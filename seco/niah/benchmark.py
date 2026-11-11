"""Evaluate a held-out NIAH grid or explicitly selected NIAH training rows."""
import argparse
import json
import math
import os
import time

from seco.data import Collator, limit_grid_rows, tokenizer_fingerprint
from seco.experiment import artifact_path, check_method, memory_token_bound, validate_ratios
from seco.generation import add_generation_arguments, generation_options
from seco.io import fingerprint, iter_jsonl, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.niah.metrics import aggregate, aggregate_training, score_prediction
from seco.niah.train_eval import load_training_rows
from seco.progress import BatchProgress, log_progress


def main(*, train_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", help="Relocated base directory for adapter checkpoints")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--grid-dir", help="Held-out grid directory (default: method's prepared/grid_v1)")
    data.add_argument("--prepared-dir", help="Training data directory (default: method's prepared/train_v1)")
    parser.add_argument("--eval-train", action="store_true", help="Explicitly evaluate the checkpoint's own NIAH training data")
    parser.add_argument("--output", required=True, help="New independent NIAH benchmark directory")
    parser.add_argument("--ratios", nargs="+", type=float, default=[4, 8, 16, 32])
    parser.add_argument("--max-test-samples", type=int, help="Cap the total number of test samples per ratio")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation cap within the prepared budget (default: 64)")
    add_generation_arguments(parser)
    parser.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Verify inputs and report workload without importing torch")
    parser.add_argument("--seed", type=int, help="Generation seed override (defaults to the dataset seed)")
    args = parser.parse_args()
    generation = generation_options(parser, args, args.max_new_tokens)
    if train_only and not args.eval_train:
        parser.error("Training-set evaluation requires --eval-train")
    if (args.eval_train and args.grid_dir) or (not args.eval_train and args.prepared_dir):
        parser.error("Use --prepared-dir with --eval-train, or --grid-dir for held-out evaluation")
    if args.max_test_samples is not None and args.max_test_samples < 1:
        parser.error("--max-test-samples must be positive")
    if len(set(args.ratios)) != len(args.ratios) or not args.ratios or any(not math.isfinite(r) or r < 1 for r in args.ratios):
        parser.error("ratios must be distinct finite numbers >= 1")
    if args.batch_size < 1 or min(args.workers, args.warmup_batches) < 0:
        parser.error("Invalid batch/generation options")
    source, output = project_path(args.checkpoint), project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new NIAH benchmark directory: {output}")
    experiment = read_json(source / "experiment.json")
    if experiment.get("format") != "seco-adapters-v1":
        raise ValueError("Unsupported checkpoint format")
    model_config = experiment["config"]["model"]
    check_method(model_config)
    validate_ratios(model_config, args.ratios)
    checkpoint_file = source / ("artifact.json" if (source / "artifact.json").exists() else "model.safetensors")
    split = "train" if args.eval_train else "test"
    if args.eval_train:
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
        args.prepared_dir = args.prepared_dir or artifact_path("niah", "prepared/train_v1")
        prepared = project_path(args.prepared_dir)
        manifest, manifest_hash, rows, total = load_training_rows(prepared, experiment, args.max_test_samples)
        settings = manifest["arguments"]
        if min(args.ratios) < settings["min_ratio"]:
            raise ValueError("Ratio below training data min_ratio")
        for row in rows:
            if decoder_tokens(row, args, model_config) > settings["max_decoder_tokens"]:
                raise ValueError(f"Training sample exceeds the prepared decoder budget: {row['id']}")
        seed = manifest["preparation"]["seed"]
        sample_count = len(rows)
        workload = {"training_examples": total, "maximum_encoder_length": max(len(row["input_ids"]) for row in rows)}
    else:
        args.grid_dir = args.grid_dir or artifact_path("niah", "prepared/grid_v1")
        prepared = project_path(args.grid_dir)
        manifest, training_identity = read_test_grid(prepared, experiment, args)
        manifest_hash = sha256_file(prepared / "manifest.json")
        seed = manifest["grid"]["seed"]
        sample_count = min(manifest["count"], args.max_test_samples or manifest["count"])
        workload = {"grid_examples": manifest["count"], "maximum_encoder_length": max(manifest["grid"]["lengths"])}

    if args.plan_only:
        print(json.dumps({"checkpoint": str(source), "split": split, **workload,
                          "test_samples": sample_count, "ratios": args.ratios,
                          "scored_generation_examples": sample_count * len(args.ratios),
                          "world_size": int(os.getenv("WORLD_SIZE", "1")),
                          "model_position_limits_checked": False}, indent=2))
        return
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from transformers import set_seed
    from seco.model import load_checkpoint_model

    if not torch.cuda.is_available():
        raise RuntimeError("NIAH generation requires a visible Linux CUDA GPU")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    worker_label = f"[niah {split} rank={rank}]"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    generation_seed = seed if args.seed is None else args.seed
    set_seed(generation_seed + rank)
    log_progress(f"{worker_label} loading checkpoint {source}")
    model, experiment = load_checkpoint_model(source, model_path=args.model, attention=args.attention)
    if tokenizer_fingerprint(model.tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Prepared NIAH tokenizer differs from the loaded model tokenizer")
    model.to(device).eval()
    if not args.no_merge:
        model.merge_for_inference()
    log_progress(f"{worker_label} validating prepared samples")
    if not args.eval_train:
        rows = list(iter_jsonl(prepared / "grid.jsonl"))
        validate_test_rows(rows, manifest, training_identity)
    for row in rows:
        decoder_length = decoder_tokens(row, args, model_config)
        budget = settings["max_decoder_tokens"] if args.eval_train else manifest["grid"]["max_decoder_tokens"]
        if decoder_length > budget:
            raise ValueError(f"NIAH sample exceeds the prepared decoder budget: {row['id']}")
        if max(len(row["input_ids"]), decoder_length) > model.config.max_position_embeddings:
            raise ValueError(f"NIAH sample exceeds model position limit: {row['id']}")
    if not args.eval_train:
        rows = limit_grid_rows(rows, args.max_test_samples, cell_keys=("sequence_length", "depth_percent"))
    expected_ids = {row["id"] for row in rows}
    selection = {}
    if args.eval_train or len(rows) < manifest["count"]:
        selection = {"count": len(rows), "id_hash": fingerprint(sorted(expected_ids))}
    if args.eval_train:
        metric_identity = {"training_manifest_sha256": manifest_hash, "sample_selection": selection}
        dataset_metadata = {"training_manifest": manifest, "training_examples": total, **metric_identity}
    else:
        grid_identity = fingerprint([manifest, selection] if selection else manifest)
        metric_identity = {"grid_manifest_sha256": manifest_hash, "grid_identity": grid_identity}
        dataset_metadata = {"grid_manifest": manifest, "sample_selection": selection, **metric_identity}
    if rank == 0:
        write_json(output / "benchmark.json", {"task": "niah", "split": split, "arguments": vars(args),
                   "method": model.method, "model_config": model_config,
                   **dataset_metadata,
                   "generation": generation, "generation_seed": generation_seed,
                   "checkpoint_identity": sha256_file(checkpoint_file), "world_size": world,
                   "gpu": torch.cuda.get_device_name(device), "attention": model.model_config["attn_implementation"],
                   "timing": "amortized batch seconds per example; encoder+compression+decoder, excluding H2D/loading"})
    shard = sorted(rows[rank::world], key=lambda row: len(row["input_ids"]))
    del rows
    collator = Collator(model.tokenizer.pad_token_id, training=False)
    for ratio_index, ratio in enumerate(args.ratios, start=1):
        label = f"{worker_label} R={ratio:g} ({ratio_index}/{len(args.ratios)})"
        progress = BatchProgress(label, len(shard))
        directory = output / f"ratio_{ratio:g}"
        directory.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(shard, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                            num_workers=args.workers, pin_memory=True)
        offset = 0
        torch.cuda.reset_peak_memory_stats(device)
        with (directory / f"predictions.rank{rank}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for number, batch in enumerate(loader):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                if number == 0:
                    for _ in range(args.warmup_batches):
                        model.generate_batch(batch, ratio=ratio, generation=generation)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                tokens = model.generate_batch(batch, ratio=ratio, generation=generation)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                predictions = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                if len(predictions) != batch["input_ids"].size(0):
                    raise RuntimeError("NIAH generation batch size mismatch")
                for row, prediction in zip(shard[offset:offset + len(predictions)], predictions):
                    result = prediction_metadata(row, args.eval_train)
                    result.update({"prediction": prediction, "ratio": ratio, "status": "ok",
                                   "memory_tokens_upper_bound": memory_token_bound(model_config, row["context_tokens"], ratio),
                                   "seconds_per_sample": elapsed / len(predictions),
                                   **score_prediction(prediction, row["answer"])})
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                offset += len(predictions)
                progress.advance(len(predictions))
        write_json(directory / f"runtime.rank{rank}.json", {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30})
        if world > 1:
            log_progress(f"{label}: shard saved, waiting for all ranks")
            dist.barrier()
        if rank == 0:
            log_progress(f"{label}: aggregating {len(expected_ids)} samples")
            combined = [row for worker in range(world) for row in iter_jsonl(directory / f"predictions.rank{worker}.jsonl")]
            if len(combined) != len(expected_ids) or {row["id"] for row in combined} != expected_ids:
                raise ValueError("NIAH shard IDs differ from selected sample IDs")
            if args.eval_train:
                metrics = aggregate_training(combined)
                combined.sort(key=lambda row: row["id"])
            else:
                metrics = aggregate(combined, manifest["grid"], allow_incomplete=bool(selection))
                combined.sort(key=lambda row: (row["sequence_length"], row["depth_percent"], row["trial"]))
            write_jsonl(directory / "predictions.jsonl", combined)
            metrics.update({"ratio": ratio, "checkpoint": str(source),
                            **metric_identity,
                            "mean_seconds_per_sample": sum(row["seconds_per_sample"] for row in combined) / len(combined),
                            "peak_allocated_gib": max(read_json(directory / f"runtime.rank{worker}.json")["peak_allocated_gib"] for worker in range(world))})
            write_json(directory / "metrics.json", metrics)
            log_progress(f"{label}: n={metrics['count']}, accuracy={metrics['summary']['accuracy']:.4f}, "
                         f"F1={metrics['summary']['f1']:.4f}")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


def decoder_tokens(row, args, model_config=None):
    context = row["context_tokens"]
    centers = max(memory_token_bound(model_config or {}, context, ratio) for ratio in args.ratios)
    return centers + row["prompt_tokens"] + args.max_new_tokens


def read_test_grid(prepared, experiment, args):
    manifest = read_json(prepared / "manifest.json")
    if manifest.get("format") != "seco-niah-grid-v1" or manifest.get("kind") != "test":
        raise ValueError("NIAH benchmark requires an independently prepared NIAH test grid")
    if sha256_file(prepared / "grid.jsonl") != manifest["files"]["grid.jsonl"]:
        raise ValueError("NIAH grid hash mismatch")
    if min(args.ratios) < manifest["grid"]["min_ratio"]:
        raise ValueError("Ratio below grid min_ratio; prepare a new grid with a compatible decoder budget")
    if args.max_new_tokens > manifest["grid"]["generation_budget"]:
        raise ValueError("max_new_tokens exceeds the prepared generation budget")
    training_identity = experiment.get("niah_training_identity", {})
    documents = {row["text_sha256"] for row in manifest["haystack"]["documents"]}
    if documents.intersection(training_identity.get("document_hashes", [])):
        raise ValueError("Checkpoint training documents overlap this NIAH test haystack")
    return manifest, training_identity


def validate_test_rows(rows, manifest, training_identity):
    if len(rows) != manifest["count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Grid count or sample identity mismatch")
    blocked_answers = set(training_identity.get("answer_hashes", []))
    for row in rows:
        if any(fingerprint(answer) in blocked_answers for answer in row["answer"]):
            raise ValueError(f"Evaluation answer appears in this checkpoint's NIAH training data: {row['id']}")
        if len(row["input_ids"]) != row["sequence_length"]:
            raise ValueError(f"Grid length mismatch: {row['id']}")


def prediction_metadata(row, eval_train):
    common = {key: row[key] for key in ("id", "task", "answer", "prompt")}
    if eval_train:
        return {**common, "split": "train", "source": row.get("source", "synthetic"),
                "sequence_length": len(row["input_ids"])}
    fields = ("sequence_length", "depth_percent", "actual_depth_percent", "needle_start_token",
              "needle_tokens", "trial", "seed", "needle", "haystack_start", "source_document_ids")
    return {**common, **{key: row[key] for key in fields}}
