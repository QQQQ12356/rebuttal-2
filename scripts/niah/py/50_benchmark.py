"""Evaluate SeCo on every held-out NIAH grid cell using disjoint GPU shards."""
import argparse
import json
import math
import os
import time

from niah_bootstrap import require_linux
from seco.data import Collator, tokenizer_fingerprint
from seco.io import fingerprint, iter_jsonl, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.niah.metrics import aggregate, score_prediction
from seco.progress import BatchProgress, log_progress


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", help="Relocated base directory for adapter checkpoints")
    parser.add_argument("--grid-dir", default="artifacts/niah/prepared/grid_v1")
    parser.add_argument("--output", required=True, help="New independent NIAH benchmark directory")
    parser.add_argument("--ratios", nargs="+", type=float, default=[4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Verify grid and checkpoint metadata without importing torch")
    args = parser.parse_args()
    if len(set(args.ratios)) != len(args.ratios) or not args.ratios or any(not math.isfinite(r) or r < 1 for r in args.ratios):
        parser.error("ratios must be distinct finite numbers >= 1")
    if args.batch_size < 1 or min(args.workers, args.warmup_batches) < 0 or args.max_new_tokens < 1 or args.repetition_penalty <= 0:
        parser.error("Invalid batch/generation options")
    prepared, source, output = project_path(args.grid_dir), project_path(args.checkpoint), project_path(args.output)
    manifest = read_json(prepared / "manifest.json")
    if manifest.get("format") != "seco-niah-grid-v1" or manifest.get("kind") != "test":
        raise ValueError("NIAH benchmark requires an independently prepared NIAH test grid")
    if sha256_file(prepared / "grid.jsonl") != manifest["files"]["grid.jsonl"]:
        raise ValueError("NIAH grid hash mismatch")
    if min(args.ratios) < manifest["grid"]["min_ratio"]:
        raise ValueError("Ratio below grid min_ratio; prepare a new grid with a compatible decoder budget")
    if args.max_new_tokens > manifest["grid"]["generation_budget"]:
        raise ValueError("max_new_tokens exceeds the prepared generation budget")
    experiment = read_json(source / "experiment.json")
    if experiment.get("format") != "seco-adapters-v1":
        raise ValueError("Unsupported SeCo checkpoint format")
    training_identity = experiment.get("niah_training_identity", {})
    if set(training_identity.get("document_hashes", [])).intersection(row["text_sha256"] for row in manifest["haystack"]["documents"]):
        raise ValueError("Checkpoint training documents overlap this NIAH test haystack")
    if output.exists():
        raise FileExistsError(f"Choose a new NIAH benchmark directory: {output}")
    if args.plan_only:
        print(json.dumps({"checkpoint": str(source), "grid_examples": manifest["count"],
                          "ratios": args.ratios, "scored_generation_examples": manifest["count"] * len(args.ratios),
                          "maximum_encoder_length": max(manifest["grid"]["lengths"]),
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
    worker_label = f"[niah rank={rank}]"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    set_seed(manifest["grid"]["seed"] + rank)
    log_progress(f"{worker_label} loading checkpoint {source}")
    model, experiment = load_checkpoint_model(source, model_path=args.model, attention=args.attention)
    if tokenizer_fingerprint(model.tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("NIAH grid tokenizer differs from checkpoint tokenizer")
    model.to(device).eval()
    if not args.no_merge:
        model.merge_for_inference()
    log_progress(f"{worker_label} validating test grid")
    rows = list(iter_jsonl(prepared / "grid.jsonl"))
    if len(rows) != manifest["count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Grid count or sample identity mismatch")
    blocked_answers = set(training_identity.get("answer_hashes", []))
    for row in rows:
        if any(fingerprint(answer) in blocked_answers for answer in row["answer"]):
            raise ValueError(f"Evaluation answer appears in this checkpoint's NIAH training data: {row['id']}")
        if len(row["input_ids"]) != row["sequence_length"]:
            raise ValueError(f"Grid length mismatch: {row['id']}")
        centers = min(row["context_tokens"], max(2, math.ceil(row["context_tokens"] / min(args.ratios))))
        if max(len(row["input_ids"]), centers + row["prompt_tokens"] + args.max_new_tokens) > model.config.max_position_embeddings:
            raise ValueError(f"NIAH sample exceeds model position limit: {row['id']}")
    if rank == 0:
        identity = "artifact.json" if (source / "artifact.json").exists() else "model.safetensors"
        write_json(output / "benchmark.json", {"task": "niah", "arguments": vars(args), "experiment": experiment,
                   "grid_manifest": manifest, "grid_manifest_sha256": sha256_file(prepared / "manifest.json"),
                   "checkpoint_identity": sha256_file(source / identity), "world_size": world,
                   "gpu": torch.cuda.get_device_name(device), "attention": model.model_config["attn_implementation"],
                   "timing": "amortized batch seconds per example; encoder+compression+decoder, excluding H2D/loading"})
    shard = sorted(rows[rank::world], key=lambda row: row["sequence_length"])
    collator = Collator(model.tokenizer.pad_token_id, training=False)
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                  "repetition_penalty": args.repetition_penalty,
                  "return_dict_in_generate": False, "output_scores": False}
    for ratio_index, ratio in enumerate(args.ratios, start=1):
        label = f"{worker_label} R={ratio:g} ({ratio_index}/{len(args.ratios)})"
        progress = BatchProgress(f"{label} grid", len(shard))
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
                    result = {key: row[key] for key in ("id", "task", "sequence_length", "depth_percent",
                              "actual_depth_percent", "needle_start_token", "needle_tokens", "trial", "seed",
                              "answer", "needle", "prompt", "haystack_start", "source_document_ids")}
                    result.update({"prediction": prediction, "ratio": ratio, "status": "ok",
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
            log_progress(f"{label}: aggregating {len(rows)} samples")
            combined = [row for worker in range(world) for row in iter_jsonl(directory / f"predictions.rank{worker}.jsonl")]
            if {row["id"] for row in combined} != {row["id"] for row in rows}:
                raise ValueError("NIAH shard IDs differ from fixed grid IDs")
            metrics = aggregate(combined, manifest["grid"])
            combined.sort(key=lambda row: (row["sequence_length"], row["depth_percent"], row["trial"]))
            write_jsonl(directory / "predictions.jsonl", combined)
            metrics.update({"ratio": ratio, "checkpoint": str(source),
                            "grid_identity": fingerprint(manifest),
                            "grid_manifest_sha256": sha256_file(prepared / "manifest.json"),
                            "mean_seconds_per_sample": sum(row["seconds_per_sample"] for row in combined) / len(combined),
                            "peak_allocated_gib": max(read_json(directory / f"runtime.rank{worker}.json")["peak_allocated_gib"] for worker in range(world))})
            write_json(directory / "metrics.json", metrics)
            log_progress(f"{label}: n={metrics['count']}, accuracy={metrics['summary']['accuracy']:.4f}, "
                         f"F1={metrics['summary']['f1']:.4f}")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
