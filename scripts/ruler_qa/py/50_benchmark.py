"""Generate on one fixed RULER QA grid at each requested compression ratio."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
import time

from ruler_bootstrap import require_linux
from seco.data import Collator, length_reason, limit_grid_rows, tokenizer_fingerprint
from seco.generation import add_generation_arguments, generation_options
from seco.io import fingerprint, iter_jsonl, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.progress import BatchProgress
from seco.ruler_qa.metrics import aggregate, ratio_directory, score_prediction, validate_grid


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", help="Relocated base model for an adapter checkpoint")
    parser.add_argument("--test-dir", default="artifacts/ruler_qa/prepared/test_v1")
    parser.add_argument("--eval-train", action="store_true", help="Evaluate the checkpoint's training split")
    parser.add_argument("--prepared-dir", default="artifacts/ruler_qa/prepared/train_v1",
                        help="Prepared training directory used with --eval-train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[4, 8, 16, 32])
    parser.add_argument("--max-test-samples", type=int, help="Cap the total number of test samples per ratio")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation cap within the prepared budget (default: 32)")
    add_generation_arguments(parser)
    parser.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generation = generation_options(parser, args, args.max_new_tokens)
    if args.max_test_samples is not None and args.max_test_samples < 1:
        parser.error("--max-test-samples must be positive")
    if len(set(args.ratios)) != len(args.ratios) or any(not math.isfinite(r) or r < 1 for r in args.ratios):
        parser.error("Ratios must be distinct finite numbers >= 1")
    if args.batch_size < 1 or min(args.workers, args.warmup_batches) < 0:
        parser.error("Invalid batch or generation options")

    split = "train" if args.eval_train else "test"
    dataset = project_path(args.prepared_dir if args.eval_train else args.test_dir)
    checkpoint, output = map(project_path, (args.checkpoint, args.output))
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    if rank == 0 and output.exists():
        raise FileExistsError(f"Choose a new benchmark directory: {output}")
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    data_format = "seco-data-v1" if args.eval_train else "seco-ruler-qa-test-v1"
    if (manifest.get("format"), manifest.get("task"), manifest.get("kind")) != (data_format, "ruler_qa", split):
        raise ValueError(f"Expected prepared RULER QA {split} data")
    data_path = dataset / f"{split}.jsonl"
    if sha256_file(data_path) != manifest["files"][data_path.name]:
        raise ValueError(f"{split} data hash mismatch")
    settings = manifest["arguments"]
    if min(args.ratios) < settings["min_ratio"] or args.max_new_tokens > settings["generation_budget"]:
        raise ValueError("Ratio or generation limit is outside the prepared evaluation budget")
    experiment = read_json(checkpoint / "experiment.json")
    if experiment.get("format") != "seco-adapters-v1":
        raise ValueError("Unsupported SeCo checkpoint format")
    if experiment["tokenizer_fingerprint"] != manifest["tokenizer_fingerprint"]:
        raise ValueError("Prepared data and checkpoint tokenizer fingerprints differ")
    if args.eval_train and (experiment.get("task") != "ruler_qa"
                            or experiment.get("data_manifest_sha256") != sha256_file(manifest_path)):
        raise ValueError("Checkpoint and --prepared-dir must refer to the same RULER QA training manifest")
    identity = checkpoint / ("artifact.json" if (checkpoint / "artifact.json").is_file() else "model.safetensors")
    if not identity.is_file():
        raise FileNotFoundError(identity)
    rows = list(iter_jsonl(data_path))
    validate_grid(rows, manifest)
    if not args.eval_train:
        training_keys = set(experiment.get("ruler_qa_training_question_keys", []))
        if experiment.get("task") == "ruler_qa" and not training_keys:
            raise ValueError("RULER QA checkpoint lacks training question identity")
        if any(row["question_key"] in training_keys for row in rows):
            raise ValueError("Test questions overlap the checkpoint's RULER QA training set")
    prepared_count = len(rows)
    rows = limit_grid_rows(rows, args.max_test_samples, cell_keys=("task", "target_length"))
    if len(rows) < prepared_count:
        counts, ids = defaultdict(Counter), defaultdict(list)
        for row in rows:
            counts[row["task"]][str(row["target_length"])] += 1
            ids[row["task"]].append(row["id"])
        manifest = {**manifest, "prepared_test_identity": fingerprint(manifest), "counts": dict(counts),
                    "sample_id_hashes": {task: fingerprint(sorted(task_ids)) for task, task_ids in ids.items()}}
    if args.eval_train:
        for row in rows:
            reason = length_reason(
                {**row, "answer_tokens": args.max_new_tokens},
                max_input_tokens=settings["max_input_tokens"], max_answer_tokens=settings["generation_budget"],
                max_decoder_tokens=settings["max_decoder_tokens"], min_ratio=min(args.ratios), min_context_tokens=1,
            )
            if reason:
                raise ValueError(f"{row['id']}: {reason} for generation; check --max-new-tokens and --ratios")
    if args.plan_only:
        print(json.dumps({"split": split, "counts": manifest["counts"], "ratios": args.ratios, "world_size": world,
                          "prepared_test_samples": prepared_count, "test_samples": len(rows),
                          "generation_examples": len(rows) * len(args.ratios)}, indent=2))
        return

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from transformers import set_seed
    from seco.model import load_checkpoint_model

    if not torch.cuda.is_available():
        raise RuntimeError("RULER QA generation requires a visible CUDA GPU")
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    set_seed(args.seed + rank)
    if world > 1:
        dist.init_process_group("nccl")
    model, _ = load_checkpoint_model(checkpoint, model_path=args.model, attention=args.attention)
    if tokenizer_fingerprint(model.tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Loaded model tokenizer differs from the prepared data tokenizer")
    if max(settings["max_input_tokens"], settings["max_decoder_tokens"]) > model.config.max_position_embeddings:
        raise ValueError("Prepared token limits exceed the loaded model position limit")
    model.to(device).eval()
    if not args.no_merge:
        model.merge_for_inference()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "benchmark.json", {
            "task": "ruler_qa", "split": split, "arguments": vars(args), "experiment": experiment,
            "generation": generation, "test_manifest": manifest,
            "checkpoint_identity": sha256_file(identity), "world_size": world,
            "gpu": torch.cuda.get_device_name(device), "attention": model.model_config["attn_implementation"],
            "timing": "amortized generation seconds per sample, including encoder and compression",
        })
    if world > 1:
        dist.barrier()

    order = {row["id"]: index for index, row in enumerate(rows)}
    shard = sorted(rows[rank::world], key=lambda row: len(row["input_ids"]))
    loader = DataLoader(shard, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                        collate_fn=Collator(model.tokenizer.pad_token_id, training=False))
    for ratio in args.ratios:
        directory = output / ratio_directory(ratio)
        progress = BatchProgress(f"[ruler_qa rank={rank} R={ratio:g}]", len(shard))
        offset = 0
        torch.cuda.reset_peak_memory_stats(device)

        def predict():
            nonlocal offset
            for index, batch in enumerate(loader):
                batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
                if index == 0:
                    for _ in range(args.warmup_batches):
                        model.generate_batch(batch, ratio=ratio, generation=generation)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                tokens = model.generate_batch(batch, ratio=ratio, generation=generation)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                texts = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                if len(texts) != batch["input_ids"].size(0):
                    raise RuntimeError("Generated batch size differs from input batch")
                for row, prediction in zip(shard[offset:offset + len(texts)], texts):
                    result = {key: row[key] for key in ("id", "task", "source_id", "question", "answer",
                                                        "target_length", "context_tokens", "prompt_tokens")}
                    yield {**result, "prediction": prediction, "ratio": ratio,
                           "seconds_per_sample": elapsed / len(texts), **score_prediction(prediction, row["answer"])}
                offset += len(texts)
                progress.advance(len(texts))

        write_jsonl(directory / f"predictions.rank{rank}.jsonl", predict())
        write_json(directory / f"runtime.rank{rank}.json",
                   {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30})
        if world > 1:
            dist.barrier()
        if rank == 0:
            combined = [row for worker in range(world)
                        for row in iter_jsonl(directory / f"predictions.rank{worker}.jsonl")]
            metrics = aggregate(combined, manifest, ratio)
            combined.sort(key=lambda row: order[row["id"]])
            metrics["mean_seconds_per_sample"] = sum(row["seconds_per_sample"] for row in combined) / len(combined)
            metrics["peak_allocated_gib"] = max(read_json(directory / f"runtime.rank{worker}.json")["peak_allocated_gib"]
                                                for worker in range(world))
            write_jsonl(directory / "predictions.jsonl", combined)
            write_json(directory / "metrics.json", metrics)
            print(f"R={ratio:g}: {metrics['summary']}", flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
