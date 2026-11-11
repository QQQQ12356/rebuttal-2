"""Batched generation with disjoint GPU shards and EM/F1 results."""
import argparse
import math
import os
import random
import time

from _bootstrap import require_linux
from seco.experiment import artifact_path, check_method, memory_token_bound, validate_ratios
from seco.data import BENCHMARKS, DATASETS, Collator, limit_grid_rows, tokenizer_fingerprint
from seco.generation import add_generation_arguments, generation_options
from seco.io import iter_jsonl, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.metrics import score_answer, summarize
from seco.progress import BatchProgress, log_progress


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="final/checkpoint-N adapter folder OR a merged export")
    parser.add_argument("--model", help="Override base location for an adapter checkpoint")
    parser.add_argument("--prepared-dir", default=artifact_path("benchmark", "prepared/default"))
    parser.add_argument("--output", required=True, help="New benchmark directory; never overwrite a previous run")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, help="Default: all prepared datasets for the split")
    parser.add_argument("--eval-train", action="store_true", help="Evaluate the checkpoint's prepared training split")
    parser.add_argument("--ratios", nargs="+", type=float, default=[4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=0, help="Per dataset; 0 = all prepared samples")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Global cap after --samples; 0 = unlimited")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--qa-max-new-tokens", type=int, default=64, help="QA generation cap (default: 64)")
    parser.add_argument("--news-max-new-tokens", type=int, default=512, help="MultiNews generation cap (default: 512)")
    add_generation_arguments(parser)
    parser.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), help="Default: checkpoint dtype")
    parser.add_argument("--no-merge", action="store_true", help="Keep LoRA modules active for equivalence comparisons")
    parser.add_argument("--include-context", action="store_true", help="Include source context in prediction JSONL/Excel details")
    parser.add_argument("--plan-only", action="store_true", help="Validate and print the sample selection without CUDA")
    args = parser.parse_args()
    if not args.ratios or any(not math.isfinite(r) or r < 1 for r in args.ratios):
        parser.error("Ratios must be finite and >= 1")
    if args.batch_size < 1 or min(args.samples, args.max_test_samples, args.workers, args.warmup_batches) < 0:
        parser.error("Invalid batch size, samples, workers or warmup")
    output, prepared, source = project_path(args.output), project_path(args.prepared_dir), project_path(args.checkpoint)
    manifest_path = prepared / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("format") != "seco-data-v1" or manifest.get("task", "qa") != "qa":
        raise ValueError("Expected prepared Benchmark data")
    experiment = read_json(source / "experiment.json")
    model_config = experiment["config"]["model"]
    check_method(model_config)
    validate_ratios(model_config, args.ratios)
    split = "train" if args.eval_train else "test"
    available = tuple(name for name in (DATASETS if args.eval_train else BENCHMARKS)
                      if f"{split}/{name}.jsonl" in manifest["files"])
    args.datasets = args.datasets or list(available)
    if not args.datasets or set(args.datasets) - set(available):
        parser.error(f"Available {split} datasets: {', '.join(available)}")
    manifest_hash = sha256_file(manifest_path)
    if args.eval_train and (experiment.get("task") != "qa"
                            or experiment.get("data_manifest_sha256") != manifest_hash
                            or experiment.get("tokenizer_fingerprint") != manifest["tokenizer_fingerprint"]):
        raise ValueError("Checkpoint and --prepared-dir must refer to the same Benchmark training manifest")
    limits = {name: args.news_max_new_tokens if name == "multinews" else args.qa_max_new_tokens
              for name in args.datasets}
    decoding = {name: generation_options(parser, args, limit) for name, limit in limits.items()}
    if len(set(args.ratios)) != len(args.ratios) or len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate ratios/datasets would duplicate predictions")
    settings = manifest["arguments"]
    if min(args.ratios) < settings["min_ratio"] or max(limits.values()) > settings["test_generation_budget"]:
        raise ValueError("Ratio or generation length is outside the prepared evaluation budget")
    selected = []
    for index, name in enumerate(args.datasets):
        relative = f"{split}/{name}.jsonl"
        if sha256_file(prepared / relative) != manifest["files"][relative]:
            raise ValueError(f"Prepared data hash mismatch: {relative}")
        rows = list(iter_jsonl(prepared / relative))
        if args.shuffle:
            random.Random(args.seed + index).shuffle(rows)
        selected.extend(rows[:args.samples] if args.samples else rows)
    selected = limit_grid_rows(selected, args.max_test_samples or None, cell_keys=("dataset",))
    ids = [row["id"] for row in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation requires nonempty samples with unique IDs")
    for row in selected:
        memory = max(memory_token_bound(model_config, row["context_tokens"], ratio) for ratio in args.ratios)
        if memory + row["prompt_tokens"] + limits[row["dataset"]] > settings["max_decoder_tokens"]:
            raise ValueError(f"Sample exceeds the prepared decoder budget: {row['id']}")
    identity_file = "artifact.json" if (source / "artifact.json").exists() else "model.safetensors"
    if not (source / identity_file).is_file():
        raise FileNotFoundError(source / identity_file)
    if args.plan_only:
        print(f"Benchmark {split}: {len(selected)} samples per ratio, {len(selected) * len(args.ratios)} generations")
        return
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from transformers import set_seed
    from seco.model import load_checkpoint_model

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark entry requires a Linux CUDA GPU")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    worker_label = f"[benchmark rank={rank}]"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    set_seed(args.seed + rank)
    log_progress(f"{worker_label} loading checkpoint {source}")
    model, experiment = load_checkpoint_model(source, model_path=args.model, attention=args.attention, dtype=args.dtype)
    if tokenizer_fingerprint(model.tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Evaluation tokenizer differs from preprocessing")
    model.to(device).eval()
    if not args.no_merge:
        model.merge_for_inference()
    if rank == 0:
        write_json(output / "benchmark.json", {"split": split, "arguments": vars(args), "experiment": experiment,
                   "checkpoint_identity": sha256_file(source / identity_file),
                   "generation_by_dataset": decoding,
                   "data_manifest_sha256": manifest_hash,
                   "samples": len(selected), "world_size": world, "gpu": torch.cuda.get_device_name(device),
                   "torch": torch.__version__, "attention": model.model_config["attn_implementation"],
                   "merged": model.merged, "timing": "amortized seconds per sample within each generation batch; includes encoder, compression and decoder; excludes loading/tokenization/H2D"})
    collator = Collator(model.tokenizer.pad_token_id, training=False)
    # Bucket by dataset and then length; restore canonical selection order on rank 0.
    order = {row["id"]: index for index, row in enumerate(selected)}
    shard = selected[rank::world]
    for ratio_index, ratio in enumerate(args.ratios, start=1):
        label = f"{worker_label} R={ratio:g} ({ratio_index}/{len(args.ratios)})"
        log_progress(f"{label}: starting ratio, local samples={len(shard)}")
        directory = output / f"ratio_{ratio:g}"
        directory.mkdir(parents=True, exist_ok=True)
        predictions = []
        torch.cuda.reset_peak_memory_stats(device)
        for name in args.datasets:
            rows = sorted((row for row in shard if row["dataset"] == name), key=lambda row: len(row["input_ids"]))
            if not rows:
                continue
            max_new = limits[name]
            for row in rows:
                centers = memory_token_bound(model.model_config, row["context_tokens"], ratio)
                decoder_length = centers + row["prompt_tokens"] + max_new
                if max(len(row["input_ids"]), decoder_length) > model.config.max_position_embeddings:
                    raise ValueError(f"Sample exceeds model position limit: {row['id']}")
            progress = BatchProgress(f"{label} dataset={name}", len(rows))
            loader = DataLoader(rows, batch_size=args.batch_size, collate_fn=collator,
                                num_workers=args.workers, pin_memory=True, shuffle=False)
            generation = decoding[name]
            offset = 0
            for batch_number, batch in enumerate(loader):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                if batch_number == 0:
                    for _ in range(args.warmup_batches):
                        model.generate_batch(batch, ratio=ratio, generation=generation)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                token_ids = model.generate_batch(batch, ratio=ratio, generation=generation)
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - start
                texts = model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                if len(texts) != batch["input_ids"].size(0):
                    raise RuntimeError("Generated batch size differs from input batch")
                batch_rows = rows[offset:offset + len(texts)]
                for row, prediction in zip(batch_rows, texts):
                    em, f1 = score_answer(prediction, row["answer"])
                    result = {"id": row["id"], "dataset": name, "prompt": row["prompt"],
                              "memory_tokens_upper_bound": memory_token_bound(model.model_config, row["context_tokens"], ratio),
                              "answer": row["answer"], "prediction": prediction, "em": em, "f1": f1,
                              "ratio": ratio, "context_tokens": row["context_tokens"],
                              "seconds_per_sample": seconds / len(texts)}
                    if args.include_context:
                        result["input"] = row["input"]
                    predictions.append(result)
                offset += len(texts)
                progress.advance(len(texts))
        write_jsonl(directory / f"predictions.rank{rank}.jsonl", predictions)
        write_json(directory / f"runtime.rank{rank}.json", {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30})
        if world > 1:
            log_progress(f"{label}: shard saved, waiting for all ranks")
            dist.barrier()
        if rank == 0:
            log_progress(f"{label}: aggregating {len(selected)} samples")
            combined = [row for worker in range(world) for row in iter_jsonl(directory / f"predictions.rank{worker}.jsonl")]
            if len(combined) != len(selected) or {row["id"] for row in combined} != set(ids):
                raise RuntimeError("Missing or duplicated GPU shard predictions")
            combined.sort(key=lambda row: order[row["id"]])
            write_jsonl(directory / "predictions.jsonl", combined)
            metrics = summarize(combined)
            metrics.update({"ratio": ratio, "split": split, "benchmark": str(output),
                            "mean_seconds_per_sample": sum(row["seconds_per_sample"] for row in combined) / len(combined),
                            "peak_allocated_gib": max(read_json(directory / f"runtime.rank{worker}.json")["peak_allocated_gib"]
                                                      for worker in range(world))})
            write_json(directory / "metrics.json", metrics)
            log_progress(f"{label}: n={metrics['count']}, EM={metrics['micro']['em']:.4f}, "
                         f"F1={metrics['micro']['f1']:.4f}")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
