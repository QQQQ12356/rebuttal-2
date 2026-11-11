"""Normalize, filter, decontaminate and sample five training and four test datasets."""
import argparse
from collections import Counter
from itertools import islice
import math
import random

from _bootstrap import require_linux
from seco.experiment import artifact_path, default_model_path
from seco.data import (BENCHMARKS, DATASETS, length_reason, normalize_record, reservoir_add,
                       tokenize_record, tokenizer_fingerprint)
from seco.io import iter_jsonl, model_source, project_path, read_json, sha256_file, write_json, write_jsonl


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=artifact_path("benchmark", "raw"))
    parser.add_argument("--output", default=artifact_path("benchmark", "prepared/default"))
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--revision", default="main")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--train-samples", type=int, nargs="+", default=[5000],
                        help="One count shared by all, or one per --datasets entry; 0 = all eligible")
    parser.add_argument("--test-samples", type=int, default=500, help="Per benchmark; 0 = all eligible")
    parser.add_argument("--candidate-limit", type=int, default=0, help="Scan first N raw training rows; 0 = all downloaded")
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-answer-tokens", type=int, default=1024, help="Training answers only; no reference-length test filtering")
    parser.add_argument("--max-decoder-tokens", type=int, default=8192)
    parser.add_argument("--min-context-tokens", type=int, default=32)
    parser.add_argument("--min-ratio", type=float, default=4, help="Smallest ratio intended for training AND evaluation")
    parser.add_argument("--test-generation-budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-count", action="store_true", help="Fail if requested sample count cannot be filled")
    args = parser.parse_args()
    if len(args.train_samples) not in (1, len(args.datasets)):
        parser.error("--train-samples needs one count or exactly one per dataset")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets cannot contain duplicate entries")
    if any(v < 0 for v in [*args.train_samples, args.test_samples, args.candidate_limit, args.min_context_tokens]):
        parser.error("Sample counts and minimum context must be >= 0")
    if not math.isfinite(args.min_ratio) or args.min_ratio < 1 or any(v < 1 for v in (args.max_input_tokens, args.max_answer_tokens,
                                                args.max_decoder_tokens, args.test_generation_budget)):
        parser.error("Invalid token limits or minimum ratio")
    from transformers import AutoTokenizer

    raw_dir, output = project_path(args.raw_dir), project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new prepared output directory: {output}")
    tokenizer = AutoTokenizer.from_pretrained(model_source(args.model), revision=args.revision, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    limits = {key: getattr(args, key) for key in ("max_input_tokens", "max_answer_tokens", "max_decoder_tokens",
                                                 "min_ratio", "min_context_tokens")}
    sources, statistics, file_hashes = {}, {}, {}
    blocked_contexts, blocked_questions = set(), set()
    output.mkdir(parents=True)

    def prepare(name, role, limit, index):
        path = raw_dir / role / f"{name}.jsonl"
        source_manifest = path.with_suffix(".source.json")
        digest = sha256_file(path)
        if source_manifest.is_file():
            source = read_json(source_manifest)
            if source["sha256"] != digest:
                raise ValueError(f"Raw file modified after download: {path}")
        else:
            # Local JSONL is an explicit supported input; provenance stays visible.
            source = {"local_file": str(path), "sha256": digest}
        sources[f"{role}/{name}"] = source
        stats = Counter()
        sample, seen_keys = [], set()
        rng = random.Random(args.seed + index)
        iterator = iter_jsonl(path)
        if role == "training" and args.candidate_limit:
            iterator = islice(iterator, args.candidate_limit)
        for number, row in enumerate(iterator):
            stats["scanned"] += 1
            try:
                record = normalize_record(row, name, number, "train" if role == "training" else "test")
            except ValueError:
                stats["invalid_or_unanswerable"] += 1
                continue
            # Unknown schema raises KeyError/TypeError instead of silently discarding a dataset.
            if role == "benchmark":
                blocked_contexts.add(record["context_key"])
                if name != "multinews":
                    blocked_questions.add(record["question_key"])
            elif record["context_key"] in blocked_contexts or record["question_key"] in blocked_questions:
                stats["test_overlap"] += 1
                continue
            # Preserve different questions about the same document.
            key = record["content_key"]
            if key in seen_keys:
                stats["duplicates"] += 1
                continue
            seen_keys.add(key)
            tokens = tokenize_record(record, tokenizer)
            check_tokens = tokens.copy()
            if role == "benchmark":
                check_tokens["answer_tokens"] = args.test_generation_budget
            reason = length_reason(check_tokens, **{**limits, "max_answer_tokens":
                                   args.max_answer_tokens if role == "training" else args.test_generation_budget})
            if reason:
                stats[reason] += 1
                continue
            stats["eligible"] += 1
            record.update(tokens)
            reservoir_add(sample, record, stats["eligible"], limit, rng)
        if not sample:
            raise ValueError(f"No eligible {role}/{name} examples. Inspect schema/length limits: {dict(stats)}")
        if limit and len(sample) < limit:
            message = f"Requested {limit}, found {len(sample)}: {role}/{name}"
            if args.strict_count:
                raise ValueError(message)
            print(f"WARNING: {message}")
        rng.shuffle(sample)
        stats["selected"] = len(sample)
        statistics[f"{role}/{name}"] = dict(stats)
        relative = f"{'test' if role == 'benchmark' else 'train'}/{name}.jsonl"
        write_jsonl(output / relative, sample)
        file_hashes[relative] = sha256_file(output / relative)
        print(f"{role}/{name}: {dict(stats)}")
        return sample

    # Block against ALL normalized raw benchmark rows, even rows later filtered or unsampled.
    for index, name in enumerate(BENCHMARKS):
        prepare(name, "benchmark", args.test_samples, index)
    counts = args.train_samples * len(args.datasets) if len(args.train_samples) == 1 else args.train_samples
    for index, (name, count) in enumerate(zip(args.datasets, counts), 100):
        prepare(name, "training", count, index)
    # Interleave with a deterministic global shuffle; Trainer reshuffles each epoch.
    train_rows = [row for name in args.datasets for row in iter_jsonl(output / "train" / f"{name}.jsonl")]
    random.Random(args.seed).shuffle(train_rows)
    write_jsonl(output / "train.jsonl", train_rows)
    file_hashes["train.jsonl"] = sha256_file(output / "train.jsonl")
    write_json(output / "manifest.json", {"format": "seco-data-v1", "arguments": vars(args),
               "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer), "sources": sources,
               "statistics": statistics, "files": file_hashes})
    print(f"Prepared data complete: {output}")


if __name__ == "__main__":
    main()
