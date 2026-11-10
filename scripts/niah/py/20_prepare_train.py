"""Filter and sample NIAH supervised train splits plus fresh synthetic retrieval examples."""
import argparse
from collections import Counter
from itertools import islice
import random

from niah_bootstrap import require_linux
from seco.data import length_reason, reservoir_add, tokenizer_fingerprint
from seco.io import fingerprint, iter_jsonl, model_source, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.niah.config import positive, validate_limits
from seco.niah.data import Haystack, construct_sample, make_split_needle, normalize_supervised


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/niah/prepare_train.json")
    parser.add_argument("--raw-dir", default="artifacts/niah/raw")
    parser.add_argument("--output", default="artifacts/niah/prepared/train_v1")
    parser.add_argument("--model", default="artifacts/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--synthetic-samples", type=int, help="Override target count; 0 disables synthetic training")
    parser.add_argument("--synthetic-tasks", nargs="+", choices=("uuid", "passkey", "custom"),
                        help="Override JSON task list, e.g. --synthetic-tasks custom")
    parser.add_argument("--supervised-samples", type=int, help="Override each supervised source target; 0 = all eligible")
    parser.add_argument("--no-supervised", action="store_true", help="Skip all supervised sources without reading their files")
    args = parser.parse_args()
    config = read_json(project_path(args.config))
    if args.no_supervised:
        config["supervised"] = []
    if args.synthetic_samples is not None:
        config["synthetic"]["samples"] = args.synthetic_samples
    if args.synthetic_tasks is not None:
        config["synthetic"]["tasks"] = args.synthetic_tasks
    if args.supervised_samples is not None:
        for spec in config["supervised"]:
            spec["samples"] = args.supervised_samples
    limits, synthetic = config["limits"], config["synthetic"]
    validate_limits(limits)
    positive(config["candidate_limit"], "candidate_limit", allow_zero=True)
    positive(synthetic["samples"], "synthetic.samples", allow_zero=True)
    positive(synthetic["candidates"], "synthetic.candidates")
    for spec in config["supervised"]:
        positive(spec["samples"], "supervised.samples", allow_zero=True)
    if len({spec["source"] for spec in config["supervised"]}) != len(config["supervised"]):
        raise ValueError("Duplicate supervised sources")
    if not synthetic["lengths"] or not synthetic["tasks"] or set(synthetic["tasks"]) - {"uuid", "passkey", "custom"}:
        raise ValueError("Synthetic training needs lengths and tasks chosen from uuid, passkey, and custom")
    for length in synthetic["lengths"]:
        positive(length, "synthetic length")
    from transformers import AutoTokenizer

    raw, output = project_path(args.raw_dir), project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new NIAH training data version: {output}")
    tokenizer = AutoTokenizer.from_pretrained(model_source(args.model), revision=args.revision, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    heldout_hashes, provenance = set(), {}
    for name in config["heldout_corpora"]:
        path = raw / f"{name}.jsonl"
        provenance[name] = {"sha256": sha256_file(path), "role": "heldout_haystack"}
        heldout_hashes.update(fingerprint(" ".join(row["text"].split())) for row in iter_jsonl(path))
    statistics, selected, used_ids, used_content = {}, [], set(), set()
    pool_identity = None
    answer_hashes = set()

    def select_rows(rows, name, target, seed):
        stats, sample, rng = Counter(), [], random.Random(seed)
        for row in rows:
            stats["scanned"] += 1
            reason = length_reason(row, **limits)
            if reason:
                stats[reason] += 1
                continue
            digest = fingerprint([row["input_ids"], row["prompt_mask"]])
            if row["id"] in used_ids or digest in used_content:
                stats["duplicates"] += 1
                continue
            if "input" in row and fingerprint(" ".join(row["input"].split())) in heldout_hashes:
                stats["heldout_overlap"] += 1
                continue
            used_ids.add(row["id"])
            used_content.add(digest)
            stats["eligible"] += 1
            reservoir_add(sample, row, stats["eligible"], target, rng)
        if not sample:
            raise ValueError(f"No eligible NIAH training examples: {name}: {dict(stats)}")
        if target and len(sample) < target:
            message = f"{name}: requested {target}, selected {len(sample)}"
            if config["strict_count"]:
                raise ValueError(message)
            print(f"WARNING: {message}")
        stats["selected"] = len(sample)
        statistics[name] = dict(stats)
        selected.extend(sample)
        answer_hashes.update(fingerprint(answer) for row in sample for answer in row["answer"])
        print(f"{name}: {dict(stats)}")

    if synthetic["samples"]:
        sidecar = (raw / f"{synthetic['corpus']}.jsonl").with_suffix(".source.json")
        if sidecar.exists():
            source_spec = read_json(sidecar)["identity"]["spec"]
            if source_spec.get("role") != "haystack_train" or source_spec.get("split") != "train":
                raise ValueError("Synthetic training must use a declared training haystack")
        pool = Haystack(raw / f"{synthetic['corpus']}.jsonl", tokenizer,
                        max_tokens=synthetic["max_corpus_tokens"], per_document_tokens=synthetic["per_document_tokens"],
                        excluded_hashes=heldout_hashes)
        pool_identity = pool.identity()
        provenance[synthetic["corpus"]] = {"sha256": pool.sha256, "role": "training_haystack"}
        if not synthetic["repeat_haystack"] and len(pool.tokens) < max(synthetic["lengths"]):
            raise ValueError("Training haystack is shorter than the largest requested sequence; add corpus text")

        def synthetic_rows():
            rng = random.Random(config["seed"])
            for index in range(synthetic["candidates"]):
                length = rng.choice(synthetic["lengths"])
                task, depth = rng.choice(synthetic["tasks"]), rng.uniform(0, 100)
                start = rng.randrange(max(1, len(pool.tokens) - length + 1))
                needle = make_split_needle(task, config["seed"] + index, "train")
                yield construct_sample(tokenizer, pool, length=length, depth=depth, needle=needle,
                                       start=start, sample_id=f"niah:train:synthetic:{index}",
                                       repeat=synthetic["repeat_haystack"])
        select_rows(synthetic_rows(), "synthetic", synthetic["samples"], config["seed"] + 1)
    for index, spec in enumerate(config["supervised"]):
        name, path = spec["source"], raw / f"{spec['source']}.jsonl"
        digest = sha256_file(path)
        sidecar = path.with_suffix(".source.json")
        if sidecar.is_file():
            identity = read_json(sidecar)
            if identity["sha256"] != digest or identity["identity"]["spec"]["split"] != "train":
                raise ValueError(f"Expected unmodified training split: {path}")
        provenance[name] = {"sha256": digest, "role": "supervised_train",
                            "source": read_json(sidecar) if sidecar.exists() else {"local_file": str(path)}}

        def supervised_rows():
            source = iter_jsonl(path)
            if config["candidate_limit"]:
                source = islice(source, config["candidate_limit"])
            for number, row in enumerate(source):
                if row.get("source_split", "train") != "train":
                    raise ValueError(f"Non-training row found in {path}")
                yield normalize_supervised(row, name, number, tokenizer)
        select_rows(supervised_rows(), name, spec["samples"], config["seed"] + index + 10)
    if not selected:
        raise ValueError("No NIAH training sources were enabled")
    random.Random(config["seed"]).shuffle(selected)
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", selected)
    write_json(output / "manifest.json", {"format": "seco-data-v1", "task": "niah", "kind": "train",
               "arguments": limits, "preparation": config, "sources": provenance,
               "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer), "statistics": statistics,
               "training_haystack": pool_identity, "answer_hashes": sorted(answer_hashes),
               "files": {"train.jsonl": sha256_file(output / "train.jsonl")}})
    print(f"Independent NIAH training data: {output}; {len(selected)} examples")


if __name__ == "__main__":
    main()
