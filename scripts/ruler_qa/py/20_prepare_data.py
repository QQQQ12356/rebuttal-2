"""Prepare a SeCo training corpus or a held-out RULER QA length grid."""
import argparse
from collections import Counter
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from ruler_bootstrap import require_linux
from seco.data import tokenizer_fingerprint
from seco.io import fingerprint, model_source, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.ruler_qa.data import TASKS, build_rows, read_questions, sample_train, validate_settings


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--config", default="configs/ruler_qa/prepare.json")
    parser.add_argument("--raw-dir", default="artifacts/ruler_qa/raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    settings = read_json(project_path(args.config))
    validate_settings(settings)
    output, raw = project_path(args.output), project_path(args.raw_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new prepared-data directory: {output}")
    sources = {filename: sha256_file(raw / filename) for paths in TASKS.values()
               for split, filename in paths.items() if split == "test" or args.split == "train"}

    from transformers import AutoConfig, AutoTokenizer

    model = model_source(args.model)
    config = AutoConfig.from_pretrained(model, revision=args.revision)
    revision = getattr(config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if max(max(settings[f"{args.split}_lengths"]), settings["max_decoder_tokens"]) > config.max_position_embeddings:
        raise ValueError("Configured token limits exceed the model position limit")
    if args.split == "train":
        blocked = {question.key for task, paths in TASKS.items()
                   for question in read_questions(raw / paths["test"], task)}

    counts, rejected, identities = {}, {}, {}
    train_keys = set()
    filename = f"{args.split}.jsonl"

    def generate():
        for task, paths in TASKS.items():
            print(f"Preparing {task}/{args.split} from {paths[args.split]}", flush=True)
            candidates = read_questions(raw / paths[args.split], task)
            if args.split == "train":
                questions = sample_train(candidates, blocked=blocked | train_keys,
                                         limit=settings["train_candidates_per_task"],
                                         rng=random.Random(f"{settings['seed']}:{task}:candidate"))
            else:
                questions = list(candidates)
            rows, rejected[task] = build_rows(questions, tokenizer, settings, task=task, split=args.split)
            counts[task] = dict(Counter(str(row["target_length"]) for row in rows))
            identities[task] = fingerprint(sorted(row["id"] for row in rows))
            if args.split == "train":
                train_keys.update(row["question_key"] for row in rows)
            print(f"{task}: {counts[task]}; rejected={rejected[task]}", flush=True)
            yield from rows

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ruler-qa-", dir=output.parent) as directory:
        payload = Path(directory) / "data"
        write_jsonl(payload / filename, generate())
        local_download = Path(model) / "download.json"
        manifest = {"task": "ruler_qa", "kind": args.split,
                    "format": "seco-data-v1" if args.split == "train" else "seco-ruler-qa-test-v1",
                    "arguments": {**settings, "max_input_tokens": max(settings[f"{args.split}_lengths"])},
                    "model": {"name_or_path": model, "revision": revision},
                    "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer), "sources": sources,
                    "counts": counts, "sample_id_hashes": identities, "rejected": rejected,
                    "files": {filename: sha256_file(payload / filename)}}
        if local_download.is_file():
            manifest["model"]["download"] = read_json(local_download)
        if args.split == "train":
            manifest["question_keys"] = sorted(train_keys)
        write_json(payload / "manifest.json", manifest)
        payload.rename(output)
    print(f"Prepared {args.split}: {output}")


if __name__ == "__main__":
    main()
