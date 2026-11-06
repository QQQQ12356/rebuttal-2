"""Build a fixed, paired length × needle-depth × trial benchmark without truncating needles."""
import argparse
import random

from niah_bootstrap import require_linux
from seco.data import length_reason, tokenizer_fingerprint
from seco.io import fingerprint, model_source, project_path, read_json, sha256_file, write_json, write_jsonl
from seco.niah.config import validate_grid
from seco.niah.data import Haystack, construct_sample, make_needle


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/niah/grid.json")
    parser.add_argument("--raw-dir", default="artifacts/niah/raw")
    parser.add_argument("--output", default="artifacts/niah/prepared/grid_v1")
    parser.add_argument("--model", default="artifacts/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--training-manifest", help="Optional explicit training manifest for overlap checks")
    args = parser.parse_args()
    config = read_json(project_path(args.config))
    validate_grid(config)
    from transformers import AutoTokenizer

    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new NIAH evaluation grid directory: {output}")
    tokenizer = AutoTokenizer.from_pretrained(model_source(args.model), revision=args.revision, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pool = Haystack(project_path(args.raw_dir) / f"{config['corpus']}.jsonl", tokenizer,
                    max_tokens=config["max_corpus_tokens"], per_document_tokens=config["per_document_tokens"])
    if not config["repeat_haystack"] and len(pool.tokens) < max(config["lengths"]):
        raise ValueError("Held-out corpus too short for largest grid length; add text or explicitly enable repeat_haystack")
    train_manifest_path = project_path(args.training_manifest) if args.training_manifest else project_path("artifacts/niah/prepared/train_v1/manifest.json")
    train_manifest = read_json(train_manifest_path) if train_manifest_path.exists() else None
    if args.training_manifest and train_manifest is None:
        raise FileNotFoundError(train_manifest_path)
    if train_manifest and (train_manifest.get("task"), train_manifest.get("kind")) != ("niah", "train"):
        raise ValueError("Overlap guard requires a NIAH training manifest")
    training_pool = train_manifest.get("training_haystack") if train_manifest else None
    if training_pool:
        training_hashes = {row["text_sha256"] for row in training_pool["documents"]}
        if training_hashes.intersection(row["text_sha256"] for row in pool.documents):
            raise ValueError("Evaluation haystack overlaps training documents")
    blocked_answers = set(train_manifest.get("answer_hashes", [])) if train_manifest else set()
    limits = {"max_input_tokens": max(config["lengths"]), "max_answer_tokens": config["generation_budget"],
              "max_decoder_tokens": config["max_decoder_tokens"], "min_context_tokens": 1,
              "min_ratio": config["min_ratio"]}
    count = 0

    def rows():
        nonlocal count
        for trial in range(config["repeats"]):
            seed = config["seed"] + trial
            needle = make_needle(config["task"], seed, "niah-test-v1", config.get("custom"))
            if fingerprint(needle["answer"]) in blocked_answers:
                raise ValueError("Evaluation answer was used in NIAH training; change evaluation seed")
            # Same needle and haystack starting point across all lengths/depths in
            # a trial; trials use independent seeds. A depth sweep is not IID trials.
            start = random.Random(seed).randrange(max(1, len(pool.tokens) - max(config["lengths"]) + 1))
            for length in config["lengths"]:
                for depth in config["depths"]:
                    row = construct_sample(tokenizer, pool, length=length, depth=depth, needle=needle,
                                           start=start, sample_id=f"niah:test:{trial}:{length}:{depth:g}",
                                           repeat=config["repeat_haystack"], include_text=config["include_text"])
                    check = {**row, "answer_tokens": config["generation_budget"]}
                    if row["answer_tokens"] - 1 > config["generation_budget"]:
                        raise ValueError("Reference answer cannot fit the generation budget")
                    reason = length_reason(check, **limits)
                    if reason:
                        raise ValueError(f"Grid cell exceeds generation/decoder budget ({reason}): {row['id']}")
                    row["trial"], row["seed"] = trial, seed
                    count += 1
                    yield row

    output.mkdir(parents=True)
    write_jsonl(output / "grid.jsonl", rows())
    write_json(output / "manifest.json", {"format": "seco-niah-grid-v1", "task": "niah", "kind": "test",
               "grid": config, "count": count, "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
               "haystack": pool.identity(), "length_definition": "complete encoder input_ids including description, needle and prompt",
               "depth_definition": "insertion offset divided by available haystack tokens; 0=before, 100=after background",
               "training_manifest_sha256": sha256_file(train_manifest_path) if train_manifest else None,
               "files": {"grid.jsonl": sha256_file(output / "grid.jsonl")}})
    print(f"NIAH grid complete: {count} examples; {output}")


if __name__ == "__main__":
    main()
