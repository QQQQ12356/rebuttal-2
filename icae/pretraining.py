"""Prepare original ICAE reconstruction and continuation targets from documents."""
import argparse
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from seco.data import Collator, tokenizer_fingerprint
from seco.experiment import artifact_path, default_model_path
from seco.io import fingerprint, iter_jsonl, model_source, project_path, sha256_file, write_json, write_jsonl


TASK = "icae_pretrain"


def corpus_rows(paths):
    for path in paths:
        if path.suffix == ".jsonl":
            yield from iter_jsonl(path)
        else:
            from datasets import load_dataset

            yield from load_dataset("parquet", data_files=str(path), split="train", streaming=True)


def split_document(ids, bos_id, eos_id):
    middle = len(ids) // 2
    prefix = [bos_id, *ids[:middle]]
    return {"input_ids": prefix, "ae_labels": [*prefix, eos_id], "labels": [*ids[middle:], eos_id]}


class PretrainingCollator:
    def __init__(self, pad_token_id):
        self.collate = Collator(pad_token_id)

    def __call__(self, features):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        batch = self.collate([{**row, "prompt_mask": [0] * len(row["input_ids"])} for row in features])
        batch["ae_labels"] = pad_sequence([torch.tensor(row["ae_labels"], dtype=torch.long) for row in features],
                                          batch_first=True, padding_value=-100)
        return batch


def prepare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Ordered list of local JSONL or Parquet files")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default=artifact_path(TASK, "prepared/v1"))
    parser.add_argument("--min-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    args = parser.parse_args()
    if not 2 <= args.min_tokens <= args.max_tokens or args.train_samples < 1 or args.eval_samples < 0:
        parser.error("Use 2 <= min_tokens <= max_tokens, positive train_samples and nonnegative eval_samples")
    sources = [project_path(value).resolve() for value in args.input]
    if len(set(sources)) != len(sources) or any(
        path.suffix not in (".jsonl", ".parquet") or not path.is_file() for path in sources
    ):
        parser.error("--input requires distinct, existing .jsonl or .parquet files")
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(output)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source(args.model), revision=args.revision, use_fast=True)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("ICAE pretraining requires BOS and EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    maxima = {"input_ids": 0, "ae_labels": 0, "labels": 0}
    seen = set()

    def examples():
        for record in corpus_rows(sources):
            text = record.get(args.text_field)
            if not isinstance(text, str):
                raise ValueError(f"Corpus field {args.text_field!r} must contain text")
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=args.max_tokens + 1)
            if not args.min_tokens <= len(ids) <= args.max_tokens:
                continue
            row = split_document(ids, tokenizer.bos_token_id, tokenizer.eos_token_id)
            identity = fingerprint(row["input_ids"])
            if identity in seen:
                continue
            seen.add(identity)
            for field in maxima:
                maxima[field] = max(maxima[field], len(row[field]))
            yield row

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".icae-prepare-", dir=output.parent) as temporary:
        prepared = Path(temporary) / "prepared"
        stream = examples()
        files = {}
        counts = {"eval": args.eval_samples, "train": args.train_samples}
        for split, count in counts.items():
            if not count:
                continue
            name = f"{split}.jsonl"
            before = len(seen)
            write_jsonl(prepared / name, islice(stream, count))
            if len(seen) - before != count:
                raise ValueError(f"Corpus exhausted before collecting {count} unique {split} examples")
            files[name] = sha256_file(prepared / name)
        write_json(prepared / "manifest.json", {
            "format": "seco-data-v1", "task": TASK, "kind": "train",
            "arguments": vars(args), "counts": counts, "max_lengths": maxima, "files": files,
            "sources": [{"path": str(path), "sha256": sha256_file(path)} for path in sources],
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        })
        prepared.rename(output)
    print(f"ICAE pretraining data: {output} ({counts})")
