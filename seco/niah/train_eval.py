"""Read the exact prepared NIAH training split for explicit in-sample evaluation."""
import random

from seco.data import reservoir_add
from seco.io import iter_jsonl, read_json, sha256_file


def load_training_rows(prepared, experiment, limit):
    manifest_path = prepared / "manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("format"), manifest.get("task"), manifest.get("kind")) != ("seco-data-v1", "niah", "train"):
        raise ValueError("Expected a prepared NIAH training split")
    manifest_hash = sha256_file(manifest_path)
    if experiment.get("task") != "niah" or experiment.get("data_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint and --prepared-dir must refer to the same NIAH training manifest")
    if experiment.get("tokenizer_fingerprint") != manifest["tokenizer_fingerprint"]:
        raise ValueError("Training data tokenizer differs from the checkpoint tokenizer")
    data_path = prepared / "train.jsonl"
    if sha256_file(data_path) != manifest["files"]["train.jsonl"]:
        raise ValueError("NIAH training data hash mismatch")
    rng = random.Random(manifest["preparation"]["seed"])
    rows, seen = [], set()
    for count, row in enumerate(iter_jsonl(data_path), 1):
        if row["id"] in seen:
            raise ValueError(f"Duplicate NIAH training sample: {row['id']}")
        seen.add(row["id"])
        reservoir_add(rows, row, count, limit or 0, rng)
    if not rows:
        raise ValueError("No prepared NIAH training samples")
    return manifest, manifest_hash, rows, len(seen)
