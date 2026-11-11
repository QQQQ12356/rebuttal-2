"""Validation and provenance for the ICAE pretraining-to-SFT handoff."""
import math

from seco.experiment import memory_token_bound
from seco.io import project_path, read_json, sha256_file

from .pretraining import TASK


def validate_stage(model, task):
    pretraining = model.get("pretraining")
    if task != TASK:
        if pretraining is not None:
            raise ValueError("ICAE pretraining settings belong in the pretraining launcher")
        return
    if model.get("method") != "icae" or not isinstance(pretraining, dict) or set(pretraining) != {"ae_weight", "lm_weight"}:
        raise ValueError("ICAE pretraining requires ae_weight and lm_weight")
    values = list(pretraining.values())
    if (any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in values)
            or not 0 < sum(values) < math.inf):
        raise ValueError("Pretraining weights must be finite, nonnegative and have a finite positive sum")


def pretraining_limits(manifest, model, ratios):
    lengths = manifest["max_lengths"]
    encoder = min(lengths["input_ids"], model["memory_slots"] * int(max(ratios))) + model["memory_slots"]
    memory = max(memory_token_bound(model, lengths["input_ids"], ratio) for ratio in ratios)
    objectives = model["pretraining"]
    target = max(lengths[field] for name, field in (("ae", "ae_labels"), ("lm", "labels"))
                 if objectives[f"{name}_weight"] > 0)
    return encoder, memory + target


def initialization_checkpoint(path, model, tokenizer_identity):
    checkpoint = project_path(path).resolve()
    source = read_json(checkpoint / "experiment.json")
    if (source.get("task") != TASK or source.get("format") != "seco-adapters-v1"
            or (checkpoint / "artifact.json").exists()):
        raise ValueError("--init-from must name an ICAE pretraining adapter directory")
    pretrained = source["config"]["model"]
    validate_stage(pretrained, TASK)
    if not model.get("keep_ae_marker"):
        raise ValueError("Set model.keep_ae_marker=true when initializing from ICAE pretraining")
    if model["revision"] == "main" and model["name_or_path"] == pretrained["name_or_path"]:
        model["revision"] = pretrained["revision"]
    keys = ("name_or_path", "revision", "lora_r", "lora_alpha", "lora_target_modules",
            "memory_slots", "full_tail_slots")
    for key in keys:
        if model.get(key) != pretrained.get(key):
            raise ValueError(f"ICAE initialization requires matching model.{key}")
    if source["tokenizer_fingerprint"] != tokenizer_identity:
        raise ValueError("ICAE pretraining and SFT tokenizer fingerprints differ")
    return {"checkpoint": str(checkpoint), "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
            "experiment_sha256": sha256_file(checkpoint / "experiment.json")}
