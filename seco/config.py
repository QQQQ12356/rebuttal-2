"""Validate experiment settings before allocating model memory."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .io import model_source, project_path, read_json
from .experiment import check_method, validate_ratios


def load_config(path: str | Path) -> dict:
    config = copy.deepcopy(read_json(path))
    model, train = config["model"], config["training"]
    check_method(model)
    method = model.get("method", "seco")
    positive_keys = {"comi": ("encoder_layers", "segment_size"), "icae": ("memory_slots",),
                     "ram": ("encoder_layers", "segment_batch_size")}.get(method, ())
    for key in positive_keys:
        if type(model.get(key)) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if method in ("comi", "ram") and (type(model.get("fusion_layers")) is not int or model["fusion_layers"] < 0):
        raise ValueError("model.fusion_layers must be a nonnegative integer")
    if method == "ram" and not 0 <= model.get("keep_ratio", -1) < 1:
        raise ValueError("model.keep_ratio must be in [0, 1)")
    if method == "ram":
        for key, default, minimum in (("contrastive_loss_weight", 0, 0), ("contrastive_temperature", 0.1, 1e-8)):
            value = model.get(key, default)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"model.{key} must be finite and >= {minimum}")
    if model["ratio_mode"] not in ("fixed", "random"):
        raise ValueError("ratio_mode must be fixed or random")
    ratios = [model["compress_ratio"], *model["random_ratios"]]
    if not ratios or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 1 for v in ratios):
        raise ValueError("Compression ratios must be finite numbers >= 1")
    validate_ratios(model, ratios)
    if model["ratio_mode"] == "random" and not model["random_ratios"]:
        raise ValueError("random_ratios cannot be empty in random mode")
    if model["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported dtype")
    if model["attn_implementation"] not in ("eager", "sdpa", "flash_attention_2"):
        raise ValueError("Unsupported attention backend")
    if not isinstance(model.get("similarity_chunk_tokens", 16), int) or model.get("similarity_chunk_tokens", 16) < 0:
        raise ValueError("similarity_chunk_tokens must be an integer >= 0")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "save_steps", "logging_steps"):
        if train[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if train.get("eval_strategy", "no") != "no":
        raise ValueError("Use the separate benchmark command for evaluation; eval_strategy must be no")
    if train.get("load_best_model_at_end", False):
        raise ValueError("load_best_model_at_end requires a validation loop and is unsupported")
    forbidden = {"output_dir", "deepspeed", "bf16", "fp16", "remove_unused_columns", "save_only_model"}
    if forbidden.intersection(train):
        raise ValueError(f"Training fields are controlled by the launcher: {sorted(forbidden.intersection(train))}")
    config["data"]["prepared_dir"] = str(project_path(config["data"]["prepared_dir"]))
    config["model"]["name_or_path"] = model_source(config["model"]["name_or_path"])
    return config


def deepspeed_config(backend: str):
    if backend == "ddp":
        return None
    if backend not in ("zero2", "zero2_cpu", "zero3_cpu"):
        raise ValueError(f"Unknown distributed backend: {backend}")
    return str(project_path(f"configs/{backend}.json"))
