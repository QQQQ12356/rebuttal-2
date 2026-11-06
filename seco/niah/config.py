"""Validate NIAH dimensions and budgets before creating output artifacts."""
from __future__ import annotations

import math


def positive(value, name: str, *, allow_zero: bool = False):
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer {'>= 0' if allow_zero else '> 0'}")


def validate_limits(limits: dict):
    for key in ("max_input_tokens", "max_answer_tokens", "max_decoder_tokens"):
        positive(limits[key], key)
    positive(limits["min_context_tokens"], "min_context_tokens", allow_zero=True)
    ratio = limits["min_ratio"]
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 1:
        raise ValueError("min_ratio must be finite and >= 1")


def validate_grid(config: dict):
    lengths, depths = config["lengths"], config["depths"]
    if not lengths or len(set(lengths)) != len(lengths) or lengths != sorted(lengths):
        raise ValueError("lengths must be a non-empty strictly increasing array")
    for length in lengths:
        positive(length, "length")
    if not depths or len(set(depths)) != len(depths) or depths != sorted(depths):
        raise ValueError("depths must be a non-empty strictly increasing array")
    if any(not isinstance(depth, (int, float)) or not math.isfinite(depth) or not 0 <= depth <= 100 for depth in depths):
        raise ValueError("depths must lie within 0–100")
    for key in ("repeats", "generation_budget", "max_decoder_tokens", "max_corpus_tokens", "per_document_tokens"):
        positive(config[key], key)
    validate_limits({"max_input_tokens": max(lengths), "max_answer_tokens": config["generation_budget"],
                     "max_decoder_tokens": config["max_decoder_tokens"], "min_context_tokens": 1,
                     "min_ratio": config["min_ratio"]})
    if config["task"] not in ("uuid", "passkey", "custom"):
        raise ValueError("task must be uuid, passkey or custom")
