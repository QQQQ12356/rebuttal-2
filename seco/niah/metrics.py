"""Locally reproducible EM/F1 and complete length-by-depth cell aggregation."""
from __future__ import annotations

import re

from seco.metrics import score_answer


def score_prediction(prediction: str, answers: list[str]) -> dict:
    em, f1 = score_answer(prediction, answers)
    hit = any(re.search(r"(?<![A-Za-z0-9])" + re.escape(answer.strip()) + r"(?![A-Za-z0-9])",
                        prediction, flags=re.IGNORECASE) is not None for answer in answers)
    return {"accuracy": em, "em": em, "f1": f1, "hit_rate": float(hit)}


def aggregate_training(rows: list[dict]) -> dict:
    """Summarize scored training rows without imposing a held-out grid shape."""
    if not rows:
        raise ValueError("No NIAH training predictions")
    by_source, by_task = {}, {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
        by_task.setdefault(row["task"], []).append(row)

    def summarize(items):
        return {"count": len(items),
                **{metric: sum(row[metric] for row in items) / len(items)
                   for metric in ("accuracy", "em", "f1", "hit_rate")}}

    return {"format": "seco-niah-train-metrics-v1", "scale": "0-1", "split": "train", "count": len(rows),
            "summary": summarize(rows),
            "sources": {source: summarize(items) for source, items in sorted(by_source.items())},
            "tasks": {task: summarize(items) for task, items in sorted(by_task.items())}}


def aggregate(rows: list[dict], grid: dict, *, allow_incomplete: bool = False) -> dict:
    lengths, depths, repeats = grid["lengths"], grid["depths"], grid["repeats"]
    expected = {(length, depth, trial) for length in lengths for depth in depths for trial in range(repeats)}
    seen, ids, groups, scored = set(), set(), {}, []
    for row in rows:
        key = (row["sequence_length"], row["depth_percent"], row["trial"])
        if key not in expected or key in seen or row["id"] in ids:
            raise ValueError(f"Unexpected/duplicate NIAH sample: {row['id']}")
        seen.add(key)
        ids.add(row["id"])
        if row.get("status", "ok") != "ok":
            raise ValueError(f"Failed prediction must be resolved before scoring: {row['id']}")
        scores = score_prediction(row["prediction"], row["answer"])
        scored.append(scores)
        groups.setdefault(key[:2], []).append(scores)
    if seen != expected and not allow_incomplete:
        raise ValueError(f"Incomplete NIAH grid: expected {len(expected)}, got {len(seen)}")
    if not rows:
        raise ValueError("No NIAH predictions")
    cells = []
    for depth in depths:
        for length in lengths:
            values = groups.get((length, depth), [])
            cells.append({"sequence_length": length, "depth_percent": depth, "count": len(values),
                          "expected_count": repeats,
                          **{metric: sum(row[metric] for row in values) / len(values) if values else None
                             for metric in ("accuracy", "em", "f1", "hit_rate")}})
    return {"format": "seco-niah-metrics-v1", "scale": "0-1", "grid": grid, "count": len(rows),
            "expected_count": len(expected), "complete": seen == expected,
            "summary": {metric: sum(row[metric] for row in scored) / len(scored)
                        for metric in ("accuracy", "em", "f1", "hit_rate")}, "cells": cells}
