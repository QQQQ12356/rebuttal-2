"""RULER QA substring scoring plus normalized EM/F1 on a fixed grid."""
from collections import Counter, defaultdict
import re

from seco.io import fingerprint
from seco.metrics import score_answer
from seco.ruler_qa.data import TASKS


METRICS = ("ruler_score", "em", "f1")


def ratio_directory(ratio):
    return f"ratio_{str(ratio).removesuffix('.0')}"


def score_prediction(prediction, references):
    cleaned = re.sub(r"[\x00-\x1f]", "\n", prediction).strip()
    em, f1 = score_answer(cleaned, references)
    hit = any(answer.lower() in cleaned.lower() for answer in references)
    return {"ruler_score": int(hit), "em": em, "f1": f1, "empty": int(not cleaned)}


def validate_grid(rows, manifest):
    split = manifest.get("kind", "test")
    groups = defaultdict(list)
    for row in rows:
        expected_id = f"{row['task']}:{split}:{row['source_id']}:{row['target_length']}"
        if row["id"] != expected_id:
            raise ValueError(f"Grid row identity does not match its task, split, source ID and length: {row['id']}")
        groups[row["task"]].append(row)
    expected_tasks = set(manifest["counts"]) if "prepared_test_identity" in manifest else set(TASKS)
    if not expected_tasks or not expected_tasks.issubset(TASKS) or set(groups) != expected_tasks:
        raise ValueError("Grid tasks do not match the expected RULER QA tasks")
    for task, items in groups.items():
        ids = {row["id"] for row in items}
        counts = Counter(str(row["target_length"]) for row in items)
        if (len(ids) != len(items) or counts != manifest["counts"][task]
                or fingerprint(sorted(ids)) != manifest["sample_id_hashes"][task]):
            raise ValueError(f"{task}: missing, duplicate, or unexpected grid rows")


def aggregate(rows, manifest, ratio):
    validate_grid(rows, manifest)
    cells, tasks, overall = defaultdict(list), defaultdict(list), []
    for row in rows:
        if row["ratio"] != ratio:
            raise ValueError("Prediction file contains an unexpected compression ratio")
        result = score_prediction(row["prediction"], row["answer"])
        cells[row["task"], row["target_length"]].append(result)
        tasks[row["task"]].append(result)
        overall.append(result)

    def averages(items):
        return {"count": len(items), "empty_count": sum(row["empty"] for row in items),
                **{metric: sum(row[metric] for row in items) / len(items) for metric in METRICS}}

    return {"task": "ruler_qa", "ratio": ratio, "scale": "0-1", "split": manifest.get("kind", "test"),
            "test_identity": fingerprint(manifest),
            "summary": averages(overall), "tasks": {task: averages(items) for task, items in tasks.items()},
            "cells": [{"task": task, "target_length": target, **averages(items)}
                      for (task, target), items in sorted(cells.items())]}
