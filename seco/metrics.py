"""SQuAD-style normalized EM and bag-of-token F1, reported on a 0–1 scale."""
from __future__ import annotations

from collections import Counter, defaultdict
import re
import string


def normalize_answer(text: str) -> str:
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(prediction: str, references: list[str]) -> tuple[float, float]:
    if not references or not all(isinstance(ref, str) for ref in references):
        raise ValueError("Reference answers must be a non-empty list of strings")
    prediction = normalize_answer(prediction)
    pred_tokens = prediction.split()
    scores = []
    for reference in references:
        reference = normalize_answer(reference)
        ref_tokens = reference.split()
        common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if not pred_tokens or not ref_tokens:
            f1 = float(pred_tokens == ref_tokens)
        else:
            f1 = 2 * common / (len(pred_tokens) + len(ref_tokens))
        scores.append((float(prediction == reference), f1))
    return max(value[0] for value in scores), max(value[1] for value in scores)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty prediction file")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate prediction id: {row['id']}")
        seen.add(row["id"])
        em, f1 = score_answer(row["prediction"], row["answer"])
        groups[row["dataset"]].append((em, f1))
    subsets = {key: {"count": len(values), "em": sum(v[0] for v in values) / len(values),
                     "f1": sum(v[1] for v in values) / len(values)} for key, values in sorted(groups.items())}
    count = len(rows)
    return {"scale": "0-1", "count": count, "datasets": subsets,
            "micro": {metric: sum(value[metric] * value["count"] for value in subsets.values()) / count
                      for metric in ("em", "f1")},
            "macro": {metric: sum(value[metric] for value in subsets.values()) / len(subsets)
                      for metric in ("em", "f1")}}
