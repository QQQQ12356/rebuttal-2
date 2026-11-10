"""Deterministic needles, exact-token insertion and independent haystack windows."""
from __future__ import annotations

from array import array
from bisect import bisect_right
import math
import random
import uuid

from seco.data import answer_list, tokenize_record
from seco.io import fingerprint, iter_jsonl, read_json, sha256_file


DESCRIPTION = "Read the following documents carefully and retrieve the requested information.\n\n"
PROMPT = "\n\nQuestion: {question}"


def make_needle(task: str, seed: int, namespace: str, custom: dict | None = None) -> dict:
    if task == "custom":
        if not custom or any(not isinstance(custom.get(key), str) or not custom[key].strip()
                             for key in ("needle", "question", "answer")):
            raise ValueError("custom task requires non-empty needle/question/answer")
        if custom["answer"].casefold() not in custom["needle"].casefold():
            raise ValueError("Custom reference answer must occur inside the needle")
        return {**custom, "task": task}
    if task not in ("uuid", "passkey"):
        raise ValueError(f"Unsupported needle task: {task}")
    rng = random.Random(int(fingerprint([namespace, seed, task]), 16))
    key = f"archive-{rng.getrandbits(64):016x}"
    answer = str(uuid.UUID(int=rng.getrandbits(128), version=4)) if task == "uuid" else f"{rng.randrange(10**8):08d}"
    return {"task": task, "needle": f"\n\nThe access value for {key} is {answer}.\n\n",
            "question": f"What is the access value for {key}?", "answer": answer, "key": key}


CUSTOM_DESTINATIONS = {
    "train": (
        "the orchid conservatory", "the maritime museum", "the riverside gardens",
        "the butterfly pavilion", "the hilltop observatory", "the sculpture courtyard",
        "the railway museum", "the lakeside nature center",
    ),
    "test": (
        "the tidal aquarium", "the woodland planetarium", "the pottery gallery",
        "the cliffside lighthouse", "the aviation museum", "the fern greenhouse",
        "the riverside amphitheater", "the glassblowing studio",
    ),
}


def make_custom_needle(seed: int, split: str) -> dict:
    namespace = f"niah-{split}-v1"
    rng = random.Random(int(fingerprint([namespace, seed, "custom"]), 16))
    center = rng.choice(("harbor visitor center", "riverside information desk",
                         "old town tourism office", "lakeside welcome center"))
    time = rng.choice(("at dawn", "after lunch", "in the afternoon", "at sunset"))
    answer = rng.choice(CUSTOM_DESTINATIONS[split])
    return make_needle("custom", seed, namespace, {
        "needle": f"\n\nThe {center} recommends visiting {answer} {time}.\n\n",
        "question": f"Which place does the {center} recommend visiting {time}?",
        "answer": answer,
    })


def make_split_needle(task: str, seed: int, split: str, custom: dict | None = None) -> dict:
    """Use the same generated needle format for training and evaluation."""
    if task == "custom" and custom is None:
        return make_custom_needle(seed, split)
    return make_needle(task, seed, f"niah-{split}-v1", custom)


class Haystack:
    """Compact token pool. Never repeats a corpus unless explicitly requested."""
    def __init__(self, path, tokenizer, *, max_tokens: int, per_document_tokens: int,
                 excluded_hashes: set[str] | None = None):
        self.tokens = array("I")
        self.documents = []
        self.path, self.sha256 = str(path), sha256_file(path)
        sidecar = path.with_suffix(".source.json")
        if sidecar.is_file() and read_json(sidecar)["sha256"] != self.sha256:
            raise ValueError(f"Haystack raw file hash mismatch: {path}")
        seen = set()
        separator = tokenizer.encode("\n\n", add_special_tokens=False)
        for index, row in enumerate(iter_jsonl(path)):
            if len(self.tokens) >= max_tokens:
                break
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                continue
            digest = fingerprint(" ".join(text.split()))
            if digest in seen or digest in (excluded_hashes or set()):
                continue
            seen.add(digest)
            remaining = max_tokens - len(self.tokens) - (len(separator) if self.tokens else 0)
            if remaining <= 0:
                break
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                   max_length=min(per_document_tokens, remaining))
            if not ids:
                continue
            if self.tokens:
                self.tokens.extend(separator)
            start = len(self.tokens)
            self.tokens.extend(ids)
            self.documents.append({"id": str(row.get("id", index)), "text_sha256": digest,
                                   "start": start, "end": len(self.tokens)})
        if not self.tokens:
            raise ValueError(f"No haystack tokens: {path}")
        self.ends = [row["end"] for row in self.documents]

    def window(self, length: int, start: int, *, repeat: bool = False) -> tuple[list[int], list[str]]:
        if length < 0 or start < 0:
            raise ValueError("Window length/start must be non-negative")
        if not repeat and start + length > len(self.tokens):
            raise ValueError("Haystack pool is too short; download more text or explicitly enable repeat_haystack")
        if repeat:
            ids = [self.tokens[(start + index) % len(self.tokens)] for index in range(length)]
            documents = [row["id"] for row in self.documents]
        else:
            ids = self.tokens[start:start + length].tolist()
            first = bisect_right(self.ends, start)
            documents = [row["id"] for row in self.documents[first:] if row["start"] < start + length]
        return ids, documents

    def identity(self) -> dict:
        from pathlib import Path
        sidecar = Path(self.path).with_suffix(".source.json")
        return {"path": self.path, "sha256": self.sha256, "tokens": len(self.tokens),
                "documents": self.documents,
                "source": read_json(sidecar) if sidecar.exists() else {"local_file": self.path}}


def construct_sample(tokenizer, haystack: Haystack, *, length: int, depth: float,
                     needle: dict, start: int, sample_id: str, repeat: bool = False,
                     include_text: bool = False) -> dict:
    if not math.isfinite(depth) or not 0 <= depth <= 100:
        raise ValueError("Needle depth must be finite and within 0–100")
    description = tokenizer.encode(DESCRIPTION, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle["needle"], add_special_tokens=False)
    prompt_text = PROMPT.format(question=needle["question"])
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    budget = length - len(description) - len(needle_ids) - len(prompt_ids)
    if budget < 1:
        raise ValueError(f"Requested length {length} cannot fit description + needle + prompt")
    background, document_ids = haystack.window(budget, start, repeat=repeat)
    background_text = tokenizer.decode(background, skip_special_tokens=False)
    if needle["answer"].casefold() in background_text.casefold() or needle["needle"].strip().casefold() in background_text.casefold():
        raise ValueError("Needle answer already occurs in the background; choose a different seed/window")
    offset = min(budget, max(0, round(budget * depth / 100)))
    context = description + background[:offset] + needle_ids + background[offset:]
    inputs = context + prompt_ids
    if len(inputs) != length:
        raise AssertionError("Construction did not preserve the requested encoder length")
    labels = tokenizer.encode(needle["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
    row = {"id": sample_id, "dataset": "niah", "task": needle["task"],
           "input_ids": inputs, "prompt_mask": [0] * len(context) + [1] * len(prompt_ids),
           "labels": labels, "prompt": prompt_text, "answer": [needle["answer"]],
           "context_tokens": len(context), "prompt_tokens": len(prompt_ids), "answer_tokens": len(labels),
           "sequence_length": length, "depth_percent": depth,
           "actual_depth_percent": 100 * offset / budget,
           "needle_start_token": len(description) + offset, "needle_tokens": len(needle_ids),
           "needle": needle["needle"], "haystack_start": start, "haystack_tokens": budget,
           "source_document_ids": document_ids, "repeat_haystack": repeat}
    if include_text:
        row["input"] = tokenizer.decode(context, skip_special_tokens=False)
    return row


def normalize_supervised(row: dict, source: str, index: int, tokenizer) -> dict:
    if "split" in row and row["split"] != "train":
        raise ValueError("Only declared training rows may be used for supervised NIAH training")
    context, query = row["context"], row["query"]
    if not isinstance(context, str) or not context.strip() or not isinstance(query, str) or not query.strip():
        raise ValueError("Empty supervised context/query")
    record = {"id": f"{source}:{row.get('id', index)}", "dataset": "niah", "source": source,
              "task": row.get("task", source), "input": context,
              "prompt": PROMPT.format(question=query), "answer": answer_list(row["answer"])}
    record.update(tokenize_record(record, tokenizer))
    return record
