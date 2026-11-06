"""Dataset adapters, shared token boundaries, deterministic sampling and padding."""
from __future__ import annotations

import html
import math
import random
from typing import Any

from .io import fingerprint


DATASETS = ("naturalqa", "2wikimqa", "hotpotqa", "narrativeqa", "multinews")
BENCHMARKS = ("naturalqa", "2wikimqa", "hotpotqa", "multinews")
QA_PROMPT = "\nQuestion: {question}\nAnswer the question using the context. Give only the answer.\nAnswer:"
NEWS_PROMPT = "\nNow, write a one-page summary of all the news. Summary:"


def records(value: Any) -> list[dict]:
    """Accept both Arrow's dict-of-lists and JSON's list-of-dicts."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        lengths = {len(v) for v in value.values() if isinstance(v, list)}
        if len(lengths) != 1 or not all(isinstance(v, list) for v in value.values()):
            raise ValueError("Expected a column-oriented sequence")
        return [dict(zip(value, values)) for values in zip(*value.values())]
    raise ValueError("Expected a sequence of records")


def answer_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict) and "text" in value:
        value = value["text"]
        if isinstance(value, str):
            value = [value]
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        value = [entry["text"] for entry in value]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError("No non-empty reference answers")
    return list(dict.fromkeys(value))


def context_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "title" in value:
        contents = value.get("sentences", value.get("content"))
        if contents is None or len(value["title"]) != len(contents):
            raise ValueError("Context titles and paragraphs do not align")
        value = list(zip(value["title"], contents))
    parts = []
    for entry in value:
        if isinstance(entry, dict):
            title, text = entry.get("title", ""), entry.get("text", "")
        else:
            title, text = entry
        text = " ".join(text) if isinstance(text, list) else text
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("Context title/text must be strings")
        parts.append(f"{title}\n{text}" if title else text)
    # Keep every passage in source order, including distractors. Never use isgold.
    return "\n\n".join(parts)


def natural_questions(row: dict) -> tuple[str, str, list[str]]:
    document = row["document"]
    tokens = records(document["tokens"])
    words = [html.unescape(token["token"]) for token in tokens]
    context = " ".join(word for word, token in zip(words, tokens) if not token["is_html"])
    answers = []
    for annotation in records(row["annotations"]):
        # HF Natural Questions ClassLabel: -1 = NONE, 0 = NO, 1 = YES.
        yes_no = annotation.get("yes_no_answer", -1)
        if yes_no in (0, 1, "NO", "YES"):
            answers.append("yes" if yes_no in (1, "YES") else "no")
        spans = records(annotation.get("short_answers", []))
        parts = []
        for span in spans:
            start, end = span["start_token"], span["end_token"]
            if 0 <= start < end <= len(words):
                parts.append(" ".join(word for word, token in zip(words[start:end], tokens[start:end])
                                      if not token["is_html"]))
        if parts:
            # Multiple short spans in one annotation form one answer, not aliases.
            answers.append(" ".join(parts))
    return context, row["question"]["text"], answer_list(answers)


def normalize_record(row: dict, dataset: str, source_index: int, source_split: str) -> dict:
    if all(key in row for key in ("input", "prompt", "answer")):
        context, prompt, answers = row["input"], row["prompt"], answer_list(row["answer"])
        question = row.get("question", prompt)
        if not isinstance(question, str):
            question = question["text"]
    elif "ctxs" in row:  # All four Twwilght benchmarks use this schema.
        context = context_text(row["ctxs"])
        question = row["question"]
        answers = answer_list(row["answers"])
        prompt = "\n" + question if dataset == "multinews" else QA_PROMPT.format(question=question)
    elif dataset == "naturalqa":
        context, question, answers = natural_questions(row)
        prompt = QA_PROMPT.format(question=question)
    elif dataset in ("2wikimqa", "hotpotqa"):
        context, question = context_text(row["context"]), row["question"]
        answers = answer_list(row["answer"])
        prompt = QA_PROMPT.format(question=question)
    elif dataset == "narrativeqa":
        # Use the full story, never silently substitute its human-written summary.
        context, question = row["document"]["text"], row["question"]["text"]
        answers = answer_list(row["answers"])
        prompt = QA_PROMPT.format(question=question)
    elif dataset == "multinews":
        context, question = row["document"], NEWS_PROMPT.strip()
        answers, prompt = answer_list(row["summary"]), NEWS_PROMPT
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not isinstance(context, str) or not context.strip() or not isinstance(prompt, str) or not prompt:
        raise ValueError("Empty context or prompt")
    question_key = fingerprint({"dataset": dataset, "question": " ".join(question.lower().split())})
    context_key = fingerprint(" ".join(context.split()))
    content_key = fingerprint({"input": context, "prompt": prompt})
    source_id = row.get("id", row.get("_id", source_index))
    return {"id": f"{dataset}:{source_split}:{source_id}", "dataset": dataset,
            "input": context, "prompt": prompt, "answer": answers,
            "question_key": question_key, "content_key": content_key, "context_key": context_key,
            "source_index": source_index, "source_split": source_split}


def tokenizer_fingerprint(tokenizer) -> str:
    # Includes fast tokenizer normalizer/merges and special-token definitions.
    return fingerprint({"backend": tokenizer.backend_tokenizer.to_str(),
                        "special_tokens": tokenizer.special_tokens_map})


def tokenize_record(row: dict, tokenizer) -> dict:
    context = tokenizer.encode(row["input"], add_special_tokens=False)
    prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
    answer = tokenizer.encode(row["answer"][0], add_special_tokens=False) + [tokenizer.eos_token_id]
    return {"input_ids": context + prompt, "prompt_mask": [0] * len(context) + [1] * len(prompt),
            "labels": answer, "context_tokens": len(context), "prompt_tokens": len(prompt),
            "answer_tokens": len(answer)}


def length_reason(tokens: dict, *, max_input_tokens: int, max_answer_tokens: int,
                  max_decoder_tokens: int, min_ratio: float, min_context_tokens: int) -> str | None:
    if tokens["context_tokens"] < min_context_tokens:
        return "short_context"
    if len(tokens["input_ids"]) > max_input_tokens:
        return "long_input"
    if tokens["answer_tokens"] > max_answer_tokens:
        return "long_answer"
    centers = min(tokens["context_tokens"], max(2, math.ceil(tokens["context_tokens"] / min_ratio)))
    if centers + tokens["prompt_tokens"] + tokens["answer_tokens"] > max_decoder_tokens:
        return "long_decoder"
    return None


def reservoir_add(sample: list, row: dict, seen: int, limit: int, rng: random.Random) -> None:
    if limit == 0 or len(sample) < limit:
        sample.append(row)
    else:
        index = rng.randrange(seen)
        if index < limit:
            sample[index] = row


class Collator:
    def __init__(self, pad_token_id: int, training: bool = True):
        self.pad_token_id, self.training = pad_token_id, training

    def __call__(self, features: list[dict]) -> dict:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in features]
        prompts = [torch.tensor(row["prompt_mask"], dtype=torch.bool) for row in features]
        lengths = torch.tensor([len(value) for value in ids])
        padded = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        batch = {"input_ids": padded, "prompt_mask": pad_sequence(prompts, batch_first=True),
                 "attention_mask": torch.arange(padded.size(1))[None, :] < lengths[:, None]}
        if self.training:
            batch["labels"] = pad_sequence([torch.tensor(row["labels"], dtype=torch.long)
                                             for row in features], batch_first=True, padding_value=-100)
        return batch
