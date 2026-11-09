"""Read RULER's source datasets and build length-controlled QA examples."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
import math
import random

from seco.data import answer_list, length_reason, reservoir_add
from seco.io import fingerprint


TASKS = {
    "qa_1": {"train": "train-v2.0.json", "test": "dev-v2.0.json"},
    "qa_2": {"train": "hotpot_train_v1.1.json", "test": "hotpot_dev_distractor_v1.json"},
}
PROMPT_STYLES = ("raw", "legacy")
LEGACY_INTRO = "Use the documents below to answer the question. Give only the answer."


@dataclass(frozen=True)
class Question:
    source_id: str
    text: str
    answers: list[str]
    required: tuple[str, ...]
    related: tuple[str, ...] = ()

    @property
    def key(self):
        return fingerprint(" ".join(self.text.casefold().split()))


def read_questions(path, task):
    import ijson

    with path.open("rb") as handle:
        if task == "qa_1":
            for article in ijson.items(handle, "data.item"):
                documents = tuple(p["context"] for p in article["paragraphs"])
                for paragraph in article["paragraphs"]:
                    evidence = paragraph["context"]
                    for row in paragraph["qas"]:
                        if not row["is_impossible"]:
                            yield Question(row["id"], row["question"], answer_list(row["answers"]),
                                           (evidence,), documents)
        else:
            for row in ijson.items(handle, "item"):
                documents = tuple(f"{title}\n{''.join(sentences)}" for title, sentences in row["context"])
                yield Question(row["_id"], row["question"], answer_list(row["answer"]), documents)


def sample_train(questions, *, blocked, limit, rng):
    sample, seen = [], set()
    for question in questions:
        key = question.key
        if key not in blocked and key not in seen:
            seen.add(key)
            reservoir_add(sample, question, len(seen), limit, rng)
    return sample


def validate_settings(settings):
    if settings.get("prompt_style", "raw") not in PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}")
    for key in ("train_lengths", "test_lengths"):
        values = settings[key]
        if not values or any(type(n) is not int or n < 1 for n in values) or len(set(values)) != len(values):
            raise ValueError(f"{key} must contain distinct positive integer token budgets")
    for key in ("train_candidates_per_task", "train_samples_per_task", "test_samples_per_task",
                "max_decoder_tokens", "max_answer_tokens", "generation_budget"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["train_candidates_per_task"] < settings["train_samples_per_task"]:
        raise ValueError("train_candidates_per_task must be at least train_samples_per_task")
    if settings["train_samples_per_task"] < len(settings["train_lengths"]):
        raise ValueError("train_samples_per_task must cover every train length")
    if not 0 < settings["min_fill"] <= 1 or not math.isfinite(settings["min_ratio"]) or settings["min_ratio"] < 1:
        raise ValueError("Require 0 < min_fill <= 1 and finite min_ratio >= 1")
    if type(settings["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if min(settings["test_lengths"]) <= settings["generation_budget"]:
        raise ValueError("Test lengths must exceed generation_budget")


def build_rows(questions, tokenizer, settings, *, task, split):
    """Use complete documents, retaining all evidence before adding distractors."""
    pool = tuple(dict.fromkeys(doc for q in questions for doc in chain(q.required, q.related)))
    rng = random.Random(f"{settings['seed']}:{split}:{task}")
    rng.shuffle(questions)
    encode = lru_cache(maxsize=8192)(lambda text: tokenizer.encode(text, add_special_tokens=False))
    legacy_prompt = settings.get("prompt_style", "raw") == "legacy"
    intro = encode(LEGACY_INTRO) if legacy_prompt else []
    lengths = settings[f"{split}_lengths"]
    limit = settings[f"{split}_samples_per_task"]
    rows, rejected, accepted_keys = [], Counter(), set()

    def build(question, target):
        test = split == "test"
        prompt_text = f"\n\nQuestion: {question.text}\nAnswer:" if legacy_prompt else f"\n{question.text}"
        prompt = encode(prompt_text)
        labels = encode(question.answers[0]) + [tokenizer.eos_token_id] if not test else []
        reserve = settings["generation_budget"] if test else len(labels)
        input_budget = target - reserve
        # Fit the decoder at the lowest permitted ratio, including its two-center floor.
        decoder_room = settings["max_decoder_tokens"] - len(prompt) - reserve
        if decoder_room < 2:
            return None, "long_decoder"
        context_budget = min(input_budget - len(prompt), math.floor(decoder_room * settings["min_ratio"]))
        required = list(dict.fromkeys(question.required))
        documents = [encode("\n\nDocument:\n" + doc) for doc in required]
        size = len(intro) + sum(map(len, documents))
        if size > context_budget:
            return None, "long_required_context"
        if not test and len(labels) > settings["max_answer_tokens"]:
            return None, "long_answer"

        sample_rng = random.Random(f"{settings['seed']}:{split}:{task}:{question.source_id}")
        seen = set(required)
        related = list(dict.fromkeys(question.related))
        sample_rng.shuffle(related)
        minimum = math.ceil(settings["min_fill"] * target)
        desired = max(minimum, target - min(64, target // 200)) - len(prompt) - reserve
        # Bound candidate draws even when no remaining document fits.
        extra = (pool[sample_rng.randrange(len(pool))] for _ in range(max(128, target // 8)))
        for doc in chain(related, extra):
            if size >= desired:
                break
            if doc in seen:
                continue
            seen.add(doc)
            tokens = encode("\n\nDocument:\n" + doc)
            if size + len(tokens) <= context_budget:
                documents.append(tokens)
                size += len(tokens)
        if size + len(prompt) + reserve < minimum:
            return None, "low_fill"
        sample_rng.shuffle(documents)
        context = intro + list(chain.from_iterable(documents))
        row = {"id": f"{task}:{split}:{question.source_id}:{target}", "task": task,
               "source_id": question.source_id, "question_key": question.key, "question": question.text,
               "answer": question.answers, "target_length": target, "context_tokens": len(context),
               "prompt_tokens": len(prompt), "input_ids": context + prompt,
               "prompt_mask": [0] * len(context) + [1] * len(prompt)}
        tokens = {**row, "answer_tokens": reserve}
        reason = length_reason(tokens, max_input_tokens=input_budget,
                               max_answer_tokens=reserve if test else settings["max_answer_tokens"],
                               max_decoder_tokens=settings["max_decoder_tokens"],
                               min_ratio=settings["min_ratio"], min_context_tokens=1)
        if reason:
            return None, reason
        if not test:
            row["labels"] = labels
        return row, None

    for question in questions:
        if question.key in accepted_keys:
            continue
        targets = lengths if split == "test" else [lengths[len(accepted_keys) % len(lengths)]]
        examples = []
        for target in targets:
            row, reason = build(question, target)
            if reason:
                rejected[f"{target}:{reason}"] += 1
                break
            examples.append(row)
        else:
            rows.extend(examples)
            accepted_keys.add(question.key)
            if len(accepted_keys) == limit:
                return rows, dict(rejected)
    raise ValueError(f"{task}/{split}: accepted {len(accepted_keys)}/{limit}; rejected={dict(rejected)}. "
                     "Increase the candidate pool, reduce quotas or relax length limits.")
