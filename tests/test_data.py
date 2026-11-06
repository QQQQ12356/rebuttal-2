"""Pure adapters and sampling, run on Linux without GPU/downloads."""
import random
import unittest

from seco.data import normalize_record, records, reservoir_add, length_reason


class DatasetTests(unittest.TestCase):
    def test_twwilght_keeps_distractors_and_answer_aliases(self):
        row = {"question": "Where?", "answers": ["Paris", "Paris, France"],
               "ctxs": [{"title": "Other", "text": "London", "isgold": False},
                        {"title": "France", "text": "Paris", "isgold": True}]}
        result = normalize_record(row, "naturalqa", 1, "test")
        self.assertEqual(result["input"], "Other\nLondon\n\nFrance\nParis")
        self.assertEqual(result["answer"], ["Paris", "Paris, France"])

    def test_multihop_supports_both_context_layouts(self):
        for field in ("sentences", "content"):
            row = {"question": "Who?", "answer": "Alice", "context": {"title": ["Title"], field: [["One.", "Two."]]}}
            result = normalize_record(row, "2wikimqa", 0, "train")
            self.assertEqual(result["input"], "Title\nOne. Two.")

    def test_nq_multi_span_annotation_is_one_reference(self):
        row = {"question": {"text": "Who?"},
               "document": {"tokens": {"token": ["<p>", "Larry", "and", "Sergey", "</p>"],
                                        "is_html": [True, False, False, False, True]}},
               "annotations": [{"yes_no_answer": -1, "short_answers": [
                   {"start_token": 1, "end_token": 2}, {"start_token": 3, "end_token": 4}]}]}
        result = normalize_record(row, "naturalqa", 0, "train")
        self.assertEqual(result["input"], "Larry and Sergey")
        self.assertEqual(result["answer"], ["Larry Sergey"])
        row["annotations"] = [{"yes_no_answer": 0, "short_answers": []}]
        self.assertEqual(normalize_record(row, "naturalqa", 0, "train")["answer"], ["no"])

    def test_narrativeqa_uses_story_not_summary(self):
        row = {"document": {"text": "Full story", "summary": {"text": "Spoiler"}},
               "question": {"text": "Who?"}, "answers": [{"text": "Alice"}, {"text": "A woman"}]}
        result = normalize_record(row, "narrativeqa", 0, "train")
        self.assertEqual(result["input"], "Full story")
        self.assertEqual(result["answer"], ["Alice", "A woman"])

    def test_news_constant_instruction_does_not_define_content_identity(self):
        first = normalize_record({"document": "First news", "summary": "First summary"}, "multinews", 0, "train")
        second = normalize_record({"document": "Other news", "summary": "Other summary"}, "multinews", 1, "train")
        self.assertNotEqual(first["context_key"], second["context_key"])

    def test_length_filter_counts_prompt_and_decoder_budget(self):
        tokens = {"input_ids": list(range(16)), "context_tokens": 12, "prompt_tokens": 4, "answer_tokens": 5}
        limits = dict(max_input_tokens=16, max_answer_tokens=8, max_decoder_tokens=11,
                      min_ratio=4, min_context_tokens=1)
        self.assertEqual(length_reason(tokens, **limits), "long_decoder")
        limits["max_decoder_tokens"] = 12
        self.assertIsNone(length_reason(tokens, **limits))

    def test_reservoir_reproducible_and_bounded(self):
        samples = []
        for _ in range(2):
            sample, rng = [], random.Random(42)
            for number in range(100):
                reservoir_add(sample, {"id": number}, number + 1, 7, rng)
            samples.append(sample)
        self.assertEqual(samples[0], samples[1])
        self.assertEqual(len({row["id"] for row in samples[0]}), 7)

    def test_arrow_empty_sequence(self):
        self.assertEqual(records({"start_token": [], "end_token": []}), [])
