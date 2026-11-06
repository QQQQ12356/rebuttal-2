from pathlib import Path
import tempfile
import unittest

from seco.io import write_jsonl
from seco.niah.config import validate_grid
from seco.niah.data import Haystack, construct_sample, make_needle, normalize_supervised


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) + 3 for char in text]
        return ids[:max_length] if truncation else ids

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token - 3) for token in tokens)


class NeedleDataTests(unittest.TestCase):
    def test_lengths_depth_endpoints_and_full_needles_survive(self):
        tokenizer = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            write_jsonl(path, [{"id": "book", "text": "A quiet background passage. " * 300}])
            pool = Haystack(path, tokenizer, max_tokens=8000, per_document_tokens=8000)
            needle = make_needle("passkey", 42, "test")
            for depth in (0, 50, 100):
                row = construct_sample(tokenizer, pool, length=512, depth=depth, needle=needle,
                                       start=7, sample_id=str(depth), include_text=True)
                self.assertEqual(len(row["input_ids"]), 512)
                self.assertEqual(sum(row["prompt_mask"]), row["prompt_tokens"])
                self.assertEqual(row["input"].count(needle["needle"]), 1)
                self.assertEqual(row["answer"], [needle["answer"]])
                begin = row["needle_start_token"]
                end = begin + row["needle_tokens"]
                self.assertEqual(tokenizer.decode(row["input_ids"][begin:end]), needle["needle"])
                if depth == 0:
                    self.assertEqual(begin, row["context_tokens"] - row["haystack_tokens"] - row["needle_tokens"])
                if depth == 100:
                    self.assertEqual(end, row["context_tokens"])

    def test_splits_have_different_answers_even_for_same_seed(self):
        first = make_needle("uuid", 42, "niah-train-v1")
        second = make_needle("uuid", 42, "niah-test-v1")
        self.assertEqual(first, make_needle("uuid", 42, "niah-train-v1"))
        self.assertNotEqual(first["answer"], second["answer"])

    def test_short_haystack_never_repeats_silently(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            write_jsonl(path, [{"text": "abcd"}])
            pool = Haystack(path, CharacterTokenizer(), max_tokens=10, per_document_tokens=10)
            with self.assertRaises(ValueError):
                pool.window(8, 0)
            self.assertEqual(pool.window(8, 0, repeat=True)[0], [100, 101, 102, 103] * 2)

    def test_supervised_rows_are_retokenized_and_holdout_rejected(self):
        row = {"context": "Facts", "query": "Q alice", "answer": "red", "input_ids": [9999]}
        result = normalize_supervised(row, "bits", 1, CharacterTokenizer())
        self.assertEqual(result["input_ids"][:5], [73, 100, 102, 119, 118])
        self.assertEqual(result["labels"][-1], 2)
        row["split"] = "test"
        with self.assertRaises(ValueError):
            normalize_supervised(row, "bits", 1, CharacterTokenizer())

    def test_duplicate_grid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            validate_grid({"lengths": [512, 512], "depths": [0, 100]})
