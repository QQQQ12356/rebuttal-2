import unittest

from seco.metrics import score_answer, summarize


class MetricTests(unittest.TestCase):
    def test_normalization_aliases_and_partial_overlap(self):
        self.assertEqual(score_answer("The Paris!", ["London", "Paris"]), (1.0, 1.0))
        self.assertEqual(score_answer("red green", ["red blue"]), (0.0, 0.5))
        self.assertEqual(score_answer("", [""]), (1.0, 1.0))

    def test_macro_differs_from_micro(self):
        rows = [{"id": "1", "dataset": "A", "prediction": "yes", "answer": ["yes"]},
                {"id": "2", "dataset": "A", "prediction": "no", "answer": ["yes"]},
                {"id": "3", "dataset": "B", "prediction": "yes", "answer": ["yes"]}]
        metrics = summarize(rows)
        self.assertAlmostEqual(metrics["micro"]["em"], 2 / 3)
        self.assertEqual(metrics["macro"]["em"], 0.75)

    def test_duplicate_shards_are_rejected(self):
        row = {"id": "1", "dataset": "A", "prediction": "yes", "answer": ["yes"]}
        with self.assertRaises(ValueError):
            summarize([row, row])
