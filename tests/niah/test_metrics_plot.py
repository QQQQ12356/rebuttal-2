from pathlib import Path
import tempfile
import unittest

from seco.io import write_json
from seco.niah.metrics import aggregate, score_prediction
from seco.niah.plotting import plot_heatmaps


GRID = {"lengths": [512, 1024], "depths": [0, 100], "repeats": 2,
        "task": "passkey", "seed": 7, "corpus": "toy"}


def predictions():
    return [{"id": f"{length}:{depth}:{trial}", "sequence_length": length, "depth_percent": depth,
             "trial": trial, "prediction": "12345678" if trial == 0 else "wrong", "answer": ["12345678"]}
            for length in GRID["lengths"] for depth in GRID["depths"] for trial in range(2)]


class NeedleMetricTests(unittest.TestCase):
    def test_accuracy_is_empirical_success_rate(self):
        metrics = aggregate(predictions(), GRID)
        self.assertTrue(metrics["complete"])
        self.assertEqual(metrics["summary"]["accuracy"], 0.5)
        self.assertTrue(all(cell["accuracy"] == 0.5 and cell["count"] == 2 for cell in metrics["cells"]))

    def test_duplicate_and_incomplete_shards_fail(self):
        with self.assertRaises(ValueError):
            aggregate(predictions() + [predictions()[0]], GRID)
        with self.assertRaises(ValueError):
            aggregate(predictions()[:2], GRID)
        partial = aggregate(predictions()[:2], GRID, allow_incomplete=True)
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["cells"][-1]["accuracy"])

    def test_contains_does_not_accept_partial_numeric_matches(self):
        self.assertEqual(score_prediction("The value is 12345678.", ["12345678"])["hit_rate"], 1.0)
        self.assertEqual(score_prediction("9123456789", ["12345678"])["hit_rate"], 0.0)

    def test_plot_writes_png_pdf_and_auditable_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = {**aggregate(predictions(), GRID), "ratio": 4}
            write_json(root / "metrics.json", metrics)
            outputs = plot_heatmaps([root / "metrics.json"], root / "NIAH.png", dpi=60)
            self.assertEqual({path.suffix for path in outputs}, {".png", ".pdf", ".csv"})
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))
            import csv
            with (root / "NIAH.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["accuracy"]), 0.5)
