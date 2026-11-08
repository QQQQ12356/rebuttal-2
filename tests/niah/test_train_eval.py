from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from seco.io import sha256_file, write_json, write_jsonl
from seco.niah.benchmark import prediction_metadata
from seco.niah.metrics import aggregate_training, score_prediction
from seco.niah.train_eval import load_training_rows


class TrainingEvaluationTests(unittest.TestCase):
    def test_training_manifest_binding_and_deterministic_cap(self):
        with TemporaryDirectory() as directory:
            prepared = Path(directory)
            data = prepared / "train.jsonl"
            manifest = prepared / "manifest.json"
            write_jsonl(data, ({"id": str(index)} for index in range(8)))
            write_json(manifest, {"format": "seco-data-v1", "task": "niah", "kind": "train",
                                 "tokenizer_fingerprint": "fixture", "preparation": {"seed": 42},
                                 "files": {"train.jsonl": sha256_file(data)}})
            experiment = {"task": "niah", "data_manifest_sha256": sha256_file(manifest),
                          "tokenizer_fingerprint": "fixture"}
            _, _, rows, total = load_training_rows(prepared, experiment, 3)
            self.assertEqual(total, 8)
            self.assertEqual(len({row["id"] for row in rows}), 3)
            self.assertEqual(rows, load_training_rows(prepared, experiment, 3)[2])
            self.assertEqual(len(load_training_rows(prepared, experiment, None)[2]), total)
            with self.assertRaisesRegex(ValueError, "same NIAH training manifest"):
                load_training_rows(prepared, {**experiment, "data_manifest_sha256": "other"}, 3)
            write_jsonl(data, [{"id": "changed"}])
            with self.assertRaisesRegex(ValueError, "training data hash mismatch"):
                load_training_rows(prepared, experiment, 3)

    def test_training_scores_include_cogito_without_grid_fields(self):
        rows = [{"id": "one", "task": "passkey", "answer": ["0123"], "input_ids": [1], "prompt": "q"},
                {"id": "two", "task": "bind", "answer": ["Bob"], "input_ids": [1, 2], "prompt": "q",
                 "source": "cogito_bind"}]
        predictions = [{**prediction_metadata(row, True), **score_prediction(prediction, row["answer"])}
                       for row, prediction in zip(rows, ("0123", "Alice"))]
        metrics = aggregate_training(predictions)
        self.assertEqual(metrics["split"], "train")
        self.assertEqual(metrics["summary"]["em"], 0.5)
        self.assertEqual(metrics["sources"]["synthetic"]["em"], 1.0)
        self.assertEqual(metrics["sources"]["cogito_bind"]["count"], 1)
        self.assertEqual(metrics["tasks"]["bind"]["f1"], 0.0)
