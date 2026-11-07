from pathlib import Path
import tempfile
import unittest

from seco.io import write_json, write_jsonl
from seco.reporting import export_excel


class ReportingTests(unittest.TestCase):
    def test_scores_and_literal_prediction_content_survive_excel_export(self):
        from openpyxl import load_workbook

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "ratio_4/metrics.json", {"ratio": 4, "datasets": {
                "naturalqa": {"count": 1, "em": 0.0, "f1": 0.5}}})
            write_jsonl(root / "ratio_4/predictions.jsonl", [{"id": "sample", "dataset": "naturalqa",
                "prediction": "=1+1", "answer": ["2"], "input": "x" * 40000}])
            output = root / "report.xlsx"
            result = export_excel([root / "ratio_4/metrics.json"], output, include_details=True)
            workbook = load_workbook(output)
            self.assertEqual(result["prediction_rows"], 1)
            self.assertEqual(workbook["Summary_1"]["F2"].value, 0.5)
            self.assertEqual(workbook["Predictions_1"]["E2"].data_type, "s")
            self.assertEqual(workbook["Predictions_1"]["E2"].value, "=1+1")
            long_sheet = workbook["LongText_1"]
            self.assertEqual(long_sheet["E2"].value + long_sheet["E3"].value, "x" * 40000)
            workbook.close()
