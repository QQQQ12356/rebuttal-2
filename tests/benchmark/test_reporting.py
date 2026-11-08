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
            for run, f1 in (("test_a", 0.5), ("test_b", 0.75)):
                for ratio in (4, 8):
                    write_json(root / run / f"ratio_{ratio}/metrics.json", {"ratio": ratio, "datasets": {
                        "naturalqa": {"count": 1, "em": 0.0, "f1": f1}}})
            write_jsonl(root / "test_a/ratio_4/predictions.jsonl", [{"id": "sample", "dataset": "naturalqa",
                "prediction": "=1+1", "answer": ["2"], "input": "x" * 40000}])
            output = root / "report.xlsx"
            result = export_excel(list(root.glob("*/ratio_*/metrics.json")), output, include_details=True)
            workbook = load_workbook(output)
            try:
                self.assertEqual(result, {"metrics_files": 4, "summary_rows": 4, "prediction_rows": 1})
                self.assertEqual(workbook.sheetnames, ["test_a", "test_b"])
                for run, f1 in (("test_a", 0.5), ("test_b", 0.75)):
                    sheet = workbook[run]
                    self.assertEqual([sheet["B2"].value, sheet["B3"].value], [4, 8])
                    self.assertEqual([sheet["F2"].value, sheet["F3"].value], [f1, f1])
                details = workbook["test_a"]
                self.assertEqual(details["E6"].data_type, "s")
                self.assertEqual(details["E6"].value, "=1+1")
                self.assertEqual(details["M7"].value + details["M8"].value, "x" * 40000)
                self.assertEqual(workbook["test_b"].max_row, 3)
            finally:
                workbook.close()
