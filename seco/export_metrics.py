"""Export one NIAH or RULER QA benchmark run to Excel."""

import argparse
from decimal import Decimal
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .io import project_path, read_json


METHODS = ("seco", "comi", "icae", "ram")
SCORES = {
    "niah": ("accuracy", "em", "f1", "hit_rate"),
    "ruler_qa": ("ruler_score", "em", "f1"),
}
RATIO_NAME = re.compile(r"ratio_(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
RUNTIME_FIELDS = ("mean_seconds_per_sample", "peak_allocated_gib")


def write_table(sheet, headers, rows, *, metric_names=()):
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    sheet.append(headers)
    for cell in sheet[sheet.max_row]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="24476A")
    for index, header in enumerate(headers, start=1):
        column = sheet.column_dimensions[get_column_letter(index)]
        column.width = max(column.width, 16, len(header) + 3)

    for values in rows:
        sheet.append(values)
        for cell, header in zip(sheet[sheet.max_row], headers):
            if isinstance(cell.value, str):
                cell.data_type = "s"
            metric = values[0] if header == "Value" else header
            if metric in metric_names:
                cell.number_format = "0.00%"
            elif metric in RUNTIME_FIELDS:
                cell.number_format = "0.0000"


def write_metrics_sheet(sheet, metrics, ratio, task):
    scores = SCORES[task]
    summary = metrics["summary"]
    fields = ("split", "count", "expected_count", "empty_count", "complete", *RUNTIME_FIELDS)
    overall = {**metrics, **summary}
    write_table(sheet, ("Metric", "Value"), [
        ("ratio", metrics.get("ratio", float(ratio))),
        *((key, summary[key]) for key in scores if key in summary),
        *((key, overall[key]) for key in fields if key in overall),
    ], metric_names=scores)
    sheet.column_dimensions["A"].width = max(sheet.column_dimensions["A"].width, 29)
    sheet.freeze_panes = "A2"

    for group, label in (("sources", "source"), ("tasks", "task")):
        entries = metrics.get(group)
        if entries:
            columns = ("count", "empty_count") if task == "ruler_qa" else ("count",)
            columns += scores
            sheet.append([])
            write_table(sheet, (label, *columns),
                        ([name, *(row.get(key) for key in columns)] for name, row in sorted(entries.items())),
                        metric_names=scores)

    if cells := metrics.get("cells"):
        dimensions = (("sequence_length", "depth_percent", "count", "expected_count") if task == "niah"
                      else ("task", "target_length", "count", "empty_count"))
        columns = (*dimensions, *scores)
        sheet.append([])
        write_table(sheet, columns, ([row.get(key) for key in columns] for row in cells), metric_names=scores)


def export_run(run: Path, output: Path, *, method: str, task: str) -> int:
    from openpyxl import Workbook

    if output.suffix.lower() != ".xlsx":
        raise ValueError(f"Output path must end in .xlsx: {output}")
    files = []
    for path in run.glob("ratio_*/metrics.json"):
        match = RATIO_NAME.fullmatch(path.parent.name)
        if match and path.is_file():
            files.append((Decimal(match[1]), match[1], path))
    if not files:
        raise ValueError(f"No ratio_*/metrics.json files found in {run}")

    workbook = Workbook()
    workbook.remove(workbook.active)
    for ratio, label, path in sorted(files):
        title = f"{method}-{task}-ratio-{label}"
        if len(title) > 31:
            raise ValueError(f"Excel sheet name exceeds 31 characters: {title}")
        metrics = read_json(path)
        summary = metrics.get("summary") if isinstance(metrics, dict) else None
        if not isinstance(summary, dict) or not any(key in summary for key in SCORES[task]):
            raise ValueError(f"Expected a {task} summary in {path}")
        write_metrics_sheet(workbook.create_sheet(title), metrics, ratio, task)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".metrics-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "metrics.xlsx"
        workbook.save(staged)
        os.replace(staged, output)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Run directory containing ratio_*/metrics.json")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--task", choices=SCORES, required=True)
    parser.add_argument("--output", help="Default: metrics.xlsx in the benchmark directory")
    args = parser.parse_args()
    run = project_path(args.benchmark_dir)
    output = project_path(args.output) if args.output else run / "metrics.xlsx"
    try:
        count = export_run(run, output, method=args.method, task=args.task)
    except ModuleNotFoundError as error:
        if error.name != "openpyxl":
            raise
        parser.error("Install openpyxl with: python -m pip install -r requirements-tools.txt")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported {count} sheets to {output}")


if __name__ == "__main__":
    main()
