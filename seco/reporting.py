"""Convert scored benchmark artifacts to a reviewable Excel workbook on Linux."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re

from .io import iter_jsonl, read_json
from .metrics import score_answer


def metric_rows(path: Path) -> list[dict]:
    metrics = read_json(path)
    benchmark_path = path.parent.parent / "benchmark.json"
    metadata = read_json(benchmark_path) if benchmark_path.exists() else {}
    checkpoint = metadata.get("arguments", {}).get("checkpoint", "")
    experiment = metadata.get("experiment", {})
    model = experiment.get("config", {}).get("model", {})
    subsets = metrics.get("datasets", metrics.get("subset_summary"))
    if not isinstance(subsets, dict) or not subsets:
        raise ValueError(f"Expected datasets or subset_summary in {path}")
    rows = []
    for name, value in subsets.items():
        count = value["count"]
        em = value.get("em", value.get("avg_em"))
        f1 = value.get("f1", value.get("avg_f1"))
        if not isinstance(count, int) or count <= 0 or any(not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 1 for v in (em, f1)):
            raise ValueError(f"Invalid count/EM/F1 in {path}: {name}; expected scale 0–1")
        rows.append({"benchmark": str(path.parent.parent), "ratio": metrics.get("ratio", ""),
                     "dataset": name, "count": count, "em": em, "f1": f1,
                     "seconds_per_sample": metrics.get("mean_seconds_per_sample"),
                     "peak_allocated_gib": metrics.get("peak_allocated_gib"), "checkpoint": checkpoint,
                     "base_model": model.get("name_or_path", ""), "train_ratio_mode": model.get("ratio_mode", ""),
                     "train_ratios": json.dumps(model.get("random_ratios", [])),
                     "config_hash": experiment.get("config_hash", ""), "source": str(path)})
    for average in ("micro", "macro"):
        if average in metrics:
            rows.append({**rows[0], "dataset": f"__{average}__", "count": sum(row["count"] for row in rows
                          if not row["dataset"].startswith("__")),
                         "em": metrics[average]["em"], "f1": metrics[average]["f1"]})
    return rows


def export_excel(paths: list[Path], output: Path, *, include_details: bool = False) -> dict:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise ValueError("No matching metrics.json files")
    if output.exists():
        raise FileExistsError(f"Choose a new workbook path: {output}")
    workbook = Workbook(write_only=True)
    headers = ["benchmark", "ratio", "dataset", "count", "em", "f1", "seconds_per_sample",
               "peak_allocated_gib", "checkpoint", "base_model", "train_ratio_mode", "train_ratios",
               "config_hash", "source"]
    detail_headers = ["source", "id", "dataset", "ratio", "prediction", "answer", "em", "f1", "prompt", "input",
                      "field", "part", "text"]
    runs = defaultdict(list)
    for path in paths:
        runs[path.parent.parent].append(path)
    sheet_titles = {"history"}  # Reserved by Excel.
    wide_columns = {"source", "prediction", "answer", "prompt", "input", "text"}

    def new_sheet(run):
        base = re.sub(r"[\x00-\x1f\[\]:*?/\\]", "_", run.name).strip("'") or "Benchmark"
        suffix = ""
        version = 1
        while True:
            title = base[:31 - len(suffix)].rstrip("'") + suffix
            if title.casefold() not in sheet_titles:
                break
            version += 1
            suffix = f"_{version}"
        sheet_titles.add(title.casefold())
        sheet = workbook.create_sheet(title)
        sheet.freeze_panes = "A2"
        for index, header in enumerate(headers):
            wide = header in wide_columns or (include_details and index < len(detail_headers)
                                             and detail_headers[index] in wide_columns)
            sheet.column_dimensions[get_column_letter(index + 1)].width = 55 if wide else 20
        return sheet

    def append(sheet, columns, values, *, header=False):
        cells = []
        for column, value in zip(columns, values):
            cell = WriteOnlyCell(sheet, value=value)
            if isinstance(value, str):
                # Treat model output and context as literal data, including '='.
                cell.data_type = "s"
            if header:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="24476A")
            elif column in ("em", "f1"):
                cell.number_format = "0.00%"
            elif column in ("seconds_per_sample", "peak_allocated_gib"):
                cell.number_format = "0.000"
            cells.append(cell)
        sheet.append(cells)

    def text_value(value, field, chunks):
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value if value is not None else "")
        # XML 1.0 cannot represent these control characters; preserve visible escapes.
        text = "".join(f"\\u{ord(char):04x}" if ord(char) < 32 and char not in "\t\n\r" else char for char in text)
        if len(text) > 30000:
            for index in range(0, len(text), 30000):
                chunks.append((field, index // 30000 + 1, text[index:index + 30000]))
            return f"[Full text in rows below: field={field}]"
        return text

    def prediction_rows(run_paths):
        for path in run_paths:
            prediction_path = path.parent / "predictions.jsonl"
            if not prediction_path.is_file():
                continue
            for index, row in enumerate(iter_jsonl(prediction_path)):
                prediction = row.get("prediction", row.get("model_output"))
                answers = row.get("answer", row.get("ground_truth"))
                answers = [answers] if isinstance(answers, str) else answers
                em, f1 = score_answer(prediction, answers)
                detail = {"source": str(prediction_path), "id": row.get("id", f"external:{index}"),
                          "dataset": row.get("dataset", row.get("subset", "unknown")),
                          "ratio": row.get("ratio", ""), "prediction": prediction,
                          "answer": answers, "em": em, "f1": f1,
                          "prompt": row.get("prompt", ""), "input": row.get("input", "")}
                chunks = []
                for field in ("id", "prediction", "answer", "prompt", "input"):
                    detail[field] = text_value(detail[field], field, chunks)
                yield detail
                for field, part, text in chunks:
                    yield {"source": detail["source"], "id": detail["id"], "ratio": detail["ratio"],
                           "field": field, "part": part, "text": text}

    total = details = 0
    for run, run_paths in runs.items():
        sheet = new_sheet(run)
        used_rows = 0

        def write_section(columns, rows):
            nonlocal sheet, used_rows
            needs_header = True
            count = 0
            for row in rows:
                # Keep each header with at least one data row.
                if used_rows + (3 if needs_header else 1) > 1048576:
                    sheet = new_sheet(run)
                    used_rows = 0
                    needs_header = True
                if needs_header:
                    if used_rows:
                        sheet.append([])
                        used_rows += 1
                    append(sheet, columns, columns, header=True)
                    used_rows += 1
                    needs_header = False
                append(sheet, columns, [row.get(column) for column in columns])
                used_rows += 1
                if "field" not in row:
                    count += 1
            return count

        summary = (row for path in run_paths for row in metric_rows(path))
        total += write_section(headers, summary)
        if include_details:
            details += write_section(detail_headers, prediction_rows(run_paths))
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return {"metrics_files": len(paths), "summary_rows": total, "prediction_rows": details}
