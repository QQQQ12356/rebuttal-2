"""Convert scored benchmark artifacts to a reviewable Excel workbook on Linux."""
from __future__ import annotations

import json
import math
from pathlib import Path

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

    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise ValueError("No matching metrics.json files")
    if output.exists():
        raise FileExistsError(f"Choose a new workbook path: {output}")
    workbook = Workbook(write_only=True)
    headers = ["benchmark", "ratio", "dataset", "count", "em", "f1", "seconds_per_sample",
               "peak_allocated_gib", "checkpoint", "base_model", "train_ratio_mode", "train_ratios",
               "config_hash", "source"]
    detail_headers = ["source", "id", "dataset", "ratio", "prediction", "answer", "em", "f1", "prompt", "input"]
    long_headers = ["source", "id", "field", "part", "text"]
    sheets = {}
    counters = {}

    def append(kind, columns, values):
        if kind not in sheets or counters[kind] >= 1048576:
            version = 1 if kind not in sheets else int(sheets[kind].title.rsplit("_", 1)[-1]) + 1
            sheet = workbook.create_sheet(f"{kind}_{version}")
            sheet.freeze_panes = "A2"
            for index, header in enumerate(columns):
                from openpyxl.utils import get_column_letter
                sheet.column_dimensions[get_column_letter(index + 1)].width = 20 if header not in ("prediction", "answer", "prompt", "input", "source") else 55
            header_cells = []
            for header in columns:
                cell = WriteOnlyCell(sheet, value=header)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="24476A")
                header_cells.append(cell)
            sheet.append(header_cells)
            sheets[kind], counters[kind] = sheet, 1
        cells = []
        for column, value in zip(columns, values):
            cell = WriteOnlyCell(sheets[kind], value=value)
            if isinstance(value, str):
                # Treat model output and context as literal data, including '='.
                cell.data_type = "s"
            if column in ("em", "f1"):
                cell.number_format = "0.00%"
            elif column in ("seconds_per_sample", "peak_allocated_gib"):
                cell.number_format = "0.000"
            cells.append(cell)
        sheets[kind].append(cells)
        counters[kind] += 1

    def text_value(value, source, row_id, field):
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value if value is not None else "")
        # XML 1.0 cannot represent these control characters; preserve visible escapes.
        text = "".join(f"\\u{ord(char):04x}" if ord(char) < 32 and char not in "\t\n\r" else char for char in text)
        if len(text) > 30000:
            for index in range(0, len(text), 30000):
                append("LongText", long_headers, [source, row_id, field, index // 30000 + 1, text[index:index + 30000]])
            return f"[Full text in LongText sheets: id={row_id}, field={field}]"
        return text

    total = details = 0
    for path in paths:
        rows = metric_rows(path)
        for row in rows:
            append("Summary", headers, [row.get(header) for header in headers])
            total += 1
        prediction_path = path.parent / "predictions.jsonl"
        if include_details and prediction_path.exists():
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
                for key in ("id", "prediction", "answer", "prompt", "input"):
                    detail[key] = text_value(detail[key], str(prediction_path), row.get("id", index), key)
                append("Predictions", detail_headers, [detail[header] for header in detail_headers])
                details += 1
    append("Readme", ["item", "meaning"], ["EM / F1", "Stored as numbers on a 0–1 scale; percentage display in Excel."])
    append("Readme", ["item", "meaning"], ["__micro__ / __macro__", "micro: weighted by sample count; macro: unweighted dataset mean."])
    append("Readme", ["item", "meaning"], ["Timing", "Amortized batch time/sample, not individual request latency or distributed throughput."])
    append("Readme", ["item", "meaning"], ["LongText", "Text over 30,000 characters is split in order; control characters use literal Unicode escapes."])
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return {"metrics_files": len(paths), "summary_rows": total, "prediction_rows": details}
