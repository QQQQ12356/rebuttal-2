"""Publication-style length/depth heatmaps using a fixed red–yellow–green scale."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from seco.io import read_json


def plot_heatmaps(paths: list[Path], output: Path, *, metric: str = "accuracy", columns: int = 2,
                  title: str = "SeCo Needle-in-a-Haystack", dpi: int = 300,
                  allow_incomplete: bool = False, annotate_all: bool = False, pdf: bool = True) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import numpy as np

    if metric not in ("accuracy", "em", "f1", "hit_rate") or columns < 1 or dpi < 1:
        raise ValueError("Invalid plot metric/columns/dpi")
    if not paths:
        raise ValueError("No NIAH metric files")
    records = [read_json(path) for path in paths]
    outputs = [output, output.with_suffix(".csv")]
    if pdf:
        outputs.append(output.with_suffix(".pdf"))
    if output.suffix.lower() != ".png" or any(path.exists() for path in outputs):
        raise ValueError("Use a new .png output path; related PNG/PDF/CSV outputs must not exist")
    reference = records[0]["grid"]
    for value in records:
        if value.get("format") != "seco-niah-metrics-v1":
            raise ValueError("Not a NIAH metrics artifact")
        if not value["complete"] and not allow_incomplete:
            raise ValueError("Incomplete grid: explicitly set allow_incomplete to plot observed cells")
        if any(value["grid"][key] != reference[key] for key in ("lengths", "depths", "repeats", "task", "seed", "corpus")):
            raise ValueError("Panels must use the same task, seed, corpus, length/depth axes and repeats")
        if value.get("grid_identity") != records[0].get("grid_identity"):
            raise ValueError("Panels refer to different prepared grid identities")
    columns = min(columns, len(records))
    rows = math.ceil(len(records) / columns)
    width = max(7, len(reference["lengths"]) * 0.65 + 2)
    height = max(4.5, len(reference["depths"]) * 0.32 + 1.8)
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                               squeeze=False, constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("niah", ["#F0496E", "#EBB839", "#0CD79F"])
    cmap.set_bad("#D9D9D9")
    normalizer = Normalize(0, 100)
    used_axes, exported = [], []
    for axis, value, source in zip(axes.flat, records, paths):
        lookup = {(cell["sequence_length"], cell["depth_percent"]): cell for cell in value["cells"]}
        if len(lookup) != len(reference["lengths"]) * len(reference["depths"]):
            raise ValueError("Missing or duplicated heatmap cell definitions")
        matrix = np.full((len(reference["depths"]), len(reference["lengths"])), np.nan)
        for y, depth in enumerate(reference["depths"]):
            for x, length in enumerate(reference["lengths"]):
                cell = lookup[(length, depth)]
                score = cell[metric]
                if score is not None:
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("NIAH metrics must use scale 0–1")
                    matrix[y, x] = score * 100
                partial = cell["count"] < cell["expected_count"]
                annotation = "NA" if score is None else f"{score * 100:.0f}"
                if partial:
                    annotation += f"\n{cell['count']}/{cell['expected_count']}"
                if annotate_all or partial or score is None or score < 1:
                    axis.text(x, y, annotation, ha="center", va="center", fontsize=9, color="#202020")
                exported.append({"source": str(source), "ratio": value.get("ratio", ""), **cell})
        image = axis.pcolormesh(np.arange(len(reference["lengths"]) + 1) - 0.5,
                                np.arange(len(reference["depths"]) + 1) - 0.5,
                                np.ma.masked_invalid(matrix), cmap=cmap, norm=normalizer,
                                shading="flat", edgecolors="white", linewidth=0.6)
        axis.set_ylim(len(reference["depths"]) - 0.5, -0.5)
        axis.set_xticks(range(len(reference["lengths"])), [str(length) for length in reference["lengths"]], rotation=45, ha="right")
        axis.set_yticks(range(len(reference["depths"])), [f"{depth:g}" for depth in reference["depths"]])
        axis.set_xticks(np.arange(-0.5, len(reference["lengths"]), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(reference["depths"]), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.6)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xlabel("Sequence Length (tokens)")
        axis.set_ylabel("Needle Depth (%)")
        axis.set_title(f"SeCo | R={value.get('ratio', '?'):g}" if isinstance(value.get("ratio"), (int, float)) else "SeCo")
        used_axes.append(axis)
    for axis in list(axes.flat)[len(records):]:
        axis.set_visible(False)
    labels = {"accuracy": "Accuracy (EM, %)", "em": "Exact Match (%)", "f1": "F1 (%)", "hit_rate": "Answer Hit Rate (%)"}
    figure.colorbar(image, ax=used_axes, label=labels[metric], ticks=[0, 20, 40, 60, 80, 100], shrink=0.9)
    figure.suptitle(title + (" [INCOMPLETE]" if any(not value["complete"] for value in records) else ""), fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    if pdf:
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    return outputs
