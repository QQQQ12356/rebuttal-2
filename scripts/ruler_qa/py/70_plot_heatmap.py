"""Plot a pair of RULER QA heatmaps and export numeric cells for all metrics."""
import argparse
import csv
from io import StringIO

from ruler_bootstrap import require_linux
from seco.io import fingerprint, project_path, read_json, write_text
from seco.ruler_qa.data import TASKS
from seco.ruler_qa.metrics import METRICS, ratio_directory


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output", help="PNG path; defaults to BENCHMARK_DIR/ruler_qa_heatmap.png")
    parser.add_argument("--metric", choices=METRICS, default="ruler_score")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    run = project_path(args.benchmark_dir)
    output = project_path(args.output) if args.output else run / "ruler_qa_heatmap.png"
    if output.suffix.lower() != ".png" or args.dpi < 1:
        parser.error("Use a .png output and positive DPI")
    metadata = read_json(run / "benchmark.json")
    manifest = metadata["test_manifest"]
    identity = fingerprint(manifest)
    ratios = sorted(metadata["arguments"]["ratios"])
    split = manifest.get("kind", "test")
    lengths = sorted(manifest["arguments"][f"{split}_lengths"])
    expected = {(task, length) for task in TASKS for length in lengths}
    if "prepared_test_identity" in manifest:
        expected = {(task, int(length)) for task, counts in manifest["counts"].items() for length in counts}
    values = {}
    for ratio in ratios:
        metrics = read_json(run / ratio_directory(ratio) / "metrics.json")
        if metrics["test_identity"] != identity or metrics["ratio"] != ratio:
            raise ValueError(f"Metrics do not match this benchmark at R={ratio:g}")
        cells = {(cell["task"], cell["target_length"]): cell for cell in metrics["cells"]}
        if len(cells) != len(metrics["cells"]) or set(cells) != expected:
            raise ValueError(f"Incomplete or duplicated cells at R={ratio:g}")
        values[ratio] = cells

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(max(9, 2.1 * len(lengths)), max(3.5, 0.5 * len(ratios) + 2)),
                             layout="constrained", squeeze=False)
    fields = ["task", "target_length", "ratio", "count", "empty_count", *METRICS]
    csv_text = StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=fields)
    writer.writeheader()
    for axis, (task, title) in zip(axes[0], (("qa_1", "SQuAD 2.0"), ("qa_2", "HotpotQA"))):
        matrix = [[values[ratio].get((task, length), {}).get(args.metric, float("nan")) * 100
                   for length in lengths] for ratio in ratios]
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        axis.set_title(f"{task} · {title}")
        axis.set_xticks(range(len(lengths)), [f"{n // 1024}K" if n % 1024 == 0 else str(n) for n in lengths])
        axis.set_yticks(range(len(ratios)), [f"{ratio:g}×" for ratio in ratios])
        axis.set_xlabel("Target token budget")
        axis.set_ylabel("Compression ratio")
        for y, ratio in enumerate(ratios):
            for x, length in enumerate(lengths):
                value = matrix[y][x]
                cell = values[ratio].get((task, length))
                label = f"{value:.1f}" if cell else "NA"
                axis.text(x, y, label, ha="center", va="center", color="white" if cell and value < 50 else "black")
                if cell:
                    writer.writerow({"ratio": ratio, **cell})
    metric_label = {"ruler_score": "RULER QA substring score", "em": "Exact match", "f1": "Token F1"}[args.metric]
    subset = " [SUBSET]" if "prepared_test_identity" in manifest else ""
    dataset_label = " [TRAIN]" if split == "train" else ""
    method = metadata.get("experiment", {}).get("config", {}).get("model", {}).get("method", "seco")
    fig.suptitle(f"{method.upper()} · {metric_label}{dataset_label}{subset}")
    fig.colorbar(image, ax=list(axes[0]), label="Score (%)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    write_text(output.with_suffix(".csv"), csv_text.getvalue())
    print(f"RULER QA figure and numeric cells: {output}, {output.with_suffix('.pdf')}, {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
