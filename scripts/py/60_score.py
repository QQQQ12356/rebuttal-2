"""Recompute EM/F1 from a saved predictions JSONL, including external benchmarks."""
import argparse

from _bootstrap import require_linux
from seco.io import iter_jsonl, project_path, read_json, write_json, write_jsonl
from seco.metrics import score_answer, summarize


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", help="Default: metrics.json beside predictions")
    parser.add_argument("--scored-output", help="Optional new JSONL containing recomputed per-sample scores")
    args = parser.parse_args()
    source = project_path(args.predictions)
    rows = list(iter_jsonl(source))
    # Accept the original SeCo benchmark vocabulary as well as the new pipeline.
    for index, row in enumerate(rows):
        row.setdefault("id", f"external:{index}")
        row.setdefault("dataset", row.get("subset", "unknown"))
        if "prediction" not in row:
            row["prediction"] = row["model_output"]
        if "answer" not in row:
            row["answer"] = row["ground_truth"]
        if isinstance(row["answer"], str):
            row["answer"] = [row["answer"]]
        row["em"], row["f1"] = score_answer(row["prediction"], row["answer"])
    metrics = summarize(rows)
    ratios = {row["ratio"] for row in rows if "ratio" in row}
    if len(ratios) == 1:
        metrics["ratio"] = ratios.pop()
    destination = project_path(args.output) if args.output else source.parent / "metrics.json"
    # Preserve timing/model annotations from the generated benchmark when rescoring.
    if destination.exists():
        previous = read_json(destination)
        metrics = {**previous, **metrics}
    write_json(destination, metrics)
    if args.scored_output:
        scored = project_path(args.scored_output)
        if scored == source or scored.exists():
            raise FileExistsError("--scored-output must name a new file")
        write_jsonl(scored, rows)
    print(f"n={metrics['count']} EM={metrics['micro']['em']:.6f} F1={metrics['micro']['f1']:.6f}\n{destination}")


if __name__ == "__main__":
    main()
