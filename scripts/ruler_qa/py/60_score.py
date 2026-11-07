"""Recompute RULER QA scores from a complete saved prediction file."""
import argparse

from ruler_bootstrap import require_linux
from seco.io import iter_jsonl, project_path, read_json, write_json
from seco.ruler_qa.metrics import aggregate


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test-manifest", help="Default: use the embedded manifest in benchmark.json")
    parser.add_argument("--output", help="Default: sibling metrics.json")
    args = parser.parse_args()
    source = project_path(args.predictions)
    manifest = (read_json(project_path(args.test_manifest)) if args.test_manifest else
                read_json(source.parent.parent / "benchmark.json")["test_manifest"])
    rows = list(iter_jsonl(source))
    ratios = {row["ratio"] for row in rows}
    if len(ratios) != 1:
        raise ValueError("Score one compression ratio per prediction file")
    metrics = aggregate(rows, manifest, ratios.pop())
    output = project_path(args.output) if args.output else source.parent / "metrics.json"
    if output.is_file():
        previous = read_json(output)
        if previous.get("ratio") == metrics["ratio"] and previous.get("test_identity") == metrics["test_identity"]:
            metrics = {**previous, **metrics}
    write_json(output, metrics)
    print(f"RULER QA metrics: {output} {metrics['summary']}")


if __name__ == "__main__":
    main()
