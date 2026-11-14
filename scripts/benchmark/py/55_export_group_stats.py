#!/usr/bin/env python3
"""Extract one compact JSONL row per sample and compression ratio."""
import argparse

from _bootstrap import require_linux
from seco.io import iter_jsonl, project_path, read_json, write_jsonl


def group_rows(predictions):
    for path in predictions:
        found = False
        for number, row in enumerate(iter_jsonl(path), 1):
            found = True
            count = row.get("group_count")
            sizes = row.get("group_token_counts")
            if count is None or sizes is None:
                raise ValueError(f"{path} (record {number}): rerun SECO with RECORD_GROUP_STATS=1")
            if (type(count) is not int or not isinstance(sizes, list) or count != len(sizes)
                    or any(type(size) is not int or size < 1 for size in sizes)):
                raise ValueError(f"{path} (record {number}): invalid group counts")
            dataset_key = "dataset" if "dataset" in row else "task"
            yield {key: row[key] for key in ("id", dataset_key, "ratio", "group_count", "group_token_counts")}
        if not found:
            raise ValueError(f"Empty predictions file: {path}")


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Completed benchmark run directory")
    parser.add_argument("--output", help="New JSONL file (default: group_stats.jsonl inside the run)")
    args = parser.parse_args()
    run = project_path(args.benchmark_dir)
    output = project_path(args.output) if args.output else run / "group_stats.jsonl"
    try:
        metadata = read_json(run / "benchmark.json")
        predictions = sorted(run.glob("ratio_*/predictions.jsonl"))
        ratios = metadata["arguments"]["ratios"]
        if not predictions or len(predictions) != len(ratios):
            raise ValueError(f"Expected {len(ratios)} merged predictions files in {run}, found {len(predictions)}")
        if output.exists():
            raise ValueError(f"Output already exists: {output}. Choose a different --output path.")
        write_jsonl(output, group_rows(predictions))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Group statistics: {output}")


if __name__ == "__main__":
    main()
