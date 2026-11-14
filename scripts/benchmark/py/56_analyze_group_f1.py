#!/usr/bin/env python3
"""Analyze SECO group-size imbalance and token F1 by dataset and ratio."""
import argparse
from collections import defaultdict
import csv
import io
import math
import random
from statistics import correlation, fmean, pstdev, quantiles

from _bootstrap import require_linux
from seco.io import iter_jsonl, project_path, read_json, write_text


DATASETS = ("naturalqa", "hotpotqa", "2wikimqa", "narrativeqa", "multinews")
SAMPLE_FIELDS = ("dataset", "ratio", "id", "context_tokens", "group_count", "f1",
                 "group_size_cv", "max_group_over_mean")


def ranks(values):
    """Assign the mean rank to tied values, including repeated F1 scores."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(order)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = mean_rank
        start = end
    return result


def pearson(x, y):
    if len(x) < 3 or max(x) == min(x) or max(y) == min(y):
        return None
    return correlation(x, y)


def rank_correlations(x, f1, lengths):
    x_rank, f1_rank, length_rank = map(ranks, (x, f1, lengths))
    raw = pearson(x_rank, f1_rank)
    if raw is None or max(length_rank) == min(length_rank):
        return raw, raw
    if len(x) < 4:
        return raw, None
    x_length = pearson(x_rank, length_rank)
    f1_length = pearson(f1_rank, length_rank)
    denominator = (1 - x_length**2) * (1 - f1_length**2)
    if denominator <= 1e-12:
        return raw, None
    adjusted = (raw - x_length * f1_length) / math.sqrt(denominator)
    return raw, max(-1.0, min(1.0, adjusted))


def read_samples(run):
    metadata = read_json(run / "benchmark.json")
    if "generation_by_dataset" not in metadata:
        raise ValueError("Expected a general Benchmark run from scripts/benchmark/sh/seco/50_benchmark.sh")
    for ratio in sorted(metadata["arguments"]["ratios"]):
        path = run / f"ratio_{ratio:g}" / "predictions.jsonl"
        seen = set()
        for number, row in enumerate(iter_jsonl(path), 1):
            dataset, sample_id, f1 = row["dataset"], row["id"], row["f1"]
            if row["ratio"] != ratio or sample_id in seen:
                raise ValueError(f"{path}, record {number}: unexpected ratio or duplicate sample ID")
            seen.add(sample_id)
            if dataset not in DATASETS:
                continue
            sizes, count = row.get("group_token_counts"), row.get("group_count")
            if sizes is None or count is None:
                raise ValueError(f"{path}, record {number}: rerun SECO with RECORD_GROUP_STATS=1")
            if (not isinstance(sizes, list) or type(count) is not int or len(sizes) != count
                    or any(type(size) is not int or size <= 0 for size in sizes)):
                raise ValueError(f"{path}, record {number}: invalid group sizes")
            if type(f1) not in (int, float) or not 0 <= f1 <= 1:
                raise ValueError(f"{path}, record {number}: expected F1 on the 0–1 scale")
            length = sum(sizes)
            mean_size = length / count if count >= 2 else None
            yield dict(zip(SAMPLE_FIELDS, (
                dataset, ratio, sample_id, length, count, f1,
                pstdev(sizes) / mean_size if mean_size else None,
                max(sizes) / mean_size if mean_size else None,
            )))
        if not seen or len(seen) != metadata["samples"]:
            raise ValueError(f"{path}: expected {metadata['samples']} predictions, found {len(seen)}")


def summarize(samples, resamples, seed):
    strata = defaultdict(list)
    for row in samples:
        strata[row["dataset"], row["ratio"]].append(row)

    rng = random.Random(seed)
    for (dataset, ratio), rows in sorted(strata.items()):
        for metric in ("group_size_cv", "max_group_over_mean"):
            observations = [(row[metric], row["f1"], row["context_tokens"])
                            for row in rows if row[metric] is not None]
            n = len(observations)
            x, f1, lengths = zip(*observations) if n else ((), (), ())
            raw, adjusted = rank_correlations(x, f1, lengths)
            draws = []
            if adjusted is not None and n >= 4 and resamples:
                for _ in range(resamples):
                    draw = rng.choices(observations, k=n)
                    _, estimate = rank_correlations(*zip(*draw))
                    if estimate is not None:
                        draws.append(estimate)
            ci_low = ci_high = None
            if resamples and len(draws) >= max(2, math.ceil(0.95 * resamples)):
                cuts = quantiles(draws, n=40, method="inclusive")
                ci_low, ci_high = cuts[0], cuts[-1]
            yield {
                "dataset": dataset, "ratio": ratio, "metric": metric,
                "n_total": len(rows), "n_analyzed": n, "n_excluded": len(rows) - n,
                "mean_f1": fmean(f1) if n else None,
                "spearman_rho": raw, "length_adjusted_rho": adjusted,
                "adjusted_ci95_low": ci_low, "adjusted_ci95_high": ci_high,
                "bootstrap_valid": len(draws), "bootstrap_requested": resamples,
            }


def write_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Completed general Benchmark run directory")
    parser.add_argument("--output-dir", help="Output directory (default: group_f1_analysis inside the run)")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000, help="95%% CI resamples per stratum; 0 disables")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed (default: 42)")
    args = parser.parse_args()
    if args.bootstrap_resamples < 0:
        parser.error("--bootstrap-resamples must be non-negative")
    run = project_path(args.benchmark_dir)
    output = project_path(args.output_dir) if args.output_dir else run / "group_f1_analysis"
    try:
        samples = list(read_samples(run))
        if not samples:
            raise ValueError("No predictions for the five supported datasets")
        summary = list(summarize(samples, args.bootstrap_resamples, args.seed))
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / "samples.csv", samples)
        write_csv(output / "correlations.csv", summary)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    missing = set(DATASETS) - {row["dataset"] for row in samples}
    if missing:
        print(f"Datasets absent from this run: {', '.join(sorted(missing))}")
    print(f"Analyzed {len(samples)} sample-ratio records. Results: {output}")


if __name__ == "__main__":
    main()
