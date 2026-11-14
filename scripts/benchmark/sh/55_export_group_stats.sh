#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

BENCHMARK_DIR="${BENCHMARK_DIR:-}"                     # Required: completed SECO run directory from step 50.
GROUP_STATS_OUTPUT="${GROUP_STATS_OUTPUT:-}"           # New JSONL path; empty writes group_stats.jsonl inside the run.

if [[ -z "$BENCHMARK_DIR" ]]; then
    echo 'Set BENCHMARK_DIR to the completed benchmark run directory.' >&2
    exit 2
fi
options=(--benchmark-dir "$BENCHMARK_DIR")
[[ -n "$GROUP_STATS_OUTPUT" ]] && options+=(--output "$GROUP_STATS_OUTPUT")
exec "$PYTHON_BIN" scripts/benchmark/py/55_export_group_stats.py "${options[@]}" "$@"
