#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

BENCHMARK_DIR="${BENCHMARK_DIR:?Set BENCHMARK_DIR to the completed general SECO Benchmark run directory}"
GROUP_F1_OUTPUT_DIR="${GROUP_F1_OUTPUT_DIR:-}"           # New output directory; empty uses group_f1_analysis inside the run.
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"      # Resamples per dataset and ratio; 0 disables confidence intervals.
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"                  # Seed for reproducible confidence intervals.

options=(--benchmark-dir "$BENCHMARK_DIR" --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" --seed "$BOOTSTRAP_SEED")
[[ -n "$GROUP_F1_OUTPUT_DIR" ]] && options+=(--output-dir "$GROUP_F1_OUTPUT_DIR")
exec "$PYTHON_BIN" scripts/benchmark/py/56_analyze_group_f1.py "${options[@]}" "$@"
