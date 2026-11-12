#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

BENCHMARK_DIR="${BENCHMARK_DIR:?Set BENCHMARK_DIR to the directory containing ratio_*/metrics.json}"
options=(--benchmark-dir "$BENCHMARK_DIR" --method "$METHOD" --task niah)
if [[ -n "${EXCEL_PATH:-}" ]]; then
    options+=(--output "$EXCEL_PATH")
fi
"$PYTHON_BIN" -m seco.export_metrics "${options[@]}" "$@"
