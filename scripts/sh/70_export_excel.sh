#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

INPUTS=("artifacts/benchmarks/**/metrics.json")             # Quote globs; Python expands recursively.
EXCEL_PATH="${EXCEL_PATH:-artifacts/reports/benchmark_$(timestamp)_${RANDOM}.xlsx}"
INCLUDE_DETAILS="${INCLUDE_DETAILS:-1}"                    # 1 imports prediction/answer/prompt/context contents.
extra=()
[[ "$INCLUDE_DETAILS" == 1 ]] && extra+=(--include-details)
"$PYTHON_BIN" scripts/py/70_export_excel.py \
    --inputs "${INPUTS[@]}" --output "$EXCEL_PATH" "${extra[@]}" "$@"
