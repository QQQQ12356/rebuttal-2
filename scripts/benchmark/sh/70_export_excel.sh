#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

INPUTS=("$BENCHMARK_ROOT/benchmarks/**/metrics.json")             # Quote globs; Python expands recursively.
EXCEL_PATH="${EXCEL_PATH:-$BENCHMARK_ROOT/reports/benchmark_$(timestamp)_${RANDOM}.xlsx}"
INCLUDE_DETAILS="${INCLUDE_DETAILS:-1}"                    # 1 imports prediction/answer/prompt/context contents.
extra=()
[[ "$INCLUDE_DETAILS" == 1 ]] && extra+=(--include-details)
"$PYTHON_BIN" "$PY_ENTRY_DIR/70_export_excel.py" \
    --inputs "${INPUTS[@]}" --output "$EXCEL_PATH" "${extra[@]}" "$@"
