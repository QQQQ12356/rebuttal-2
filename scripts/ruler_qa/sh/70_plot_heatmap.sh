#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

BENCHMARK_DIR="${BENCHMARK_DIR:?Set BENCHMARK_DIR to a completed benchmark}"
options=(--benchmark-dir "$BENCHMARK_DIR" --metric "${METRIC:-ruler_score}" --dpi "${DPI:-300}")
[[ -n "${FIGURE_PATH:-}" ]] && options+=(--output "$FIGURE_PATH")
"$PYTHON_BIN" "$PY_ENTRY_DIR/70_plot_heatmap.py" "${options[@]}" "$@"
