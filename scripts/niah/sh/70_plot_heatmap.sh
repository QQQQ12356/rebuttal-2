#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

BENCHMARK_DIR="${BENCHMARK_DIR:-$NIAH_ROOT/benchmarks/REPLACE_WITH_BENCHMARK}"
INPUTS=("$BENCHMARK_DIR/ratio_*/metrics.json")                # Quoted patterns; Python expands them.
FIGURE_PATH="${FIGURE_PATH:-$BENCHMARK_DIR/figures/NIAH.png}" # New PNG; PDF and cell CSV also generated.
METRIC="${METRIC:-accuracy}"                              # accuracy/em / f1 / hit_rate; fixed 0–100 colorbar.
COLUMNS="${COLUMNS:-2}"                                   # One panel per ratio; 1 for a single-column figure.
DPI="${DPI:-300}"
TITLE="${TITLE:-$METHOD Needle-in-a-Haystack}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/70_plot_heatmap.py" --inputs "${INPUTS[@]}" \
    --output "$FIGURE_PATH" --metric "$METRIC" --columns "$COLUMNS" --dpi "$DPI" --title "$TITLE" "$@"
