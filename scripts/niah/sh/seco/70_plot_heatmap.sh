#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export BENCHMARK_DIR="${BENCHMARK_DIR:-$NIAH_ROOT/benchmarks/REPLACE_WITH_BENCHMARK}"
export METRIC="${METRIC:-accuracy}"                              # accuracy/em / f1 / hit_rate; fixed 0–100 colorbar.
export COLUMNS="${COLUMNS:-2}"                                   # One panel per ratio; 1 for a single-column figure.
exec bash "$ENTRY_DIR/70_plot_heatmap.sh" "$@"
