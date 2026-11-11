#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=ram
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export GRID_CONFIG="${GRID_CONFIG:-configs/niah/grid.json}"          # Lengths/depths/trials/task/budgets; optional grid_128k.json.
export GRID_REPEATS="${GRID_REPEATS:-}"                             # Positive trial count; empty uses JSON.
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
export GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"           # Test-only fixed grid, separate from train data.
export TRAINING_MANIFEST="${TRAINING_MANIFEST:-}"                  # Optional explicit train manifest to check overlap.
exec bash "$ENTRY_DIR/21_prepare_grid.sh" "$@"
