#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

GRID_CONFIG="${GRID_CONFIG:-configs/niah/grid.json}"          # Lengths/depths/trials/task/budgets; optional grid_128k.json.
MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
RAW_DIR="${RAW_DIR:-$NIAH_ROOT/raw}"
GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"           # Test-only fixed grid, separate from train data.
TRAINING_MANIFEST="${TRAINING_MANIFEST:-}"                  # Optional explicit train manifest to check overlap.
if [[ -z "$TRAINING_MANIFEST" && -f "$NIAH_ROOT/prepared/train_v1/manifest.json" ]]; then
    TRAINING_MANIFEST="$NIAH_ROOT/prepared/train_v1/manifest.json"
fi
extra=()
[[ -n "$TRAINING_MANIFEST" ]] && extra+=(--training-manifest "$TRAINING_MANIFEST")
"$PYTHON_BIN" scripts/niah/py/21_prepare_grid.py --config "$GRID_CONFIG" \
    --model "$MODEL_PATH" --raw-dir "$RAW_DIR" --output "$GRID_DIR" "${extra[@]}" "$@"
