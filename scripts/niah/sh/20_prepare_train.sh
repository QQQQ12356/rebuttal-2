#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREPARE_CONFIG="${PREPARE_CONFIG:-configs/niah/prepare_train.json}" # Counts, candidate limits, length filters, sources.
MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
RAW_DIR="${RAW_DIR:-$NIAH_ROOT/raw}"
PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}"        # Must be a new directory.
SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-}"                       # Empty uses JSON; 0 disables generated examples.
SYNTHETIC_TASKS="${SYNTHETIC_TASKS:-}"                           # Space-separated tasks, e.g. "custom" or "uuid passkey custom"; empty uses JSON.
SUPERVISED_SAMPLES="${SUPERVISED_SAMPLES:-}"                     # Empty uses per-source JSON; 0 = all eligible/source.
extra=()
[[ -n "$SYNTHETIC_SAMPLES" ]] && extra+=(--synthetic-samples "$SYNTHETIC_SAMPLES")
if [[ -n "$SYNTHETIC_TASKS" ]]; then
    read -r -a tasks <<< "$SYNTHETIC_TASKS"
    extra+=(--synthetic-tasks "${tasks[@]}")
fi
[[ -n "$SUPERVISED_SAMPLES" ]] && extra+=(--supervised-samples "$SUPERVISED_SAMPLES")
"$PYTHON_BIN" scripts/niah/py/20_prepare_train.py --config "$PREPARE_CONFIG" \
    --model "$MODEL_PATH" --raw-dir "$RAW_DIR" --output "$PREPARED_DIR" "${extra[@]}" "$@"
