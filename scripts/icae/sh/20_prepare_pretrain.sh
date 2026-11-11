#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

CORPUS_FILE="${CORPUS_FILE:-}"                 # Single file; use --input for multiple shards.
TEXT_FIELD="${TEXT_FIELD:-text}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
REVISION="${REVISION:-main}"
PREPARED_DIR="${PREPARED_DIR:-$ICAE_PRETRAIN_ROOT/prepared/v1}"
MIN_TOKENS="${MIN_TOKENS:-512}"                # Document length before splitting, excluding BOS/EOS.
MAX_TOKENS="${MAX_TOKENS:-4096}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-10000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"            # First eligible examples; 0 skips validation.

input=()
[[ -n "$CORPUS_FILE" ]] && input+=(--input "$CORPUS_FILE")
"$PYTHON_BIN" scripts/icae/py/20_prepare_pretrain.py "${input[@]}" \
    --text-field "$TEXT_FIELD" --model "$MODEL_PATH" --revision "$REVISION" --output "$PREPARED_DIR" \
    --min-tokens "$MIN_TOKENS" --max-tokens "$MAX_TOKENS" --train-samples "$TRAIN_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES" "$@"
