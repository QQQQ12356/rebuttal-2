#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
RAW_DIR="${RAW_DIR:-artifacts/raw}"
PREPARED_DIR="${PREPARED_DIR:-artifacts/prepared/default}"  # Must be a new directory each time.
DATASETS=(naturalqa 2wikimqa hotpotqa narrativeqa multinews) # Training datasets, in this order.
TRAIN_SAMPLES=(5000 5000 5000 5000 5000)                   # Eligible samples per dataset; 0 = all.
TEST_SAMPLES="${TEST_SAMPLES:-500}"                      # Eligible examples per Twwilght benchmark; 0 = all.
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"                   # First N downloaded training rows to scan; 0 = all.
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"             # Context + prompt. Overlength examples are filtered.
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-1024}"            # Training answer + EOS only; never truncates targets.
MAX_DECODER_TOKENS="${MAX_DECODER_TOKENS:-8192}"          # Centers + prompt + answer/generation reserve.
MIN_CONTEXT_TOKENS="${MIN_CONTEXT_TOKENS:-32}"
MIN_RATIO="${MIN_RATIO:-4}"                              # Smallest ratio planned for training AND testing.
TEST_GENERATION_BUDGET="${TEST_GENERATION_BUDGET:-512}"   # Must cover the largest inference max_new_tokens.
SEED="${SEED:-42}"
STRICT_COUNT="${STRICT_COUNT:-0}"                        # 1 fails when eligible examples are insufficient.
extra=()
[[ "$STRICT_COUNT" == 1 ]] && extra+=(--strict-count)
"$PYTHON_BIN" scripts/py/20_prepare_data.py \
    --model "$MODEL_PATH" --raw-dir "$RAW_DIR" --output "$PREPARED_DIR" \
    --datasets "${DATASETS[@]}" --train-samples "${TRAIN_SAMPLES[@]}" --test-samples "$TEST_SAMPLES" \
    --candidate-limit "$CANDIDATE_LIMIT" --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-answer-tokens "$MAX_ANSWER_TOKENS" --max-decoder-tokens "$MAX_DECODER_TOKENS" \
    --min-context-tokens "$MIN_CONTEXT_TOKENS" --min-ratio "$MIN_RATIO" \
    --test-generation-budget "$TEST_GENERATION_BUDGET" --seed "$SEED" "${extra[@]}" "$@"
