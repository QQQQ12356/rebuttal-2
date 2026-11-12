#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

SPLIT="${SPLIT:-train}"
PREPARE_CONFIG="${PREPARE_CONFIG:-configs/ruler_qa/prepare.json}"
RAW_DIR="${RAW_DIR:-$RULER_QA_ROOT/raw}"
PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/${SPLIT}_v1}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
REVISION="${REVISION:-main}"
PROMPT_STYLE="${PROMPT_STYLE:-raw}"  # raw: question only; legacy: NIAH Question: suffix.

# Leave an override empty to use PREPARE_CONFIG. Comments show the shipped defaults.
TRAIN_LENGTHS="${TRAIN_LENGTHS:-}"                         # 4096 8192. Space-separated token budgets.
TEST_LENGTHS="${TEST_LENGTHS:-}"                           # 4096 8192 16384 32768. Space-separated token budgets.
TRAIN_CANDIDATES_PER_TASK="${TRAIN_CANDIDATES_PER_TASK:-}"   # 4096. Training candidate pool per task.
TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-}"         # 512. Accepted training questions per task.
TEST_SAMPLES_PER_TASK="${TEST_SAMPLES_PER_TASK:-}"           # 50. Accepted test questions per task.
MIN_FILL="${MIN_FILL:-}"                                   # 0.85. Minimum fraction of the target token budget.
MIN_RATIO="${MIN_RATIO:-}"                                 # 4. Lowest ratio used to check decoder capacity.
MAX_DECODER_TOKENS="${MAX_DECODER_TOKENS:-}"                 # 16384. Maximum decoder sequence length.
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-}"                   # 64. Training answer limit including EOS.
GENERATION_BUDGET="${GENERATION_BUDGET:-}"                   # 32. Test tokens reserved for generation.
SEED="${SEED:-}"                                           # 42. Question and document selection seed.

overrides=()
if [[ -n "$TRAIN_LENGTHS" ]]; then
    read -r -a train_lengths <<< "$TRAIN_LENGTHS"
    overrides+=(--train-lengths "${train_lengths[@]}")
fi
if [[ -n "$TEST_LENGTHS" ]]; then
    read -r -a test_lengths <<< "$TEST_LENGTHS"
    overrides+=(--test-lengths "${test_lengths[@]}")
fi
[[ -n "$TRAIN_CANDIDATES_PER_TASK" ]] && overrides+=(--train-candidates-per-task "$TRAIN_CANDIDATES_PER_TASK")
[[ -n "$TRAIN_SAMPLES_PER_TASK" ]] && overrides+=(--train-samples-per-task "$TRAIN_SAMPLES_PER_TASK")
[[ -n "$TEST_SAMPLES_PER_TASK" ]] && overrides+=(--test-samples-per-task "$TEST_SAMPLES_PER_TASK")
[[ -n "$MIN_FILL" ]] && overrides+=(--min-fill "$MIN_FILL")
[[ -n "$MIN_RATIO" ]] && overrides+=(--min-ratio "$MIN_RATIO")
[[ -n "$MAX_DECODER_TOKENS" ]] && overrides+=(--max-decoder-tokens "$MAX_DECODER_TOKENS")
[[ -n "$MAX_ANSWER_TOKENS" ]] && overrides+=(--max-answer-tokens "$MAX_ANSWER_TOKENS")
[[ -n "$GENERATION_BUDGET" ]] && overrides+=(--generation-budget "$GENERATION_BUDGET")
[[ -n "$SEED" ]] && overrides+=(--seed "$SEED")

"$PYTHON_BIN" "$PY_ENTRY_DIR/20_prepare_data.py" --split "$SPLIT" --config "$PREPARE_CONFIG" \
    --raw-dir "$RAW_DIR" --output "$PREPARED_DIR" --model "$MODEL_PATH" --revision "$REVISION" \
    --prompt-style "$PROMPT_STYLE" "${overrides[@]}" "$@"
