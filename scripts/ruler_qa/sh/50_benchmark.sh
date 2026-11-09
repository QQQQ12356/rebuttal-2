#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an adapter or merged export}"
TEST_DIR="${TEST_DIR:-$RULER_QA_ROOT/prepared/test_v1}"
PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/train_v1}"
EVAL_TRAIN="${EVAL_TRAIN:-0}"                             # 0: held-out test grid. 1: checkpoint training split.
OUTPUT_DIR="${OUTPUT_DIR:-$RULER_QA_ROOT/benchmarks/ruler_qa_$(timestamp)_${RANDOM}}"
read -r -a ratios <<< "${RATIOS:-4 8 16 32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"                    # Must fit the prepared generation budget.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
MODEL_PATH="${MODEL_PATH:-}"
ATTENTION="${ATTENTION:-}"
NO_MERGE="${NO_MERGE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$EVAL_TRAIN" in
    0) dataset_options=(--test-dir "$TEST_DIR" --prepared-dir "$PREPARED_DIR") ;;
    1) dataset_options=(--eval-train --prepared-dir "$PREPARED_DIR") ;;
    *) echo 'EVAL_TRAIN must be 0 or 1.' >&2; exit 2 ;;
esac

options=(--checkpoint "$CHECKPOINT" "${dataset_options[@]}" --output "$OUTPUT_DIR"
         --ratios "${ratios[@]}" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
         --warmup-batches "$WARMUP_BATCHES" --max-new-tokens "$MAX_NEW_TOKENS"
         --repetition-penalty "$REPETITION_PENALTY")
[[ -n "$MAX_TEST_SAMPLES" ]] && options+=(--max-test-samples "$MAX_TEST_SAMPLES")
[[ -n "$MODEL_PATH" ]] && options+=(--model "$MODEL_PATH")
[[ -n "$ATTENTION" ]] && options+=(--attention "$ATTENTION")
[[ "$NO_MERGE" == 1 ]] && options+=(--no-merge)
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE="$NPROC_PER_NODE" "$PYTHON_BIN" scripts/ruler_qa/py/50_benchmark.py \
        "${options[@]}" --plan-only "$@"
    exit 0
fi

mkdir -p "$RULER_QA_ROOT/logs"
LOG_FILE="$RULER_QA_ROOT/logs/benchmark_$(timestamp)_${RANDOM}.log"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
    scripts/ruler_qa/py/50_benchmark.py "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'RULER QA benchmark: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG_FILE"
