#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # Full model pair on each GPU; no tensor parallel.
CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"
PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}" # Only used for training-set evaluation.
EVAL_TRAIN="${EVAL_TRAIN:-0}"                              # 0: held-out grid. 1: checkpoint's training data.
read -r -a ratios <<< "${RATIOS:-4 8 16 32}"               # Same examples at every fixed ratio.
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"                     # Must fit the selected data's decoder budget.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$EVAL_TRAIN" in
    0)
        entry=scripts/niah/py/50_benchmark.py
        dataset_options=(--grid-dir "$GRID_DIR")
        run_prefix=niah
        log_prefix=benchmark
        ;;
    1)
        entry=scripts/niah/py/51_benchmark_train.py
        dataset_options=(--eval-train --prepared-dir "$PREPARED_DIR")
        run_prefix=niah_train
        log_prefix=train_benchmark
        ;;
    *)
        echo 'EVAL_TRAIN must be 0 or 1.' >&2
        exit 2
        ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$NIAH_ROOT/benchmarks/${run_prefix}_$(timestamp)_${RANDOM}}"
options=(--checkpoint "$CHECKPOINT" "${dataset_options[@]}" --output "$OUTPUT_DIR"
         --ratios "${ratios[@]}" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
         --warmup-batches "$WARMUP_BATCHES" --max-new-tokens "$MAX_NEW_TOKENS"
         --repetition-penalty "$REPETITION_PENALTY")
[[ -n "$MAX_TEST_SAMPLES" ]] && options+=(--max-test-samples "$MAX_TEST_SAMPLES")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE="$NPROC_PER_NODE" "$PYTHON_BIN" "$entry" "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p "$NIAH_ROOT/logs"
LOG_FILE="$NIAH_ROOT/logs/${log_prefix}_$(timestamp)_${RANDOM}.log"
printf 'Log: %s\n' "$LOG_FILE"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
    "$entry" "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'NIAH benchmark: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG_FILE"
