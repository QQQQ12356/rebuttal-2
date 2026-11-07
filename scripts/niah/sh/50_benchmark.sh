#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # Full model pair on each GPU; no tensor parallel.
CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$NIAH_ROOT/benchmarks/niah_$(timestamp)_${RANDOM}}"
RATIOS=(4 8 16 32)                                       # Same test grid at every fixed ratio.
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"                     # Must fit grid generation_budget.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
options=(--checkpoint "$CHECKPOINT" --grid-dir "$GRID_DIR" --output "$OUTPUT_DIR"
         --ratios "${RATIOS[@]}" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
         --warmup-batches "$WARMUP_BATCHES" --max-new-tokens "$MAX_NEW_TOKENS"
         --repetition-penalty "$REPETITION_PENALTY")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE="$NPROC_PER_NODE" "$PYTHON_BIN" scripts/niah/py/50_benchmark.py "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p "$NIAH_ROOT/logs"
LOG_FILE="$NIAH_ROOT/logs/benchmark_$(timestamp)_${RANDOM}.log"
printf 'Log: %s\n' "$LOG_FILE"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
    scripts/niah/py/50_benchmark.py "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'NIAH benchmark: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG_FILE"
