#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"    # Each process holds one complete encoder/decoder pair.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # More GPUs increase throughput, not per-model capacity.
CHECKPOINT="${CHECKPOINT:-artifacts/runs/REPLACE_WITH_RUN/final_merged}"
PREPARED_DIR="${PREPARED_DIR:-artifacts/prepared/default}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/benchmarks/benchmark_$(timestamp)_${RANDOM}}"
DATASETS=(naturalqa 2wikimqa hotpotqa multinews)
RATIOS=(4 8 16 32)                                       # Fixed ratios, each applied to exactly the same examples.
BATCH_SIZE="${BATCH_SIZE:-1}"                             # Increase only when each GPU has enough memory.
SAMPLES="${SAMPLES:-0}"                                  # Per dataset; 0 = all preprocessed test examples.
WORKERS="${WORKERS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"                      # Repeat first batch; all examples still scored once.
QA_MAX_NEW_TOKENS="${QA_MAX_NEW_TOKENS:-64}"
NEWS_MAX_NEW_TOKENS="${NEWS_MAX_NEW_TOKENS:-512}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"             # Same decoding configuration for comparisons.
SEED="${SEED:-42}"
INCLUDE_CONTEXT="${INCLUDE_CONTEXT:-0}"                    # 1 stores full input for later Excel details.
extra=()
[[ "$INCLUDE_CONTEXT" == 1 ]] && extra+=(--include-context)
mkdir -p logs
LOG_FILE="logs/benchmark_$(timestamp)_${RANDOM}.log"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
    scripts/py/50_benchmark.py --checkpoint "$CHECKPOINT" --prepared-dir "$PREPARED_DIR" \
    --output "$OUTPUT_DIR" --datasets "${DATASETS[@]}" --ratios "${RATIOS[@]}" \
    --batch-size "$BATCH_SIZE" --samples "$SAMPLES" --workers "$WORKERS" \
    --warmup-batches "$WARMUP_BATCHES" --qa-max-new-tokens "$QA_MAX_NEW_TOKENS" \
    --news-max-new-tokens "$NEWS_MAX_NEW_TOKENS" --repetition-penalty "$REPETITION_PENALTY" \
    --seed "$SEED" "${extra[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'Benchmark directory: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG_FILE"
