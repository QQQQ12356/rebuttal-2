#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29600}"
BACKEND="${BACKEND:-zero2}"                                # ddp / zero2 / zero2_cpu / zero3_cpu.
CONFIG="${CONFIG:-configs/niah/train.json}"                  # Independent NIAH optimizer and fixed/random ratios.
MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}"
RUN_NAME="${RUN_NAME:-niah_qwen3_4b}"
RESUME="${RESUME:-}"                                      # Native checkpoint-N; final is inference-only.
MAX_STEPS="${MAX_STEPS:-}"                                # Empty = config; positive = cumulative optimizer steps.
PLAN_ONLY="${PLAN_ONLY:-0}"                               # 1 validates data/config without model allocation.
if [[ -n "$RESUME" ]]; then
    RUN_DIR="${RUN_DIR:-$(dirname -- "$RESUME")}"
elif [[ "$NNODES" != 1 && -z "${RUN_DIR:-}" ]]; then
    echo 'All nodes must specify the same shared RUN_DIR.' >&2
    exit 2
else
    RUN_DIR="${RUN_DIR:-$NIAH_ROOT/runs/${RUN_NAME}_$(timestamp)_${RANDOM}}"
fi
options=(--config "$CONFIG" --model "$MODEL_PATH" --prepared-dir "$PREPARED_DIR" --output "$RUN_DIR" --backend "$BACKEND")
[[ -n "$RESUME" ]] && options+=(--resume "$RESUME")
[[ -n "$MAX_STEPS" ]] && options+=(--max-steps "$MAX_STEPS")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE=$((NNODES * NPROC_PER_NODE)) "$PYTHON_BIN" scripts/niah/py/30_train.py "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p "$NIAH_ROOT/logs"
LOG_FILE="$NIAH_ROOT/logs/train_$(timestamp)_node${NODE_RANK}_${RANDOM}.log"
"$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --nproc_per_node "$NPROC_PER_NODE" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    scripts/niah/py/30_train.py "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'NIAH run: %s\nLog: %s\n' "$RUN_DIR" "$LOG_FILE"
