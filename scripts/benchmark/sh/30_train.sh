#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" # Physical GPUs on this node.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"                     # One process per visible GPU.
NNODES="${NNODES:-1}"                                   # Nodes; all nodes share data/model/run storage.
NODE_RANK="${NODE_RANK:-0}"                             # 0..NNODES-1, unique on each node.
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"                   # Rank-zero hostname/IP; change for multiple nodes.
MASTER_PORT="${MASTER_PORT:-29500}"
BACKEND="${BACKEND:-zero2}"                             # ddp / zero2 / zero2_cpu / zero3_cpu.
CONFIG="${CONFIG:-configs/benchmark/train.json}"                   # LoRA, ratio mode/array, optimizer and schedule.
MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
PREPARED_DIR="${PREPARED_DIR:-artifacts/prepared/default}"
RUN_NAME="${RUN_NAME:-qwen3_4b_seco}"                    # Human-readable tag used for new run names.
RESUME="${RESUME:-}"                                   # Existing artifacts/runs/<run>/checkpoint-N directory.
MAX_STEPS="${MAX_STEPS:-}"                             # Empty uses JSON; positive value is cumulative target.
PLAN_ONLY="${PLAN_ONLY:-0}"                            # 1 validates data/config and prints the plan.

if [[ -n "$RESUME" ]]; then
    RUN_DIR="${RUN_DIR:-$(dirname -- "$RESUME")}"
elif [[ "$NNODES" != 1 && -z "${RUN_DIR:-}" ]]; then
    echo 'Multiple nodes require the same explicit RUN_DIR on every node.' >&2
    exit 2
else
    RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_NAME}_$(timestamp)_${RANDOM}}"
fi
options=(--config "$CONFIG" --model "$MODEL_PATH" --prepared-dir "$PREPARED_DIR"
         --output "$RUN_DIR" --backend "$BACKEND")
[[ -n "$RESUME" ]] && options+=(--resume "$RESUME")
[[ -n "$MAX_STEPS" ]] && options+=(--max-steps "$MAX_STEPS")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE=$((NNODES * NPROC_PER_NODE)) "$PYTHON_BIN" scripts/benchmark/py/30_train.py "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p logs
LOG_FILE="logs/train_$(timestamp)_node${NODE_RANK}_${RANDOM}.log"
"$PYTHON_BIN" -m torch.distributed.run \
    --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    scripts/benchmark/py/30_train.py "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'Run directory: %s\nLog: %s\n' "$RUN_DIR" "$LOG_FILE"
