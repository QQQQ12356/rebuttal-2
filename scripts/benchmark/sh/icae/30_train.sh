#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" # Physical GPUs on this node.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"                     # One process per visible GPU.
export BACKEND="${BACKEND:-zero2}"                             # ddp / zero2 / zero2_cpu.
export CONFIG="${CONFIG:-configs/benchmark/$METHOD/train.json}"                   # LoRA, ratio mode/array, optimizer and schedule.
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
export PREPARED_DIR="${PREPARED_DIR:-$BENCHMARK_ROOT/prepared/default}"
export MAX_STEPS="${MAX_STEPS:-}"                             # Empty uses JSON; positive value is cumulative target.
export PLAN_ONLY="${PLAN_ONLY:-0}"                            # 1 validates data/config and prints the plan.
INIT_FROM="${INIT_FROM:-}"                                # ICAE pretraining final or checkpoint-N.
options=()
[[ -n "$INIT_FROM" ]] && options+=(--init-from "$INIT_FROM")
exec bash "$ENTRY_DIR/30_train.sh" "${options[@]}" "$@"
