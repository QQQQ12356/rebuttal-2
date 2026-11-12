#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export BACKEND="${BACKEND:-zero2}"                                # ddp / zero2 / zero2_cpu.
export CONFIG="${CONFIG:-configs/niah/$METHOD/train.json}"                  # Independent NIAH optimizer and fixed/random ratios.
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
export PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}"
export MAX_STEPS="${MAX_STEPS:-}"                                # Empty = config; positive = cumulative optimizer steps.
export PLAN_ONLY="${PLAN_ONLY:-0}"                               # 1 validates data/config without model allocation.
# Choose one source for a new SFT run. To resume, leave both at their defaults.
INIT_FROM="${INIT_FROM:-}"             # Pretraining final/ or checkpoint-N/.
INIT_FROM_BASE="${INIT_FROM_BASE:-0}"  # 1 starts from MODEL_PATH with fresh ICAE parameters.

options=()
[[ -n "$INIT_FROM" ]] && options+=(--init-from "$INIT_FROM")
case "$INIT_FROM_BASE" in
    0) ;;
    1) options+=(--init-from-base) ;;
    *) echo 'INIT_FROM_BASE must be 0 or 1.' >&2; exit 2 ;;
esac
exec bash "$ENTRY_DIR/30_train.sh" "${options[@]}" "$@"
