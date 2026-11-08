#!/usr/bin/env bash
set -euo pipefail

EVAL_TRAIN="${EVAL_TRAIN:-0}"  # Explicit opt-in to evaluation on the checkpoint's training split.
if [[ "$EVAL_TRAIN" != 1 ]]; then
    echo 'Set EVAL_TRAIN=1 to evaluate on the NIAH training split.' >&2
    exit 2
fi
export EVAL_TRAIN
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/50_benchmark.sh" "$@"
