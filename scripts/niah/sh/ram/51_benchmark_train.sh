#!/usr/bin/env bash
set -euo pipefail
# Shared settings are in this method's 50_benchmark.sh.
export EVAL_TRAIN=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/50_benchmark.sh" "$@"
