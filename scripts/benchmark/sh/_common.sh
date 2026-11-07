#!/usr/bin/env bash
# Shared path/environment setup. Source this file; edit settings in numbered entries.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo 'Run experiments on Linux; Windows is for editing only.' >&2; exit 2; }
cd -- "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/artifacts/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

timestamp() {
    date -u +%Y%m%dT%H%M%SZ
}
