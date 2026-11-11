#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo 'Run RULER QA on Linux; Windows is for editing only.' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
METHOD="${COMPRESSION_METHOD:-seco}"
BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
EXPERIMENT_ROOT="${ARTIFACT_ROOT:-artifacts}/$METHOD/$BACKBONE"
RULER_QA_ROOT="${RULER_QA_ROOT:-$EXPERIMENT_ROOT/ruler_qa}"
DEFAULT_MODEL_PATH="$EXPERIMENT_ROOT/model"
PY_ENTRY_DIR="scripts/ruler_qa/py"
if [[ -n "${COMPRESSION_METHOD:-}" ]]; then
    PY_ENTRY_DIR="$PY_ENTRY_DIR/$METHOD"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/artifacts/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

timestamp() {
    date -u +%Y%m%dT%H%M%SZ
}
