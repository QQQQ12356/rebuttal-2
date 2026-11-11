#!/usr/bin/env bash
# Linux verification only. No data/model download, training or CUDA allocation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

"$PYTHON_BIN" -m ruff check seco comi icae ram scripts tests
"$PYTHON_BIN" "$PY_ENTRY_DIR/90_validate.py"
shopt -s globstar
for entry in scripts/**/*.sh; do
    bash -n "$entry"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests -v
