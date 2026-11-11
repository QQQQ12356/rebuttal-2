#!/usr/bin/env bash
# Linux-only validation; random toy corpora and synthetic scores, never downloads/training.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

"$PYTHON_BIN" -m ruff check seco scripts/benchmark/py scripts/niah/py tests
"$PYTHON_BIN" "$PY_ENTRY_DIR/90_validate.py"
for entry in scripts/niah/sh/*.sh; do
    bash -n "$entry"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests/niah -v
