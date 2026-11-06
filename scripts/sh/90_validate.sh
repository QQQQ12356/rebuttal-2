#!/usr/bin/env bash
# Linux verification only. No data/model download, training or CUDA allocation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

"$PYTHON_BIN" -m ruff check seco scripts/py tests
"$PYTHON_BIN" scripts/py/90_validate.py
for entry in scripts/sh/*.sh; do
    bash -n "$entry"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests -v
