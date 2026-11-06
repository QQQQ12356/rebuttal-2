#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Activate a Linux Python 3.10/3.11 environment first. Existing SeCo env can be reused.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-1}"                 # 0 for DDP-only training.
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install torch==2.8.0 --index-url "$TORCH_INDEX"
"$PYTHON_BIN" -m pip install -r requirements-niah.txt -r requirements-tools.txt
if [[ "$INSTALL_DEEPSPEED" == 1 ]]; then
    "$PYTHON_BIN" -m pip install -r requirements-deepspeed.txt
fi
"$PYTHON_BIN" -m pip check
