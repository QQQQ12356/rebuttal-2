#!/usr/bin/env bash
# First create/activate the conda or venv environment in docs/benchmark/01_environment.md.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Torch wheel source: CUDA 12.6 by default; adapt to your Linux driver/toolkit.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
# INSTALL_DEEPSPEED: 1 installs multi-GPU ZeRO support; 0 installs DDP-only dependencies.
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-1}"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install torch==2.8.0 --index-url "$TORCH_INDEX"
"$PYTHON_BIN" -m pip install -r requirements.txt -r requirements-tools.txt
if [[ "$INSTALL_DEEPSPEED" == 1 ]]; then
    "$PYTHON_BIN" -m pip install -r requirements-deepspeed.txt
fi
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" scripts/benchmark/py/00_check_environment.py
