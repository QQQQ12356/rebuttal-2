#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=comi
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
export INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"
exec bash "$ENTRY_DIR/00_setup_environment.sh" "$@"
