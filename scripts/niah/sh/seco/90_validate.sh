#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

exec bash "$ENTRY_DIR/90_validate.sh" "$@"
