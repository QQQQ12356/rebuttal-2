#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=ram
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export REGISTRY="${REGISTRY:-configs/niah/sources.json}"             # Explicit train/test source roles.
export RAW_DIR="${RAW_DIR:-$NIAH_ROOT/raw}"
exec bash "$ENTRY_DIR/10_download_data.sh" "$@"
