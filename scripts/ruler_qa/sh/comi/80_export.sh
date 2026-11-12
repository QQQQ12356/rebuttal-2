#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=comi
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ENTRY_DIR/80_export.sh" "$@"
