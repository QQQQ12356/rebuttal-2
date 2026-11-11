#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

REGISTRY="${REGISTRY:-configs/niah/sources.json}"             # Explicit train/test source roles.
RAW_DIR="${RAW_DIR:-$NIAH_ROOT/raw}"
# Default: PG19 training prose, Paul Graham test prose, Cogito Bits/Bind train splits.
# Append --sources pg19_test to download the optional PG19 test corpus separately.
"$PYTHON_BIN" "$PY_ENTRY_DIR/10_download_data.py" --registry "$REGISTRY" --output "$RAW_DIR" "$@"
