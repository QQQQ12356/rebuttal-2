#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

SOURCES="${SOURCES:-configs/ruler_qa/sources.json}"
RAW_DIR="${RAW_DIR:-$RULER_QA_ROOT/raw}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/10_download_data.py" --sources "$SOURCES" --output "$RAW_DIR" "$@"
