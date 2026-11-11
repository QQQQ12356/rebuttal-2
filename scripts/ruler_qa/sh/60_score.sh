#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl for one ratio}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/60_score.py" --predictions "$PREDICTIONS" "$@"
