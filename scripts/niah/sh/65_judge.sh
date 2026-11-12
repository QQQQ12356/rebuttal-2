#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl for one ratio}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-60}"
JUDGE_ATTEMPTS="${JUDGE_ATTEMPTS:-4}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/65_judge.py" --predictions "$PREDICTIONS" \
    --timeout "$JUDGE_TIMEOUT" --attempts "$JUDGE_ATTEMPTS" "$@"
