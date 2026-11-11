#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl for one ratio}"
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL to a chat model served by JUDGE_BASE_URL}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-60}"
JUDGE_ATTEMPTS="${JUDGE_ATTEMPTS:-4}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/65_judge.py" --predictions "$PREDICTIONS" \
    --judge-model "$JUDGE_MODEL" --base-url "$JUDGE_BASE_URL" \
    --timeout "$JUDGE_TIMEOUT" --attempts "$JUDGE_ATTEMPTS" "$@"
