#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl for one ratio}"
"$PYTHON_BIN" scripts/ruler_qa/py/60_score.py --predictions "$PREDICTIONS" "$@"
