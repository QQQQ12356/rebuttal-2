#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:-$NIAH_ROOT/benchmarks/REPLACE_WITH_BENCHMARK/ratio_16/predictions.jsonl}"
"$PYTHON_BIN" scripts/niah/py/60_score.py --predictions "$PREDICTIONS" "$@"
