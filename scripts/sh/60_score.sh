#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:-artifacts/benchmarks/REPLACE_WITH_BENCHMARK/ratio_16/predictions.jsonl}"
# Default overwrites/recomputes only sibling metrics.json, preserving timing annotations.
"$PYTHON_BIN" scripts/py/60_score.py --predictions "$PREDICTIONS" "$@"
