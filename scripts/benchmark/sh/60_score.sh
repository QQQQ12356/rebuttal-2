#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

PREDICTIONS="${PREDICTIONS:-$BENCHMARK_ROOT/benchmarks/REPLACE_WITH_BENCHMARK/ratio_16/predictions.jsonl}"
# Default overwrites/recomputes only sibling metrics.json, preserving timing annotations.
"$PYTHON_BIN" "$PY_ENTRY_DIR/60_score.py" --predictions "$PREDICTIONS" "$@"
