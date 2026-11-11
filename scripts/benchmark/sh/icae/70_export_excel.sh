#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export EXCEL_PATH="${EXCEL_PATH:-$BENCHMARK_ROOT/reports/benchmark_$(timestamp)_${RANDOM}.xlsx}"
exec bash "$ENTRY_DIR/70_export_excel.sh" "$@"
