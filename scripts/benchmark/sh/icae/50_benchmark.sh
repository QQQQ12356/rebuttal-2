#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"    # Each process holds one complete encoder/decoder pair.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # More GPUs increase throughput, not per-model capacity.
export CHECKPOINT="${CHECKPOINT:-$BENCHMARK_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
export PREPARED_DIR="${PREPARED_DIR:-$BENCHMARK_ROOT/prepared/default}"
export BATCH_SIZE="${BATCH_SIZE:-1}"                             # Increase only when each GPU has enough memory.
export QA_MAX_NEW_TOKENS="${QA_MAX_NEW_TOKENS:-64}"
export NEWS_MAX_NEW_TOKENS="${NEWS_MAX_NEW_TOKENS:-512}"
export DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
export RATIOS="${RATIOS:-4 8 16 32}"
exec bash "$ENTRY_DIR/50_benchmark.sh" "$@"
