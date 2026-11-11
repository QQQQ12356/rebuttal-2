#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
source "$(dirname -- "${BASH_SOURCE[0]}")/../../benchmark/sh/_common.sh"
export ICAE_PRETRAIN_ROOT="${ICAE_PRETRAIN_ROOT:-$EXPERIMENT_ROOT/icae_pretrain}"
