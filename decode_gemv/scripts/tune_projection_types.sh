#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" decode_gemv/benchmarks/tune_projection_types.py \
  --tokens "${GEMV_TOKENS:-1}" \
  --hidden "${GEMV_HIDDEN:-2048,4096}" \
  --intermediate "${GEMV_INTERMEDIATE:-8192,11008}" \
  --projections "${GEMV_PROJECTIONS:-wo,mlp_down}" \
  --block-k "${GEMV_BLOCK_K:-32,64,128,256}" \
  --block-n "${GEMV_BLOCK_N:-16,32,64,128}" \
  --dtype "${GEMV_DTYPE:-bfloat16}" \
  --warmup "${GEMV_WARMUP:-10}" \
  --runs "${GEMV_RUNS:-50}" \
  --run-name "${GEMV_RUN_NAME:-projection-type-tuning-wo-mlp-down}" \
  --dedupe-shapes
