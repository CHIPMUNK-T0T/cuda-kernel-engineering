#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export UV_CACHE_DIR

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" decode_gemv/benchmarks/bench_projection_types.py \
  --tokens "${GEMV_TOKENS:-1}" \
  --hidden "${GEMV_HIDDEN:-2048,4096}" \
  --intermediate "${GEMV_INTERMEDIATE:-8192,11008}" \
  --projections "${GEMV_PROJECTIONS:-qkv,wo,mlp_up,mlp_down}" \
  --dtype "${GEMV_DTYPE:-bfloat16}" \
  --warmup "${GEMV_WARMUP:-10}" \
  --runs "${GEMV_RUNS:-50}" \
  --implementations "${GEMV_IMPLEMENTATIONS:-torch_linear,triton_tuned}" \
  --run-name "${GEMV_RUN_NAME:-projection-types-deduped}" \
  --dedupe-shapes
