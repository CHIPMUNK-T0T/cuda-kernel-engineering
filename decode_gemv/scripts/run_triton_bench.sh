#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

if [[ -x ".venv/bin/python" ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

TOKENS="${TOKENS:-1}"
RUNS="${RUNS:-50}"
WARMUP="${WARMUP:-10}"
IMPLEMENTATIONS="${IMPLEMENTATIONS:-torch_matmul,torch_linear,triton_gemv}"

"${PYTHON_BIN}" decode_gemv/benchmarks/bench_gemv.py \
  --tokens "${TOKENS}" \
  --runs "${RUNS}" \
  --warmup "${WARMUP}" \
  --implementations "${IMPLEMENTATIONS}" \
  --run-name "triton-gemv-tokens${TOKENS}" \
  "$@"
