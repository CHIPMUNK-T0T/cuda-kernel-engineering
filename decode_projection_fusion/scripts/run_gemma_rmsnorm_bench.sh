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

TOKENS="${TOKENS:-1,8,128}"
HIDDEN="${HIDDEN:-2048,4096,8192}"
RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-20}"
IMPLEMENTATIONS="${IMPLEMENTATIONS:-torch_gemma_native,triton_gemma_fused,cuda_gemma_fused}"

"${PYTHON_BIN}" decode_projection_fusion/benchmarks/bench_gemma_rmsnorm.py \
  --tokens "${TOKENS}" \
  --hidden "${HIDDEN}" \
  --runs "${RUNS}" \
  --warmup "${WARMUP}" \
  --implementations "${IMPLEMENTATIONS}" \
  --run-name "gemma-rmsnorm" \
  "$@"
