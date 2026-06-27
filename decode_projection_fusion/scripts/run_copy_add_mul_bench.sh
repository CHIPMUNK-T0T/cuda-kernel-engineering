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
FEATURES="${FEATURES:-2048,4096,8192,11008,16384}"
RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-20}"
IMPLEMENTATIONS="${IMPLEMENTATIONS:-torch_add_mul,torch_clone_add_mul,triton_add_mul,triton_copy_add_mul}"

"${PYTHON_BIN}" decode_projection_fusion/benchmarks/bench_copy_add_mul.py \
  --tokens "${TOKENS}" \
  --features "${FEATURES}" \
  --runs "${RUNS}" \
  --warmup "${WARMUP}" \
  --implementations "${IMPLEMENTATIONS}" \
  --run-name "copy-add-mul" \
  "$@"
