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

"${PYTHON_BIN}" mini_llm_decode/benchmarks/bench_decode.py "$@"
