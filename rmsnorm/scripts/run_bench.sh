#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PATH="$PWD/.venv/bin:$PATH"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: UV_CACHE_DIR=.uv-cache uv venv .venv" >&2
  exit 1
fi

.venv/bin/python rmsnorm/benchmarks/bench_rmsnorm.py --out-dir rmsnorm/results/rtx4070 "$@"
