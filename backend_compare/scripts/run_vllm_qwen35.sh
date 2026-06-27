#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

size="${1:-2b}"
base_url="${2:-http://127.0.0.1:8000}"
runs="${3:-5}"
max_tokens="${4:-128}"
warmup="${5:-1}"

case "$size" in
  2b)
    model="Qwen/Qwen3.5-2B"
    ;;
  4b)
    model="Qwen/Qwen3.5-4B"
    ;;
  *)
    echo "Usage: $0 [2b|4b] [base_url] [runs] [max_tokens]" >&2
    exit 2
    ;;
esac

.venv/bin/python backend_compare/benchmarks/bench_backend.py \
  --backend openai_compatible \
  --model "$model" \
  --base-url "$base_url" \
  --prompt backend_compare/prompts/decode_japanese.txt \
  --warmup "$warmup" \
  --runs "$runs" \
  --max-tokens "$max_tokens"
