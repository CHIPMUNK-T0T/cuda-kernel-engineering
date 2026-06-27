#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

size="${1:-2b}"
runs="${2:-3}"
max_tokens="${3:-128}"
num_ctx="${4:-4096}"
warmup="${5:-1}"

case "$size" in
  2b|4b)
    model="qwen3.5:${size}"
    ;;
  *)
    echo "Usage: $0 [2b|4b] [runs] [max_tokens] [num_ctx]" >&2
    exit 2
    ;;
esac

.venv/bin/python backend_compare/benchmarks/bench_backend.py \
  --backend ollama \
  --model "$model" \
  --base-url "http://127.0.0.1:11434" \
  --prompt backend_compare/prompts/decode_japanese.txt \
  --warmup "$warmup" \
  --runs "$runs" \
  --max-tokens "$max_tokens" \
  --num-ctx "$num_ctx"
