#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

size="${1:-2b}"
base_url="${2:-http://127.0.0.1:8000}"
runs="${3:-3}"
max_tokens="${4:-128}"
warmup="${5:-1}"
request_delay="${REQUEST_DELAY:-0}"
request_wait_timeout="${REQUEST_WAIT_TIMEOUT:-300}"
request_after_ready_delay="${REQUEST_AFTER_READY_DELAY:-0}"

case "$size" in
  2b)
    model="Qwen/Qwen3.5-2B"
    ;;
  4b)
    model="Qwen/Qwen3.5-4B"
    ;;
  *)
    echo "Usage: $0 [2b|4b] [base_url] [runs] [max_tokens] [warmup]" >&2
    exit 2
    ;;
esac

if [[ "$request_delay" != "0" ]]; then
  echo "request_delay=${request_delay}s"
  sleep "$request_delay"
fi

echo "waiting_for_server=$base_url timeout=${request_wait_timeout}s"
.venv/bin/python - "$base_url" "$request_wait_timeout" <<'PY'
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
timeout_s = float(sys.argv[2])
deadline = time.monotonic() + timeout_s
last_error = None

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as resp:
            if 200 <= resp.status < 500:
                print("server_ready=true")
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(2)

print(f"server_ready=false last_error={last_error}", file=sys.stderr)
raise SystemExit(1)
PY

if [[ "$request_after_ready_delay" != "0" ]]; then
  echo "request_after_ready_delay=${request_after_ready_delay}s"
  sleep "$request_after_ready_delay"
fi

.venv/bin/python backend_compare/benchmarks/bench_backend.py \
  --backend openai_compatible \
  --model "$model" \
  --base-url "$base_url" \
  --prompt backend_compare/prompts/decode_japanese.txt \
  --warmup "$warmup" \
  --runs "$runs" \
  --max-tokens "$max_tokens" \
  --out-dir backend_compare/results/rtx4070/profile_requests
