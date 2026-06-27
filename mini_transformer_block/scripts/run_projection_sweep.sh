#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

runs="${RUNS:-50}"
warmup="${WARMUP:-10}"

for out_features in 512 1024 4096; do
  bash mini_transformer_block/scripts/run_bench.sh \
    --tokens 1 \
    --hidden 4096 \
    --out-features "$out_features" \
    --runs "$runs" \
    --warmup "$warmup" \
    --run-name "projection-sweep-decode-out${out_features}"
done

for out_features in 512 1024 8192; do
  bash mini_transformer_block/scripts/run_bench.sh \
    --tokens 512 \
    --hidden 8192 \
    --out-features "$out_features" \
    --runs "$runs" \
    --warmup "$warmup" \
    --run-name "projection-sweep-prefill-out${out_features}"
done

