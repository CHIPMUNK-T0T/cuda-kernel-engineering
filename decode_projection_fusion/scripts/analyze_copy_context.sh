#!/usr/bin/env bash
set -euo pipefail

TRACE=${TRACE:-backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only/cuda_gpu_trace.csv}
START_S=${START_S:-45}
END_S=${END_S:-70}
OUT_DIR=${OUT_DIR:-decode_projection_fusion/results/rtx4070/copy_context}
TARGET_FAMILY=${TARGET_FAMILY:-copy / cast}
RADIUS=${RADIUS:-2}
TOP=${TOP:-40}

.venv/bin/python decode_projection_fusion/benchmarks/analyze_copy_context.py \
  --trace "$TRACE" \
  --start-s "$START_S" \
  --end-s "$END_S" \
  --target-family "$TARGET_FAMILY" \
  --radius "$RADIUS" \
  --out-dir "$OUT_DIR" \
  --top "$TOP"
