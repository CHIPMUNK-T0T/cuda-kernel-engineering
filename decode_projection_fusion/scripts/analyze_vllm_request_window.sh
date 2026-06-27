#!/usr/bin/env bash
set -euo pipefail

TRACE=${TRACE:-backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only/cuda_gpu_trace.csv}
START_S=${START_S:-45}
END_S=${END_S:-70}
OUT_DIR=${OUT_DIR:-decode_projection_fusion/results/rtx4070/request_window}
TOP=${TOP:-40}

.venv/bin/python decode_projection_fusion/benchmarks/analyze_nsys_kernels.py \
  --trace "$TRACE" \
  --start-s "$START_S" \
  --end-s "$END_S" \
  --out-dir "$OUT_DIR" \
  --top "$TOP"

