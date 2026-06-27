# Decode Projection Fusion Request-Window Analysis

## Source

- trace: `decode_projection_fusion/results/rtx4070/vllm_gemma_patch_nsys/20260624-230057-vllm-qwen35-2b-request_only_gemma_patch/cuda_gpu_trace.csv`
- window: `45.0s-70.0s`
- kernels in window: `0`
- total GPU kernel time: `0` ns

## Family Summary

| family | total time ns | share | instances | avg ns |
|---|---:|---:|---:|---:|

## Candidate Kernels

GEMV/GEMM/attention/norm を除いた、fusion 候補になりうる上位 kernel。

| family | total time ns | share | instances | avg ns | name |
|---|---:|---:|---:|---:|---|

## Initial Read

- cuBLAS GEMV 本体は別テーマ `decode_gemv/` で扱ったため、ここでは主対象にしない。
- 上位 candidate が PyTorch native の copy/cast/elementwise/fill に偏るなら、次は mini reproduction を作る。
- candidate が pre-ready/warmup 由来に見える場合は、window を狭めて再集計する。
