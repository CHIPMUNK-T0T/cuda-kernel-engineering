# Copy/Cast Context Analysis

## Source

- trace: `decode_projection_fusion/results/rtx4070/vllm_gemma_patch_nsys/20260624-230057-vllm-qwen35-2b-request_only_gemma_patch/cuda_gpu_trace.csv`
- window: `45.0s-70.0s`
- target family: `copy / cast`
- context radius: `2`
- kernels in window: `0`
- target kernels: `0`
- target total time: `0` ns

Note: context is based on `Start (ns)` order. It is a practical adjacency signal, not a strict dependency graph.

## Target Kernel Summary

| target | total target time ns | share | instances | avg ns |
|---|---:|---:|---:|---:|

## Previous Family

| previous family | total target time ns | share | instances | avg target ns |
|---|---:|---:|---:|---:|

## Next Family

| next family | total target time ns | share | instances | avg target ns |
|---|---:|---:|---:|---:|

## Family Triplets

| previous | target | next | total target time ns | share | instances | avg target ns |
|---|---|---|---:|---:|---:|---:|

## Initial Read Guide

- `cuBLAS GEMV -> copy/cast -> elementwise` が多ければ、projection 後処理の layout/cast が候補。
- `copy/cast -> copy/cast` が多ければ、連続 copy/cast の削減候補。
- `Qwen hybrid/state-space` 周辺に偏るなら、Qwen3.5 固有 path の調査が必要。
- `sampling / softmax` 周辺に偏るなら、decode 後段の logits/sampling 側を調査する。
