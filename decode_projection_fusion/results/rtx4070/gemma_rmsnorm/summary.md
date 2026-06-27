# Gemma-style RMSNorm Mini Benchmark

| implementation | tokens | hidden | dtype | latency us | est. GB/s | max abs error | vs torch native |
|---|---:|---:|---|---:|---:|---:|---:|
| torch_gemma_native | 1 | 2048 | bfloat16 | 37.888 | 1.622 | 0 | 1.000 |
| triton_gemma_fused | 1 | 2048 | bfloat16 | 13.840 | 0.888 | 0 | 0.365 |
| cuda_gemma_fused | 1 | 2048 | bfloat16 | 7.168 | 1.714 | 0 | 0.189 |
| torch_gemma_native | 1 | 4096 | bfloat16 | 39.776 | 3.089 | 0 | 1.000 |
| triton_gemma_fused | 1 | 4096 | bfloat16 | 13.536 | 1.816 | 0.00390625 | 0.340 |
| cuda_gemma_fused | 1 | 4096 | bfloat16 | 6.928 | 3.547 | 0 | 0.174 |
| torch_gemma_native | 1 | 8192 | bfloat16 | 38.976 | 6.305 | 0 | 1.000 |
| triton_gemma_fused | 1 | 8192 | bfloat16 | 13.312 | 3.692 | 0 | 0.342 |
| cuda_gemma_fused | 1 | 8192 | bfloat16 | 7.168 | 6.857 | 0 | 0.184 |
| torch_gemma_native | 8 | 2048 | bfloat16 | 39.936 | 12.308 | 0 | 1.000 |
| triton_gemma_fused | 8 | 2048 | bfloat16 | 14.144 | 6.950 | 0 | 0.354 |
| cuda_gemma_fused | 8 | 2048 | bfloat16 | 7.344 | 13.386 | 0 | 0.184 |
| torch_gemma_native | 8 | 4096 | bfloat16 | 39.936 | 24.615 | 0 | 1.000 |
| triton_gemma_fused | 8 | 4096 | bfloat16 | 14.160 | 13.885 | 0.00390625 | 0.355 |
| cuda_gemma_fused | 8 | 4096 | bfloat16 | 7.152 | 27.490 | 0 | 0.179 |
| torch_gemma_native | 8 | 8192 | bfloat16 | 39.936 | 49.231 | 0 | 1.000 |
| triton_gemma_fused | 8 | 8192 | bfloat16 | 13.536 | 29.050 | 0.015625 | 0.339 |
| cuda_gemma_fused | 8 | 8192 | bfloat16 | 7.168 | 54.857 | 0 | 0.179 |
| torch_gemma_native | 128 | 2048 | bfloat16 | 39.936 | 196.923 | 0 | 1.000 |
| triton_gemma_fused | 128 | 2048 | bfloat16 | 13.424 | 117.168 | 0.00390625 | 0.336 |
| cuda_gemma_fused | 128 | 2048 | bfloat16 | 7.168 | 219.429 | 0 | 0.179 |
| torch_gemma_native | 128 | 4096 | bfloat16 | 39.936 | 393.846 | 0 | 1.000 |
| triton_gemma_fused | 128 | 4096 | bfloat16 | 13.312 | 236.308 | 0.00390625 | 0.333 |
| cuda_gemma_fused | 128 | 4096 | bfloat16 | 7.168 | 438.857 | 0.0078125 | 0.179 |
| torch_gemma_native | 128 | 8192 | bfloat16 | 51.200 | 614.400 | 0 | 1.000 |
| triton_gemma_fused | 128 | 8192 | bfloat16 | 13.456 | 467.558 | 0.0078125 | 0.263 |
| cuda_gemma_fused | 128 | 8192 | bfloat16 | 8.192 | 768.000 | 0.0078125 | 0.160 |

## Best By Shape

| tokens | hidden | best implementation | latency us | speedup vs torch native |
|---:|---:|---|---:|---:|
| 1 | 2048 | cuda_gemma_fused | 7.168 | 5.286 |
| 1 | 4096 | cuda_gemma_fused | 6.928 | 5.741 |
| 1 | 8192 | cuda_gemma_fused | 7.168 | 5.437 |
| 8 | 2048 | cuda_gemma_fused | 7.344 | 5.438 |
| 8 | 4096 | cuda_gemma_fused | 7.152 | 5.584 |
| 8 | 8192 | cuda_gemma_fused | 7.168 | 5.571 |
| 128 | 2048 | cuda_gemma_fused | 7.168 | 5.571 |
| 128 | 4096 | cuda_gemma_fused | 7.168 | 5.571 |
| 128 | 8192 | cuda_gemma_fused | 8.192 | 6.250 |
