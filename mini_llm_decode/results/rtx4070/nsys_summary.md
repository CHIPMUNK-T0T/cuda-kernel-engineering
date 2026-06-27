# Mini LLM Decode Nsight Systems Summary

Environment: RTX 4070, `dtype=float16`, `tokens=1`, `hidden=4096`, `layers=32`, distinct projection weights, `iters=1`.

Run records:

```text
mini_llm_decode/results/rtx4070/nsys/20260620-113716-pytorch_unfused-layers32-hidden4096-distinct/
mini_llm_decode/results/rtx4070/nsys/20260620-113748-cuda_residual_fused-layers32-hidden4096-distinct/
mini_llm_decode/results/rtx4070/nsys/20260620-113822-triton_residual_fused-layers32-hidden4096-distinct/
```

| implementation | RMSNorm side us | matmul us | total GPU kernel us | RMSNorm side share | matmul share |
|---|---:|---:|---:|---:|---:|
| PyTorch unfused | 495.202 | 2417.168 | 2912.370 | 17.0% | 83.0% |
| CUDA fused | 99.521 | 2719.312 | 2818.833 | 3.5% | 96.5% |
| Triton fused | 50.528 | 2411.375 | 2461.903 | 2.1% | 97.9% |

Kernel counts:

| implementation | RMSNorm side kernel instances | matmul kernel instances |
|---|---:|---:|
| PyTorch unfused | 352 | 32 |
| CUDA fused | 32 | 32 |
| Triton fused | 32 | 32 |

Notes:

- PyTorch unfused launches multiple kernels per layer for add / pow / reduce / rsqrt / mul / copy.
- CUDA and Triton fused residual RMSNorm reduce the RMSNorm side to one kernel per layer.
- After fusion, matmul dominates the GPU kernel time. This explains why mini decode speedup is around 1.23x to 1.25x rather than matching the larger RMSNorm-only speedup.
