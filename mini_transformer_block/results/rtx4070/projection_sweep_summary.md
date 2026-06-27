# Projection Sweep Summary

Purpose: change projection output size to control GEMM weight and measure how much fused residual RMSNorm still contributes to the block.

## Latency

`RUNS=50`, `WARMUP=10`, `dtype=float16`, RTX 4070.

| mode | tokens | hidden | out features | PyTorch unfused us | CUDA fused us | Triton fused us | best | CUDA vs PyTorch | Triton vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| decode | 1 | 4096 | 512 | 47.984 | 15.184 | 76.608 | CUDA fused | 3.16x | 0.63x |
| decode | 1 | 4096 | 1024 | 49.152 | 23.312 | 32.496 | CUDA fused | 2.11x | 1.51x |
| decode | 1 | 4096 | 4096 | 114.688 | 104.224 | 92.048 | Triton fused | 1.10x | 1.25x |
| prefill | 512 | 8192 | 512 | 348.160 | 123.904 | 123.840 | Triton fused | 2.81x | 2.81x |
| prefill | 512 | 8192 | 1024 | 411.648 | 214.864 | 232.448 | CUDA fused | 1.92x | 1.77x |
| prefill | 512 | 8192 | 8192 | 1445.888 | 1184.720 | 1154.560 | Triton fused | 1.22x | 1.25x |

## Nsight Systems: out_features=512

`iters=1`, values from `cuda_gpu_kern_sum.csv`.

| mode | implementation | tokens | hidden | out features | RMSNorm side us | matmul side us | RMSNorm side share |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | PyTorch unfused | 1 | 4096 | 512 | 17.344 | 12.224 | 58.7% |
| decode | CUDA fused | 1 | 4096 | 512 | 3.648 | 13.376 | 21.4% |
| decode | Triton fused | 1 | 4096 | 512 | 1.952 | 13.024 | 13.0% |
| prefill | PyTorch unfused | 512 | 8192 | 512 | 244.068 | 97.058 | 71.5% |
| prefill | CUDA fused | 512 | 8192 | 512 | 64.417 | 94.434 | 40.6% |
| prefill | Triton fused | 512 | 8192 | 512 | 51.393 | 96.225 | 34.8% |

## Conclusion

When projection/GEMM is heavy, fused residual RMSNorm improves only part of the block latency. When projection is lighter, the RMSNorm side becomes a large share of GPU kernel time, so fusion has a much larger visible effect.

