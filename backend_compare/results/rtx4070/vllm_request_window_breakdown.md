# vLLM Qwen3.5 2B Request Window Breakdown

## Purpose

`cuda_gpu_kern_sum.csv` の全体集計だけでは、server ready 前の vLLM warmup / init kernel が混ざる。
ここでは `cuda_gpu_trace.csv` を request が走っていた時間帯に寄せて再集計し、次の最適化候補を選ぶ。

## Source

- record: `backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only`
- request record: `backend_compare/results/rtx4070/profile_requests/runs/20260620-152413-openai_compatible-Qwen-Qwen3-5-2B`
- model: `Qwen/Qwen3.5-2B`
- request: warmup `1`, measured runs `3`, generated tokens `128`

## Important Correction

The initial `cuda_gpu_kern_sum.csv` aggregate included kernels before the server became ready.
The large `FillFunctor<int>` entry was mostly pre-ready / warmup work, not the measured request path.

Approximate windows from `cuda_gpu_trace.csv`:

| window | interpretation |
|---|---|
| `0-43s` | pre-ready / warmup-heavy |
| `45-70s` | request window |

## Request Window Summary

Approximate request window: `45s-70s` from capture start.

| family | total time ns | share | kernels |
|---|---:|---:|---:|
| cuBLAS GEMV | 4,149,161,560 | 86.788% | 58,424 |
| PyTorch copy / cast | 158,452,960 | 3.314% | 109,376 |
| PyTorch elementwise math | 157,019,871 | 3.284% | 160,205 |
| norm-related | 68,315,257 | 1.429% | 71,659 |
| PyTorch RMSNorm reduce | 61,151,991 | 1.279% | 31,223 |
| GEMM / CUTLASS / cuBLAS | 57,975,130 | 1.213% | 6,320 |
| Qwen hybrid / Mamba-like | 56,636,335 | 1.185% | 18,495 |
| vLLM SwiGLU `act_and_mul_kernel` | 19,479,816 | 0.407% | 12,285 |
| KV cache auxiliary | 8,317,129 | 0.174% | 5,119 |
| PyTorch fill bf16 | 6,975,662 | 0.146% | 9,282 |
| RoPE | 3,195,355 | 0.067% | 3,072 |
| sampling / softmax | 2,796,348 | 0.058% | 512 |
| PyTorch fill int | 812,413 | 0.017% | 1,040 |

## Candidate Assessment

| candidate | request impact | implementation difficulty | vLLM novelty / risk | read |
|---|---:|---|---|---|
| decode GEMV / small-batch linear | very high | high | medium | Biggest request bottleneck. Meaningful if the goal is vLLM tokens/sec, but hard to beat cuBLAS. |
| PyTorch copy/cast + elementwise fusion | medium | medium | medium | Smaller than GEMV, but many tiny PyTorch native kernels remain. More realistic custom CUDA/Triton target. |
| SwiGLU fused activation | low | low-medium | low | vLLM already uses `vllm::act_and_mul_kernel`; not a strong target for vLLM impact. |
| RMSNorm continuation | low | already done | low | Request path still has norm kernels, but impact is small. |
| Qwen hybrid / Mamba-like kernels | low-medium | high | high | Model-specific and already has Triton/FLA kernels. Risky as a next portfolio step. |

## Recommendation

For vLLM impact, the strongest target is decode GEMV / small-batch linear projection.
For implementation feasibility, the strongest target is PyTorch native copy/cast + elementwise fusion.

SwiGLU should not be the first vLLM-driven target, because vLLM already has a dedicated `act_and_mul_kernel` and its measured request-window share is only about `0.407%`.

The next theme is:

1. `decode_gemv/`: harder, but directly targets the dominant vLLM request bottleneck.

`elementwise_fusion/` remains a secondary candidate for residual PyTorch native copy/cast/math kernels.
