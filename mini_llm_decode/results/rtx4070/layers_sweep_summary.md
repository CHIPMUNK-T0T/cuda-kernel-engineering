# Mini LLM Decode Layers Sweep

Environment: RTX 4070, `dtype=float16`, `tokens=1`, `hidden=4096`, `runs=50`, `warmup=10`.

## Distinct Projection Weights

This uses one `hidden x hidden` projection matrix per layer. It is closer to a real layer stack than the shared-weight fallback.

| layers | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 788.528 | 633.856 | 632.528 | 1.24x | 1.25x | Triton fused |
| 16 | 1542.112 | 1246.128 | 1244.160 | 1.24x | 1.24x | Triton fused |
| 32 | 3071.488 | 2488.848 | 2474.560 | 1.23x | 1.24x | Triton fused |

Run records:

```text
mini_llm_decode/results/rtx4070/runs/20260620-112946-decode-distinct-layers8/
mini_llm_decode/results/rtx4070/runs/20260620-112925-decode-distinct-layers16/
mini_llm_decode/results/rtx4070/runs/20260620-112934-decode-distinct-layers32/
```

## Shared Projection Weights

This reuses one projection matrix across layers to reduce VRAM usage.

| layers | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 780.448 | 627.712 | 630.640 | 1.24x | 1.24x | CUDA fused |
| 16 | 1540.576 | 1242.640 | 1238.512 | 1.24x | 1.24x | Triton fused |
| 32 | 3083.968 | 2484.224 | 2474.336 | 1.24x | 1.25x | Triton fused |

Run records:

```text
mini_llm_decode/results/rtx4070/runs/20260620-111254-decode-shared-layers8/
mini_llm_decode/results/rtx4070/runs/20260620-111345-decode-shared-layers16/
mini_llm_decode/results/rtx4070/runs/20260620-111400-decode-shared-layers32/
```

Notes:

- This is not a production LLM backend measurement.
- The result shows that residual RMSNorm fusion still contributes after returning from kernel/block evaluation to a repeated decode-style workload.
