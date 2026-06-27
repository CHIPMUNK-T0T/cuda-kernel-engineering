# Mini Decoder KV Context Sweep

Environment: RTX 4070, `dtype=float16`, `preset=qwen_like_2b`, `hidden=2048`, `layers=16`, `heads=16`, `kv_heads=4`, `head_dim=128`, `runs=30`, `warmup=5`.

Run records:

```text
mini_decoder_kv/results/rtx4070/runs/20260620-115605-qwen2b-ctx128/
mini_decoder_kv/results/rtx4070/runs/20260620-115613-qwen2b-ctx512/
mini_decoder_kv/results/rtx4070/runs/20260620-115619-qwen2b-ctx2048/
```

| context len | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 2224.288 | 1570.112 | 1798.608 | 1.42x | 1.24x | CUDA fused |
| 512 | 2070.256 | 1528.784 | 1762.304 | 1.35x | 1.17x | CUDA fused |
| 2048 | 2671.760 | 2292.736 | 2064.736 | 1.17x | 1.29x | Triton fused |

Notes:

- Fused residual RMSNorm still improves latency after adding QKV projection, attention, KV cache read, and output projection.
- The best implementation changes by context length in this synthetic workload.
- Longer context increases the attention / KV cache side, so the RMSNorm contribution can become less dominant.
