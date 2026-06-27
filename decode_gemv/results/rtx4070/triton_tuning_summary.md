# Triton GEMV Tuning Result

Run:

```text
decode_gemv/results/rtx4070/triton_tuning/runs/20260620-165310-triton-gemv-tuning/
```

## Best Configs

| shape | best block_k | best block_n | latency us | speedup vs torch_linear | GB/s | TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| `1x2048x8192` | 128 | 32 | 41.104 | 1.868 | 816.828 | 0.816 |
| `1x2048x11008` | 128 | 32 | 108.544 | 0.933 | 415.637 | 0.415 |
| `1x4096x8192` | 128 | 32 | 154.832 | 0.956 | 433.589 | 0.433 |
| `1x4096x11008` | 128 | 32 | 202.752 | 0.964 | 444.917 | 0.445 |

## Read

- Best config was consistently `BLOCK_K=128, BLOCK_N=32`.
- `1x2048x8192` is the clear win: tuned Triton is `1.868x` faster than `torch_linear`.
- Larger `in_features` / `out_features` shapes are close to cuBLAS but still slightly slower.
- Small `BLOCK_K=32` is generally weak, likely because the kernel loops too many times over the reduction dimension.
- Large `BLOCK_N=128` is not best in this implementation, likely because each program carries a wider output tile and becomes less efficient for this memory-streaming pattern.

## Interpretation

The first Triton kernel was not generally faster than cuBLAS, but tuning exposed a meaningful shape-specific win.

This is useful for the project story:

- vLLM profiling identified decode GEMV as the dominant backend cost.
- Baseline showed cuBLAS is strong.
- Nsight showed GEMV is memory-throughput limited.
- Tuning showed that a custom Triton kernel can beat cuBLAS for at least one representative decode projection shape.

The next technical question is whether the win survives in a projection block or real backend-like sequence, not just a standalone GEMV microbenchmark.

## Next

Profile the best config with Nsight:

```bash
GEMV_BLOCK_K=128 GEMV_BLOCK_N=32 \
  bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 2048 8192
```

Then compare it against:

```text
decode_gemv/results/rtx4070/nsight/20260620-163742-torch_linear-tokens1-in2048-out8192/
```
