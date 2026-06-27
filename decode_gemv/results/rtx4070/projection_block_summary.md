# Projection Block Result

Run:

```text
decode_gemv/results/rtx4070/projection_block/runs/20260620-170301-projection-block/
```

## Summary

| tokens | hidden | intermediate | torch linear us | triton tuned us | triton / torch |
|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 8192 | 278.784 | 357.376 | 1.282 |
| 1 | 2048 | 11008 | 365.216 | 461.744 | 1.264 |
| 1 | 4096 | 8192 | 712.704 | 772.096 | 1.083 |
| 1 | 4096 | 11008 | 857.104 | 972.208 | 1.134 |

## Read

- Projection block total latency does not preserve the standalone GEMV win.
- `torch_linear` is faster for every measured projection-block configuration.
- Triton gets closer at `hidden=4096`, but still does not beat cuBLAS-backed `torch_linear`.
- Correctness is acceptable for the measured bf16 outputs.

## Interpretation

The tuned Triton kernel won one standalone shape, but the block contains a mix of different projection shapes:

- QKV: `hidden -> 3 * hidden`
- Wo: `hidden -> hidden`
- MLP gate/up: `hidden -> 2 * intermediate`
- MLP down: `intermediate -> hidden`

That means a single `BLOCK_K=128, BLOCK_N=32` choice is not enough for all projection types.
The block result suggests that cuBLAS handles the mixed projection set better overall.

This is a useful boundary for the project claim:

- Good claim: standalone profiling found a decode GEMV bottleneck, and custom Triton can beat cuBLAS for a selected shape.
- Stronger claim not yet supported: replacing all projection GEMVs with this Triton kernel improves a decoder projection block.

## Next

Break down the projection block by projection type:

1. QKV only
2. Wo only
3. MLP gate/up only
4. MLP down only

Then decide whether to:

- specialize block sizes per projection type,
- fuse adjacent operations where possible,
- or move to CUDA C++ for finer control.
