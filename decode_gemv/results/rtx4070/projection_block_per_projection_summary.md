# Projection Block Per-Projection Tuning Summary

Source: `decode_gemv/results/rtx4070/projection_block/runs/20260623-163716-projection-block-per-projection-tuned-v2/summary.csv`

| shape | torch us | fixed Triton us | per-projection Triton us | fixed/torch | per-proj/torch | per-proj/fixed |
|---|---:|---:|---:|---:|---:|---:|
| 1xhidden2048xintermediate8192 | 278.528 | 357.376 | 319.088 | 1.283 | 1.146 | 0.893 |
| 1xhidden2048xintermediate11008 | 345.408 | 460.624 | 395.168 | 1.334 | 1.144 | 0.858 |
| 1xhidden4096xintermediate8192 | 693.184 | 770.032 | 738.304 | 1.111 | 1.065 | 0.959 |
| 1xhidden4096xintermediate11008 | 837.632 | 979.728 | 884.656 | 1.170 | 1.056 | 0.903 |

Read:

- Per-projection Triton tuning improves every measured projection-block shape versus fixed `BLOCK_K=128, BLOCK_N=32`.
- The improvement is not enough to beat `torch_linear` / cuBLAS at block level yet.
- Current direction: projection-specific tuning is useful, but block-level speedup likely needs either a stronger CUDA GEMV implementation or fusion around projection outputs.
