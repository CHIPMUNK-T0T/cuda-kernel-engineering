# Projection Type Breakdown Summary

Source:

```text
decode_gemv/results/rtx4070/projection_types/summary.csv
```

## Result

This is the deduped projection-type result. QKV and Wo shapes do not depend on `intermediate`, so repeated QKV/Wo shapes were skipped before summarizing.

| projection | hidden | intermediate | shape | torch us | triton us | triton / torch | winner |
|---|---:|---:|---|---:|---:|---:|---|
| MLP down | 2048 | 8192 | `8192 -> 2048` | 54.928 | 77.488 | 1.411 | torch |
| MLP up | 2048 | 8192 | `2048 -> 16384` | 126.976 | 163.232 | 1.286 | torch |
| QKV | 2048 | 8192 | `2048 -> 6144` | 44.096 | 60.400 | 1.370 | torch |
| Wo | 2048 | 8192 | `2048 -> 2048` | 33.792 | 29.696 | 0.879 | triton |
| MLP down | 2048 | 11008 | `11008 -> 2048` | 79.872 | 160.768 | 2.013 | torch |
| MLP up | 2048 | 11008 | `2048 -> 22016` | 174.784 | 226.048 | 1.293 | torch |
| MLP down | 4096 | 8192 | `8192 -> 4096` | 125.056 | 175.104 | 1.400 | torch |
| MLP up | 4096 | 8192 | `4096 -> 16384` | 265.632 | 310.656 | 1.169 | torch |
| QKV | 4096 | 8192 | `4096 -> 12288` | 196.608 | 225.280 | 1.146 | torch |
| Wo | 4096 | 8192 | `4096 -> 4096` | 56.224 | 49.920 | 0.888 | triton |
| MLP down | 4096 | 11008 | `11008 -> 4096` | 175.104 | 231.184 | 1.320 | torch |
| MLP up | 4096 | 11008 | `4096 -> 22016` | 361.472 | 454.656 | 1.258 | torch |

## Read

- Triton wins only Wo in this deduped matrix: `0.879-0.888x` triton / torch.
- QKV is slower with the current Triton mapping: `1.146-1.370x`.
- MLP up is consistently slower: `1.169-1.293x`.
- MLP down is consistently slower: `1.320-2.013x`.
- The stable finding is that the current global Triton config helps Wo, but does not generalize to QKV or MLP projections.

## Implication

The current single tuned config, `BLOCK_K=128, BLOCK_N=32`, is not enough for all projection types.
The next useful step is per-projection tuning, starting from Wo as the positive control and MLP down as the hardest negative case.

Recommended next order:

1. tune Wo to confirm and stabilize the win
2. tune MLP down because it is the clearest loss
3. tune MLP up
4. tune QKV

If per-projection tuning still fails for MLP, move to CUDA C++ or focus on fusion around MLP projection outputs rather than GEMV replacement alone.
