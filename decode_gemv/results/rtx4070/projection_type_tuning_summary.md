# Projection Type Triton Tuning Summary

Source: `decode_gemv/results/rtx4070/projection_type_tuning/runs/*/summary.csv`

## Best Per Shape

| projection | shape | block_k | block_n | torch us | triton us | triton/torch | speedup | GB/s | TFLOP/s | max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_down | 1x8192x2048 | 256 | 16 | 55.296 | 49.792 | 0.900 | 1.111 | 674.303 | 0.674 | 0 |
| mlp_down | 1x8192x4096 | 64 | 32 | 125.040 | 158.720 | 1.269 | 0.788 | 422.968 | 0.423 | 0 |
| mlp_down | 1x11008x2048 | 256 | 16 | 79.872 | 120.816 | 1.513 | 0.661 | 373.418 | 0.373 | 0 |
| mlp_down | 1x11008x4096 | 64 | 32 | 175.104 | 209.920 | 1.199 | 0.834 | 429.724 | 0.430 | 0 |
| mlp_up | 1x2048x16384 | 256 | 64 | 126.816 | 154.464 | 1.218 | 0.821 | 434.701 | 0.434 | 0.00195312 |
| mlp_up | 1x2048x22016 | 128 | 64 | 175.040 | 203.040 | 1.160 | 0.862 | 444.374 | 0.444 | 1.52588e-05 |
| mlp_up | 1x4096x16384 | 128 | 128 | 266.064 | 294.416 | 1.107 | 0.904 | 456.017 | 0.456 | 0.000976562 |
| mlp_up | 1x4096x22016 | 128 | 128 | 361.968 | 391.168 | 1.081 | 0.925 | 461.202 | 0.461 | 0.00195312 |
| qkv | 1x2048x6144 | 256 | 32 | 44.656 | 32.768 | 0.734 | 1.363 | 768.500 | 0.768 | 0 |
| qkv | 1x4096x12288 | 128 | 32 | 196.608 | 224.256 | 1.141 | 0.877 | 449.023 | 0.449 | 0.000976562 |
| wo | 1x2048x2048 | 256 | 16 | 14.208 | 22.416 | 1.578 | 0.634 | 374.590 | 0.374 | 9.53674e-07 |
| wo | 1x4096x4096 | 256 | 32 | 56.032 | 41.872 | 0.747 | 1.338 | 801.749 | 0.801 | 0 |

Read:

- Triton best config wins 3 / 12 projection shapes in this tuning matrix.
- Wins are shape-specific; projection-specific tuning improves some shapes but is not a universal replacement for cuBLAS.
- Use this aggregate result as config input for projection block evaluation.
