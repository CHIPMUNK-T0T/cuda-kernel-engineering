# Decode Projection Fusion Copy/Add/Mul Benchmark

| implementation | tokens | features | dtype | latency us | est. GB/s | max abs error | vs torch clone+add+mul |
|---|---:|---:|---|---:|---:|---:|---:|
| torch_add_mul | 1 | 2048 | bfloat16 | 9.824 | 2.502 | 0 | 0.751 |
| torch_clone_add_mul | 1 | 2048 | bfloat16 | 13.088 | 2.504 | 0 | 1.000 |
| triton_add_mul | 1 | 2048 | bfloat16 | 21.504 | 0.762 | 0.00390625 | 1.643 |
| triton_copy_add_mul | 1 | 2048 | bfloat16 | 16.720 | 0.980 | 0.00390625 | 1.278 |
| torch_add_mul | 1 | 4096 | bfloat16 | 9.376 | 5.242 | 0 | 0.726 |
| torch_clone_add_mul | 1 | 4096 | bfloat16 | 12.912 | 5.076 | 0 | 1.000 |
| triton_add_mul | 1 | 4096 | bfloat16 | 15.088 | 2.172 | 0.00390625 | 1.169 |
| triton_copy_add_mul | 1 | 4096 | bfloat16 | 14.528 | 2.256 | 0.00390625 | 1.125 |
| torch_add_mul | 1 | 8192 | bfloat16 | 9.344 | 10.521 | 0 | 0.718 |
| torch_clone_add_mul | 1 | 8192 | bfloat16 | 13.008 | 10.076 | 0 | 1.000 |
| triton_add_mul | 1 | 8192 | bfloat16 | 14.848 | 4.414 | 0.0078125 | 1.141 |
| triton_copy_add_mul | 1 | 8192 | bfloat16 | 14.768 | 4.438 | 0.0078125 | 1.135 |
| torch_add_mul | 1 | 11008 | bfloat16 | 9.440 | 13.993 | 0 | 0.723 |
| torch_clone_add_mul | 1 | 11008 | bfloat16 | 13.056 | 13.490 | 0 | 1.000 |
| triton_add_mul | 1 | 11008 | bfloat16 | 15.360 | 5.733 | 0.0078125 | 1.176 |
| triton_copy_add_mul | 1 | 11008 | bfloat16 | 14.416 | 6.109 | 0.0078125 | 1.104 |
| torch_add_mul | 1 | 16384 | bfloat16 | 9.360 | 21.005 | 0 | 0.732 |
| torch_clone_add_mul | 1 | 16384 | bfloat16 | 12.784 | 20.506 | 0 | 1.000 |
| triton_add_mul | 1 | 16384 | bfloat16 | 14.720 | 8.904 | 0.0078125 | 1.151 |
| triton_copy_add_mul | 1 | 16384 | bfloat16 | 14.784 | 8.866 | 0.0078125 | 1.156 |
| torch_add_mul | 8 | 2048 | bfloat16 | 9.280 | 21.186 | 0 | 0.720 |
| torch_clone_add_mul | 8 | 2048 | bfloat16 | 12.896 | 20.328 | 0 | 1.000 |
| triton_add_mul | 8 | 2048 | bfloat16 | 14.432 | 9.082 | 0.0078125 | 1.119 |
| triton_copy_add_mul | 8 | 2048 | bfloat16 | 14.336 | 9.143 | 0.0078125 | 1.112 |
| torch_add_mul | 8 | 4096 | bfloat16 | 9.504 | 41.374 | 0 | 0.737 |
| torch_clone_add_mul | 8 | 4096 | bfloat16 | 12.896 | 40.655 | 0 | 1.000 |
| triton_add_mul | 8 | 4096 | bfloat16 | 15.152 | 17.301 | 0.0078125 | 1.175 |
| triton_copy_add_mul | 8 | 4096 | bfloat16 | 14.480 | 18.104 | 0.0078125 | 1.123 |
| torch_add_mul | 8 | 8192 | bfloat16 | 9.376 | 83.877 | 0 | 0.720 |
| torch_clone_add_mul | 8 | 8192 | bfloat16 | 13.024 | 80.511 | 0 | 1.000 |
| triton_add_mul | 8 | 8192 | bfloat16 | 14.944 | 35.084 | 0.0078125 | 1.147 |
| triton_copy_add_mul | 8 | 8192 | bfloat16 | 14.608 | 35.890 | 0.0078125 | 1.122 |
| torch_add_mul | 8 | 11008 | bfloat16 | 9.312 | 113.485 | 0 | 0.717 |
| torch_clone_add_mul | 8 | 11008 | bfloat16 | 12.992 | 108.453 | 0 | 1.000 |
| triton_add_mul | 8 | 11008 | bfloat16 | 15.360 | 45.867 | 0.0078125 | 1.182 |
| triton_copy_add_mul | 8 | 11008 | bfloat16 | 15.360 | 45.867 | 0.0078125 | 1.182 |
| torch_add_mul | 8 | 16384 | bfloat16 | 9.920 | 158.555 | 0 | 0.753 |
| torch_clone_add_mul | 8 | 16384 | bfloat16 | 13.168 | 159.261 | 0 | 1.000 |
| triton_add_mul | 8 | 16384 | bfloat16 | 15.248 | 68.768 | 0.0078125 | 1.158 |
| triton_copy_add_mul | 8 | 16384 | bfloat16 | 15.264 | 68.696 | 0.0078125 | 1.159 |
| torch_add_mul | 128 | 2048 | bfloat16 | 9.904 | 317.622 | 0 | 0.744 |
| torch_clone_add_mul | 128 | 2048 | bfloat16 | 13.312 | 315.077 | 0 | 1.000 |
| triton_add_mul | 128 | 2048 | bfloat16 | 15.360 | 136.533 | 0.0078125 | 1.154 |
| triton_copy_add_mul | 128 | 2048 | bfloat16 | 15.264 | 137.392 | 0.0078125 | 1.147 |
| torch_add_mul | 128 | 4096 | bfloat16 | 9.632 | 653.183 | 0 | 0.724 |
| torch_clone_add_mul | 128 | 4096 | bfloat16 | 13.296 | 630.912 | 0 | 1.000 |
| triton_add_mul | 128 | 4096 | bfloat16 | 15.200 | 275.941 | 0.0078125 | 1.143 |
| triton_copy_add_mul | 128 | 4096 | bfloat16 | 14.336 | 292.571 | 0.0078125 | 1.078 |
| torch_add_mul | 128 | 8192 | bfloat16 | 11.264 | 1117.091 | 0 | 0.733 |
| torch_clone_add_mul | 128 | 8192 | bfloat16 | 15.360 | 1092.267 | 0 | 1.000 |
| triton_add_mul | 128 | 8192 | bfloat16 | 15.360 | 546.133 | 0.0078125 | 1.000 |
| triton_copy_add_mul | 128 | 8192 | bfloat16 | 16.352 | 513.002 | 0.0078125 | 1.065 |
| torch_add_mul | 128 | 11008 | bfloat16 | 13.312 | 1270.154 | 0 | 0.712 |
| torch_clone_add_mul | 128 | 11008 | bfloat16 | 18.688 | 1206.356 | 0 | 1.000 |
| triton_add_mul | 128 | 11008 | bfloat16 | 17.408 | 647.529 | 0.0078125 | 0.932 |
| triton_copy_add_mul | 128 | 11008 | bfloat16 | 17.408 | 647.529 | 0.0078125 | 0.932 |
| torch_add_mul | 128 | 16384 | bfloat16 | 17.408 | 1445.647 | 0 | 0.719 |
| torch_clone_add_mul | 128 | 16384 | bfloat16 | 24.224 | 1385.173 | 0 | 1.000 |
| triton_add_mul | 128 | 16384 | bfloat16 | 20.480 | 819.200 | 0.0078125 | 0.845 |
| triton_copy_add_mul | 128 | 16384 | bfloat16 | 20.272 | 827.605 | 0.0078125 | 0.837 |

## Best By Shape

| tokens | features | best implementation | latency us |
|---:|---:|---|---:|
| 1 | 2048 | torch_add_mul | 9.824 |
| 1 | 4096 | torch_add_mul | 9.376 |
| 1 | 8192 | torch_add_mul | 9.344 |
| 1 | 11008 | torch_add_mul | 9.440 |
| 1 | 16384 | torch_add_mul | 9.360 |
| 8 | 2048 | torch_add_mul | 9.280 |
| 8 | 4096 | torch_add_mul | 9.504 |
| 8 | 8192 | torch_add_mul | 9.376 |
| 8 | 11008 | torch_add_mul | 9.312 |
| 8 | 16384 | torch_add_mul | 9.920 |
| 128 | 2048 | torch_add_mul | 9.904 |
| 128 | 4096 | torch_add_mul | 9.632 |
| 128 | 8192 | torch_add_mul | 11.264 |
| 128 | 11008 | torch_add_mul | 13.312 |
| 128 | 16384 | torch_add_mul | 17.408 |
