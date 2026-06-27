# Decode GEMV Benchmark Summary

| implementation | tokens | in features | out features | dtype | device | latency us | effective GB/s | effective TFLOP/s | max abs error | relative L2 error |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| torch_matmul | 1 | 2048 | 2048 | bfloat16 | cuda | 13.552 | 619.599 | 0.619 | 0 | 0 |
| torch_linear | 1 | 2048 | 2048 | bfloat16 | cuda | 14.320 | 586.369 | 0.586 | 0 | 0 |
| triton_gemv | 1 | 2048 | 2048 | bfloat16 | cuda | 28.960 | 289.945 | 0.290 | 9.53674e-07 | 2.09394e-08 |
| torch_matmul | 1 | 2048 | 4096 | bfloat16 | cuda | 23.536 | 713.354 | 0.713 | 0 | 0 |
| torch_linear | 1 | 2048 | 4096 | bfloat16 | cuda | 20.480 | 819.800 | 0.819 | 0.00195312 | 3.11324e-05 |
| triton_gemv | 1 | 2048 | 4096 | bfloat16 | cuda | 35.744 | 469.715 | 0.469 | 9.53674e-07 | 1.52014e-08 |
| torch_matmul | 1 | 2048 | 8192 | bfloat16 | cuda | 77.856 | 431.244 | 0.431 | 0 | 0 |
| torch_linear | 1 | 2048 | 8192 | bfloat16 | cuda | 33.168 | 1012.268 | 1.012 | 0.000488281 | 7.65512e-06 |
| triton_gemv | 1 | 2048 | 8192 | bfloat16 | cuda | 49.152 | 683.083 | 0.683 | 0.000488281 | 7.65512e-06 |
| torch_matmul | 1 | 2048 | 11008 | bfloat16 | cuda | 103.424 | 436.213 | 0.436 | 0 | 0 |
| torch_linear | 1 | 2048 | 11008 | bfloat16 | cuda | 101.376 | 445.025 | 0.445 | 1.52588e-05 | 1.46e-07 |
| triton_gemv | 1 | 2048 | 11008 | bfloat16 | cuda | 110.512 | 408.235 | 0.408 | 1.52588e-05 | 1.46e-07 |
| torch_matmul | 1 | 4096 | 2048 | bfloat16 | cuda | 20.544 | 817.246 | 0.817 | 0 | 0 |
| torch_linear | 1 | 4096 | 2048 | bfloat16 | cuda | 20.480 | 819.800 | 0.819 | 0 | 0 |
| triton_gemv | 1 | 4096 | 2048 | bfloat16 | cuda | 45.056 | 372.636 | 0.372 | 0 | 0 |
| torch_matmul | 1 | 4096 | 4096 | bfloat16 | cuda | 79.632 | 421.574 | 0.421 | 0 | 0 |
| torch_linear | 1 | 4096 | 4096 | bfloat16 | cuda | 32.768 | 1024.500 | 1.024 | 3.8147e-06 | 6.0902e-08 |
| triton_gemv | 1 | 4096 | 4096 | bfloat16 | cuda | 82.912 | 404.897 | 0.405 | 3.8147e-06 | 6.0902e-08 |
| torch_matmul | 1 | 4096 | 8192 | bfloat16 | cuda | 151.264 | 443.816 | 0.444 | 0 | 0 |
| torch_linear | 1 | 4096 | 8192 | bfloat16 | cuda | 147.456 | 455.278 | 0.455 | 0 | 0 |
| triton_gemv | 1 | 4096 | 8192 | bfloat16 | cuda | 160.064 | 419.416 | 0.419 | 1.52588e-05 | 2.41031e-07 |
| torch_matmul | 1 | 4096 | 11008 | bfloat16 | cuda | 202.752 | 444.917 | 0.445 | 0 | 0 |
| torch_linear | 1 | 4096 | 11008 | bfloat16 | cuda | 194.560 | 463.650 | 0.463 | 7.45058e-08 | 7.14454e-10 |
| triton_gemv | 1 | 4096 | 11008 | bfloat16 | cuda | 207.472 | 434.795 | 0.435 | 7.45058e-08 | 7.14454e-10 |
