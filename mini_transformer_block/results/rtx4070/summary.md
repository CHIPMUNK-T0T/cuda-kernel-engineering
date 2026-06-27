# Mini Transformer Block Benchmark Summary

| implementation | tokens | hidden | out features | dtype | device | latency us | max abs error | mean abs error | relative L2 error | max rel error |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| pytorch_unfused | 512 | 8192 | 8192 | float16 | cuda | 1445.888 | 0 | 0 | 0 | 0 |
| cuda_residual_fused | 512 | 8192 | 8192 | float16 | cuda | 1184.720 | 0.25 | 0.000125647 | 3.02643e-05 | 4.85655 |
| triton_residual_fused | 512 | 8192 | 8192 | float16 | cuda | 1154.560 | 0.25 | 0.000108863 | 2.82407e-05 | 4.85655 |
