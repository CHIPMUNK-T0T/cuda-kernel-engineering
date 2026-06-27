# Mini LLM Decode Benchmark Summary

| implementation | tokens | hidden | layers | projection weights | dtype | device | latency us | tokens/s | max abs error | mean abs error | relative L2 error | max rel error |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| pytorch_unfused | 1 | 4096 | 8 | distinct | float16 | cuda | 788.528 | 1268.186 | 0 | 0 | 0 | 0 |
| cuda_residual_fused | 1 | 4096 | 8 | distinct | float16 | cuda | 633.856 | 1577.645 | 0 | 0 | 0 | 0 |
| triton_residual_fused | 1 | 4096 | 8 | distinct | float16 | cuda | 632.528 | 1580.958 | 0 | 0 | 0 | 0 |
