# Mini Decoder KV Benchmark Summary

| implementation | preset | context | hidden | layers | heads | kv heads | head dim | dtype | device | latency us | tokens/s | max abs error | relative L2 error |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| pytorch_unfused | qwen_like_2b | 2048 | 2048 | 16 | 16 | 4 | 128 | float16 | cuda | 2671.760 | 374.285 | 0 | 0 |
| cuda_residual_fused | qwen_like_2b | 2048 | 2048 | 16 | 16 | 4 | 128 | float16 | cuda | 2292.736 | 436.160 | 0 | 0 |
| triton_residual_fused | qwen_like_2b | 2048 | 2048 | 16 | 16 | 4 | 128 | float16 | cuda | 2064.736 | 484.323 | 0 | 0 |
