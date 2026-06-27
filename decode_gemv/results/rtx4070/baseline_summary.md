# Decode GEMV PyTorch/cuBLAS Baseline

## Conditions

- GPU: RTX 4070
- dtype: `bfloat16`
- implementations: `torch_matmul`, `torch_linear`
- rows: `96`
- record: `decode_gemv/results/rtx4070/runs/20260620-160838`

## Summary

`tokens=1` の decode shape では、`torch_linear` が全 shape で最速だった。

| metric | value |
|---|---:|
| shapes | 48 |
| `torch_linear` wins | 31 |
| `torch_matmul` wins | 17 |
| `tokens=1` `torch_linear` wins | 8 / 8 |

Median latency by tokens:

| tokens | torch_matmul us | torch_linear us |
|---:|---:|---:|
| 1 | 78.800 | 33.264 |
| 2 | 58.880 | 36.696 |
| 4 | 53.088 | 34.304 |
| 8 | 51.712 | 36.376 |
| 32 | 53.656 | 48.056 |
| 128 | 102.880 | 103.512 |

Best TFLOP/s by tokens:

| tokens | implementation | in | out | TFLOP/s | latency us |
|---:|---|---:|---:|---:|---:|
| 1 | torch_linear | 4096 | 4096 | 1.024 | 32.768 |
| 2 | torch_linear | 4096 | 4096 | 1.928 | 34.816 |
| 4 | torch_linear | 4096 | 4096 | 4.096 | 32.768 |
| 8 | torch_linear | 4096 | 4096 | 7.746 | 34.656 |
| 32 | torch_linear | 2048 | 8192 | 26.214 | 40.960 |
| 128 | torch_matmul | 4096 | 11008 | 55.222 | 209.024 |

## Read

- `tokens=1` は arithmetic intensity が低く、latency は tens of microseconds に留まる。
- `torch_linear` は decode-like shape で `torch_matmul` より安定して強い。
- tokens が増えると compute utilization が上がり、`tokens=32/128` では TFLOP/s が大きく伸びる。
- custom kernel の最初の比較対象は、`tokens=1` の `torch_linear` path にするのが自然。
- cuBLAS-backed baseline が強いため、次の Triton / CUDA 実装は「勝つ」だけでなく「どの shape で近づくか」を見る。

