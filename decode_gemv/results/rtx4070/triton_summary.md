# Triton GEMV First Result

Run:

```text
decode_gemv/results/rtx4070/runs/20260620-162334-triton-gemv-tokens1/
```

## Result

`tokens=1` の全 shape で `triton_gemv` の correctness は通った。

現時点の単純 Triton kernel は、cuBLAS-backed `torch_linear` には勝っていない。
ただし、出力幅が大きい shape では差がかなり縮まる。

| shape | torch_linear us | triton_gemv us | triton / linear | triton TFLOP/s | note |
|---|---:|---:|---:|---:|---|
| t=1 in=2048 out=2048 | 14.320 | 28.960 | 2.02x | 0.290 | |
| t=1 in=2048 out=4096 | 20.480 | 35.744 | 1.75x | 0.469 | |
| t=1 in=2048 out=8192 | 33.168 | 49.152 | 1.48x | 0.683 | faster than torch_matmul |
| t=1 in=2048 out=11008 | 101.376 | 110.512 | 1.09x | 0.408 | |
| t=1 in=4096 out=2048 | 20.480 | 45.056 | 2.20x | 0.372 | |
| t=1 in=4096 out=4096 | 32.768 | 82.912 | 2.53x | 0.405 | |
| t=1 in=4096 out=8192 | 147.456 | 160.064 | 1.09x | 0.419 | |
| t=1 in=4096 out=11008 | 194.560 | 207.472 | 1.07x | 0.435 | |

## Initial Read

- `tokens=1` GEMV は compute を十分に埋めにくく、cuBLAS でも TFLOP/s は低い。
- 単純 Triton 版は output tile ごとに `x` を読み直すため、`x` reuse と scheduling の面で cuBLAS に不利。
- 小さい output width では launch / scheduling / tile overhead が目立ち、差が大きい。
- 大きい output width では weight streaming が支配的になり、単純 Triton でも `torch_linear` に近づく。
- `in=2048, out=8192` では `torch_matmul` よりは速く、custom path の余地は見える。

## Next

次は Nsight Compute で `torch_linear` と `triton_gemv` を同じ representative shape で見る。

優先 shape:

- `tokens=1, in=2048, out=8192`: Triton が `torch_matmul` に勝ち、`torch_linear` との差は 1.48x。
- `tokens=1, in=4096, out=11008`: `torch_linear` との差が 1.07x まで縮まる。

見る指標:

- kernel duration
- achieved occupancy
- DRAM throughput
- L2 hit rate
- global load efficiency
- warp stall reason
- registers per thread

その後、CUDA C++ naive / optimized で `x` reuse、vectorized load、warp-level reduction を詰める。
