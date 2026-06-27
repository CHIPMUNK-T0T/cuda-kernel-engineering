# RMSNorm Analysis Notes

## CUDA optimized v1

今回の CUDA optimized RMSNorm は、naive kernel の実験骨格を保ったまま reduction の実装を変えた。

変更点:

- naive: shared memory に全 thread の partial sum を置き、`__syncthreads()` を挟みながら block 全体で二分 reduction する。
- optimized: warp 内は `__shfl_down_sync` で reduction し、warp ごとの合計だけを shared memory に置く。
- 最終合計は thread 0 が 8 個の warp sum を畳み、shared memory 経由で block 全体に broadcast する。

## 結果

default benchmark matrix では、CUDA optimized は全 20 shape で PyTorch eager より速く、CUDA naive / Triton を含めた比較では 17 shape で最速だった。

| metric | value |
|---|---:|
| CUDA optimized min speedup vs PyTorch eager | 5.11x |
| CUDA optimized median speedup vs PyTorch eager | 5.30x |
| CUDA optimized max speedup vs PyTorch eager | 12.20x |
| CUDA optimized median speedup vs CUDA naive | 1.14x |
| max abs error | 0.00390625 |

## 観察

- naive でも PyTorch eager より十分速い。RMSNorm を 1 kernel にまとめるだけで、複数 op / launch overhead を避けられる。
- optimized は shared memory reduction の同期回数を減らすことで、多くの shape で naive を上回った。
- 改善幅は小さい shape では限定的。kernel launch overhead と固定コストが支配的になりやすい。
- 大きい prefill 寄り shape では memory traffic が増えるため、reduction と memory access の差が latency に出やすい。

## 次に確認すること

- Nsight Compute で `tokens=1, hidden=4096` と `tokens=512, hidden=8192` を見る。
- achieved occupancy、registers per thread、shared memory usage、memory throughput、warp stall reason を確認する。
- vectorized load/store を追加した場合に、今回の warp reduction 版からさらに改善するかを見る。

## Nsight Compute 実行メモ

Nsight Compute 用に以下を追加した。

- `rmsnorm/benchmarks/profile_rmsnorm.py`
- `rmsnorm/scripts/run_nsight.sh`

実行例:

```bash
bash rmsnorm/scripts/run_nsight.sh cuda_naive 1 4096
bash rmsnorm/scripts/run_nsight.sh cuda_optimized 1 4096
bash rmsnorm/scripts/run_nsight.sh cuda_naive 512 8192
bash rmsnorm/scripts/run_nsight.sh cuda_optimized 512 8192
```

現在の環境では `ERR_NVGPUCTRPERM` により NVIDIA GPU performance counter を読めなかった。

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

このため、Nsight Compute の本計測は権限設定後に行う。Ubuntu では管理者権限で NVIDIA driver の profiling 制限を解除するか、root 権限で `ncu` を実行する必要がある。

追加確認:

```bash
cat /proc/driver/nvidia/params | rg -n "Profil|Restrict|Admin"
```

結果:

```text
RmProfilingAdminOnly: 1
```

`sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` は実行されたが、すでに NVIDIA module がロード済みのため、現在値には反映されていない。

## Nsight Compute 結果

`sudo bash rmsnorm/scripts/run_nsight.sh ...` で Nsight Compute の profile を取得した。今回のスクリプト修正前は warmup も profile 対象に入っていたため、以下は最後の launch ID を比較対象として読んでいる。

保存先:

- `rmsnorm/results/rtx4070/nsight/20260619-215036-cuda_naive-tokens1-hidden4096/profile.ncu-rep`
- `rmsnorm/results/rtx4070/nsight/20260619-215112-cuda_optimized-tokens1-hidden4096/profile.ncu-rep`
- `rmsnorm/results/rtx4070/nsight/20260619-215123-cuda_naive-tokens512-hidden8192/profile.ncu-rep`
- `rmsnorm/results/rtx4070/nsight/20260619-215134-cuda_optimized-tokens512-hidden8192/profile.ncu-rep`

### Decode 寄り: tokens=1, hidden=4096

| metric | cuda naive | cuda optimized |
|---|---:|---:|
| Nsight duration | 12.03 us | 4.26 us |
| Memory Workload Throughput | 2.20 GB/s | 11.82 GB/s |
| DRAM Throughput | 0.45% | 2.43% |
| L2 Hit Rate | 51.71% | 65.80% |
| Executed Instructions | 4,128 | 2,572 |
| Registers Per Thread | 16 | 27 |
| Dynamic Shared Memory Per Block | 1.02 KB | 0 B |
| Static Shared Memory Per Block | 0 B | 36 B |
| Achieved Occupancy | 16.67% | 15.82% |

観察:

- optimized は実行命令数が約 38% 減った。
- naive は 256 thread 分の partial sum を dynamic shared memory に置いて多段 reduction する。optimized は warp sum だけを shared memory に置くため、shared memory 使用量が大きく減った。
- grid size が 1 block なので GPU 全体はほぼ埋まらない。decode 寄りでは occupancy や throughput より、kernel 内の固定コスト削減が効いている。

### Prefill 寄り: tokens=512, hidden=8192

| metric | cuda naive | cuda optimized |
|---|---:|---:|
| Nsight duration | 49.73 us | 36.42 us |
| Memory Workload Throughput | 323.42 GB/s | 350.81 GB/s |
| DRAM Throughput | 65.88% | 71.44% |
| L2 Hit Rate | 66.72% | 67.92% |
| Executed Instructions | 3,751,936 | 2,238,464 |
| Registers Per Thread | 16 | 27 |
| Dynamic Shared Memory Per Block | 1.02 KB | 0 B |
| Static Shared Memory Per Block | 0 B | 36 B |
| Achieved Occupancy | 94.55% | 83.42% |

観察:

- optimized は実行命令数が約 40% 減った。
- registers per thread は 16 から 27 に増え、achieved occupancy は 94.55% から 83.42% に下がった。
- それでも duration は短く、DRAM throughput は 65.88% から 71.44% に上がった。occupancy を少し犠牲にしても、reduction の同期・命令数削減が効いた。
- この shape では launch overhead より memory traffic と reduction の実装差が見えやすい。

## ここまでの結論

速くなった理由:

- shared memory 全段 reduction をやめ、warp shuffle に寄せた。
- dynamic shared memory 使用量が 1.02 KB/block から 0 B になった。
- 実行命令数が decode / prefill の両方で大きく減った。
- prefill 寄りでは DRAM throughput も上がった。

遅かった試行:

- 最初の optimized 実装は block sum を全 thread に正しく broadcast できず、`max_abs_error=5993.93` で correctness NG だった。
- 次の修正版は correctness は通ったが、最終 reduction に余計な warp shuffle が残り `tokens=1, hidden=4096` で naive より遅かった。
- 最終的に、warp sum 8 個だけを thread 0 が単純に足す形にして速くなった。

## Fused Residual RMSNorm

対象演算:

```text
z = x + residual
y = z * rsqrt(mean(z^2) + eps) * weight
```

比較対象:

- PyTorch unfused: `add + rmsnorm`
- CUDA unfused: `torch add + cuda_optimized RMSNorm`
- CUDA fused: `add + rmsnorm` を 1 CUDA kernel
- Triton fused: `add + rmsnorm` を 1 Triton kernel

### Benchmark matrix の結果

default benchmark matrix では、CUDA fused が 19 / 20 shape で最速だった。Triton fused は `tokens=512, hidden=8192` の 1 shape だけ最速だった。

| metric | value |
|---|---:|
| CUDA fused fastest | 19 / 20 |
| Triton fused fastest | 1 / 20 |
| CUDA fused median speedup vs CUDA unfused | 1.401x |
| Triton fused median speedup vs CUDA fused | 0.500x |
| Triton fused max speedup vs CUDA fused | 1.083x |
| max abs error | 0.00390625 |

代表結果:

| shape | CUDA unfused | CUDA fused | Triton fused | fastest |
|---|---:|---:|---:|---|
| tokens=1, hidden=4096 | 10.016 us | 7.168 us | 14.192 us | CUDA fused |
| tokens=512, hidden=8192 | 28.976 us | 26.624 us | 24.576 us | Triton fused |

### Nsight Compute 結果

`NCU_SET=basic` で CUDA fused / Triton fused の代表 shape を取得した。

保存先:

- `rmsnorm/results/rtx4070/nsight/20260620-101115-triton_residual_fused-tokens1-hidden4096/profile.ncu-rep`
- `rmsnorm/results/rtx4070/nsight/20260620-101118-triton_residual_fused-tokens512-hidden8192/profile.ncu-rep`
- `rmsnorm/results/rtx4070/nsight/20260620-101120-cuda_residual_fused-tokens1-hidden4096/profile.ncu-rep`
- `rmsnorm/results/rtx4070/nsight/20260620-101150-cuda_residual_fused-tokens512-hidden8192/profile.ncu-rep`

| shape | metric | CUDA fused | Triton fused |
|---|---|---:|---:|
| tokens=1, hidden=4096 | Nsight kernel duration | 4.90 us | 2.75 us |
| tokens=1, hidden=4096 | DRAM Throughput | 1.72% | 2.25% |
| tokens=1, hidden=4096 | Registers / thread | 39 | 40 |
| tokens=1, hidden=4096 | Achieved Occupancy | 15.90% | 17.49% |
| tokens=512, hidden=8192 | Nsight kernel duration | 59.74 us | 53.12 us |
| tokens=512, hidden=8192 | DRAM Throughput | 88.41% | 88.87% |
| tokens=512, hidden=8192 | Registers / thread | 39 | 64 |
| tokens=512, hidden=8192 | Achieved Occupancy | 124.03% | 79.09% |

### 観察

- Nsight の duration は kernel 単体の時間で、benchmark harness の CUDA event latency とは一致しない。
- 小さい decode 寄り shape では、Nsight kernel duration だけなら Triton fused が短い。しかし benchmark latency では CUDA fused が速い。
- この差は、Triton の Python/runtime dispatch overhead が小さい shape で支配的になるためと考えるのが自然。
- 大きい prefill 寄り shape では dispatch overhead が相対的に小さくなり、kernel 本体の差が表に出る。この条件では benchmark / Nsight の両方で Triton fused が CUDA fused より速い。
- CUDA fused は `tokens=1` から `tokens=128` 付近まで安定して強く、decode 寄りの用途に向いている。
- Triton fused は最大 shape では強いが、小さい shape の end-to-end latency では不利だった。
