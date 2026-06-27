# RMSNorm Benchmark Log

この記事・README で使う計測結果の一次記録をここにまとめる。

## 2026-06-20 - Triton fused residual RMSNorm Nsight attempt

目的:

- Triton fused residual RMSNorm を Nsight Compute で確認する。
- `tokens=1, hidden=4096` と `tokens=512, hidden=8192` を CUDA fused と比較する準備をする。

追加・変更したもの:

- `profile_rmsnorm.py` に `cuda_residual_fused` / `triton_residual_fused` を追加。
- `run_nsight.sh` に `cuda_residual_fused` / `triton_residual_fused` を追加。
- `run_nsight.sh` で `NCU_SET` / `NCU_TARGET_PROCESSES` / `NCU_LAUNCH_SKIP` を環境変数から指定できるようにした。

実行したコマンド:

```bash
env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh triton_residual_fused 1 4096 1
```

保存先:

- `rmsnorm/results/rtx4070/nsight/20260620-100558-triton_residual_fused-tokens1-hidden4096/`

結果:

- Python runner は実行され、`max_abs_error=0` を確認した。
- Nsight Compute は `ERR_NVGPUCTRPERM` で performance counter を読めず終了した。
- この Codex セッションからは `sudo -n true` が `sudo: a password is required` になり、sudo 実行できなかった。

ユーザー側で実行するコマンド:

```bash
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh triton_residual_fused 1 4096 1
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh triton_residual_fused 512 8192 1
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh cuda_residual_fused 1 4096 1
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh cuda_residual_fused 512 8192 1
```

メモ:

- Triton では `--set full` が非常に重かったため、まず `NCU_SET=basic` で取得する。
- `NCU_TARGET_PROCESSES=application-only` は Triton JIT の child process 追跡を避けるために使う。
- `NCU_LAUNCH_SKIP=0` は profile runner 側で warmup 後の NVTX range 内を実行しているため、まず確実に対象 kernel を拾うために使う。

## 2026-06-20 - Fused residual RMSNorm Nsight results

目的:

- CUDA fused と Triton fused の kernel 単体 metrics を比較する。
- benchmark で CUDA fused が小さい shape に強く、Triton fused が最大 shape で勝った理由を説明する材料を作る。

保存先:

- `rmsnorm/results/rtx4070/nsight/20260620-101115-triton_residual_fused-tokens1-hidden4096/`
- `rmsnorm/results/rtx4070/nsight/20260620-101118-triton_residual_fused-tokens512-hidden8192/`
- `rmsnorm/results/rtx4070/nsight/20260620-101120-cuda_residual_fused-tokens1-hidden4096/`
- `rmsnorm/results/rtx4070/nsight/20260620-101150-cuda_residual_fused-tokens512-hidden8192/`

要約:

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

メモ:

- Nsight の duration は kernel 単体で、benchmark harness の CUDA event latency とは一致しない。
- benchmark では `tokens=1, hidden=4096` で CUDA fused `7.168 us`、Triton fused `14.192 us` だった。
- 一方 Nsight の kernel 単体では Triton fused の duration が短い。小さい shape では Triton の Python/runtime dispatch overhead が支配的になっている可能性が高い。
- `tokens=512, hidden=8192` では benchmark でも Triton fused が勝ち、Nsight kernel duration でも Triton fused が短い。
- 大きい shape では dispatch overhead が相対的に小さくなり、kernel 本体の効率が latency に出やすい。

## 2026-06-20 - Triton fused residual RMSNorm default matrix

目的:

- Triton fused residual RMSNorm を追加し、CUDA fused と比較する。
- 高水準 kernel 実装で、CUDA C++ fused にどこまで近づけるかを見る。
- fused residual RMSNorm の次の分析対象 shape を決める。

追加したもの:

- `rmsnorm/kernels/rmsnorm_triton/fused_residual_rmsnorm.py`
- Python wrapper `triton_fused_residual_rmsnorm(x, residual, weight, eps)`
- benchmark implementation `triton_residual_fused`

実行したコマンド:

```bash
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --tokens 1 --hidden 4096 --runs 10 --warmup 3 --implementations pytorch_residual_unfused,cuda_residual_fused,triton_residual_fused --run-name smoke-triton-residual-fused
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --tokens 512 --hidden 8192 --runs 10 --warmup 3 --implementations pytorch_residual_unfused,cuda_residual_fused,triton_residual_fused --run-name smoke-triton-residual-fused-prefill
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --runs 100 --warmup 20 --implementations pytorch_residual_unfused,cuda_residual_unfused,cuda_residual_fused,triton_residual_fused --run-name residual-fused-cuda-triton-matrix
```

保存先:

- `rmsnorm/results/rtx4070/runs/20260620-095248-smoke-triton-residual-fused/`
- `rmsnorm/results/rtx4070/runs/20260620-095257-smoke-triton-residual-fused-prefill/`
- `rmsnorm/results/rtx4070/runs/20260620-095305-residual-fused-cuda-triton-matrix/`

要約:

| metric | value |
|---|---:|
| compared shapes | 20 |
| CUDA fused fastest | 19 |
| Triton fused fastest | 1 |
| CUDA unfused fastest | 0 |
| CUDA fused median speedup vs CUDA unfused | 1.401x |
| Triton fused median speedup vs CUDA unfused | 0.705x |
| Triton fused median speedup vs CUDA fused | 0.500x |
| Triton fused max speedup vs CUDA fused | 1.083x |
| Triton fused max abs error | 0.00390625 |

代表結果:

| tokens | hidden | CUDA unfused us | CUDA fused us | Triton fused us | fastest |
|---:|---:|---:|---:|---:|---|
| 1 | 4096 | 10.016 | 7.168 | 14.192 | CUDA fused |
| 128 | 8192 | 12.288 | 10.240 | 15.360 | CUDA fused |
| 512 | 4096 | 16.576 | 13.376 | 18.432 | CUDA fused |
| 512 | 8192 | 28.976 | 26.624 | 24.576 | Triton fused |

メモ:

- CUDA fused は 19 / 20 shape で最速だった。
- Triton fused は decode 寄りでは CUDA fused の約半分程度の速度に留まる。
- `tokens=512, hidden=8192` では Triton fused が CUDA fused を 1.083x 上回った。
- 次は Nsight Compute で `tokens=1, hidden=4096` と `tokens=512, hidden=8192` を見ると、Triton が小さい shape で遅い理由と最大 shape で勝つ理由を説明しやすい。

## 2026-06-20 - CUDA fused residual RMSNorm default matrix

目的:

- CUDA fused residual RMSNorm が default benchmark matrix 全体で効くか確認する。
- PyTorch unfused / CUDA unfused / CUDA fused を同じ条件で比較する。
- Triton fused 実装に進む前に、CUDA fused の勝ち負けと弱い条件を把握する。

実行したコマンド:

```bash
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --runs 100 --warmup 20 --implementations pytorch_residual_unfused,cuda_residual_unfused,cuda_residual_fused --run-name residual-cuda-fused-matrix
```

保存先:

- `rmsnorm/results/rtx4070/runs/20260620-094959-residual-cuda-fused-matrix/`

要約:

| metric | value |
|---|---:|
| compared shapes | 20 |
| CUDA fused wins vs CUDA unfused | 20 |
| CUDA fused min speedup vs CUDA unfused | 1.069x |
| CUDA fused median speedup vs CUDA unfused | 1.419x |
| CUDA fused max speedup vs CUDA unfused | 1.460x |
| CUDA fused min speedup vs PyTorch unfused | 5.196x |
| CUDA fused median speedup vs PyTorch unfused | 5.714x |
| CUDA fused max speedup vs PyTorch unfused | 8.623x |
| CUDA fused max abs error | 0.00390625 |

代表結果:

| tokens | hidden | PyTorch unfused us | CUDA unfused us | CUDA fused us | fused speedup vs CUDA unfused |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 40.736 | 10.304 | 7.840 | 1.314x |
| 1 | 8192 | 41.984 | 10.256 | 7.296 | 1.406x |
| 128 | 8192 | 57.216 | 12.288 | 10.240 | 1.200x |
| 512 | 4096 | 91.744 | 18.432 | 15.360 | 1.200x |
| 512 | 8192 | 256.064 | 31.744 | 29.696 | 1.069x |

メモ:

- CUDA fused は 20 / 20 shape で CUDA unfused より速かった。
- decode 寄り、小さい token 数では launch / temporary 削減の効果が見えやすい。
- prefill 寄りの大きい shape では差が縮み、`tokens=512, hidden=8192` では 1.069x に留まった。
- 現在の fused kernel は reduction pass と output pass で `x` / `residual` を 2 回読むため、大きい shape では memory traffic 削減が限定的になる可能性がある。
- 次は Triton fused を追加して、高水準 kernel でも同じ傾向になるか比較する。その後、必要なら Nsight Compute で `tokens=1, hidden=4096` と `tokens=512, hidden=8192` を見る。

## 2026-06-19 - CUDA fused residual RMSNorm smoke test

目的:

- `z = x + residual` と RMSNorm を 1 CUDA kernel に fuse する。
- PyTorch unfused / CUDA unfused / CUDA fused を同じ benchmark harness で比較できるようにする。
- decode 寄り shape と prefill 寄り shape の両方で correctness と latency を確認する。

追加したもの:

- `rmsnorm/kernels/rmsnorm_cuda/fused_residual_rmsnorm.cu`
- `binding.cpp` の `forward_fused_residual`
- Python wrapper `fused_residual_rmsnorm(x, residual, weight, eps)`
- benchmark implementation `cuda_residual_fused`

実行したコマンド:

```bash
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --tokens 1 --hidden 4096 --runs 10 --warmup 3 --run-name smoke-cuda-residual-fused-v2
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --tokens 512 --hidden 8192 --runs 10 --warmup 3 --run-name smoke-cuda-residual-fused-prefill
```

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-221136-smoke-cuda-residual-fused-v2/`
- `rmsnorm/results/rtx4070/runs/20260619-221148-smoke-cuda-residual-fused-prefill/`

結果:

| shape | implementation | latency us | effective GB/s | max abs error |
|---|---|---:|---:|---:|
| tokens=1, hidden=4096 | pytorch_residual_unfused | 41.616 | 1.378 | 0 |
| tokens=1, hidden=4096 | cuda_residual_unfused | 10.304 | 5.565 | 0 |
| tokens=1, hidden=4096 | cuda_residual_fused | 7.280 | 6.752 | 0 |
| tokens=512, hidden=8192 | pytorch_residual_unfused | 254.528 | 230.703 | 0 |
| tokens=512, hidden=8192 | cuda_residual_unfused | 41.264 | 1423.038 | 0.00390625 |
| tokens=512, hidden=8192 | cuda_residual_fused | 38.096 | 1321.179 | 0.00390625 |

メモ:

- 初回実装では `x + residual` を FP32 のまま扱ったため、PyTorch unfused と丸め位置が違い `max_abs_error=0.00390625` になった。
- 比較基準を揃えるため、kernel 内で `z` を output dtype に一度丸めてから reduction / output に使う形へ修正した。
- decode 寄りでは CUDA unfused `10.304 us` に対して CUDA fused `7.280 us`。
- prefill 寄りでは CUDA unfused `41.264 us` に対して CUDA fused `38.096 us`。
- fused kernel は `z` 一時 tensor を作らないが、RMSNorm の reduction pass と output pass で `x` / `residual` を 2 回読む実装になっている。

## 2026-06-19 - Residual RMSNorm unfused smoke test

目的:

- Fused Residual RMSNorm に進む前に、unfused baseline を benchmark harness に追加する。
- `z = x + residual` の add と RMSNorm を別々に実行する比較基準を作る。
- PyTorch unfused と CUDA unfused の correctness / latency を同じ出力形式で記録する。

追加した implementation:

- `pytorch_residual_unfused`: PyTorch の `add + rmsnorm`
- `cuda_residual_unfused`: PyTorch の `add + cuda_optimized RMSNorm`

実行したコマンド:

```bash
bash rmsnorm/scripts/run_bench.sh --operation residual_rmsnorm --tokens 1 --hidden 4096 --runs 5 --warmup 2 --run-name smoke-residual-unfused
```

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-220655-smoke-residual-unfused/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-220655-smoke-residual-unfused/metadata.json`
- `rmsnorm/results/rtx4070/runs/20260619-220655-smoke-residual-unfused/summary.csv`
- `rmsnorm/results/rtx4070/runs/20260619-220655-smoke-residual-unfused/summary.md`
- `rmsnorm/results/rtx4070/runs/20260619-220655-smoke-residual-unfused/console.txt`

結果:

| operation | implementation | tokens | hidden | latency us | effective GB/s | max abs error |
|---|---|---:|---:|---:|---:|---:|
| residual_rmsnorm | pytorch_residual_unfused | 1 | 4096 | 42.208 | 1.165 | 0 |
| residual_rmsnorm | cuda_residual_unfused | 1 | 4096 | 11.488 | 4.279 | 0 |

メモ:

- `bench_rmsnorm.py` に `--operation rmsnorm|residual_rmsnorm` を追加した。
- 既存の RMSNorm 単体 benchmark はデフォルト `--operation rmsnorm` のまま維持した。
- `residual_rmsnorm` の default implementations は `pytorch_residual_unfused,cuda_residual_unfused` にした。
- effective bandwidth は unfused の概算として `x read + residual read + z write + two z reads + weight read + output write` を使う。
- 次は CUDA fused residual RMSNorm kernel を追加し、同じ harness に接続する。

## 2026-06-19 - Nsight Compute setup attempt

目的:

- CUDA naive / CUDA optimized を Nsight Compute で比較する準備をする。
- `tokens=1, hidden=4096` と `tokens=512, hidden=8192` の代表 shape を計測できる導線を作る。

追加したもの:

- `rmsnorm/benchmarks/profile_rmsnorm.py`
- `rmsnorm/scripts/run_nsight.sh`

実行したコマンド:

```bash
bash rmsnorm/scripts/run_nsight.sh cuda_naive 1 4096
```

保存先:

- `rmsnorm/results/rtx4070/nsight/20260619-213535-cuda_naive-tokens1-hidden4096/metadata.md`
- `rmsnorm/results/rtx4070/nsight/20260619-213535-cuda_naive-tokens1-hidden4096/ncu.log`
- `rmsnorm/results/rtx4070/nsight/20260619-213535-cuda_naive-tokens1-hidden4096/console.txt`

結果:

- Python runner は実行でき、`max_abs_error=0` を確認した。
- Nsight Compute は `ERR_NVGPUCTRPERM` で performance counter を読めず終了した。
- `sudo -n true` は `sudo: a password is required` で、このセッションから権限変更はできなかった。
- ユーザーが `sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` 実行後も、`/proc/driver/nvidia/params` は `RmProfilingAdminOnly: 1` のままだった。
- 新しい 4 実行でも Python runner は動いたが、Nsight Compute は同じ `ERR_NVGPUCTRPERM` で `.ncu-rep` を生成できなかった。

次の対応:

- ユーザー側で NVIDIA performance counter 権限を有効化し、`RmProfilingAdminOnly: 0` になったことを確認する。
- 権限設定後、以下を実行する。

```bash
bash rmsnorm/scripts/run_nsight.sh cuda_naive 1 4096
bash rmsnorm/scripts/run_nsight.sh cuda_optimized 1 4096
bash rmsnorm/scripts/run_nsight.sh cuda_naive 512 8192
bash rmsnorm/scripts/run_nsight.sh cuda_optimized 512 8192
```

## 2026-06-19 - Nsight Compute profile results

目的:

- CUDA naive と CUDA optimized の差を Nsight Compute で確認する。
- 何を変えて速くなったかを、命令数、shared memory、throughput の観点で説明できるようにする。

実行したコマンド:

```bash
sudo bash rmsnorm/scripts/run_nsight.sh cuda_naive 1 4096
sudo bash rmsnorm/scripts/run_nsight.sh cuda_optimized 1 4096
sudo bash rmsnorm/scripts/run_nsight.sh cuda_naive 512 8192
sudo bash rmsnorm/scripts/run_nsight.sh cuda_optimized 512 8192
```

保存先:

- `rmsnorm/results/rtx4070/nsight/20260619-215036-cuda_naive-tokens1-hidden4096/`
- `rmsnorm/results/rtx4070/nsight/20260619-215112-cuda_optimized-tokens1-hidden4096/`
- `rmsnorm/results/rtx4070/nsight/20260619-215123-cuda_naive-tokens512-hidden8192/`
- `rmsnorm/results/rtx4070/nsight/20260619-215134-cuda_optimized-tokens512-hidden8192/`

注意:

- この時点の `run_nsight.sh` は warmup launch も profile 対象にしていた。
- 比較には最後の launch ID を使った。
- 以後のため、`run_nsight.sh` に `--launch-skip 10` と `--launch-count "$iters"` を追加した。

要約:

| shape | metric | cuda naive | cuda optimized |
|---|---|---:|---:|
| tokens=1, hidden=4096 | Nsight duration | 12.03 us | 4.26 us |
| tokens=1, hidden=4096 | Executed Instructions | 4,128 | 2,572 |
| tokens=1, hidden=4096 | Dynamic Shared Memory / Block | 1.02 KB | 0 B |
| tokens=512, hidden=8192 | Nsight duration | 49.73 us | 36.42 us |
| tokens=512, hidden=8192 | DRAM Throughput | 65.88% | 71.44% |
| tokens=512, hidden=8192 | Executed Instructions | 3,751,936 | 2,238,464 |
| tokens=512, hidden=8192 | Dynamic Shared Memory / Block | 1.02 KB | 0 B |

メモ:

- CUDA optimized は shared memory の全段 reduction をやめ、warp shuffle + warp sum 集約に変えたことで命令数が減った。
- prefill 寄りでは achieved occupancy は下がったが、DRAM throughput と duration は改善した。
- 次の最適化候補は vectorized load/store。

## 2026-06-19 - CUDA optimized RMSNorm smoke test

目的:

- benchmark harness から `cuda_optimized` を選択できることを確認する。
- PyTorch eager / CUDA naive / Triton RMSNorm / CUDA optimized を同じ shape で比較する。
- optimized kernel の correctness と latency を確認する。

実行条件:

- GPU: NVIDIA GeForce RTX 4070
- CUDA capability: 8.9
- PyTorch: 2.11.0+cu128
- torch CUDA: 12.8
- dtype: FP16 input / FP32 accumulate / FP16 output
- shape: `tokens=1, hidden=4096`
- warmup: 5
- runs: 20

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-213019-smoke-cuda-optimized-v2/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-213019-smoke-cuda-optimized-v2/metadata.json`
- `rmsnorm/results/rtx4070/runs/20260619-213019-smoke-cuda-optimized-v2/summary.csv`
- `rmsnorm/results/rtx4070/runs/20260619-213019-smoke-cuda-optimized-v2/summary.md`
- `rmsnorm/results/rtx4070/runs/20260619-213019-smoke-cuda-optimized-v2/console.txt`

結果:

| implementation | latency us | effective GB/s | max abs error |
|---|---:|---:|---:|
| pytorch_eager | 37.568 | 0.654 | 0 |
| cuda_naive | 8.176 | 3.006 | 0 |
| triton_rmsnorm | 15.152 | 1.622 | 0 |
| cuda_optimized | 7.184 | 3.421 | 0 |

メモ:

- `cuda_optimized` は smoke test で PyTorch reference と一致した。
- この shape では CUDA naive より少し速い。
- optimized kernel は shared memory の全段 reduction ではなく、warp shuffle で warp 内を畳み、warp ごとの合計だけを shared memory 経由で集約する。

## 2026-06-19 - CUDA optimized RMSNorm baseline matrix

目的:

- PyTorch eager / CUDA naive / Triton RMSNorm / CUDA optimized を default benchmark matrix で比較する。
- CUDA optimized がどの shape で効くか、CUDA naive と比べて改善する条件を見る。
- 次の Nsight Compute 分析に使う代表 shape を選ぶ材料にする。

実行条件:

- GPU: NVIDIA GeForce RTX 4070
- CUDA capability: 8.9
- PyTorch: 2.11.0+cu128
- torch CUDA: 12.8
- dtype: FP16 input / FP32 accumulate / FP16 output
- hidden size: `2048 / 3072 / 4096 / 8192`
- num_tokens: `1 / 8 / 32 / 128 / 512`
- warmup: 20
- runs: 100

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-213034-baseline-matrix-cuda-optimized/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-213034-baseline-matrix-cuda-optimized/metadata.json`
- `rmsnorm/results/rtx4070/runs/20260619-213034-baseline-matrix-cuda-optimized/summary.csv`
- `rmsnorm/results/rtx4070/runs/20260619-213034-baseline-matrix-cuda-optimized/summary.md`
- `rmsnorm/results/rtx4070/runs/20260619-213034-baseline-matrix-cuda-optimized/console.txt`

要約:

| metric | value |
|---|---:|
| compared shapes | 20 |
| CUDA optimized min speedup vs PyTorch eager | 5.11x |
| CUDA optimized median speedup vs PyTorch eager | 5.30x |
| CUDA optimized max speedup vs PyTorch eager | 12.20x |
| CUDA optimized median speedup vs CUDA naive | 1.14x |
| CUDA optimized wins vs CUDA naive / Triton | 17 shapes |
| CUDA naive wins vs CUDA optimized / Triton | 3 shapes |
| Triton wins vs CUDA naive / CUDA optimized | 0 shapes |
| max abs error | 0.00390625 |

代表結果:

| tokens | hidden | PyTorch eager us | CUDA naive us | Triton us | CUDA optimized us | speedup vs PyTorch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 38.272 | 7.600 | 14.336 | 7.264 | 5.27x |
| 128 | 8192 | 53.184 | 12.624 | 14.336 | 7.984 | 6.66x |
| 512 | 4096 | 84.960 | 16.384 | 15.936 | 10.624 | 8.00x |
| 512 | 8192 | 212.992 | 25.648 | 20.480 | 17.456 | 12.20x |

メモ:

- CUDA optimized は 17 / 20 shape で最速だった。
- 小さい shape でも launch overhead の下限に近いが、hidden が大きい条件では naive より差が出た。
- `tokens=512, hidden=8192` は effective bandwidth が `1441.672 GB/s` まで出た。次の Nsight Compute ではこの shape と decode 寄りの `tokens=1, hidden=4096` を代表候補にする。

## 2026-06-19 - Triton RMSNorm smoke test

目的:

- benchmark harness から `triton_rmsnorm` を選択できることを確認する。
- PyTorch eager / CUDA naive / Triton RMSNorm の 3 実装を同じ shape で比較する。
- Triton JIT compile 後の correctness と latency を確認する。

実行条件:

- GPU: NVIDIA GeForce RTX 4070
- CUDA capability: 8.9
- PyTorch: 2.11.0+cu128
- torch CUDA: 12.8
- dtype: FP16 input / FP32 accumulate / FP16 output
- shape: `tokens=1, hidden=4096`
- warmup: 5
- runs: 20

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-212113-smoke-triton-rmsnorm/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-212113-smoke-triton-rmsnorm/metadata.json`
- `rmsnorm/results/rtx4070/runs/20260619-212113-smoke-triton-rmsnorm/summary.csv`
- `rmsnorm/results/rtx4070/runs/20260619-212113-smoke-triton-rmsnorm/summary.md`
- `rmsnorm/results/rtx4070/runs/20260619-212113-smoke-triton-rmsnorm/console.txt`

結果:

| implementation | latency us | effective GB/s | max abs error |
|---|---:|---:|---:|
| pytorch_eager | 39.664 | 0.620 | 0 |
| cuda_naive | 23.648 | 1.039 | 0 |
| triton_rmsnorm | 14.736 | 1.668 | 0 |

メモ:

- `triton_rmsnorm` は smoke test では PyTorch reference と一致した。
- この shape では Triton が CUDA naive より速い。
- 次は default benchmark matrix で token / hidden ごとの傾向を見る。

## 2026-06-19 - Triton RMSNorm baseline matrix

目的:

- PyTorch eager / CUDA naive / Triton RMSNorm を default benchmark matrix で比較する。
- Triton がどの shape で効くか、CUDA naive と比べてどの条件で強いかを見る。
- CUDA optimized RMSNorm に進む前の比較基準を作る。

実行条件:

- GPU: NVIDIA GeForce RTX 4070
- CUDA capability: 8.9
- PyTorch: 2.11.0+cu128
- torch CUDA: 12.8
- dtype: FP16 input / FP32 accumulate / FP16 output
- hidden size: `2048 / 3072 / 4096 / 8192`
- num_tokens: `1 / 8 / 32 / 128 / 512`
- warmup: 20
- runs: 100

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-212317-baseline-matrix-triton-rmsnorm/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-212317-baseline-matrix-triton-rmsnorm/metadata.json`
- `rmsnorm/results/rtx4070/runs/20260619-212317-baseline-matrix-triton-rmsnorm/summary.csv`
- `rmsnorm/results/rtx4070/runs/20260619-212317-baseline-matrix-triton-rmsnorm/summary.md`
- `rmsnorm/results/rtx4070/runs/20260619-212317-baseline-matrix-triton-rmsnorm/console.txt`

要約:

| metric | value |
|---|---:|
| compared shapes | 20 |
| Triton min speedup vs PyTorch eager | 2.67x |
| Triton median speedup vs PyTorch eager | 2.88x |
| Triton max speedup vs PyTorch eager | 10.67x |
| CUDA naive wins vs Triton | 18 shapes |
| Triton wins vs CUDA naive | 2 shapes |
| max abs error | 0.00390625 |

代表結果:

| tokens | hidden | PyTorch eager us | CUDA naive us | Triton us | Triton speedup vs PyTorch |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 38.624 | 8.192 | 13.360 | 2.89x |
| 512 | 4096 | 76.368 | 14.896 | 14.576 | 5.24x |
| 512 | 8192 | 207.504 | 24.384 | 19.456 | 10.67x |

メモ:

- Triton は全 shape で PyTorch eager より速い。
- CUDA naive は 18 / 20 shape で Triton より速い。現時点の Triton 実装は 1 program = 1 row の素直な実装なので、小さい shape では launch / program overhead の影響が見える。
- Triton は `tokens=512, hidden=4096` と `tokens=512, hidden=8192` で CUDA naive を上回った。大きい prefill 寄りの条件では Triton の実装でも帯域が出やすい。
- 次は CUDA optimized RMSNorm で、CUDA naive の reduction / memory access を詰める。

## 2026-06-19 - CUDA naive baseline matrix

目的:

- PyTorch eager と CUDA C++ naive RMSNorm を同じ入力 shape で比較する。
- CUDA extension が PyTorch から呼べることを確認する。
- 今後の Triton / CUDA optimized 実装の比較基準を作る。

実行条件:

- GPU: NVIDIA GeForce RTX 4070
- CUDA capability: 8.9
- PyTorch: 2.11.0+cu128
- torch CUDA: 12.8
- dtype: FP16 input / FP32 accumulate / FP16 output
- warmup: 20
- runs: 100

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-211027-baseline-matrix-cuda-naive/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-211027-baseline-matrix-cuda-naive/metadata.json`
- `rmsnorm/results/rtx4070/runs/20260619-211027-baseline-matrix-cuda-naive/summary.csv`
- `rmsnorm/results/rtx4070/runs/20260619-211027-baseline-matrix-cuda-naive/summary.md`
- `rmsnorm/results/rtx4070/runs/20260619-211027-baseline-matrix-cuda-naive/console.txt`

要約:

| metric | value |
|---|---:|
| compared shapes | 20 |
| min speedup vs PyTorch eager | 2.92x |
| median speedup vs PyTorch eager | 4.98x |
| max speedup vs PyTorch eager | 7.99x |

代表結果:

| tokens | hidden | PyTorch eager us | CUDA naive us | speedup |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 54.864 | 7.840 | 7.00x |
| 32 | 8192 | 35.840 | 12.288 | 2.92x |
| 512 | 8192 | 211.968 | 26.528 | 7.99x |

メモ:

- CUDA naive は最適化版ではないが、PyTorch eager の複数 op / launch overhead を避けられるため小さい shape でも差が出る。
- `max_abs_error` は最大で `0.00390625`。FP16 出力としては初回比較では許容範囲だが、optimized 実装でも継続確認する。
- effective bandwidth は大きい shape ほど上がる。次は token / hidden ごとの傾向を整理して、memory-bound の説明につなげる。

## 2026-06-19 - CUDA naive smoke test

保存先:

- `rmsnorm/results/rtx4070/runs/20260619-211005-smoke-cuda-naive/metadata.md`
- `rmsnorm/results/rtx4070/runs/20260619-211005-smoke-cuda-naive/summary.csv`

目的:

- run record 生成が動くことを確認する。
- `metadata.md`, `metadata.json`, `summary.csv`, `summary.md`, `console.txt` が保存されることを確認する。
