# Decode GEMV Plan

## Current Status

PyTorch/cuBLAS baseline done.

## Steps

1. baseline harness を作る
   - PyTorch `matmul` / `linear`
   - correctness check
   - latency measurement
   - status: done

2. benchmark matrix を決める
   - `tokens=1` を中心にする
   - `in_features=2048/4096`
   - QKV / Wo / MLP projection 相当の `out_features`
   - status: initial default done

3. baseline を計測する
   - `bash decode_gemv/scripts/run_bench.sh`
   - status: done
   - summary: `decode_gemv/results/rtx4070/baseline_summary.md`

4. Triton baseline を追加する
   - まずは単純な GEMV / matmul kernel
   - PyTorch / cuBLAS path と比較する
   - status: first GPU measurement done
   - run: `bash decode_gemv/scripts/run_triton_bench.sh`
   - summary: `decode_gemv/results/rtx4070/triton_summary.md`

5. CUDA C++ naive GEMV を追加する
   - PyTorch extension から呼ぶ
   - correctness と latency を確認する

6. CUDA C++ optimized GEMV を追加する
   - warp-level reduction
   - vectorized load
   - shape specialization

7. Nsight で見る
   - representative decode shape
   - PyTorch / Triton / CUDA を比較
   - cuBLAS に対して何が足りないかを見る
   - next target: `torch_linear` vs `triton_gemv`
   - priority shapes: `1x2048x8192`, `1x4096x11008`
   - run: `bash decode_gemv/scripts/run_nsight.sh torch_linear 1 2048 8192`
   - run: `bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 2048 8192`
   - status: done for `torch_linear` / `triton_gemv`
   - summary: `decode_gemv/results/rtx4070/nsight_compare.md`

8. Triton tuning
   - tune `BLOCK_N` / `BLOCK_K`
   - compare memory throughput and duration
   - decide whether Triton has enough headroom
   - status: tuning and tuned Nsight check done
   - run: `bash decode_gemv/scripts/tune_triton.sh`
   - summary: `decode_gemv/results/rtx4070/triton_tuning_summary.md`
   - Nsight summary: `decode_gemv/results/rtx4070/nsight_compare.md`

9. mini decode / projection block に戻す
   - QKV / Wo / MLP projection に近い構成で比較する
   - status: first measurement done
   - run: `bash decode_gemv/scripts/run_projection_block.sh`
   - output: `decode_gemv/results/rtx4070/projection_block/`
   - compare: `torch_linear` vs tuned `triton_tuned`
   - summary: `decode_gemv/results/rtx4070/projection_block_summary.md`

10. projection type ごとに分解する
   - QKV only
   - Wo only
   - MLP gate/up only
   - MLP down only
   - goal: block 全体で Triton が負けた原因を shape 別に切り分ける
   - status: done
   - run: `bash decode_gemv/scripts/run_projection_types.sh`
   - output: `decode_gemv/results/rtx4070/projection_types/`
   - summary: `decode_gemv/results/rtx4070/projection_type_summary.md`
   - read: deduped result shows Wo wins, QKV/MLP up/MLP down are slower

11. projection type の deduped 再計測
   - QKV/Wo の重複 shape を除外する
   - run: `bash decode_gemv/scripts/run_projection_types.sh`
   - status: done

12. projection type ごとに Triton tuning する
   - QKV / Wo / MLP up / MLP down を別々に tune する
   - current global config: `BLOCK_K=128, BLOCK_N=32`
   - goal: block 全体の負けが config mismatch か、kernel design の限界かを切り分ける
   - first order:
     - Wo: current win が block 設定を変えても安定するか確認する
     - MLP down: もっとも負けが大きい projection として重点的に見る
   - run: `bash decode_gemv/scripts/tune_projection_types.sh`
   - output: `decode_gemv/results/rtx4070/projection_type_tuning/`
   - summary: `decode_gemv/results/rtx4070/projection_type_tuning_summary.md`
   - status: done for Wo / MLP down
   - read:
     - Wo は `4096 -> 4096` で勝つが、`2048 -> 2048` では負ける
     - MLP down は `8192 -> 2048` だけ勝ち、他 shape はまだ負ける
     - projection-specific tuning は有効だが、単一 Triton mapping では全 projection に一般化しない

13. QKV / MLP up の projection type tuning
   - QKV / MLP up も同じ tuner で確認する
   - status: done
   - summary: `decode_gemv/results/rtx4070/projection_type_tuning_summary.md`
   - read:
     - best Triton config が勝つのは `3 / 12` projection shapes
     - QKV は小さい shape で勝つが、大きい QKV では負ける
     - MLP up は全 measured shape でまだ負ける

14. per-projection best config を projection block に戻す
   - `triton_projection_tuned` を追加
   - projection type tuning の run summary から shape ごとの best `BLOCK_K/BLOCK_N` を使う
   - status: done
   - summary: `decode_gemv/results/rtx4070/projection_block_per_projection_summary.md`
   - read:
     - fixed Triton より全 block shape で改善
     - ただし `torch_linear` / cuBLAS には未達で、`triton_projection_tuned / torch_linear = 1.056-1.146x`
     - 全体高速化には GEMV 単体 tuning だけでは足りず、CUDA C++ GEMV または projection 後段 fusion を検討する

15. 次の実装判断
   - Wo / QKV の勝ち shape を Nsight Compute で確認する
   - MLP up / MLP down の負け理由を Nsight で見る
   - 次に作るなら CUDA C++ GEMV より、projection 後段との fusion 候補を先に設計する
   - status: next

## Baseline Read

- `tokens=1` では `torch_linear` が全 shape で最速。
- median latency は `torch_matmul 78.800 us`、`torch_linear 33.264 us`。
- tokens が増えると TFLOP/s が伸び、`tokens=128` では `torch_matmul` が勝つ shape も出る。
- 次の Triton / CUDA は、まず `tokens=1` の `torch_linear` path にどこまで迫れるかを見る。
- Triton first result では、単純 Triton kernel は `torch_linear` に未勝利。
- ただし大きい output width では差が `1.07-1.09x` まで縮み、`in=2048, out=8192` では `torch_matmul` より速い。
- Nsight では cuBLAS / Triton とも DRAM throughput が `84-92%` で、decode GEMV が memory-throughput limited であることを確認。
- 大きい shape では Triton は `206.56 us`、cuBLAS は `203.68 us` でかなり近い。
- Triton tuning では `BLOCK_K=128, BLOCK_N=32` が全代表 shape で最速。
- `1x2048x8192` では tuned Triton が `torch_linear` 比 `1.868x`。
- tuned Triton の Nsight では、初期 Triton 比で duration `81.54 us -> 78.08 us`、DRAM throughput `84.47% -> 90.78%`、achieved occupancy `23.20% -> 46.29%`。
- 次は単体 GEMV ではなく、`QKV / Wo / MLP up / MLP down` 相当をまとめた projection block で、単体の改善が合計 latency に残るかを見る。
- projection block では `torch_linear` が全構成で速く、`triton_tuned / torch_linear` は `1.08-1.28x`。
- 単体 GEMV の shape-specific win は、混合 projection block 全体にはまだ残っていない。

## First Milestone

PyTorch baseline と benchmark harness を作り、decode GEMV の latency matrix を出す。

Run:

```bash
bash decode_gemv/scripts/run_bench.sh
```

Triton first measurement:

```bash
bash decode_gemv/scripts/run_triton_bench.sh
```

Projection block measurement:

```bash
bash decode_gemv/scripts/run_projection_block.sh
```

Projection type breakdown:

```bash
bash decode_gemv/scripts/run_projection_types.sh
```
