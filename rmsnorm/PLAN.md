# PLAN.md - RMSNorm kernel lab

## 目的

RTX 4070 12GB 上で、LLM 推論に使われる RMSNorm / Fused Residual RMSNorm kernel を PyTorch、Triton、CUDA C++ で実装比較し、GitHub と Qiita で就職アピールに使える成果物にする。

## 最初に作るもの

```text
rmsnorm/
  DESIGN.md
  PLAN.md
  README.md
  kernels/
    rmsnorm_cuda/
      rmsnorm_naive.cu
      rmsnorm_optimized.cu
      fused_residual_rmsnorm.cu
      binding.cpp
    rmsnorm_triton/
      rmsnorm.py
      fused_residual_rmsnorm.py
  benchmarks/
    bench_rmsnorm.py
    bench_matrix.yaml
  results/
    rtx4070/
      summary.csv
      summary.md
      nsight_notes.md
  scripts/
    build.sh
    run_bench.sh
    run_nsight.sh
  docs/
    analysis_rmsnorm.md
    roofline_notes.md
```

## 実行順

1. README とディレクトリ骨格を作る。
2. PyTorch eager の RMSNorm baseline を作る。
3. benchmark matrix を作る。
4. CUDA extension の build path を作る。
5. CUDA C++ naive RMSNorm を実装する。
6. PyTorch baseline と CUDA naive の correctness check を通す。
7. CUDA event で latency を測る。
8. effective bandwidth を計算して CSV / Markdown に出す。
9. Triton RMSNorm を追加する。
10. CUDA optimized RMSNorm を追加する。
11. Fused Residual RMSNorm を CUDA / Triton で追加する。
12. 代表 shape を Nsight Compute で分析する。
13. README と `docs/analysis_rmsnorm.md` に結果と考察をまとめる。
14. Qiita 記事ドラフトを作る。

## 残タスク

次の優先順で進める。

### 1. Triton RMSNorm 実装

目的:

- PyTorch eager / CUDA naive / Triton の 3 者比較にする。
- CUDA optimized に入る前に、高水準 custom kernel でどこまで速くなるかを確認する。
- 記事1の比較軸を作る。

完了条件:

- `triton_rmsnorm` を benchmark harness から選択できる。
- PyTorch baseline との correctness check を通す。
- default benchmark matrix を保存する。
- `docs/benchmark_log.md` に実行記録を追記する。

作業ステップ:

- [x] Triton kernel 本体を `rmsnorm/kernels/rmsnorm_triton/rmsnorm.py` に作る。
- [x] Python wrapper `triton_rmsnorm(x, weight, eps)` を作る。
- [x] benchmark harness に `triton_rmsnorm` を接続する。
- [x] smoke test を実行する。
- [x] default benchmark matrix を実行する。
- [x] `docs/benchmark_log.md` / README / PLAN を更新する。

### 2. CUDA optimized RMSNorm 実装

目的:

- naive CUDA の改善余地を具体的に示す。
- warp reduction、vectorized load/store、block size 比較を行う。
- effective bandwidth の改善を見る。

完了条件:

- [x] `cuda_optimized` を benchmark harness から選択できる。
- [x] CUDA naive / Triton / CUDA optimized の比較表を保存する。
- [x] どの変更が効いたかを `docs/analysis_rmsnorm.md` に整理する。

作業ステップ:

- [x] `rmsnorm_optimized.cu` を追加する。
- [x] `binding.cpp` に `forward_optimized` を追加する。
- [x] Python wrapper `rmsnorm_optimized(x, weight, eps)` を追加する。
- [x] benchmark harness に `cuda_optimized` を接続する。
- [x] smoke test を実行する。
- [x] default benchmark matrix を実行する。
- [x] `docs/benchmark_log.md` / README / PLAN を更新する。

### 3. Nsight Compute 分析

目的:

- latency だけでなく、なぜ速い/遅いかを説明する。
- memory throughput、occupancy、register 数、stall reason を確認する。

完了条件:

- 代表 shape の Nsight Compute 結果を `results/rtx4070/` に保存する。
- `docs/roofline_notes.md` または `docs/analysis_rmsnorm.md` に観察を書く。

作業ステップ:

- [x] Nsight Compute 用の単体 runner を作る。
- [x] `run_nsight.sh` から implementation / shape を指定できるようにする。
- [x] NVIDIA performance counter 権限を有効化する。
- [x] `tokens=1, hidden=4096` の naive / optimized を計測する。
- [x] `tokens=512, hidden=8192` の naive / optimized を計測する。
- [x] 結果を `docs/analysis_rmsnorm.md` に追記する。

### 4. Fused Residual RMSNorm 実装

目的:

- `z = x + residual` と RMSNorm を fuse し、memory traffic と launch overhead の削減を見る。
- fused が必ず速い前提にせず、decode / prefill 条件で差を見る。
- RMSNorm 単体の高速化から一段進めて、LLM block 内でより実際に近い `residual add + RMSNorm` の効果を見る。

完了条件:

- CUDA / Triton の fused 実装を benchmark harness から選択できる。
- unfused と fused の比較結果を保存する。
- 効果が出る条件、出ない条件を整理する。

作業方針:

- ここまでは `rmsnorm/` の中で続ける。
- `mini_llm_backend/` のような end-to-end 評価用フォルダは、Fused Residual RMSNorm の単体評価後に別途作る。

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

測ること:

- correctness
- latency
- effective bandwidth
- memory traffic 削減の見積もり
- decode 寄り shape と prefill 寄り shape の差

作業ステップ:

- [x] PyTorch unfused baseline を benchmark harness に追加する。
- [x] CUDA unfused baseline を benchmark harness に追加する。
- [x] CUDA fused residual RMSNorm kernel を追加する。
- [x] Triton fused residual RMSNorm kernel を追加する。
- [x] CUDA fused residual RMSNorm の smoke test を実行する。
- [x] CUDA fused residual RMSNorm の default benchmark matrix を実行する。
- [x] Triton fused 追加後の default benchmark matrix を実行する。
- [x] Nsight Compute で代表 shape を確認する。
- [x] `docs/benchmark_log.md` / `docs/analysis_rmsnorm.md` / README / PLAN を更新する。

### 5. README / docs 整理

目的:

- GitHub で見た人が、何を実装し、何を測り、何が分かったかを短時間で理解できるようにする。

完了条件:

- README に結果表と要点を反映する。
- `docs/benchmark_log.md` を一次記録の索引にする。
- `docs/analysis_rmsnorm.md` に技術的な考察をまとめる。

### 6. Qiita 記事ドラフト

目的:

- GitHub 実装と測定結果を、就職アピールとして読める記事にする。

候補:

- 記事1: PyTorch vs CUDA naive vs Triton vs CUDA optimized
- 記事2: Nsight Compute で memory-bound を分析
- 記事3: Fused Residual RMSNorm の効果

## 次タスク 1 の前提準備

Triton RMSNorm 実装に入る前に、以下を確認する。

- [x] `triton` が `.venv` で import できる。
- [x] `triton==3.6.0` を直接依存として明示する。
- [x] `rmsnorm/kernels/rmsnorm_triton/` に実装方針を書く。
- [x] benchmark harness に追加する implementation 名を `triton_rmsnorm` に固定する。
- [x] 初回 smoke test の shape を `tokens=1, hidden=4096` に固定する。
- [x] 本計測は default benchmark matrix を使う。

## 最初のマイルストーン

まずは以下を成立させる。

- [x] PyTorch eager RMSNorm
- [x] CUDA C++ naive RMSNorm
- [x] correctness check
- [x] 1 shape 以上の latency 測定
- [x] `rmsnorm/results/rtx4070/summary.csv`
- [x] `rmsnorm/results/rtx4070/summary.md`

この時点では最速化より、実験の骨格と再現性を優先する。

次は benchmark matrix の複数 shape を流し、CUDA naive の遅い条件を確認してから Triton / CUDA optimized に進む。

## ベンチ条件

### shape

- hidden size: `2048 / 3072 / 4096 / 8192`
- num_tokens: `1 / 8 / 32 / 128 / 512`

### dtype

- input: FP16
- weight: FP16
- accumulate: FP32
- output: FP16
- reference: PyTorch

## Qiita 記事構成

### 記事1

RTX4070でLLMのRMSNorm kernelを自作する: PyTorch / Triton / CUDA C++比較

- なぜ RMSNorm か
- PyTorch baseline
- CUDA naive
- Triton
- CUDA optimized
- latency / GB/s / 誤差比較

### 記事2

RMSNorm kernelはなぜ速くならないのか: Nsight Computeで見るmemory-bound最適化

- effective bandwidth
- occupancy
- register pressure
- memory coalescing
- warp stall
- block size 比較

### 記事3

Fused Residual RMSNormはRTX4070で効くのか: kernel fusionの効果と限界

- residual add + RMSNorm の fusion
- memory traffic 削減
- launch overhead 削減
- register pressure 悪化
- decode / prefill 条件での差

## 今回やらないもの

- FlashAttention 自作
- GEMM / Tensor Core kernel 自作
- FP8 GEMM 自作
- vLLM / SGLang 本体改造
- backward kernel

## 注意点

- RMSNorm は memory-bound として考える。
- latency だけでなく effective bandwidth を必ず出す。
- fused は必ず速い前提にしない。
- Nsight Compute は全 shape ではなく代表 shape でよい。
- GitHub README は success path 中心にする。
- 失敗ログや試行錯誤は必要なら別ファイルに分ける。
