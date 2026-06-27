# PLAN.md - Mini Transformer Block Evaluation

## 目的

RMSNorm / Fused Residual RMSNorm の kernel 単体高速化が、LLM block に近い処理へ入ったときにどこまで効くかを測る。

見せ方は「自作 kernel で LLM が爆速」ではなく、kernel 単体評価から block 評価へ進め、効果が残る範囲と薄まる範囲を実測で切り分ける。

## 対象

まずは attention や MLP 全体ではなく、最小構成から始める。

```text
y = residual_rmsnorm(x, residual, norm_weight)
out = y @ projection_weight
```

この構成にする理由:

- residual add + RMSNorm は LLM block 内で頻出する。
- 直後に Linear / GEMM を置くことで、kernel 単体の改善が大きい演算に埋もれるかを確認できる。
- attention / KV cache / MLP まで入れる前に、切り分けやすい。

## 比較対象

- `pytorch_unfused`: PyTorch の `x + residual` + RMSNorm + `torch.matmul`
- `cuda_residual_fused`: CUDA fused residual RMSNorm + `torch.matmul`
- `triton_residual_fused`: Triton fused residual RMSNorm + `torch.matmul`

Linear / GEMM は全実装で同じ PyTorch `matmul` を使い、差分を residual RMSNorm 部分に限定する。

## 測定 shape

- decode 寄り: `tokens=1`, `hidden=4096`
- prefill 寄り: `tokens=512`, `hidden=8192`

追加で必要なら `tokens=8,32,128` や `hidden=2048,3072` を足す。

## 完了条件

- [x] benchmark harness を作る。
- [x] PyTorch unfused / CUDA fused / Triton fused を選択できる。
- [x] correctness check を通す。
- [x] smoke test を実行する。
- [x] decode / prefill shape の初回計測結果を保存する。
- [x] RMSNorm 単体結果と block 結果の初期差分を README / docs に整理する。
- [x] Nsight Systems で block 内の RMSNorm 側と matmul 側の比率を見る。
- [x] `out_features` を変えて、GEMM が軽い条件で RMSNorm fusion の効果を見る。

## 実行順

1. [x] `bench_block.py` を作る。
2. [x] smoke test を `tokens=1, hidden=4096` で通す。
3. [x] decode / prefill representative shape を測る。
4. [x] 結果を `docs/analysis_block.md` にまとめる。
5. [x] Nsight Systems で block 内の kernel 並びと時間比率を見る。
6. [x] projection sweep を実行する。

## 初回計測

run:

```text
mini_transformer_block/results/rtx4070/runs/20260620-103339-block-matrix-initial-v2/
```

観察:

- block 全体でも CUDA fused は PyTorch unfused より速い。
- ただし RMSNorm 単体ほどの倍率ではなく、projection の GEMM によって改善幅は薄まる。
- Triton fused は block 全体では CUDA fused とほぼ同等からやや遅い。
- prefill の大きい shape では `max_abs_error` は 0.125 / 0.25 まで出るが、`relative_l2_error` は `1.65e-05` から `3.03e-05` 程度。

次に見ること:

- mini LLM decode 評価に進み、複数 layer / attention 風 projection / MLP 風 projection を含めた latency を見る。

## Nsight Systems 結果

代表 profile:

```text
mini_transformer_block/results/rtx4070/nsys/20260620-105025-pytorch_unfused-tokens1-hidden4096-out4096/
mini_transformer_block/results/rtx4070/nsys/20260620-104609-cuda_residual_fused-tokens1-hidden4096-out4096/
mini_transformer_block/results/rtx4070/nsys/20260620-104716-triton_residual_fused-tokens1-hidden4096-out4096/
mini_transformer_block/results/rtx4070/nsys/20260620-105024-pytorch_unfused-tokens512-hidden8192-out8192/
mini_transformer_block/results/rtx4070/nsys/20260620-104714-cuda_residual_fused-tokens512-hidden8192-out8192/
mini_transformer_block/results/rtx4070/nsys/20260620-104858-triton_residual_fused-tokens512-hidden8192-out8192/
```

要点:

- PyTorch unfused の RMSNorm 側は、decode で 17.082 us / 18.5%、prefill で 245.220 us / 17.0%。
- CUDA fused の RMSNorm 側は、decode で 3.040 us / 4.6%、prefill で 56.868 us / 4.5%。
- Triton fused の RMSNorm 側は、decode で 2.080 us / 2.5%、prefill で 50.114 us / 4.0%。
- fused 後は matmul が GPU kernel 時間の約95%を占めるため、block 全体 speedup は RMSNorm 単体ほど大きくならない。

## Projection Sweep 結果

summary:

```text
mini_transformer_block/results/rtx4070/projection_sweep_summary.md
```

要点:

- decode `tokens=1 hidden=4096 out_features=512`: CUDA fused は PyTorch unfused 比 3.16x。
- prefill `tokens=512 hidden=8192 out_features=512`: CUDA / Triton fused は PyTorch unfused 比 約2.81x。
- `out_features=hidden` まで GEMM を重くすると、block 全体の speedup は 1.1x から 1.25x 程度に戻る。
- Nsight Systems では、`out_features=512` の PyTorch unfused は RMSNorm 側が 58.7% / 71.5% を占める。fusion 後は CUDA で 21.4% / 40.6%、Triton で 13.0% / 34.8%。
- GEMM が軽いほど RMSNorm fusion の寄与が大きく、GEMM が重いほど効果は薄まる。
