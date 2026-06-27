# DESIGN.md - RMSNorm kernel lab

## 結論

このリポジトリでは、RTX 4070 12GB 上で LLM 推論向け RMSNorm / Fused Residual RMSNorm kernel を実装し、PyTorch・Triton・CUDA C++ の性能と実装差を比較する。

狙いは、単なる速度比較ではなく、memory bandwidth、reduction、occupancy、register pressure、kernel fusion の観点から、なぜ速いか・なぜ遅いかを説明できるポートフォリオにすること。

## 対象演算

### RMSNorm

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

入力 shape は `[num_tokens, hidden_size]` とする。各 row ごとに hidden dimension 方向へ sum of squares を計算し、RMS で正規化する。

### Fused Residual RMSNorm

```text
z = x + residual
y = z * rsqrt(mean(z^2) + eps) * weight
```

通常は residual add と RMSNorm が別 kernel になりうる。fusion により global memory read/write と kernel launch を減らせる可能性がある。一方で、register pressure が増え、occupancy が落ちる可能性もある。

## 実装対象

| 実装 | 目的 |
|---|---|
| PyTorch eager | correctness baseline |
| torch.compile | compiler baseline |
| Triton RMSNorm | 実用寄り custom kernel |
| CUDA C++ naive | 自作 CUDA の最小実装 |
| CUDA C++ optimized | 最適化の本命 |
| CUDA C++ fused residual RMSNorm | fusion 効果の評価 |

## ベンチ設計

### shape

| hidden size | 意味 |
|---:|---|
| 2048 | 小型 LLM 相当 |
| 3072 | 2B-4B 級でよくある規模 |
| 4096 | 7B 級でよくある規模 |
| 8192 | 大きめ hidden の負荷確認 |

| num_tokens | 想定 |
|---:|---|
| 1 | decode 1 token |
| 8 | small batch decode |
| 32 | moderate decode |
| 128 | prefill 寄り |
| 512 | long prefill 寄り |

decode 的な小さい token 条件と、prefill 的な大きい token 条件で kernel の効き方が変わるかを見る。

## 測定指標

| 指標 | 用途 |
|---|---|
| latency us | kernel 単体の実行時間 |
| effective bandwidth GB/s | memory-bound 性の中心指標 |
| max abs error | PyTorch baseline との差分 |
| max relative error | 数値誤差の確認 |
| achieved occupancy | SM 上の warp 利用率 |
| registers per thread | fusion / 最適化の副作用確認 |
| shared memory usage | reduction 実装の負荷 |
| warp stall reason | 待ち要因の分析 |
| memory throughput | global memory 効率 |
| L2 hit rate | cache の効き方 |

## 方針

RMSNorm は GEMM と違い、主に memory-bound な kernel として扱う。そのため「何倍速いか」だけではなく、読み書き byte 数と latency から effective bandwidth を計算する。

naive 実装では、あえて改善余地が分かる構成にする。optimized 実装では reduction、coalescing、vectorized load、global memory access 削減を順に検討する。

fused residual RMSNorm では、fusion すれば必ず速いとは扱わない。memory traffic 削減、kernel launch 削減、register pressure 増加、occupancy 低下を分けて評価する。

## 成功条件

以下を GitHub README と Qiita 記事で説明できる状態を成功とする。

- RMSNorm がなぜ memory-bound になりやすいか。
- naive 実装がどこで遅いか。
- optimized 実装で何を改善したか。
- fused residual RMSNorm がどの条件で効き、どの条件で効かないか。
- RTX 4070 12GB の実測値が Nsight Compute の指標とどう対応するか。
