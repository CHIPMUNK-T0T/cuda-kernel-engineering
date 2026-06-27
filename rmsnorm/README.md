# RMSNorm Kernel Lab on RTX 4070

RTX 4070 12GB 上で、LLM 推論に使われる RMSNorm / Fused Residual RMSNorm kernel を PyTorch、Triton、CUDA C++ で実装比較する実験リポジトリです。

目的は単なる速度比較ではなく、RMSNorm が memory-bound になりやすい理由、naive 実装の遅い理由、optimized / fused 実装で何が改善するかを、latency、effective bandwidth、occupancy、register pressure、warp stall reason の観点から説明できる成果物にすることです。

## Current Status

現在は、PyTorch baseline、CUDA C++ naive RMSNorm、Triton RMSNorm、CUDA C++ optimized RMSNorm、CUDA / Triton fused residual RMSNorm を同じベンチで呼び出し、correctness、latency、Nsight Compute の代表 metrics を比較できる状態です。

実装予定:

| 実装 | 状態 | 目的 |
|---|---|---|
| PyTorch eager | done | correctness baseline |
| torch.compile | planned | compiler baseline |
| Triton RMSNorm | done | 実用寄り custom kernel |
| CUDA C++ naive | done | 自作 CUDA の最小実装 |
| CUDA C++ optimized | done | warp reduction 改善 |
| CUDA C++ fused residual RMSNorm | done | decode 寄りで fusion 効果を見る |
| Triton fused residual RMSNorm | done | prefill 寄りの大きい shape で比較する |

## Latest Result

RTX 4070 上の RMSNorm 単体 default benchmark matrix では、CUDA C++ optimized RMSNorm は全 20 shape で PyTorch eager より高速でした。CUDA naive / Triton / CUDA optimized の比較では、CUDA optimized が 17 shape、CUDA naive が 3 shape で最速でした。

| metric | value |
|---|---:|
| CUDA optimized min speedup vs PyTorch eager | 5.11x |
| CUDA optimized median speedup vs PyTorch eager | 5.30x |
| CUDA optimized max speedup vs PyTorch eager | 12.20x |
| CUDA optimized median speedup vs CUDA naive | 1.14x |
| max abs error | 0.00390625 |

Fused Residual RMSNorm では、CUDA fused が 19 / 20 shape で最速、Triton fused が最大 shape `tokens=512, hidden=8192` で最速でした。

| metric | value |
|---|---:|
| CUDA fused fastest | 19 / 20 |
| Triton fused fastest | 1 / 20 |
| CUDA fused median speedup vs CUDA unfused | 1.401x |
| Triton fused max speedup vs CUDA fused | 1.083x |
| max abs error | 0.00390625 |

代表結果:

| shape | CUDA unfused | CUDA fused | Triton fused | fastest |
|---|---:|---:|---:|---|
| tokens=1, hidden=4096 | 10.016 us | 7.168 us | 14.192 us | CUDA fused |
| tokens=512, hidden=8192 | 28.976 us | 26.624 us | 24.576 us | Triton fused |

Nsight Compute の kernel 単体 duration では、Triton fused は `tokens=1, hidden=4096` でも CUDA fused より短く出ました。ただし benchmark latency では小さい shape で CUDA fused が速く、これは Triton の Python/runtime dispatch overhead が decode 寄りで支配的になるためと考えています。

## LLM Inference Takeaway

今回の結果は「LLM では不向き」という意味ではありません。むしろ、LLM 推論では decode と prefill で効く実装が違い、さらに実 backend ではどの構成要素が支配的かを確認する必要がある、という結果です。

| 場面 | 今回の観察 | 解釈 |
|---|---|---|
| decode 寄り: `tokens=1` | CUDA fused が benchmark latency で強い | 1 token 処理では dispatch overhead と固定コストが目立つ |
| prefill 寄り: `tokens=512, hidden=8192` | Triton fused が最速 | 大きい shape では kernel 本体の効率が出やすい |
| RMSNorm 単体 | PyTorch より custom kernel が速い | ただし LLM 全体の高速化は end-to-end 測定なしには断言しない |

言えること:

- residual add + RMSNorm を fuse すると、CUDA unfused より速くなる条件が多い。
- decode 寄りでは CUDA C++ fused が安定して強い。
- prefill 寄りの大きい shape では Triton fused も有力。

実 backend で確認したこと:

- vLLM + Qwen3.5 2B の request-only profile では、norm-related kernel は見える。
- ただし share は `1.677%` に留まり、`GEMM / GEMV` と `elementwise / copy / misc` が支配的だった。
- そのため、custom RMSNorm kernel だけで vLLM tokens/sec を大きく改善できるとは言わない。

まだ言えないこと:

- LLM 全体の tokens/sec がどれだけ上がるか。
- vLLM や llama.cpp のような実バックエンドで同じ効果が出るか。
- attention / KV cache / GEMM を含めた全体ボトルネックの中で RMSNorm fusion がどれだけ効くか。

そのため、この段階の結論は「LLM で意味がない」ではなく、「単体 kernel としては効果があり、mini decoder でも改善は残る。ただし production backend では RMSNorm 以外の比率が大きく、次は elementwise / copy fusion を検討する」です。

## Target Operation

```text
RMSNorm:
y = x * rsqrt(mean(x^2) + eps) * weight

Fused Residual RMSNorm:
z = x + residual
y = z * rsqrt(mean(z^2) + eps) * weight
```

## Benchmark Matrix

hidden size:

- `2048`
- `3072`
- `4096`
- `8192`

num_tokens:

- `1`
- `8`
- `32`
- `128`
- `512`

小さい `num_tokens` は decode、大きい `num_tokens` は prefill 寄りの条件として扱います。

## Repository Layout

```text
rmsnorm/
  DESIGN.md
  PLAN.md
  README.md
  kernels/
    rmsnorm_cuda/
    rmsnorm_triton/
  benchmarks/
    bench_rmsnorm.py
    bench_matrix.yaml
  results/
    rtx4070/
  scripts/
    run_bench.sh
  docs/
```

## Quick Start

リポジトリトップで `uv` 仮想環境を作り、PyTorch CUDA wheel を入れてから実行します。

```bash
UV_CACHE_DIR=.uv-cache uv venv .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python numpy
```

```bash
bash rmsnorm/scripts/run_bench.sh
```

1 条件だけ確認する場合:

```bash
bash rmsnorm/scripts/run_bench.sh --tokens 1 --hidden 4096 --runs 100 --warmup 20
```

PyTorch baseline だけ確認する場合:

```bash
bash rmsnorm/scripts/run_bench.sh --implementations pytorch_eager
```

Nsight Compute で代表 shape を見る場合:

```bash
bash rmsnorm/scripts/run_nsight.sh cuda_naive 1 4096
bash rmsnorm/scripts/run_nsight.sh cuda_optimized 1 4096
bash rmsnorm/scripts/run_nsight.sh cuda_naive 512 8192
bash rmsnorm/scripts/run_nsight.sh cuda_optimized 512 8192
```

Fused Residual RMSNorm を見る場合:

```bash
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh cuda_residual_fused 1 4096 1
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh cuda_residual_fused 512 8192 1
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh triton_residual_fused 1 4096 1
sudo env NCU_SET=basic NCU_TARGET_PROCESSES=application-only NCU_LAUNCH_SKIP=0 bash rmsnorm/scripts/run_nsight.sh triton_residual_fused 512 8192 1
```

Nsight Compute が `ERR_NVGPUCTRPERM` で失敗する場合は、NVIDIA performance counter の権限設定が必要です。

結果は以下に出力します。

```text
rmsnorm/results/rtx4070/summary.csv
rmsnorm/results/rtx4070/summary.md
```

各実行の一次記録は timestamp 付きで以下にも保存します。

```text
rmsnorm/results/rtx4070/runs/
```

記事用の計測ログ一覧は `rmsnorm/docs/benchmark_log.md` にまとめます。

## Metrics

最低限、以下を記録します。

| 指標 | 意味 |
|---|---|
| latency_us | kernel / 実装単体の実行時間 |
| effective_bandwidth_gb_s | 概算 memory traffic から計算した実効帯域 |
| max_abs_error | PyTorch baseline との差分 |
| max_rel_error | PyTorch baseline との差分 |

Nsight Compute 分析では、代表 shape について以下を確認します。

- achieved occupancy
- registers per thread
- shared memory usage
- warp stall reason
- memory throughput
- L2 hit rate

## Notes

RMSNorm は GEMM と違い、主に memory-bound な処理として扱います。そのため FLOPS ではなく、読み書き byte 数、coalescing、reduction、fusion による memory traffic 削減を中心に分析します。
