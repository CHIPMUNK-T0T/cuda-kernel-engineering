# CUDA Kernel Engineering for LLM Decode

**JP** — LLM 推論の decode フェーズに現れる小さな kernel bottleneck を題材に、CUDA C++ / Triton で kernel を自作し、PyTorch baseline・Nsight profiling・実 vLLM backend まで一気通貫で評価した、カーネルエンジニアリングの実践記録です。目的は「vLLM を速くすること」ではなく、*decode workload を構成要素に分解し、自作 kernel がどこで効き、どこで他の処理に埋もれるのかを benchmark と profiler で定量的に切り分ける力* を示すことです。

**EN** — A hands-on record of CUDA kernel engineering, using the small kernel bottlenecks of the LLM decode phase as the problem domain. I implement kernels in CUDA C++ / Triton and evaluate them end to end — against a PyTorch baseline, under Nsight profiling, and finally inside a real vLLM backend. The goal is *not* to "make vLLM faster", but to demonstrate the ability to decompose a decode workload into its components and quantify, with benchmarks and profilers, where a custom kernel helps and where it gets buried under other work.

```text
RMSNorm kernel
  -> residual RMSNorm + matmul block
  -> multi-layer mini decode
  -> mini decoder with attention / KV cache
  -> real vLLM backend baseline + profiling
```

## What This Demonstrates / 示している力

**JP**

- **CUDA C++ kernel 実装**: warp shuffle / shared memory reduction による RMSNorm、residual add + RMSNorm を 1 kernel に fuse して memory traffic と kernel launch overhead を削減。
- **Triton kernel 実装と autotuning**: 同一演算を Triton でも実装し、block size を tuning して cuBLAS / CUDA C++ と比較。
- **PyTorch C++ extension 連携**: 自作 `.cu` を PyTorch から呼び出す binding を実装。
- **計測 harness**: 各 stage で correctness check（max abs error）と latency 測定を統一手順で実施。
- **Nsight による分析**: Nsight Compute で kernel 単体、Nsight Systems で workload 全体の kernel time 比率を分解。
- **bottleneck-driven な進め方**: kernel 単体 → block → multi-layer decode → attention/KV cache → 実 vLLM backend と段階的に戻し、profile から次の最適化対象を選ぶ。
- **実 backend への接続**: Qwen3.5-2B を vLLM で測り、request-window profile を再集計して支配的 kernel を特定。さらに vLLM への patch と upstream issue draft まで踏み込む。

**EN**

- **CUDA C++ kernels**: RMSNorm using warp-shuffle / shared-memory reductions, and a fused residual-add + RMSNorm that cuts memory traffic and kernel-launch overhead.
- **Triton kernels with autotuning**: the same ops re-implemented in Triton, with block sizes tuned and compared against cuBLAS / CUDA C++.
- **PyTorch C++ extension integration**: bindings that call hand-written `.cu` kernels from PyTorch.
- **Measurement harness**: a uniform per-stage flow for correctness checks (max abs error) and latency measurement.
- **Nsight-based analysis**: Nsight Compute for single-kernel detail, Nsight Systems for the kernel-time breakdown of the whole workload.
- **Bottleneck-driven methodology**: walking the workload back up — single kernel → block → multi-layer decode → attention/KV cache → real vLLM backend — and choosing the next target from the profile.
- **Connection to a real backend**: profiling Qwen3.5-2B under vLLM, re-aggregating the request window to identify the dominant kernel, and going as far as a vLLM patch and an upstream issue draft.

## Key Numbers / 主要な数値

**JP**

- Fused residual RMSNorm: CUDA optimized が PyTorch eager 比 **median 5.3x / max 12.2x**。
- attention + KV cache 付き mini decoder で **最大 1.42x** の latency 改善。
- Nsight Systems 上で RMSNorm 側の kernel time 比率を **23–29% → 9–14%** に低減。
- 実 vLLM request-window では **cuBLAS GEMV が 86.8%** を占めると特定し、削減対象を選定。
- その profile 起点で Qwen3.5 の `GemmaRMSNorm` path を fused kernel 化し、**実 vLLM backend の decode TPOT を約 15% 改善（decode tokens/s 94 → 112）**（限定条件下）。

**EN**

- Fused residual RMSNorm: the CUDA-optimized kernel reaches **median 5.3x / max 12.2x** over PyTorch eager.
- Up to **1.42x** latency improvement on the mini decoder with attention + KV cache.
- RMSNorm's share of kernel time dropped from **23–29% to 9–14%** under Nsight Systems.
- In the real vLLM request window, **cuBLAS GEMV accounts for 86.8%** of kernel time — which guided what to optimize.
- Acting on that profile, fusing Qwen3.5's `GemmaRMSNorm` path **improved real vLLM decode TPOT by ~15% (decode tokens/s 94 → 112)** under bounded conditions.

## What This Does NOT Claim / まだ言わないこと

**JP** — RMSNorm 単体 kernel が任意の実 backend で tokens/sec を上げるとは言いません。§7 の実 backend 改善（約 15%）は **Qwen3.5-2B / vLLM nightly / `--enforce-eager` / decode-heavy** という限定条件での結果で、全 workload・batching・CUDA Graph / `torch.compile`・long prefill・他モデルへの一般化は主張しません。GEMV 本体は cuBLAS が強く、正面からの置き換えは未達です。

**EN** — I do not claim the standalone RMSNorm kernel raises tokens/sec on arbitrary real backends. The ~15% real-backend gain in §7 holds under **bounded conditions** (Qwen3.5-2B / vLLM nightly / `--enforce-eager` / decode-heavy) and does not generalize to all workloads, batching, CUDA Graph / `torch.compile`, long prefill, or other models. cuBLAS remains strong on GEMV itself, which I did not replace head-on.

## What Was Built

| stage | folder | 目的 |
|---|---|---|
| kernel 単体 | `rmsnorm/` | RMSNorm / Fused Residual RMSNorm を PyTorch / CUDA C++ / Triton で比較 |
| block 評価 | `mini_transformer_block/` | `residual_rmsnorm + matmul` に戻し、GEMM で効果が薄まるかを見る |
| mini decode 評価 | `mini_llm_decode/` | 複数 layer の decode 風 workload で改善が積み上がるかを見る |
| mini decoder KV 評価 | `mini_decoder_kv/` | attention / KV cache / GQA / QKV / Wo を入れた本物寄り decode で比率を見る |
| real backend 評価 | `backend_compare/` | Qwen3.5 2B を vLLM で測り、tokens/sec 検証へ進む |
| decode GEMV 評価 | `decode_gemv/` | vLLM request の最大要因だった decode GEMV / small-batch linear を調べる |
| projection / norm fusion | `decode_projection_fusion/` | Gemma RMSNorm fused kernel と copy/add/mul elementwise を実装し、vLLM への patch + request-window 再解析、upstream issue draft まで進める |
| 次テーマ候補 | `elementwise_fusion/` | vLLM request に残る PyTorch native copy/cast/math kernel を削減できるか検討する |

## Target Operation

```text
RMSNorm:
y = x * rsqrt(mean(x^2) + eps) * weight

Fused Residual RMSNorm:
z = x + residual
y = z * rsqrt(mean(z^2) + eps) * weight
```

LLM decoder block では residual add と RMSNorm が頻出します。これを separate kernels で実行すると、intermediate tensor の read/write と kernel launch が増えます。この実験では、その部分を 1 kernel に fused して memory traffic と launch overhead を減らします。

## Key Results

### 1. RMSNorm / Fused Residual RMSNorm

RMSNorm 単体では、CUDA optimized RMSNorm が PyTorch eager より高速でした。

| metric | value |
|---|---:|
| CUDA optimized median speedup vs PyTorch eager | 5.30x |
| CUDA optimized max speedup vs PyTorch eager | 12.20x |
| CUDA fused median speedup vs CUDA unfused | 1.401x |
| max abs error | 0.00390625 |

代表 shape:

| shape | CUDA unfused | CUDA fused | Triton fused | fastest |
|---|---:|---:|---:|---|
| tokens=1, hidden=4096 | 10.016 us | 7.168 us | 14.192 us | CUDA fused |
| tokens=512, hidden=8192 | 28.976 us | 26.624 us | 24.576 us | Triton fused |

読み取り:

- decode 寄りの小さい shape では CUDA fused が安定して強い。
- prefill 寄りの大きい shape では Triton fused が強くなる条件がある。
- kernel 単体で速いだけでは LLM 全体の改善は断言できないため、次の stage に進めた。

詳細: `rmsnorm/README.md`

### 2. Mini Transformer Block

`residual_rmsnorm + matmul` に戻すと、GEMM が重い条件では改善幅が薄まります。一方、GEMM が軽い条件では RMSNorm fusion の寄与が大きく出ます。

| mode | tokens | hidden | out features | PyTorch unfused | CUDA fused | Triton fused | best |
|---|---:|---:|---:|---:|---:|---:|---|
| decode | 1 | 4096 | 512 | 47.984 us | 15.184 us | 76.608 us | CUDA fused |
| prefill | 512 | 8192 | 512 | 348.160 us | 123.904 us | 123.840 us | Triton fused |
| prefill | 512 | 8192 | 8192 | 1445.888 us | 1184.720 us | 1154.560 us | Triton fused |

読み取り:

- RMSNorm fusion の効果は block に戻しても残る。
- ただし GEMM が支配的になるほど、全体 speedup は小さくなる。

詳細: `mini_transformer_block/README.md`

### 3. Mini LLM Decode

`tokens=1`, `hidden=4096`, distinct projection weights の multi-layer decode 風 workload でも、fused residual RMSNorm の効果は残りました。

| layers | PyTorch unfused | CUDA fused | Triton fused | CUDA vs PyTorch | Triton vs PyTorch |
|---:|---:|---:|---:|---:|---:|
| 8 | 788.528 us | 633.856 us | 632.528 us | 1.24x | 1.25x |
| 16 | 1542.112 us | 1246.128 us | 1244.160 us | 1.24x | 1.24x |
| 32 | 3071.488 us | 2488.848 us | 2474.560 us | 1.23x | 1.24x |

Nsight Systems では、RMSNorm 側の GPU kernel 時間比率が PyTorch unfused の 17.0% から、CUDA fused で 3.5%、Triton fused で 2.1% まで下がりました。

読み取り:

- layer を重ねても改善は消えない。
- fused 後は matmul 側が支配的になる。

詳細: `mini_llm_decode/README.md`

### 4. Mini Decoder With Attention / KV Cache

attention / KV cache / GQA / QKV projection / Wo を入れた `qwen_like_2b` 合成 decode workload でも、fused residual RMSNorm の効果は残りました。

| context len | PyTorch unfused | CUDA fused | Triton fused | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 2224.288 us | 1570.112 us | 1798.608 us | 1.42x | 1.24x | CUDA fused |
| 512 | 2070.256 us | 1528.784 us | 1762.304 us | 1.35x | 1.17x | CUDA fused |
| 2048 | 2671.760 us | 2292.736 us | 2064.736 us | 1.17x | 1.29x | Triton fused |

Nsight Systems section ratio:

| context | implementation | RMSNorm | KV cache | attention total | QKV + Wo |
|---:|---|---:|---:|---:|---:|
| 128 | PyTorch unfused | 29.2% | 14.2% | 33.2% | 23.5% |
| 128 | CUDA fused | 8.6% | 19.6% | 46.6% | 25.4% |
| 2048 | PyTorch unfused | 23.1% | 21.3% | 28.6% | 27.1% |
| 2048 | CUDA fused | 9.0% | 26.2% | 39.8% | 24.9% |
| 2048 | Triton fused | 10.7% | 28.0% | 25.6% | 35.8% |

読み取り:

- PyTorch unfused では RMSNorm 側が 23-29% 程度を占める。
- CUDA/Triton fused 後は 9-14% 程度まで下がる。
- attention / KV cache を含めても、RMSNorm fusion の効果は完全には埋もれない。
- ただし fused 後の改善余地は、RMSNorm 単体ではなく decoder body 側に移る。

詳細: `mini_decoder_kv/README.md`

### 5. Real Backend Baseline / Profiling

Qwen3.5 2B を vLLM で測る baseline harness を追加しました。Ollama と 4B は後段で扱います。

| backend | primary | later |
|---|---|---|
| vLLM | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3.5-4B` |
| Ollama | later | `qwen3.5:2b`, `qwen3.5:4b` |

この段階では、まだ custom RMSNorm kernel を backend に組み込んでいません。目的は、vLLM の tokens/sec baseline を取り、Nsight Systems で backend 内の比率を見ることです。

初回 baseline:

| model | prompt tokens | generated tokens | warmup | steady median tokens/s |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-2B` | 54 | 128 | 1 | 92.328 |

これは backend baseline であり、custom RMSNorm kernel による vLLM speedup ではありません。

Nsight Systems 手順は `backend_compare/docs/nsight_vllm.md` にあります。実 backend 内で RMSNorm / attention / GEMM / KV cache の比率を確認しました。

初回 Nsight Systems では、vLLM server 起動から request までを含む whole-session profile を取得した。norm-related kernel は見えたが、GPU kernel time の主成分は FlashAttention と GEMM/GEMV だった。

| category | share |
|---|---:|
| FlashAttention | 31.082% |
| GEMM / GEMV | 22.070% |
| norm-related | 0.780% |

request-only profile の全体集計では、起動・model load を外した状態で次の傾向になった。

| category | share |
|---|---:|
| GEMM / GEMV | 49.943% |
| elementwise / copy / misc | 45.729% |
| norm-related | 1.677% |
| Qwen hybrid / Mamba-like | 1.669% |

さらに request window だけに寄せて `cuda_gpu_trace.csv` を再集計すると、cuBLAS GEMV が `86.788%` を占めた。`SwiGLU` 相当の `vllm::act_and_mul_kernel` は `0.407%` で、既に vLLM 側で custom kernel 化されている。

この結果から、実 vLLM backend の end-to-end tokens/sec では RMSNorm 単体より、decode GEMV と残存する PyTorch native copy/cast/math kernel の影響を見る方が自然です。

### 6. Decode GEMV

vLLM request-window で最大だった `cuBLAS GEMV` を、standalone benchmark で PyTorch / cuBLAS / Triton と比較しました。

| 観点 | 結果 |
|---|---|
| baseline | `tokens=1` decode shape では cuBLAS-backed `torch_linear` が全 8 shape で最速。48 shape 中 31 勝。 |
| Triton GEMV（初期） | 概ね cuBLAS に届かず。 |
| Triton GEMV（tuning 後） | `block_k`/`block_n` を tuning すると、代表 shape `1x2048x8192` で **1.87x**、`qkv 1x2048x6144` で **1.36x** と cuBLAS を上回る条件が出た。 |
| Nsight | GEMV は memory-throughput 律速。 |

読み取り:

- cuBLAS は非常に強く、GEMV 本体を正面から置き換えるのは難しい。
- ただし autotuning により、特定の decode projection shape では custom Triton が cuBLAS を超えうる。
- 「standalone microbenchmark の勝ち」が projection block / 実 backend でも残るかが次の問い。

詳細: `decode_gemv/README.md`

### 7. Decode Projection Fusion — Real Backend Improvement

GEMV 本体ではなく、vLLM trace に残る **copy / cast / norm / reduce** に着目しました。発生源として Qwen3.5 の `GemmaRMSNorm` path（fp32 cast → reduction → `weight + 1` → output cast）を特定し、Triton fused kernel に近づけて vLLM に patch しました。

`stream=True` の decode-heavy workload（real vLLM backend 実測）:

| max tokens | TPOT | decode tokens/s | total latency |
|---:|---|---|---:|
| 128 | `10.615 -> 8.919 ms` | `94.214 -> 112.129` | `-15.88%` |
| 512 | `10.557 -> 8.927 ms` | `94.720 -> 112.017` | `-15.43%` |
| 2048 | `10.625 -> 9.037 ms` | `94.121 -> 110.659` | `-14.94%` |

読み取り:

- 自作 fused kernel が **実 vLLM backend の decode per-token latency を約 15% 改善**した（後述の限定条件下）。
- GEMV を置き換えるのではなく、その周辺に残る tiny kernel を fusion する、という profile 起点の判断が効いた。
- ここまで来ると価値は RMSNorm 単体ではなく、「実 backend trace から削減対象を見つけて差し込む」プロセスにある。

Claim boundary（確認済み条件）:

- confirmed: Qwen3.5-2B / vLLM nightly / `--enforce-eager` / short prompt + decode-heavy workload
- not claimed: 全 vLLM workload、batching、CUDA Graph / `torch.compile`、long prefill、他モデル

このテーマでは upstream issue draft（`decode_projection_fusion/upstream_issue/`）と vLLM patch（`decode_projection_fusion/vllm_patch/`）まで作成しました。

詳細: `decode_projection_fusion/README.md`

## Engineering Points

このリポジトリで見せたい技術要素:

- PyTorch baseline と custom CUDA / Triton kernel を同一 harness で比較
- correctness check と latency 測定を各 stage で実施
- CUDA C++ extension を PyTorch から呼び出し
- warp shuffle reduction / shared memory reduction を使った RMSNorm
- residual add + RMSNorm fusion による memory traffic 削減
- vLLM profile から次の最適化対象を選ぶ bottleneck analysis
- Nsight Compute / Nsight Systems による kernel 単体と workload 全体の分析
- 「速くなった」で止めず、どの stage で効果が薄まるかを確認

## Environment

主な測定環境:

| item | value |
|---|---|
| GPU | RTX 4070 |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Python | 3.12 |

Python はリポジトリ直下の `.venv` を使います。

```bash
source .venv/bin/activate
```

## Reproduce

代表 benchmark:

```bash
bash rmsnorm/scripts/run_bench.sh
bash mini_transformer_block/scripts/run_bench.sh
bash mini_llm_decode/scripts/run_bench.sh
bash mini_decoder_kv/scripts/run_bench.sh --preset qwen_like_2b --context-len 2048 --runs 30 --warmup 5
bash backend_compare/scripts/start_vllm_qwen35.sh
bash backend_compare/scripts/run_vllm_qwen35.sh 2b
```

Nsight Systems:

```bash
bash mini_decoder_kv/scripts/run_nsys.sh pytorch_unfused qwen_like_2b 2048 1
bash mini_decoder_kv/scripts/run_nsys.sh cuda_residual_fused qwen_like_2b 2048 1
bash mini_decoder_kv/scripts/run_nsys.sh triton_residual_fused qwen_like_2b 2048 1
```

Nsight Compute:

```bash
bash rmsnorm/scripts/run_nsight.sh cuda_residual_fused 1 4096 1
bash rmsnorm/scripts/run_nsight.sh triton_residual_fused 512 8192 1
```

## Repository Layout

```text
.
├── rmsnorm/                   # RMSNorm / fused residual RMSNorm kernels (CUDA C++ / Triton)
│   ├── kernels/
│   ├── benchmarks/
│   ├── scripts/
│   ├── results/
│   └── docs/
├── mini_transformer_block/    # residual RMSNorm + matmul block
├── mini_llm_decode/           # multi-layer decode workload
├── mini_decoder_kv/           # mini decoder with attention / KV cache
├── backend_compare/           # real vLLM Qwen3.5-2B baseline + Nsight profiling
├── decode_gemv/               # decode GEMV / small-batch linear (Triton, tuning)
├── decode_projection_fusion/  # Gemma RMSNorm / elementwise fusion, vLLM patch + upstream issue
├── elementwise_fusion/        # design notes for the next theme
├── docker/
├── pyproject.toml
└── README.md
```

## Current Claim

このリポジトリで現在言えること:

> LLM decode に現れる residual add + RMSNorm を CUDA/Triton で fused kernel 化し、kernel 単体 → block → multi-layer decode → attention/KV cache 付き mini decoder まで段階的に評価した。RMSNorm 側の時間比率を大きく下げ、mini decoder latency で最大 1.42x の改善を確認。さらに実 vLLM backend を Nsight で分解し、profile から選んだ Qwen3.5 `GemmaRMSNorm` path を fused kernel 化することで、限定条件下ながら decode TPOT を約 15% 改善した（decode tokens/s 94 → 112）。

> Built fused residual-add + RMSNorm kernels in CUDA/Triton and evaluated them stage by stage from a single kernel up to a mini decoder with attention/KV cache (up to 1.42x latency, RMSNorm share cut to ~9–14%). Then profiled a real vLLM backend with Nsight and, acting on the profile, fused Qwen3.5's `GemmaRMSNorm` path to improve decode TPOT by ~15% under bounded conditions (decode tokens/s 94 → 112).

次に確認すること / Next:

- cuBLAS GEMV 本体に対する custom kernel の余地（現状は cuBLAS が支配的）
- batching / CUDA Graph / `torch.compile` 下での再現性
- 他モデル・long prefill への一般化
- upstream（vLLM）への還元可能性
