# CUDA Kernel LLM Inference Lab

RTX 4070 上で、LLM decode に現れる小さな kernel bottleneck を CUDA C++ / Triton で実装・比較し、PyTorch baseline や vLLM 実 backend profiling と接続する実験リポジトリです。

単体 kernel の速度だけで終わらせず、LLM 推論を構成要素に分解して、次の順に戻しながら測定しています。

```text
RMSNorm kernel
  -> residual RMSNorm + matmul block
  -> multi-layer mini decode
  -> attention / KV cache 付き mini decoder
  -> real backend baseline
```

目的は「自作 kernel で LLM 全体が爆速」と断言することではありません。自作 kernel が LLM decode のどの範囲で効き、どこから別の処理に埋もれるのかを、benchmark と Nsight で切り分けることです。

## Result Summary

ここまでの結論は、次の通りです。

- `residual add + RMSNorm` を 1 kernel に fused すると、RMSNorm 側の kernel 時間を大きく削減できる。
- kernel 単体だけでなく、block / mini decode / attention + KV cache 付き mini decoder でも latency 改善が残る。
- attention / KV cache 付き mini decoder では、最大 1.42x の latency 改善を確認した。
- Nsight Systems では、PyTorch unfused の RMSNorm 側比率 23-29% が、CUDA/Triton fused 後に 9-14% 程度まで下がった。
- vLLM + Qwen3.5 2B の request-window profile では、norm-related は小さく、cuBLAS GEMV が支配的だった。
- PyTorch native の copy/cast/math kernel も小さく多数残っていた。
- 次段階として、`decode_gemv/` で decode GEMV / small-batch linear を扱う。

まだ言わないこと:

- custom RMSNorm kernel で vLLM / llama.cpp / Ollama の tokens/sec が上がるとは言わない。
- decode GEMV の custom kernel で実 backend の tokens/sec が上がるかは、これから別テーマとして検証する。

## What Was Built

| stage | folder | 目的 |
|---|---|---|
| kernel 単体 | `rmsnorm/` | RMSNorm / Fused Residual RMSNorm を PyTorch / CUDA C++ / Triton で比較 |
| block 評価 | `mini_transformer_block/` | `residual_rmsnorm + matmul` に戻し、GEMM で効果が薄まるかを見る |
| mini decode 評価 | `mini_llm_decode/` | 複数 layer の decode 風 workload で改善が積み上がるかを見る |
| mini decoder KV 評価 | `mini_decoder_kv/` | attention / KV cache / GQA / QKV / Wo を入れた本物寄り decode で比率を見る |
| real backend 評価 | `backend_compare/` | Qwen3.5 2B を vLLM で測り、tokens/sec 検証へ進む |
| 次テーマ | `decode_gemv/` | vLLM request の最大要因だった decode GEMV / small-batch linear を調べる |
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

### 6. Next Theme: Decode GEMV

RMSNorm の次は、vLLM request-window profile で最大だった `cuBLAS GEMV` を対象にする。

`tokens=1` の LLM decode では、linear projection が GEMV / small-batch matmul として現れやすい。ここを standalone benchmark で PyTorch / cuBLAS / Triton / CUDA C++ と比較する。

最初に見ること:

- `tokens=1` decode shape の latency
- custom CUDA / Triton が cuBLAS にどこまで迫れるか
- shape を変えたときに GEMV 支配がどう変わるか
- cuBLAS に勝てない場合、何が律速か

詳細: `decode_gemv/README.md`

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
├── rmsnorm/
│   ├── kernels/
│   ├── benchmarks/
│   ├── scripts/
│   ├── results/
│   └── docs/
├── mini_transformer_block/
├── mini_llm_decode/
├── mini_decoder_kv/
├── backend_compare/
├── AGENTS.md
├── pyproject.toml
└── README.md
```

## Current Claim

このリポジトリで現在言えること:

> LLM decode に現れる residual add + RMSNorm を CUDA/Triton で fused kernel 化し、kernel 単体、matmul を含む block、multi-layer decode、attention/KV cache 付き mini decoder まで段階的に評価した。結果として、RMSNorm 側の時間比率を大きく下げ、mini decoder latency でも最大 1.42x の改善を確認した。

次に確認すること:

- 実 Qwen 系 config へのさらに厳密な合わせ込み
- Qwen3.5 2B の vLLM backend baseline
- vLLM / llama.cpp / Ollama など実 backend での組み込み可能性
- production inference の tokens/sec に対する寄与
