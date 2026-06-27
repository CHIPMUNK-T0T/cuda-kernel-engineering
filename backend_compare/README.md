# Real Backend Evaluation

Qwen3.5 2B を実 backend で動かし、RMSNorm kernel 評価を LLM inference の tokens/sec 検証へ接続するためのフォルダです。

ここでは、まだ自作 CUDA / Triton RMSNorm kernel を backend に組み込みません。まず vLLM + Qwen3.5 2B の baseline を作り、次に Nsight Systems で backend 内の比率を見ます。Ollama と 4B は後段で扱います。

## Position

ここまでの repo では、次のことを確認済みです。

- kernel 単体では custom RMSNorm / fused residual RMSNorm が速い。
- block / mini decode に戻しても改善が残る。
- attention / KV cache 付き mini decoder でも改善が残る。

次の問いはこれです。

> 実 backend で Qwen3.5 2B を動かしたとき、RMSNorm 最適化が効きそうな余地はどれだけ残っているか。

## Model Targets

| backend | primary | later |
|---|---|---|
| vLLM | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3.5-4B` |
| Ollama | later | `qwen3.5:2b`, `qwen3.5:4b` |

## Benchmark

vLLM server:

```bash
bash backend_compare/scripts/start_vllm_qwen35.sh
```

vLLM benchmark:

```bash
bash backend_compare/scripts/run_vllm_qwen35.sh 2b
```

既定では `warmup=1`, `runs=5`, `max_tokens=128` です。初回 request を測定値に混ぜず、steady-state を見るためです。

Ollama は後段:

```bash
bash backend_compare/scripts/run_ollama_qwen35.sh 2b
```

## vLLM Defaults

RTX 4070 12GB 向けに、まず保守的な条件で起動します。

| setting | value |
|---|---|
| model | `Qwen/Qwen3.5-2B` |
| image | `vllm/vllm-openai:nightly` |
| max model len | `4096` |
| max num batched tokens | `4096` |
| gpu memory utilization | `0.90` |
| CUDA graph | disabled by `--enforce-eager` |
| allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

必要なら環境変数で変えます。

```bash
MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.92 bash backend_compare/scripts/start_vllm_qwen35.sh
```

## Output

Results are written under:

```text
backend_compare/results/rtx4070/
```

Each run records:

- backend
- model
- prompt path
- requested max tokens
- wall latency
- generated tokens/sec
- prompt tokens / generated tokens if available
- backend-specific raw response summary

## First Baseline

summary:

```text
backend_compare/results/rtx4070/vllm_qwen35_2b_baseline.md
```

`Qwen/Qwen3.5-2B`, prompt tokens `54`, generated tokens `128`, warmup `1`, measured runs `5`, RTX 4070。

| metric | value |
|---|---:|
| median wall tokens/s | 92.328 |
| mean wall tokens/s | 92.337 |
| min wall tokens/s | 91.170 |
| max wall tokens/s | 93.684 |

読み取り:

- warmup request は集計から除外した。
- steady-state は約 `92 tok/s`。
- これは backend baseline であり、まだ custom RMSNorm kernel による speedup ではない。
- Nsight Systems で vLLM server 内の RMSNorm / attention / GEMM 比率を確認した。

## Nsight Systems

request-only 手順:

```bash
docker stop vllm-qwen35-2b
NSYS_DELAY=90 NSYS_DURATION=180 bash backend_compare/scripts/start_vllm_qwen35_nsys_request_only.sh 2b 8000
```

別 terminal:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

`NSYS_DELAY` は container launch から capture 開始までの秒数。request script は `/v1/models` を待ってから benchmark を投げる。Qwen3.5 2B は起動が 50-70 秒程度なので、まず `NSYS_DELAY=90` / `NSYS_DURATION=180` で、server ready 後の request を広めに捕まえる。

whole-session 手順:

```bash
docker stop vllm-qwen35-2b
NSYS_DURATION=90 bash backend_compare/scripts/start_vllm_qwen35_nsys.sh 2b 8000
```

別 terminal:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

詳細: `backend_compare/docs/nsight_vllm.md`

初回結果:

```text
backend_compare/results/rtx4070/nsys_summary.md
```

300秒 profile で vLLM server 起動、model load、warmup、request を含めて取得した。request 中の measured throughput は profiler overhead 込みで median `73.734 tok/s`。

| category | share |
|---|---:|
| elementwise / copy / misc | 44.536% |
| FlashAttention | 31.082% |
| GEMM / GEMV | 22.070% |
| norm-related | 0.780% |

request-only profile では、起動・model load を外した状態で次の傾向になった。

| category | share |
|---|---:|
| GEMM / GEMV | 49.943% |
| elementwise / copy / misc | 45.729% |
| norm-related | 1.677% |
| Qwen hybrid / Mamba-like | 1.669% |

ただし、この集計には server ready 前の warmup-heavy な部分が一部残っていた。`cuda_gpu_trace.csv` を request window だけに寄せて再集計すると、次の傾向になった。

| family | share |
|---|---:|
| cuBLAS GEMV | 86.788% |
| PyTorch copy / cast | 3.314% |
| PyTorch elementwise math | 3.284% |
| vLLM SwiGLU `act_and_mul_kernel` | 0.407% |

読み取り:

- 実 vLLM backend では norm-related kernel は見えるが、whole-session profile ではかなり小さい。
- request window では GEMV が支配的。
- `SwiGLU` は vLLM 側で `act_and_mul_kernel` として既に custom kernel 化されており、request-window share も小さい。
- mini decoder では fused RMSNorm の効果が見えたが、vLLM backend 全体では他の kernel と runtime overhead に薄まる。

## Current Claim Boundary

ここで baseline が取れても、まだ「自作 kernel で vLLM / Ollama が速くなった」とは言わない。

言えるようになるのは、次のどちらかを満たした後です。

- backend 内の RMSNorm 相当処理を差し替えて tokens/sec を比較する。
- Nsight Systems で backend 内の RMSNorm 比率を確認し、mini decoder の結果と対応づける。

## Decision From Profiling

vLLM request window では、RMSNorm より `cuBLAS GEMV` が支配的だった。次点で PyTorch native の copy/cast/math kernel が細かく残っている。

次の候補:

| path | 目的 | 難しさ |
|---|---|---|
| elementwise / copy fusion | 多数の小 kernel と memory traffic を減らす | medium |
| decode GEMV / small-batch matmul | 最大比率の GEMM/GEMV を狙う | high |
| backend 直接改造 | vLLM / llama.cpp に custom kernel を入れる | high |

次は `decode_gemv/` と `elementwise_fusion/` のどちらを先に扱うかを決める。
