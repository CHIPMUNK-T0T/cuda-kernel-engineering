# PLAN.md - Real Backend Evaluation

## 目的

Qwen3.5 2B を実 backend で動かし、ここまでの custom RMSNorm kernel 評価を production inference の tokens/sec 検証へ接続する。

この段階では、まだ自作 kernel を vLLM / Ollama / llama.cpp に組み込まない。まず vLLM + Qwen3.5 2B で同じ prompt、同じ max tokens、同じ測定形式の backend baseline を作る。

## 対象 model

| backend | primary | later |
|---|---|---|
| vLLM | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3.5-4B` |
| Ollama | later | `qwen3.5:2b`, `qwen3.5:4b` |

ローカルに model がない場合は、先に download / pull が必要。

## 比較するもの

- TTFT に近い first request latency
- generated tokens/sec
- prompt tokens / generated tokens
- total latency
- backend 固有の計測値が取れる場合は記録する

通常の steady-state 評価では、warmup request を 1 回以上実行し、集計対象から外す。

## 実行順

1. [x] `backend_compare/` を作る。
2. [x] Ollama / OpenAI-compatible backend を叩ける benchmark script を作る。
3. [x] Qwen3.5 2B first の実行 script を作る。
4. [x] vLLM で Qwen3.5 2B server を起動する。
5. [x] vLLM で Qwen3.5 2B baseline を測る。
6. [x] vLLM Qwen3.5 2B 用の Nsight Systems 実行 script を作る。
7. [x] vLLM Qwen3.5 2B を Nsight Systems で見る。
8. [x] README に実測値を反映する。
9. [ ] 次に custom kernel を差し込める backend を選ぶ。
10. [ ] 必要なら Ollama / Qwen3.5 4B / FP8 4B を追加する。

## 判断基準

最初の目的は、どの backend が速いかを決めることではない。

見るべき点:

- RTX 4070 で Qwen3.5 2B が安定して動くか。
- decode tokens/sec のばらつきが大きすぎないか。
- Nsight Systems で backend 内の RMSNorm / attention / GEMM 比率を追えるか。
- 自作 RMSNorm kernel を差し込む余地がある backend か。

## vLLM 起動設定

RTX 4070 12GB では、まず以下の保守的な条件から始める。

- `Qwen/Qwen3.5-2B`
- `--max-model-len 4096`
- `--max-num-batched-tokens 4096`
- `--gpu-memory-utilization 0.90`
- `--enforce-eager`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

4B や FP8 4B は、2B baseline と Nsight が安定してから追加する。

## 初回 baseline

summary:

```text
backend_compare/results/rtx4070/vllm_qwen35_2b_baseline.md
```

条件:

- backend: vLLM OpenAI-compatible server
- model: `Qwen/Qwen3.5-2B`
- prompt tokens: `54`
- generated tokens: `128`
- warmup: `1`
- measured runs: `5`

| metric | value |
|---|---:|
| median wall tokens/s | 92.328 |
| mean wall tokens/s | 92.337 |
| min wall tokens/s | 91.170 |
| max wall tokens/s | 93.684 |

観察:

- warmup request は集計から除外した。
- steady-state は約 `92 tok/s`。
- これは backend baseline であり、まだ custom RMSNorm kernel による backend speedup ではない。
- 次は vLLM server を Nsight Systems で見て、実 backend 内で RMSNorm がどれだけ見えるか確認する。

## Nsight Systems 手順

詳細:

```text
backend_compare/docs/nsight_vllm.md
```

通常起動の vLLM container が動いている場合は先に止める。

```bash
docker stop vllm-qwen35-2b
```

Terminal 1:

```bash
NSYS_DURATION=90 bash backend_compare/scripts/start_vllm_qwen35_nsys.sh 2b 8000
```

Terminal 2:

```bash
REQUEST_DELAY=125 bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

Terminal 2 は Terminal 1 の直後に起動する。`REQUEST_DELAY=125` により、default の capture 開始 `120s` より少し後に request を投げる。

## 初回 Nsight Systems 結果

summary:

```text
backend_compare/results/rtx4070/nsys_summary.md
```

300秒 profile で、vLLM server 起動、model load、warmup、request を含めて取得した。

| category | share |
|---|---:|
| elementwise / copy / misc | 44.536% |
| FlashAttention | 31.082% |
| GEMM / GEMV | 22.070% |
| norm-related | 0.780% |

観察:

- vLLM backend 内で norm-related kernel は見える。
- ただし whole-session profile では attention と GEMM/GEMV が支配的。
- request-only profile ではないため、次にやるなら capture window を server ready 後の request へ絞る。

## 次: request-only Nsight Systems

startup / model load / warmup を除き、server ready 後の request だけを capture する。

Terminal 1:

```bash
docker stop vllm-qwen35-2b
NSYS_DELAY=90 NSYS_DURATION=180 bash backend_compare/scripts/start_vllm_qwen35_nsys_request_only.sh 2b 8000
```

Terminal 2:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

Terminal 2 は Terminal 1 の直後に起動する。request script は `/v1/models` を待ってから benchmark を投げるため、server 起動前の `ConnectionRefusedError` を避ける。

見るもの:

- norm-related kernel が request-only でも何%見えるか
- FlashAttention / GEMM / GEMV / elementwise の比率
- profiler overhead 込みの request throughput

## request-only Nsight Systems 結果

- record: `backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only`
- request record: `backend_compare/results/rtx4070/profile_requests/runs/20260620-152413-openai_compatible-Qwen-Qwen3-5-2B`
- request median: `71.979 tok/s`

| category | share |
|---|---:|
| GEMM / GEMV | 49.943% |
| elementwise / copy / misc | 45.729% |
| norm-related | 1.677% |
| Qwen hybrid / Mamba-like | 1.669% |

読み取り:

- request-only でも norm-related kernel は見える。
- ただし vLLM backend では GEMM/GEMV と elementwise/copy が支配的。
- 自作 RMSNorm kernel の価値は kernel 単体・mini decoder では説明できるが、production backend の tokens/sec 改善としては、RMSNorm だけでは寄与が小さい可能性が高い。

## 注意

- Ollama は後で baseline 用に扱う。内部実装に直接 custom CUDA kernel を差し込む用途には向きにくい。
- vLLM は production backend として比較しやすいが、内部 kernel 差し替えは設計確認が必要。
- llama.cpp は C/C++ 側の改造対象としては有力だが、CUDA graph / ggml / quantization 経路の理解が必要。
- ここではまず baseline を作り、custom kernel 組み込みは次段階で判断する。
