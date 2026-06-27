# vLLM GemmaRMSNorm Patch Comparison

## Condition

- model: `Qwen/Qwen3.5-2B`
- backend: `vllm/vllm-openai:nightly`
- prompt: `backend_compare/prompts/decode_japanese.txt`
- generated tokens: `128`
- measured runs: `5`
- warmup requests before measurement: `3`
- server option: `--enforce-eager`
- patch backend: `triton`

## Inputs

| variant | record dir |
|---|---|
| unpatched | `backend_compare/results/rtx4070/profile_requests/runs/20260624-231130-openai_compatible-Qwen-Qwen3-5-2B` |
| patched | `backend_compare/results/rtx4070/profile_requests/runs/20260624-231429-openai_compatible-Qwen-Qwen3-5-2B` |

## Result

| variant | mean tokens/s | median tokens/s | min tokens/s | max tokens/s | mean latency ms | median latency ms |
|---|---:|---:|---:|---:|---:|---:|
| unpatched | `93.910` | `93.904` | `93.444` | `94.335` | `1363.022` | `1363.092` |
| patched | `104.659` | `104.644` | `104.526` | `104.816` | `1223.019` | `1223.193` |

## Delta

| metric | delta |
|---|---:|
| mean tokens/s speedup | `1.114x` |
| median tokens/s speedup | `1.114x` |
| mean latency reduction | `10.27%` |

## Read

- 同一条件の request-level 測定では、GemmaRMSNorm Triton fused patch は throughput を改善した。
- warmup 1 はどちらも遅く、初回 JIT / request 初期化の影響があるため、比較には warmup 後の measured runs だけを使った。
- Nsight の別測定では、patch 後に `copy/cast`, `norm/reduce`, `elementwise` の share が下がっており、throughput 改善の説明と整合する。
- 最終記事では、この結果を「Qwen3.5 の GemmaRMSNorm native decomposition を fused kernel に差し替え、実 backend decode で約 1.11x の改善を確認」と表現できる。
