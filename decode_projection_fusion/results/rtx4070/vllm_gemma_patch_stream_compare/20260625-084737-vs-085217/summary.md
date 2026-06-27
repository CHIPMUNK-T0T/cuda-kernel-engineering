# vLLM GemmaRMSNorm Patch Streaming Comparison

## Condition

- model: `Qwen/Qwen3.5-2B`
- backend: `vllm/vllm-openai:nightly`
- endpoint: `/v1/completions`
- mode: `stream=True`
- prompt: `backend_compare/prompts/decode_japanese.txt`
- max tokens: `128`, `512`, `2048`
- measured runs per max_tokens: `3`
- warmup requests per max_tokens before measurement: `1`
- server option: `--enforce-eager`
- patch backend: `triton`

## Inputs

| variant | record dir |
|---|---|
| unpatched | `backend_compare/results/rtx4070/stream_requests/runs/20260625-084737-openai_compatible_stream-Qwen-Qwen3-5-2B` |
| patched | `backend_compare/results/rtx4070/stream_requests/runs/20260625-085217-openai_compatible_stream-Qwen-Qwen3-5-2B` |

All measured runs ended with `finish_reason=length`, so the requested output lengths were actually generated.

## Aggregate Result

| max tokens | variant | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean tokens/s | mean decode tokens/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | unpatched | `34.092` | `10.615` | `10.618` | `11.130` | `1382.165` | `92.614` | `94.214` |
| 128 | patched | `30.048` | `8.919` | `8.897` | `9.333` | `1162.700` | `110.091` | `112.129` |
| 512 | unpatched | `34.173` | `10.557` | `10.550` | `10.829` | `5429.002` | `94.308` | `94.720` |
| 512 | patched | `29.531` | `8.927` | `8.918` | `9.094` | `4591.356` | `111.514` | `112.017` |
| 2048 | unpatched | `34.349` | `10.625` | `10.596` | `10.919` | `21783.064` | `94.018` | `94.121` |
| 2048 | patched | `30.378` | `9.037` | `9.025` | `9.250` | `18528.584` | `110.532` | `110.659` |

## Delta

| max tokens | TTFT reduction | TPOT reduction | ITL p50 reduction | ITL p95 reduction | total latency reduction | tokens/s speedup | decode tokens/s speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | `11.86%` | `15.98%` | `16.21%` | `16.15%` | `15.88%` | `1.189x` | `1.190x` |
| 512 | `13.58%` | `15.44%` | `15.48%` | `16.03%` | `15.43%` | `1.182x` | `1.183x` |
| 2048 | `11.56%` | `14.95%` | `14.83%` | `15.29%` | `14.94%` | `1.176x` | `1.176x` |

## Read

- `max_tokens=128/512/2048` の全条件で patched が改善した。
- TPOT と ITL が一貫して約 `15%` 改善しており、decode 中の per-token 処理に効いたという説明を支える。
- TTFT も約 `12-14%` 改善したが、この測定では first token に prefill と最初の decode が混ざるため、TTFT 改善だけを RMSNorm kernel 効果と断定しない。
- output length が長くなるほど total latency は decode per-token の積み重ねに支配されるため、`2048` tokens でも約 `1.18x` の decode tokens/s 改善が残ったことが重要。
- Nsight の `copy/cast`, `norm/reduce`, `elementwise` 減少と、streaming の TPOT/ITL 改善が同じ方向を示している。
