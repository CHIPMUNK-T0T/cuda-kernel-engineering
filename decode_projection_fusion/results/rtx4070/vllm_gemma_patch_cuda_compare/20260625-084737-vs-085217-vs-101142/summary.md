# vLLM GemmaRMSNorm Backend Comparison: unpatched vs Triton vs CUDA C++

## Condition

- GPU: RTX 4070
- model: `Qwen/Qwen3.5-2B`
- backend: vLLM nightly
- mode: `--enforce-eager`
- API: OpenAI-compatible streaming
- prompt: short prompt from `backend_compare`
- warmup: 1
- runs: 3
- max tokens: `128,512,2048`

## Runs

| variant | record |
|---|---|
| unpatched | `backend_compare/results/rtx4070/stream_requests/runs/20260625-084737-openai_compatible_stream-Qwen-Qwen3-5-2B` |
| Triton GemmaRMSNorm patch | `backend_compare/results/rtx4070/stream_requests/runs/20260625-085217-openai_compatible_stream-Qwen-Qwen3-5-2B` |
| CUDA C++ GemmaRMSNorm patch | `backend_compare/results/rtx4070/stream_requests/runs/20260625-101142-openai_compatible_stream-Qwen-Qwen3-5-2B` |

## Result

| max tokens | variant | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean total latency ms | mean decode tokens/s |
|---:|---|---:|---:|---:|---:|---:|
| 128 | unpatched | 34.092 | 10.615 | 10.618 | 1382.165 | 94.214 |
| 128 | Triton patched | 30.048 | 8.919 | 8.897 | 1162.700 | 112.129 |
| 128 | CUDA patched | 27.853 | 8.879 | 8.917 | 1155.458 | 112.628 |
| 512 | unpatched | 34.173 | 10.557 | 10.550 | 5429.002 | 94.720 |
| 512 | Triton patched | 29.531 | 8.927 | 8.918 | 4591.356 | 112.017 |
| 512 | CUDA patched | 28.331 | 8.920 | 8.929 | 4586.286 | 112.112 |
| 2048 | unpatched | 34.349 | 10.625 | 10.596 | 21783.064 | 94.121 |
| 2048 | Triton patched | 30.378 | 9.037 | 9.025 | 18528.584 | 110.659 |
| 2048 | CUDA patched | 30.052 | 9.039 | 9.036 | 18533.316 | 110.629 |

## Delta vs Unpatched

| max tokens | patched backend | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---|---:|---:|---:|
| 128 | Triton | 15.98% | 15.88% | 1.190x |
| 128 | CUDA C++ | 16.35% | 16.45% | 1.195x |
| 512 | Triton | 15.44% | 15.43% | 1.183x |
| 512 | CUDA C++ | 15.51% | 15.52% | 1.184x |
| 2048 | Triton | 14.95% | 14.94% | 1.176x |
| 2048 | CUDA C++ | 14.93% | 14.92% | 1.175x |

## Delta: CUDA C++ vs Triton

| max tokens | TPOT change | total latency change | decode tokens/s change |
|---:|---:|---:|---:|
| 128 | 0.45% faster | 0.62% faster | 0.45% faster |
| 512 | 0.08% faster | 0.11% faster | 0.08% faster |
| 2048 | 0.03% slower | 0.03% slower | 0.03% slower |

## Read

CUDA C++ backend integration now works in the vLLM runtime container after removing the heavier CUDA include path that required `cusparse.h`.

However, backend-level streaming results show almost no meaningful uplift over the Triton backend. CUDA C++ is clearly faster in the isolated mini benchmark, but in vLLM request-level decode the remaining latency is dominated by the whole execution path, not only the single RMSNorm kernel implementation.

The main claim should remain:

- fused GemmaRMSNorm reduces PyTorch native decomposition around RMSNorm
- both Triton and CUDA C++ patched paths preserve the backend-level improvement vs unpatched
- Triton is the better article baseline because it is easier to integrate and already captures almost all request-level gain

CUDA C++ should be presented as an additional engineering validation:

- the runtime build issue was not `ninja`; it was the CUDA header dependency path
- CUDA C++ can be made to load in the vLLM runtime
- mini benchmark speed does not automatically translate to backend-level speed when the optimized kernel is only one part of the decode step
