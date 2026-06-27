# vLLM Qwen3.5 2B Baseline

## Conditions

- backend: vLLM OpenAI-compatible server
- model: `Qwen/Qwen3.5-2B`
- GPU: RTX 4070
- prompt: `backend_compare/prompts/decode_japanese.txt`
- prompt tokens: `54`
- max tokens: `128`
- warmup: `1`
- measured runs: `5`
- record: `backend_compare/results/rtx4070/runs/20260620-133020-openai_compatible-Qwen-Qwen3-5-2B`

## Results

| run | latency ms | generated tokens | wall tokens/s |
|---:|---:|---:|---:|
| 1 | 1385.818 | 128 | 92.364 |
| 2 | 1389.202 | 128 | 92.139 |
| 3 | 1366.302 | 128 | 93.684 |
| 4 | 1403.976 | 128 | 91.170 |
| 5 | 1386.363 | 128 | 92.328 |

## Aggregate

| metric | value |
|---|---:|
| median wall tokens/s | 92.328 |
| mean wall tokens/s | 92.337 |
| min wall tokens/s | 91.170 |
| max wall tokens/s | 93.684 |

## Read

- Warmup is excluded from the measured runs.
- Steady-state decode throughput for this short prompt is about `92 tok/s`.
- This is a backend baseline only. It does not show a custom RMSNorm kernel speedup yet.
- Next step is Nsight Systems on the vLLM server to see whether RMSNorm remains visible inside the real backend.
