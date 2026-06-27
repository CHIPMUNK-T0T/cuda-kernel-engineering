# Backend Benchmark Summary

| backend | model | run | latency ms | prompt tokens | generated tokens | wall tokens/s | backend eval tokens/s | load ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openai_compatible | `Qwen/Qwen3.5-2B` | 1 | 1385.818 | 54 | 128 | 92.364 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 2 | 1389.202 | 54 | 128 | 92.139 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 3 | 1366.302 | 54 | 128 | 93.684 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 4 | 1403.976 | 54 | 128 | 91.170 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 5 | 1386.363 | 54 | 128 | 92.328 | n/a | n/a |

## Aggregate

- median wall tokens/s: `92.328`
- min wall tokens/s: `91.170`
- max wall tokens/s: `93.684`
- mean wall tokens/s: `92.337`
