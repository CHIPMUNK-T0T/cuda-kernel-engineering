# Backend Benchmark Summary

| backend | model | run | latency ms | prompt tokens | generated tokens | wall tokens/s | backend eval tokens/s | load ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openai_compatible | `Qwen/Qwen3.5-2B` | 1 | 1741.044 | 54 | 128 | 73.519 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 2 | 1734.369 | 54 | 128 | 73.802 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 3 | 1726.554 | 54 | 128 | 74.136 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 4 | 1736.349 | 54 | 128 | 73.718 | n/a | n/a |
| openai_compatible | `Qwen/Qwen3.5-2B` | 5 | 1717.080 | 54 | 128 | 74.545 | n/a | n/a |

## Aggregate

- median wall tokens/s: `73.802`
- min wall tokens/s: `73.519`
- max wall tokens/s: `74.545`
- mean wall tokens/s: `73.944`
