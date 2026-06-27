# Mini Decoder KV Analysis

## 現在の位置づけ

この評価は、vLLM / Ollama / llama.cpp に入る前の、本物寄り decode workload です。

実 backend ではなく合成 workload ですが、attention / KV cache / GQA / projection を入れることで、RMSNorm fusion が LLM decode の中でどこまで効くかをより現実に近い形で見ます。

## まだ言えないこと

- vLLM / Ollama / llama.cpp の tokens/sec が上がるか。
- 実モデル重み、実 tokenizer、scheduler、batching を含む production inference でどれだけ効くか。
- backend 内部の fused attention kernel や paged KV cache で同じ比率になるか。

## 見たいこと

- PyTorch unfused と CUDA / Triton fused の latency 差が context length に対してどう変わるか。
- context length が長くなるほど attention / KV cache が支配的になり、RMSNorm の寄与が薄まるか。
- Nsight Systems で RMSNorm / QKV / attention / output projection の比率を説明できるか。

## 初回結果

summary:

```text
mini_decoder_kv/results/rtx4070/context_sweep_summary.md
```

`preset=qwen_like_2b`, `hidden=2048`, `layers=16`, `heads=16`, `kv_heads=4`, `head_dim=128`, `runs=30`, RTX 4070。

| context len | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 2224.288 | 1570.112 | 1798.608 | 1.42x | 1.24x | CUDA fused |
| 512 | 2070.256 | 1528.784 | 1762.304 | 1.35x | 1.17x | CUDA fused |
| 2048 | 2671.760 | 2292.736 | 2064.736 | 1.17x | 1.29x | Triton fused |

観察:

- attention / KV cache を入れても、RMSNorm fusion の効果は latency に残った。
- context 128/512 では CUDA fused が最速。
- context 2048 では Triton fused が最速。
- context length によって最速実装が変わるため、kernel 単体結果だけでは判断できない。

次に必要な確認:

- Nsight Systems で RMSNorm / QKV projection / attention / output projection の比率を見る。
- context 2048 で Triton が逆転した理由を kernel timeline で見る。

## Nsight Systems 結果

summary:

```text
mini_decoder_kv/results/rtx4070/nsys_summary.md
```

`qwen_like_2b`, `hidden=2048`, `layers=16`, `iters=1`。

| context | implementation | RMSNorm | QKV | KV cache | attention score | softmax | attention value | Wo |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | PyTorch unfused | 29.2% | 13.9% | 14.2% | 14.3% | 8.9% | 9.9% | 9.6% |
| 128 | CUDA fused | 8.6% | 15.9% | 19.6% | 21.9% | 12.1% | 12.5% | 9.5% |
| 128 | Triton fused | 13.8% | 17.8% | 17.8% | 12.6% | 12.8% | 10.9% | 14.2% |
| 2048 | PyTorch unfused | 23.1% | 19.6% | 21.3% | 10.0% | 8.3% | 10.3% | 7.5% |
| 2048 | CUDA fused | 9.0% | 15.4% | 26.2% | 15.0% | 16.2% | 8.6% | 9.5% |
| 2048 | Triton fused | 10.7% | 19.9% | 28.0% | 9.8% | 7.9% | 7.9% | 15.9% |

読み取り:

- PyTorch unfused では residual RMSNorm 側が 23-29% 程度を占める。
- CUDA/Triton fused 後は RMSNorm 側が 9-14% 程度まで下がる。
- fusion 後は KV cache、attention、QKV/Wo projection が支配的になる。
- `context_len=2048` でも fused residual RMSNorm の効果は残るが、次に狙うべき対象は RMSNorm 単体ではなく decoder body 全体になる。

注意:

- この比率は profile runner の NVTX range と CUDA event による区間計測であり、通常 benchmark の絶対 latency とは別に読む。
- 実 backend の paged KV cache / fused attention / scheduler を含む tokens/sec 改善はまだ未確認。
