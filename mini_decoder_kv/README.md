# Mini Decoder KV Evaluation

attention / KV cache 付きの mini decoder で、Fused Residual RMSNorm の効果が本物寄り decode workload に残るかを見る評価です。

## 位置づけ

ここまでの評価では、kernel 単体、mini block、mini decode で効果を確認しました。

このフォルダでは、さらに attention / KV cache を入れて、次の問いに進みます。

- context length が伸びても RMSNorm fusion の効果は残るか。
- attention / KV cache / projection が支配的になり、効果が薄まるか。
- Nsight Systems で RMSNorm / QKV / attention / output projection の比率を説明できるか。

## 評価対象

実 model weight は使わず、Qwen 系に寄せた shape の合成 workload として測ります。

```text
for layer in layers:
    y = residual_rmsnorm(x, residual, norm_weight[layer])
    q = y @ Wq[layer]
    k_new = y @ Wk[layer]
    v_new = y @ Wv[layer]
    K = append(K_cache[layer], k_new)
    V = append(V_cache[layer], v_new)
    attn = softmax(q @ K.T / sqrt(head_dim)) @ V
    out = attn @ Wo[layer]
    residual = x
    x = out
```

比較対象:

| implementation | residual RMSNorm | QKV / attention / Wo |
|---|---|---|
| `pytorch_unfused` | PyTorch add + RMSNorm | PyTorch |
| `cuda_residual_fused` | CUDA fused kernel | PyTorch |
| `triton_residual_fused` | Triton fused kernel | PyTorch |

## Preset

| preset | hidden | layers | heads | kv heads | head dim |
|---|---:|---:|---:|---:|---:|
| `qwen_like_2b` | 2048 | 16 | 16 | 4 | 128 |
| `qwen_like_4b` | 2560 | 24 | 20 | 4 | 128 |

これは実モデル config の厳密再現ではありません。Qwen 系 decoder に近い GQA decode workload として使います。

## 実行

```bash
source .venv/bin/activate
bash mini_decoder_kv/scripts/run_bench.sh --preset qwen_like_2b --layers 2 --context-len 128 --runs 10 --warmup 3 --run-name smoke
```

context sweep:

```bash
bash mini_decoder_kv/scripts/run_bench.sh --preset qwen_like_2b --context-len 128 --runs 30 --warmup 5 --run-name qwen2b-ctx128
bash mini_decoder_kv/scripts/run_bench.sh --preset qwen_like_2b --context-len 512 --runs 30 --warmup 5 --run-name qwen2b-ctx512
bash mini_decoder_kv/scripts/run_bench.sh --preset qwen_like_2b --context-len 2048 --runs 30 --warmup 5 --run-name qwen2b-ctx2048
```

Nsight Systems:

```bash
bash mini_decoder_kv/scripts/run_nsys.sh pytorch_unfused qwen_like_2b 2048 1
bash mini_decoder_kv/scripts/run_nsys.sh cuda_residual_fused qwen_like_2b 2048 1
bash mini_decoder_kv/scripts/run_nsys.sh triton_residual_fused qwen_like_2b 2048 1
```

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

attention / KV cache を含めた本物寄り decode workload でも、fused residual RMSNorm の効果は残りました。ただし context length によって最速実装が変わるため、次に Nsight Systems で RMSNorm / QKV projection / attention / output projection の比率を確認します。

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

PyTorch unfused では RMSNorm 側が 23-29% 程度を占めます。CUDA/Triton fused 後は 9-14% 程度まで下がり、残りの支配要因は KV cache、attention、QKV/Wo projection 側に移ります。

このため、ここで言えることは「RMSNorm fusion は attention / KV cache 付きの decode でも効果が残る。ただし fusion 後の主な改善余地は decoder body 側に移る」です。
