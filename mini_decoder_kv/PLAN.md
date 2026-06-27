# PLAN.md - Mini Decoder KV Evaluation

## 目的

attention / KV cache 付きの mini decoder を作り、Fused Residual RMSNorm の改善が本物寄り decode workload でどこまで残るかを測る。

ここでは vLLM / Ollama に直接入る前に、自分で制御できる小さい decoder で以下を切り分ける。

- RMSNorm 側の短縮が decode latency に残るか。
- context length が伸びたとき、attention / KV cache 側に効果が埋もれるか。
- Nsight Systems で RMSNorm / QKV projection / attention / output projection の比率を見られるか。

## 対象

実 model weight は使わず、Qwen 系に寄せた shape の合成 workload とする。

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

## 比較対象

- `pytorch_unfused`: PyTorch add + RMSNorm
- `cuda_residual_fused`: CUDA fused residual RMSNorm
- `triton_residual_fused`: Triton fused residual RMSNorm

QKV projection、attention、output projection は全実装で PyTorch を使う。

## 初期 preset

まず RTX 4070 で安定して測れる `qwen_like_2b` から始める。

| preset | hidden | layers | heads | kv heads | head dim |
|---|---:|---:|---:|---:|---:|
| `qwen_like_2b` | 2048 | 16 | 16 | 4 | 128 |
| `qwen_like_4b` | 2560 | 24 | 20 | 4 | 128 |

この preset は実モデル config の厳密再現ではなく、Qwen 系 decoder に近い GQA decode workload として扱う。

## 測定 shape

- `tokens=1`
- `context_len=128,512,2048`
- `preset=qwen_like_2b`

余力があれば `qwen_like_4b` も測る。

## 完了条件

- [x] benchmark harness を作る。
- [x] PyTorch unfused / CUDA fused / Triton fused を選択できる。
- [x] correctness check を通す。
- [x] smoke test を実行する。
- [x] context length sweep を実行する。
- [x] Nsight Systems で kernel 比率を見る。
- [x] README / docs に結果を整理する。

## 実行順

1. [x] `bench_decode_kv.py` を作る。
2. [x] `run_bench.sh` を作る。
3. [x] smoke test を `qwen_like_2b context_len=128 layers=2` で通す。
4. [x] `context_len=128,512,2048` を測る。
5. [x] `profile_decode_kv.py` と `run_nsys.sh` を追加する。
6. [x] Nsight Systems 結果をまとめる。

## 初回計測

summary:

```text
mini_decoder_kv/results/rtx4070/context_sweep_summary.md
```

条件:

- RTX 4070
- `preset=qwen_like_2b`
- `hidden=2048`
- `layers=16`
- `heads=16`
- `kv_heads=4`
- `head_dim=128`
- `runs=30`, `warmup=5`

| context len | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 2224.288 | 1570.112 | 1798.608 | 1.42x | 1.24x | CUDA fused |
| 512 | 2070.256 | 1528.784 | 1762.304 | 1.35x | 1.17x | CUDA fused |
| 2048 | 2671.760 | 2292.736 | 2064.736 | 1.17x | 1.29x | Triton fused |

観察:

- attention / KV cache を入れても fused residual RMSNorm の効果は残った。
- context 128/512 では CUDA fused が最速。
- context 2048 では Triton fused が最速。
- context が長くなると attention / KV cache 側の比率が上がる可能性があるため、次は Nsight Systems で比率を見る。

## Nsight Systems

summary:

```text
mini_decoder_kv/results/rtx4070/nsys_summary.md
```

`qwen_like_2b`, `hidden=2048`, `layers=16`, `iters=1` で `context_len=128` と `2048` を測った。

| context | implementation | RMSNorm | QKV | KV cache | attention score | softmax | attention value | Wo |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | PyTorch unfused | 29.2% | 13.9% | 14.2% | 14.3% | 8.9% | 9.9% | 9.6% |
| 128 | CUDA fused | 8.6% | 15.9% | 19.6% | 21.9% | 12.1% | 12.5% | 9.5% |
| 128 | Triton fused | 13.8% | 17.8% | 17.8% | 12.6% | 12.8% | 10.9% | 14.2% |
| 2048 | PyTorch unfused | 23.1% | 19.6% | 21.3% | 10.0% | 8.3% | 10.3% | 7.5% |
| 2048 | CUDA fused | 9.0% | 15.4% | 26.2% | 15.0% | 16.2% | 8.6% | 9.5% |
| 2048 | Triton fused | 10.7% | 19.9% | 28.0% | 9.8% | 7.9% | 7.9% | 15.9% |

観察:

- PyTorch unfused では residual RMSNorm 側が 23-29% 程度を占める。
- CUDA/Triton fused 後は RMSNorm 側が 9-14% 程度まで下がる。
- fusion 後の支配要因は KV cache、attention、QKV/Wo projection 側に移る。
- ここまでで「RMSNorm fusion は本物寄り decode でも効くが、次の改善対象は decoder body 側」と言える。
