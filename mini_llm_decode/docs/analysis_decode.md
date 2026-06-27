# Mini LLM Decode Analysis

## 現在の位置づけ

この評価は、本物の LLM backend の tokens/sec を測る前段階です。

RMSNorm kernel 単体では大きな改善が見え、mini transformer block では projection / GEMM によって改善幅が薄まることを確認しました。次に mini decode workload で複数 layer にしたとき、その差分が積み上がるかを確認します。

## まだ言えないこと

- vLLM / llama.cpp / 実 model backend の tokens/sec が上がるか。
- attention / KV cache / MLP を含む実 LLM block 全体でどれだけ効くか。
- scheduler、batching、KV cache layout まで含めた production inference での効果。

## 言えるようにしたいこと

- residual RMSNorm fusion は、decode 風 workload の layer stack でも latency に寄与するか。
- GEMM が支配的になるほど効果が薄まるか。
- PyTorch unfused / CUDA fused / Triton fused のどれが decode 条件で有利か。

## 初回結果

summary:

```text
mini_llm_decode/results/rtx4070/layers_sweep_summary.md
```

`tokens=1`, `hidden=4096`, `dtype=float16`, `runs=50`, RTX 4070。

Distinct projection weights:

| layers | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 788.528 | 633.856 | 632.528 | 1.24x | 1.25x | Triton fused |
| 16 | 1542.112 | 1246.128 | 1244.160 | 1.24x | 1.24x | Triton fused |
| 32 | 3071.488 | 2488.848 | 2474.560 | 1.23x | 1.24x | Triton fused |

Shared projection weights:

| layers | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 780.448 | 627.712 | 630.640 | 1.24x | 1.24x | CUDA fused |
| 16 | 1540.576 | 1242.640 | 1238.512 | 1.24x | 1.24x | Triton fused |
| 32 | 3083.968 | 2484.224 | 2474.336 | 1.24x | 1.25x | Triton fused |

観察:

- fused residual RMSNorm の改善は、複数 layer の mini decode 風 workload でも残った。
- 速度差は約1.24xで、RMSNorm 単体よりは小さい。
- 1 block 評価と同じく、projection / GEMM が入ることで改善幅は薄まる。
- それでも layer stack に戻して改善が消えなかったため、kernel 単体評価だけで終わっていない点を記事・READMEで説明できる。

制約:

- shared projection weight は実 LLM の layer ごとに異なる重み配置とは違う。distinct projection weight 条件は測定済み。
- attention / KV cache / MLP はまだ含めていない。
- production backend の tokens/sec 改善はまだ未測定。

## Nsight Systems

summary:

```text
mini_llm_decode/results/rtx4070/nsys_summary.md
```

`tokens=1`, `hidden=4096`, `layers=32`, distinct projection weights, `iters=1`。

| implementation | RMSNorm side us | matmul us | total GPU kernel us | RMSNorm side share | matmul share |
|---|---:|---:|---:|---:|---:|
| PyTorch unfused | 495.202 | 2417.168 | 2912.370 | 17.0% | 83.0% |
| CUDA fused | 99.521 | 2719.312 | 2818.833 | 3.5% | 96.5% |
| Triton fused | 50.528 | 2411.375 | 2461.903 | 2.1% | 97.9% |

この結果から、mini decode の約1.23xから1.25xの差は、RMSNorm側の kernel 数と時間が減ったこととして説明できる。PyTorch unfused は 32 layer で RMSNorm 側が 352 kernel instance に分かれる一方、CUDA / Triton fused は 32 kernel instance まで減る。

一方で、fusion 後は matmul が GPU kernel 時間の 96% 以上を占める。つまり、RMSNorm kernel を速くしても、全体の上限は matmul によって抑えられる。
