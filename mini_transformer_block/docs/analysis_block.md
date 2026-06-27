# Mini Transformer Block Analysis

## 目的

RMSNorm / Fused Residual RMSNorm の kernel 単体評価を、LLM block に近い形へ接続する。

## 現時点の仮説

- decode 寄りの小さい shape では、fused residual RMSNorm の fixed overhead 削減が block latency に残る可能性がある。
- prefill 寄りの大きい shape では、projection の GEMM が支配的になり、RMSNorm 単体の改善は block 全体では薄まる可能性がある。
- 効果が薄まった場合も、LLM 推論全体でどこが支配的かを説明する材料になる。

## 初回結果

run:

```text
mini_transformer_block/results/rtx4070/runs/20260620-103339-block-matrix-initial-v2/
```

条件:

- `runs=20`
- `warmup=5`
- `dtype=float16`
- projection は全実装で PyTorch `matmul`

| tokens | hidden | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 118.656 | 105.856 | 130.480 | 1.12x |
| 1 | 8192 | 327.712 | 293.360 | 301.120 | 1.12x |
| 512 | 4096 | 431.104 | 352.256 | 357.376 | 1.22x |
| 512 | 8192 | 1435.088 | 1256.960 | 1258.496 | 1.14x |

## 観察

- CUDA fused residual RMSNorm は、projection を含む最小 block でも PyTorch unfused より速い。
- ただし RMSNorm 単体ほどの speedup は残らない。block 全体では projection の GEMM が大きくなり、RMSNorm の改善は全体 latency の一部に薄まる。
- Triton fused は block 全体では CUDA fused と近いが、今回の初回 matrix では CUDA fused が安定して速い。
- prefill shape の `max_abs_error` は 0.125 / 0.25 まで出るが、`relative_l2_error` は `1.65e-05` から `3.03e-05` 程度。matmul 後の絶対誤差は増幅されるため、最大相対誤差だけでなく L2 相対誤差も併記する。

## 現時点で言えること

RMSNorm kernel 単体の改善は、LLM block に近い処理へ入れても一部は残る。ただし、Linear / GEMM を含むと改善幅は小さくなる。

したがって、ここでの強い主張は「LLM が爆速」ではなく、「kernel 単体改善を block 評価へ接続し、どこまで効くかを実測で切り分けた」である。

## 次に見ること

- `out_features` を小さくした条件を追加し、GEMM が軽い場合に RMSNorm 改善がどれだけ残るかを見る。
- attention / KV cache を含む mini decode block へ進む前に、projection が支配的な条件と支配的でない条件を分ける。

## Nsight Systems

目的:

- block 内で residual RMSNorm 側と matmul 側の GPU kernel 時間比率を見る。
- benchmark latency で見えた「block 全体では改善幅が薄まる」理由を確認する。

取得した profile:

```text
mini_transformer_block/results/rtx4070/nsys/20260620-105025-pytorch_unfused-tokens1-hidden4096-out4096/
mini_transformer_block/results/rtx4070/nsys/20260620-104609-cuda_residual_fused-tokens1-hidden4096-out4096/
mini_transformer_block/results/rtx4070/nsys/20260620-104716-triton_residual_fused-tokens1-hidden4096-out4096/
mini_transformer_block/results/rtx4070/nsys/20260620-105024-pytorch_unfused-tokens512-hidden8192-out8192/
mini_transformer_block/results/rtx4070/nsys/20260620-104714-cuda_residual_fused-tokens512-hidden8192-out8192/
mini_transformer_block/results/rtx4070/nsys/20260620-104858-triton_residual_fused-tokens512-hidden8192-out8192/
```

`cuda_gpu_kern_sum.csv` から見た GPU kernel 時間:

| implementation | tokens | hidden | RMSNorm side us | matmul us | RMSNorm side share | note |
|---|---:|---:|---:|---:|---:|---|
| PyTorch unfused | 1 | 4096 | 17.082 | 75.492 | 18.5% | add / pow / reduce / rsqrt / mul / copy が複数 kernel |
| CUDA fused | 1 | 4096 | 3.040 | 62.752 | 4.6% | fused residual RMSNorm 1 kernel |
| Triton fused | 1 | 4096 | 2.080 | 80.006 | 2.5% | fused residual RMSNorm 1 kernel |
| PyTorch unfused | 512 | 8192 | 245.220 | 1200.825 | 17.0% | unfused RMSNorm 側が複数 kernel |
| CUDA fused | 512 | 8192 | 56.868 | 1201.361 | 4.5% | fused residual RMSNorm 1 kernel |
| Triton fused | 512 | 8192 | 50.114 | 1201.987 | 4.0% | fused residual RMSNorm 1 kernel |

観察:

- PyTorch unfused では、RMSNorm 側が複数の小さい CUDA kernel に分解される。
- CUDA / Triton fused では residual add + RMSNorm が 1 kernel になり、RMSNorm 側の GPU kernel 時間は PyTorch unfused より大きく減る。
- fused 後の block は matmul が約95%前後を占める。したがって、RMSNorm kernel 単体を高速化しても、projection を含む block 全体では speedup が 1.1x から 1.2x 程度に薄まる。
- これは「意味がない」ではなく、「GEMM を含めると効果の上限が決まる」という結果。LLM 全体へ進む前に、効く範囲を切り分けられている。

次に見ること:

- `out_features` を小さくして、GEMM が軽い条件で RMSNorm fusion の効果がどこまで戻るかを見る。
- attention / KV cache を含む decode block では、RMSNorm 以外の支配要因がさらに増える可能性があるため、tokens/sec 主張はその後に回す。

## Projection Sweep

目的:

- projection の `out_features` を小さくして GEMM を軽くする。
- GEMM が軽いほど RMSNorm fusion の効果が block latency に戻るかを確認する。

結果:

```text
mini_transformer_block/results/rtx4070/projection_sweep_summary.md
```

latency:

| mode | tokens | hidden | out features | PyTorch unfused us | CUDA fused us | Triton fused us | best | CUDA vs PyTorch | Triton vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| decode | 1 | 4096 | 512 | 47.984 | 15.184 | 76.608 | CUDA fused | 3.16x | 0.63x |
| decode | 1 | 4096 | 1024 | 49.152 | 23.312 | 32.496 | CUDA fused | 2.11x | 1.51x |
| decode | 1 | 4096 | 4096 | 114.688 | 104.224 | 92.048 | Triton fused | 1.10x | 1.25x |
| prefill | 512 | 8192 | 512 | 348.160 | 123.904 | 123.840 | Triton fused | 2.81x | 2.81x |
| prefill | 512 | 8192 | 1024 | 411.648 | 214.864 | 232.448 | CUDA fused | 1.92x | 1.77x |
| prefill | 512 | 8192 | 8192 | 1445.888 | 1184.720 | 1154.560 | Triton fused | 1.22x | 1.25x |

Nsight Systems, `out_features=512`:

| mode | implementation | tokens | hidden | out features | RMSNorm side us | matmul side us | RMSNorm side share |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | PyTorch unfused | 1 | 4096 | 512 | 17.344 | 12.224 | 58.7% |
| decode | CUDA fused | 1 | 4096 | 512 | 3.648 | 13.376 | 21.4% |
| decode | Triton fused | 1 | 4096 | 512 | 1.952 | 13.024 | 13.0% |
| prefill | PyTorch unfused | 512 | 8192 | 512 | 244.068 | 97.058 | 71.5% |
| prefill | CUDA fused | 512 | 8192 | 512 | 64.417 | 94.434 | 40.6% |
| prefill | Triton fused | 512 | 8192 | 512 | 51.393 | 96.225 | 34.8% |

観察:

- `out_features=hidden` では matmul が重く、fusion の効果は 1.1x から 1.25x 程度に薄まる。
- `out_features=512` では GEMM が軽くなり、RMSNorm 側の比率が上がる。decode では CUDA fused が PyTorch unfused に対して 3.16x、prefill では CUDA / Triton fused が約2.81x。
- Nsight Systems でも、PyTorch unfused の `out_features=512` では RMSNorm 側が GPU kernel 時間の 58.7% / 71.5% を占める。fusion 後は 13.0% から 40.6% まで下がる。
- つまり、fusion は「常に LLM 全体を大きく速くする」ものではなく、周辺の GEMM / attention / KV cache がどれだけ重いかで見える効果が変わる。

この結果により、次の LLM 風 decode 評価で効果が小さくても説明できる。GEMM が支配的な条件では効果が薄まり、RMSNorm 側の比率が高い条件では効果が大きく出る。
