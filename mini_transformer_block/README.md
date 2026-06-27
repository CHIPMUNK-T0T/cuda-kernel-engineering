# Mini Transformer Block Evaluation

RMSNorm kernel 単体で得た改善が、LLM block に近い処理へ入ったときにどこまで残るかを測るための評価です。

## 評価方針

対象は最小の block です。

```text
y = residual_rmsnorm(x, residual, norm_weight)
out = y @ projection_weight
```

比較対象:

| implementation | residual RMSNorm | projection |
|---|---|---|
| `pytorch_unfused` | PyTorch add + RMSNorm | PyTorch matmul |
| `cuda_residual_fused` | CUDA fused kernel | PyTorch matmul |
| `triton_residual_fused` | Triton fused kernel | PyTorch matmul |

Linear / GEMM は同じ PyTorch 実装に固定し、差分を residual RMSNorm に寄せます。

## 実行

```bash
source .venv/bin/activate
bash mini_transformer_block/scripts/run_bench.sh --tokens 1 --hidden 4096 --runs 50 --warmup 10 --run-name smoke
```

default matrix:

```bash
bash mini_transformer_block/scripts/run_bench.sh --runs 50 --warmup 10 --run-name block-matrix
```

Nsight Systems:

```bash
bash mini_transformer_block/scripts/run_nsys.sh cuda_residual_fused 1 4096 1
bash mini_transformer_block/scripts/run_nsys.sh cuda_residual_fused 512 8192 1
```

## 読み方

この評価で見たいことは、kernel 単体の speedup そのものではありません。

- decode shape で fixed overhead 削減が block latency に残るか
- prefill shape で GEMM が支配的になり、RMSNorm 改善が薄まるか
- CUDA fused と Triton fused の差が block 全体ではどう見えるか

ここで効果が小さくても失敗ではありません。LLM 推論では GEMM、attention、KV cache などが支配的になるため、kernel 単体の改善が全体にどれだけ効くかを切り分けること自体が目的です。

## 初回結果

run:

```text
mini_transformer_block/results/rtx4070/runs/20260620-103339-block-matrix-initial-v2/
```

`runs=20`, `warmup=5`, `dtype=float16`, RTX 4070。

| tokens | hidden | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 118.656 | 105.856 | 130.480 | 1.12x |
| 1 | 8192 | 327.712 | 293.360 | 301.120 | 1.12x |
| 512 | 4096 | 431.104 | 352.256 | 357.376 | 1.22x |
| 512 | 8192 | 1435.088 | 1256.960 | 1258.496 | 1.14x |

初回結果では、CUDA fused residual RMSNorm は block 全体でも PyTorch unfused より速いです。ただし RMSNorm 単体評価ほどの倍率ではありません。projection の GEMM を含めると、RMSNorm 改善は block latency の一部として効くためです。

この結果は「LLM 全体が速くなる」と断言するものではありません。kernel 単体で見えた改善が、GEMM を含む block 評価ではどの程度残るかを切り分けた段階です。

## Nsight Systems

`iters=1` で block 内の GPU kernel 時間を見た結果です。

| implementation | tokens | hidden | RMSNorm side us | matmul us | RMSNorm side share |
|---|---:|---:|---:|---:|---:|
| PyTorch unfused | 1 | 4096 | 17.082 | 75.492 | 18.5% |
| CUDA fused | 1 | 4096 | 3.040 | 62.752 | 4.6% |
| Triton fused | 1 | 4096 | 2.080 | 80.006 | 2.5% |
| PyTorch unfused | 512 | 8192 | 245.220 | 1200.825 | 17.0% |
| CUDA fused | 512 | 8192 | 56.868 | 1201.361 | 4.5% |
| Triton fused | 512 | 8192 | 50.114 | 1201.987 | 4.0% |

PyTorch unfused は add / pow / reduce / rsqrt / mul / copy などが複数 kernel に分かれます。CUDA / Triton fused は residual add + RMSNorm が 1 kernel になり、block 内の GPU 時間では matmul が約95%前後を占める状態になります。

## Projection Sweep

GEMM を軽くしたときに fused residual RMSNorm の寄与が増えるかを確認しました。

| mode | tokens | hidden | out features | PyTorch unfused us | CUDA fused us | Triton fused us | best | CUDA vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| decode | 1 | 4096 | 512 | 47.984 | 15.184 | 76.608 | CUDA fused | 3.16x |
| decode | 1 | 4096 | 1024 | 49.152 | 23.312 | 32.496 | CUDA fused | 2.11x |
| decode | 1 | 4096 | 4096 | 114.688 | 104.224 | 92.048 | Triton fused | 1.10x |
| prefill | 512 | 8192 | 512 | 348.160 | 123.904 | 123.840 | Triton fused | 2.81x |
| prefill | 512 | 8192 | 1024 | 411.648 | 214.864 | 232.448 | CUDA fused | 1.92x |
| prefill | 512 | 8192 | 8192 | 1445.888 | 1184.720 | 1154.560 | Triton fused | 1.22x |

`out_features=512` の Nsight Systems では、PyTorch unfused の RMSNorm 側が GPU kernel 時間の 58.7% / 71.5% を占めました。fusion 後は CUDA で 21.4% / 40.6%、Triton で 13.0% / 34.8% まで下がります。

この結果から、GEMM が重い条件では fusion の効果は薄まり、GEMM が軽い条件では効果が大きく見えることが確認できます。
