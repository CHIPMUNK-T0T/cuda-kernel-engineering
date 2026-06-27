# Mini LLM Decode Evaluation

RMSNorm kernel 単体、mini transformer block 評価の次に、複数 layer の decode 風 workload で fused residual RMSNorm の寄与を見るための評価です。

## 評価方針

対象は attention / KV cache をまだ含まない mini decode workload です。

```text
for layer in layers:
    y = residual_rmsnorm(x, residual, norm_weight[layer])
    out = y @ projection_weight
    residual = x
    x = out
```

比較対象:

| implementation | residual RMSNorm | projection |
|---|---|---|
| `pytorch_unfused` | PyTorch add + RMSNorm | PyTorch matmul |
| `cuda_residual_fused` | CUDA fused kernel | PyTorch matmul |
| `triton_residual_fused` | Triton fused kernel | PyTorch matmul |

この評価は本物の LLM backend ではありません。目的は、kernel 単体で速かった処理が、複数 layer の decode 風 workload でも latency に残るかを切り分けることです。

既定では projection weight は layer 間で共有します。これは VRAM 使用量を抑えて RTX 4070 上で測りやすくするためです。実 LLM のように layer ごとに別 weight を使う場合は `--distinct-projection-weights` を指定します。初回結果では、空きメモリを確保したうえで distinct 条件も測定しています。

## 実行

```bash
source .venv/bin/activate
bash mini_llm_decode/scripts/run_bench.sh --tokens 1 --hidden 4096 --layers 2 --runs 10 --warmup 3 --run-name smoke
```

layers sweep:

```bash
bash mini_llm_decode/scripts/run_bench.sh --tokens 1 --hidden 4096 --layers 8 --distinct-projection-weights --runs 50 --warmup 10 --run-name decode-distinct-layers8
bash mini_llm_decode/scripts/run_bench.sh --tokens 1 --hidden 4096 --layers 16 --distinct-projection-weights --runs 50 --warmup 10 --run-name decode-distinct-layers16
bash mini_llm_decode/scripts/run_bench.sh --tokens 1 --hidden 4096 --layers 32 --distinct-projection-weights --runs 50 --warmup 10 --run-name decode-distinct-layers32
```

Nsight Systems:

```bash
bash mini_llm_decode/scripts/run_nsys.sh pytorch_unfused 32 4096 1 distinct
bash mini_llm_decode/scripts/run_nsys.sh cuda_residual_fused 32 4096 1 distinct
bash mini_llm_decode/scripts/run_nsys.sh triton_residual_fused 32 4096 1 distinct
```

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

mini decode 風の複数 layer workload でも、CUDA / Triton fused residual RMSNorm は PyTorch unfused より速いです。distinct / shared のどちらでも約1.23xから1.25xで、RMSNorm単体評価ほどの倍率ではありません。projection / GEMM が入ることで、kernel 単体の改善は end-to-end 風 latency の一部として効くためです。

## 読み方

ここで見たいことは、次の2点です。

- layer 数が増えたときに residual RMSNorm fusion の効果が積み上がるか。
- projection / GEMM が支配的になり、kernel 単体の改善がどこまで薄まるか。

ここで効果が小さい場合でも、block 評価と同じく「LLM 全体では GEMM / attention / KV cache が支配的になる」ことを示す材料になります。

今回の初回結果では、効果は小さくならず約1.24xで残りました。これは「LLM を構成要素へ分解し、kernel を置換し、block と mini decode に戻して実測した」という説明に使えます。ただし、まだ vLLM / llama.cpp / 実 model backend の tokens/sec 改善を示すものではありません。

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

PyTorch unfused は 32 layer で RMSNorm 側が 352 kernel instance に分かれます。CUDA / Triton fused は 1 layer あたり 1 kernel になり、32 layer で 32 instance です。これにより RMSNorm 側の比率は大きく下がり、残りは matmul が支配的になります。
