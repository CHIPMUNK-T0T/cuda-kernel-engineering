# PLAN.md - Mini LLM Decode Evaluation

## 目的

RMSNorm / Fused Residual RMSNorm の改善が、1 block を超えて複数 layer の decode 風 workload に入ったときにどこまで残るかを測る。

ここでの見せ方は「実 LLM 全体が速くなった」ではなく、kernel 単体、block 単体、mini decode の順に戻して、改善がどこで残り、どこで薄まるかを実測すること。

## 対象

最初は attention / KV cache を入れず、decode 1 token の複数 layer projection workload として切り分ける。

```text
for layer in layers:
    y = residual_rmsnorm(x, residual, norm_weight[layer])
    out = y @ projection_weight
    residual = x
    x = out
```

この構成にする理由:

- residual RMSNorm を layer 数ぶん繰り返し、kernel 改善が積み上がるかを見られる。
- projection は全実装で同じ PyTorch matmul に固定し、差分を residual RMSNorm に寄せられる。
- attention / KV cache / MLP を入れる前に、decode 風 end-to-end latency の下限評価として扱える。

既定では projection weight は layer 間で共有する。これは RTX 4070 上で別プロセスが動いていても測れるように VRAM 使用量を抑えるためで、実 LLM の重み配置そのものではない。必要なら `--distinct-projection-weights` で layer ごとに別 weight を使う。初回結果では、空きメモリを確保して distinct 条件も測定済み。

## 比較対象

- `pytorch_unfused`: PyTorch の `x + residual` + RMSNorm + `torch.matmul`
- `cuda_residual_fused`: CUDA fused residual RMSNorm + `torch.matmul`
- `triton_residual_fused`: Triton fused residual RMSNorm + `torch.matmul`

## 測定 shape

初回:

- `tokens=1`
- `hidden=4096`
- `layers=8,16,32`

必要に応じて:

- `hidden=8192`
- `tokens=1` 固定の decode
- `tokens=8,32` の micro-batch decode
- `--distinct-projection-weights` による layer ごとの projection weight

## 完了条件

- [x] benchmark harness を作る。
- [x] PyTorch unfused / CUDA fused / Triton fused を選択できる。
- [x] correctness check を通す。
- [x] smoke test を実行する。
- [x] layers sweep を実行する。
- [x] 結果を README / docs に整理する。
- [x] Nsight Systems で layer 内の RMSNorm 側と matmul 側の比率を見る。

## 実行順

1. [x] `bench_decode.py` を作る。
2. [x] `run_bench.sh` を作る。
3. [x] smoke test を `tokens=1 hidden=4096 layers=2` で通す。
4. [x] `layers=8,16,32` を測る。
5. [x] block 評価との差分をまとめる。
6. [x] Nsight Systems を追加する。

## 初回計測

summary:

```text
mini_llm_decode/results/rtx4070/layers_sweep_summary.md
```

条件:

- RTX 4070
- `tokens=1`
- `hidden=4096`
- `dtype=float16`
- `runs=50`, `warmup=10`

distinct projection weights:

| layers | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 788.528 | 633.856 | 632.528 | 1.24x | 1.25x | Triton fused |
| 16 | 1542.112 | 1246.128 | 1244.160 | 1.24x | 1.24x | Triton fused |
| 32 | 3071.488 | 2488.848 | 2474.560 | 1.23x | 1.24x | Triton fused |

shared projection weights:

| layers | PyTorch unfused us | CUDA fused us | Triton fused us | CUDA vs PyTorch | Triton vs PyTorch | best |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 780.448 | 627.712 | 630.640 | 1.24x | 1.24x | CUDA fused |
| 16 | 1540.576 | 1242.640 | 1238.512 | 1.24x | 1.24x | Triton fused |
| 32 | 3083.968 | 2484.224 | 2474.336 | 1.24x | 1.25x | Triton fused |

観察:

- mini decode 風の複数 layer workload でも、fused residual RMSNorm の改善は latency に残った。
- speedup は約1.24xで、RMSNorm 単体ほど大きくはない。
- block 評価と同じく projection / GEMM によって改善幅は薄まる。
- ただし layer 数を増やしても改善が消えていないため、「kernel 単体から block、mini decode へ戻して寄与を確認した」というストーリーには使える。

## Nsight Systems 結果

summary:

```text
mini_llm_decode/results/rtx4070/nsys_summary.md
```

条件:

- RTX 4070
- `tokens=1`
- `hidden=4096`
- `layers=32`
- distinct projection weights
- `iters=1`

| implementation | RMSNorm side us | matmul us | total GPU kernel us | RMSNorm side share | matmul share |
|---|---:|---:|---:|---:|---:|
| PyTorch unfused | 495.202 | 2417.168 | 2912.370 | 17.0% | 83.0% |
| CUDA fused | 99.521 | 2719.312 | 2818.833 | 3.5% | 96.5% |
| Triton fused | 50.528 | 2411.375 | 2461.903 | 2.1% | 97.9% |

観察:

- PyTorch unfused は 32 layer で RMSNorm 側が 352 kernel instance に分かれる。
- CUDA / Triton fused は 32 layer で RMSNorm 側が 32 kernel instance になる。
- fusion 後は matmul が 96% 以上を占めるため、mini decode 全体の speedup は約1.23xから1.25xに留まる。
