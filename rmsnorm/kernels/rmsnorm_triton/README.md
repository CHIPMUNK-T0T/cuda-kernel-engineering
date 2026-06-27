# Triton RMSNorm Implementation Notes

このフォルダには Triton 版 RMSNorm / Fused Residual RMSNorm を置く。

## 次に実装するもの

最初に実装する benchmark 名:

```text
triton_rmsnorm
```

対象演算:

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

入力:

- `x`: `[tokens, hidden]`, FP16
- `weight`: `[hidden]`, FP16
- accumulate: FP32
- output: FP16

## 最初の実装方針

- 1 program が 1 row を処理する。
- hidden dimension は `BLOCK_SIZE = next_power_of_2(hidden)` で読む。
- mask 付き load で hidden size の端を扱う。
- sum of squares は FP32 で reduction する。
- output は FP16 で `y` に書く。

## 初回 smoke test

```bash
bash rmsnorm/scripts/run_bench.sh \
  --tokens 1 \
  --hidden 4096 \
  --runs 20 \
  --warmup 5 \
  --implementations pytorch_eager,cuda_naive,triton_rmsnorm \
  --run-name smoke-triton-rmsnorm
```

## 本計測

```bash
bash rmsnorm/scripts/run_bench.sh \
  --runs 100 \
  --warmup 20 \
  --implementations pytorch_eager,cuda_naive,triton_rmsnorm \
  --run-name baseline-matrix-triton-rmsnorm
```

## 記録

実行後、以下を更新する。

- `rmsnorm/results/rtx4070/runs/<timestamp>-*/`
- `rmsnorm/docs/benchmark_log.md`
- `rmsnorm/README.md`
