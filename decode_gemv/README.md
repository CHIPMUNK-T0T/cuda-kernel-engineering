# Decode GEMV / Small-Batch Linear

vLLM + Qwen3.5 2B の request-window profiling で最大だった `cuBLAS GEMV` を、次の最適化テーマとして扱うフォルダです。

目的は、いきなり cuBLAS に勝つことではありません。LLM decode の `tokens=1` small-batch linear projection がなぜ支配的になるのか、consumer GPU 上で custom CUDA / Triton がどこまで迫れるのか、どの shape なら余地があるのかを実測で切り分けることです。

## Why This Target

前段の `backend_compare/` で、Qwen3.5 2B の vLLM request window を `cuda_gpu_trace.csv` から再集計したところ、次の傾向でした。

| family | share |
|---|---:|
| cuBLAS GEMV | 86.788% |
| PyTorch copy / cast | 3.314% |
| PyTorch elementwise math | 3.284% |
| vLLM SwiGLU `act_and_mul_kernel` | 0.407% |

RMSNorm や SwiGLU は、vLLM request path では主要ボトルネックではありませんでした。特に SwiGLU は vLLM 側ですでに dedicated kernel 化されています。

そのため、次に tokens/sec へ近い問いを立てるなら、decode 中の small-batch linear / GEMV を見るのが最も自然です。

## Target Operation

まずは LLM decode の linear projection を単純化して扱います。

```text
y = x @ W
```

条件:

- `x`: `[tokens, in_features]`
- `W`: `[in_features, out_features]`
- `tokens`: decode では主に `1`
- dtype: `bf16` / `fp16`

QKV / Wo / MLP projection のような実際の decoder block に近い shape を、段階的に増やします。

## Comparison

最初の比較対象:

| implementation | role |
|---|---|
| PyTorch `matmul` / `linear` | high-level baseline |
| cuBLAS path | practical library baseline |
| Triton GEMV / matmul | rapid custom kernel baseline |
| CUDA C++ GEMV | low-level custom implementation |

Current status:

| implementation | status |
|---|---|
| PyTorch `matmul` | done |
| PyTorch `linear` | done |
| Triton GEMV / matmul | first measurement done |
| CUDA C++ GEMV | later |

Baseline summary:

```text
decode_gemv/results/rtx4070/baseline_summary.md
```

Initial read:

- `tokens=1` では `torch_linear` が全 shape で最速。
- `tokens=1` の median latency は `torch_matmul 78.800 us`、`torch_linear 33.264 us`。
- custom Triton / CUDA は、まず `tokens=1` の `torch_linear` path にどこまで迫れるかを見る。

Triton first result:

- 単純 Triton GEMV は、現時点では `torch_linear` に未勝利。
- 小さい output width では `torch_linear` の `1.75-2.53x` 程度遅い。
- 大きい output width では差が `1.07-1.09x` まで縮まる。
- `tokens=1, in=2048, out=8192` では `torch_matmul` より速く、custom kernel の余地は見える。
- summary: `decode_gemv/results/rtx4070/triton_summary.md`

Nsight first result:

- `torch_linear` は cuBLAS GEMV に落ちている。
- cuBLAS / Triton とも DRAM throughput が `84-92%` で、decode GEMV は memory-throughput limited。
- `1x4096x11008` では Triton `206.56 us`、cuBLAS `203.68 us` まで近づいた。
- ただし Triton は compute throughput が低く、cuBLAS の方が hardware utilization は良い。
- summary: `decode_gemv/results/rtx4070/nsight_compare.md`

## Triton Tuning

Sweep `BLOCK_K` / `BLOCK_N` for representative decode shapes:

```bash
bash decode_gemv/scripts/tune_triton.sh
```

Default tuning matrix:

- shape: `tokens=1`, `in_features=2048,4096`, `out_features=8192,11008`
- `BLOCK_K=32,64,128`
- `BLOCK_N=32,64,128`

Results are written to:

```text
decode_gemv/results/rtx4070/triton_tuning/
```

First tuning result:

- best config was `BLOCK_K=128, BLOCK_N=32`
- `1x2048x8192`: tuned Triton `41.104 us`, `1.868x` vs `torch_linear`
- larger shapes stayed close to cuBLAS but did not clearly beat it
- summary: `decode_gemv/results/rtx4070/triton_tuning_summary.md`

Nsight check for tuned `1x2048x8192`:

- `BLOCK_K=128, BLOCK_N=32` improved Nsight duration from `81.54 us` to `78.08 us` versus the initial Triton mapping.
- DRAM throughput rose from `84.47%` to `90.78%`.
- achieved occupancy rose from `23.20%` to `46.29%`.
- This supports the interpretation that the tuned shape improves memory streaming and parallel decomposition, not just Python-level benchmark noise.

Profile a tuned Triton config with Nsight Compute:

```bash
  GEMV_BLOCK_K=128 GEMV_BLOCK_N=32 \
  bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 2048 8192
```

## Projection Block Check

単体 GEMV で見えた改善が、複数 projection のまとまりでも残るかを確認します。

対象は decoder layer 内の linear projection に近い構成です。

| projection | shape |
|---|---|
| QKV | `hidden -> 3 * hidden` |
| Wo | `hidden -> hidden` |
| MLP gate/up | `hidden -> 2 * intermediate` |
| MLP down | `intermediate -> hidden` |

Run:

```bash
bash decode_gemv/scripts/run_projection_block.sh
```

Output:

```text
decode_gemv/results/rtx4070/projection_block/
```

This benchmark still excludes attention, KV cache, and activation kernels.
The purpose is to check whether the standalone GEMV result survives when several projection kernels are executed as a block.

First projection-block result:

- `torch_linear` is faster for every measured block configuration.
- `triton_tuned / torch_linear` was `1.08-1.28x`, so the tuned Triton path is still slower at block level.
- The standalone win does not automatically carry over to mixed QKV / Wo / MLP projection shapes.
- summary: `decode_gemv/results/rtx4070/projection_block_summary.md`

Next, split the block by projection type:

```bash
bash decode_gemv/scripts/run_projection_types.sh
```

Output:

```text
decode_gemv/results/rtx4070/projection_types/
```

This identifies whether QKV, Wo, MLP gate/up, or MLP down is responsible for the block-level loss before choosing per-shape Triton tuning or CUDA C++.

Projection-type result (deduped):

- `triton_tuned` wins Wo only in the deduped matrix: `0.879-0.888x` triton / torch.
- QKV is slower with the current mapping: `1.146-1.370x`
- MLP projections are consistently slower with the current Triton mapping:
  - MLP up: `1.169-1.293x`
  - MLP down: `1.400-2.013x`
- The stable result is that the global Triton config helps Wo but does not generalize to QKV or MLP projections.
- summary: `decode_gemv/results/rtx4070/projection_type_summary.md`

## Projection-Type Tuning

The next step is to tune Triton per projection type instead of using one global config.

Default target:

1. Wo: confirm whether the current win is stable across block settings.
2. MLP down: focus on the clearest loss case and check whether it is a config mismatch or a kernel-design limit.

Run:

```bash
bash decode_gemv/scripts/tune_projection_types.sh
```

Default sweep:

- projections: `wo,mlp_down`
- shape: `tokens=1`, `hidden=2048,4096`, `intermediate=8192,11008`
- `BLOCK_K=32,64,128,256`
- `BLOCK_N=16,32,64,128`

Output:

```text
decode_gemv/results/rtx4070/projection_type_tuning/
decode_gemv/results/rtx4070/projection_type_tuning_summary.md
```

To include all projection types:

```bash
GEMV_PROJECTIONS=qkv,wo,mlp_up,mlp_down \
  bash decode_gemv/scripts/tune_projection_types.sh
```

First projection-type tuning result:

- Wo is shape-dependent, not universally stable.
  - `1x4096x4096`: Triton wins, `41.872 us` vs torch `56.032 us`, `0.747x` triton / torch.
  - `1x2048x2048`: Triton loses, `22.416 us` vs torch `14.208 us`, `1.578x` triton / torch.
- MLP down improves in one shape but does not generalize yet.
  - `1x8192x2048`: Triton wins, `49.792 us` vs torch `55.296 us`, `0.900x` triton / torch.
  - Other MLP down shapes still lose: `1.199-1.513x` triton / torch.
- Best configs are not the old global `BLOCK_K=128, BLOCK_N=32`; tuning selected `BLOCK_K=256, BLOCK_N=16/32` for the winning Wo / small MLP down cases.
- Current read: projection-specific tuning has real signal, but a single Triton GEMV mapping is not enough for all decoder projections.
- summary: `decode_gemv/results/rtx4070/projection_type_tuning_summary.md`

Aggregate projection-type tuning result:

- Best Triton configs win `3 / 12` measured projection shapes.
- Wins:
  - `qkv 1x2048x6144`: `0.734x` triton / torch.
  - `wo 1x4096x4096`: `0.747x` triton / torch.
  - `mlp_down 1x8192x2048`: `0.900x` triton / torch.
- Losses remain in larger QKV and most MLP up/down shapes.

Projection block with per-projection configs:

- Per-projection Triton improves every measured block shape versus fixed `BLOCK_K=128, BLOCK_N=32`.
- Block-level `triton_projection_tuned / torch_linear` is still `1.056-1.146x`, so it has not beaten cuBLAS at block level yet.
- This is the important whole-system read: per-shape tuning helps, but LLM projection-block speedup likely needs either a stronger CUDA GEMV implementation or fusion around projection outputs.
- summary: `decode_gemv/results/rtx4070/projection_block_per_projection_summary.md`

## Initial Question

最初に答える問い:

1. `tokens=1` の decode shape で、PyTorch / cuBLAS の latency はどの程度か
2. custom CUDA / Triton は cuBLAS にどこまで迫れるか
3. shape を変えると、GEMV が支配的になる条件はどう変わるか
4. cuBLAS に勝てない場合、どの要因が支配しているか

## Claim Boundary

このテーマの初期段階では、まだ vLLM の tokens/sec 改善は主張しません。

まずは standalone benchmark で decode GEMV の性質を確認し、次に mini decode / mini block へ戻します。

## Run

```bash
bash decode_gemv/scripts/run_bench.sh
```

Triton first measurement:

```bash
bash decode_gemv/scripts/run_triton_bench.sh
```

Nsight Compute comparison:

```bash
bash decode_gemv/scripts/run_nsight.sh torch_linear 1 2048 8192
bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 2048 8192

bash decode_gemv/scripts/run_nsight.sh torch_linear 1 4096 11008
bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 4096 11008
```

If Nsight Compute reports `ERR_NVGPUCTRPERM`, run with sudo while preserving the CUDA and venv PATH:

```bash
sudo env "PATH=$PWD/.venv/bin:/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH" \
  UV_CACHE_DIR=.uv-cache \
  NCU_SET=basic \
  NCU_TARGET_PROCESSES=application-only \
  bash decode_gemv/scripts/run_nsight.sh torch_linear 1 2048 8192
```

Repeat the same command for `triton_gemv` and the second shape.

Small smoke test:

```bash
bash decode_gemv/scripts/run_bench.sh --tokens 1 --in-features 256 --out-features 512 --runs 5 --warmup 2 --no-record-run
```
