# Decode Projection Fusion

vLLM + Qwen3.5-2B の decode request を Nsight で分解し、
GemmaRMSNorm 周辺の PyTorch native decomposition を Triton fused kernel に差し替えた検証です。

結果として、`stream=True` の decode-heavy workload で TPOT / ITL / decode tokens/s が改善しました。

| max tokens | TPOT | decode tokens/s | total latency |
|---:|---:|---:|---:|
| 128 | `10.615 -> 8.919 ms` | `94.214 -> 112.129` | `15.88%` reduction |
| 512 | `10.557 -> 8.927 ms` | `94.720 -> 112.017` | `15.43%` reduction |
| 2048 | `10.625 -> 9.037 ms` | `94.121 -> 110.659` | `14.94%` reduction |

このテーマのポイント:

- GEMV 本体を無理に置き換えるのではなく、vLLM trace から残っている copy/cast/norm/reduce を探した
- `elementwise -> copy/cast -> norm/reduce` の発生源として Qwen3.5 の `GemmaRMSNorm` path を特定した
- fp32 cast、reduction、`weight + 1`、output cast を 1 kernel に近づけた
- backend 実測で TPOT が約 `15%` 改善し、decode per-token に効いたことを確認した

Claim boundary:

- confirmed: Qwen3.5-2B / vLLM nightly / `--enforce-eager` / short prompt + decode-heavy workload
- not claimed: all vLLM workloads, batching, CUDA Graph / torch.compile, long prefill, other models

## Why This Exists

前段の `decode_gemv/` では、decode の small-batch linear / GEMV を調査しました。

結果として、cuBLAS-backed `torch_linear` は非常に強く、Triton GEMV は一部 shape で勝つものの、
projection block 全体ではまだ cuBLAS を超えませんでした。

そのため次は、cuBLAS GEMV 本体を正面から置き換えるのではなく、
GEMV 周辺の tiny kernel / copy / cast / layout / elementwise を調べます。

## Question

実 backend の decode request で、自作 kernel / fusion が効きうる領域はどこか。

特に見るもの:

- QKV projection 後の split / reshape / transpose / copy
- MLP projection 後の activation 周辺の elementwise / copy
- residual / norm / projection 前後の cast / copy
- PyTorch native kernel として残っている小さい処理

## First Analysis

既存の vLLM Nsight Systems trace を request window で再集計します。

```bash
bash decode_projection_fusion/scripts/analyze_vllm_request_window.sh
```

Output:

```text
decode_projection_fusion/results/rtx4070/request_window/
```

First result:

| family | share | instances | avg |
|---|---:|---:|---:|
| cuBLAS GEMV | `86.788%` | `58,424` | `71.02 us` |
| copy / cast | `3.695%` | `132,994` | `1.33 us` |
| elementwise | `2.342%` | `122,729` | `0.91 us` |
| fill | `0.166%` | `10,391` | `0.77 us` |

Read:

- GEMV 本体は引き続き最大だが、これは `decode_gemv/` で扱った。
- 次に自作 kernel が効きうるのは、`copy / cast` と `elementwise` の小さい kernel 群。
- これらは単体 share は小さいが、instance 数が多く、kernel launch と memory traffic 削減の題材にしやすい。
- classifier は `nocast` を `cast` と誤判定しないように修正済み。

## Copy/Cast Context

copy/cast kernel の直前・直後に何が走っているかを集計します。

```bash
bash decode_projection_fusion/scripts/analyze_copy_context.sh
```

Output:

```text
decode_projection_fusion/results/rtx4070/copy_context/
```

Result:

| context | share of copy/cast target | instances |
|---|---:|---:|
| `elementwise -> copy/cast -> norm/reduce` | `29.184%` | `31,223` |
| `copy/cast -> copy/cast -> elementwise` | `19.921%` | `19,464` |
| `cuBLAS GEMV -> copy/cast -> elementwise` | `13.141%` | `12,192` |
| `elementwise -> copy/cast -> cuBLAS GEMV` | `12.252%` | `24,384` |

Read:

- copy/cast は GEMV 直後だけではなく、elementwise/norm 周辺にも多い。
- `copy/cast -> copy/cast` が目立つため、連続 copy/cast の発生源特定が次の候補。
- `cuBLAS GEMV -> copy/cast -> elementwise` もあり、projection 後処理の copy/layout も候補。

## Completed Mini Benchmark

最初の mini benchmark として、PyTorch native の copy/cast/add/mul を分けて実行する unfused baseline と、
それらを 1 kernel にまとめる fused Triton kernel を比較しました。

Example:

```text
unfused:
  tmp = x.to(dtype) or x.clone()
  z = tmp + residual
  y = z * scale

fused:
  y = (cast/copy(x) + residual) * scale
```

この段階で狙う主張は、vLLM tokens/sec 改善ではなく、
「実 backend trace で見えた tiny copy/elementwise を再現し、fusion で kernel 数と memory traffic を減らせるか」です。

Result:

| shape | torch clone+add+mul | triton copy+add+mul | ratio |
|---|---:|---:|---:|
| tokens=1, features=4096 | `12.912 us` | `14.528 us` | `1.125x` |
| tokens=1, features=8192 | `13.008 us` | `14.768 us` | `1.135x` |
| tokens=128, features=11008 | `18.688 us` | `17.408 us` | `0.932x` |
| tokens=128, features=16384 | `24.224 us` | `20.272 us` | `0.837x` |

Read:

- decode `tokens=1` では単純な Triton fused kernel は PyTorch baseline に勝てない。
- 大きい tensor では clone を含む unfused baseline に勝つ shape がある。
- 次は単純な add/mul fusion ではなく、本当に不要な copy/layout の発生源を特定する。

## Next Investigation

context analyzer の結果から、vLLM source を確認した。

```bash
bash decode_projection_fusion/scripts/inspect_vllm_sources.sh
```

Result:

- Qwen3.5 は `GemmaRMSNorm` を `Qwen3_5RMSNorm` として使っている。
- input layernorm, post-attention layernorm, final norm が対象。
- `GemmaRMSNorm.forward_cuda()` は `forward_native()` に流れる。
- `forward_native()` は `weight.float() + 1.0` を作る。
- `vllm_c` RMSNorm は input と weight dtype が同じ場合に fast C kernel 条件を満たす。
- bf16 activation + fp32 Gemma-style weight では PyTorch-native decomposition に落ちやすい。
- native decomposition は fp32 cast, pow, mean, rsqrt, multiply, output cast を含む。

This matches the trace pattern:

| context | share of copy/cast target | instances |
|---|---:|---:|
| `elementwise -> copy/cast -> norm/reduce` | `29.184%` | `31,223` |

次の実装候補は `Gemma-style RMSNorm`。

Compare:

- PyTorch native GemmaRMSNorm-style baseline
- Triton/CUDA fused GemmaRMSNorm
- optional fused residual GemmaRMSNorm

## Gemma-style RMSNorm Mini Benchmark

次の mini benchmark では、Qwen3.5 で使われる `GemmaRMSNorm` に寄せて比較します。

```bash
bash decode_projection_fusion/scripts/run_gemma_rmsnorm_bench.sh
```

比較対象:

| implementation | description |
|---|---|
| `torch_gemma_native` | `weight.float() + 1.0` と fp32 cast/reduction を PyTorch native decomposition で実行 |
| `triton_gemma_fused` | reduction、rsqrt、`weight + 1`、output cast を Triton 1 kernel に fusion |
| `cuda_gemma_fused` | 同じ処理を CUDA C++ 1 kernel で実行 |

見る観点:

- vLLM trace の `elementwise -> copy/cast -> norm/reduce` に対応する処理を削れるか
- decode shape の `tokens=1` で効くか
- `tokens=8/128` の小さい prefill/複数 token 条件でも傾向が残るか
- Triton で十分か、CUDA C++ optimized に進む意味があるか

Result:

| shape | torch native | triton fused | cuda fused | best speedup |
|---|---:|---:|---:|---:|
| tokens=1, hidden=2048 | `37.888 us` | `13.840 us` | `7.168 us` | `5.286x` |
| tokens=1, hidden=4096 | `39.776 us` | `13.536 us` | `6.928 us` | `5.741x` |
| tokens=1, hidden=8192 | `38.976 us` | `13.312 us` | `7.168 us` | `5.437x` |
| tokens=128, hidden=8192 | `51.200 us` | `13.456 us` | `8.192 us` | `6.250x` |

Read:

- 単純な add/mul fusion は decode `tokens=1` で弱かったが、Gemma-style RMSNorm は明確に効いた。
- 理由は、PyTorch native decomposition の fp32 cast、pow/mean/rsqrt、multiply、output cast を 1 kernel にまとめられるため。
- ここでの結果は mini benchmark であり、まだ vLLM tokens/sec 改善そのものではない。
- 次は vLLM の `GemmaRMSNorm` path に差し込めるかを調べ、request-only profile を再測定する。

## vLLM Patch Trial

vLLM 本体は直接編集せず、`sitecustomize.py` による opt-in monkey patch で
`GemmaRMSNorm.forward_native/forward_cuda` を差し替える。

CUDA C++ fused kernel は host mini benchmark では使えるが、`vllm/vllm-openai:nightly`
runtime image では CUDA header `cusparse.h` がなく JIT build できなかった。
そのため、vLLM 組み込みの first trial は Triton fused kernel で行う。

Normal run:

```bash
bash decode_projection_fusion/scripts/start_vllm_qwen35_gemma_patch.sh 2b 8000
```

別 terminal:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

Nsight Systems:

```bash
NSYS_DELAY=120 NSYS_DURATION=180 \
  bash decode_projection_fusion/scripts/start_vllm_qwen35_gemma_patch_nsys.sh 2b 8000
```

別 terminal:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

この trial で見るのは、tokens/sec だけではない。
`copy/cast`, `norm/reduce`, `elementwise -> copy/cast -> norm/reduce` が減るかを確認する。

Result:

| run | wall tokens/s |
|---|---:|
| 1 | `93.174` |
| 2 | `94.213` |
| 3 | `95.005` |
| mean | `94.131` |

Request-window Nsight result:

| family | baseline share | patched share |
|---|---:|---:|
| cuBLAS GEMV | `86.788%` | `92.592%` |
| copy / cast | `3.695%` | `1.443%` |
| norm / reduce | `2.708%` | `0.741%` |
| elementwise | `2.342%` | `0.484%` |

Copy/cast context:

| context | baseline share of copy/cast target | patched share of copy/cast target |
|---|---:|---:|
| `elementwise -> copy/cast -> norm/reduce` | `29.184%` | `16.751%` |

Read:

- vLLM の `GemmaRMSNorm` path に Triton fused kernel を差し込むと、request-level tokens/sec は改善方向に出た。
- Nsight でも、狙っていた `copy/cast`, `norm/reduce`, `elementwise` の share が下がった。
- cuBLAS GEMV の share が上がったのは、GEMV が遅くなったというより、周辺の小さい kernel が減って相対的に支配的になったためと読む。
- ただし、JIT warning が request 中に出ているため、最終主張には warmup 条件を揃えた再測定を使う。

## Same-Condition Backend Comparison

JIT / 初回 request の影響を避けるため、unpatched / patched の両方で
`warmup=3`, `runs=5`, `max_tokens=128` に揃えて再測定しました。

| variant | mean tokens/s | median tokens/s | mean latency |
|---|---:|---:|---:|
| unpatched | `93.910` | `93.904` | `1363.022 ms` |
| GemmaRMSNorm patched | `104.659` | `104.644` | `1223.019 ms` |

Delta:

| metric | result |
|---|---:|
| mean tokens/s speedup | `1.114x` |
| median tokens/s speedup | `1.114x` |
| mean latency reduction | `10.27%` |

Read:

- same-condition request-level 測定でも、GemmaRMSNorm fused patch は改善した。
- mini benchmark の勝ちが、vLLM backend の decode request にも残った。
- Nsight で `copy/cast`, `norm/reduce`, `elementwise` が減ったことと、tokens/sec 改善が同じ方向を向いている。
- これにより「profile で見つけた GemmaRMSNorm native decomposition を fused kernel に置き換え、実 backend decode で約 `1.11x` 改善した」というストーリーにできる。

## Streaming TTFT / TPOT Comparison

decode 中の per-token 処理に効いたかを見るため、`stream=True` で
`max_tokens=128/512/2048` を測定しました。

| max tokens | variant | mean TTFT | mean TPOT | mean ITL p50 | mean total latency | mean decode tokens/s |
|---:|---|---:|---:|---:|---:|---:|
| 128 | unpatched | `34.092 ms` | `10.615 ms` | `10.618 ms` | `1382.165 ms` | `94.214` |
| 128 | patched | `30.048 ms` | `8.919 ms` | `8.897 ms` | `1162.700 ms` | `112.129` |
| 512 | unpatched | `34.173 ms` | `10.557 ms` | `10.550 ms` | `5429.002 ms` | `94.720` |
| 512 | patched | `29.531 ms` | `8.927 ms` | `8.918 ms` | `4591.356 ms` | `112.017` |
| 2048 | unpatched | `34.349 ms` | `10.625 ms` | `10.596 ms` | `21783.064 ms` | `94.121` |
| 2048 | patched | `30.378 ms` | `9.037 ms` | `9.025 ms` | `18528.584 ms` | `110.659` |

Delta:

| max tokens | TTFT reduction | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---:|---:|---:|---:|
| 128 | `11.86%` | `15.98%` | `15.88%` | `1.190x` |
| 512 | `13.58%` | `15.44%` | `15.43%` | `1.183x` |
| 2048 | `11.56%` | `14.95%` | `14.94%` | `1.176x` |

Read:

- `max_tokens=128/512/2048` のすべてで patched が改善した。
- TPOT / ITL が一貫して改善しており、decode 中の per-token 処理に効いたという説明ができる。
- TTFT も改善しているが、first token には prefill と最初の decode が混ざるため、TTFT 単独では断定しない。
- 長い `2048` tokens でも decode tokens/s の改善が残っているため、per-token の積み重なりに効いたという見せ方が強い。

## Claim Boundary

現時点では、vLLM request-level の decode benchmark で改善したことは主張できます。

ただし、主張の範囲は次に限定します。

- model は `Qwen/Qwen3.5-2B`
- backend は `vllm/vllm-openai:nightly`
- mode は `--enforce-eager`
- workload は short prompt + `128` generated tokens
- patch は `sitecustomize.py` による opt-in monkey patch
- fused kernel backend は Triton

まだ主張しないこと:

- すべての vLLM workload で速くなる
- CUDA Graph / torch.compile 有効時にも同じ改善率が出る
- 他モデル、長文 prefill、batching 条件でも同じ改善率が出る
