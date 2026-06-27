# Decode Projection Fusion Plan

## Current Status

Start.

`decode_gemv/` では cuBLAS GEMV 本体が強く、Triton GEMV の per-shape tuning だけでは
projection block 全体の高速化には届かないことを確認した。

次は vLLM request trace から、GEMV 周辺の copy / cast / elementwise / layout 系 kernel を切り出す。

## Steps

1. テーマフォルダと設計を作る
   - `decode_projection_fusion/`
   - `DESIGN.md`
   - `PLAN.md`
   - `README.md`
   - status: done

2. 既存 vLLM Nsight Systems trace を再集計する
   - input: `backend_compare/results/rtx4070/nsys/*/cuda_gpu_trace.csv`
   - request window を指定して、kernel family ごとの時間・回数・平均時間を出す
   - run: `bash decode_projection_fusion/scripts/analyze_vllm_request_window.sh`
   - output: `decode_projection_fusion/results/rtx4070/request_window/summary.md`
   - status: done
   - read:
     - cuBLAS GEMV が `86.788%` で最大
     - classifier の `nocast` 誤分類を修正後、GEMV 以外では `copy / cast 3.695%`、`elementwise 2.342%`
     - `copy / cast` は `132,994` instances、平均 `1.33 us`
     - `elementwise` は `122,729` instances、平均 `0.91 us`
     - 大きい単発 kernel ではなく、小さい PyTorch native kernel が大量に残っている

3. candidate kernel を抽出する
   - GEMV/GEMM/FlashAttention を除外
   - copy / cast / elementwise / fill / layout 系を上位表示
   - status: done
   - read:
     - 上位候補は `direct_copy_kernel_cuda`
     - float / bf16 の add / mul 系 elementwise
     - bf16 copy
     - device-to-device memcpy
     - fill bf16
     - この段階では QKV後段 layout と断定するより、まず PyTorch native copy/cast/elementwise の mini reproduction が自然

4. 実装対象を1つに絞る
   - first target: copy/cast + add/mul elementwise fusion
   - example operation:
     - unfused: `tmp = x.to(dtype)` or copy
     - unfused: `z = tmp + residual`
     - unfused: `y = z * scale`
     - fused: copy/cast + add + mul を 1 kernel
   - reason:
     - request trace では該当 family が合計 `6.037%`
     - 1回あたり `~1 us` 前後で、kernel launch / memory traffic 削減の説明がしやすい
     - cuBLAS GEMV 本体を置き換えないため、既存最適化と競合しにくい
   - status: done

5. mini benchmark を作る
   - PyTorch baseline
   - custom Triton fused kernel
   - correctness
   - latency
   - run: `bash decode_projection_fusion/scripts/run_copy_add_mul_bench.sh --runs 50 --warmup 10`
   - output: `decode_projection_fusion/results/rtx4070/copy_add_mul/summary.md`
   - status: done
   - read:
     - `tokens=1` の decode 代表 shape では Triton fused は PyTorch より遅い
     - `tokens=1, features=4096`: `torch_clone_add_mul 12.912 us` vs `triton_copy_add_mul 14.528 us`
     - `tokens=1, features=8192`: `torch_clone_add_mul 13.008 us` vs `triton_copy_add_mul 14.768 us`
     - `tokens=128, features=11008`: `torch_clone_add_mul 18.688 us` vs `triton_copy_add_mul 17.408 us`
     - `tokens=128, features=16384`: `torch_clone_add_mul 24.224 us` vs `triton_copy_add_mul 20.272 us`
     - pure `torch_add_mul` は全 shape で最速
     - 単純な elementwise fusion だけでは、decode `tokens=1` の実利には届きにくい
     - より意味がある候補は「本当に copy/layout を消せる処理」または「複数の後続処理までまとめる fusion」

6. vLLM / mini decode との接続を判断する
   - `copy + add + mul` 単体は、実 backend に直接入れる優先度を下げる
   - 次は vLLM trace の copy/cast kernel の発生源を絞る
   - QKV split/reshape/contiguous、KV cache layout、sampling/softmax 周辺のどれかを特定する
   - status: done

7. copy/cast context analyzer を追加する
   - run: `bash decode_projection_fusion/scripts/analyze_copy_context.sh`
   - output: `decode_projection_fusion/results/rtx4070/copy_context/summary.md`
   - status: done
   - read:
     - copy/cast target: `132,994` kernels, `176,649,452 ns`
     - target 内訳は `direct_copy_kernel_cuda` が `64.118%`
     - `bfloat16_copy_kernel_cuda` が `15.797%`
     - `[CUDA memcpy Device-to-Device]` が `8.415%`
     - previous family は `elementwise 42.486%`, `copy / cast 26.706%`, `cuBLAS GEMV 22.504%`
     - next family は `elementwise 35.248%`, `norm / reduce 29.256%`, `copy / cast 17.452%`, `cuBLAS GEMV 12.252%`
     - family triplet 上位:
       - `elementwise -> copy/cast -> norm/reduce`: `29.184%`
       - `copy/cast -> copy/cast -> elementwise`: `19.921%`
       - `cuBLAS GEMV -> copy/cast -> elementwise`: `13.141%`
       - `elementwise -> copy/cast -> cuBLAS GEMV`: `12.252%`
     - 単純な add/mul fusion より、RMSNorm 近辺の dtype/copy と projection 前後 copy の発生源を見に行くべき

8. 次の候補を決める
   - first candidate: `elementwise -> copy/cast -> norm/reduce`
   - second candidate: `copy/cast -> copy/cast -> elementwise`
   - third candidate: `cuBLAS GEMV -> copy/cast -> elementwise`
   - status: done

9. vLLM source で発生源候補を確認する
   - run: `bash decode_projection_fusion/scripts/inspect_vllm_sources.sh`
   - output: `decode_projection_fusion/results/rtx4070/source_inspection/summary.md`
   - status: done
   - read:
     - Qwen3.5 は `GemmaRMSNorm` を `Qwen3_5RMSNorm` として使う
     - input layernorm, post-attention layernorm, final norm が対象
     - `GemmaRMSNorm.forward_cuda()` は `forward_native()` に流れる
     - `GemmaRMSNorm.forward_native()` は `weight.float() + 1.0` を作る
     - `vllm_c` RMSNorm は input と weight dtype が一致する場合に高速 C kernel 条件を満たす
     - bf16 activation + fp32 Gemma-style weight では native decomposition に落ちやすい
     - native decomposition は `to(float32)`, `pow`, `mean`, `rsqrt`, multiply, `to(orig_dtype)` を含む
     - trace の `elementwise -> copy/cast -> norm/reduce` と整合する

10. 次の実装テーマ
   - first target: Gemma-style RMSNorm
   - baseline: PyTorch native GemmaRMSNorm-style decomposition
   - custom: Triton/CUDA fused GemmaRMSNorm
   - optional: fused residual GemmaRMSNorm
   - status: done

11. Gemma-style RMSNorm mini benchmark を作る
   - PyTorch baseline:
     - `weight.float() + 1.0`
     - `x.to(float32)`
     - `pow -> mean -> rsqrt -> multiply`
     - `to(orig_dtype)`
   - Triton fused:
     - fp32 reduction と `(weight + 1.0)` multiply を 1 kernel にまとめる
   - CUDA fused:
     - 1 row = 1 block の CUDA C++ 実装で比較対象を置く
   - run:
     - `bash decode_projection_fusion/scripts/run_gemma_rmsnorm_bench.sh`
   - output:
     - `decode_projection_fusion/results/rtx4070/gemma_rmsnorm/summary.md`
   - status: done
   - read:
     - `tokens=1, hidden=2048`: PyTorch native `37.888 us`, Triton fused `13.840 us`, CUDA fused `7.168 us`
     - `tokens=1, hidden=4096`: PyTorch native `39.776 us`, Triton fused `13.536 us`, CUDA fused `6.928 us`
     - `tokens=1, hidden=8192`: PyTorch native `38.976 us`, Triton fused `13.312 us`, CUDA fused `7.168 us`
     - CUDA fused は全 shape で best
     - Triton fused も全 shape で PyTorch native より速い
     - max abs error は bf16 許容範囲内
     - 単純 add/mul fusion と違い、GemmaRMSNorm native decomposition は fusion 対象として有望

12. 次の判断
   - mini benchmark では勝ち筋あり
   - 次は vLLM 組み込み方針を決める
   - まずは vLLM の `GemmaRMSNorm` path を最小 patch で差し替えられるか調査する
   - 組み込み後に request-only profile を再測定し、`elementwise -> copy/cast -> norm/reduce` が減るかを見る
   - status: done
   - read:
     - vLLM runtime image では CUDA extension の JIT build が `cusparse.h` missing で失敗した
     - そのため vLLM 差し込み検証の first path は Triton backend にする
     - `sitecustomize.py` で `GemmaRMSNorm.forward_native/forward_cuda` を monkey patch する
     - `residual is None` と `residual` 付きの両方を Triton fused kernel に流す
     - unsupported case は元の native path に fallback する
     - vLLM container 内の small tensor smoke test は通った

13. vLLM patched server を測る
   - normal run:
     - `bash decode_projection_fusion/scripts/start_vllm_qwen35_gemma_patch.sh 2b 8000`
     - `bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1`
   - Nsight run:
     - `NSYS_DELAY=120 NSYS_DURATION=180 bash decode_projection_fusion/scripts/start_vllm_qwen35_gemma_patch_nsys.sh 2b 8000`
     - `bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1`
   - compare:
     - `TRACE=<patched cuda_gpu_trace.csv> OUT_DIR=decode_projection_fusion/results/rtx4070/request_window_gemma_patch bash decode_projection_fusion/scripts/analyze_vllm_request_window.sh`
     - baseline の `copy/cast`, `norm/reduce`, `elementwise -> copy/cast -> norm/reduce` と比較する
   - status: done
   - read:
     - patched request:
       - run1 `93.174 tok/s`
       - run2 `94.213 tok/s`
       - run3 `95.005 tok/s`
       - mean `94.131 tok/s`
     - baseline request-window share:
       - `copy/cast 3.695%`
       - `norm/reduce 2.708%`
       - `elementwise 2.342%`
     - patched request-window share:
       - `copy/cast 1.443%`
       - `norm/reduce 0.741%`
       - `elementwise 0.484%`
     - target context:
       - baseline `elementwise -> copy/cast -> norm/reduce`: `29.184%`
       - patched `elementwise -> copy/cast -> norm/reduce`: `16.751%`
     - cuBLAS GEMV share は `86.788%` から `92.592%` に上がった
     - これは GEMV が悪化したというより、GemmaRMSNorm 周辺の PyTorch native decomposition が減り、残った主処理が GEMV に寄ったと読む
     - request 中に Triton JIT warning があるため、final 用には warmup を厚くした再測定が必要

14. same-condition backend comparison を取る
   - condition:
     - unpatched / patched
     - `warmup=3`
     - `runs=5`
     - `max_tokens=128`
     - `--enforce-eager`
   - unpatched:
     - record: `backend_compare/results/rtx4070/profile_requests/runs/20260624-231130-openai_compatible-Qwen-Qwen3-5-2B`
     - mean `93.910 tok/s`
     - median `93.904 tok/s`
     - mean latency `1363.022 ms`
   - patched:
     - record: `backend_compare/results/rtx4070/profile_requests/runs/20260624-231429-openai_compatible-Qwen-Qwen3-5-2B`
     - mean `104.659 tok/s`
     - median `104.644 tok/s`
     - mean latency `1223.019 ms`
   - delta:
     - mean tokens/s speedup `1.114x`
     - median tokens/s speedup `1.114x`
     - mean latency reduction `10.27%`
   - output:
     - `decode_projection_fusion/results/rtx4070/vllm_gemma_patch_compare/20260624-231130-vs-231429/summary.md`
   - status: done
   - read:
     - warmup を厚くした同一条件でも patched が改善した
     - Nsight の `copy/cast`, `norm/reduce`, `elementwise` 減少と request-level throughput 改善が整合する
     - 記事では「実 backend decode で約 `1.11x` 改善」と言える

15. streaming benchmark で TTFT / TPOT / ITL を分ける
   - condition:
     - unpatched / patched
     - `stream=True`
     - `max_tokens=128,512,2048`
     - `warmup=1`
     - `runs=3`
     - `--enforce-eager`
   - unpatched:
     - record: `backend_compare/results/rtx4070/stream_requests/runs/20260625-084737-openai_compatible_stream-Qwen-Qwen3-5-2B`
   - patched:
     - record: `backend_compare/results/rtx4070/stream_requests/runs/20260625-085217-openai_compatible_stream-Qwen-Qwen3-5-2B`
   - output:
     - `decode_projection_fusion/results/rtx4070/vllm_gemma_patch_stream_compare/20260625-084737-vs-085217/summary.md`
   - status: done
   - read:
     - all runs ended with `finish_reason=length`
     - `max_tokens=128`: TPOT `10.615 ms -> 8.919 ms`, decode tokens/s `94.214 -> 112.129`
     - `max_tokens=512`: TPOT `10.557 ms -> 8.927 ms`, decode tokens/s `94.720 -> 112.017`
     - `max_tokens=2048`: TPOT `10.625 ms -> 9.037 ms`, decode tokens/s `94.121 -> 110.659`
     - TPOT reduction は `14.95-15.98%`
     - decode tokens/s speedup は `1.176-1.190x`
     - TTFT も `11.56-13.58%` 改善したが、first token には prefill と最初の decode が混ざるため、TTFT 単独では断定しない
     - per-token decode 改善を主張する根拠として、TPOT / ITL / 2048 tokens の改善が重要

16. CUDA C++ vLLM integration を確認する
   - status: done
   - background:
     - mini benchmark では CUDA C++ fused GemmaRMSNorm が最速
     - vLLM backend 実測では Triton fused GemmaRMSNorm でも TPOT 約 `15%` 改善、decode tokens/s 約 `1.18x` 改善を確認済み
     - `ninja` は `.venv/bin/ninja` に存在する
     - CUDA extension loader は `.venv/bin/ninja` を拾うよう修正済み
     - vLLM Docker runtime 内で CUDA C++ extension を JIT build できなかった主因は `ninja` ではなく `cusparse.h` missing
     - `gemma_rmsnorm.cu` の include を軽くし、`ATen/cuda/CUDAContext.h` 経由で `cusparse.h` を要求しない形に変更した
     - vLLM runtime container 内で CUDA C++ extension の JIT build と smoke test が通った
   - priority:

     | priority | item | reason |
     |---:|---|---|
     | P0 | 現在の Triton backend 結果をまとめる | 完了。すでに成果あり。TPOT 約 `15%` 改善、decode tokens/s 約 `1.18x` |
     | P1 | CUDA C++ extension を vLLM runtime で読み込ませる | 完了。JIT build と request-level benchmark が通った |
     | P2 | vLLM devel image を作って container 内 build | 現時点では不要。runtime 内 JIT build が通ったため |
     | P3 | flash-attn を host に追加 | 優先度低い。vLLM container 側では既に FlashAttention / FlashInfer 系 kernel が使われている |

   - P1 result:
     - runtime container 内で `VLLM_GEMMA_RMSNORM_PATCH_BACKEND=cuda` が読み込めた
     - smoke test は bf16 input / bf16 weight で通った
     - streaming benchmark:
       - record: `backend_compare/results/rtx4070/stream_requests/runs/20260625-101142-openai_compatible_stream-Qwen-Qwen3-5-2B`
       - compare output: `decode_projection_fusion/results/rtx4070/vllm_gemma_patch_cuda_compare/20260625-084737-vs-085217-vs-101142/summary.md`
     - CUDA C++ patched vs unpatched:
       - `max_tokens=128`: TPOT `10.615 ms -> 8.879 ms`, decode tokens/s `94.214 -> 112.628`
       - `max_tokens=512`: TPOT `10.557 ms -> 8.920 ms`, decode tokens/s `94.720 -> 112.112`
       - `max_tokens=2048`: TPOT `10.625 ms -> 9.039 ms`, decode tokens/s `94.121 -> 110.629`
     - CUDA C++ patched vs Triton patched:
       - `max_tokens=128`: CUDA C++ が TPOT `0.45%` 程度速い
       - `max_tokens=512`: CUDA C++ が TPOT `0.08%` 程度速い
       - `max_tokens=2048`: CUDA C++ が TPOT `0.03%` 程度遅い
   - read:
     - 記事本筋は Triton backend 結果で成立している
     - CUDA C++ backend integration は追加検証として扱う
     - CUDA C++ 版は mini benchmark で明確に速いが、backend-level では Triton 版からの上積みはほぼない
     - これは「単体 kernel の速さが、そのまま request-level 改善になるとは限らない」という記事の主張を補強する

## Initial Run

既存の request-only trace を解析する。

```bash
bash decode_projection_fusion/scripts/analyze_vllm_request_window.sh
```

## First Read

Request window `45s-70s` の再集計では、GEMV 本体以外の候補は次の通り。

| family | share | instances | avg |
|---|---:|---:|---:|
| copy / cast | `3.695%` | `132,994` | `1.33 us` |
| elementwise | `2.342%` | `122,729` | `0.91 us` |
| fill | `0.166%` | `10,391` | `0.77 us` |

最初は、GEMV 本体ではなく `copy/cast + add/mul` のような小さい連続処理を fusion し、
kernel 数と memory traffic が減るかを見た。

結果として、単純な add/mul fusion は decode `tokens=1` では PyTorch baseline に勝ちにくい。
そのため、次は copy/cast の発生源を文脈付きで見る。

## Copy/Cast Context Analysis

```bash
bash decode_projection_fusion/scripts/analyze_copy_context.sh
```

結果:

| context | share of copy/cast target | instances |
|---|---:|---:|
| `elementwise -> copy/cast -> norm/reduce` | `29.184%` | `31,223` |
| `copy/cast -> copy/cast -> elementwise` | `19.921%` | `19,464` |
| `cuBLAS GEMV -> copy/cast -> elementwise` | `13.141%` | `12,192` |
| `elementwise -> copy/cast -> cuBLAS GEMV` | `12.252%` | `24,384` |

読み:

- copy/cast は GEMV 直後だけでなく、elementwise/norm 周辺にも多い。
- `copy/cast -> copy/cast` が `17.452%` あり、連続 copy/cast の削減余地がある。
- 次は kernel 名だけではなく、vLLM/PyTorch の発生源を絞る必要がある。

## Source Inspection

```bash
bash decode_projection_fusion/scripts/inspect_vllm_sources.sh
```

結果:

- Qwen3.5 は `GemmaRMSNorm` を `Qwen3_5RMSNorm` として使っている。
- `GemmaRMSNorm.forward_cuda()` は vLLM C kernel を直接呼ばず、`forward_native()` に流れる。
- `forward_native()` は fp32 の Gemma-style weight を作り、`ir.ops.rms_norm` を呼ぶ。
- `ir.ops.rms_norm` の native path は fp32 cast, pow, mean, rsqrt, multiply, output cast を含む。
- これは trace の `elementwise -> copy/cast -> norm/reduce` と整合する。

次の実装対象は `Gemma-style RMSNorm` が第一候補。

## Gemma-style RMSNorm Benchmark

実 backend trace と vLLM source inspection から、次は Qwen3.5 の
`GemmaRMSNorm` native decomposition を mini benchmark 化する。

```bash
bash decode_projection_fusion/scripts/run_gemma_rmsnorm_bench.sh
```

比較する実装:

| implementation | meaning |
|---|---|
| `torch_gemma_native` | vLLM native path に近い PyTorch decomposition |
| `triton_gemma_fused` | fp32 reduction / rsqrt / `(weight + 1)` / output cast を Triton 1 kernel 化 |
| `cuda_gemma_fused` | 同じ処理を CUDA C++ 1 kernel 化 |

この実験で見たいこと:

- trace の `elementwise -> copy/cast -> norm/reduce` に対応する処理を本当に削れるか
- decode `tokens=1` で PyTorch native より速いか
- `tokens=8/128` でも効果が残るか
- Triton で十分か、CUDA C++ optimized に進む価値があるか

結果:

| shape | torch native | triton fused | cuda fused | best speedup |
|---|---:|---:|---:|---:|
| tokens=1, hidden=2048 | `37.888 us` | `13.840 us` | `7.168 us` | `5.286x` |
| tokens=1, hidden=4096 | `39.776 us` | `13.536 us` | `6.928 us` | `5.741x` |
| tokens=1, hidden=8192 | `38.976 us` | `13.312 us` | `7.168 us` | `5.437x` |
| tokens=128, hidden=8192 | `51.200 us` | `13.456 us` | `8.192 us` | `6.250x` |

読み:

- PyTorch native decomposition は小さい kernel 群と dtype/copy が残りやすい。
- Triton/CUDA fused は reduction, rsqrt, `(weight + 1)`, output cast を 1 kernel にまとめるため、mini benchmark では明確に速い。
- 次に必要なのは、vLLM 内でこの path を差し替えたときに request profile と tokens/sec が改善するかの確認。

## vLLM Patch Plan

vLLM 本体を直接編集せず、`sitecustomize.py` を `PYTHONPATH` に置いて
`GemmaRMSNorm.forward_native/forward_cuda` を monkey patch する。

CUDA C++ extension は host の mini benchmark では動くが、vLLM runtime image では
CUDA header `cusparse.h` がなく JIT build できなかった。
そのため、vLLM 組み込み検証はまず Triton backend で行う。

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

見るもの:

- request throughput が baseline `~72 tok/s` から動くか
- request window の `copy/cast`, `norm/reduce`, `elementwise` が減るか
- `elementwise -> copy/cast -> norm/reduce` triplet が減るか
- Triton kernel launch が増えすぎて decode latency を悪化させないか

## Claim Boundary

途中計測では、Qwen3.5-2B / vLLM nightly / `--enforce-eager` /
short prompt + decode-heavy workload で request-level 改善を確認した。
ただし、最終主張は `--enforce-eager` なしで再測定した結果に寄せる。

途中結果として主張できること:

- GemmaRMSNorm Triton fused patch で TPOT / ITL が改善した
- `max_tokens=128/512/2048` で decode tokens/s が `1.176-1.190x` 改善した
- Nsight 上でも `copy/cast`, `norm/reduce`, `elementwise` の share が減った

次に確認すること:

- `--enforce-eager` なしの unpatched / Triton patched / CUDA patched を同じ条件で再測定する
- `max_tokens=128/512/2048` の streaming TPOT / ITL 改善が残るかを見る
- Nsight request-window で `copy/cast`, `norm/reduce`, `elementwise` の減少が残るかを見る

まだ主張しないこと:

- すべての vLLM workload で速くなる
- default vLLM execution path でも同じ改善率が出る
- batch size > 1 や long prefill でも同じ改善が出る
- 他モデルでも同じ改善が出る
