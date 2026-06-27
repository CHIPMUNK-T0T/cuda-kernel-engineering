# Handoff: vLLM GemmaRMSNorm Upstream Work

## Current Goal

The immediate upstream step is to file a maintainer-facing Issue before opening
a PR.

Current positioning:

1. Share the `GemmaRMSNorm.forward_cuda()` / `Qwen3_5RMSNorm` eager-mode
   fragmentation evidence.
2. Present the fused Triton implementation as a measurement vehicle /
   implementation candidate, not as a final merge request.
3. Ask maintainers whether any non-`torch.compile` remedy is in scope for
   compile-disabled / `--enforce-eager` cases.

This is not ready for a normal PR. A Draft PR may make sense only after the
Issue clarifies the preferred upstream direction.

## Read First

Use these files as the handoff entry points:

- `decode_projection_fusion/upstream_issue/INSTRUCTIONS.md`
- `decode_projection_fusion/upstream_issue/vllm_upstream_issue_draft.md`
- `decode_projection_fusion/upstream_issue/community_landscape.md`
- `decode_projection_fusion/OSS_UPSTREAM_PLAN.md`
- `decode_projection_fusion/vllm_upstream_patch_draft.md`
- `decode_projection_fusion/ARTICLE_OUTLINE.md`

The actual vLLM fork is under:

```text
vllm/
```

The active vLLM branch is:

```text
exp/fused-gemma-rmsnorm-triton
```

## What Was Found

In current vLLM source:

- `vllm/model_executor/models/qwen3_5.py` aliases
  `GemmaRMSNorm` as `Qwen3_5RMSNorm`.
- Qwen3.5 uses that norm in decoder layer norms and final norm.
- `GemmaRMSNorm.forward_cuda()` previously returned `forward_native()`.
- `forward_native()` builds `weight.float() + 1.0`, casts/accumulates in fp32,
  applies RMSNorm, then casts output back to the original activation dtype.

This matched the earlier Nsight observation that copy/cast, norm/reduce, and
elementwise kernels were visible around the norm path.

## What Was Implemented In The vLLM Fork

Files changed in `vllm/`:

```text
vllm/model_executor/layers/gemma_rmsnorm.py
vllm/model_executor/layers/layernorm.py
tests/kernels/core/test_layernorm.py
```

Implementation summary:

- Added a Triton fused Gemma-style RMSNorm helper.
- Added no-residual and residual paths:
  - `gemma_rms_norm(x, weight, eps)`
  - `gemma_fused_add_rms_norm(x, residual, weight, eps)`
- Integrated it into `GemmaRMSNorm.forward_cuda()`.
- Unsupported cases fall back to `forward_native()`.
- Added tests comparing `forward_cuda()` against `forward_native()` for:
  - no residual / residual
  - fp16 / bf16 / fp32
  - representative hidden sizes

Supported path is intentionally narrow:

- CUDA tensors only
- activation dtype: fp16, bf16, or fp32
- `x.dim() == 2`
- `weight.dim() == 1`
- fp32 weight is allowed, matching native GemmaRMSNorm semantics
- residual must match `x` shape and dtype

## Verification Already Done

From `vllm/`:

```text
/home/ubuntu/Desktop/CUDA_kernel/.venv/bin/python -m py_compile \
  vllm/model_executor/layers/gemma_rmsnorm.py \
  vllm/model_executor/layers/layernorm.py \
  tests/kernels/core/test_layernorm.py
```

Result:

```text
passed
```

Also checked:

```text
git diff --check
```

Result:

```text
passed
```

Runtime tests were not completed because the top-level project `.venv` does not
currently contain the vLLM development dependencies. The focused pytest attempt
failed because `pytest` was missing, and a manual import failed because
`packaging` was missing.

Update 2026-06-26:

- Built a real vLLM editable development environment under `vllm/.venv`.
- Build was run with constrained parallelism:
  - `MAX_JOBS=2`
  - `NVCC_THREADS=1`
  - `CMAKE_BUILD_PARALLEL_LEVEL=2`
  - `NINJAFLAGS=-j2`
- Installed the missing test dependency `tblib`.
- Focused correctness test passed:

```text
.venv/bin/python -m pytest \
  tests/kernels/core/test_layernorm.py::test_gemma_rms_norm_cuda_matches_native -q
```

Result:

```text
54 passed, 16 warnings in 9.16s
```

Nearby layernorm regression test also passed:

```text
.venv/bin/python -m pytest tests/kernels/core/test_layernorm.py -q
```

Result:

```text
1027 passed, 16 warnings in 216.67s
```

The focused test now covers:

- residual: no residual / residual
- dtype: fp16 / bf16 / fp32
- hidden size: 2048 / 4096 / 8192
- tokens: 1 / 8 / 128

Also validated Qwen3.5-2B with the real vLLM fork implementation, not the
monkey patch.

Run condition:

- model: `Qwen/Qwen3.5-2B`
- endpoint: `/v1/completions`
- mode: `stream=True`
- prompt: `backend_compare/prompts/decode_japanese.txt`
- max tokens: `128`, `512`, `2048`
- measured runs per max_tokens: `3`
- warmup requests per max_tokens: `1`
- server option: `--enforce-eager`
- server: local editable vLLM fork from `vllm/.venv`
- record:
  `backend_compare/results/rtx4070/stream_requests/runs/20260626-212809-openai_compatible_stream-Qwen-Qwen3-5-2B`

Aggregate:

| max tokens | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean tokens/s | mean decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | `27.228` | `8.740` | `8.738` | `9.021` | `1137.210` | `112.556` | `114.416` |
| 512 | `27.958` | `8.824` | `8.768` | `9.300` | `4537.078` | `112.848` | `113.326` |
| 2048 | `28.113` | `8.986` | `8.924` | `9.595` | `18422.480` | `111.169` | `111.285` |

Compared with the earlier unpatched baseline
`20260625-084737-openai_compatible_stream-Qwen-Qwen3-5-2B`:

| max tokens | TTFT reduction | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---:|---:|---:|---:|
| 128 | `20.14%` | `17.66%` | `17.72%` | `1.214x` |
| 512 | `18.19%` | `16.42%` | `16.43%` | `1.196x` |
| 2048 | `18.16%` | `15.42%` | `15.43%` | `1.182x` |

Compared with the earlier Triton monkey patch run
`20260625-085217-openai_compatible_stream-Qwen-Qwen3-5-2B`:

| max tokens | TPOT delta | decode tokens/s speedup |
|---:|---:|---:|
| 128 | `2.00%` faster | `1.020x` |
| 512 | `1.16%` faster | `1.012x` |
| 2048 | `0.56%` faster | `1.006x` |

Qwen3.5-2B was also measured without `--enforce-eager`.

Official nightly no-eager record:
`backend_compare/results/rtx4070/stream_requests/runs/20260626-232255-openai_compatible_stream-Qwen-Qwen3-5-2B`

Patched fork no-eager record:
`backend_compare/results/rtx4070/stream_requests/runs/20260626-232832-openai_compatible_stream-Qwen-Qwen3-5-2B`

No-eager aggregate:

| backend | max tokens | mean TTFT ms | mean TPOT ms | mean total latency ms | mean decode tokens/s |
|---|---:|---:|---:|---:|---:|
| official nightly | 128 | `22.801` | `9.355` | `1210.910` | `106.902` |
| patched fork | 128 | `20.636` | `9.123` | `1179.311` | `109.608` |
| official nightly | 512 | `24.427` | `9.146` | `4698.210` | `109.334` |
| patched fork | 512 | `20.606` | `9.151` | `4696.765` | `109.279` |
| official nightly | 2048 | `22.653` | `9.230` | `18916.693` | `108.344` |
| patched fork | 2048 | `20.075` | `9.211` | `18874.557` | `108.571` |

No-eager comparison:

| max tokens | TTFT reduction | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---:|---:|---:|---:|
| 128 | `9.49%` | `2.48%` | `2.61%` | `1.025x` |
| 512 | `15.64%` | `-0.05%` | `0.03%` | `0.999x` |
| 2048 | `11.38%` | `0.21%` | `0.22%` | `1.002x` |

Gemma3 1B was measured as the real Gemma-family comparison target.

Run condition:

- model: `google/gemma-3-1b-it`
- endpoint: `/v1/chat/completions`
- mode: `stream=True`
- request option: `ignore_eos=true`
- prompt: `backend_compare/prompts/decode_japanese.txt`
- max tokens: `128`, `512`, `2048`
- measured runs per max_tokens: `3`
- warmup requests per max_tokens: `1`
- server option: `--enforce-eager`
- server: `vllm/vllm-openai:nightly`
- record:
  `backend_compare/results/rtx4070/stream_requests/runs/20260626-222906-official_nightly_chat_stream_ignore_eos-google-gemma-3-1b-it`

Aggregate:

| max tokens | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | `28.733` | `11.691` | `11.703` | `12.199` | `1513.449` | `85.540` |
| 512 | `29.020` | `11.662` | `11.658` | `11.966` | `5988.188` | `85.751` |
| 2048 | `28.953` | `11.670` | `11.979` | `23.648` | `23917.067` | `85.691` |

The patched fork was then measured under the same Gemma3 1B condition.

Run condition:

- model: `google/gemma-3-1b-it`
- endpoint: `/v1/chat/completions`
- mode: `stream=True`
- request option: `ignore_eos=true`
- prompt: `backend_compare/prompts/decode_japanese.txt`
- max tokens: `128`, `512`, `2048`
- measured runs per max_tokens: `3`
- warmup requests per max_tokens: `1`
- server option: `--enforce-eager`
- server: local editable vLLM fork from `vllm/.venv`
- record:
  `backend_compare/results/rtx4070/stream_requests/runs/20260626-225100-patched_fork_chat_stream_ignore_eos-google-gemma-3-1b-it`

Aggregate:

| max tokens | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | `19.286` | `7.925` | `7.963` | `8.195` | `1025.779` | `126.182` |
| 512 | `24.681` | `8.025` | `7.993` | `8.370` | `4125.419` | `124.627` |
| 2048 | `19.600` | `7.976` | `7.997` | `16.067` | `16346.254` | `125.378` |

Compared with the official nightly Gemma3 1B baseline:

| max tokens | TTFT reduction | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---:|---:|---:|---:|
| 128 | `32.88%` | `32.21%` | `32.22%` | `1.475x` |
| 512 | `14.95%` | `31.19%` | `31.11%` | `1.453x` |
| 2048 | `32.30%` | `31.65%` | `31.65%` | `1.463x` |

Gemma3 1B was also measured without `--enforce-eager`, which is closer to the
default vLLM execution path.

Official nightly no-eager record:
`backend_compare/results/rtx4070/stream_requests/runs/20260626-230708-official_nightly_noeager_chat_stream_ignore_eos-google-gemma-3-1b-it`

Patched fork no-eager record:
`backend_compare/results/rtx4070/stream_requests/runs/20260626-231737-patched_fork_noeager_chat_stream_ignore_eos-google-gemma-3-1b-it`

No-eager aggregate:

| backend | max tokens | mean TTFT ms | mean TPOT ms | mean total latency ms | mean decode tokens/s |
|---|---:|---:|---:|---:|---:|
| official nightly | 128 | `12.411` | `5.652` | `730.207` | `176.932` |
| patched fork | 128 | `9.693` | `5.614` | `722.685` | `178.123` |
| official nightly | 512 | `11.585` | `5.903` | `3028.170` | `169.397` |
| patched fork | 512 | `9.375` | `5.875` | `3011.529` | `170.211` |
| official nightly | 2048 | `11.634` | `6.123` | `12546.056` | `163.311` |
| patched fork | 2048 | `9.975` | `6.095` | `12486.470` | `164.069` |

No-eager comparison:

| max tokens | TTFT reduction | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---:|---:|---:|---:|
| 128 | `21.90%` | `0.67%` | `1.03%` | `1.007x` |
| 512 | `19.08%` | `0.48%` | `0.55%` | `1.005x` |
| 2048 | `14.26%` | `0.46%` | `0.47%` | `1.005x` |

Interpretation: the fused path produces a large eager-mode gain for Gemma3 1B,
but under the default no-eager path the decode throughput gain is currently
small. For upstream claims, present eager-mode results as kernel-path evidence
and no-eager results as default-path compatibility / limited-throughput-impact
evidence unless further investigation finds why the default path already hides
most of the norm cost.

Follow-up investigation on the small no-eager delta:

- vLLM's default no-eager path uses `CompilationMode.VLLM_COMPILE` with the
  Inductor backend.
- In that mode, `VllmConfig` appends `none` to `compilation_config.custom_ops`
  by default.
- `CustomOp.enabled()` therefore disables custom ops unless they are explicitly
  listed with `+op_name`.
- When `gemma_rms_norm` is disabled, `GemmaRMSNorm.forward()` dispatches to
  `forward_native()`, not `forward_cuda()`.
- `forward_native()` uses vLLM IR ops:
  - `ir.ops.rms_norm`
  - `ir.ops.fused_add_rms_norm`
- Those IR ops are compile-friendly and can be lowered/fused by the no-eager
  Inductor path.

This means the default no-eager comparison likely measured:

```text
official nightly: native GemmaRMSNorm IR path under Inductor/CUDA graph
patched fork:     native GemmaRMSNorm IR path under Inductor/CUDA graph
```

It probably did not measure the new Triton `forward_cuda()` path. The small
no-eager delta should therefore be treated as compatibility/no-regression
evidence, not as evidence that the fused Triton path itself has no effect under
all non-eager configurations.

Next no-eager experiment should force-enable only this custom op:

```text
--compilation-config '{"custom_ops":["none","+gemma_rms_norm"]}'
```

If that changes performance, the explanation is confirmed: default no-eager was
using the native compiled IR path, while forced custom-op no-eager uses the new
Triton path.

Official nightly was measured with the forced custom-op config:

Record:
`backend_compare/results/rtx4070/stream_requests/runs/20260627-100953-official_nightly_noeager_forced_gemma_customop_chat_stream_ignore_eos-google-gemma-3-1b-it`

Aggregate:

| max tokens | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | `12.111` | `5.687` | `5.677` | `6.090` | `734.414` | `175.827` |
| 512 | `11.446` | `5.936` | `5.936` | `6.543` | `3044.739` | `168.464` |
| 2048 | `11.687` | `6.127` | `6.461` | `12.833` | `12553.670` | `163.212` |

Compared with the previous official nightly default no-eager run, this is
effectively unchanged:

| max tokens | default no-eager decode tokens/s | forced custom-op config decode tokens/s | ratio |
|---:|---:|---:|---:|
| 128 | `176.932` | `175.827` | `0.994x` |
| 512 | `169.397` | `168.464` | `0.994x` |
| 2048 | `163.311` | `163.212` | `0.999x` |

This is expected for official nightly because it does not contain the new
`GemmaRMSNorm.forward_cuda()` Triton path. The useful next comparison is the
patched fork with the same forced custom-op config.

The patched fork was then measured with the same forced custom-op config.

Record:
`backend_compare/results/rtx4070/stream_requests/runs/20260627-101425-patched_fork_noeager_forced_gemma_customop_chat_stream_ignore_eos-google-gemma-3-1b-it`

Aggregate:

| max tokens | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | `10.131` | `5.769` | `5.758` | `6.180` | `742.806` | `173.339` |
| 512 | `10.021` | `6.030` | `6.047` | `6.525` | `3091.535` | `165.828` |
| 2048 | `9.740` | `6.232` | `6.182` | `6.816` | `12766.197` | `160.470` |

Compared with official nightly under the same forced custom-op config:

| max tokens | official forced decode tokens/s | patched forced decode tokens/s | ratio |
|---:|---:|---:|---:|
| 128 | `175.827` | `173.339` | `0.986x` |
| 512 | `168.464` | `165.828` | `0.984x` |
| 2048 | `163.212` | `160.470` | `0.983x` |

Interpretation: forcing `gemma_rms_norm` as a custom op in no-eager does not
improve throughput. It is slightly slower than the official/native compiled IR
path on this RTX 4070 setup. This supports the conclusion that the current
Triton `forward_cuda()` path is valuable as eager-mode kernel-path evidence,
while the default no-eager path should keep using the existing compile-friendly
native/IR path unless a better compiled/custom-op integration is designed.

## Nsight Systems Evidence For Qwen3.5 Eager

Qwen3.5-2B was profiled under `--enforce-eager` to verify that the request-level
speedup comes from the GemmaRMSNorm / Qwen3_5RMSNorm path, not only from noisy
end-to-end timing.

Profiler records:

| backend | profiler record | request record | mean tokens/s |
|---|---|---|---:|
| official nightly | `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/20260627-104321-official-nightly-qwen35-2b` | `backend_compare/results/rtx4070/profile_requests/runs/20260627-104605-openai_compatible-Qwen-Qwen3-5-2B` | `73.944` |
| patched real fork | `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/20260627-103554-patched-real-fork-qwen35-2b` | `backend_compare/results/rtx4070/profile_requests/runs/20260627-103756-openai_compatible-Qwen-Qwen3-5-2B` | `89.221` |

Request-level result:

| comparison | result |
|---|---:|
| throughput speedup | `1.207x` |
| approximate latency reduction | `17.1%` |

Kernel-family comparison from `cuda_gpu_trace.csv` over the profiled request
window:

| kernel family | official time | patched time | time reduction | official launches | patched launches | launch reduction |
|---|---:|---:|---:|---:|---:|---:|
| copy / cast | `358.584 ms` | `78.774 ms` | `78.0%` | `266,160` | `78,816` | `70.4%` |
| norm / reduce | `261.943 ms` | `21.513 ms` | `91.8%` | `205,824` | `18,432` | `91.0%` |
| elementwise | `224.340 ms` | `21.706 ms` | `90.3%` | `245,536` | `21,672` | `91.2%` |
| combined | `844.867 ms` | `121.993 ms` | `85.6%` | `717,520` | `118,920` | `83.4%` |

The official profile shows the native PyTorch decomposition around
GemmaRMSNorm:

- `direct_copy_kernel_cuda`
- `MeanOps reduce_kernel`
- `CUDAFunctorOnSelf_add<float>`
- `rsqrt_kernel_cuda`
- `bfloat16_copy_kernel_cuda`
- `pow_tensor_scalar_kernel_impl`
- `BinaryFunctor MulFun`
- `CUDAFunctor_add<c10::BFloat16>`

The patched profile shows the expected fused kernels:

- `_gemma_fused_add_rms_norm_kernel`: `71.148 ms`, `49,152` launches
- `_gemma_rms_norm_kernel`: `15.578 ms`, `13,312` launches

One caveat: the local analyzer labels `_gemma_*` kernels as
`GEMM / cuBLAS / CUTLASS` because the substring `gemm` appears in `gemma`.
For Issue/PR text, use the explicit kernel names above rather than that family
label.

Issue-ready summary:

```text
Qwen3.5-2B uses the GemmaRMSNorm path through Qwen3_5RMSNorm. Under
--enforce-eager, the official path emits many PyTorch native copy/cast,
elementwise, and norm/reduce kernels. A fused GemmaRMSNorm path replaces those
with dedicated _gemma_*rms_norm_kernel launches, reducing the combined
copy/norm/elementwise GPU time by ~85.6% and improving request throughput from
73.9 tok/s to 89.2 tok/s on RTX 4070.
```

## Current vLLM Git Status

Expected status inside `vllm/`:

```text
 M tests/kernels/core/test_layernorm.py
 M vllm/model_executor/layers/layernorm.py
?? vllm/model_executor/layers/gemma_rmsnorm.py
```

No commit or push has been made.

## Why This Is Not PR-Ready Yet

The patch shape is reasonable, but upstream PR quality needs more evidence.

Remaining gaps:

- Decide whether the upstream maintainers prefer this as:
  - a Triton Python kernel in layer code,
  - a different existing vLLM kernel location,
  - or an extension of an existing RMSNorm custom op.
- Keep claims scoped to measured conditions.

Important: eager-mode performance data shows the strongest kernel-path gain.
The default no-eager Gemma3 1B comparison was measured and shows only a small
decode-throughput gain, so do not claim a large default-path speedup yet.

## Next Commands To Try

From the vLLM fork:

```text
cd /home/ubuntu/Desktop/CUDA_kernel/vllm
git status --short
```

Focused correctness and nearby layernorm regression tests already passed.
Next validate at least one Gemma-family model. Keep the Qwen3.5-2B real-fork
benchmark result above as PR evidence for the Qwen3.5 path.

```text
PYTHONPATH=. python -m pytest tests/kernels/core/test_layernorm.py -q
```

If correctness passes, benchmark the real vLLM fork implementation using the
same Qwen3.5-2B decode-heavy conditions previously used for the monkey patch.

## PR Direction

Possible PR title:

```text
Add fused Triton path for GemmaRMSNorm
```

Safe claim:

```text
This adds a fused Triton path for supported CUDA GemmaRMSNorm inputs and keeps
the existing native implementation as fallback for unsupported cases.
```

Avoid claiming:

```text
Speeds up all vLLM workloads.
Speeds up all Qwen/Gemma models.
Replaces GEMV/GEMM bottlenecks.
```

PR-ready evidence should include:

- correctness table
- Qwen3.5-2B benchmark
- one Gemma-family benchmark
- supported/fallback conditions
- note that unsupported cases preserve existing native behavior
