# GemmaRMSNorm eager-fusion — experiment / measurement branch

**This branch is a reference for a vLLM Issue, not a merge proposal.**

It exists so an Issue can link to a concrete, reproducible implementation. It is
*not* a PR and is *not* intended to be merged. The canonical direction for this
problem is the existing upstream work, especially:

- vllm-project/vllm#42251 — *Auto-compile trivial CustomOp fallbacks to complete
  GemmaRMSNorm fusion under enforce_eager* (the `torch.compile`-based fix, which
  is the maintainable approach for the cases it covers)
- vllm-project/vllm#38780 / #39014 — vLLM IR port of `gemma_rms_norm` (merged;
  this is what already fuses the norm on the default compiled path)
- vllm-project/vllm#42048 — bypass IR wrap so Inductor can fuse the native
  decomposition
- vllm-project/vllm#19817 — `CustomOp` cleanup (the direction away from
  hand-written custom kernels)
- vllm-project/vllm#29810 — an earlier hand-written CUDA Gemma `rms_norm` kernel
  (went stale / auto-closed)

## What's here

A narrow fused Triton path for `GemmaRMSNorm.forward_cuda`, used purely as a
measurement vehicle to quantify the achievable ceiling of removing the
`enforce_eager` norm fragmentation that #42251 describes.

- `vllm/model_executor/layers/gemma_rmsnorm.py` — Triton kernels
  (`gemma_rms_norm`, `gemma_fused_add_rms_norm`, `can_use_gemma_rms_norm`)
- `vllm/model_executor/layers/layernorm.py` — `GemmaRMSNorm.forward_cuda` uses
  the fused path when supported, else falls back to `forward_native`
- `tests/kernels/core/test_layernorm.py` — correctness test vs `forward_native`

## Supported (fused) conditions; everything else falls back to native

- Triton available; `x` and `weight` are CUDA tensors
- `x.dtype` ∈ {fp16, bf16, fp32}; `x` is 2D; `weight` is 1D, `x.shape[-1] == weight.shape[0]`
- `x.shape[-1] <= 65536`
- with residual: residual is CUDA and matches `x` shape and dtype
- fp32 accumulation, `(1.0 + weight)` offset, output cast back to activation dtype
  (matches native semantics); residual path returns `(out, updated_residual)`

## Verification (editable build)

```text
tests/kernels/core/test_layernorm.py::test_gemma_rms_norm_cuda_matches_native
  -> 54 passed
tests/kernels/core/test_layernorm.py (full regression)
  -> 1027 passed
```

## Measurement summary (single RTX 4070, batch size 1, decode-heavy)

Single-stream only; higher batch likely shrinks the gain (cf. #42251 bs>1 ≈0%).

| path | model | TPOT reduction | decode tok/s |
|---|---|---:|---:|
| `--enforce-eager` | Qwen3.5-2B (Qwen3_5RMSNorm = GemmaRMSNorm) | 16–18% | 1.20–1.21× |
| `--enforce-eager` | Gemma3-1B | 31–32% | 1.45–1.48× |
| default (compiled) | Qwen3.5-2B | ~0–2.5% | 1.00–1.03× |
| default (compiled) | Gemma3-1B | ~0.5–0.7% | 1.01× |

The default-path ≈0 is consistent with the merged IR fusion (#38780/#39014)
already handling the compiled path; the eager gain is the surface #42251 targets,
and it is much larger on these 1–2B models than the Gemma3-4B figure in #42251.

Forcing this op under no-eager
(`--compilation-config '{"custom_ops":["none","+gemma_rms_norm"]}'`, Gemma3-1B)
was slightly *slower* than the native compiled IR path (decode 0.984–0.986×,
max-tokens 128/512) — i.e. on the default path the compiler-generated fusion
beats this hand kernel, matching the #19817 direction. The fused path only wins
under `enforce_eager`.

## Kernel-level evidence (Nsight Systems, Qwen3.5-2B, `--enforce-eager`)

The eager gain is the GemmaRMSNorm decomposition, not timing noise. Over the
profiled request window the native copy/cast + norm/reduce + elementwise kernels
are replaced by `_gemma_fused_add_rms_norm_kernel` / `_gemma_rms_norm_kernel`:

| kernel group | official nightly | fused vehicle | reduction |
|---|---:|---:|---:|
| copy / cast | 358.6 ms | 78.8 ms | 78.0% |
| norm / reduce | 261.9 ms | 21.5 ms | 91.8% |
| elementwise | 224.3 ms | 21.7 ms | 90.3% |
| combined | 844.9 ms | 122.0 ms | 85.6% |
| combined launches | 717,520 | 118,920 | 83.4% |

Same profiled setup: mean request throughput 73.9 → 89.2 tok/s (1.21×).
Records: `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/`
(`20260627-104321-official-nightly-qwen35-2b`,
`20260627-103554-patched-real-fork-qwen35-2b`).

Tracking issue: <ISSUE_URL — fill in after filing>
