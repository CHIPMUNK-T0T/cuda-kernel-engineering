# OSS Upstream Plan: Fused GemmaRMSNorm for vLLM

## Goal

`sitecustomize.py` monkey patch で確認した GemmaRMSNorm fused kernel の効果を、
vLLM 本体へ upstream できる形に整理する。

現時点の目的は PR を即作成することではなく、次の 2 点を明確にすること。

1. vLLM current source で GemmaRMSNorm がどの path を通るか
2. monkey patch ではなく、vLLM の layer 実装へ入れるならどの形が自然か

## Current Source Finding

対象 image:

- `vllm/vllm-openai:nightly`

確認済み path:

- `qwen3_5.py` は `GemmaRMSNorm` を `Qwen3_5RMSNorm` として alias している
- Qwen3.5 decoder layer は input layernorm / post-attention layernorm / final norm で `Qwen3_5RMSNorm` を使う
- `GemmaRMSNorm.forward_cuda()` は専用 CUDA kernel を呼ばず、`forward_native()` に流れる
- `forward_native()` は `weight = self.weight.data.float() + 1.0` を作る
- native path は fp32 cast, reduction, multiply, output cast を含む

該当箇所:

```python
class GemmaRMSNorm(CustomOp):
    def forward_native(self, x, residual=None):
        orig_dtype = x.dtype
        weight = self.weight.data.float() + 1.0
        if residual is not None:
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x
        out = ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return (
            out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
        )

    def forward_cuda(self, x, residual=None):
        return self.forward_native(x, residual)
```

## Why This Is a Reasonable OSS Candidate

Gemma-style RMSNorm is not Qwen-only.

The target operation is:

```text
y = x * rsqrt(mean(x^2) + eps) * (weight + 1)
```

This appears as a model-layer primitive rather than a benchmark-only trick.
The current path can create multiple small kernels and temporary tensors around:

- `weight.float() + 1.0`
- activation fp32 upcast
- norm reduction
- final multiply
- output cast

Nsight request-window analysis showed that the related families decreased after patching:

| family | baseline share | patched share |
|---|---:|---:|
| copy / cast | `3.695%` | `1.443%` |
| norm / reduce | `2.708%` | `0.741%` |
| elementwise | `2.342%` | `0.484%` |

Streaming benchmark also showed TPOT reduction of roughly `15%` under the measured
Qwen3.5-2B / vLLM nightly / `--enforce-eager` decode-heavy condition.

## Upstream Integration Shape

The natural upstream target is:

```text
vllm/model_executor/layers/layernorm.py
```

The intended logic is:

```python
def forward_cuda(self, x, residual=None):
    if can_use_fused_gemma_rmsnorm(x, residual, self.weight):
        return fused_gemma_rmsnorm(x, residual, self.weight, self.variance_epsilon)
    return self.forward_native(x, residual)
```

The important upstream requirement is safe fallback.
The fused path should be used only for supported cases, and the current native path
must remain the fallback.

## Initial Supported Conditions

Start narrow.

- CUDA tensor only
- activation dtype: `torch.bfloat16`, `torch.float16`, or `torch.float32`
- weight is 1D and may remain fp32, matching GemmaRMSNorm native semantics
- input is flattened over the last dimension, matching `[tokens, hidden]` style use
- hidden size supported by Triton block size
- residual path supported only when shape and dtype match

Fallback to `forward_native()` for:

- CPU
- unsupported dtype
- unsupported rank/layout
- hidden size too large for the Triton kernel

The upstream patch should avoid broad `try/except` fallback around the kernel.
It is better to keep the supported-condition gate narrow and let real kernel bugs
fail during testing.

## Why Start With Triton

Triton is the right first upstream candidate.

- vLLM already depends heavily on Triton-style kernels
- it avoids CUDA extension ABI/build issues in the first PR
- the request-level result was already strong with Triton
- CUDA C++ was faster in mini benchmark, but backend-level uplift over Triton was almost zero

CUDA C++ can be a follow-up only if default vLLM conditions still show a meaningful gap.

## Validation Needed For OSS PR

Before opening an upstream PR, validate:

1. Correctness
   - no residual
   - residual
   - bf16 / fp16 / fp32
   - representative hidden sizes

2. Model coverage
   - Qwen3.5-2B
   - at least one Gemma-family model

3. Runtime coverage
   - `--enforce-eager` condition, because current data uses it
   - default vLLM execution if Qwen/Gemma starts successfully
   - batch size 1 first, then batch size > 1

4. Performance
   - TPOT / ITL / decode tokens/s
   - no regression in unsupported fallback path
   - Nsight kernel family reduction if possible

## Current Boundary

This repository currently has evidence for:

- Qwen3.5-2B
- vLLM nightly
- decode-heavy benchmark
- monkey patch integration
- `--enforce-eager` measurement condition

This is enough for a strong engineering article and a PR candidate.
It is not yet enough for a production-quality vLLM PR.

## Next Steps

1. Draft the vLLM layer-level patch shape - done
2. Convert monkey patch logic into a minimal `GemmaRMSNorm.forward_cuda()` branch - done in `vllm/` fork branch `exp/fused-gemma-rmsnorm-triton`
3. Add correctness tests for no-residual / residual and bf16 / fp16 / fp32 - drafted
4. Run local correctness tests in a vLLM development environment
5. Re-run Qwen3.5-2B request benchmark with the real vLLM patch
6. Re-run on a Gemma-family model

## Fork Implementation Status

Branch:

```text
exp/fused-gemma-rmsnorm-triton
```

Implemented files:

- `vllm/model_executor/layers/gemma_rmsnorm.py`
- `vllm/model_executor/layers/layernorm.py`
- `tests/kernels/core/test_layernorm.py`

Implementation summary:

- Added common Triton helper functions for Gemma-style RMSNorm.
- Added both no-residual and fused add + RMSNorm paths.
- `GemmaRMSNorm.forward_cuda()` now uses the Triton path only when inputs are supported.
- Unsupported cases fall back to `forward_native()`.
- Added correctness tests comparing `forward_cuda()` against `forward_native()`.

Verification status:

- Python syntax check passed.
- Runtime correctness test has not run yet because the current project `.venv` does not contain vLLM development dependencies such as `pytest` and `packaging`.

For a PR-quality validation, the next required step is to install or use a proper vLLM dev environment and run the focused `GemmaRMSNorm` tests there.
