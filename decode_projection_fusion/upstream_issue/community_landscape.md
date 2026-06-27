# vLLM Community Landscape: GemmaRMSNorm Fusion

Research notes (2026-06-27) on existing upstream work, so the Issue is grounded
in the community and not a duplicate. Source: GitHub API search on
`vllm-project/vllm`.

## Directly overlapping work (must reference)

| # | Type | State | Relevance |
|---|---|---|---|
| #42251 | PR | open (2026-05-10) | **Same problem as ours.** "Auto-compile trivial CustomOp fallbacks to complete GemmaRMSNorm fusion under enforce_eager." Documents `GemmaRMSNorm.forward_cuda` → `forward_native` = 8 unfused kernels, ~40% GPU kernel time in Gemma3-4B under enforce_eager. Audited 14 CustomOp subclasses. **Solution = torch.compile auto-fuse (`VLLM_AUTO_FUSE_OPS=1`, `__init_subclass__`), NOT a hand kernel.** Does NOT mention Qwen3.5. **Key details for our Issue:** (a) Limitation #2 in the PR body — "only matters for `enforce_eager=True`; under default compile mode `forward_cuda` is bypassed" → exactly matches our eager-vs-default split. (b) Decode throughput measured only on **Gemma3-4B, bs=1, +5.1%**; bs>1 ≈0% (attention/memory-bound). (c) Fix is `torch.compile`-based, yet its own enforce_eager-trigger list includes `TORCH_COMPILE_DISABLE=1`, under which the auto-fuse is a no-op → residual gap. |
| #38780 | PR | **merged** 2026-04-04 | "[vLLM IR][RMSNorm] Port GemmaRMSNorm to vLLM IR Ops." Tested on Qwen3.5-9B (A100). Caused a CI dtype blowup (`weight.dtype()==input.dtype()`). |
| #39014 | PR | **merged** 2026-04-07 | "[vLLM IR] rework gemma_rms_norm" — rework of #38780. Disables allreduce_rms fusion on dtype mismatch (accuracy issue for quantized models). |
| #42048 | PR | open | "Bypass vllm_ir custom op wrap when only native impl is registered" — lets Inductor fuse the native decomposition on the default path. Closes #41804. |
| #19817 | Issue | open, `help wanted`, `good first issue`, `keep-open` | **Governing philosophy.** "CustomOp cleanup" parent issue: with torch.compile default in V1, the plan is to *reduce reliance on hand-written custom kernels*; recent torch produces Triton faster than custom ops anyway. Exception: custom ops sometimes faster on AMD. |
| #29810 | PR | **closed (auto-stale 2026-04-10)** | "[Kernel] Support rms_norm kernel for Gemma" — a hand-written CUDA Gemma rms_norm kernel (`y = x/sqrt(mean(x^2)+eps)*(1+weight)`), with test_layernorm.py cases + microbench. **Exactly our approach in CUDA. Went stale and was auto-closed.** Precedent that the hand-kernel route stalls. |

## Adjacent / dtype sharp edges

| # | Type | State | Relevance |
|---|---|---|---|
| #43242 | PR | closed | "fp32 residual dtype leak in GemmaRMSNorm.forward_native" (fixes #42588). Native upcasts residual to fp32 and must cast it back to orig dtype. Any fused path replicating native semantics must match this residual contract. |
| #44694 | PR | closed | "Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion" (Qwen3.5-FP8). dtype mismatch is a recurring failure mode in the fused norm path. |
| #44611 | Issue | closed | "rms_norm_per_block_quant crashes on Qwen3.5 models" (RTX 5090). |

## Implications for our Issue

1. The **problem** we found (forward_cuda trivial fallback → enforce_eager
   fragmentation) is already well-described by #42251. Restating it alone =
   duplicate.
2. The **solution** we prototyped (hand-written Triton in forward_cuda) runs
   against the merged IR direction (#38780/#39014) and the stated philosophy
   (#19817), and mirrors the already-stalled #29810.
3. Our **non-duplicate value**:
   - **Qwen3.5 coverage**: `qwen3_5.py` aliases `GemmaRMSNorm` as
     `Qwen3_5RMSNorm` (lines 44, 177, 180, 248). Not mentioned in #42251 or the
     IR PRs in this context. Search for `Qwen3_5RMSNorm` in issues = 0 results.
   - **Consumer-GPU (RTX 4070) eager-vs-default measurements** for Qwen3.5-2B
     and Gemma3-1B, which corroborate that the merged IR fusion already recovers
     most of the default-path cost and quantify the residual enforce_eager gap
     that #42251 targets.
4. Correct framing: **share findings + problem context + a direction question**,
   contributing to the active #42251 / #19817 thread — not "merge my kernel."
