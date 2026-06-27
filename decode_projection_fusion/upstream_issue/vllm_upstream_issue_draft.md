# vLLM Upstream Issue Draft

A maintainer-facing **Issue** (new, filed by the human submitter), referencing
the existing GemmaRMSNorm-fusion work by issue number. It is positioned as
*additional evidence + an unmeasured regime + one open question* for #42251, not
a competing proposal.

Grounding research: see `community_landscape.md` in this folder.

---

## Title options

Primary:

```text
[Perf] enforce_eager GemmaRMSNorm cost is much larger on 1–2B models than the Gemma3-4B figure in #42251 (consumer-GPU data, incl. Qwen3.5)
```

Alternatives:

```text
[Perf] Small-model (1–2B) enforce_eager data for #42251: GemmaRMSNorm/Qwen3_5RMSNorm fragmentation + the TORCH_COMPILE_DISABLE gap
[Perf] Independent confirmation of #42251 on RTX 4070; norm-fusion payoff scales down with model size — plus a compile-disabled question
```

---

## Issue body

### Summary

This is supporting evidence and a follow-up question for #42251 (*Auto-compile
trivial CustomOp fallbacks to complete GemmaRMSNorm fusion under enforce_eager*),
not a new proposal. Three points:

1. **Independent confirmation.** On a consumer GPU (RTX 4070), the
   `enforce_eager`-vs-default split behaves exactly as #42251's Limitation #2
   predicts: the gain is real under `enforce_eager` and ~0 on the default
   compiled path.
2. **An unmeasured regime.** #42251's throughput numbers are for Gemma3-**4B**
   (bs=1, +5.1%). On **1–2B** models the same `enforce_eager` norm fragmentation
   costs far more — single-stream decode is **1.2–1.5×** here — because the norm
   is a larger fraction of per-token work. This strengthens the motivation for
   #42251 in the small-model regime it didn't cover.
3. **One open question** #42251's approach doesn't resolve: it is
   `torch.compile`-based, yet `TORCH_COMPILE_DISABLE=1` is one of the
   `enforce_eager` triggers #42251 itself lists — so under that flag the
   auto-fuse is a no-op. Is that slice in scope for a non-compile fix, or
   out of scope per #19817?

Related work I'm building on: #42251 (open), the merged vLLM IR port #38780 /
#39014, #42048 (open), the `CustomOp` cleanup philosophy #19817, and the earlier
stalled hand-written kernel #29810.

### Setup

- GPU: single RTX 4070; models: `Qwen/Qwen3.5-2B`, `google/gemma-3-1b-it`
- Decode-heavy single-stream (**batch size 1**), streaming, 3 runs + 1 warmup
- Note: `Qwen3_5RMSNorm` is an alias of `GemmaRMSNorm`
  (`vllm/model_executor/models/qwen3_5.py` imports
  `GemmaRMSNorm as Qwen3_5RMSNorm` and uses it for the decoder input /
  post-attention norms and the final norm), so Qwen3.5 exercises the same
  `forward_cuda` → `forward_native` path. It's the same class #42251 already
  auto-detects; I'm just reporting that the small-model data below includes it.

### Point 1 — the eager/default split matches #42251's Limitation #2

#42251 states the patch "only matters for `enforce_eager=True`" because under the
default compile mode `forward_cuda` is bypassed. I see exactly that:

**Default (no-eager / compiled)** — norm already fused via the IR path
(#38780 / #39014 / #42048); near-zero headroom left:

| model | max tokens | TPOT reduction | decode tok/s |
|---|---:|---:|---:|
| Qwen3.5-2B | 128 | 2.5% | 1.03× |
| Qwen3.5-2B | 512 | ~0% (-0.1%) | 1.00× |
| Gemma3-1B | 128 | 0.7% | 1.01× |
| Gemma3-1B | 512 | 0.5% | 1.01× |

So on the default path there is essentially nothing left to win — consistent with
the merged IR fusion doing its job, and with #42251 only targeting eager.

To rule out "the fused path just wasn't exercised under compile," I re-ran
Gemma3-1B no-eager while *force-enabling* this op
(`--compilation-config '{"custom_ops":["none","+gemma_rms_norm"]}'`) so the
Triton `forward_cuda` is actually used under Inductor/CUDA-graph instead of the
native IR decomposition. Forcing the custom op was **slightly slower** than the
native compiled path (decode 0.984–0.986×, max-tokens 128/512), not faster. That
is direct evidence for the #19817 direction on the default path: hand-maintaining
a custom kernel there loses to the compiler-generated fusion, so the only regime
where the fused path wins is `enforce_eager`.

### Point 2 — under enforce_eager, the cost is much larger on small models

`--enforce-eager`, same single-stream setup:

| model | max tokens | TPOT reduction | decode tok/s |
|---|---:|---:|---:|
| Qwen3.5-2B | 128 | 17.7% | 1.21× |
| Qwen3.5-2B | 512 | 16.4% | 1.20× |
| Gemma3-1B | 128 | 32.2% | 1.48× |
| Gemma3-1B | 512 | 31.2% | 1.45× |

For comparison, #42251's decode table (Gemma3-4B, bs=1) reports **+5.1%**. The
gap is model size: on 1–2B models the norm is a larger share of per-token work,
so removing the `enforce_eager` fragmentation #42251 describes buys much more.
This is the regime #42251 didn't benchmark, and where its motivation is
strongest.

**Honest caveat on batch size.** These are bs=1. #42251's own bs>1 numbers go to
~0% because decode becomes attention/memory-bandwidth bound; I'd expect the same
here, so this is a *single-stream / low-concurrency* result, not a
high-throughput-serving claim.

### Kernel-level evidence on Qwen3.5-2B

I also profiled Qwen3.5-2B under `--enforce-eager` with Nsight Systems. This
confirms the request-level gain is coming from the norm decomposition described
in #42251, and not just from noisy end-to-end timing.

The Qwen3.5 path matters because `Qwen3_5RMSNorm` is the same
`GemmaRMSNorm` class. In the official path, the profiler shows many native
PyTorch copy/cast, elementwise, and norm/reduce kernels around this path. In the
patched measurement vehicle, those are replaced by dedicated fused kernels:

- `_gemma_fused_add_rms_norm_kernel`
- `_gemma_rms_norm_kernel`

Summary over the profiled request window:

| kernel group | official nightly | fused measurement vehicle | reduction |
|---|---:|---:|---:|
| copy / cast GPU time | 358.6 ms | 78.8 ms | 78.0% |
| norm / reduce GPU time | 261.9 ms | 21.5 ms | 91.8% |
| elementwise GPU time | 224.3 ms | 21.7 ms | 90.3% |
| combined GPU time | 844.9 ms | 122.0 ms | 85.6% |
| combined launches | 717,520 | 118,920 | 83.4% |

The same profiled request setup improved mean request throughput from
**73.9 tok/s** to **89.2 tok/s** on RTX 4070 (**1.21x**). This lines up with the
broader Qwen3.5-2B eager benchmark above and directly shows that the native
GemmaRMSNorm decomposition is the removed work.

Profiler records can be attached if useful:

- official nightly:
  `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/20260627-104321-official-nightly-qwen35-2b`
- fused measurement vehicle:
  `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/20260627-103554-patched-real-fork-qwen35-2b`

### How the numbers were produced

To get an achievable-ceiling figure for the eager path I wired a narrow fused
Triton `forward_cuda` (CUDA-only; fp16/bf16/fp32; 2D `x`; 1D weight; native
fallback otherwise) and validated it against `forward_native`:

- `test_layernorm.py::test_gemma_rms_norm_cuda_matches_native` across residual
  on/off, fp16/bf16/fp32, hidden 2048/4096/8192, tokens 1/8/128 → **54 passed**
- full `tests/kernels/core/test_layernorm.py` regression → **1027 passed**

This is only the measurement vehicle / upper bound. I'm **not** proposing to
merge a hand-written kernel: #19817 is explicit about reducing reliance on custom
kernels in favor of `torch.compile`-generated ones, and #29810 already tried a
hand-written CUDA Gemma `rms_norm` kernel and went stale. The numbers, not the
kernel, are the contribution. #42251's `torch.compile` auto-fuse is the more
maintainable way to capture the same win for the cases it covers.

Baseline note: the Qwen3.5-2B eager row compares the fused path against an
unpatched build; the Gemma3-1B rows and the Qwen3.5-2B no-eager row compare
against the official nightly (`vllm/vllm-openai:nightly`). The latter are
cross-build, but the ~0% default-path delta bounds that build noise, which is why
I attribute the large eager delta to the norm path. Full TTFT / ITL p50/p95 /
total-latency records can be attached.

### Point 3 — the one case #42251 may not cover

#42251's fix replaces `forward_cuda` with a `torch.compile`'d `forward_native`
under `VLLM_AUTO_FUSE_OPS=1`. But its own "Why enforce_eager Matters" list
includes **`TORCH_COMPILE_DISABLE=1`** as a way users end up in eager. When
`torch.compile` is globally disabled (or Inductor is otherwise unavailable), an
auto-`torch.compile` fix is a no-op, and neither the merged IR fusion nor #42251
removes the fragmentation. That residual slice is the only place a non-compile
(e.g. custom-kernel) path would still matter.

**Question:** is that compile-disabled `enforce_eager` slice considered in scope
for any non-`torch.compile` remedy, or is it explicitly out of scope under the
#19817 direction (accept the perf cost rather than maintain a custom kernel)?

### Questions for maintainers

1. Does the small-model (1–2B) `enforce_eager` data above add useful motivation
   for #42251, given it only benchmarked Gemma3-4B (bs=1, +5.1%)?
2. For the `TORCH_COMPILE_DISABLE=1` / no-Inductor `enforce_eager` slice, is a
   non-compile remedy in scope, or out of scope per #19817?
3. Is it worth explicitly noting Qwen3.5 (`Qwen3_5RMSNorm = GemmaRMSNorm`) as an
   affected model in #42251's scope, or is that already implied since it's the
   same class?

### Scope / caveats

- Single RTX 4070, **batch size 1**, decode-heavy. Not a claim about all
  workloads, batch sizes, models, or hardware; bs>1 likely shrinks the gain
  (cf. #42251).
- **No default-path speedup claim** — the default path is already fused; the gain
  there is ~0.
- The eager-mode gain is for the *supported* `GemmaRMSNorm` inputs only.
- AI assistance was used for the prototyping and measurement; numbers were
  recomputed from the raw benchmark records. Duplicate-work checks were run
  against #42251, #38780, #39014, #42048, #29810, and #19817 before filing.
