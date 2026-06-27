# Draft comment for vllm-project/vllm#42251

> Positioning: this is a **comment on #42251**, not a new Issue and not a competing PR.
> Tone: I took a different approach to the *same* enforce_eager GemmaRMSNorm problem;
> #42251's torch.compile direction is the better/more maintainable one, so I'm just
> leaving my data + a reference branch here for anyone working on the same thing,
> plus one scope question.

---

I was looking at the same `enforce_eager` GemmaRMSNorm fragmentation from a
different angle — a hand-written fused Triton `forward_cuda` — before finding this
PR. I think the `torch.compile` auto-fuse direction here is the better and more
maintainable approach (and it generalizes to the other trivial-fallback CustomOps
you audited, not just this one), so this isn't worth a separate issue. I just
wanted to leave some supporting data and one scope question on the thread in case
it's useful.

All numbers are single RTX 4070, **batch size 1**, decode-heavy single-stream, 3
runs + 1 warmup. bs=1 only — I'd expect bs>1 to shrink toward ~0 like your own
table, so this is a low-concurrency result, not a serving-throughput claim.

### 1. The eager/default split matches your Limitation #2

Default (compiled) path — norm already fused via the merged IR port
(#38780 / #39014); essentially no headroom left:

| model | max tok | TPOT reduction | decode tok/s |
|---|---:|---:|---:|
| Qwen3.5-2B | 128 | 2.5% | 1.03× |
| Qwen3.5-2B | 512 | -0.1% | 1.00× |
| Gemma3-1B | 128 | 0.7% | 1.01× |
| Gemma3-1B | 512 | 0.5% | 1.01× |

`--enforce-eager` — where the fragmentation actually costs:

| model | max tok | TPOT reduction | decode tok/s |
|---|---:|---:|---:|
| Qwen3.5-2B | 128 | 17.7% | 1.21× |
| Qwen3.5-2B | 512 | 16.4% | 1.20× |
| Gemma3-1B | 128 | 32.2% | 1.48× |
| Gemma3-1B | 512 | 31.2% | 1.45× |

Two things that might be useful for the PR:

- **Small models cost more.** Your decode table is Gemma3-4B (bs=1, +5.1%). On
  1–2B the same eager fragmentation is a larger share of per-token work, so the
  eager payoff is bigger — i.e. the motivation is strongest in the small-model
  regime the PR didn't benchmark.
- **Qwen3.5 is covered too.** `Qwen3_5RMSNorm` is an alias of `GemmaRMSNorm`
  (`qwen3_5.py` imports `GemmaRMSNorm as Qwen3_5RMSNorm` for the layer/final
  norms), so it exercises the same path. Same class your detection already
  catches; just noting the data above includes it.

### 2. Kernel-level confirmation (Nsight Systems, Qwen3.5-2B, --enforce-eager)

Over the profiled request window, the native copy/cast + norm/reduce +
elementwise kernels collapse into the two fused kernels:

| kernel group | official nightly | fused | reduction |
|---|---:|---:|---:|
| copy / cast | 358.6 ms | 78.8 ms | 78.0% |
| norm / reduce | 261.9 ms | 21.5 ms | 91.8% |
| elementwise | 224.3 ms | 21.7 ms | 90.3% |
| combined | 844.9 ms | 122.0 ms | 85.6% |
| launches | 717,520 | 118,920 | 83.4% |

Mean request throughput 73.9 → 89.2 tok/s (1.21×). This is just confirming the
decomposition you describe is the real cost, not timing noise.

### 3. A control experiment that supports *your* direction

I force-enabled the hand kernel under the default compiled path
(`--compilation-config '{"custom_ops":["none","+gemma_rms_norm"]}'`, Gemma3-1B)
so `forward_cuda` actually runs under Inductor instead of the native IR
decomposition. It was **slightly slower** than the native compiled path
(decode 0.984–0.986×, max-tokens 128/512), not faster. So on the default path the
compiler-generated fusion beats the hand kernel — which is exactly the #19817
argument for not maintaining a custom kernel. The hand kernel only wins under
`enforce_eager`.

### One scope question

Your "Why enforce_eager Matters" list includes `TORCH_COMPILE_DISABLE=1`. Since
the fix here is itself `torch.compile`-based, under that flag (or wherever
Inductor is unavailable) the auto-fuse is a no-op, and neither the IR fusion nor
this PR removes the fragmentation. Is that compile-disabled `enforce_eager` slice
considered in scope for any non-`torch.compile` remedy, or explicitly out of
scope per #19817 (accept the cost rather than carry a custom kernel)? That's the
only place a direct kernel would still matter, and I'd rather defer to your call
before doing anything with it.

### Reference

Measurement vehicle (narrow fused Triton `forward_cuda`, native fallback
otherwise; correctness vs `forward_native`: 54 focused + 1027 full
`test_layernorm.py` passing) — not proposed for merge, just so the numbers are
reproducible:

- branch: `<FORK_URL>/tree/exp/gemma-rmsnorm-eager-fusion-measurement`

Caveats: bs=1 only; the Qwen3.5-2B eager row is fork-vs-unpatched while the
Gemma3-1B rows are fork-vs-official-nightly (cross-build, but the ~0% default
delta bounds that build noise). AI assistance was used for the prototype and
measurement; the numbers were recomputed from the raw benchmark records.
