# Upstream Issue Notes

This folder contains the maintainer-facing Issue material for the vLLM
GemmaRMSNorm / Qwen3_5RMSNorm eager-mode fragmentation work.

Read order:

1. `INSTRUCTIONS.md`
2. `community_landscape.md`
3. `vllm_upstream_issue_draft.md`

Current policy:

- File an Issue before opening a PR.
- Treat the fused Triton implementation as a measurement vehicle /
  implementation candidate.
- Do not claim broad default-path speedups.
- Use the Nsight Systems Qwen3.5-2B eager evidence as the strongest kernel-level
  support.
