目的:
vLLM upstream にPRする前段として、まずIssue本文を作りたい。
PR提案ではなく、GemmaRMSNorm向け fused Triton path の実装方針について maintainer feedback をもらうIssueにしたい。

今回の方針:
- まず通常PRではなく、Issueとして出す。
- 目的は「この実装をmergeしてほしい」ではなく、
  `GemmaRMSNorm.forward_cuda()` / `Qwen3_5RMSNorm` の eager-mode fragmentation
  が小型モデルで大きく効いていることを示し、maintainerに設計方針を確認すること。
- 手元の fused Triton 実装は、現時点では PR-ready な最終提案ではなく、
  問題の大きさと実現可能な上限を測るための measurement vehicle / implementation candidate
  として扱う。
- PRは Issue の反応後に判断する。
  - maintainer が hand-written Triton/custom op path を許容するなら Draft PR 化を検討する。
  - `torch.compile` / vLLM IR 方向に寄せるべきという反応なら、その方向に実装方針を切り替える。
  - compile-disabled / `--enforce-eager` slice は out of scope と判断された場合は、
    現実装を upstream PR にしない。
- Issue本文では #42251 の重複に見えないように、
  Qwen3.5 coverage、RTX 4070 の小型モデル結果、Nsight Systems の kernel-level evidence、
  そして `TORCH_COMPILE_DISABLE=1` で残る可能性のある gap に絞る。
- Issueのトーンは「提案」ではなく「追加データ + 質問」にする。

まず読むもの:
1. /home/ubuntu/Desktop/CUDA_kernel/decode_projection_fusion/HANDOFF.md
   - 現状のSSOT
   - 実装内容、テスト結果、benchmark結果、PRで避けるべきclaimがまとまっている

2. /home/ubuntu/Desktop/CUDA_kernel/decode_projection_fusion/vllm_upstream_patch_draft.md
   - PR向け説明の下書き
   - Issue本文に流用できる表現があるか確認

3. vLLM fork のdiff:
   cd /home/ubuntu/Desktop/CUDA_kernel/vllm
   git diff -- tests/kernels/core/test_layernorm.py vllm/model_executor/layers/layernorm.py vllm/model_executor/layers/gemma_rmsnorm.py

Issueで伝えたいこと:
- GemmaRMSNorm.forward_cuda() は現在 native path に落ちている
- Qwen3.5 は GemmaRMSNorm を Qwen3_5RMSNorm として使っている
- Gemma3 も GemmaRMSNorm path を通る
- narrow supported CUDA path に fused Triton implementation を入れる候補を試した
- unsupported cases は native fallback に戻す
- correctness test は residualあり/なし、fp16/bf16/fp32、hidden 2048/4096/8192、tokens 1/8/128 で通った
- layernorm test全体も通った
- eager modeでは Gemma3-1B と Qwen3.5-2B で大きい改善が出た
- no-eager/default pathでは改善は小さいので、大きなdefault-path speedupとは主張しない
- Qwen3.5-2B の `--enforce-eager` Nsight Systems で、公式 nightly の
  native decomposition が fused `_gemma_*rms_norm_kernel` に置き換わる
  kernel-level evidence も取れている
- その profiler では copy/cast + norm/reduce + elementwise の合計GPU時間が
  `844.867 ms -> 121.993 ms`、約 `85.6%` 減少
- 同じ profiler request の mean throughput は
  `73.944 tok/s -> 89.221 tok/s`、約 `1.207x`
- maintainerに聞きたいのは、この実装をどこに置くべきか:
  - Triton Python kernel as layer helper
  - existing RMSNorm/custom op extension
  - other vLLM kernel location

Issueで避けること:
- 「全vLLM workloadが速くなる」と書かない
- 「default pathで大幅高速化」と書かない
- 「全Gemma/Qwenで効果あり」と書かない
- いきなりPR-readyと断定しない

成果物:
- GitHub Issue本文のMarkdown
- title案
- maintainerへの質問を明確にした短い末尾セクション

Profiler evidence:
- official nightly profiler:
  `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/20260627-104321-official-nightly-qwen35-2b`
- patched real fork profiler:
  `decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/20260627-103554-patched-real-fork-qwen35-2b`
- official request record:
  `backend_compare/results/rtx4070/profile_requests/runs/20260627-104605-openai_compatible-Qwen-Qwen3-5-2B`
- patched request record:
  `backend_compare/results/rtx4070/profile_requests/runs/20260627-103756-openai_compatible-Qwen-Qwen3-5-2B`
