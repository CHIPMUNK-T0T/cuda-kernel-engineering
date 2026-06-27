# Decode Projection Fusion Design

## Goal

vLLM + Qwen3.5 2B の decode request で、cuBLAS GEMV 本体以外に残る
copy / cast / elementwise / layout 系 kernel を切り出し、自作 kernel や fusion が効く領域を探す。

## Background

`decode_gemv/` では small-batch linear / GEMV を調査した。

- cuBLAS-backed `torch_linear` は非常に強い。
- Triton GEMV は一部 shape で勝つが、projection block 全体ではまだ cuBLAS に届かない。
- そのため、次は GEMV 本体を正面から倒すより、GEMV 周辺の memory traffic / tiny kernel / layout 処理を調査する。

## Scope

対象にするもの:

- QKV projection 後の split / reshape / transpose / copy
- MLP projection 後の activation 周辺の elementwise / copy
- residual / norm / projection 前後の cast / copy / small elementwise
- vLLM request trace に実際に出ている PyTorch native kernel

対象にしないもの:

- cuBLAS GEMV を汎用的に置き換えること
- FlashAttention 本体
- vLLM の既存 optimized SwiGLU kernel を根拠なく置き換えること

## First Principle

最初から kernel を書かない。

まず Nsight Systems の request window から kernel 名・時間・出現回数を分類し、
「どの処理が、どれだけ、どの粒度で残っているか」を見てから実装対象を選ぶ。

## Candidate Criteria

次の条件を満たすものを優先する。

1. request window で一定以上の GPU time を持つ
2. 1回あたりが小さく、kernel launch / memory traffic の影響を受けやすい
3. GEMV/GEMM 本体ではなく、周辺処理として fusion 余地がある
4. vLLM ですでに専用 kernel 化されていない、または PyTorch native kernel として残っている
5. mini benchmark で再現でき、before / after を説明できる

## Expected Story

このテーマの狙いは「自作 kernel で vLLM を即座に高速化した」と言うことではない。

狙いは、実 backend profile から候補を選び、
既存最適化が強い領域と自作 kernel が効きうる領域を切り分けること。

