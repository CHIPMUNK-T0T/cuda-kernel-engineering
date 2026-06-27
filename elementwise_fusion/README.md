# Elementwise Fusion

vLLM request に残る PyTorch native copy/cast/math kernel を、次点の最適化候補として扱うためのフォルダです。

次の本命テーマは、vLLM request-window profile で最大だった `cuBLAS GEMV` を扱う `decode_gemv/` です。このフォルダは、より実装しやすい次点候補として残します。

## Starting Point

前段の `backend_compare/` では、Qwen3.5 2B の vLLM profile を確認しました。

最初の全体集計では次の傾向でした。

| category | share |
|---|---:|
| GEMM / GEMV | 49.943% |
| elementwise / copy / misc | 45.729% |
| norm-related | 1.677% |

その後、server ready 前の warmup を外して request window だけを再集計すると、次のようになりました。

| family | share |
|---|---:|
| cuBLAS GEMV | 86.788% |
| PyTorch copy / cast | 3.314% |
| PyTorch elementwise math | 3.284% |
| vLLM SwiGLU `act_and_mul_kernel` | 0.407% |

RMSNorm は custom kernel として高速化できましたが、実 backend では主要ボトルネックではありませんでした。
また、SwiGLU は vLLM 側ですでに `act_and_mul_kernel` として custom kernel 化されており、request-window share も小さいため、最初の vLLM-driven target としては弱いです。

## Next

次に決めること:

- 最初に fuse する elementwise 処理
- PyTorch / CUDA / Triton の比較範囲
- decode-like / prefill-like shape
- mini block へ戻すか、kernel 単体から始めるか

候補は次のどちらかです。

- `decode_gemv/`: vLLM request の最大ボトルネックに近いが、実装難度が高い
- `elementwise_fusion/`: PyTorch native copy/cast/math kernel を狙いやすいが、vLLM tokens/sec への寄与は小さめ
