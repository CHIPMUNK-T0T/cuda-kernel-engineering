# Decode GEMV Design

## Background

前段の RMSNorm 実験では、custom RMSNorm / fused residual RMSNorm は kernel 単体や mini decoder では効果がありました。
しかし vLLM + Qwen3.5 2B の request-window profile では、RMSNorm は主要ボトルネックではありませんでした。

request window の再集計では、`cuBLAS GEMV` が GPU kernel time の大半を占めました。

```text
cuBLAS GEMV: 86.788%
```

この結果から、次の対象を LLM decode の small-batch linear projection にします。

## Goal

LLM decode で支配的になりやすい `tokens=1` GEMV / small-batch matmul を、PyTorch / cuBLAS / Triton / CUDA C++ で比較します。

主目的:

- vLLM profile から最大ボトルネックを選んだことを示す
- cuBLAS という強い baseline に対して、custom kernel がどこまで迫れるかを見る
- decode shape で何が律速になるかを Nsight で説明する

## Non-Goal

- 初手から vLLM に custom kernel を組み込まない
- 初手から cuBLAS 超えを前提にしない
- quantized GEMV は最初の scope に入れない
- batching / scheduler / paged KV cache は最初の scope に入れない

## Operation

```text
y = x @ W
```

Tensor layout:

```text
x: [tokens, in_features]
W: [in_features, out_features]
y: [tokens, out_features]
```

最初は contiguous tensor のみを扱います。

## Shape Direction

decode-like:

- `tokens=1`
- `in_features`: `2048`, `4096`
- `out_features`: QKV / Wo / MLP projection を想定して段階的に設定

prefill-like reference:

- `tokens=8`, `32`, `128`

decode 専用最適化が、tokens が増えたときにどこで崩れるかも見る。

## Implementation Direction

1. PyTorch baseline
2. Triton baseline
3. CUDA C++ naive GEMV
4. CUDA C++ optimized GEMV

CUDA optimized の検討軸:

- one row / one output tile
- warp-level reduction
- vectorized load
- bf16 / fp16 accumulation strategy
- occupancy
- memory coalescing

Triton baseline の最初の設計:

- one program = one token + one output-column tile
- `W` は `[in_features, out_features]` contiguous として、output 列方向を coalesced load する
- `x` は token ごとに読み、output tile 内で再利用する
- `tokens=1` decode を主対象にしつつ、tokens が増えたときに weight reuse がない設計の限界も測る

## Evaluation

- correctness
- latency
- effective bandwidth
- achieved occupancy
- memory throughput
- kernel launch count
- comparison against cuBLAS-backed PyTorch path

## Risk

cuBLAS は非常に強い baseline です。custom CUDA が勝てない可能性は高いです。

その場合でも、成果は次の形で残します。

- cuBLAS がなぜ強いか
- custom kernel がどの shape で近づくか
- decode GEMV がなぜ vLLM request で支配的になるか
- 次に見るべき余地が quantization / batching / fusion なのか
