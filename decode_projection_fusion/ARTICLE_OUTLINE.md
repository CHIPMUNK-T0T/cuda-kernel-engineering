# Article Outline: vLLM decode を profile し、GemmaRMSNorm fused kernel で per-token latency を下げる

## この記事で伝えること

vLLM + Qwen3.5-2B の decode request を Nsight で分解し、GEMV 本体ではなく
GemmaRMSNorm 周辺の PyTorch native decomposition に着目した。

`GemmaRMSNorm` を Triton fused kernel に差し替えた結果、streaming benchmark で
TPOT / ITL / decode tokens/s が改善した。

主張の中心:

- LLM backend 全体を見て、効きそうな小さい kernel 群を探した
- cuBLAS GEMV 本体ではなく、copy/cast/norm/reduce の発生源を追った
- Qwen3.5 の `GemmaRMSNorm` path が PyTorch native decomposition に落ちることを確認した
- fp32 cast、reduction、`weight + 1`、output cast を fused kernel にまとめた
- vLLM 実 backend の decode workload で TPOT が約 `15%` 改善した

避ける言い方:

- 「自作 kernel で LLM が爆速」
- 「すべての vLLM workload で速くなる」
- 「TTFT 改善はすべて RMSNorm kernel の効果」

安全な言い方:

- 「Qwen3.5-2B / vLLM nightly / decode-heavy 条件で検証する」
- 「TPOT / ITL が改善しており、decode 中の per-token 処理に効いたと考えられる」
- 「Nsight の copy/cast/norm/reduce 減少と streaming benchmark の改善が整合した」

## 章立て案

### 1. 問題設定: kernel 単体高速化は LLM 全体で効くのか

RMSNorm kernel 単体を速くしても、それだけでは LLM 全体の tokens/sec 改善は主張できない。
LLM 推論では GEMM/GEMV、attention、KV cache、sampling、framework overhead などが絡むため。

この記事では、先に backend 全体を profile し、どこに自作 kernel の余地があるかを見た。

書くポイント:

- kernel 単体 benchmark と backend 実測を分ける
- 速くする対象を先に決め打ちしない
- Nsight で bottleneck と candidate を見る

### 2. 最初の仮説: decode の GEMV を自作する

decode は batch size が小さく、linear projection は GEMV 的になる。
そのため最初は `decode_gemv/` で small-batch GEMV を見た。

結果:

- cuBLAS-backed `torch_linear` は非常に強い
- Triton GEMV は一部 shape で近づく/勝つが、projection block 全体では決定打になりにくい
- 汎用 GEMV 本体を正面から置き換えるのは、既存最適化と競合しやすい

考察:

- GEMV は memory bandwidth と launch overhead の影響が大きい
- cuBLAS は shape ごとの実装選択や低レベル最適化が強い
- 個人実装で汎用的に勝つより、周辺の framework decomposition を減らす方が現実的

ここで方針転換:

- GEMV 本体ではなく、GEMV 周辺に残る tiny kernel / copy / cast / norm / elementwise を探す

### 3. vLLM request trace を見る

vLLM + Qwen3.5-2B の request-only Nsight trace を request window で集計した。

主な内訳:

| family | share | instances | avg |
|---|---:|---:|---:|
| cuBLAS GEMV | `86.788%` | `58,424` | `71.02 us` |
| copy / cast | `3.695%` | `132,994` | `1.33 us` |
| norm / reduce | `2.708%` | `102,882` | `1.26 us` |
| elementwise | `2.342%` | `122,729` | `0.91 us` |

読み:

- GEMV が最大だが、ここは既に cuBLAS が強い
- copy/cast/norm/reduce/elementwise は 1 個あたり小さいが、回数が非常に多い
- decode では per-token で同じ処理が繰り返されるため、小さい処理でも積み重なる

技術的な意味:

- 1 us 前後の kernel が大量にあると、計算量そのものより launch overhead / memory traffic が効きやすい
- PyTorch native decomposition は複数 kernel に分かれやすい
- fusion の価値は FLOPs 削減ではなく、主に memory traffic と kernel launch 数の削減

### 4. copy/cast の文脈を見る

copy/cast kernel の前後を集計した。

上位 context:

| context | share of copy/cast target | instances |
|---|---:|---:|
| `elementwise -> copy/cast -> norm/reduce` | `29.184%` | `31,223` |
| `copy/cast -> copy/cast -> elementwise` | `19.921%` | `19,464` |
| `cuBLAS GEMV -> copy/cast -> elementwise` | `13.141%` | `12,192` |
| `elementwise -> copy/cast -> cuBLAS GEMV` | `12.252%` | `24,384` |

ここで分かったこと:

- copy/cast は projection 直後だけでなく、norm 周辺にも多い
- `elementwise -> copy/cast -> norm/reduce` が大きい
- 単純 add/mul fusion より、norm 周辺の decomposition を見る方がよい

### 5. 失敗した小実験: 単純な add/mul fusion

copy/cast + add/mul を Triton で 1 kernel にした。

結果:

| shape | torch clone+add+mul | triton copy+add+mul | ratio |
|---|---:|---:|---:|
| tokens=1, features=4096 | `12.912 us` | `14.528 us` | `1.125x` |
| tokens=1, features=8192 | `13.008 us` | `14.768 us` | `1.135x` |
| tokens=128, features=11008 | `18.688 us` | `17.408 us` | `0.932x` |
| tokens=128, features=16384 | `24.224 us` | `20.272 us` | `0.837x` |

読み:

- decode `tokens=1` では単純な Triton fused kernel は PyTorch baseline に勝てない
- 大きい tensor では勝つ shape もあるが、vLLM decode の主張には弱い

技術的な理由:

- tokens=1 では work が小さく、Triton kernel launch overhead が効きやすい
- ただ add/mul をまとめるだけでは memory access や framework overhead の削減量が足りない
- 「本当に不要な dtype conversion / temporary tensor / reduction を消す」対象の方がよい

この失敗が次の判断につながる:

- 単純な elementwise fusion ではなく、より重い PyTorch native decomposition を探す

### 6. vLLM source を見て GemmaRMSNorm に到達

Qwen3.5 は `GemmaRMSNorm` を `Qwen3_5RMSNorm` として使っている。

観察した path:

- input layernorm
- post-attention layernorm
- final norm
- `GemmaRMSNorm.forward_cuda()` は `forward_native()` に流れる
- `forward_native()` は `weight.float() + 1.0` を作る
- native RMSNorm path は `x.to(float32)`, `pow`, `mean`, `rsqrt`, multiply, `to(orig_dtype)` を含む

なぜ候補として良いか:

- trace の `elementwise -> copy/cast -> norm/reduce` と一致する
- RMSNorm は layer ごと、decode token ごとに繰り返される
- PyTorch native decomposition は複数 kernel と temporary tensor を生みやすい
- fusion によって memory traffic と launch overhead を減らせる

### 7. 実装した kernel: Gemma-style RMSNorm fused

対象の式:

```text
y = x * rsqrt(mean(x^2) + eps) * (weight + 1)
```

PyTorch native baseline:

```text
weight.float() + 1
x.to(float32)
pow -> mean -> rsqrt
multiply
to(orig_dtype)
```

fused kernel:

- 1 row = 1 hidden vector
- fp32 で reduction
- `rsqrt(mean(x^2) + eps)` を計算
- `(weight + 1)` を適用
- output dtype に cast して store

実装:

- mini benchmark では Triton 版と CUDA C++ 版を比較
- vLLM への差し込みは Triton 版を使用
- vLLM runtime image では CUDA extension JIT build が `cusparse.h` missing で難しかったため
- `sitecustomize.py` による opt-in monkey patch で `GemmaRMSNorm.forward_native/forward_cuda` を差し替え

### 8. mini benchmark 結果

| shape | torch native | triton fused | cuda fused | best speedup |
|---|---:|---:|---:|---:|
| tokens=1, hidden=2048 | `37.888 us` | `13.840 us` | `7.168 us` | `5.286x` |
| tokens=1, hidden=4096 | `39.776 us` | `13.536 us` | `6.928 us` | `5.741x` |
| tokens=1, hidden=8192 | `38.976 us` | `13.312 us` | `7.168 us` | `5.437x` |
| tokens=128, hidden=8192 | `51.200 us` | `13.456 us` | `8.192 us` | `6.250x` |

読み:

- GemmaRMSNorm native decomposition は fusion 対象として有望
- 単純 add/mul fusion と違い、decode `tokens=1` でも明確に勝った
- CUDA C++ 版が最速だが、vLLM runtime integration では Triton 版を選んだ

技術的な理由:

- 複数の PyTorch native kernel を 1 kernel にまとめられる
- fp32 temporary tensor の生成と copy/cast を減らせる
- reduction と final multiply/store を同じ pass に近づけられる
- RMSNorm は arithmetic intensity が高い処理ではなく、memory traffic と launch 数の影響を受けやすい

### 9. vLLM に差し込んだ結果: request-level

同一条件:

- Qwen3.5-2B
- vLLM nightly
- `--enforce-eager`
- `warmup=3`
- `runs=5`
- `max_tokens=128`

結果:

| variant | mean tokens/s | median tokens/s | mean latency |
|---|---:|---:|---:|
| unpatched | `93.910` | `93.904` | `1363.022 ms` |
| GemmaRMSNorm patched | `104.659` | `104.644` | `1223.019 ms` |

差分:

- mean tokens/s speedup: `1.114x`
- median tokens/s speedup: `1.114x`
- mean latency reduction: `10.27%`

読み:

- mini benchmark の勝ちが backend request でも残った
- ただし non-stream では total latency しか分からない
- decode per-token に効いたかは TTFT/TPOT/ITL を分ける必要がある

### 10. vLLM に差し込んだ結果: streaming TPOT / ITL

`stream=True` で `max_tokens=128/512/2048` を測定。
すべて `finish_reason=length` で、指定 token 数まで生成された。

| max tokens | variant | mean TTFT | mean TPOT | mean ITL p50 | mean total latency | mean decode tokens/s |
|---:|---|---:|---:|---:|---:|---:|
| 128 | unpatched | `34.092 ms` | `10.615 ms` | `10.618 ms` | `1382.165 ms` | `94.214` |
| 128 | patched | `30.048 ms` | `8.919 ms` | `8.897 ms` | `1162.700 ms` | `112.129` |
| 512 | unpatched | `34.173 ms` | `10.557 ms` | `10.550 ms` | `5429.002 ms` | `94.720` |
| 512 | patched | `29.531 ms` | `8.927 ms` | `8.918 ms` | `4591.356 ms` | `112.017` |
| 2048 | unpatched | `34.349 ms` | `10.625 ms` | `10.596 ms` | `21783.064 ms` | `94.121` |
| 2048 | patched | `30.378 ms` | `9.037 ms` | `9.025 ms` | `18528.584 ms` | `110.659` |

Delta:

| max tokens | TTFT reduction | TPOT reduction | total latency reduction | decode tokens/s speedup |
|---:|---:|---:|---:|---:|
| 128 | `11.86%` | `15.98%` | `15.88%` | `1.190x` |
| 512 | `13.58%` | `15.44%` | `15.43%` | `1.183x` |
| 2048 | `11.56%` | `14.95%` | `14.94%` | `1.176x` |

読み:

- TPOT / ITL が一貫して改善した
- 長い `2048` tokens でも decode tokens/s 改善が残った
- これは per-token decode 処理の積み重なりに効いたという説明を支える
- TTFT も改善したが、first token には prefill と最初の decode が混ざるため、単独では断定しない

### 11. Nsight との整合

patch 後の request-window では、狙っていた family が減った。

| family | baseline share | patched share |
|---|---:|---:|
| copy / cast | `3.695%` | `1.443%` |
| norm / reduce | `2.708%` | `0.741%` |
| elementwise | `2.342%` | `0.484%` |

一方で cuBLAS GEMV share は `86.788% -> 92.592%` に上がった。

これは GEMV が遅くなったという意味ではなく、周辺の small kernel が減ったため、
残った主処理である GEMV の相対比率が上がったと読む。

技術的な説明:

- RMSNorm 周辺の PyTorch decomposition を fusion すると、copy/cast/norm/reduce/elementwise が減る
- その結果、decode step の non-GEMV overhead が下がる
- 総時間が短くなると、最適化済みの GEMV がより支配的に見える
- これは「次のボトルネックが GEMV に戻る」正常なプロファイル変化

### 12. なぜ LLM decode に効いたのか

RMSNorm は transformer block の中で繰り返し実行される。
decode では 1 token ずつ処理するため、layer 数分の norm と projection が token ごとに発生する。

Qwen3.5-2B のこの path では、GemmaRMSNorm が PyTorch native decomposition に落ち、
複数の小 kernel と copy/cast を発生させていた。

fusion が効いた理由:

1. kernel launch 数を減らした
   - PyTorch native の複数 kernel を 1 kernel 化
   - decode は token ごとに繰り返すため launch overhead が積み重なる

2. memory traffic を減らした
   - fp32 temporary tensor
   - output cast
   - intermediate read/write
   - これらを減らすことで GPU memory access を削減

3. framework decomposition を避けた
   - 高水準 PyTorch op は便利だが、実行時には小 kernel に分解される
   - 1つ1つは小さくても、layer 数 x token 数で支配的になりうる

4. decode workload と相性がよかった
   - batch が小さいため、1 kernel の仕事量が小さく launch overhead が相対的に目立つ
   - per-token latency である TPOT / ITL に直接効きやすい

### 13. `--enforce-eager` を使わない方針にする理由

ここまでの途中計測は `--enforce-eager` 条件で行った。
ただし、記事や採用アピールの最終主張では、この flag に依存しない形で再測定する方が強い。

`--enforce-eager` は CUDA Graph や一部の graph capture / replay を避け、
PyTorch op や vLLM layer 実装を比較的そのまま実行させる。
そのため、PyTorch native decomposition が作る small kernel / copy / cast / reduce は trace に出やすい。
これは調査段階では有利だった。

一方で、実 backend の高速化として見せるなら、通常の vLLM 実行経路に近い条件で改善が残るかを確認したい。
`--enforce-eager` ありで改善しても、それは「eager path に残っていた overhead を減らした」結果に見えやすい。
flag なしでも TPOT / ITL の改善が残れば、主張はかなり強くなる。

今回狙った `GemmaRMSNorm` では、native path が概念的には次のような処理に分解される。

```text
weight.float() + 1
x.to(float32)
pow -> mean -> rsqrt
multiply
to(orig_dtype)
```

この分解が flag なしの実行経路でも残る、または同等の copy/cast/norm overhead として見えるなら、
fused kernel は引き続き有効な候補になる。
逆に、CUDA Graph / compile / vLLM 側の kernel 選択によってこの overhead が隠れるなら、改善幅は小さくなる。

したがって次の方針は明確。

- 起動スクリプトから `--enforce-eager` を外す
- unpatched / Triton patched / CUDA patched を同じ条件で再測定する
- `max_tokens=128/512/2048` の streaming TPOT / ITL を比較する
- Nsight request-window で copy/cast, norm/reduce, elementwise が本当に減るかを見る

記事では、`--enforce-eager` ありの数値は探索段階の補助結果として扱う。
最終的な主張は、flag なしで再測定した結果を使う。

### 14. 限界と今後

今回主張できる範囲:

- Qwen3.5-2B
- vLLM nightly
- default vLLM execution path without `--enforce-eager`
- short prompt + decode-heavy workload
- `stream=True/False` の request-level benchmark
- Triton fused GemmaRMSNorm monkey patch

まだ主張しないこと:

- すべての vLLM workload で速くなる
- CUDA Graph / torch.compile 有効時も同じ改善率が出る
- batch size > 1 でも同じ
- 長文 prefill でも同じ
- 他モデルでも同じ

次に見るなら:

- batch size > 1
- Qwen3.5-4B
- CUDA C++ fused kernel を vLLM runtime に安全に組み込む方法
- Nsight warmup 後 request window を再度取り、最終図にする

### 15. CUDA C++ 版を vLLM に入れた追加検証

mini benchmark では CUDA C++ fused GemmaRMSNorm が最速だった。
そのため、Triton 版で得た backend 改善に対して、CUDA C++ backend でさらに上積みが出るかを確認した。

まず分かったこと:

- host 環境では CUDA C++ extension を build / benchmark できる
- `ninja` は `.venv/bin/ninja` に存在する
- ただし loader が `PATH` 上の `ninja` しか見ていなかったため、未検出になることがあった
- この検出問題は loader 側で `.venv/bin/ninja` も見るように修正した
- vLLM runtime Docker image で CUDA C++ extension build が詰まった主因は `ninja` ではなく、CUDA header `cusparse.h` missing
- `ATen/cuda/CUDAContext.h` 経由で重い CUDA header を引かないように include を軽くすると、runtime container 内 JIT build が通った

比較結果:

| max tokens | variant | mean TPOT | mean total latency | mean decode tokens/s |
|---:|---|---:|---:|---:|
| 128 | unpatched | `10.615 ms` | `1382.165 ms` | `94.214` |
| 128 | Triton patched | `8.919 ms` | `1162.700 ms` | `112.129` |
| 128 | CUDA patched | `8.879 ms` | `1155.458 ms` | `112.628` |
| 512 | unpatched | `10.557 ms` | `5429.002 ms` | `94.720` |
| 512 | Triton patched | `8.927 ms` | `4591.356 ms` | `112.017` |
| 512 | CUDA patched | `8.920 ms` | `4586.286 ms` | `112.112` |
| 2048 | unpatched | `10.625 ms` | `21783.064 ms` | `94.121` |
| 2048 | Triton patched | `9.037 ms` | `18528.584 ms` | `110.659` |
| 2048 | CUDA patched | `9.039 ms` | `18533.316 ms` | `110.629` |

読み:

- CUDA C++ patched も unpatched に対しては TPOT 約 `15%` 改善を維持した
- ただし Triton patched に対する上積みは `0.5%` 未満で、ほぼ同等
- mini benchmark では CUDA C++ が明確に速いが、backend-level では RMSNorm kernel 以外の処理が支配的になる
- 記事の本筋は Triton 版で成立させるのがよい
- CUDA C++ 版は「単体 kernel の速さが request-level 改善にそのまま出るとは限らない」ことを示す追加検証として扱う

優先度の最終判断:

| priority | item | judgment |
|---:|---|---|
| P0 | 現在の Triton backend 結果をまとめる | 完了。本筋。TPOT 約 `15%` 改善、decode tokens/s 約 `1.18x` |
| P1 | CUDA C++ extension を vLLM runtime に入れる | 完了。動くが Triton からの request-level 上積みはほぼない |
| P2 | vLLM devel image を作って container 内 build | 現時点では不要。runtime 内 JIT build が通ったため |
| P3 | flash-attn を host に追加 | 優先度低い。今回の bottleneck ではない |


## ClaudeCode に渡す記事方針

### 冒頭

結果だけから入らず、問題設定から入る。

良い流れ:

1. LLM の kernel を自作しても、単体 benchmark だけでは意味があると言えない
2. そこで vLLM backend を profile し、実際に残っている overhead を探した
3. cuBLAS GEMV 本体ではなく、GemmaRMSNorm 周辺の native decomposition に着目した
4. fused kernel を差し込んだところ、TPOT / ITL が改善した

### 表現のトーン

強く言ってよい:

- profile-driven に実装対象を選んだ
- GemmaRMSNorm 周辺の copy/cast/norm/reduce を減らした
- Qwen3.5-2B decode 条件で TPOT が約 `15%` 改善した
- decode tokens/s が `1.176-1.190x` 改善した

控える:

- vLLM 全体を高速化した
- どのモデルでも効く
- TTFT 改善は RMSNorm のみが原因
- GEMV を倒した

### 最終タイトル案

- vLLM の Qwen3.5 decode を Nsight で分解し、GemmaRMSNorm fused kernel で TPOT を約15%下げた
- RTX 4070 で vLLM decode を profile し、GemmaRMSNorm の PyTorch decomposition を Triton kernel に置き換える
- 自作 kernel は LLM 全体で効くのか: vLLM + Qwen3.5 の GemmaRMSNorm を fused して検証した

### この記事の中心メッセージ

自作 kernel の価値は、単体 benchmark の速さだけでは決まらない。
実 backend の profile から、既存 backend に残っている decomposition / copy / cast を見つけ、
そこを狙って fusion し、最後に TTFT/TPOT/ITL で backend 効果を確認する。

今回の実験では、Qwen3.5-2B の GemmaRMSNorm path がその対象になり、
Triton fused kernel に差し替えることで TPOT / ITL / decode tokens/s の改善を確認できた。
