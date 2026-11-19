# Projected decode contract

The operator prepares and attends to one token at a time. The upstream sources are [the checkpoint configuration](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/1df2d97d45210d94f781f42a6521392e5a539e46/config.json) and [the Transformers implementation](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/inkling2/modeling_inkling2.py). That implementation's Git blob is `9cf6b654153ef04793d0ea98fd4fb1c826d8c24d`. The experiment uses FP16 storage and FP32 reductions to compare schedules on Metal. Trained checkpoint weights and the production BF16/quantized model are outside its scope.

## Arrays

Let B be batch size, Q query heads, H KV heads, D head width, R relative-feature width, E relative extent and C cache capacity. The benchmark uses R=16 and E=1024. Every array at the operator boundary uses FP16.

| Dictionary | Key | Shape | Meaning |
| --- | --- | --- | --- |
| token | q, gate | B,Q,D | Raw query and per-channel output gate logits |
| token | k, v | B,H,D | New raw key and value |
| token | r, sink | B,Q,R and B,Q | Relative projection and one zero-value sink logit per query head |
| token | dk, dv | B,H,4 | Global layer's projected token-specific taps, ordered by lag, position zero is the present |
| weights | wk, wv | 4,H,D | Static per-channel taps in lag order, also used in global layers |
| weights | q_scale, r_scale | Q,D and Q,R | Learned scales before clipping |
| weights | rel | E,R | Relative-position bank, transposed from checkpoint layout for contiguous access by lag |
| state | cache_k, cache_v | B,H,C,D | Processed keys and values at absolute position modulo C |
| state | hist_k, hist_v | B,H,3,D | Previous raw projections, newest first |

`reference.attention.step(token, weights, state, position=p, window=W, dynamic=global_layer)` returns the gated heads and advances the state in place. All sequences share a scalar zero-based position. W=0 attends to the entire prefix. A positive window includes the current token. The visible span may not exceed C. Histories contain zeros before a sequence begins.

## Cast points and math

- Convolve K/V in FP32 with lag 0 through 3. Add an identity path from the current raw projection. Static convolution uses the static taps and divides the sum by √5. Dynamic convolution adds the token's four taps to the static taps and divides by 3. Round both outputs to FP16.
- Scale Q/R per channel in FP32. Clip Q and K to an L2 norm of 22.627416997969522, R to 8, using `min(1, threshold / (norm + 1e-6))`. Round to FP16. Normalize each V head with unweighted FP32 RMSNorm (epsilon 1e-6) and round to FP16. Save K/V and advance raw K/V histories. For global attention multiply the rounded Q and R by `1 + 0.1 * log(max(1, (p+1)/8192))` in FP32 and round again. The multiplier equals one throughout the planned timing sweep.
- Query head h reads KV head `h // (Q/H)`. For lag d, add the R·rel[d] inner product rounded to FP16 if d<E. Older global entries keep their content score. Compute Q·K/√D, softmax and the weighted value sum with FP32 accumulation. Include the sink in the denominator exactly once. Round the weighted average to FP16, multiply by sigmoid(gate) in FP32 and round the returned output to FP16.
- The FP64 prefix oracle implements temporally padded delay windows, ordinary softmax with an explicit sink column and explicit window masks/indices from the raw sequence. It applies each agreed FP16 storage boundary. Fixed-length performance fixtures use seeded, already-prepared historical K/V, while trajectory tests construct these from actual preceding projected tokens.

### Acceptance before timing

Gated outputs must satisfy `atol=0.002, rtol=0.01`. Written cache elements must satisfy `atol=0.002, rtol=0.002`. Raw convolution histories and every untouched cache element must match exactly. No acceptance test admits NaNs. These limits apply before any schedule is timed. A failed case is invalid for a speedup claim.

From the repository root:

```sh
python3 -W error -m unittest discover -s reference -v
python3 -W error -m reference.validate
python3 -m reference.fixtures --length 64 --seed 7 --output /tmp/metal-decode-fixture
```

The validation command emits CPU correctness metrics for a full 64-token decode trajectory at 48 Q heads, eight KV heads and head dimension 128. The fixture exporter saves independent NPY files for all inputs and expected outputs, plus a JSON manifest of shapes and generation parameters.
