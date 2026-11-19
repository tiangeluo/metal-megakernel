# Connected decoder replay

This slice runs layers modeled on Inkling 2 Small's local layer 10 and global layer 11. It measures one token through input projections, grouped attention, two output Conv4s per layer, depth reads, and eight selected experts out of 384. Hidden width is 1,152, expert width is 576, and the attention layout is 12 query heads and two KV heads with dimension 128. Both FP16 expert banks together occupy 3,057,647,616 bytes. The separate attention benchmark uses the published 48/8 geometry.

The functions accept synthetic intermediate states for a token inside the model. The incoming bank contains a seeded embedding and the previous completed six-layer block. The incoming partial sum represents layers 6–9. The local layer updates that sum, then the global layer updates it again. Its output is the completed summary that upstream layer 12 would add to the bank.

For each attention or MoE depth read, the implementation RMS-normalizes the three source vectors, scores each source with a learned query scaled by the inverse square root of the hidden width, and sums the normalized values using softmax weights. A learned affine RMSNorm prepares each mixture for projection. Global attention has separate Q/K/V mixtures. R, gate and sink depend on the Q mixture. Dynamic K/V taps use the corresponding K or V mixture.

The router computes sigmoid scores in FP32. Its bias only affects selection. Selected unbiased scores are normalized to sum to eight. Exact ties prefer the lower expert ID. The expert activation applies a sigmoid to the raw gate, multiplies by the tanh-softcapped gate and up projections, then runs the down projection. Benchmark storage is FP16 with FP32 depth, router and expert-output reductions. The replay uses seeded weights and a seeded mid-sequence cache. It does not provide language generation or model quality data.

## Run from the project root

```sh
python -m unittest decoder.test_decoder -v
python -m decoder.benchmark --method cpu --length 64
python -m decoder.benchmark --method cpu --length 4096 --seed 1
```

The compact sequence test builds a real prefix from empty caches. It evaluates all tokens through the first layer, then through the second, using independent dense attention and direct time-domain convolutions. It checks sequential cached decode, final state, local cache wrap, and selected expert IDs. An independent dense-expert test evaluates all 384 compact experts before selecting and weighting eight. The recorded CPU full-width smoke outputs are in `cpu-smoke.json`.

GPU methods are `compiled`, `hybrid`, `staged`, and `single`. The surrounding MLX projection and MoE regions are shared across methods. MLX expert matmuls gather selected matrix indices on-device. The two native methods keep their KV and convolution state in persistent Metal buffers. They bind the current projected inputs and a private exclusive output buffer via Metal DLPack. Cross-queue synchronization at each handoff is included in whole-stack latency. All cache readback for verification happens after the timer. Native operator-only comparisons omit those interop fences. Use `--compact` for small diagnostic fixtures and `python -m bench.run` for experimental timing.

## How sensitive is the second MoE to rounding?

The first M4 Pro study stopped at the two-layer, 64-token replay. Four elements of the second layer's MoE output history exceeded the elementwise bound `0.002 + 0.01 × abs(reference)`. That history contains 1,152 new channels and two rows copied from the input. The log reports a maximum absolute difference of 0.00372314 among the four violations.

We replayed the same two seeds and lengths entirely on the CPU, varying only the accumulation used in the dense and expert projections. The benchmark oracle uses NumPy FP32 `einsum`. The three controls use FP32 matrix multiplication, 32 FP32 partial sums, or FP64 `einsum`. Each rounds to FP16 at the original layer boundaries. All twelve comparisons selected the same eight experts in the same order at both layers and passed the summary check.

| Seed | Context | FP32 matmul | 32 FP32 partials | FP64 einsum |
| --- | ---: | ---: | ---: | ---: |
| 20261118 | 64 | 0 | 0 | 1 |
| 20261118 | 4,096 | 2 | 0 | 0 |
| 20261119 | 64 | 0 | 1 | 2 |
| 20261119 | 4,096 | 0 | 1 | 0 |

Cells count new second-layer MoE channels outside the original bound. For the historical seed at length 64, using FP64 for the dots changed channel 946 from −0.01200104 to −0.00975037. Its absolute error was 0.00225067 against a bound of 0.00212001. A change in projection arithmetic alone can cross this threshold on the CPU. Establishing the cause of the four GPU differences requires replaying the device intermediates. The JSON records every outlying channel and its bound, selected routes, and error norms on the newly computed vector. [CPU control](../diagnostics/projection_sensitivity.py) · [Observed results](../analysis/projection-sensitivity.json)

```sh
.venv/bin/python -m diagnostics.projection_sensitivity --output /tmp/projection-sensitivity.json
```

## Sources

- [Pinned Hugging Face architecture and forward passes](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/inkling2/modeling_inkling2.py)
- [Checkpoint configuration](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/main/config.json)
- [MLX gathered matrix multiplication](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html)
- [MLX DLPack interop](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
