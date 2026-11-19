# Where the two-layer replay diverges

On the M4 Pro, all four attention schedules passed the connected decoder’s checks at both context lengths for seed 20261119. For seed 20261118, all four exceeded the tolerance on one of the 15 observables, the global layer’s expert-output history. The final decoder summary passed in all 16 comparisons, and both layers selected the same eight experts in the same order as the CPU reference in every comparison.

| Visible tokens | Seed | Compiled MLX | Hybrid | Staged Metal | Single dispatch | Staged / single vs. compiled MLX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 20261118 | 8 | 8 | 4 | 4 | 3 each |
| 4,096 | 20261118 | 1 | 1 | 1 | 1 | 1 each |
| 64 | 20261119 | 0 | 0 | 0 | 0 | 0 each |
| 4,096 | 20261119 | 0 | 0 | 0 | 0 | 0 each |

The four schedule columns count values in `layer11_post_moe` that exceeded `0.002 + 0.01 × abs(CPU reference)`. The last column applies the same check using compiled MLX as the reference. Each history contains 3,456 elements. The first 1,152 are the current token’s raw expert output. The other two rows shift previously seeded history. All recorded worst violations occur in the new row. Hybrid and compiled MLX returned numerically identical values across every recorded observable in all four fixtures.

At 64 tokens, the largest violating difference for compiled MLX was at channel 801 (−0.101929 on GPU versus −0.105713 on CPU). Staged and single-dispatch Metal had their largest violating difference at channel 1,032 (0.078979 versus 0.082703). At 4,096 tokens the compiled and native worst channels were 6 and 997, respectively. This makes both the shared decoder arithmetic and the choice of attention path relevant to tracing the discrepancy.

## What the stage replay resolves

The harness traced compiled MLX on both failing fixtures. At each length, all 81 numerical checks passed. The instrumented replay also reproduced the original execution’s checked output histories, route IDs and final summary exactly. Recomputing the global layer’s finish in NumPy using the GPU’s incoming residual and attention heads brought every expert-history entry under tolerance. For 64 tokens, 223 values still differed, with maximum absolute error 0.001953125. At 4,096 tokens, 358 differed, with maximum absolute error 0.00390625. Elementwise bounds grow with the reference magnitude.

This pattern is consistent with small differences accumulating across layers and operations. The local checks condition on the GPU's inputs. They cannot identify which upstream differences caused a particular end-to-end error. The collected stage trace covers compiled MLX only. The registered whole-stack gate remains open and neither two-layer latency was certified.

## A targeted replay

An untimed crossover records the incoming residual and attention heads at layer 11 for the CPU, compiled MLX and staged Metal. It feeds all nine residual/head combinations through the **same** CPU layer finish. Each source contributes a residual and heads from a normal two-layer call. Mixing the sources intervenes at the input to the finish. The global attention heads themselves can already carry differences from layer 10. The same-source pairs let us compare the GPU finish against a CPU recomputation with identical inputs.

```sh
.venv/bin/python -m diagnostics.stack_crossover --length 64 --seed 20261118 --out results/crossover-64-20261118.json
```

The JSON reports the number of failing channels for every pair and the channel values where any original pair exceeded tolerance. The adjacent NPZ preserves the FP16 inputs, observed outputs and crossed expert outputs for offline analysis, along with the common depth states and output-convolution histories. Captures must reproduce the original call's checked histories, routes and summary exactly. Changing the length or seed runs the same diagnostic on the 4,096-token fixture or the passing control. A full-width CPU self-replay at 64 tokens reproduced the original expert output exactly.

The crossover also exposes the selected expert inputs, route weights, gate/up projections, activations and down projections. It compares this decomposed expert computation against the observed production output. Using the decomposed graph’s inputs, it recomputes all eight gate/up projections and the down projections at the offending channels with FP64 accumulation, followed by the registered FP16 cast. The JSON includes the eight signed contributions per channel and their absolute sum so cancellation is measurable. The GPU readbacks and these diagnostic recomputations run outside timed trials.

[Four original M4 Pro replay records](../results/m4pro-02-diagnostics/) · [CPU projection sensitivity controls](../decoder/README.md) · [Elementwise acceptance rules](../bench/protocol.py)
