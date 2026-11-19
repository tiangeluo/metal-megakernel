# Megakernels on a Mac

A GPU can have plenty of arithmetic capacity and still wait for work. Between one kernel and the next, an intermediate may need to be written to memory, a dependent dispatch must become runnable, and different parts of the GPU may finish at different times. A megakernel moves some of that scheduling inside a single dispatch. The operating-system analogy helps explain the motivation. While one group of workers waits for data, another can advance an independent part of the computation.

Flapping Airplanes describes an instruction-driven Kittens Virtual Machine that can schedule training, inference and a game inside a persistent kernel. LMSYS chose a narrower boundary for Muse Spark Open. Its megakernel spans one layer's attention through the next layer's QKV projection and KV write. LMSYS reports 1,109 versus 517 tokens/s for a batch-one comparison on eight GB300s. Those are serving-system results with speculative decoding. On my MacBook, I wanted to isolate a scheduling decision I could actually measure. [Flapping Airplanes](https://x.com/flappyairplanes/status/2112955825367269788) · [LMSYS](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast)

## One token writes what it is about to read

When a decoder advances by one token, it prepares new keys and values, writes them to the cache and immediately attends over the cache, including that new entry. Should the writer also perform the scan? Combining them saves a dispatch boundary and lets the six query heads that share a KV head consume the current entry directly from threadgroup memory. Keeping the scan separate lets a specialized attention kernel distribute long histories across more workers.

![The local layer feeds the global layer. The outlined portion starts at projected inputs and ends at the gated heads and cache updates.](architecture.svg)

The experiment uses two adjacent layer types from Inkling 2 Small. Both use grouped-query attention, four-tap causal convolutions, clipped queries and keys, normalized values, learned relative scores, a zero-value attention sink and a per-channel output gate. The local layer reads the last 1,024 positions. The global layer has separate Q/K/V depth reads and token-dependent K/V convolution taps. Both layers execute their eight selected experts. The [model diagram and numerical specification](architecture.md) give the block-state inputs, projection boundaries and exact operations.

## Four implementations of the same operation

The standalone kernel uses the published head geometry, 48 query heads sharing eight KV heads with 128 channels per head. At batch one, a 4,096-entry FP16 K/V cache occupies 16 MiB. The single-dispatch version maps one threadgroup to each KV head. It forwards the freshly rounded current entry on-chip while streaming older entries in tiles. This exposes eight independent threadgroups. At long contexts, extra parallelism may be worth a second dispatch to combine partial softmax results.

The staged Metal implementation gives the same scan its own dispatch. The compiled MLX baseline uses its optimized attention primitive. The hybrid uses custom Metal for preparation and MLX for cache append, attention and gating. A fifth diagnostic schedule partitions the history and merges partial softmax statistics. All methods start from the same projected inputs and end with the same observable gated heads and persistent state.

## What has been verified

The CPU reference processed a 64-token trajectory at the published head dimensions. An independent prefix implementation, with FP64 arithmetic at the agreed FP16 storage boundaries, gave a maximum absolute gated-output difference of 0.0009765625. In the connected two-layer test, the width is 1,152 and each layer contains 384 experts, eight active per token. The full expert matrices occupy 2.85 GiB. CPU passes at visible lengths 64 and 4,096 returned finite summaries and cache states. The weights and incoming upstream block states are seeded.

## Performance measurements

The performance figure will plot absolute synchronized latency for all four schedules against six context lengths. The accompanying table will also show the two-layer latency, which includes projections and routed expert execution. That distinction matters. A gain inside attention can have little effect on a complete decode step if most time is spent elsewhere.

Run `./mac-run study` to generate the figure and table from two checked GPU sessions. The current draft contains the measurements available on CPU. The GPU comparison and interpretation are pending verified M4 Pro results.
