# Studying Megakernels on a Mac

When decoding one token, should one Metal threadgroup scan all 4,096 cached tokens for its KV head? I tested this on my M4 Pro with isolated attention and an Inkling-inspired two-layer decoder.

At batch one, dividing the isolated scan across threadgroups cut latency from 2,875 to 432 µs. That’s a **6.66× speedup over single-dispatch Metal** and **1.10× over compiled MLX**. In the two-layer decoder with FP32 expert arithmetic, compiled MLX and hybrid Metal/MLX both took **1.56 ms** at the same context length.

[Read the published blog post](https://tiangeluo.github.io/#/blog/studying-megakernels-on-a-mac)

I measured decode latency on the M4 Pro GPU. You can regenerate the figure from the saved timings on a CPU, or collect fresh timings on an Apple-silicon GPU. See the [reproduction guide](REPRODUCIBILITY.md).
