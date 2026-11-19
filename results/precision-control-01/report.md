# Connected decoder with FP32 expert arithmetic

FP32 MoE depth-read output, router input, gate/up and down projection outputs, expert activation and weighted sum. FP16 expert banks; one FP16 cast after mixing. Attention, projections outside the MoE, cache, convolution and residual storage retain the original cast points. The GPU uses a custom selected-expert matvec.

Median synchronized wall time in µs. Brackets give 95% paired bootstrap intervals within the two sessions.

| Visible tokens | Compiled MLX | Staged Metal | Hybrid | Single dispatch |
|---:|---:|---:|---:|---:|
| 64 | 1,422.9 [1,414.0, 1,431.1] | 2,063.6 [2,046.2, 2,085.5] | 1,417.1 [1,407.9, 1,429.3] | 2,081.2 [2,064.1, 2,097.8] |
| 4,096 | 1,561.8 [1,544.4, 1,590.2] | 5,036.2 [4,923.5, 5,155.8] | 1,559.5 [1,541.1, 1,583.4] | 6,016.8 [5,928.5, 6,143.4] |

Two processes, two fixed seeds, 100 paired timing blocks per process and length. Every schedule passed the original 15-observable checks against this control’s CPU equations before and after timing.

Shared benchmark SHA-256 `f807d8a4329016e1d365e31fefd3a44796f7442ccb80d4b1f50d227acc706237`.
Control source SHA-256 `5465d00aa8bb28381f019994f5d3f2a7b3d79073ce246ca81b774ae49cc84cfc`.
