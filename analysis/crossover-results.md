# What carries the difference into layer 11?

Each of the four M4 Pro captures reproduced its production call’s expert histories, attention-output histories, route IDs and final residual exactly. I took the incoming residual from one implementation and the gated attention heads from another, then ran the rest of layer 11 in NumPy. Each row below compares the resulting 1,152-channel expert output with the original CPU result. A failure is an element that exceeds the registered tolerance of 0.002 + 0.01 × abs(reference).

| Tokens | Seed | GPU inputs from | GPU residual only | GPU heads only | Both GPU inputs | Observed on GPU |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 64 | 20261118 | Compiled MLX | 3 | 0 | 8 | 8 |
| 64 | 20261118 | Staged Metal | 2 | 0 | 4 | 4 |
| 64 | 20261119 | Compiled MLX | 1 | 0 | 0 | 0 |
| 64 | 20261119 | Staged Metal | 1 | 0 | 0 | 0 |
| 4,096 | 20261118 | Compiled MLX | 2 | 0 | 2 | 1 |
| 4,096 | 20261118 | Staged Metal | 1 | 0 | 1 | 1 |
| 4,096 | 20261119 | Compiled MLX | 0 | 0 | 0 | 0 |
| 4,096 | 20261119 | Staged Metal | 1 | 0 | 0 | 0 |

On the failing seed, substituting just the GPU residual was sufficient to cross tolerance at both lengths. Substituting just the heads passed for both implementations and both seeds. The heads still influence the outcome when combined with a different residual. The passing seed illustrates that interaction: a mixed pair can fail even when the two inputs from its production run pass.

The crossover locates a difference at the boundary between the layers. It cannot distinguish the individual operations in layer 10 that contributed to it. A CPU finish on a GPU’s *own* inputs passed against that GPU’s observed output in all eight cases. Individual channels can still land on opposite sides of the tolerance relative to the original CPU baseline.

## The eight expert contributions

Each implementation chose the identical ordered expert IDs. Summing its eight recorded output vectors with its own route weights reproduced the observed expert output bit for bit. Two offline swaps then varied the route weights and expert vectors independently.

| Tokens | Seed | GPU inputs from | GPU weights only | GPU expert vectors only | CPU experts on GPU expert input |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 20261118 | Compiled MLX | 0 | 8 | 8 |
| 64 | 20261118 | Staged Metal | 0 | 4 | 4 |
| 64 | 20261119 | Compiled MLX | 0 | 1 | 0 |
| 64 | 20261119 | Staged Metal | 0 | 1 | 0 |
| 4,096 | 20261118 | Compiled MLX | 0 | 1 | 1 |
| 4,096 | 20261118 | Staged Metal | 0 | 2 | 1 |
| 4,096 | 20261119 | Compiled MLX | 0 | 0 | 0 |
| 4,096 | 20261119 | Staged Metal | 0 | 0 | 0 |

The recorded GPU route weights alone never pushed the CPU expert vectors across tolerance. The vectors alone did at both lengths for the failing seed. Recomputing the experts in NumPy on the recorded GPU input also retained failures. This input sensitivity helps explain how a cancellation in the expert sum can expose small differences already present before the sum.

The acquisition used Apple M4 Pro, MLX 0.32.4 and source `f807d8a4329016e1d365e31fefd3a44796f7442ccb80d4b1f50d227acc706237`. Run `.venv/bin/python -m analysis.crossover_report` from the project root to verify the array digests and rerun all nine CPU finishes per fixture. The script also prints the individual violating channel IDs.

[Four captured fixtures](../results/m4pro-02-followup-20261119T020918Z-tf3imdmi/) · [Original stack parity](stack-parity.md)
