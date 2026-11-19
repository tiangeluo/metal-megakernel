# Megakernels on a Mac

Should a Metal kernel that prepares one token's keys and values also scan its KV cache?

This repository compares compiled MLX, staged Metal, hybrid Metal/MLX and single-dispatch Metal on an Inkling-inspired cached decoder. The isolated global-attention measurement preserves the published 48 query heads, eight KV heads and 128-dimensional heads. The connected two-layer experiment uses a 1,152-wide residual stream, retains separate global Q/K/V depth reads, and routes each token to eight of 384 experts in each layer. The [architecture](docs/architecture.md) explains both workloads and the exact benchmark boundary.

## Current evidence

Read the [article in your browser](docs/article.html) or open the [Markdown source](docs/article.md). To rebuild the figure, measured tables and tracker from the included trial files, run `./mac-run report` with NumPy and Matplotlib installed. If Pandoc 3 is available, the same command regenerates the HTML article with its figures embedded. Report generation runs on CPU.

Two complete M4 Pro attention sweeps measured all seven workloads. At batch one and 4,096 tokens, the median synchronized latency was 2,875 µs for single-dispatch Metal, 432 µs for the split-context control and 476 µs for compiled MLX. The single-to-split ratios were 6.63× and 6.74× in the two sessions. All measurements and correctness certificates are included. [Figure and timing table](analysis/attention-verified.md)

The connected two-layer precision control passed all 15 registered output and state checks for each schedule, fixture and endpoint before and after timing. At 4,096 tokens, compiled MLX and hybrid took 1.562 and 1.560 ms. Staged and single-dispatch Metal took 5.036 and 6.017 ms. Both layers retained FP16 expert weights and used FP32 expert arithmetic. The split-context schedule was measured only for isolated attention. [Connected timing table](results/precision-control-01/report.md) · [Article and figure](docs/article.md)

One fixture exceeded tolerance in the original FP16 expert-output history, so the original stack has no certified timings. Both seeds passed its final residual and expert-selection checks. [Original stack diagnostics](analysis/stack-parity.md)

Four Instruments captures isolate a single warmed global-attention step at 4,096 tokens. Each shows one command buffer and one compute encoder in the target process. Shader dispatches within the encoder remain unresolved in these captures. [Trace inspection and provenance](analysis/trace-observations.json)

The independent CPU prefilling and incremental decoding implementations have been checked at the published head geometry. At 64 successive tokens, the maximum absolute gated-output difference was 0.0009765625. The connected decoder completed CPU forward passes at visible lengths 64 and 4,096 with both full expert banks allocated. See the [decoder smoke checks](decoder/cpu-smoke.json) and [standalone progress page](dist/index.html) for the evidence and GPU run status.

The first M4 Pro sweep certified all seven isolated attention cases. At batch one and 4,096 tokens, the single-dispatch implementation took 2,840 µs and the split-context control took 436 µs. The pilot measured 2,861 and 453 µs. The first stack precheck stopped on four elements in the second layer’s expert-output history. Both runs used benchmark source from commit `29cf099`. Their original CSVs, parity certificates and manifests are included under `results/`. [Measured table and figure](analysis/attention-preliminary.md)

## Run on an Apple-silicon Mac

Use a local macOS session with working Metal access, Xcode command-line tools, and uv or Python 3.11–3.13. The reference storage precision is FP16 and reductions use FP32. Dependencies are pinned in [requirements-mac.txt](requirements-mac.txt). Runtime shader compilation is used on the GPU.

From the root of the checkout:

```sh
./mac-run probe
./mac-run study
```

The probe dispatches 128 threads using 22 KiB of threadgroup storage and checks every output. The study runs the tests and pilot, then replays all four schedules on each of the two stack fixtures at 64 and 4,096 tokens. For a failed fixture, the replay records the affected schedule, observable, coordinate, and a stage-by-stage comparison using the actual inputs to the first failing schedule. It then measures two independent attention sessions and writes `results/report-attention/`. If the stack checks pass, the study also measures both stack sessions and produces the combined report. Each timed cell still has to pass the CPU comparison before and after timing.

The completed `m4pro-02` attention sessions and all four decoder diagnostic records are included. To collect another independent study, use a fresh label. Reuse a label to resume interrupted work on the same code revision. The study records the environment, power state and available thermal data for each suite. Its HTML tracker refreshes on disk as each case or diagnostic completes.

Use `./mac-run followup` on the Mac to investigate the original stack discrepancy. It captures the incoming residual and gated attention heads at layer 11 for NumPy, compiled MLX and staged Metal, then swaps the sources independently and reruns the same expert calculation. Each capture must reproduce the original wrapper outputs exactly. The command covers the failing seed and passing control at both lengths, saves small arrays for offline analysis, and attempts a separate long-context Instruments capture for four attention schedules. Trace collection has its own status. The results directory printed at launch holds the evidence, separately from the timing sessions. This follow-up does not repeat the attention sweeps.

The original crossover is complete. The captured incoming residual alone reproduced expert-output discrepancies at both tested lengths. The follow-on control keeps the expert weights in FP16 and performs the expert computation in FP32. On the recorded inputs, all 12 pairwise CPU comparisons of the control’s layer-11 expert output passed the original tolerance. A full connected CPU replay under an alternate projection summation order passed all 15 checks on both seeds and lengths. The control uses a different expert projection kernel, so any certified latency describes that implementation. [Crossover results](analysis/crossover-results.md) · [Captured-input precision replay](analysis/moe-precision-replay.json) · [Connected CPU replay](analysis/precision-sensitivity.json)

The completed `precision-control-01` sessions are included. To run or resume the precision workflow from a Metal-visible terminal, use `./mac-run precision`. It checks the new GPU projection against a CPU reference, validates the four connected schedules, and writes a separate report under `results/precision-control-01/` after two compatible sessions pass. It also retries the original isolated attention traces with the project’s MLX environment. The trace captures run after timing. Refresh `dist/index.html` when the command finishes.

For just the two-point pilot:

```sh
./mac-run pilot
```

The generated files include raw per-trial CSVs, parity checks, the publication-sized figure at `results/report/latency.svg`, the comparison table at `results/report/performance.md`, and a refreshed progress page at `dist/index.html`. The pilot shows preliminary medians in the page. The combined report uses two completed compatible sessions. Profile representative cases separately using the [Instruments instructions](bench/README.md).

For an existing downloaded tracker, `./mac-run connect` verifies the known copy in Downloads and replaces it with a link to this project's progress page. Refresh the open Chrome tab as the study completes cases. Other users can open `dist/index.html` directly. The tracker is self-contained when downloaded as a snapshot.

## Run the CPU checks separately

```sh
python -m pytest -q
python -m reference.validate
python -m decoder.benchmark --method cpu --length 64
```

The final command allocates approximately 2.85 GiB for the two FP16 expert banks. Add `--compact` for a quick functional check. The [reference contract](reference/README.md) records upstream revisions, numerical tolerances, layout and rounding choices.

## Source map

| Folder | Contents |
| --- | --- |
| `reference/` | Batched cached NumPy reference, independent dense FP64 oracle, and deterministic fixtures |
| `native/` | Staged, single and split-context Metal schedules, Objective-C++ bridge and GPU parity tests |
| `mlx_backend/` | Compiled MLX baseline, custom Metal preparation with MLX attention, and trace capture |
| `decoder/` | Two connected layers, expert routing, CPU and GPU adapters |
| `bench/` | Paired timing harness, correctness gates, analysis, plotting and trace markers |
| `docs/` | Architecture figure and article draft |
| `tracker/` | Task ledger and portable progress-page generator |

The isolated comparison measures one attention path. The two-layer comparison charges projections, depth reads, expert execution, cache updates and backend handoffs. Inputs and weights are seeded. These timings characterize this M4 Pro workload.
