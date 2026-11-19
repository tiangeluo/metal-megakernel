# Megakernels on a Mac

Should a Metal kernel that prepares one token's keys and values also scan its KV cache?

This repository compares compiled MLX, staged Metal, hybrid Metal/MLX and single-dispatch Metal on an Inkling-inspired cached decoder. The isolated global-attention measurement preserves the published 48 query heads, eight KV heads and 128-dimensional heads. The connected two-layer experiment uses a 1,152-wide residual stream, retains separate global Q/K/V depth reads, and routes each token to eight of 384 experts in each layer. The [architecture](docs/architecture.md) explains both workloads and the exact benchmark boundary.

## Current evidence

The independent CPU prefilling and incremental decoding implementations have been checked at the published head geometry. At 64 successive tokens, the maximum absolute gated-output difference was 0.0009765625. The connected decoder completed CPU forward passes at visible lengths 64 and 4,096 with both full expert banks allocated. See the [decoder smoke checks](decoder/cpu-smoke.json) and [standalone progress page](dist/index.html) for the evidence and GPU run status.

## Run on an Apple-silicon Mac

Use a local macOS session with working Metal access, Xcode command-line tools, and uv or Python 3.11–3.13. The reference storage precision is FP16 and reductions use FP32. Dependencies are pinned in [requirements-mac.txt](requirements-mac.txt). Runtime shader compilation is used on the GPU.

From the root of the checkout:

```sh
./mac-run probe
./mac-run study
```

The probe compiles a native host program, dispatches 128 threads using 22 KiB of threadgroup storage, and checks every output. The study command creates a project-local Python environment, runs the tests, then runs the two-point pilot and two full timing sessions in separate processes. Each method must match the CPU oracle before a cell is timed. A failure leaves already checked cells available for inspection. Use a stable label such as `./mac-run study m4pro-01` to resume the same run after interruption. Remote changes to configs or source require a new label. Each suite records the environment, including power state and available thermal data.

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
