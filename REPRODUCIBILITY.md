# Reproduction guide for *Studying Megakernels on a Mac*

The article reports **6,000 isolated-attention timings** from `m4pro-02-a/b` and **1,600 two-layer timings** from `precision-control-01/a,b`. The repository contains each trial, the parity certificates and the code that computes the medians and paired bootstrap intervals.

## Rebuild the published results on CPU

From a clean checkout or extracted ZIP, run the following in its root. The reference environment was macOS on arm64 with CPython 3.13.9. The CPU path uses NumPy and Matplotlib. The release pins their dependencies and includes pytest for the tests.

```sh
uv venv --python 3.13.9 .venv
uv pip install --python .venv/bin/python -r requirements-report.txt
./mac-run verify
./mac-run report
```

`verify` reads the archive before you regenerate anything. It checks the evidence digests, separates source revisions, recomputes 38 measurement rows including the interval bounds, and compares the rounded values with the five rows printed in the article. It also replays the original layer-11 comparisons from four saved array captures and attributes four Instruments windows using the archived process-specific rows.

`report` regenerates the [article figure](analysis/decoder-performance.svg), [attention table](analysis/attention-verified.md), [two-layer table](results/precision-control-01/report.md) and the [HTML tracker](dist/index.html). With Pandoc 3.8 on the path, it also rebuilds the [standalone HTML article](docs/article.html). Run `shasum -a 256 -c analysis/release-outputs.sha256` to compare the four SVGs, six table files and HTML article with the reference build, byte for byte. The tracker displays the local time of its rebuild. Run `.venv/bin/python -m pytest -q` for the test suite. GPU-specific tests require a Metal-visible device.

## Collect independent GPU measurements

The acquisition machine was a 20-GPU-core M4 Pro MacBook Pro with 48 GB of unified memory, on AC power, running macOS 26.6.2, Xcode 26.6 (17F113), CPython 3.13.9 and MLX/MLX Metal 0.32.4. Both attention sweeps began at nominal thermal state. Use a local Apple-silicon macOS session with working Metal access, Xcode command-line tools and uv. Leave the GPU idle before starting. The runner installs the pinned Mac requirements and records the actual device, software and power state.

```sh
./mac-run replicate my-repeat
```

The runner takes the study lock and creates `results/replications/my-repeat/`. Choose an unused label for each repeat. It probes Metal, starts two independent processes per workload, retains the original fixture seeds and method-order randomization, and validates outputs before and after timing. The attention study covers seven cases. The connected study uses the FP32 expert control at both context lengths. Add `--traces` to the same command to capture four warmed, marked steps **after** the timing sessions.

| Output inside the new run | What to compare |
| --- | --- |
| `attention-report/measurements.csv` | Synchronized wall time for each case and schedule, including the split-context control at the batch-one endpoints |
| `precision-control/measurements.csv` | Two-layer medians, interval bounds and per-session values for all four schedules |
| `decoder-performance.svg` | Both workloads plotted from this repeat's trials |
| `attention/` and `precision-control/{a,b}/` | Raw timing CSVs, before/after parity certificates and machine manifests |

At batch one and 4,096 tokens, the recorded **single/split** attention ratios were **6.63× and 6.74×**. For the connected decoder at the same length, the **compiled/hybrid** pooled ratio was **1.001×**, with a paired 95% interval of **0.987–1.016×**. Compare those ratios and the direction of the effect per session. Absolute latency can vary with device, runtime, temperature and other GPU work. The published intervals describe timing-block variation within the recorded sessions. A new Mac can produce a different interval.

## Numerical gates and provenance

The [reference contract](reference/README.md) sets the gated-output tolerances to `atol=0.002`, `rtol=0.01` and the written-cache tolerances to `atol=0.002`, `rtol=0.002`. Unchanged cache entries and raw convolution histories must match exactly. All four connected schedules must pass 15 registered output and state checks against the control’s CPU reference on both seeds and lengths, before and after timing.

The original FP16 expert stack exceeded tolerance on one seed. Its saved arrays reproduce the layer-11 discrepancy. Only the separately identified control with FP16 expert banks and FP32 expert arithmetic contributes connected timings. The earlier `m4pro-01` pilot and partial sweep belong to Git revision `29cf099`. Their evidence remains separate from the published medians. To inspect that historical code from the optional offline Git bundle, run `git clone metal-megakernel-history.bundle pilot-history` and then `git -C pilot-history switch --detach 29cf099`.

The [checksum inventory](analysis/release-inputs.sha256) identifies the recorded inputs. The [trace tables](analysis/trace-tables.xml) retain the benchmark process's relevant rows for all four captures. Run `.venv/bin/python -m analysis.trace_replay` to recompute the command-buffer, encoder and active GPU intervals. The full system-wide Instruments bundles remain on the acquisition Mac. These captures resolve one compute encoder per marked step and leave individual shader dispatches and achieved occupancy unresolved.
