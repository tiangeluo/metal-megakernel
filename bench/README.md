# Decode benchmark protocol

From the repository root, run the hardware probe and the pinned GPU environment through `./mac-run pilot`. A pilot measures lengths 64 and 4,096, including a split-context control. For the complete table, collect two full sessions in separate processes, preferably with a cooldown and the same power mode.

```sh
./mac-run full s1
./mac-run full s2
.venv/bin/python -m bench.report --input results --sessions s1 s2 --out results/report
```

The harness requires a fresh positive native Metal probe and verifies an MLX computation on the GPU. Each schedule first has to pass output and state comparisons against the independent CPU reference. Preparation and reset finish outside the timed region. The primary clock starts just before dispatch from the common Python caller and stops after device completion. MLX and native inputs are already device-resident. Native command-buffer timestamps are a separate raw column. The hybrid and connected decoder charge crossings between backends.

Each session collects at least 100 randomized, paired blocks per case. All methods see the same snapshot in each block. Each successive pair of blocks uses both fixtures in randomized order. The two decoder fixtures reuse the same expert banks. Changing the session name changes interleaving order while preserving fixture seeds. Every schedule gets another correctness check after timing. A completed case writes raw CSV, a digest, and parity evidence. Repeating the same command can resume certified cases after interruption. Use a fresh session label if the code, versions or settings change.

`bench.report` verifies certificates, complete pairings and source/environment agreement in two independent collection sessions. It writes `performance.md`, `measurements.csv` and `latency.svg`. The CSV includes medians, the ratio of median latencies against compiled MLX, and 95% intervals from 2,000 bootstrap draws. Each draw resamples paired blocks within each session and fixture. Intervals describe variability in the collected sessions. A column flags changes in the fastest implementation between sessions. No report is generated from unavailable, partial or unverified cells. For an attention-only publication, collect two sessions with `python -m bench.run --suite attention`, then use `python -m bench.report --scope attention`.

## Verify dispatch counts

Use a separate profiling run. List templates with `xcrun xctrace list templates` and enable Metal System Trace and Points of Interest. Where the installed Metal System Trace template includes these tracks, an example capture is:

```sh
xcrun xctrace record --template 'Metal System Trace' --output results/single-l64.trace --launch -- "$PWD/.venv/bin/python" -m bench.trace --case attention-b1-l64 --method single
```

Locate the “Decode step” signpost labeled with the case and method, then follow its command buffers and count GPU compute dispatches across the interval. The module warms and checks the result before opening the marker. It reads back the output again after closing it. Keep the Instruments trace, record its path and observed dispatches alongside that case, and collect the same trace for the comparison schedule. To include inspected counts in the publication CSV, supply `bench.report --trace-index results/trace-index.csv`. Use the columns `case,method,observed_dispatches,trace_path`. Each trace path must point to an existing .trace bundle relative to that index. Missing evidence leaves the report fields blank. The raw CSV field `encoded_dispatches` records how many dispatches the native code encoded. MLX dispatch counts require a trace. Benchmark timings come from the profiling-free runs.

## Protocol tests

```sh
python -m unittest bench.test_protocol
```

Tests synthesize paired data inside temporary directories, including the publication export test. Nothing from those fixtures is stored as a measured result.
