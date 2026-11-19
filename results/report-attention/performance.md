# Decode latency on Apple M4 Pro

Median synchronized wall time in µs. Brackets enclose 95% paired, stratified bootstrap intervals.

| Workload | Batch | Length | Compiled MLX | Staged Metal | Hybrid | Single dispatch |
|---|---:|---:|---:|---:|---:|---:|
| attention | 1 | 64 | 368.52 [351.38, 380.25] | 301.31 [291.38, 304.50] | 300.50 [288.31, 318.59] | 320.52 [316.73, 323.60] |
| attention | 1 | 4096 | 476.02 [469.73, 485.29] | 2146.29 [2137.25, 2154.02] | 448.46 [442.25, 456.77] | 2875.27 [2863.12, 2886.12] |
| attention | 8 | 4096 | 1683.65 [1666.83, 1695.62] | 3126.08 [3119.50, 3134.75] | 1618.67 [1605.88, 1627.08] | 3583.02 [3568.70, 3599.75] |

measurements.csv includes speedups, confidence intervals and each session’s median and winner. Raw timing CSVs distinguish native command-buffer GPU intervals from total wall time. Encoded native dispatch counts describe the submitted schedule. The observed_dispatches and trace_path columns carry inspected Instruments evidence when supplied.

Sessions: m4pro-02-a, m4pro-02-b

Source SHA-256: f807d8a4329016e1d365e31fefd3a44796f7442ccb80d4b1f50d227acc706237
