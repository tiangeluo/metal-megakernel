"""Rebuild the article's M4 Pro attention figure and table from certified raw trials."""
import numpy as np

from analysis.preliminary import HERE, load, make_figure, split_effects, table
from bench.protocol import ATTENTION, PILOT, SCHEDULES
from bench.report import LABELS, intervals


def main():
    names = ("m4pro-02-a", "m4pro-02-b")
    sources = [load(name, "attention", ATTENTION, require_complete=True) for name in names]
    first, second = (entry[0] for entry in sources)
    if any(first[k] != second[k] for k in ("machine", "seed", "trials", "warmups", "fixtures")):
        raise ValueError("The two sessions have different machines, sources or fixtures.")

    records = {label: values for label, (_, values, _) in zip(("Run A", "Run B"), sources)}
    pooled = {case: np.concatenate([values[case][0] for values in records.values()]) for case in ATTENTION}
    rng = np.random.default_rng(71023)
    effects = split_effects(records, rng)
    provenance = {
        "source_sha256": first["machine"]["sources_sha256"], "sessions": list(names),
        "case_sha256": {name: digests for name, (_, _, digests) in zip(names, sources)},
    }
    make_figure(pooled, effects, provenance, path=HERE / "attention-verified.svg",
                subtitle="Apple M4 Pro  /  isolated global attention, batch one  /  two complete sweeps",
                groups=tuple(records),
                caption="The curve pools 200 paired blocks per case. Whiskers show each session’s 95% paired bootstrap interval for the ratio.")

    rows = [["Batch", "Visible tokens", *(LABELS[m] for m in (*SCHEDULES, "split"))],
            ["---:", "---:", *(["---:"] * 5)]]
    for case in ATTENTION:
        medians = dict(zip(case.methods, np.median(pooled[case], axis=0)))
        rows.append([str(case.batch), f"{case.length:,}",
                     *(f"{medians[m]:,.1f}" if m in medians else "—" for m in (*SCHEDULES, "split"))])

    comparisons = [["Tokens", "Run", "Single µs", "Split µs", "Single / split (95% CI)", "Single GPU µs", "Split GPU µs"],
                   ["---:", "---", "---:", "---:", "---:", "---:", "---:"]]
    for case in PILOT:
        cols = [case.methods.index(m) for m in ("single", "split")]
        session_pairs = []
        for label, values in records.items():
            wall, gpu = values[case]
            pair = wall[:, cols]
            session_pairs.append(pair)
            ratio, (lo, hi) = effects[label, case]
            comparisons.append([f"{case.length:,}", label, *(f"{x:,.1f}" for x in np.median(pair, axis=0)),
                                f"{ratio:.2f}× ({lo:.2f}–{hi:.2f})", f"{gpu['single']:,.1f}", f"{gpu['split']:,.1f}"])
        med, _, ratios, bounds = intervals(session_pairs, rng)
        comparisons.append([f"{case.length:,}", "Pooled", *(f"{x:,.1f}" for x in med),
                            f"{ratios[1]:.2f}× ({bounds[0, 1]:.2f}–{bounds[1, 1]:.2f})", "—", "—"])

    report = f"""# Attention on the M4 Pro

![Pooled latency across the context sweep and individual single-versus-split ratios](attention-verified.svg)

## Median latency by schedule

Each cell gives synchronized wall time in microseconds for one cached global-attention step, pooled across two complete sessions. Each session contributes 100 paired blocks per case across two seeded states. The split control ran at the two batch-one endpoints.

{table(rows)}

## How well did the split effect repeat?

{table(comparisons)}

Each ratio divides the median single-dispatch time by the median split-context time. The 95% intervals use 2,000 bootstrap draws, resampling whole paired blocks within each session and fixture. They describe trial variability on the recorded Mac in these two consecutive processes. Run A and Run B had different randomized schedule orders. Both started on AC power with the reported thermal state nominal.

At 4,096 tokens, single dispatch gives eight threadgroups the entire cache scan. Splitting it into 32-token chunks issues 1,024 scan threadgroups followed by a merge. The native encoder records one dispatch for single, three for staged and four for split. The GPU columns come from Metal command-buffer timestamps. Achieved occupancy and the internal MLX dispatch counts require a trace.

At 64 tokens, the fastest of the four main schedules changed from staged in A to hybrid in B. The separate split control had the lowest median at both tested endpoints in both sessions. Among the four main schedules, hybrid had the lowest median in both sessions at every measured length from 256 onward and at batch eight. The batch-eight row measures latency for eight simultaneous sequences.

## Reproduce

Run `.venv/bin/python -m analysis.verified` from the repository root. The generator checks the CSV digests, pre- and post-timing certificates, pairing, hardware, source revision and seeded fixtures. Each figure embeds its source and per-case CSV hashes. The single, staged, split and MLX timings all charge CPU dispatch and GPU completion. The prepared inputs are device resident when timing starts. Whole-decoder timings have a separate correctness gate.

Benchmark source SHA-256 `{provenance['source_sha256']}`. [Run A](../results/{names[0]}/attention/) · [Run B](../results/{names[1]}/attention/) · [Registered measurement protocol](../bench/README.md)
"""
    (HERE / "attention-verified.md").write_text(report)


if __name__ == "__main__":
    main()
