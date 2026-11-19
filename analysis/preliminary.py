"""Regenerate the preliminary attention-only table and figure from the first Mac runs."""
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from bench.protocol import ATTENTION, PILOT, REPO, SCHEMA, SCHEDULES
from bench.report import LABELS, intervals, read_case


HERE = Path(__file__).resolve().parent
COLORS = {"compiled": "#275777", "staged": "#8c638a", "hybrid": "#007c74",
          "single": "#ad4b23", "split": "#253129"}
SHAPES = {"compiled": "o", "staged": "s", "hybrid": "^", "single": "D", "split": "X"}


def load(session, suite, cases):
    directory = REPO / "results" / session / suite
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = manifest["spec"]
    if ((spec["session"], spec["suite"], spec["schema"]) != (session, suite, SCHEMA)
            or manifest["environment"]["gpu_probe"]["passed"] is not True):
        raise ValueError(f"Invalid session manifest at {directory}")
    values = {}
    digests = {}
    for case in cases:
        wall = read_case(directory, case, spec)
        proof = json.loads((directory / (case.name + ".proof.json")).read_text())
        digests[case.name] = proof["sha256"]
        with (directory / (case.name + ".csv")).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        gpu = {}
        for method, expected in (("staged", 3), ("single", 1), ("split", 4)):
            if method in case.methods:
                selected = [r for r in rows if r["method"] == method]
                if {r["encoded_dispatches"] for r in selected} != {str(expected)}:
                    raise ValueError(f"Unexpected dispatch count, {case.name} {method}")
                gpu[method] = np.median([float(r["native_gpu_us"]) for r in selected])
        values[case] = wall, gpu
    return spec, values, digests


def split_effects(records, rng):
    result = {}
    for label, cases in records.items():
        for case in PILOT:
            wall, _ = cases[case]
            columns = [case.methods.index(method) for method in ("single", "split")]
            _, _, ratio, ci = intervals([wall[:, columns]], rng)
            result[label, case] = ratio[1], ci[:, 1]
    return result


def make_figure(sweep, effects, provenance):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
                         "svg.hashsalt": "m4pro-attention", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, comparison) = plt.subplots(1, 2, figsize=(11.4, 4.6), width_ratios=(1.6, 1))
    fig.subplots_adjust(left=.077, right=.98, bottom=.21, top=.7, wspace=.32)
    fig.text(.077, .93, "Splitting the KV scan across threadgroups", fontsize=20, weight="bold")
    fig.text(.077, .855, "Preliminary  /  Apple M4 Pro  /  isolated global attention, batch one",
             fontsize=10, color="#45515a")
    cases = [case for case in ATTENTION if case.batch == 1]
    for method in (*SCHEDULES, "split"):
        used = [case for case in cases if method in case.methods]
        ax.plot([case.length for case in used],
                [np.median(sweep[case][0][:, case.methods.index(method)]) for case in used],
                marker=SHAPES[method], markersize=6 if method == "split" else 4.5,
                color=COLORS[method], linewidth=1.7, linestyle="None" if method == "split" else "-",
                label=LABELS[method])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([case.length for case in cases], [f"{case.length:,}" for case in cases])
    ax.yaxis.set_major_locator(FixedLocator([200, 400, 800, 1600, 3200]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Median latency across the context sweep", loc="left", pad=13, fontweight="bold")
    ax.set_xlabel("Visible tokens")
    ax.set_ylabel("Synchronized wall time (µs, log scale)")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8, columnspacing=1.1)
    comparison.set_title("Single / split wall time", loc="left", pad=13, fontweight="bold")
    ypos = [3.0, 2.38, 1.05, .43]
    labels = []
    for case, group in zip(PILOT, (ypos[:2], ypos[2:])):
        for (label, color, marker), y in zip(
            (("Sweep", "#173b50", "o"), ("Pilot", "#728797", "s")), group
        ):
            ratio, (lo, hi) = effects[label, case]
            comparison.hlines(y, lo, hi, color=color, linewidth=1.5)
            comparison.plot(ratio, y, marker, color=color, markersize=5)
            comparison.text(max(ratio, hi) + .14, y, f"{ratio:.2f}×", va="center", fontsize=8)
            labels.append(f"{case.length:,}  ·  {label}")
    comparison.axvline(1, ls="--", color="#777777", lw=1, zorder=0)
    comparison.set_yticks(ypos, labels)
    comparison.set_ylim(0, 3.55)
    comparison.set_xlim(0, max(7.7, max(ci[1] for _, ci in effects.values()) + 1.3))
    comparison.set_xticks([1, 2, 4, 6], ["1×", "2×", "4×", "6×"])
    comparison.set_xlabel("Ratio of medians  ·  above 1 favors split")
    for panel in (ax, comparison):
        panel.grid(axis="y" if panel is ax else "x", color="#dce2e5", linewidth=.7)
        panel.set_axisbelow(True)
    fig.text(.077, .04, "Each point uses 100 paired trials across two fixtures. Whiskers show within-run 95% bootstrap intervals.",
             fontsize=8.5, color="#45515a")
    fig.savefig(HERE / "attention-preliminary.svg",
                metadata={"Date": None, "Title": "Splitting the KV scan across threadgroups",
                          "Description": json.dumps(provenance, sort_keys=True)})
    plt.close(fig)


def table(rows):
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    sweep_spec, sweep, sweep_digests = load("m4pro-01-a", "full", ATTENTION)
    pilot_spec, pilot, pilot_digests = load("m4pro-01-pilot", "pilot", PILOT)
    if any(sweep_spec[key] != pilot_spec[key] for key in
           ("machine", "seed", "trials", "warmups", "fixtures")):
        raise ValueError("The pilot and sweep protocols differ")
    records = {"Sweep": sweep, "Pilot": pilot}
    effects = split_effects(records, np.random.default_rng(71023))
    provenance = {
        "source_sha256": sweep_spec["machine"]["sources_sha256"],
        "sweep_session": sweep_spec["session"],
        "pilot_session": pilot_spec["session"],
        "case_sha256": {sweep_spec["session"]: sweep_digests, pilot_spec["session"]: pilot_digests},
    }
    make_figure(sweep, effects, provenance)
    schedules = (*SCHEDULES, "split")
    rows = [["Run", "Batch", "Visible tokens", *(LABELS[m] for m in schedules)],
            ["---", "---:", "---:", *(["---:"] * len(schedules))]]
    for label, values in records.items():
        for case, (wall, _) in values.items():
            med = dict(zip(case.methods, np.median(wall, axis=0)))
            rows.append([label, str(case.batch), f"{case.length:,}",
                         *(f"{med[m]:,.1f}" if m in med else "—" for m in schedules)])
    checks = [["Run", "Tokens", "Single / split wall time (95% CI)", "Single GPU µs", "Split GPU µs"],
              ["---", "---:", "---:", "---:", "---:"]]
    for case in PILOT:
        for label, values in records.items():
            _, gpu = values[case]
            ratio, (lo, hi) = effects[label, case]
            checks.append([label, f"{case.length:,}", f"{ratio:.2f}× ({lo:.2f}–{hi:.2f})",
                           f"{gpu['single']:,.1f}", f"{gpu['split']:,.1f}"])
    document = f"""# Preliminary M4 Pro attention results

At batch one and 4,096 visible tokens, the split-context Metal schedule finished in {np.median(sweep[PILOT[1]][0][:, -1]):,.0f} µs. Single-dispatch Metal took {np.median(sweep[PILOT[1]][0][:, -2]):,.0f} µs. The earlier pilot also found a greater than sixfold wall-time difference. At 64 tokens, the sweep’s interval included a tie and the pilot favored splitting.

![Synchronized wall time by KV length and two-run comparison of the single and split schedules](attention-preliminary.svg)

## Performance comparison

Each cell reports median synchronized wall time in microseconds for one decode step of the isolated global-attention path. The split control was measured at the two batch-one endpoints. A dash marks an unmeasured combination.

{table(rows)}

## What changes when the context is split?

Single dispatch assigns one threadgroup to each KV head. At batch one, eight threadgroups scan the entire history. The split schedule prepares the new token, scans 32-token chunks, merges their softmax statistics, then applies the gate. At 4,096 tokens its scan issues 1,024 threadgroups across eight KV heads. The staged Metal schedule also separates preparation and gating, but its scan still uses eight threadgroups. These grid dimensions describe available parallel work. A hardware trace would be needed to quantify achieved occupancy.

{table(checks)}

The ratio divides the median single-dispatch wall time by the median split-context wall time. Intervals resample paired blocks within each fixture, with 2,000 draws per run. They quantify trial variation within each observed run. GPU columns give the native Metal command buffer’s median elapsed time. The native encoder recorded one dispatch for single, three for staged and four for split. MLX and hybrid dispatch counts require a separate trace.

## Measurement scope

Both runs used the same Apple M4 Pro, MLX {sweep_spec["machine"]["mlx"]}, benchmark source and fixture seeds. Each schedule has {sweep_spec["trials"]} timings per cell over {sweep_spec["fixtures"]} token states. Inputs were resident before timing. The clock covers the caller’s dispatch through device completion. Schedule order was randomized within each paired block. Every reported case passed the CPU comparison and the post-timing check, and its CSV digest and pairings were verified.

The sweep uses seven certified attention cases from the first full-study attempt. The pilot independently collected the two batch-one endpoints. The table keeps their medians separate. Decoder-stack performance and the two-session publication report require complete compatible runs.

Run `.venv/bin/python -m analysis.preliminary` from the project root with the original [sweep](../results/m4pro-01-a/full/) and [pilot](../results/m4pro-01-pilot/pilot/) case files present. See the [benchmark protocol](../bench/README.md) and [Inkling-inspired architecture](../docs/architecture.md).
"""
    (HERE / "attention-preliminary.md").write_text(document)


if __name__ == "__main__":
    main()
