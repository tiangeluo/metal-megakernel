"""Reproduce the article figure from the certified M4 Pro attention and stack trials."""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

from analysis.preliminary import COLORS, SHAPES, load
from bench.protocol import ATTENTION, REPO, SCHEDULES, STACK
from bench.report import LABELS, metrics
from diagnostics.precision_study import load_verified


def measured_panels(results, folder):
    attention, digests, specs = {}, {}, []
    for name in ("m4pro-02-a", "m4pro-02-b"):
        spec, cases, digests[name] = load(name, "attention", ATTENTION, require_complete=True, results=results)
        attention[name] = {case.name: values[0] for case, values in cases.items()}
        specs.append(spec)
    stack, cohort = load_verified(folder)
    for field in ("machine", "fixtures", "trials", "seed", "warmups"):
        if any(s[field] != cohort[field] for s in specs):
            raise ValueError(f"Acquisitions differ in {field}.")
    control_digests = {
        name: {case.name: json.loads((folder/name/(case.name + ".proof.json")).read_text())["sha256"]
               for case in STACK} for name in stack
    }
    provenance = {"source_sha256": cohort["machine"]["sources_sha256"],
                  "control_sha256": cohort["control_sha256"], "control": cohort["control"],
                  "sessions": list(attention), "case_sha256": digests,
                  "control_sessions": list(stack), "control_case_sha256": control_digests}
    return metrics(attention, ATTENTION), metrics(stack, STACK), provenance, cohort["machine"]["device"]


@plt.style.context("default")
def draw(replication=None):
    if replication is None:
        results = REPO / "results"
        control = results / "precision-control-01"
        output = Path(__file__).with_name("decoder-performance.svg")
    else:
        results = replication / "attention"
        control = replication / "precision-control"
        output = replication / "decoder-performance.svg"
    attention, stack, provenance, device = measured_panels(results, control)
    title = "Decode latency on the " + device.removeprefix("Apple ")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "svg.fonttype": "none",
                         "svg.hashsalt": "m4pro-decoder-schedules", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#263238", "text.color": "#17282f",
                         "axes.edgecolor": "#8b989e", "xtick.color": "#43565e", "ytick.color": "#43565e"})
    fig, (curve, bars) = plt.subplots(1, 2, figsize=(11.8, 5.55), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.076, right=.974, bottom=.29, top=.72, wspace=.45)
    fig.text(.076, .935, title, fontsize=22, weight="bold")
    fig.text(.076, .874, "One cached decode step at batch one   ·   Median synchronized wall time", fontsize=11)
    for ax, panel_title, subtitle in [
            (curve, "Isolated global attention", "48 query heads / 8 KV heads"),
            (bars, "Two connected layers", "12 query heads / 2 KV heads per layer  ·  FP32 expert control")]:
        ax.text(0, 1.19, panel_title, transform=ax.transAxes, fontsize=14, weight="bold")
        ax.text(0, 1.085, subtitle, transform=ax.transAxes, fontsize=9.3, color="#43565e")
        ax.set_axisbelow(True)
    lengths = [case.length for case in ATTENTION if case.batch == 1]
    for method in (*SCHEDULES, "split"):
        rows = [r for r in attention if r["batch"] == 1 and r["method"] == method]
        values = np.array([r["wall_us"] for r in rows]) / 1000
        errors = np.array([[r["wall_us"]-r["lo_us"], r["hi_us"]-r["wall_us"]] for r in rows]).T / 1000
        curve.errorbar([r["visible_length"] for r in rows], values, yerr=errors, color=COLORS[method],
                       marker=SHAPES[method], markersize=7 if method=="split" else 4.6,
                       linestyle="none" if method=="split" else "-", linewidth=1.7, capsize=2.1)
    curve.grid(axis="y", color="#dce2e5", linewidth=.7)
    curve.set_xscale("log", base=2)
    curve.set_xticks(lengths, [f"{n:,}" for n in lengths])
    curve.set_yscale("log")
    batch_one = [r for r in attention if r["batch"] == 1]
    if all(170 <= r["lo_us"] and r["hi_us"] <= 3650 for r in batch_one):
        curve.set_ylim(.17, 3.65)
        curve.set_yticks([.2, .4, .8, 1.6, 3.2], ["0.2", "0.4", "0.8", "1.6", "3.2"])
    else:
        curve.yaxis.set_major_locator(LogLocator(base=2))
        curve.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    curve.yaxis.set_minor_formatter(NullFormatter())
    curve.set_ylabel("Latency (ms, log scale)")
    curve.set_xlabel("Visible tokens")
    short_color = "#c2d0d3"
    for index, method in enumerate(SCHEDULES):
        for length, yoff, color in [(64, .19, short_color), (4096, -.19, COLORS[method])]:
            r = next(r for r in stack if r["visible_length"] == length and r["method"] == method)
            value = r["wall_us"] / 1000
            y = -index + yoff
            bars.barh(y, value, height=.30, color=color)
            bars.hlines(y, r["lo_us"]/1000, r["hi_us"]/1000, color="#182a31", linewidth=1.2)
            bars.text(r["hi_us"]/1000 + .09, y, f"{value:.2f}", va="center", fontsize=9.1)
    bars.set_yticks([0, -1, -2, -3], [LABELS[m] for m in SCHEDULES])
    bars.set_ylim(-3.49, .52)
    stack_limit_ms = max(r["hi_us"] for r in stack) / 1000
    if 3 <= stack_limit_ms <= 6.25:
        bars.set_xlim(0, 6.95)
        bars.set_xticks([0, 2, 4, 6])
    else:
        bars.set_xlim(0, stack_limit_ms * 1.20 + .09)
    bars.grid(axis="x", color="#dce2e5", linewidth=.7)
    bars.set_xlabel("Latency (ms)")
    handles = [Line2D([], [], marker=SHAPES[m], color=COLORS[m], markersize=5.2,
                      linestyle="none" if m=="split" else "-", label=LABELS[m])
               for m in (*SCHEDULES, "split")]
    curve.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.018,-.25),
                 ncol=3, fontsize=8.5, handlelength=1.5, columnspacing=1.2, frameon=False)
    bars.legend(handles=[Patch(color=short_color, label="64 tokens"),
                         Patch(facecolor="#47606a", label="4,096 tokens (schedule colors)")],
                loc="upper left", bbox_to_anchor=(-.025,-.25), ncol=2, fontsize=8.5,
                handlelength=1.1, columnspacing=1.1, frameon=False)
    fig.text(.076, .078, "The split-context control ran only in isolated attention at 64 and 4,096 tokens. "
             "The stack includes projections, routing and eight active experts per layer.", fontsize=8.8)
    fig.text(.076, .035, "Whiskers show paired 95% bootstrap intervals within the two recorded sessions. "
             "Each median pools 200 timing blocks per method.", fontsize=8.8, color="#43565e")
    fig.savefig(output, metadata={"Date": None, "Title": title,
                          "Description": json.dumps(provenance, sort_keys=True)})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication", type=Path, help="Use the sessions in this fresh replication folder")
    draw(parser.parse_args().replication)
