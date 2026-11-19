"""Build the publication figure and measured table from independent, verified GPU sessions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .protocol import ATTENTION, FIXTURES, LENGTHS, MIN_REPEATS, SCHEMA, SCHEDULES, STACK, TABLE, finite_time

LABELS = {"compiled":"Compiled MLX","staged":"Staged Metal","hybrid":"Hybrid","single":"Single dispatch","split":"Split-context control"}


def read_case(directory, case, spec):
    path = directory / (case.name+".csv")
    proof = json.loads(path.with_suffix(".proof.json").read_text())
    if (proof.get("schema") != SCHEMA or proof.get("case") != case.name
            or proof.get("rechecked_after_timing") is not True
            or proof.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError(f"Invalid parity proof for {path}")
    for fixture in range(spec["fixtures"]):
        recorded = proof["oracle_max_abs"].get(str(fixture), {})
        if set(recorded) != set(case.methods) or any(not recorded[k] for k in case.methods):
            raise ValueError(f"Missing preflight proof for {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n, methods = spec["trials"], case.methods
    if n < MIN_REPEATS or spec["fixtures"] != FIXTURES or len(rows) != n*len(methods) or len(rows) != proof["row_count"]:
        raise ValueError(f"Insufficient samples or fixtures in {path}")
    wall = np.zeros((n,len(methods)))
    filled = set()
    orders = set()
    for row in rows:
        block, fixture, order = (int(row[k]) for k in ("block","fixture","order"))
        method = row["method"]
        if (row["session"] != spec["session"] or row["case"] != case.name or method not in methods
                or not 0 <= block < n or fixture != block%FIXTURES or not 0 <= order < len(methods)
                or (block,method) in filled or (block,order) in orders):
            raise ValueError(f"Unpaired, duplicate or mislabeled observation in {path}")
        wall[block,methods.index(method)] = finite_time(row["wall_us"], "wall time")
        if row["native_gpu_us"]:
            finite_time(row["native_gpu_us"], "GPU time", allow_zero=True)
        if method in ("compiled","hybrid") and row["native_gpu_us"]:
            raise ValueError("MLX host timing was labeled as a GPU interval")
        filled.add((block,method))
        orders.add((block,order))
    return wall


def load_sessions(root, selected, cases):
    """Choose source once per case and session; repeated suite runs cannot inflate n."""
    manifests = sorted(root.glob("*/*/manifest.json"))
    sessions = {}
    requested = set(selected or [])
    for file in manifests:
        data = json.loads(file.read_text())
        spec = data["spec"]
        name = spec["session"]
        if requested and name not in requested:
            continue
        if (spec.get("schema") != SCHEMA or data.get("status") != "complete"
                or data.get("environment",{}).get("gpu_probe",{}).get("passed") is not True):
            continue
        if (name,spec["suite"]) != (file.parent.parent.name,file.parent.name):
            raise ValueError("Session metadata does not match its directory")
        sessions.setdefault(name,{})[spec["suite"]] = (file.parent,spec)
    verified, cohort = {}, None
    for name, suites in sessions.items():
        per_case = {}
        settings = None
        for case in cases:
            options = ("full", "stack") if case.scope == "stack" else ("full", "attention", "pilot")
            candidate = next((suites[k] for k in options if k in suites
                              and (suites[k][0]/(case.name+".proof.json")).is_file()),None)
            if candidate is None:
                break
            folder, spec = candidate
            current = (spec["machine"],spec["seed"],spec["fixtures"],spec["warmups"],spec["trials"])
            if settings is not None and current != settings:
                raise ValueError(f"Incompatible suites within session {name}")
            settings = current
            per_case[case.name] = read_case(folder, case, spec)
        if len(per_case) == len(cases):
            if cohort is not None and cohort != settings:
                raise ValueError("Selected sessions differ in machine, code, seed, or protocol; select a matching cohort with --sessions")
            cohort = settings
            verified[name] = per_case
    if requested - set(verified):
        raise ValueError("Selected sessions are incomplete or unavailable: "+", ".join(sorted(requested-set(verified))))
    if len(verified) < 2:
        raise ValueError("Publication needs two independent, complete GPU sessions for the requested scope")
    return verified, cohort


def intervals(arrays, rng, replicates=2000):
    """Paired bootstrap inside each session × fixture; report ratio of medians.

    Preserve measured method pairings and equal fixture weights when sampling.
    These intervals quantify trial variation conditional on the observed sessions.
    """
    combined = np.concatenate(arrays)
    med = np.median(combined,axis=0)
    boot = np.empty((replicates, combined.shape[1]))
    strata = [a[fixture::FIXTURES] for a in arrays for fixture in range(FIXTURES)]
    for i in range(replicates):
        paired = [s[rng.integers(0,len(s),size=len(s))] for s in strata]
        boot[i] = np.median(np.concatenate(paired),axis=0)
    speed = med[0]/med
    speeds = boot[:,0:1]/boot
    return med, np.quantile(boot,[.025,.975],axis=0), speed, np.quantile(speeds,[.025,.975],axis=0)


def metrics(sessions, cases):
    names=sorted(sessions)
    rng=np.random.default_rng(71023)
    rows=[]
    for case in cases:
        arrays=[sessions[n][case.name] for n in names]
        med,ci,speed,ratio_ci=intervals(arrays,rng)
        winners=[case.methods[int(np.argmin(np.median(a,axis=0)[:len(SCHEDULES)]))] for a in arrays]
        for i,m in enumerate(case.methods):
            rows.append({"case":case.name,"workload":case.scope,"batch":case.batch,"visible_length":case.length,
                         "method":m,"wall_us":med[i],"lo_us":ci[0,i],"hi_us":ci[1,i],"speedup_vs_mlx":speed[i],
                         "lo_speedup":ratio_ci[0,i],"hi_speedup":ratio_ci[1,i],"session_count":len(names),
                         "observations":sum(a.shape[0] for a in arrays),
                         "session_medians_us":json.dumps({n:float(np.median(a[:,i])) for n,a in zip(names,arrays)},sort_keys=True),
                         "session_winners":json.dumps(dict(zip(names,winners)),sort_keys=True),
                         "same_fastest_main_schedule":len(set(winners))==1})
    return rows


def attach_trace_counts(rows, index_file):
    """Associate human-inspected Instruments captures, separately from encoded counts."""
    index = {(row["case"], row["method"]):row for row in rows}
    for row in rows:
        row.update(observed_dispatches=None,trace_path=None)
    if index_file is None:
        return
    with index_file.open(newline="") as file:
        for evidence in csv.DictReader(file):
            key=(evidence["case"],evidence["method"])
            if key not in index:
                raise ValueError(f"Trace case/method is outside the published scope: {key}")
            if index[key]["trace_path"] is not None:
                raise ValueError(f"Duplicate trace evidence for {key}")
            count=int(evidence["observed_dispatches"])
            path=Path(evidence["trace_path"])
            if (count<1 or path.is_absolute() or ".." in path.parts or path.suffix!=".trace"
                    or not (index_file.parent/path).exists()):
                raise ValueError(f"Invalid trace count or missing local Instruments capture for {key}")
            index[key].update(observed_dispatches=count, trace_path=path.as_posix())


def write_report(folder, rows, cohort, sessions, scope):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (folder/"measurements.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index={(r["case"],r["method"]):r for r in rows}
    cases = [c for c in TABLE if scope=="full" or c.scope=="attention"]
    title="Decode latency on "+str(cohort[0]["device"])
    lines=["# "+title,"","Median synchronized wall time in µs. Brackets enclose 95% paired, stratified bootstrap intervals.", "",
           "| Workload | Batch | Length | "+" | ".join(LABELS[m] for m in SCHEDULES)+" |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        cells = []
        for m in SCHEDULES:
            r=index[(c.name,m)]
            cells.append(f"{r['wall_us']:.2f} [{r['lo_us']:.2f}, {r['hi_us']:.2f}]")
        lines.append(f"| {c.scope} | {c.batch} | {c.length} | "+" | ".join(cells)+" |")
    lines += ["","measurements.csv includes speedups, confidence intervals and each session’s median and winner. "
              "Raw timing CSVs distinguish native command-buffer GPU intervals from total wall time. "
              "Encoded native dispatch counts describe the submitted schedule. The observed_dispatches and trace_path columns carry inspected Instruments evidence when supplied.",
              "", "Sessions: "+", ".join(sorted(sessions)), "","Source SHA-256: "+cohort[0]["sources_sha256"]]
    (folder/"performance.md").write_text("\n".join(lines)+"\n")
    fig, ax = plt.subplots(figsize=(7.4,4.45),layout="constrained")
    styles = [("#1a202c","o","-"),("#256099","s","--"),("#a44619","^","-."),("#387355","D",":")]
    for m,(color,mark,ls) in zip(SCHEDULES,styles):
        data=[index[(f"attention-b1-l{n}",m)] for n in LENGTHS]
        y=np.array([r["wall_us"] for r in data])
        lo=np.maximum(0,y-np.array([r["lo_us"] for r in data]))
        hi=np.maximum(0,np.array([r["hi_us"] for r in data])-y)
        ax.errorbar(LENGTHS,y,yerr=[lo,hi],label=LABELS[m],color=color,marker=mark,linestyle=ls,capsize=3)
    ax.set_xscale("log",base=2)
    ax.set_xticks(LENGTHS,[f"{n:,}" for n in LENGTHS])
    ax.set_ylabel("Synchronized wall time (µs)")
    ax.set_xlabel("Visible context, including the current token")
    ax.set_title("One token, four schedules",loc="left",weight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",alpha=.18)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,ncols=2,fontsize=9)
    fig.savefig(folder/"latency.svg",metadata={"Date":None})
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("results"))
    p.add_argument("--out",type=Path,default=Path("results/report"))
    p.add_argument("--sessions",nargs="+",help="Select matching independent collection sessions")
    p.add_argument("--scope",choices=("attention","full"),default="full")
    p.add_argument("--trace-index",type=Path,help="Optional CSV of inspected Instruments captures")
    args=p.parse_args()
    cases=ATTENTION if args.scope=="attention" else ATTENTION+STACK
    try:
        sessions, cohort=load_sessions(args.input,args.sessions,cases)
        rows=metrics(sessions,cases)
        attach_trace_counts(rows,args.trace_index)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
            source=Path(tmp)
            write_report(source,rows,cohort,sessions,args.scope)
            args.out.mkdir(exist_ok=True)
            for file in source.iterdir():
                file.replace(args.out/file.name)
        print("Wrote",args.out)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        p.exit(2,f"Report did not render: {exc}\n")


if __name__=="__main__":
    main()
