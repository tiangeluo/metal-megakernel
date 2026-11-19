"""Validate measured cases and keep each preliminary table within one run."""
import base64
import csv
from dataclasses import dataclass
import hashlib
from html import escape
import json
import math
import xml.etree.ElementTree as ET


@dataclass
class Run:
    spec: dict
    started: str
    status: str
    error: str
    cases: dict
    digests: dict

    @property
    def name(self):
        return self.spec["session"]

    @property
    def suite(self):
        return self.spec["suite"]

    def compatible(self, other):
        return all(self.spec[k] == other.spec[k] for k in ("machine", "seed", "fixtures", "trials", "warmups"))


def read_runs(results):
    files = list(results.glob("*/*/manifest.json"))
    if not files:
        return []
    from bench.protocol import SCHEMA, SUITES
    from bench.report import read_case
    from statistics import median

    runs = []
    for file in files:
        try:
            manifest = json.loads(file.read_text())
            spec = manifest["spec"]
            if ((spec["session"], spec["suite"]) != (file.parent.parent.name, file.parent.name)
                    or spec["schema"] != SCHEMA or manifest["environment"]["gpu_probe"]["passed"] is not True
                    or "source changed while timing" in manifest.get("error", "").lower()):
                continue
            run = Run(spec, manifest["started"], manifest["status"], manifest.get("error", ""), {}, {})
            for case in SUITES[run.suite]:
                proof_file = file.parent / (case.name + ".proof.json")
                if not proof_file.is_file():
                    continue
                try:
                    samples = read_case(file.parent, case, spec)
                    run.cases[case.name] = {m: median(samples[:, i]) for i, m in enumerate(case.methods)}
                    run.digests[case.name] = json.loads(proof_file.read_text())["sha256"]
                except (KeyError, OSError, TypeError, ValueError):
                    continue
            runs.append(run)
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return sorted(runs, key=lambda run: run.started)


def figure_markup(path, alt, caption):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return ('<figure class="schematic"><img style="display:block;width:100%;height:auto;margin:auto" '
            f'alt="{escape(alt)}" src="data:image/svg+xml;base64,{encoded}">'
            f'<figcaption>{escape(caption)}</figcaption></figure>')


def preliminary_figure(path, selected, runs):
    if not selected or not path.is_file():
        return ""
    try:
        tree = ET.parse(path)
        desc = tree.find(".//{http://purl.org/dc/elements/1.1/}description")
        data = json.loads(desc.text) if desc is not None else {}
        if data["sweep_session"] != selected.name or data["source_sha256"] != selected.spec["machine"]["sources_sha256"]:
            return ""
        pilot = next((r for r in runs if r.name == data["pilot_session"] and r.suite == "pilot"), None)
        if not pilot or set(data["case_sha256"]) != {selected.name, pilot.name}:
            return ""
        for run, expected_cases in ((selected, 7), (pilot, 2)):
            digests = data["case_sha256"][run.name]
            if len(digests) != expected_cases or any(run.digests.get(k) != v for k, v in digests.items()):
                return ""
        return figure_markup(
            path, "Latency by visible context, with paired single-versus-split comparisons at 64 and 4,096 tokens.",
            f"The context sweep comes from {selected.name}. The second panel compares its endpoints with "
            f"{pilot.name}. Intervals show variation across paired timing trials within each run.")
    except (KeyError, OSError, TypeError, ValueError, ET.ParseError):
        return ""


def repeated_figure(path, runs, names):
    """Attach the two-panel figure only when all source CSVs match the verified runs."""
    try:
        description = ET.parse(path).find(".//{http://purl.org/dc/elements/1.1/}description")
        provenance = json.loads(description.text) if description is not None else {}
        if sorted(provenance["sessions"]) != sorted(names) or set(provenance["case_sha256"]) != set(names):
            return ""
        from bench.protocol import ATTENTION
        cases = {case.name for case in ATTENTION}
        for name in names:
            digests = provenance["case_sha256"][name]
            if set(digests) != cases or not any(
                    run.name == name and run.status == "complete"
                    and run.spec["machine"]["sources_sha256"] == provenance["source_sha256"]
                    and all(digest == run.digests.get(case) for case, digest in digests.items())
                    for run in runs):
                return ""
        return figure_markup(path, "Attention latency across context lengths and the split-context control at the endpoints.",
                             f"Two independent M4 Pro sessions, {', '.join(names)}. Both panels use synchronized "
                             "wall time for a single decode step on the isolated global-attention path.")
    except (KeyError, OSError, TypeError, ValueError, ET.ParseError):
        return ""


def final_report(results, runs, scope="full"):
    """Accept a report only when all of its registered cases have matching provenance."""
    folder = results / ("report" if scope == "full" else "report-attention")
    try:
        lines = (folder / "performance.md").read_text().splitlines()
        names = next(s.removeprefix("Sessions: ").split(", ") for s in lines if s.startswith("Sessions: "))
        sha = next(s.removeprefix("Source SHA-256: ") for s in lines if s.startswith("Source SHA-256: "))
        from bench.protocol import SUITES
        expected_cases = {case.name for case in SUITES[scope]}
        required = {(case.name, m) for case in SUITES[scope] for m in case.methods}
        for name in names:
            complete = [r for r in runs if r.name == name and r.status == "complete"
                        and r.spec["machine"]["sources_sha256"] == sha]
            cases = {case for r in complete for case in r.cases}
            if not expected_cases <= cases:
                return None
        if len(set(names)) < 2 or len(set(names)) != len(names):
            return None
        cells, traces = {}, {}
        with (folder / "measurements.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                key = (row["case"], row["method"])
                value = float(row["wall_us"])
                if (key in cells or not math.isfinite(value) or value <= 0
                        or set(json.loads(row["session_medians_us"])) != set(names)):
                    return None
                cells[key] = value
                if row.get("observed_dispatches") and row.get("trace_path"):
                    traces[key] = int(row["observed_dispatches"])
        if cells.keys() != required:
            return None
        figure = figure_markup(folder / "latency.svg", "Global attention latency versus visible context for four GPU schedules.",
                               f"Combined medians from {', '.join(names)}. Intervals show trial variation within runs.")
        return cells, figure, names, traces
    except (KeyError, OSError, StopIteration, TypeError, ValueError):
        return None


def diagnostic_reports(results, latest):
    if latest is None:
        return []
    base, _, suffix = latest.name.rpartition("-")
    study = base if suffix in ("a", "b", "pilot") else latest.name
    folder = results / (study + "-diagnostics")
    records = []
    first_seed = latest.spec["seed"]
    for path in sorted(folder.glob("stack-b1-l*-seed*.json")):
        try:
            data = json.loads(path.read_text())
            if (data.get("status") != "complete" or data["machine"] != latest.spec["machine"]
                    or data["case"] not in ("stack-b1-l64", "stack-b1-l4096")
                    or data["seed"] not in range(first_seed, first_seed + latest.spec["fixtures"])
                    or path.name != f'{data["case"]}-seed{data["seed"]}.json'):
                continue
            statuses = {}
            for method in ("compiled", "staged", "hybrid", "single"):
                if method in data["method_errors"]:
                    statuses[method] = "Execution error"
                    continue
                result = data["comparisons"][method]["vs_cpu"]
                if not result["passed"]:
                    fields = result["failed_observables"] + result.get("extra_observables", [])
                    if fields == ["layer11_post_moe"]:
                        region = result["observables"]["layer11_post_moe"]["regions"]["all"]
                        statuses[method] = f"{region['violations']:,} / {region['size']:,}"
                    else:
                        statuses[method] = ", ".join(fields) or "Failed"
            if data["passed"] == bool(statuses):
                continue
            length = int(data["case"].split("-l")[-1])
            records.append((length, data["seed"], statuses, data["comparisons"], data.get("stage_isolation", {})))
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return sorted(records)


def followup_status(results, parent):
    """Count checksum-verified captures and candidate trace bundles for the latest follow-up."""
    import numpy as np

    for path in sorted(results.glob("m4pro-02-followup-*/manifest.json"), reverse=True):
        try:
            state = json.loads(path.read_text())
            if (state["parent"] != parent.name or state["parent_machine"] != parent.spec["machine"]
                    or state["machine"] != state["parent_machine"]):
                continue
            outcomes = []
            for length in (64, 4096):
                for seed in range(parent.spec["seed"], parent.spec["seed"] + parent.spec["fixtures"]):
                    tag = f"crossover-stack-b1-l{length}-seed{seed}"
                    if state["crossovers"].get(tag) != "complete":
                        continue
                    archive = path.parent / (tag + ".npz")
                    try:
                        record = json.loads((path.parent / (tag + ".json")).read_text())
                        if (record["length"] == length and record["seed"] == seed and record["batch"] == 1
                                and record["methods"] == ["cpu", "compiled", "staged"] and record["wrapper_replay_exact"]
                                and record["device"] == state["machine"]["device"]
                                and record["sources_sha256"] == state["machine"]["sources_sha256"]
                                and record["capture"] == archive.name and len(record["crossed_cpu_finish"]) == 9
                                and hashlib.sha256(archive.read_bytes()).hexdigest() == record["capture_sha256"]):
                            with np.load(archive, allow_pickle=False) as data:
                                metadata = json.loads(data["metadata"].item())
                                if any(record.get(k) != v for k, v in metadata.items()):
                                    continue
                                ref = data["cpu__moe"].astype(np.float32)
                                counts = []
                                for residual, heads in (("compiled", "cpu"), ("staged", "cpu"),
                                                        ("cpu", "compiled"), ("cpu", "staged")):
                                    key = f"residual_{residual}__heads_{heads}"
                                    got = data[key].astype(np.float32)
                                    if got.shape != ref.shape or not np.isfinite(got).all() or not np.isfinite(ref).all():
                                        raise ValueError("Invalid expert output")
                                    n = int(np.count_nonzero(np.abs(got - ref) > 0.002 + 0.01 * np.abs(ref)))
                                    if n != record["crossed_cpu_finish"][key]["vs_cpu"]["violations"]:
                                        raise ValueError("Crossover array differs from recorded count")
                                    counts.append(n)
                                outcomes.append((seed, length, counts))
                    except (KeyError, OSError, TypeError, ValueError):
                        continue
            traces = set()
            missing_mlx = set()
            for index in (path.parent / "traces").glob("*/capture.json"):
                trace = json.loads(index.read_text())
                if trace.get("case") != "attention-b1-l4096":
                    continue
                attestation = results.parent / "analysis/trace-preflight.json"
                verified_errors = {}
                if attestation.is_file():
                    proof = json.loads(attestation.read_text())
                    if (proof.get("followup") == path.parent.name and proof.get("case") == trace["case"]
                            and proof.get("source_sha256") == state["machine"]["sources_sha256"]
                            and proof.get("capture_sha256") == hashlib.sha256(index.read_bytes()).hexdigest()):
                        verified_errors = proof["methods"]
                for method in ("compiled", "hybrid", "single", "split"):
                    row = trace.get("captures", {}).get(method, {})
                    observed = verified_errors.get(method, {})
                    if (row.get("status") == "incomplete" and row.get("target_parity_passed") is False
                            and observed.get("failure") == "ModuleNotFoundError: No module named 'mlx'"
                            and observed.get("phase") == "preflight" and observed.get("xctrace_exit") == row.get("xctrace_exit")
                            and len(observed.get("target_log_sha256", "")) == 64):
                        missing_mlx.add(method)
                    if (row.get("status") == "awaiting_inspection" and row.get("target_parity_passed")
                            and row.get("trace") == f"{method}.trace" and (index.parent / row["trace"]).is_dir()):
                        traces.add(method)
            details = f"{len(outcomes)} of 4 GPU captures have matching JSON and array-file checksums."
            if traces:
                details += f" {len(traces)} of 4 candidate Instruments traces await inspection."
            elif len(missing_mlx) == 4:
                details += " All four Instruments targets stopped in preflight because Python could not import MLX. No dispatch counts were observed."
            elif state["trace_status"] == "unavailable or incomplete":
                details += " Instruments capture was unavailable or incomplete."
            else:
                details += " Instruments capture is pending."
            return details, sorted(outcomes)
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return "The four GPU captures and Instruments traces are awaiting collection.", []
