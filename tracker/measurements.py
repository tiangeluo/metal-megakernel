"""Validate measured cases and keep each preliminary table within one run."""
import base64
import csv
from dataclasses import dataclass
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
            statuses = []
            for method in ("compiled", "staged", "hybrid", "single"):
                if method in data["method_errors"]:
                    statuses.append(f"{method} execution error")
                    continue
                result = data["comparisons"][method]["vs_cpu"]
                if not result["passed"]:
                    fields = result["failed_observables"] + result.get("extra_observables", [])
                    statuses.append(f"{method} failed on {', '.join(fields)}")
            if data["passed"] == bool(statuses):
                continue
            length = int(data["case"].split("-l")[-1])
            records.append((length, data["seed"], statuses))
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return sorted(records)
