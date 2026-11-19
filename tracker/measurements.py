"""Read only validated, paired measurements for the standalone progress page."""
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

METHODS = ("compiled", "staged", "hybrid", "single")
TABLE = (("attention", 1, 64), ("attention", 1, 4096), ("attention", 8, 4096),
         ("stack", 1, 64), ("stack", 1, 4096))


def read_cases(results):
    cells, sessions = {}, []
    for file in sorted(results.glob("*/*/manifest.json"), key=lambda p: p.stat().st_mtime):
        try:
            manifest = json.loads(file.read_text())
            spec = manifest["spec"]
            if not manifest["environment"]["gpu_probe"]["passed"] or spec["schema"] != 1:
                continue
            accepted = []
            for proof_path in sorted(file.parent.glob("*.proof.json")):
                proof = json.loads(proof_path.read_text())
                name = proof["case"]
                if proof_path.name != name+".proof.json":
                    continue
                csv_path = file.parent / (name+".csv")
                if not proof.get("rechecked_after_timing") or proof.get("sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
                    continue
                by_method = {m: {} for m in METHODS}
                with csv_path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != proof["row_count"]:
                    continue
                for row in rows:
                    if row["session"] != spec["session"] or row["case"] != name:
                        raise ValueError("mislabeled observation")
                    method = row["method"]
                    if method in by_method:
                        b = int(row["block"])
                        if b in by_method[method]:
                            raise ValueError("duplicate observation")
                        t = float(row["wall_us"])
                        if not math.isfinite(t) or t <= 0:
                            raise ValueError("invalid wall time")
                        by_method[method][b] = t
                if spec["trials"] < 100 or any(set(x) != set(range(spec["trials"])) for x in by_method.values()):
                    continue
                if any(not proof["oracle_max_abs"][str(f)].get(m) for f in range(spec["fixtures"]) for m in METHODS):
                    continue
                for m in METHODS:
                    cells[name, m] = statistics.median(by_method[m].values())
                accepted.append(name)
            sessions.append((spec["session"], spec["suite"], manifest["status"], accepted))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return cells, sessions


def report_figure(results, sessions):
    figure = results/"report/latency.svg"
    table = results/"report/measurements.csv"
    provenance = results/"report/performance.md"
    complete = {s[0] for s in sessions if s[2] == "complete" and len(s[3]) >= 7}
    if len(complete)<2 or not all(p.is_file() for p in (figure,table,provenance)):
        return "", {}
    # Accept only the report for its listed local sessions and matching source revision.
    text = provenance.read_text()
    session_line = next((line for line in text.splitlines() if line.startswith("Sessions: ")), "")
    members = set(session_line[len("Sessions: "):].split(", "))
    if len(members)<2 or not members <= complete:
        return "", {}
    combined = {}
    with table.open(newline="") as f:
        for row in csv.DictReader(f):
            value = float(row["wall_us"])
            if row["method"] in METHODS and math.isfinite(value) and value>0:
                combined[row["case"],row["method"]] = value
    encoded=base64.b64encode(figure.read_bytes()).decode()
    markup = ('<figure class="schematic"><img style="display:block;width:100%;max-width:780px;height:auto;margin:auto" '
              'alt="One-token attention latency against visible context for four GPU schedules" '
              f'src="data:image/svg+xml;base64,{encoded}"><figcaption>Verified GPU sessions: '
              +html.escape(", ".join(sorted(members)))+'. Intervals quantify trial variation within sessions.</figcaption></figure>')
    return markup, combined
