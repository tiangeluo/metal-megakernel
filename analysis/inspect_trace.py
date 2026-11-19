"""Export Instruments tables and inspect the target process inside each decode signpost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

METHODS = ("single", "split", "compiled", "hybrid")
SCHEMAS = ("os-signpost", "metal-application-command-buffer-submissions",
           "metal-application-encoders-list", "metal-gpu-intervals", "metal-shader-profiler-intervals")
CASE = "attention-b1-l4096"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    root = ET.parse(path)
    names = [col.findtext("mnemonic") for col in root.findall(".//schema/col")]
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    for row in root.iter("row"):
        if len(row) != len(names):
            raise ValueError("Unexpected Instruments schema")
        yield {name: ids[e.get("ref")] if e.get("ref") else e for name, e in zip(names, row)}


def window(row):
    start = int(row["start"].text)
    return [start, start + int(row["duration"].text)]


def get_pid(row):
    # Instruments references may resolve to a prior process element, whose PID is in its label.
    return int(row["process"].get("fmt").rsplit("(", 1)[1].rstrip(")"))


def inspect(tables, pid, method):
    marks = [r for r in tables["os-signpost"] if r["name"].text == "Decode step"
             and r["subsystem"].text == "org.megakernel.benchmark" and get_pid(r) == pid]
    if ([r["event-type"].text for r in marks] != ["Begin", "End"]
            or marks[0]["message"].get("fmt") != f"{CASE}/{method}"
            or marks[0]["identifier"].text != marks[1]["identifier"].text):
        raise ValueError("Missing unique paired benchmark signposts")
    start, end = [int(r["time"].text) for r in marks]

    def target(schema):
        return [r for r in tables[schema] if get_pid(r) == pid]

    # MLX may create a command buffer before the marker and commit it during the marker.
    submitted = [r for r in target(SCHEMAS[1]) if start <= window(r)[1] <= end]
    buffer_ids = {int(r["cmdbuffer-id"].text) for r in submitted}
    encoders = [r for r in target(SCHEMAS[2]) if int(r["cmdbuffer-id"].text) in buffer_ids]
    encoder_ids = {int(r["encoder-id"].text) for r in encoders}
    if not buffer_ids or sum(int(r["num-encoders"].text) for r in submitted) != len(encoder_ids):
        raise ValueError("Submission and encoder tables disagree")
    gpu = [r for r in target(SCHEMAS[3]) if window(r)[0] < end and window(r)[1] > start]
    if not gpu or any(r["channel-name"].text != "Compute" or r["state"].text != "Active"
                      or int(r["cmdbuffer-id"].text) not in buffer_ids
                      or int(r["encoder-id"].text) not in encoder_ids
                      or not start <= window(r)[0] <= window(r)[1] <= end for r in gpu):
        raise ValueError("Incomplete target compute attribution")
    if {int(r["encoder-id"].text) for r in gpu} != encoder_ids:
        raise ValueError("An encoder has no matching active GPU interval")
    shader_rows = [r for r in target(SCHEMAS[4]) if window(r)[0] < end and window(r)[1] > start]
    if shader_rows:
        raise ValueError("Shader intervals are present and require individual inspection")
    # Durations are only additive if the intervals are disjoint.
    spans = sorted(window(r) for r in gpu)
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("Overlapping GPU intervals")
    return dict(pid=pid, marker_ns=[start, end], command_buffers=len(buffer_ids),
                compute_encoders=len(encoder_ids), gpu_compute_intervals=len(gpu),
                active_compute_ns=sum(b-a for a, b in spans), observed_dispatches=None,
                shader_interval_rows=len(shader_rows), buffer_ids=sorted(buffer_ids),
                encoder_ids=sorted(encoder_ids), gpu_spans_ns=spans)


def generate(folder, exported):
    manifest = folder / "capture.json"
    study = json.loads(manifest.read_text())
    environment = folder.parent / "environment.json"
    if study["case"] != CASE or study["environment_sha256"] != digest(environment):
        raise ValueError("Capture environment mismatch")
    result = dict(case=CASE, session=folder.name, capture_manifest_sha256=digest(manifest),
                  environment_sha256=digest(environment), methods={})
    for method in METHODS:
        record = study["captures"][method]
        if record["xctrace_exit"] != 0 or record["target_parity_passed"] is not True:
            raise ValueError("Benchmark trace did not complete successfully")
        toc = folder / record["toc"]
        target = ET.parse(toc).find("./run/info/target/process")
        pid = int(target.get("pid"))
        if target.get("return-exit-status") != "0" or f"--method {method} " not in target.get("arguments") + " ":
            raise ValueError("Incorrect trace target")
        files = {s: exported / f"{method}-{s}.xml" for s in SCHEMAS}
        row = inspect({s: list(table(f)) for s,f in files.items()}, pid, method)
        row.update(toc_sha256=digest(toc), export_sha256={s:digest(f) for s,f in files.items()})
        result["methods"][method] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Directory containing the completed capture.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exported", type=Path, help="Use these previously exported XML tables")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        exported = args.exported or Path(temp)
        if args.exported is None:
            cache = Path(temp) / "instruments-home"
            (cache / "Library/Caches").mkdir(parents=True)
            for method in METHODS:
                for schema in SCHEMAS:
                    query = f"/trace-toc/run[@number='1']/data/table[@schema='{schema}']"
                    subprocess.run(["xcrun", "xctrace", "export", "--input", str(args.folder / f"{method}.trace"),
                                    "--xpath", query, "--output", str(exported / f"{method}-{schema}.xml")],
                                   env={**os.environ, "CFFIXED_USER_HOME": str(cache)}, check=True)
        result = generate(args.folder, exported)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
