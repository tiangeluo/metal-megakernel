"""Replay Instruments signpost attribution from just the benchmark process's rows."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from analysis.inspect_trace import CASE, METHODS, SCHEMAS, digest, get_pid, inspect, table

HERE = Path(__file__).resolve().parent
OBSERVATIONS = HERE / "trace-observations.json"
TABLES = HERE / "trace-tables.xml"
FIELDS = {
    "os-signpost": ("time", "event-type", "identifier", "name", "subsystem", "message"),
    "metal-application-command-buffer-submissions": ("start", "duration", "cmdbuffer-id", "num-encoders"),
    "metal-application-encoders-list": ("start", "duration", "cmdbuffer-id", "encoder-id"),
    "metal-gpu-intervals": ("start", "duration", "cmdbuffer-id", "encoder-id", "channel-name", "state"),
    "metal-shader-profiler-intervals": ("start", "duration"),
}


def project(exports):
    """Keep all target PID submissions, encoders and intervals, including warm-ups."""
    observed = json.loads(OBSERVATIONS.read_text())
    doc = ET.Element("instruments-projection", case=CASE, session=observed["session"])
    for method in METHODS:
        reference = observed["methods"][method]
        pid = reference["pid"]
        capture = ET.SubElement(doc, "capture", method=method, pid=str(pid))
        for schema in SCHEMAS:
            path = exports / f"{method}-{schema}.xml"
            sha = digest(path)
            if sha != reference["export_sha256"][schema]:
                raise ValueError(f"{path.name} differs from its original export")
            dest = ET.SubElement(capture, "table", schema=schema, export_sha256=sha)
            for row in table(path):
                if get_pid(row) != pid:
                    continue
                if schema == "os-signpost" and (row["name"].text != "Decode step"
                        or row["subsystem"].text != "org.megakernel.benchmark"):
                    continue
                values = {}
                for field in FIELDS[schema]:
                    value = row[field].get("fmt") if field == "message" else row[field].text
                    if value is not None:
                        values[field] = value
                ET.SubElement(dest, "row", values)
    ET.indent(doc)
    return doc


def verify_traces(path=TABLES):
    """Reconstruct the original inspection rows and recompute its attribution."""
    observed = json.loads(OBSERVATIONS.read_text())
    doc = ET.parse(path).getroot()
    if doc.get("case") != observed["case"] or doc.get("session") != observed["session"]:
        raise ValueError("Trace projection belongs to another capture")
    if [c.get("method") for c in doc] != list(METHODS):
        raise ValueError("Missing or repeated trace capture")
    for capture in doc:
        method, pid = capture.get("method"), int(capture.get("pid"))
        reference = observed["methods"][method]
        if [t.get("schema") for t in capture] != list(SCHEMAS):
            raise ValueError(f"Missing or repeated table for {method}")
        tables = {}
        for source in capture:
            schema = source.get("schema")
            if source.get("export_sha256") != reference["export_sha256"][schema]:
                raise ValueError(f"Trace export provenance changed for {method}/{schema}")
            rows = []
            for projected in source:
                row = {key: ET.Element(key) for key in FIELDS[schema]}
                for field, value in projected.attrib.items():
                    if field == "message":
                        row[field].set("fmt", value)
                    else:
                        row[field].text = value
                row["process"] = ET.Element("process", fmt=f"target ({pid})")
                rows.append(row)
            tables[schema] = rows
        replayed = inspect(tables, pid, method)
        expected = {k: v for k, v in reference.items() if k not in ("toc_sha256", "export_sha256")}
        if replayed != expected:
            raise ValueError(f"Replayed signpost attribution differs for {method}")
    return len(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, help="Recreate the projection using the original five XML exports per method")
    parser.add_argument("--tables", type=Path, default=TABLES, help="Projected tables to verify or recreate")
    args = parser.parse_args()
    if args.exports:
        doc = project(args.exports)
        args.tables.write_text(ET.tostring(doc, encoding="unicode") + "\n", encoding="utf-8")
    print(f"Replayed all {verify_traces(args.tables)} trace captures.")
