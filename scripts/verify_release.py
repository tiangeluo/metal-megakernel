"""Audit the published M4 Pro evidence, tables and figure provenance, without Metal."""

import csv
import hashlib
import json
import math
from itertools import combinations, product
from xml.etree import ElementTree

import numpy as np

from analysis.crossover_report import METHODS, ORIGINAL
from analysis.decoder_performance import measured_panels
from analysis.preliminary import load
from analysis.trace_replay import verify_traces
from bench.protocol import ATTENTION, PILOT, REPO, SCHEDULES, TABLE, source_hash
from bench.report import attach_trace_counts
from diagnostics.precision_study import control_hash
from diagnostics.stack_crossover import compare


def verify_inputs():
    checksums = REPO / "analysis/release-inputs.sha256"
    records = checksums.read_text().splitlines()
    for record in records:
        digest, path = record.split("  ", 1)
        if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived evidence changed in {path}")
    return len(records)


def verify_attention_certificates(folder, cases):
    for case in cases:
        path = folder / (case.name + ".proof.json")
        errors = json.loads(path.read_text())["oracle_max_abs"]
        for methods in errors.values():
            for observed in methods.values():
                if (set(observed) != {"output", "cache_k", "cache_v", "hist_k", "hist_v"}
                        or not all(math.isfinite(v) and v >= 0 for v in observed.values())
                        or observed["hist_k"] != 0 or observed["hist_v"] != 0):
                    raise ValueError(f"Invalid attention oracle observations in {path}")


def verify_table(path, expected):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(expected[0]):
            raise ValueError(f"Columns differ from recomputed table in {path}")
        actual = list(reader)
    if len(actual) != len(expected):
        raise ValueError(f"Row count differs from recomputed table in {path}")
    for row, measured in zip(actual, expected):
        for field, value in measured.items():
            wanted = "" if value is None else str(value)
            if row[field] != wanted:
                raise ValueError(f"{path.name} differs at {measured['case']}/{measured['method']} "
                                 f"in {field}: stored {row[field]!r}, recomputed {wanted!r}")


def verify_article(path, rows):
    medians = {(row["case"], row["method"]): row["wall_us"] for row in rows}
    lines = path.read_text().splitlines()
    labels = ["Compiled MLX", "Staged Metal", "Hybrid", "Single dispatch"]
    for scope, title in (("attention", "Isolated global attention, median wall time"),
                         ("stack", "One decode step, batch one")):
        header = [title, *labels]
        if scope == "attention":
            header.append("Split context")
        start = lines.index("| " + " | ".join(header) + " |") + 2
        expected = []
        for case in TABLE:
            if case.scope != scope:
                continue
            if scope == "attention":
                cells = [f"Batch {case.batch}, {case.length:,} tokens"]
                for method in (*SCHEDULES, "split"):
                    value = medians.get((case.name, method))
                    cells.append(f"{value:,.1f} µs" if value is not None else "—")
            else:
                cells = [f"Two layers, {case.length:,} visible tokens"]
                cells += [f"{medians[case.name, m] / 1000:.3f} ms" for m in SCHEDULES]
            expected.append("| " + " | ".join(cells) + " |")
        for index, row in enumerate(expected, start):
            if lines[index:index + 1] != [row]:
                raise ValueError(f"Article table differs at {path.name}:{index + 1}; expected {row}")
        end = start + len(expected)
        if end < len(lines) and lines[end].startswith("|"):
            raise ValueError(f"Unexpected extra row in {scope} article table")


def verify_captures(source):
    """Reapply the original mixed-precision gate directly to archived layer-11 GPU outputs."""
    count = 0
    for length, seed in product((64, 4096), (20261118, 20261119)):
        path = ORIGINAL / f"crossover-stack-b1-l{length}-seed{seed}.npz"
        report = json.loads(path.with_suffix(".json").read_text())
        if (report["capture"] != path.name or report["sources_sha256"] != source
                or report["capture_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError(f"Capture provenance differs in {path.name}")
        with np.load(path, allow_pickle=False) as captured:
            meta = json.loads(captured["metadata"].item())
            if (meta["seed"], meta["length"], meta["methods"]) != (seed, length, list(METHODS)):
                raise ValueError(f"Fixture provenance differs in {path.name}")
            for left, right in combinations(METHODS, 2):
                evidence = compare(captured[left + "__moe"], captured[right + "__moe"])
                if evidence != report["observed"][f"{left}_vs_{right}"]:
                    raise ValueError(f"Replayed {left} vs {right} gate differs in {path.name}")
        count += 1
    return count


def main():
    n_inputs = verify_inputs()
    source = source_hash()
    root = REPO / "results"
    attention, stack, provenance, _ = measured_panels(root, root / "precision-control-01")
    if provenance["source_sha256"] != source or provenance["control_sha256"] != control_hash():
        raise ValueError("Checked-out acquisition sources differ from the published cohort")
    figure = REPO / "analysis/decoder-performance.svg"
    description = ElementTree.parse(figure).find(".//{http://purl.org/dc/elements/1.1/}description")
    if description is None or json.loads(description.text) != provenance:
        raise ValueError(f"Embedded source or trial digests differ in {figure.name}")

    for name in provenance["sessions"]:
        verify_attention_certificates(root / name / "attention", ATTENTION)
    old_pilot, _, _ = load("m4pro-01-pilot", "pilot", PILOT, require_complete=True)
    old_full, _, _ = load("m4pro-01-a", "full", ATTENTION)
    new_pilot, _, _ = load("m4pro-02-pilot", "pilot", PILOT, require_complete=True)
    if (old_pilot["machine"]["sources_sha256"] != old_full["machine"]["sources_sha256"]
            or old_full["machine"]["sources_sha256"] == source
            or new_pilot["machine"]["sources_sha256"] != source):
        raise ValueError("Historical pilot or partial sweep belongs to the wrong source cohort")

    attach_trace_counts(attention, None)
    verify_table(root / "report-attention/measurements.csv", attention)
    for row in stack:
        row["control"] = provenance["control"]
    verify_table(root / "precision-control-01/measurements.csv", stack)
    verify_article(REPO / "docs/article.md", attention + stack)
    captures = verify_captures(source)
    traces = verify_traces()
    observations = sum(row["observations"] for row in attention + stack)
    print(f"Verified {n_inputs} archived input files and both acquisition source hashes.")
    print(f"Reproduced all {len(attention) + len(stack)} published measurement rows "
          f"from {observations:,} timing observations, including the bootstrap intervals.")
    print(f"Matched all {len(TABLE)} article table rows, the figure's trial provenance, and the original "
          f"layer-11 comparisons from {captures} saved array captures.")
    print(f"Recomputed Instruments trace observations for all {traces} profiled schedules.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Release verification failed: {error}")
