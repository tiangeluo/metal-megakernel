"""Verify the four M4 Pro captures and rerun the layer and expert interventions on CPU."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from decoder.benchmark import model_weights
from decoder.model import Config, moe
from diagnostics.stack_crossover import analyze


METHODS = ("cpu", "compiled", "staged")
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/m4pro-02-followup-20261119T020918Z-tf3imdmi"


def outside(ref, actual):
    ref, actual = ref.astype(np.float32), actual.astype(np.float32)
    if not np.isfinite(ref).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite expert output")
    return np.flatnonzero(np.abs(ref - actual) > 0.002 + 0.01 * np.abs(ref)).tolist()


def mix(weights, vectors):
    out = np.zeros((vectors.shape[0], vectors.shape[2]), dtype=np.float32)
    for rank in range(weights.shape[1]):
        out += weights[:, rank, None] * vectors[:, rank, :].astype(np.float32)
    return out.astype(np.float16)


def verify_case(path, machine):
    d = json.loads(path.read_text())
    archive = path.with_suffix(".npz")
    if not (d["wrapper_replay_exact"] and d["capture"] == archive.name
            and hashlib.sha256(archive.read_bytes()).hexdigest() == d["capture_sha256"]
            and d["sources_sha256"] == machine["sources_sha256"] and d["device"] == machine["device"]
            and d["methods"] == list(METHODS) and d["batch"] == 1 and d["config"] == vars(Config())):
        raise ValueError(f"Capture provenance or geometry mismatch in {path}")
    with np.load(archive, allow_pickle=False) as arrays:
        metadata = json.loads(arrays["metadata"].item())
        if any(d[key] != value for key, value in metadata.items()):
            raise ValueError(f"Array metadata differs from JSON in {path}")
        common = {key: arrays[key] for key in ("completed", "hist_a", "hist_e")}
        captures = {m: {k.split("__", 1)[1]: arrays[k] for k in arrays.files if k.startswith(m + "__")}
                    for m in METHODS}
        rebuilt, crossed = analyze(Config(), common, captures)
        if any(rebuilt[key] != d[key] for key in rebuilt):
            raise ValueError(f"Recomputed numeric checks differ from {path}")
        for key, values in crossed.items():
            np.testing.assert_array_equal(values, arrays[key])

    layer = model_weights(Config())[1]
    reference = captures["cpu"]
    summary = {"length": d["length"], "seed": d["seed"], "cpu_finish": {}, "expert_swaps": {}}
    for r, h in product(METHODS, repeat=2):
        out = crossed[f"residual_{r}__heads_{h}"]
        summary["cpu_finish"][f"{r}/{h}"] = outside(reference["moe"], out)
    for m, values in captures.items():
        np.testing.assert_array_equal(values["routes"], reference["routes"])
        np.testing.assert_array_equal(mix(values["route_weights"], values["down"]), values["moe"])
        source_finish, routes = moe(np, values["expert_source"], layer, Config().top_k)
        np.testing.assert_array_equal(routes, reference["routes"])
        original = outside(reference["moe"], values["moe"])
        from_source = outside(reference["moe"], source_finish)
        summary["expert_swaps"][m] = {
            "observed": original, "captured_source_cpu_experts": from_source,
            "matched_input_expert_finish": outside(source_finish, values["moe"]),
            "weights_only": outside(reference["moe"], mix(values["route_weights"], reference["down"])),
            "vectors_only": outside(reference["moe"], mix(reference["route_weights"], values["down"])),
        }
    return summary


def markdown(records, directory, machine):
    lines = [
        "# What carries the difference into layer 11?",
        "",
        "Each of the four M4 Pro captures reproduced its production call’s expert histories, attention-output "
        "histories, route IDs and final residual exactly. I took the incoming residual from one implementation "
        "and the gated attention heads from another, then ran the rest of layer 11 in NumPy. Each row below "
        "compares the resulting 1,152-channel expert output with the original CPU result. A failure is an "
        "element that exceeds the registered tolerance of 0.002 + 0.01 × abs(reference).",
        "",
        "| Tokens | Seed | GPU inputs from | GPU residual only | GPU heads only | Both GPU inputs | Observed on GPU |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {"compiled": "Compiled MLX", "staged": "Staged Metal"}
    for rec in records:
        for m in labels:
            rows = rec["cpu_finish"]
            values = [len(rows[key]) for key in (f"{m}/cpu", f"cpu/{m}", f"{m}/{m}")]
            values.append(len(rec["expert_swaps"][m]["observed"]))
            lines.append(f"| {rec['length']:,} | {rec['seed']} | {labels[m]} | " + " | ".join(map(str, values)) + " |")
    lines.extend([
        "",
        "On the failing seed, substituting just the GPU residual was sufficient to cross tolerance at both "
        "lengths. Substituting just the heads passed for both implementations and both seeds. The heads "
        "still influence the outcome when combined with a different residual. The passing seed illustrates "
        "that interaction: a mixed pair can fail even when the two inputs from its production run pass.",
        "",
        "The crossover locates a difference at the boundary between the layers. It cannot distinguish the "
        "individual operations in layer 10 that contributed to it. A CPU finish on a GPU’s *own* inputs "
        "passed against that GPU’s observed output in all eight cases. Individual channels can still land "
        "on opposite sides of the tolerance relative to the original CPU baseline.",
        "",
        "## The eight expert contributions",
        "",
        "Each implementation chose the identical ordered expert IDs. Summing its eight recorded output "
        "vectors with its own route weights reproduced the observed expert output bit for bit. Two offline "
        "swaps then varied the route weights and expert vectors independently.",
        "",
        "| Tokens | Seed | GPU inputs from | GPU weights only | GPU expert vectors only | CPU experts on GPU expert input |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for rec in records:
        for m in labels:
            data = rec["expert_swaps"][m]
            values = [len(data[key]) for key in ("weights_only", "vectors_only", "captured_source_cpu_experts")]
            lines.append(f"| {rec['length']:,} | {rec['seed']} | {labels[m]} | " + " | ".join(map(str, values)) + " |")
    lines.extend([
        "",
        "The recorded GPU route weights alone never pushed the CPU expert vectors across tolerance. The "
        "vectors alone did at both lengths for the failing seed. Recomputing the experts in NumPy on the "
        "recorded GPU input also retained failures. This input sensitivity helps explain how a cancellation "
        "in the expert sum can expose small differences already present before the sum.",
        "",
        f"The acquisition used {machine['device']}, MLX {machine['mlx']} and source "
        f"`{machine['sources_sha256']}`. Run `.venv/bin/python -m analysis.crossover_report` from the "
        "project root to verify the array digests and rerun all nine CPU finishes per fixture. The script "
        "also prints the individual violating channel IDs.",
        "",
        f"[Four captured fixtures](../results/{directory.name}/) · [Original stack parity](stack-parity.md)",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ORIGINAL)
    parser.add_argument("--markdown", type=Path, help="Write the measured tables for the original acquisition.")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["machine"] != manifest["parent_machine"]:
        raise ValueError("Incomplete capture or mismatching parent run")
    records = []
    for length, seed in product((64, 4096), (20261118, 20261119)):
        name = f"crossover-stack-b1-l{length}-seed{seed}"
        if manifest["crossovers"][name] != "complete":
            raise ValueError(f"Incomplete capture for {name}")
        row = verify_case(args.directory / (name + ".json"), manifest["machine"])
        if (row["length"], row["seed"]) != (length, seed):
            raise ValueError(f"Incorrect scenario in {name}")
        records.append(row)
    if args.markdown:
        if args.directory.resolve() != ORIGINAL:
            parser.error("The Markdown interpretation is for the original acquisition.")
        args.markdown.write_text(markdown(records, args.directory, manifest["machine"]))
    else:
        print(json.dumps(records, indent=2))
