"""Replay CPU arithmetic at worst coordinates reported by a GPU stack diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from decoder import model
from decoder.benchmark import prepare_case
from diagnostics.projection_sensitivity import einsum_fp64, matmul_fp32, partials_fp32


FIELD = "layer11_post_moe"


def read_witnesses(report):
    witnesses = []
    for method, comparisons in report["comparisons"].items():
        region = comparisons["vs_cpu"]["observables"][FIELD]["regions"]["all"]
        if region["violations"]:
            witnesses.append({"method": method, **region["worst_violation"]})
    return witnesses


def terms_at_second_layer(case):
    """Capture the expert input and reproduce all channels of its weighted sum."""
    inputs = []
    original_moe = model.moe

    def capture(xp, x, layer, top_k):
        inputs.append((x, layer, top_k))
        return original_moe(xp, x, layer, top_k)

    with patch.object(model, "moe", capture):
        case.run()
    (x, layer, top_k) = inputs[1]
    ids, weights = model.route(np, x, layer["router"], layer["bias"], top_k)
    assert x.shape[0] == 1
    terms = []
    for expert, weight in zip(ids[0], weights[0]):
        gate_up = model.project(np, x[0], layer["gate_up"][expert])
        mid = model.activation(np, gate_up)
        down = model.project(np, mid, layer["down"][expert])
        terms.append(weight * down.astype(np.float32))
    total = np.zeros_like(terms[0])
    for term in terms:
        total += term
    stored = case.output_numpy()[FIELD][0, 0]
    np.testing.assert_array_equal(total.astype(np.float16), stored)
    return np.array(terms, dtype=np.float64), stored.copy(), ids[0].tolist()


def replay(path):
    report = json.loads(path.read_text())
    witnesses = read_witnesses(report)
    if not witnesses:
        return None
    length = int(report["case"].split("-l")[-1])
    case = prepare_case("cpu", length, seed=report["seed"])
    try:
        terms, baseline, ids = terms_at_second_layer(case)
        predictions = {"original_fp32_einsum": baseline}
        for impl in (matmul_fp32, partials_fp32, einsum_fp64):
            case.reset()
            with patch.object(model, "project", impl):
                case.run()
            predictions[impl.__name__] = case.output_numpy()[FIELD][0, 0].copy()
    finally:
        case.close()

    mass = np.abs(terms).sum(axis=0)
    exact_sum = terms.sum(axis=0)
    condition = np.divide(
        mass, abs(exact_sum), out=np.full_like(mass, np.inf), where=exact_sum != 0
    )
    for w in witnesses:
        batch, row, channel = w["index"]
        if (batch, row) != (0, 0) or w["expected"] != baseline[channel]:
            raise ValueError("Stored diagnostic differs from the current CPU fixture")
        w.update(
            weighted_expert_terms=terms[:, channel].tolist(),
            sum_abs_terms=float(mass[channel]),
            signed_sum=float(exact_sum[channel]),
            cancellation_ratio=float(condition[channel]),
            percentile_across_channels=float(
                100 * np.mean(condition <= condition[channel])
            ),
            cpu_projection_replays={
                name: float(v[channel]) for name, v in predictions.items()
            },
        )
    return {
        "case": report["case"],
        "seed": report["seed"],
        "diagnostic_file": path.name,
        "diagnostic_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ordered_expert_ids": ids,
        "channels": len(condition),
        "median_cancellation_ratio": float(np.median(condition)),
        "witnesses": witnesses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics", type=Path, default=Path("results/m4pro-02-diagnostics")
    )
    parser.add_argument("--out", type=Path, default=Path("analysis/moe-witnesses.json"))
    args = parser.parse_args()
    files = sorted(args.diagnostics.glob("stack-b1-l*-seed*.json"))
    if not files:
        parser.error("No matching stack diagnostics")
    records = [row for path in files if (row := replay(path)) is not None]
    args.out.write_text(
        json.dumps(
            {
                "observable": FIELD,
                "scope": "Largest recorded vs-CPU violation for each failing method and fixture.",
                "definition": "Sum of absolute weighted CPU expert contributions divided by absolute signed sum, before FP16 output storage.",
                "interpretation": "A large ratio quantifies cancellation at this coordinate. GPU expert components were not recorded. Projection replays change CPU dot accumulation with the registered FP16 casts.",
                "records": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
