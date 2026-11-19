"""CPU control for rounding sensitivity in the two-layer FP16 replay.

Run from the repository root with python -m diagnostics.projection_sensitivity.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from unittest.mock import patch

import numpy as np

from bench.protocol import equal_stack
from decoder import model
from decoder.benchmark import prepare_case


def matmul_fp32(xp, x, w):
    return (x.astype(np.float32) @ w.astype(np.float32)).astype(np.float16)


def partials_fp32(xp, x, w):
    groups = (
        32  # All full-width contracted dimensions in the study are divisible by 32.
    )
    lhs = x.reshape(*x.shape[:-1], groups, -1)
    rhs = w.reshape(groups, -1, w.shape[-1])
    partials = np.einsum("...gk,gkj->...gj", lhs, rhs, dtype=np.float32)
    return partials.sum(axis=-2, dtype=np.float32).astype(np.float16)


def einsum_fp64(xp, x, w):
    return np.einsum("...i,ij->...j", x, w, dtype=np.float64).astype(np.float16)


def compare(reference, alternative, position):
    failures = []
    for name, expected in reference.items():
        try:
            equal_stack({name: expected}, {name: alternative[name]}, position)
        except AssertionError:
            failures.append(name)

    metrics = {}
    for name in ("layer10_post_moe", "layer11_post_moe", "summary"):
        a = reference[name].astype(np.float32)
        b = alternative[name].astype(np.float32)
        # The older two Conv4 history rows come from the seed. Report errors on
        # the new vector alone so the unchanged rows cannot dilute the metrics.
        if name.endswith("post_moe"):
            a, b = a[:, 0, :], b[:, 0, :]
        delta = np.abs(b - a)
        bound = 0.002 + 0.01 * np.abs(a)
        outliers = []
        for batch, channel in np.argwhere(delta > bound):
            index = batch, channel
            outliers.append(
                dict(
                    batch=int(batch),
                    channel=int(channel),
                    reference=float(a[index]),
                    alternative=float(b[index]),
                    absolute_error=float(delta[index]),
                    allowed_error=float(bound[index]),
                )
            )
        metrics[name] = dict(
            values=int(a.size),
            max_abs=float(delta.max()),
            relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(a)),
            outside_original_bound=outliers,
        )
    return dict(
        routes={
            name: alternative[name].tolist()
            for name in reference
            if name.endswith("route_ids")
        },
        failed_observables=failures,
        metrics=metrics,
    )


def replay(seed, length):
    case = prepare_case("cpu", length, seed=seed)
    try:
        case.run()
        expected = case.output_numpy()
        runs = {}
        for variant in (matmul_fp32, partials_fp32, einsum_fp64):
            case.reset()
            # Only the diagnostic process changes this function, and only during
            # this untimed replay. All attention and routing code stays identical.
            with patch.object(model, "project", variant):
                case.run()
            runs[variant.__name__] = compare(expected, case.output_numpy(), length - 1)
        return dict(
            seed=seed,
            length=length,
            batch=1,
            reference_routes={
                k: v.tolist() for k, v in expected.items() if k.endswith("route_ids")
            },
            variants=runs,
        )
    finally:
        case.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    model_path = Path(model.__file__)
    report = dict(
        backend="NumPy CPU",
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        geometry=asdict(model.Config()),
        decoder_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        reference="Current decoder.model.project (FP32 einsum, FP16 output)",
        cases=[
            replay(seed, length)
            for seed in (20261118, 20261119)
            for length in (64, 4096)
        ],
    )
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
