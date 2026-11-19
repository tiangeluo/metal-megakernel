"""Test the connected FP32-expert control under a different CPU dot-product order."""
from contextlib import closing
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.diagnose_stack import compare_outputs
from decoder import model
from diagnostics import precision_model
from diagnostics.projection_sensitivity import matmul_fp32


def four_partial_dots(xp, x, bank, ids):
    """Partition the dot as in the selected-expert shader; NumPy controls local rounding."""
    assert xp is np
    batch, top_k = ids.shape
    out = np.empty((batch, top_k, bank.shape[-1]), dtype=np.float32)
    for b, row in enumerate(ids):
        for rank, expert in enumerate(row):
            lhs = x[b, rank] if x.ndim == 3 else x[b]
            matrix = bank[expert]
            parts = [np.einsum("i,ij->j", lhs[i::4], matrix[i::4], dtype=np.float32)
                     for i in range(4)]
            out[b, rank] = (parts[0] + parts[1]) + (parts[2] + parts[3])
    return out


def check(length, seed):
    with closing(precision_model.prepare_case("cpu", length, seed=seed)) as case:
        case.run()
        reference = case.output_numpy()
        case.reset()
        with patch.object(model, "project", matmul_fp32), \
             patch.object(precision_model, "selected_project", four_partial_dots):
            case.run()
            difference = compare_outputs(reference, case.output_numpy(), length - 1)
    return {"length": length, "seed": seed, "passed": difference["passed"],
            "observables": len(reference), "failed": difference["failed_observables"],
            "max_abs": {name: max(region["max_abs_finite"] for region in details["regions"].values())
                        for name, details in difference["observables"].items()}}


if __name__ == "__main__":
    output = Path(__file__).with_name("precision-sensitivity.json")
    rows = [check(length, seed) for length in (64, 4096) for seed in (20261118, 20261119)]
    output.write_text(json.dumps({"device": "CPU", "control": precision_model.CONTROL,
                                 "fixtures": rows}, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(row["length"], row["seed"], "passed" if row["passed"] else row["failed"])
