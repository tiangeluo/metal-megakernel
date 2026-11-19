"""Measure the FP32 MoE control on the recorded layer-11 inputs using NumPy."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from decoder import model
from decoder.benchmark import model_weights
from diagnostics.precision_model import CONTROL, PRECISION_CONTRACT, finish

METHODS = ('cpu', 'compiled', 'staged')
ROOT = Path(__file__).resolve().parents[1]


def compare(a, b):
    ref, got = a.astype(np.float32), b.astype(np.float32)
    ratio = np.abs(ref - got) / (0.002 + 0.01 * np.abs(ref))
    return {'violating_channels': int((ratio > 1).sum()), 'largest_fraction_of_original_limit': float(ratio.max())}


def replay(capture_dir):
    c = model.Config()
    layer = model_weights(c)[1]
    records = []
    for length in (64, 4096):
        for seed in (20261118, 20261119):
            stem = capture_dir / f'crossover-stack-b1-l{length}-seed{seed}'
            summary = json.loads(stem.with_suffix('.json').read_text())
            archive = stem.with_suffix('.npz')
            digest = sha256(archive.read_bytes()).hexdigest()
            if (summary['capture_sha256'] != digest or not summary['wrapper_replay_exact']
                    or summary['length'] != length or summary['seed'] != seed or summary['config'] != vars(c)):
                raise ValueError(f'Capture mismatch for {stem.name}')
            with np.load(archive, allow_pickle=False) as data:
                wide_outputs = {}
                original_cpu_finish = {}
                for method in METHODS:
                    args = (np, c, layer, data['completed'], data[method+'__residual'], data[method+'__heads'],
                            data['hist_a'], data['hist_e'])
                    _, _, hist, ids = finish(*args)
                    wide_outputs[method] = hist[:, 0], ids
                    _, _, hist, _ = model.finish_layer(*args)
                    original_cpu_finish[method] = hist[:, 0]
                pairs = {}
                for a, b in combinations(METHODS, 2):
                    out_a, ids_a = wide_outputs[a]
                    out_b, ids_b = wide_outputs[b]
                    pairs[a+'_vs_'+b] = dict(
                        original_cpu_finish=compare(original_cpu_finish[a], original_cpu_finish[b]),
                        control_cpu_finish=compare(out_a, out_b),
                        control_route_ids_equal=bool(np.array_equal(ids_a, ids_b)),
                    )
                records.append(dict(length=length, seed=seed, archive_sha256=digest, pairs=pairs))
    return dict(control=CONTROL, contract=PRECISION_CONTRACT,
                scope='Offline layer-11 MoE output on saved original residual and head inputs. '
                      'All finishes recomputed on CPU. Full connected control and GPU kernel require device validation.',
                control_source_sha256=sha256((ROOT/'diagnostics/precision_model.py').read_bytes()).hexdigest(),
                fixtures=records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture_dir', type=Path)
    p.add_argument('--out', type=Path, default=Path('analysis/moe-precision-replay.json'))
    args = p.parse_args()
    result = replay(args.capture_dir)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in result['fixtures']:
        print(row['length'], row['seed'], {
            name: (pair['control_cpu_finish']['violating_channels'],
                   round(pair['control_cpu_finish']['largest_fraction_of_original_limit'], 3))
            for name, pair in row['pairs'].items()})
