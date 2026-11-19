"""Capture one warmed and correctness-checked MLX decode step for inspection in Xcode."""
import argparse
import os
from pathlib import Path

from bench.protocol import compare_attention
from reference.attention import step as oracle
from reference.fixtures import make_case
from .schedules import require_gpu, prepare_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["compiled", "hybrid"], required=True)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if min(args.length, args.batch) < 1:
        parser.error("length and batch must be positive")
    if args.out.suffix != ".gputrace" or args.out.exists():
        parser.error("--out must be a new .gputrace path")
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        parser.error("run with MTL_CAPTURE_ENABLED=1")
    mx = require_gpu()
    token, weights, initial = make_case(batch=args.batch, length=args.length)
    pos = args.length - 1
    state = {k: v.copy() for k, v in initial.items()}
    expected = oracle(token, weights, state, position=pos), state
    case = prepare_case(token, weights, initial, position=pos, method=args.method)
    case.run()
    compare_attention(expected, case.readback(), initial, pos)
    case.fresh()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(args.out.resolve()))
    try:
        case.run()
    finally:
        mx.metal.stop_capture()
    compare_attention(expected, case.readback(), initial, pos)
    if not args.out.exists():
        raise RuntimeError("MLX finished capture without creating the requested trace")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
