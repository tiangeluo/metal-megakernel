"""Mark exactly one validated, warmed decode step for an Instruments capture.

Use a Metal System Trace configuration with Points of Interest enabled and launch
this module. Native encoded counts and timings alone do not establish framework
dispatch counts. See bench/README.md for artifact association.
"""
import argparse
import ctypes
from pathlib import Path
import subprocess
import tempfile

from .adapters import prepare
from .protocol import ATTENTION, STACK
from .run import preflight


def main():
    cases={c.name:c for c in ATTENTION+STACK}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=sorted(cases),required=True)
    parser.add_argument("--method",required=True)
    parser.add_argument("--seed",type=int,default=20261118)
    parser.add_argument("--environment",type=Path,default=Path("results/environment.json"))
    args=parser.parse_args()
    case=cases[args.case]
    if args.method not in case.methods:
        parser.error("Requested schedule is not registered for this case")
    preflight(args.environment)
    with tempfile.TemporaryDirectory() as tmp:
        dylib=Path(tmp)/"markers.dylib"
        subprocess.run(["xcrun","clang","-O2","-Wall","-Werror","-dynamiclib",
                        str(Path(__file__).with_name("trace_marker.c")),"-o",str(dylib)],check=True)
        marks=ctypes.CDLL(str(dylib))
        marks.decode_marker_begin.argtypes=[ctypes.c_char_p]
        marks.decode_marker_begin.restype=None
        marks.decode_marker_end.argtypes=[]
        marks.decode_marker_end.restype=None
        runners,check,close=prepare(case,args.seed)
        try:
            runner=runners[args.method]
            runner.reset()
            runner.step()
            check(runner.read())
            for _ in range(4):
                runner.reset()
                runner.step()
            runner.reset()
            label=(case.name+"/"+args.method).encode()
            marks.decode_marker_begin(label)
            try:
                runner.step()
            finally:
                marks.decode_marker_end()
            check(runner.read())
        finally:
            close()
        print("Captured candidate interval:",label.decode())


if __name__=="__main__":
    main()
