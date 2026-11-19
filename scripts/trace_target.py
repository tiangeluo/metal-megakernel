"""Start the traced benchmark in the package environment selected by its caller."""
from importlib import metadata, util
import json
from pathlib import Path
import sys


def configure_imports(package_dirs):
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), *package_dirs]


if __name__ == "__main__":
    separator = sys.argv.index("--")
    configure_imports(sys.argv[1:separator])
    del sys.argv[1:separator + 1]
    core = util.find_spec("mlx.core")
    if core is None or not core.origin:
        raise ModuleNotFoundError("The selected benchmark environment has no mlx.core extension")
    print("Trace target packages: " + json.dumps(
        {name: metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")}, sort_keys=True),
        flush=True)
    from bench.trace import main

    main()
