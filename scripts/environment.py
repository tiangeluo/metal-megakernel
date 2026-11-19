"""Collect non-identifying hardware/runtime data and execute a real Metal dispatch."""
import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))


def probe():
    if platform.system() != "Darwin":
        return {"passed": False, "reason": "Metal requires macOS."}
    with tempfile.TemporaryDirectory(prefix="metal-probe-") as folder:
        exe = str(Path(folder) / "probe")
        result = command("xcrun", "clang", "-fobjc-arc", "-fmodules-cache-path="+folder+"/modules",
                         str(ROOT / "scripts/probe.m"), "-framework", "Foundation", "-framework", "Metal", "-o", exe)
        if result.returncode:
            return {"passed": False, "reason": "Native probe build failed.", "detail": result.stderr[-1000:]}
        run = command(exe)
        try:
            data = json.loads(run.stdout)
            if run.returncode and data.get("passed"):
                raise ValueError("unexpected process status")
            return data
        except ValueError:
            return {"passed": False, "reason": "Native probe did not return valid JSON.", "detail": run.stderr[-500:]}


def snapshot(path):
    specs = command("system_profiler", "SPHardwareDataType", "SPDisplaysDataType").stdout
    keys = {"Model Name", "Model Identifier", "Chip", "Chipset Model", "Memory", "Total Number of Cores"}
    hardware = {}
    for k, value in re.findall(r"^\s*([^:\n]+): (.+)$", specs, re.M):
        k = k.strip()
        if k in keys:
            hardware.setdefault(k, []).append(value.strip())
    packages = {}
    for name in ("numpy", "mlx", "mlx-metal", "pytest", "matplotlib"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    power = command("pmset", "-g", "batt").stdout.splitlines()
    metal = command("xcrun", "metal", "--version")
    data = {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "platform": platform.system(),
            "macos": platform.mac_ver()[0], "arch": platform.machine(), "hardware": hardware,
            "power": power[0].strip() if power else None,
            "xcode": command("xcodebuild", "-version").stdout.strip(), "offline_metal_compiler": metal.returncode == 0,
            "python": platform.python_version(), "packages": packages, "gpu_probe": probe()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(output.name, path)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/environment.json")
    args=parser.parse_args()
    info=snapshot(args.out)
    print(json.dumps(info["gpu_probe"]))
    sys.exit(0 if info["gpu_probe"]["passed"] else 2)
