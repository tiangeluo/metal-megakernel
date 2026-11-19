"""Point a previously downloaded copy of this tracker at the current workspace page."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile

artifact = Path(__file__).resolve().parents[1] / "dist/index.html"
target = Path(sys.argv[1]).expanduser() if len(sys.argv) == 2 else Path.home() / "Downloads/megakernels-on-a-mac.html"
original_sha = "6ad0419cf859ad3d60aabb455397c994cb486e2c5e7b1e9461725e5916aa12df"
if target.is_symlink() and target.resolve() == artifact:
    print("Tracker already connected:", target)
    sys.exit(0)
approved = {original_sha, hashlib.sha256(artifact.read_bytes()).hexdigest()}
if target.is_symlink() or (target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() not in approved):
    sys.exit("The selected file is different. It was preserved. Open this file instead: "+str(artifact))
with tempfile.TemporaryDirectory(prefix=".tracker-link-", dir=target.parent) as folder:
    alias = Path(folder)/"tracker.html"
    alias.symlink_to(artifact)
    os.replace(alias, target)
print("Tracker connected. Refresh the existing Chrome tab:", target)
