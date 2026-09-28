"""Write native source provenance without changing a distribution version."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, header = map(Path, sys.argv[1:])
git = ["git", "-c", "safe.directory=" + str(root), "-C", str(root)]
revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
names = subprocess.check_output([*git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
digest = hashlib.sha256()
for name in sorted(set(names.decode().split("\0")) - {""}):
    if not (name.startswith(("src/", "src-xpu/", "src-posix/", "src-win/")) or
            name in ("scripts/build-linux-xpu.sh", "scripts/build-windows-xpu.cmd", "scripts/write-source-identity.py")):
        continue
    path = root / name
    digest.update(name.encode() + b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest() if path.is_file() else b"deleted")
header.write_text("#define AIMDO_SOURCE_REVISION " + json.dumps(revision) + "\n" +
                  "#define AIMDO_SOURCE_CONTENT_SHA256 " + json.dumps(digest.hexdigest()) + "\n")
