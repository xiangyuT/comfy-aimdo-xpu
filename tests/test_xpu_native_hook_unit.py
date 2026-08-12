import platform
import shutil
import subprocess
from pathlib import Path

import pytest


def test_linux_ur_hook_state_machine_and_test_helper_build():
    if platform.system() != "Linux":
        pytest.skip("Linux Unified Runtime hook test")
    if shutil.which("icpx") is None:
        pytest.skip("requires the oneAPI DPC++ compiler")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", str(root / "scripts" / "test-linux-xpu-hook.sh")],
        cwd=root,
        check=True,
    )
