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


def test_windows_ur_hook_state_machine_and_classification_cost():
    if platform.system() != "Windows":
        pytest.skip("Windows Unified Runtime hook test")
    root = Path(__file__).resolve().parents[1]
    if not (root / "build" / "detours-src" / "lib.X64" / "detours.lib").exists():
        pytest.skip("requires Detours; run scripts\\build-windows-detours.cmd")
    completed = subprocess.run(
        [str(root / "scripts" / "test-windows-xpu-hook.cmd")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    # The timing lines are the point of the test as much as the assertions, so
    # surface them even on success.
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASSED" in completed.stdout
