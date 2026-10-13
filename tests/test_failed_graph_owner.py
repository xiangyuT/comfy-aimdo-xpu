import platform
from pathlib import Path
import shutil
import subprocess

import pytest


def test_failed_graph_final_free_preserves_backing_until_abort(tmp_path):
    compiler = shutil.which("gcc")
    if platform.system() != "Linux" or compiler is None:
        pytest.skip("Linux C regression requires gcc")
    import resource
    source = Path(__file__).with_name("failed_graph_owner_regression.c")
    binary = tmp_path / "failed-owner"
    subprocess.run([compiler, "-std=c11", "-DAIMDO_XPU", "-ffunction-sections",
                    "-fdata-sections", str(source), "-Wl,--gc-sections", "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True,
                            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    assert result.returncode == 0, result.stderr
