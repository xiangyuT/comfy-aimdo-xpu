import platform
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def regression(tmp_path_factory):
    compiler = shutil.which("gcc")
    if platform.system() != "Linux" or compiler is None:
        pytest.skip("Linux C regression requires gcc")
    binary = tmp_path_factory.mktemp("vbar-fault") / "regression"
    source = Path(__file__).with_name("vbar_fault_regression.c")
    subprocess.run([compiler, "-std=c11", "-DAIMDO_XPU", "-ffunction-sections",
                    "-fdata-sections", str(source), "-Wl,--gc-sections", "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("scenario", range(7), ids=(
    "partial-oom", "independent-pin", "retry-oom", "unmap-retry", "destroy-remap",
    "destroy-remap-terminal", "sync-terminal",
))
def test_partial_fault_ownership_and_recovery(regression, scenario):
    import resource
    result = subprocess.run([str(regression), str(scenario)], capture_output=True, text=True,
                            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    assert result.returncode == 0, result.stderr
