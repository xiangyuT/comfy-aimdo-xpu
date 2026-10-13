"""Run real core transitions with CPU driver callbacks before device probes."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_replay_free_keeps_vmm_owned_until_work_is_drained(tmp_path):
    compiler = shutil.which('gcc')
    if compiler is None:
        pytest.skip('requires gcc for the production-core state machine')
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / 'malloc-graph-replay-free'
    compiled = subprocess.run([compiler, '-std=c11', '-DAIMDO_XPU', '-ffunction-sections', '-fdata-sections',
                    '-I' + str(root / 'src'), str(root / 'tests/malloc_graph_replay_free_unit.c'),
                    str(root / 'src/debug.c'), '-Wl,--gc-sections', '-o', str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert 'PASS replay-free eligibility' in result.stdout
