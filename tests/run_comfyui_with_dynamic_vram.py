"""Run a ComfyUI main module with DynamicVRAM explicitly enabled.

This launcher changes only the process argument list. It is intended for the
canonical workflow runner's COMFYUI_LAUNCHER hook on non-CUDA backends, where
ComfyUI does not enable DynamicVRAM automatically.
"""

import argparse
import atexit
import json
import os
import runpy
import signal
import sys
from pathlib import Path


_stats_emitted = False


def _emit_xpu_vmm_stats():
    global _stats_emitted
    if _stats_emitted:
        return
    try:
        from comfy_aimdo import control

        stats = control.get_xpu_vmm_stats()
    except Exception as error:
        print(f"[AIMDO XPU VMM STATS] unavailable: {error}", flush=True)
        return
    if stats:
        print(f"[AIMDO XPU VMM STATS] {json.dumps(stats, sort_keys=True)}",
              flush=True)
        _stats_emitted = True


def _terminate(signum, _frame):
    _emit_xpu_vmm_stats()
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comfyui-main", required=True)
    known, remaining = parser.parse_known_args()
    if "--enable-dynamic-vram" not in remaining:
        remaining.append("--enable-dynamic-vram")
    if os.environ.get("AIMDO_XPU_DEBUG_HANG") == "1" and "--debug-hang" not in remaining:
        remaining.append("--debug-hang")

    main_path = Path(known.comfyui_main).resolve()
    sys.path.insert(0, str(main_path.parent))
    sys.argv = [str(main_path), *remaining]
    atexit.register(_emit_xpu_vmm_stats)
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
