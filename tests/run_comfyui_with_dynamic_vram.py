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
import threading
import time
from pathlib import Path


_stats_emitted = False
_stats_stop = threading.Event()


def _xpu_stats_snapshot():
    from comfy_aimdo import control

    vmm = control.get_xpu_vmm_stats()
    if not vmm:
        return None
    allocator = control.get_xpu_allocator_memory_stats()
    return {
        "monotonic_seconds": round(time.monotonic(), 3),
        "recorded_vram_bytes": control.get_total_vram_usage(),
        "allocator_active_bytes": allocator[0],
        "allocator_reserved_bytes": allocator[1],
        "physical_alloc_calls": vmm["torch_allocator_physical_alloc_calls"],
        "physical_alloc_bytes": vmm["torch_allocator_physical_alloc_bytes"],
        "physical_release_calls": vmm[
            "torch_allocator_physical_release_calls"
        ],
        "physical_release_bytes": vmm[
            "torch_allocator_physical_release_bytes"
        ],
        "map_bytes": vmm["map_bytes"],
        "unmap_bytes": vmm["unmap_bytes"],
        "host_to_device_bytes": vmm["host_to_device_bytes"],
    }


def _sample_xpu_stats(interval):
    while not _stats_stop.wait(interval):
        try:
            snapshot = _xpu_stats_snapshot()
        except Exception:
            continue
        if snapshot is not None:
            print(
                f"[AIMDO XPU PERIODIC STATS] "
                f"{json.dumps(snapshot, sort_keys=True)}",
                flush=True,
            )


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
    stats_interval = float(os.environ.get("AIMDO_XPU_STATS_INTERVAL", "0"))
    if stats_interval > 0:
        stats_thread = threading.Thread(
            target=_sample_xpu_stats,
            args=(stats_interval,),
            name="aimdo-xpu-stats",
            daemon=True,
        )
        stats_thread.start()
        atexit.register(_stats_stop.set)
    atexit.register(_emit_xpu_vmm_stats)
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
