"""Minimal Linux/Windows XPU memory-policy regression reproducer.

This intentionally creates synthetic pressure with small VBARs. It does not
modify ComfyUI configuration and restores AIMDO's headroom before exiting.
"""

import gc
import platform
import sys
from pathlib import Path

import torch

# Embedded Python distributions can ignore PYTHONPATH. Make the reproducer
# test the checkout that contains this file instead of an older installed
# comfy-aimdo wheel.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from comfy_aimdo import control


MIB = 1 << 20


def _resident(vbar):
    return [value & 1 for value in vbar.get_residency()]


def main():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit("SKIP: torch.xpu is not available")
    if control.detect_vendor() != "xpu":
        raise SystemExit("SKIP: comfy-aimdo did not select the XPU backend")

    device_index = torch.xpu.current_device()
    device = torch.device("xpu", device_index)
    lower_vbar = None
    active_vbar = None
    activation = None

    assert control.init(
        implementation="xpu", simple_vram_headroom=64 * MIB
    )
    assert control.init_devices([device_index])
    from comfy_aimdo.model_vbar import ModelVBAR

    try:
        assert control.empty_xpu_allocator_cache(wait=True)

        lower_vbar = ModelVBAR(32 * MIB, device_index)
        lower_alloc = lower_vbar.alloc(32 * MIB)
        assert lower_vbar.fault(lower_alloc[1], lower_alloc[2]) is not None
        lower_vbar.unpin(lower_alloc[1], lower_alloc[2])

        active_vbar = ModelVBAR(64 * MIB, device_index)
        active_alloc = active_vbar.alloc(64 * MIB)
        assert active_vbar.fault(active_alloc[1], active_alloc[2]) is not None
        active_vbar.unpin(active_alloc[1], active_alloc[2])
        active_vbar.prioritize()

        total = torch.xpu.get_device_properties(device).total_memory
        recorded = control.get_total_vram_usage()
        if recorded >= total:
            raise RuntimeError(
                f"unexpected AIMDO usage {recorded} >= device memory {total}"
            )

        # With simple accounting, the next 64 MiB physical allocation creates
        # a 64 MiB deficit: enough to exhaust the 32 MiB lower-priority VBAR
        # and then distinguish exact Linux pressure from Windows speculation.
        control.init(simple_vram_headroom=total - recorded)

        before_lower = _resident(lower_vbar)
        before_active = _resident(active_vbar)
        activation = torch.empty(64 * MIB, dtype=torch.uint8, device=device)
        activation.fill_(17)
        torch.xpu.synchronize()
        assert int(activation[0].cpu()) == 17

        if sys.platform == "win32":
            # The Level Zero callback records growth but cannot safely unmap.
            assert _resident(active_vbar) == before_active
            del activation
            activation = None
            torch.xpu.empty_cache()
            active_vbar.prioritize()

            after_lower = _resident(lower_vbar)
            after_active = _resident(active_vbar)
            assert after_lower == [0], after_lower
            assert after_active == before_active, (before_active, after_active)
            active_vbar.set_watermark(32 * MIB)
            established_watermark = active_vbar.get_watermark()
            active_vbar.prioritize()
            reprioritized_watermark = active_vbar.get_watermark()
            assert reprioritized_watermark == active_vbar.get_nr_pages()
            expected = (
                "speculative reclaim preserved the active VBAR and "
                "reprioritization reopened its fault range"
            )
        else:
            after_lower = _resident(lower_vbar)
            after_active = _resident(active_vbar)
            assert after_lower == [0], after_lower
            assert after_active != before_active, (before_active, after_active)
            established_watermark = active_vbar.get_watermark()
            active_vbar.prioritize()
            reprioritized_watermark = active_vbar.get_watermark()
            assert reprioritized_watermark == active_vbar.get_nr_pages()
            expected = "exact live pressure was allowed to reclaim active pages"

        print(f"platform={platform.system()} device={device}")
        print(f"lower: {before_lower} -> {after_lower}")
        print(f"active: {before_active} -> {after_active}")
        print(
            "reprioritize watermark: "
            f"{established_watermark} -> {reprioritized_watermark}"
        )
        print(f"PASS: {expected}")
        return 0
    finally:
        if activation is not None:
            del activation
        torch.xpu.synchronize()
        control.empty_xpu_allocator_cache(wait=True)
        control.init(simple_vram_headroom=64 * MIB)
        if active_vbar is not None:
            active_vbar.__del__()
        if lower_vbar is not None:
            lower_vbar.__del__()
        gc.collect()
        control.deinit()


if __name__ == "__main__":
    raise SystemExit(main())
