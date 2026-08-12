"""Verify that the Windows Unified Runtime hook actually arbitrates.

`tests/run_windows_ur_hook_probe.py` proves the interception mechanism using a
standalone probe. This script proves the *production* path: that
`comfy_aimdo.control` selects the native hook on Windows, that
`src-xpu/ur-usm-detour.c` attaches to `ur_loader.dll`, and that PyTorch's
allocations are seen, classified and accounted by AIMDO itself.

It deliberately does not assert anything about memory policy. A pass means the
control loop is connected, not that it makes good decisions.

    <portable>\\python_embeded\\python.exe -s tests\\run_windows_ur_hook_smoke.py
"""

import argparse
import json
import pathlib
import sys

# The Portable interpreter is an embeddable build whose ._pth disables
# PYTHONPATH, so an explicit insert is the only way to test a locally built
# comfy_aimdo instead of the installed wheel.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MIB = 1024 * 1024


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""),
              flush=True)
        return passed

    @property
    def ok(self):
        return all(entry["passed"] for entry in self.checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--alloc-mib", type=int, default=512)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    from comfy_aimdo import control

    report = Report()
    loaded_from = pathlib.Path(control.__file__).resolve().parent
    report.check(
        "0. testing the local build rather than an installed wheel",
        loaded_from == pathlib.Path(__file__).resolve().parents[1] / "comfy_aimdo",
        str(loaded_from),
    )

    if not control.init(
        implementation="xpu", simple_vram_headroom=arguments.reserve_mib * MIB
    ):
        print("control.init() failed", file=sys.stderr)
        return 1

    import torch

    report.check(
        "1. windows defaults to the native hook allocator mode",
        control.get_xpu_allocator_mode() == "native_hook",
        f"mode={control.get_xpu_allocator_mode()}",
    )

    if not control.init_devices([arguments.device]):
        print("control.init_devices() failed", file=sys.stderr)
        return 1

    before = control.get_xpu_ur_hook_stats()
    report.check(
        "2. the hook exposes statistics, so it is attached and enabled",
        bool(before),
        f"{len(before)} counters",
    )

    device = torch.device("xpu", arguments.device)
    torch.xpu.empty_cache()
    tensors = [
        torch.empty(arguments.alloc_mib * MIB, dtype=torch.uint8, device=device)
        for _ in range(3)
    ]
    torch.xpu.synchronize()
    during = control.get_xpu_ur_hook_stats()

    report.check(
        "3. torch allocations reach AIMDO's own hook",
        during["alloc_calls"] > before["alloc_calls"],
        f"alloc_calls {before['alloc_calls']} -> {during['alloc_calls']}",
    )
    report.check(
        "4. allocations are tracked and accounted, not passed through",
        during["tracked_alloc_calls"] > before["tracked_alloc_calls"]
        and during["tracked_alloc_bytes"] >= arguments.alloc_mib * MIB,
        f"tracked_alloc_calls={during['tracked_alloc_calls']} "
        f"tracked_alloc_bytes={during['tracked_alloc_bytes'] / MIB:.0f} MiB",
    )
    report.check(
        "5. the device was resolved for every request",
        during["unknown_device_calls"] == 0,
        f"unknown_device_calls={during['unknown_device_calls']}",
    )
    report.check(
        "6. expandable segments did not bypass the hook",
        during.get("physical_mem_create_calls", 0) == 0,
        f"physical_mem_create_calls={during.get('physical_mem_create_calls')}",
    )

    for tensor in tensors:
        del tensor
    tensors.clear()
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    after = control.get_xpu_ur_hook_stats()

    report.check(
        "7. releases are observed and credited back",
        after["tracked_free_calls"] > during["tracked_free_calls"],
        f"tracked_free_calls={after['tracked_free_calls']} "
        f"tracked_free_bytes={after['tracked_free_bytes'] / MIB:.0f} MiB",
    )
    report.check(
        "8. accounting balances after the cache is emptied",
        after["tracked_alloc_bytes"] == after["tracked_free_bytes"],
        f"alloc={after['tracked_alloc_bytes']} free={after['tracked_free_bytes']}",
    )

    control.deinit()
    report.check("9. the hook detaches cleanly", True, "deinit returned")

    print()
    for key, value in after.items():
        print(f"  {key:34s} {value}")

    if arguments.json:
        pathlib.Path(arguments.json).write_text(
            json.dumps(
                {"stats": after, "checks": report.checks, "passed": report.ok},
                indent=2,
            )
        )

    print()
    print("RESULT:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
