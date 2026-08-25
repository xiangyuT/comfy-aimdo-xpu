"""Real-XPU capacity gate for same-model Windows VBAR page growth.

This is intentionally separate from the multi-queue retirement oracle.  That
test proves that a page is not unmapped too early; this one proves that a
sequence of already-retired pages does not grow without bound inside one model
while live allocator pressure is present.

Run each mode in a fresh process because the native mode switch is cached.
The default 16-page workload maps at most 512 MiB and never exceeds 2 GiB.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch


MIB = 1024 * 1024
VBAR_PAGE_SIZE = 32 * MIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("default", "reference", "async"), required=True
    )
    parser.add_argument("--pages", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.pages <= 1:
        parser.error("--pages must be greater than one")
    if arguments.pages * VBAR_PAGE_SIZE >= 2 * 1024 * MIB:
        parser.error("the mapped VBAR test ceiling must remain below 2 GiB")
    return arguments


def delta(after: dict[str, int], before: dict[str, int], name: str) -> int:
    return int(after[name]) - int(before[name])


def resident_pages(vbar: object) -> int:
    return sum(bool(state & 1) for state in vbar.get_residency())


def pressure_pulse(vbar: object) -> None:
    # Zero bytes execute the public owner-side pressure path without mapping or
    # pinning a page in the trigger VBAR.
    assert vbar.fault(vbar.base_addr, 0) is not None


def main() -> None:
    arguments = parse_args()
    if sys.platform != "win32":
        raise SystemExit("this capacity gate targets Windows XPU")
    if arguments.mode == "default":
        os.environ.pop("AIMDO_XPU_ASYNC_VBAR_RECLAIM", None)
    elif arguments.mode == "reference":
        os.environ["AIMDO_XPU_ASYNC_VBAR_RECLAIM"] = "0"
    else:
        os.environ["AIMDO_XPU_ASYNC_VBAR_RECLAIM"] = "1"

    if (
        not torch.xpu.is_available()
        or arguments.device >= torch.xpu.device_count()
    ):
        raise SystemExit(f"XPU device {arguments.device} is unavailable")

    torch.xpu.set_device(arguments.device)
    device = torch.device("xpu", arguments.device)

    # Import after the mode and device are fixed. Torch is deliberately loaded
    # first so the native Windows hook can attach to the active runtime.
    from comfy_aimdo import control

    source = None
    trigger = None
    initialized = False
    try:
        assert control.init(
            implementation="xpu",
            simple_vram_headroom=64 * MIB,
            xpu_allocator_mode="native_hook",
        )
        assert control.init_devices([arguments.device])
        initialized = True
        control.set_log_info()
        torch.xpu.empty_cache()

        from comfy_aimdo.model_vbar import ModelVBAR
        from comfy_aimdo.torch import aimdo_to_tensor

        source = ModelVBAR(
            arguments.pages * VBAR_PAGE_SIZE, arguments.device
        )
        allocations = [
            source.alloc(VBAR_PAGE_SIZE) for _ in range(arguments.pages)
        ]
        trigger = ModelVBAR(VBAR_PAGE_SIZE, arguments.device)
        source.prioritize()

        total = int(torch.xpu.get_device_properties(device).total_memory)
        stats_before = control.get_xpu_vmm_stats()
        # Force a deterministic live deficit without allocating a large tensor.
        # The physical workload remains <= pages * 32 MiB; only the budget
        # arithmetic is oversized.
        control.init(simple_vram_headroom=total * 4)

        resident_sequence: list[int] = []
        for index, allocation in enumerate(allocations):
            assert source.fault(allocation[1], allocation[2]) is not None
            tensor = aimdo_to_tensor(allocation, device)
            value = index % 251 + 1
            tensor.fill_(value)
            checksum = tensor.sum(dtype=torch.int64)
            source.unpin(allocation[1], allocation[2])
            torch.xpu.synchronize(arguments.device)
            assert int(checksum.cpu()) == value * VBAR_PAGE_SIZE
            del checksum, tensor

            current = resident_pages(source)
            resident_sequence.append(current)
            if arguments.mode == "reference":
                assert current == index + 1
            else:
                assert current <= 1

        pressure_pulse(trigger)
        torch.xpu.synchronize(arguments.device)
        pressure_pulse(trigger)
        final_resident = resident_pages(source)
        stats_after = control.get_xpu_vmm_stats()

        assert delta(stats_after, stats_before, "map_calls") == arguments.pages
        if arguments.mode == "reference":
            assert final_resident == arguments.pages
            assert delta(stats_after, stats_before, "unmap_calls") == 0
            assert delta(stats_after, stats_before, "retire_token_calls") == 0
        else:
            assert final_resident == 0
            assert delta(stats_after, stats_before, "unmap_calls") == arguments.pages
            assert (
                delta(stats_after, stats_before, "retire_token_calls")
                == arguments.pages
            )
            for name in (
                "retire_fence_submit_failures",
                "retire_queue_registration_failures",
                "retire_queue_identity_mismatches",
                "retire_fence_query_failures",
                "retire_shutdown_wait_failures",
            ):
                assert delta(stats_after, stats_before, name) == 0

        print(
            "CAPACITY_PASS "
            f"mode={arguments.mode} device={arguments.device} "
            f"pages={arguments.pages} mapped_mib={arguments.pages * 32} "
            f"resident_sequence={resident_sequence} "
            f"final_resident={final_resident} "
            f"map_delta={delta(stats_after, stats_before, 'map_calls')} "
            f"unmap_delta={delta(stats_after, stats_before, 'unmap_calls')}"
        )
    finally:
        if initialized:
            control.init(simple_vram_headroom=64 * MIB)
        gc.collect()
        if trigger is not None:
            trigger.__del__()
            trigger = None
        if source is not None:
            source.__del__()
            source = None
        gc.collect()
        if initialized:
            torch.xpu.synchronize(arguments.device)
            control.deinit()


if __name__ == "__main__":
    main()
