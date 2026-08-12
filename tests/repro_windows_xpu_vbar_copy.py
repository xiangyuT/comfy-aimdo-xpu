"""Minimal reproducer for the device-1 copy failure.

ComfyUI fails with UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY when copying a small
host tensor into a VBAR-backed destination on the second GPU. Every operand has
been checked at the failure point and is valid: the destination is mapped,
pinned, reported by the driver as device memory, and the queue shares AIMDO's
Level Zero context; the source is host USM.

The existing VBAR reproducer only writes to VBAR memory with a kernel
(``fill_``), which is a different operation. This one performs the copy that
actually fails.

    <portable>\\python_embeded\\python.exe -s tests\\repro_windows_xpu_vbar_copy.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIB = 1024 ** 2
VBAR_PAGE_SIZE = 32 * MIB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pages", type=int, default=40)
    parser.add_argument("--copy-bytes", type=int, default=607744,
                        help="size ComfyUI was copying when it failed")
    parser.add_argument("--reserve-mib", type=int, default=4096)
    args = parser.parse_args()

    from comfy_aimdo import control

    if not control.init(implementation="xpu",
                        simple_vram_headroom=args.reserve_mib * MIB):
        raise SystemExit("control.init() failed")

    import torch

    if not control.init_devices([args.device]):
        raise SystemExit("control.init_devices() failed")

    from comfy_aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin
    from comfy_aimdo.torch import aimdo_to_tensor

    device = torch.device("xpu", args.device)
    vbar = ModelVBAR(args.pages * VBAR_PAGE_SIZE, device=args.device)
    allocations = [vbar.alloc(VBAR_PAGE_SIZE) for _ in range(args.pages)]

    source = torch.empty(args.copy_bytes, dtype=torch.uint8, device="cpu")
    source.fill_(7)
    print(f"device={args.device} pages={args.pages} "
          f"copy_bytes={args.copy_bytes}")

    failures = 0
    for index, allocation in enumerate(allocations):
        if vbar_fault(allocation) is None:
            print(f"  page {index}: fault miss")
            continue
        destination = aimdo_to_tensor(allocation, device)
        view = destination.view(torch.uint8)[: args.copy_bytes]
        try:
            view.copy_(source, non_blocking=True)
            torch.xpu.synchronize()
        except Exception as error:
            failures += 1
            print(f"  page {index}: COPY FAILED "
                  f"{type(error).__name__}: {str(error)[:100]}")
            if failures >= 3:
                break
        vbar_unpin(allocation)

    print(f"pages={args.pages} copy_failures={failures}")
    control.deinit()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
