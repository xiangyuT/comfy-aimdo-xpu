import argparse
import json
import time

import torch

from comfy_aimdo import control


SIZES = tuple(size << 20 for size in (8, 12, 10, 14, 9, 13, 11))


def _physical_alloc_calls(mode):
    if mode == "native_hook":
        return control.get_xpu_ur_hook_stats()["tracked_alloc_calls"]
    if mode == "global":
        return control.get_xpu_vmm_stats()[
            "torch_allocator_physical_alloc_calls"
        ]
    return None


def _run(cycles, device):
    for _ in range(cycles):
        for size in SIZES:
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
            del tensor
    torch.xpu.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("default", "global", "native_hook"), required=True
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-cycles", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("xpu", args.device)
    if args.mode != "default":
        assert control.init(
            implementation="xpu",
            simple_vram_headroom=64 << 20,
            xpu_allocator_mode=args.mode,
        )
    torch.xpu.set_device(device)
    if args.mode != "global":
        torch.xpu.empty_cache()
    allocator_before = (
        None
        if args.mode == "global"
        else torch.xpu.memory._get_current_allocator().allocator()
    )
    if args.mode != "default":
        assert control.init_devices([args.device])
    allocator_after = torch.xpu.memory._get_current_allocator().allocator()

    _run(args.warmup_cycles, device)
    samples = []
    physical_samples = []
    for _ in range(args.trials):
        physical_before = _physical_alloc_calls(args.mode)
        start = time.perf_counter()
        _run(args.cycles, device)
        elapsed = time.perf_counter() - start
        physical_after = _physical_alloc_calls(args.mode)
        samples.append(elapsed)
        physical_samples.append(
            None
            if physical_before is None
            else int(physical_after) - int(physical_before)
        )

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "device": str(device),
                "mode": args.mode,
                "native_allocator_preserved": (
                    args.mode != "global" and allocator_before == allocator_after
                ),
                "sizes_bytes": SIZES,
                "allocations_per_trial": args.cycles * len(SIZES),
                "warmup_cycles": args.warmup_cycles,
                "seconds": samples,
                "physical_alloc_calls": physical_samples,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.mode != "default":
        control.deinit()


if __name__ == "__main__":
    main()
