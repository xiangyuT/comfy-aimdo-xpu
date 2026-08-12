"""Attribute the Windows Unified Runtime hook's cost.

The gradient ladder regressed on this branch. Two candidates were recorded in
docs/WINDOWS_XPU_UR_ALLOCATOR_HOOK.md: the caller classification, which has
since been made roughly three times cheaper, and the synthetic out-of-memory
design itself, because PyTorch answers a failed allocation by releasing its
*entire* cache before retrying.

This benchmark separates them. It runs the same allocate/free cycle twice, once
with headroom so the hook decides nothing, and once with the budget deliberately
short so every PyTorch request is refused and retried. The difference is the
cost of the decision, not the cost of making it, and the hook's own timers show
how much of that difference is spent inside the hook.

    <portable>\\python_embeded\\python.exe -s tests\\benchmark_windows_ur_hook.py
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MIB = 1024 * 1024


def cycle(torch, device, block_mib, blocks, rounds):
    """Allocate and release a working set repeatedly, like a sampler step.

    Sizes are shifted every round on purpose. A sampler does not request the
    same bytes each time, so PyTorch's free-block reuse is imperfect and real
    driver allocations keep occurring; a fixed size would be served entirely
    from cache after the first round and would measure nothing.
    """
    started = time.perf_counter()
    for round_index in range(rounds):
        held = []
        for block_index in range(blocks):
            size_mib = block_mib + (round_index * 3 + block_index) % 7
            held.append(
                torch.empty(size_mib * MIB, dtype=torch.uint8, device=device)
            )
        torch.xpu.synchronize()
        del held
    torch.xpu.synchronize()
    return time.perf_counter() - started


def snapshot(control):
    stats = control.get_xpu_ur_hook_stats()
    timing = control.get_xpu_ur_hook_timing()
    merged = dict(stats)
    merged.update(timing)
    return merged


def delta(before, after, key):
    return after.get(key, 0) - before.get(key, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-mib", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument(
        "--pressure-reserve-mib",
        type=int,
        default=29000,
        help="reserve that leaves less room than the working set needs",
    )
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    from comfy_aimdo import control

    if not control.init(implementation="xpu", simple_vram_headroom=512 * MIB):
        print("control.init() failed", file=sys.stderr)
        return 1
    import torch

    if not control.init_devices([arguments.device]):
        print("control.init_devices() failed", file=sys.stderr)
        return 1
    device = torch.device("xpu", arguments.device)

    working_set = arguments.block_mib * arguments.blocks
    print(
        f"working set {working_set} MiB, {arguments.rounds} rounds, "
        f"device {device}"
    )

    # Warm up so allocator growth is not attributed to either phase.
    cycle(torch, device, arguments.block_mib, arguments.blocks, 2)
    torch.xpu.empty_cache()

    relaxed_before = snapshot(control)
    relaxed_seconds = cycle(
        torch, device, arguments.block_mib, arguments.blocks, arguments.rounds
    )
    relaxed_after = snapshot(control)

    torch.xpu.empty_cache()
    torch.xpu.synchronize()

    # Now make the budget short enough that every PyTorch request is refused.
    control.lib.set_simple_vram_headroom(arguments.pressure_reserve_mib * MIB)
    tight_before = snapshot(control)
    tight_seconds = cycle(
        torch, device, arguments.block_mib, arguments.blocks, arguments.rounds
    )
    tight_after = snapshot(control)
    control.lib.set_simple_vram_headroom(512 * MIB)

    allocations = delta(relaxed_before, relaxed_after, "alloc_calls")
    tight_allocations = delta(tight_before, tight_after, "alloc_calls")
    relaxed_hook_ns = delta(relaxed_before, relaxed_after, "hook_ns")
    tight_hook_ns = delta(tight_before, tight_after, "hook_ns")
    synthetic = delta(tight_before, tight_after, "synthetic_oom_calls")
    reclaim_calls = delta(tight_before, tight_after, "native_reclaim_free_calls")
    reclaim_bytes = delta(tight_before, tight_after, "native_reclaim_free_bytes")
    classify_ns = delta(tight_before, tight_after, "classify_ns")
    classify_calls = delta(tight_before, tight_after, "classify_calls")

    print()
    print(f"relaxed budget : {relaxed_seconds:8.3f} s  "
          f"alloc_calls={allocations:5d}  "
          f"hook={relaxed_hook_ns / 1e9:.3f} s")
    print(f"short budget   : {tight_seconds:8.3f} s  "
          f"alloc_calls={tight_allocations:5d}  "
          f"hook={tight_hook_ns / 1e9:.3f} s")
    print()
    print(f"slowdown       : {tight_seconds / max(relaxed_seconds, 1e-9):.2f}x "
          f"({tight_seconds - relaxed_seconds:+.3f} s)")
    print(f"  of which inside the hook : "
          f"{(tight_hook_ns - relaxed_hook_ns) / 1e9:+.3f} s")
    print(f"  of which classification  : {classify_ns / 1e9:+.3f} s "
          f"over {classify_calls} calls")
    print()
    print(f"synthetic_oom_calls        : {synthetic}")
    print(f"torch cache returned       : {reclaim_calls} frees, "
          f"{reclaim_bytes / MIB:.0f} MiB")
    if synthetic:
        print(f"extra driver allocations   : "
              f"{tight_allocations - allocations} "
              f"({tight_allocations / max(allocations, 1):.2f}x)")

    summary = {
        "relaxed_seconds": relaxed_seconds,
        "tight_seconds": tight_seconds,
        "relaxed_alloc_calls": allocations,
        "tight_alloc_calls": tight_allocations,
        "relaxed_hook_ns": relaxed_hook_ns,
        "tight_hook_ns": tight_hook_ns,
        "classify_ns": classify_ns,
        "classify_calls": classify_calls,
        "synthetic_oom_calls": synthetic,
        "native_reclaim_free_calls": reclaim_calls,
        "native_reclaim_free_bytes": reclaim_bytes,
    }
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(summary, indent=2))

    control.deinit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
