"""Windows XPU native allocation pressure reproducer.

This reproducer isolates the Windows `sycl::malloc_device` behaviour that any
AIMDO DynamicVRAM design on this platform has to be built around. It needs only
Torch XPU; AIMDO itself is not loaded, so the results describe the platform, not
a particular AIMDO revision.

Modes:

``budget``
    Grow a retained, touched allocation ladder past the device's local memory
    and report whether the allocation ever fails.

``thrash``
    Hold a fixed working set and walk it repeatedly, reporting the achieved
    bandwidth. Run it once below and once above local memory to measure the
    cost of a WDDM spill.

``cache``
    Allocate, free every tensor, then flush the cache, reporting Torch's own
    accounting next to the complete-process WDDM usage at each step. This shows
    how much device memory a freed-but-cached block keeps away from VBAR pages.

Example, using the Portable interpreter:

    python_embeded\\python.exe -s tests\\repro_windows_xpu_malloc_pressure.py \\
        --mode budget --max-gib 40

WDDM numbers come from the ``GPU Process Memory`` performance counters, so the
process needs permission to read them; they are reported as -1 otherwise.
"""

import argparse
import json
import os
import subprocess
import time

import torch

MIB = 1024 ** 2
GIB = 1024 ** 3

_COUNTER_COMMAND = (
    "$p={pid};"
    "$l=(Get-Counter '\\GPU Process Memory(*)\\Local Usage')."
    "CounterSamples|Where-Object{{$_.InstanceName -like \"pid_$p*\"}}|"
    "Measure-Object CookedValue -Sum;"
    "$n=(Get-Counter '\\GPU Process Memory(*)\\Non Local Usage')."
    "CounterSamples|Where-Object{{$_.InstanceName -like \"pid_$p*\"}}|"
    "Measure-Object CookedValue -Sum;"
    "Write-Output \"$([math]::Round($l.Sum/1MB)),$([math]::Round($n.Sum/1MB))\""
)


def emit(**fields):
    fields["t"] = round(time.time(), 3)
    print(json.dumps(fields), flush=True)


def wddm_usage_mib():
    """Complete-process WDDM local and non-local usage for this process."""
    if os.name != "nt":
        return (-1, -1)
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             _COUNTER_COMMAND.format(pid=os.getpid())],
            capture_output=True, text=True, timeout=60,
        )
        local, non_local = completed.stdout.strip().splitlines()[-1].split(",")
        return (int(local), int(non_local))
    except Exception:
        return (-1, -1)


def torch_usage_mib(device):
    try:
        stats = torch.xpu.memory_stats(device)
    except Exception:
        return (-1, -1)
    return (
        int(stats.get("allocated_bytes.all.current", 0)) // MIB,
        int(stats.get("reserved_bytes.all.current", 0)) // MIB,
    )


def new_block(device, size_mib):
    return torch.empty(size_mib * MIB // 2, dtype=torch.float16, device=device)


def run_budget(args, device):
    """Does an over-budget sycl::malloc_device fail, or silently spill?"""
    sentinel_a = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    sentinel_b = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    torch.xpu.synchronize(args.device)

    blocks = []
    cumulative_mib = 0
    failures = 0
    max_mib = int(args.max_gib * 1024)

    while cumulative_mib < max_mib:
        status = "ok"
        error = ""
        touch_seconds = -1.0

        started = time.perf_counter()
        try:
            block = new_block(device, args.block_mib)
            alloc_seconds = time.perf_counter() - started
        except Exception as exception:
            alloc_seconds = time.perf_counter() - started
            status = "alloc_failed"
            error = f"{type(exception).__name__}: {exception}"
            block = None
            failures += 1

        if block is not None:
            touch_started = time.perf_counter()
            block.fill_(1.0)
            torch.xpu.synchronize(args.device)
            touch_seconds = time.perf_counter() - touch_started
            blocks.append(block)
            cumulative_mib += args.block_mib

        compute_started = time.perf_counter()
        product = sentinel_a @ sentinel_b
        torch.xpu.synchronize(args.device)
        compute_seconds = time.perf_counter() - compute_started
        del product

        allocated_mib, reserved_mib = torch_usage_mib(args.device)
        emit(
            event="step", status=status, cumulative_mib=cumulative_mib,
            alloc_s=round(alloc_seconds, 4),
            touch_s=round(touch_seconds, 4),
            sentinel_s=round(compute_seconds, 4),
            torch_allocated_mib=allocated_mib,
            torch_reserved_mib=reserved_mib, error=error,
        )
        if status != "ok":
            break

    local_mib, non_local_mib = wddm_usage_mib()
    emit(
        event="budget_result", requested_mib=cumulative_mib,
        allocation_failures=failures,
        wddm_local_mib=local_mib, wddm_nonlocal_mib=non_local_mib,
        verdict=("allocation never failed; WDDM absorbed the excess"
                 if failures == 0 else "allocation failed"),
    )


def run_thrash(args, device):
    """Cost of walking a working set that does or does not fit in local memory."""
    block_count = int(args.working_set_gib * 1024) // args.block_mib
    blocks = []
    for index in range(block_count):
        block = new_block(device, args.block_mib)
        block.fill_(float(index % 7) + 1.0)
        blocks.append(block)
    torch.xpu.synchronize(args.device)

    local_mib, non_local_mib = wddm_usage_mib()
    emit(
        event="filled", blocks=len(blocks),
        working_set_mib=len(blocks) * args.block_mib,
        wddm_local_mib=local_mib, wddm_nonlocal_mib=non_local_mib,
    )

    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        for block in blocks:
            block.add_(1.0)
        torch.xpu.synchronize(args.device)
        seconds = time.perf_counter() - started
        emit(
            event="iteration", iteration=iteration,
            seconds=round(seconds, 3),
            gib_per_second=round(
                (len(blocks) * args.block_mib / 1024.0) / seconds, 2),
        )


def run_cache(args, device):
    """How much device memory does a freed-but-cached Torch block hold?"""
    def report(phase):
        allocated_mib, reserved_mib = torch_usage_mib(args.device)
        local_mib, non_local_mib = wddm_usage_mib()
        emit(
            event="phase", phase=phase,
            torch_allocated_mib=allocated_mib,
            torch_reserved_mib=reserved_mib,
            wddm_local_mib=local_mib, wddm_nonlocal_mib=non_local_mib,
        )

    warm = torch.zeros(1024, device=device)
    torch.xpu.synchronize(args.device)
    del warm
    report("baseline")

    block_count = int(args.working_set_gib * 1024) // args.block_mib
    blocks = [new_block(device, args.block_mib) for _ in range(block_count)]
    for block in blocks:
        block.fill_(1.0)
    torch.xpu.synchronize(args.device)
    report("allocated")

    del block, blocks
    torch.xpu.synchronize(args.device)
    report("after_free_every_tensor")

    error = ""
    try:
        torch.xpu.empty_cache()
        torch.xpu.synchronize(args.device)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    emit(event="empty_cache", error=error)
    report("after_empty_cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("budget", "thrash", "cache"),
                        required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-mib", type=int, default=256)
    parser.add_argument("--max-gib", type=float, default=40.0,
                        help="budget mode: stop after this much is retained")
    parser.add_argument("--working-set-gib", type=float, default=20.0,
                        help="thrash/cache mode: retained working set")
    parser.add_argument("--iterations", type=int, default=3,
                        help="thrash mode: passes over the working set")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("requires an available Torch XPU device")

    device = torch.device("xpu", args.device)
    properties = torch.xpu.get_device_properties(args.device)
    emit(
        event="start", mode=args.mode, pid=os.getpid(), device=args.device,
        name=properties.name,
        total_memory_mib=int(properties.total_memory) // MIB,
        torch=torch.__version__,
    )

    {"budget": run_budget, "thrash": run_thrash, "cache": run_cache}[args.mode](
        args, device)
    emit(event="end", mode=args.mode)


if __name__ == "__main__":
    main()
