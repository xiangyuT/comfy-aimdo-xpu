"""Real-XPU oracle for Windows VBAR consumer retirement.

Run this script in a fresh process for each mode because the native kill
switch is intentionally read once.  The workload stays well below 2 GiB and
uses only the requested XPU device.
"""

import argparse
import gc
import os
from pathlib import Path
import sys
import time

# A script launched by absolute/relative filename puts ``tests`` ahead of the
# repository root. Pin imports to this checkout so a Portable's older installed
# wheel cannot silently become the test subject.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch


MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("reference", "async"), required=True
    )
    parser.add_argument("--matrix-size", type=int, default=12288)
    parser.add_argument("--cycles", type=int, default=0)
    return parser.parse_args()


def delta(after, before, name):
    return int(after[name]) - int(before[name])


def pressure_pulse(vbar):
    # A zero-length public fault still executes the owner-side pressure scan,
    # but cannot map or pin a trigger page. The oversized test reserve makes
    # budget_deficit(0) select one 32 MiB source page for retirement.
    assert vbar.fault(vbar.base_addr, 0) is not None


def submit_consumers(tensor, device, matrix_size):
    stream_a = torch.xpu.Stream(device=device)
    stream_b = torch.xpu.Stream(device=device)
    matrix_a = torch.ones(
        (matrix_size, matrix_size), dtype=torch.bfloat16, device=device
    )
    matrix_b = torch.ones(
        (matrix_size, matrix_size), dtype=torch.bfloat16, device=device
    )
    # Exclude oneDNN primitive/kernel initialization from the retirement race.
    # Without a warm-up, first-use host work can outlast the device command and
    # make an asynchronous GEMM look complete before AIMDO can observe it.
    warmup = torch.mm(matrix_a, matrix_b)
    torch.xpu.synchronize(device.index)
    del warmup

    # Submit the short consumer first. Submitting it after the long queue-A
    # command can cause the current Torch/oneDNN path to serialize the streams,
    # which would erase the reverse-completion interval this test needs.
    with torch.xpu.stream(stream_b):
        result_b = tensor.sum(dtype=torch.int64)
    done_b = torch.xpu.Event()
    done_b.record(stream_b)

    with torch.xpu.stream(stream_a):
        # H3-relevant GEMM creates a single long queue-A command. The VBAR
        # read follows it, so removing the mapping before queue A completes is
        # both observable and independent of Python submission speed.
        matrix_result = torch.mm(matrix_a, matrix_b)
        result_a = tensor.sum(dtype=torch.int64)
    done_a = torch.xpu.Event()
    done_a.record(stream_a)

    return {
        "stream_a": stream_a,
        "stream_b": stream_b,
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "matrix_result": matrix_result,
        "result_a": result_a,
        "result_b": result_b,
        "done_a": done_a,
        "done_b": done_b,
    }


def run_async_refault_cycles(
    source, allocation, tensor, trigger, control, device, total, cycles
):
    stats_before = control.get_xpu_vmm_stats()
    for cycle in range(cycles):
        control.init(simple_vram_headroom=64 * MIB)
        source.prioritize()
        assert source.fault(allocation[1], allocation[2]) is not None
        value = cycle % 251 + 1
        tensor.fill_(value)
        result = tensor.sum(dtype=torch.int64)
        source.unpin(allocation[1], allocation[2])

        control.init(simple_vram_headroom=total * 4)
        unmap_before = control.get_xpu_vmm_stats()["unmap_calls"]
        pressure_pulse(trigger)
        assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before
        torch.xpu.synchronize(device.index)
        assert int(result.cpu()) == value * 32 * MIB
        pressure_pulse(trigger)
        assert (
            control.get_xpu_vmm_stats()["unmap_calls"]
            == unmap_before + 1
        )
        assert source.get_residency() == [0]

    stats_after = control.get_xpu_vmm_stats()
    for name, expected in (
        ("map_calls", cycles),
        ("unmap_calls", cycles),
        ("retire_token_calls", cycles),
        ("retire_fence_submit_calls", cycles),
        ("retire_fence_complete_calls", cycles),
    ):
        assert delta(stats_after, stats_before, name) == expected
    for name in (
        "retire_fence_submit_failures",
        "retire_queue_registration_failures",
        "retire_queue_identity_mismatches",
        "retire_fence_query_failures",
        "retire_shutdown_wait_failures",
    ):
        assert delta(stats_after, stats_before, name) == 0
    summary = {
        name: delta(stats_after, stats_before, name)
        for name in (
            "map_calls",
            "unmap_calls",
            "retire_token_calls",
            "retire_fence_submit_calls",
            "retire_fence_complete_calls",
            "retire_fence_submit_failures",
            "retire_fence_query_failures",
        )
    }
    print(
        f"REPEAT_PASS mode=async device={device.index} cycles={cycles} "
        f"stats_delta={summary}"
    )


def main():
    args = parse_args()
    if sys.platform != "win32":
        raise SystemExit("this oracle targets Windows XPU")
    if args.mode == "reference":
        os.environ.pop("AIMDO_XPU_ASYNC_VBAR_RECLAIM", None)
    else:
        os.environ["AIMDO_XPU_ASYNC_VBAR_RECLAIM"] = "1"

    if not torch.xpu.is_available() or args.device >= torch.xpu.device_count():
        raise SystemExit(f"XPU device {args.device} is unavailable")

    torch.xpu.set_device(args.device)
    device = torch.device("xpu", args.device)

    # Import after selecting the process mode and device. Torch itself must be
    # imported first so the Windows UR hook can attach to the loaded runtime.
    from comfy_aimdo import control

    source = None
    trigger = None
    allocation = None
    tensor = None
    consumers = None
    initialized = False
    try:
        assert control.init(
            implementation="xpu",
            simple_vram_headroom=64 * MIB,
            xpu_allocator_mode="native_hook",
        )
        assert control.init_devices([args.device])
        initialized = True
        control.set_log_info()
        torch.xpu.empty_cache()

        # model_vbar binds native function pointers at import time, so load it
        # only after control.init() has installed the current checkout's DLL.
        from comfy_aimdo.model_vbar import ModelVBAR
        from comfy_aimdo.torch import aimdo_to_tensor

        source = ModelVBAR(32 * MIB, args.device)
        allocation = source.alloc(32 * MIB)
        assert source.fault(allocation[1], allocation[2]) is not None
        tensor = aimdo_to_tensor(allocation, device)
        tensor.fill_(37)
        torch.xpu.synchronize(args.device)

        stats_before = control.get_xpu_vmm_stats()
        consumers = submit_consumers(tensor, device, args.matrix_size)

        # Queue B is an additional consumer. Queue A performs the final unpin.
        assert source.register_consumer(
            allocation[1], allocation[2], consumers["stream_b"]
        )
        source.unpin(
            allocation[1], allocation[2], consumers["stream_a"]
        )
        if consumers["done_a"].query():
            raise RuntimeError(
                "queue A completed before the retirement oracle; increase "
                "--matrix-size without exceeding the 2 GiB test ceiling"
            )

        trigger = ModelVBAR(32 * MIB, args.device)
        total = int(torch.xpu.get_device_properties(device).total_memory)
        control.init(simple_vram_headroom=total * 4)

        if args.mode == "reference":
            start = time.perf_counter()
            trigger.prioritize()
            elapsed = time.perf_counter() - start
            assert consumers["done_a"].query()
            assert consumers["done_b"].query()
        else:
            # A pressure fault closes partial retirement batches but must not
            # wait or unmap while either queue still owns the page.
            unmap_before = control.get_xpu_vmm_stats()["unmap_calls"]
            pressure_pulse(trigger)
            after_first = control.get_xpu_vmm_stats()
            assert after_first["unmap_calls"] == unmap_before

            consumers["stream_b"].synchronize()
            assert consumers["done_b"].query()
            if consumers["done_a"].query():
                raise RuntimeError(
                    "queue A completed before the reverse-completion oracle; "
                    "increase --iterations"
                )
            pressure_pulse(trigger)
            after_b = control.get_xpu_vmm_stats()
            assert after_b["unmap_calls"] == unmap_before

            consumers["stream_a"].synchronize()
            assert consumers["done_a"].query()
            pressure_pulse(trigger)
            elapsed = 0.0

        expected = 37 * 32 * MIB
        assert int(consumers["result_a"].cpu()) == expected
        assert int(consumers["result_b"].cpu()) == expected
        assert source.get_residency() == [0]

        stats_after = control.get_xpu_vmm_stats()
        assert delta(stats_after, stats_before, "unmap_calls") == 1
        assert delta(stats_after, stats_before, "unmap_bytes") == 32 * MIB
        assert delta(
            stats_after, stats_before, "retire_queue_registration_failures"
        ) == 0
        assert delta(
            stats_after, stats_before, "retire_queue_identity_mismatches"
        ) == 0
        assert delta(
            stats_after, stats_before, "retire_fence_query_failures"
        ) == 0
        if args.mode == "reference":
            assert delta(stats_after, stats_before, "retire_token_calls") == 0
            assert delta(
                stats_after, stats_before, "retire_fence_submit_calls"
            ) == 0
            assert delta(
                stats_after, stats_before, "retire_fence_complete_calls"
            ) == 0
            assert delta(stats_after, stats_before, "context_sync_calls") >= 1
        else:
            assert delta(stats_after, stats_before, "retire_token_calls") == 2
            assert delta(
                stats_after, stats_before, "retire_fence_submit_calls"
            ) == 2
            assert delta(
                stats_after, stats_before, "retire_fence_complete_calls"
            ) == 2

        print(
            "PASS "
            f"mode={args.mode} device={args.device} "
            f"matrix_size={args.matrix_size} reference_wait_s={elapsed:.3f} "
            f"stats={stats_after}"
        )
        if args.mode == "async" and args.cycles:
            consumers.clear()
            gc.collect()
            torch.xpu.empty_cache()
            run_async_refault_cycles(
                source,
                allocation,
                tensor,
                trigger,
                control,
                device,
                total,
                args.cycles,
            )
    finally:
        control.init(simple_vram_headroom=64 * MIB) if initialized else None
        if consumers is not None:
            consumers.clear()
        tensor = None
        allocation = None
        gc.collect()
        if trigger is not None:
            trigger.__del__()
            trigger = None
        if source is not None:
            source.__del__()
            source = None
        gc.collect()
        if initialized:
            torch.xpu.synchronize(args.device)
            control.deinit()


if __name__ == "__main__":
    main()
