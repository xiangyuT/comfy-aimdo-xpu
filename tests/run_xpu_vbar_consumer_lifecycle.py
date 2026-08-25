"""Real-XPU oracle for explicit and captured VBAR consumers.

The test stays below 2 GiB.  It proves that an external-consumer lease and a
capture-lifetime lease both prevent physical unmap, then become reclaimable
only after the registered final queue has completed.
"""

import argparse
import gc
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch


MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--matrix-size", type=int, default=12288)
    parser.add_argument("--kernel-repeats", type=int, default=8)
    return parser.parse_args()


def pressure_pulse(vbar):
    assert vbar.fault(vbar.base_addr, 0) is not None


def require_pending(event, phase):
    if event.query():
        raise RuntimeError(
            f"long kernel completed before {phase}; increase --matrix-size "
            "without exceeding the 2 GiB test ceiling"
        )


def page(source):
    return source.snapshot()["pages"][0]


def prepare_page(source, allocation, tensor, value, device):
    source.prioritize()
    assert source.fault(allocation[1], allocation[2]) is not None
    tensor.fill_(value)
    torch.xpu.synchronize(device.index)


def submit_long_gemm(matrix_a, matrix_b, matrix_result, repeats):
    for _ in range(repeats):
        torch.mm(matrix_a, matrix_b, out=matrix_result)


def run_external(
    source, allocation, tensor, trigger, matrix_a, matrix_b, control, device,
    repeats,
):
    from comfy_aimdo.model_vbar import vbar_external_consumer

    prepare_page(source, allocation, tensor, 41, device)
    stream = torch.xpu.Stream(device=device)
    matrix_result = torch.empty_like(matrix_a)
    unmap_before = control.get_xpu_vmm_stats()["unmap_calls"]

    with vbar_external_consumer(allocation, stream=stream):
        with torch.xpu.stream(stream):
            submit_long_gemm(
                matrix_a, matrix_b, matrix_result, repeats
            )
            result = tensor.sum(dtype=torch.int64)
        done = torch.xpu.Event()
        done.record(stream)
        source.unpin(allocation[1], allocation[2], stream)
        assert page(source)["external_consumer_holds"] == 1
        pressure_pulse(trigger)
        assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before

    after_release = page(source)
    assert after_release["external_consumer_holds"] == 0
    assert not after_release["retire_unknown"]
    require_pending(done, "external-consumer completion check")
    pressure_pulse(trigger)
    assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before

    stream.synchronize()
    assert int(result.cpu()) == 41 * 32 * MIB
    pressure_pulse(trigger)
    assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before + 1
    assert source.get_residency() == [0]
    del matrix_result, result, done, stream
    print("EXTERNAL_CONSUMER_PASS", flush=True)


def run_capture(
    source, allocation, tensor, trigger, matrix_a, matrix_b, control, device,
    repeats,
):
    from comfy_aimdo.model_vbar import vbar_capture_begin

    prepare_page(source, allocation, tensor, 43, device)
    stream = torch.xpu.Stream(device=device)
    matrix_result = torch.empty_like(matrix_a)
    with torch.xpu.stream(stream):
        torch.mm(matrix_a, matrix_b, out=matrix_result)
        warm_sum = tensor.sum(dtype=torch.int64)
    stream.synchronize()
    del warm_sum

    capture = vbar_capture_begin(allocation)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        submit_long_gemm(matrix_a, matrix_b, matrix_result, repeats)
        result = tensor.sum(dtype=torch.int64)
        # During capture no event can prove the lifetime of future replays.
        # The explicit capture hold, rather than an irreversible unknown bit,
        # owns the page after the model pin is dropped.
        source.unpin(allocation[1], allocation[2], stream)

    captured = page(source)
    assert captured["capture_holds"] == 1
    assert not captured["retire_unknown"]
    unmap_before = control.get_xpu_vmm_stats()["unmap_calls"]
    pressure_pulse(trigger)
    assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before

    with torch.xpu.stream(stream):
        graph.replay()
    done = torch.xpu.Event()
    done.record(stream)
    # Contract boundary: no future replay is permitted after this point.  The
    # final replay is already ordered on stream, so release can publish its
    # completion dependency without synchronizing.
    capture.release(stream)
    released = page(source)
    assert released["capture_holds"] == 0
    assert not released["retire_unknown"]
    require_pending(done, "capture completion check")
    pressure_pulse(trigger)
    assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before

    stream.synchronize()
    assert int(result.cpu()) == 43 * 32 * MIB
    graph = None
    gc.collect()
    pressure_pulse(trigger)
    assert control.get_xpu_vmm_stats()["unmap_calls"] == unmap_before + 1
    assert source.get_residency() == [0]
    del matrix_result, result, done, stream
    print("CAPTURE_LIFECYCLE_PASS", flush=True)


def run_oom_snapshot(source, allocation, control, device, total):
    # Force a deterministic owner-boundary host-offload decision without
    # asking the driver to overcommit.  Native Torch remains the allocator of
    # activation blocks; the snapshot reports that domain beside the rejected
    # VBAR request instead of pretending the VBAR is a native block.
    source.set_watermark(0)
    control.init(simple_vram_headroom=total * 4)
    assert source.fault(allocation[1], allocation[2]) is None
    snapshot = control.get_last_xpu_oom_snapshot(device.index)
    assert snapshot["allocator_owner"] == "torch_xpu_native"
    assert snapshot["oom"]["stage"] == "vbar_fault_host_offload"
    assert snapshot["oom"]["request_bytes"] == 32 * MIB
    assert "stats" in snapshot["native_allocator"]
    assert "vmm" in snapshot["aimdo"]
    assert "ur_hook" in snapshot["aimdo"]
    assert "vbars" in snapshot["aimdo"]
    assert source.get_residency() == [0]
    print("NATIVE_OWNERSHIP_OOM_SNAPSHOT_PASS", flush=True)


def main():
    args = parse_args()
    if sys.platform != "win32":
        raise SystemExit("this oracle targets Windows XPU")
    os.environ.pop("UR_L0_SERIALIZE", None)
    os.environ["AIMDO_XPU_ASYNC_VBAR_RECLAIM"] = "1"
    if not torch.xpu.is_available() or args.device >= torch.xpu.device_count():
        raise SystemExit(f"XPU device {args.device} is unavailable")

    torch.xpu.set_device(args.device)
    device = torch.device("xpu", args.device)
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
        assert control.init_devices([args.device])
        initialized = True
        from comfy_aimdo.model_vbar import ModelVBAR
        from comfy_aimdo.torch import aimdo_to_tensor

        source = ModelVBAR(32 * MIB, args.device)
        allocation = source.alloc(32 * MIB)
        assert source.fault(allocation[1], allocation[2]) is not None
        tensor = aimdo_to_tensor(allocation, device)
        source.unpin(allocation[1], allocation[2])
        torch.xpu.synchronize(args.device)

        trigger = ModelVBAR(32 * MIB, args.device)
        total = int(torch.xpu.get_device_properties(device).total_memory)
        control.init(simple_vram_headroom=total * 4)

        matrix_a = torch.ones(
            (args.matrix_size, args.matrix_size),
            dtype=torch.bfloat16,
            device=device,
        )
        matrix_b = torch.ones_like(matrix_a)
        warmup = torch.mm(matrix_a, matrix_b)
        torch.xpu.synchronize(args.device)
        del warmup

        run_external(
            source, allocation, tensor, trigger, matrix_a, matrix_b,
            control, device, args.kernel_repeats,
        )
        run_capture(
            source, allocation, tensor, trigger, matrix_a, matrix_b,
            control, device, args.kernel_repeats,
        )
        run_oom_snapshot(source, allocation, control, device, total)
        stats = control.get_xpu_vmm_stats()
        for name in (
            "retire_fence_submit_failures",
            "retire_queue_registration_failures",
            "retire_queue_identity_mismatches",
            "retire_fence_query_failures",
            "retire_shutdown_wait_failures",
        ):
            assert stats[name] == 0
        snapshot = control.get_xpu_memory_snapshot(args.device)
        assert snapshot["allocator_owner"] == "torch_xpu_native"
        print(
            f"PASS device={args.device} matrix_size={args.matrix_size} "
            f"kernel_repeats={args.kernel_repeats} "
            f"stats={stats}",
            flush=True,
        )
    finally:
        if initialized:
            control.init(simple_vram_headroom=64 * MIB)
        gc.collect()
        if trigger is not None:
            trigger.__del__()
        if source is not None:
            source.__del__()
        gc.collect()
        if initialized:
            torch.xpu.synchronize(args.device)
            control.deinit()


if __name__ == "__main__":
    main()
