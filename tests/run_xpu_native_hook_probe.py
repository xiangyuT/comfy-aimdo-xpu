import argparse
import ctypes
import gc
import json
import threading

import torch

from comfy_aimdo import control


def _delta(after, before, key):
    return int(after[key]) - int(before[key])


def _allocate(size, value, device):
    tensor = torch.empty(size, dtype=torch.uint8, device=device)
    tensor.fill_(value)
    assert int(tensor[0].cpu()) == value
    torch.xpu.synchronize(device)
    return tensor


def _release(device, empty_cache=False):
    gc.collect()
    torch.xpu.synchronize(device)
    if empty_cache:
        torch.xpu.empty_cache()
        torch.xpu.synchronize(device)


def _initialize(device, headroom):
    torch.xpu.set_device(device)
    torch.xpu.empty_cache()
    allocator_before = torch.xpu.memory._get_current_allocator().allocator()
    assert control.init(
        implementation="xpu",
        simple_vram_headroom=headroom,
        xpu_allocator_mode="native_hook",
    )
    assert control.init_devices([device.index])
    allocator_after = torch.xpu.memory._get_current_allocator().allocator()
    assert allocator_before == allocator_after
    return allocator_before


def _lifecycle(device):
    _initialize(device, 64 << 20)
    before = control.get_xpu_ur_hook_stats()
    tensor = _allocate(64 << 20, 37, device)
    held = control.get_xpu_ur_hook_stats()
    usage_held = control.get_total_vram_usage()
    del tensor
    _release(device)
    cached = control.get_xpu_ur_hook_stats()
    usage_cached = control.get_total_vram_usage()

    errors = []

    def worker():
        try:
            torch.xpu.set_device(device)
            worker_tensor = _allocate(96 << 20, 39, device)
            del worker_tensor
            _release(device)
        except Exception as error:
            errors.append(repr(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if errors:
        raise RuntimeError(errors[0])
    after_worker = control.get_xpu_ur_hook_stats()

    torch.xpu.empty_cache()
    torch.xpu.synchronize(device)
    released = control.get_xpu_ur_hook_stats()
    usage_released = control.get_total_vram_usage()
    assert _delta(held, before, "tracked_alloc_calls") == 1
    assert cached["tracked_free_calls"] == held["tracked_free_calls"]
    assert _delta(after_worker, cached, "tracked_alloc_calls") == 1
    assert released["tracked_free_calls"] >= 2
    assert usage_held == usage_cached
    assert usage_released == 0
    return {
        "before": before,
        "held": held,
        "cached": cached,
        "after_worker": after_worker,
        "released": released,
        "usage_held": usage_held,
        "usage_cached": usage_cached,
        "usage_released": usage_released,
        "worker_intercepted": True,
    }


def _native_reclaim(device):
    total = int(torch.xpu.get_device_properties(device).total_memory)
    headroom = total - (448 << 20)
    _initialize(device, headroom)
    first = _allocate(256 << 20, 43, device)
    del first
    _release(device)
    before_second = control.get_xpu_ur_hook_stats()
    second = _allocate(320 << 20, 47, device)
    after_second = control.get_xpu_ur_hook_stats()
    usage_after_second = control.get_total_vram_usage()
    assert _delta(after_second, before_second, "synthetic_oom_calls") == 1
    assert _delta(
        after_second, before_second, "native_reclaim_free_calls"
    ) == 1
    assert _delta(after_second, before_second, "retry_eviction_calls") == 0
    del second
    _release(device, empty_cache=True)
    return {
        "total_bytes": total,
        "headroom_bytes": headroom,
        "before_second": before_second,
        "after_second": after_second,
        "usage_after_second": usage_after_second,
        "usage_after_release": control.get_total_vram_usage(),
    }


def _residual_eviction(device):
    total = int(torch.xpu.get_device_properties(device).total_memory)
    headroom = total - (128 << 20)
    _initialize(device, headroom)
    before = control.get_xpu_ur_hook_stats()
    tensor = _allocate(256 << 20, 41, device)
    after = control.get_xpu_ur_hook_stats()
    assert _delta(after, before, "synthetic_oom_calls") == 1
    assert _delta(after, before, "native_reclaim_free_calls") == 0
    assert _delta(after, before, "retry_eviction_calls") == 1
    assert _delta(after, before, "retry_eviction_bytes") == 128 << 20
    del tensor
    _release(device, empty_cache=True)
    return {
        "total_bytes": total,
        "headroom_bytes": headroom,
        "before": before,
        "after": after,
        "usage_after_release": control.get_total_vram_usage(),
    }


def _live_disable(device):
    _initialize(device, 64 << 20)
    tensor = _allocate(64 << 20, 51, device)
    rejected = False
    try:
        control.deinit()
    except RuntimeError as error:
        rejected = "tracked native segments remain live" in str(error)
    assert rejected
    assert control.lib is not None
    del tensor
    _release(device, empty_cache=True)
    return {
        "live_disable_rejected": True,
        "usage_after_release": control.get_total_vram_usage(),
    }


def _direct_sycl(device, library_path):
    if not library_path:
        raise ValueError("direct-sycl requires --direct-sycl-library")
    total = int(torch.xpu.get_device_properties(device).total_memory)
    headroom = total - (128 << 20)
    _initialize(device, headroom)
    helper = ctypes.CDLL(library_path)
    helper.aimdo_test_direct_sycl_alloc.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    helper.aimdo_test_direct_sycl_alloc.restype = ctypes.c_bool
    queue = torch.xpu.current_stream(device)
    before = control.get_xpu_ur_hook_stats()
    assert helper.aimdo_test_direct_sycl_alloc(
        int(queue.sycl_queue), 256 << 20
    )
    torch.xpu.synchronize(device)
    after = control.get_xpu_ur_hook_stats()
    assert _delta(after, before, "synthetic_oom_calls") == 0
    assert _delta(after, before, "direct_pressure_calls") == 1
    assert _delta(after, before, "direct_pressure_bytes") == 128 << 20
    assert _delta(after, before, "tracked_alloc_calls") == 1
    assert _delta(after, before, "tracked_free_calls") == 1
    assert control.get_total_vram_usage() == 0
    return {
        "total_bytes": total,
        "headroom_bytes": headroom,
        "before": before,
        "after": after,
        "usage_after_release": control.get_total_vram_usage(),
        "caller": "direct SYCL helper outside PyTorch",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario",
        choices=(
            "lifecycle",
            "native-reclaim",
            "residual-eviction",
            "live-disable",
            "direct-sycl",
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--direct-sycl-library")
    args = parser.parse_args()
    device = torch.device("xpu", args.device)
    scenarios = {
        "lifecycle": _lifecycle,
        "native-reclaim": _native_reclaim,
        "residual-eviction": _residual_eviction,
        "live-disable": _live_disable,
    }
    result = (
        _direct_sycl(device, args.direct_sycl_library)
        if args.scenario == "direct-sycl"
        else scenarios[args.scenario](device)
    )
    result.update(
        {
            "scenario": args.scenario,
            "torch_version": torch.__version__,
            "device": str(device),
            "allocator_mode": control.get_xpu_allocator_mode(),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    control.deinit()


if __name__ == "__main__":
    main()
