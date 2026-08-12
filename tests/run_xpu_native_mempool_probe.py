import argparse
import gc
import json
import threading

import torch

from comfy_aimdo import control


def _delta(after, before, key):
    return int(after[key]) - int(before[key])


def _allocate_and_release(size, value, device):
    tensor = torch.empty(size, dtype=torch.uint8, device=device)
    pointer = tensor.data_ptr()
    tensor.fill_(value)
    assert int(tensor[0].cpu()) == value
    del tensor
    return pointer


def _worker_default_pool_probe(device):
    errors = []

    def allocate():
        try:
            torch.xpu.set_device(device)
            _allocate_and_release(4 << 20, 29, device)
            torch.xpu.synchronize(device)
        except Exception as error:
            errors.append(repr(error))

    worker = threading.Thread(target=allocate)
    worker.start()
    worker.join()
    if errors:
        raise RuntimeError(errors[0])


def _active_pool_pressure_probe(device, fraction):
    total = int(torch.xpu.get_device_properties(device).total_memory)
    limit = int(total * fraction)
    first_size = limit * 3 // 5
    second_size = limit * 3 // 4
    if first_size <= 0 or second_size <= 0:
        raise RuntimeError("pressure fraction produced an empty allocation")

    torch.xpu.set_per_process_memory_fraction(fraction, device)
    out_of_memory = False
    error_text = None
    retained_inactive_bytes = 0
    try:
        with control.use_xpu_allocator_pool(device) as pool:
            first = torch.empty(first_size, dtype=torch.uint8, device=device)
            del first
            torch.xpu.synchronize(device)
            snapshot = pool.snapshot()
            retained_inactive_bytes = sum(
                int(segment["total_size"]) - int(segment["active_size"])
                for segment in snapshot
            )
            try:
                second = torch.empty(
                    second_size, dtype=torch.uint8, device=device
                )
                del second
                torch.xpu.synchronize(device)
            except RuntimeError as error:
                error_text = str(error)
                out_of_memory = "out of memory" in error_text.lower()
                if not out_of_memory:
                    raise
    finally:
        torch.xpu.set_per_process_memory_fraction(1.0, device)

    return {
        "fraction": fraction,
        "limit_bytes": limit,
        "first_size": first_size,
        "second_size": second_size,
        "retained_inactive_bytes": retained_inactive_bytes,
        "out_of_memory": out_of_memory,
        "active_pool_reclaim_supported": not out_of_memory,
        "error": error_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pressure-fraction", type=float, default=0.05)
    parser.add_argument("--skip-pressure", action="store_true")
    args = parser.parse_args()

    device = torch.device("xpu", args.device)
    torch.xpu.set_device(device)
    allocator_before = torch.xpu.memory._get_current_allocator().allocator()
    allocator_before_type = type(allocator_before).__name__

    assert control.init(
        implementation="xpu",
        simple_vram_headroom=64 << 20,
        xpu_allocator_mode="native_pool",
    )
    assert control.init_devices([args.device])
    assert control.get_xpu_allocator_mode() == "native_pool"
    allocator_after = torch.xpu.memory._get_current_allocator().allocator()
    allocator_after_type = type(allocator_after).__name__

    stats_before_default = control.get_xpu_vmm_stats()
    _allocate_and_release(4 << 20, 11, device)
    torch.xpu.synchronize(device)
    stats_after_default = control.get_xpu_vmm_stats()
    default_raw_allocs = _delta(
        stats_after_default,
        stats_before_default,
        "torch_allocator_physical_alloc_calls",
    )
    assert default_raw_allocs == 0

    with control.use_xpu_allocator_pool(device) as pool:
        stats_before_pool = control.get_xpu_vmm_stats()
        first_pointer = _allocate_and_release(256 << 10, 13, device)
        second_pointer = _allocate_and_release(256 << 10, 17, device)
        third_pointer = _allocate_and_release(256 << 10, 19, device)
        torch.xpu.synchronize(device)
        stats_after_pool = control.get_xpu_vmm_stats()
        snapshot = pool.snapshot()
        native_segment_bytes = sum(
            int(segment["total_size"]) for segment in snapshot
        )
        native_pool_raw_allocs = _delta(
            stats_after_pool,
            stats_before_pool,
            "torch_allocator_physical_alloc_calls",
        )
        assert native_pool_raw_allocs == 1
        assert native_segment_bytes == 2 << 20
        assert first_pointer == second_pointer == third_pointer

        stats_before_worker = control.get_xpu_vmm_stats()
        _worker_default_pool_probe(device)
        stats_after_worker = control.get_xpu_vmm_stats()
        worker_raw_allocs = _delta(
            stats_after_worker,
            stats_before_worker,
            "torch_allocator_physical_alloc_calls",
        )
        assert worker_raw_allocs == 0

    pressure = None
    if not args.skip_pressure:
        pressure = _active_pool_pressure_probe(
            device, args.pressure_fraction
        )

    pool = control.get_xpu_allocator_pool(device)
    assert pool.use_count() == 1
    assert control.release_xpu_allocator_pool(device)
    del pool
    gc.collect()
    torch.xpu.synchronize(device)
    stats_after_release = control.get_xpu_vmm_stats()

    result = {
        "torch_version": torch.__version__,
        "device": str(device),
        "allocator_identity_preserved": default_raw_allocs == 0,
        "allocator_type_before": allocator_before_type,
        "allocator_type_after": allocator_after_type,
        "default_pool_aimdo_raw_allocs": default_raw_allocs,
        "native_pool_aimdo_raw_allocs": native_pool_raw_allocs,
        "native_pool_segment_bytes": native_segment_bytes,
        "native_pool_pointer_reused": (
            first_pointer == second_pointer == third_pointer
        ),
        "worker_inherited_pool": worker_raw_allocs != 0,
        "pressure": pressure,
        "raw_release_calls": _delta(
            stats_after_release,
            stats_before_pool,
            "torch_allocator_physical_release_calls",
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    control.deinit()


if __name__ == "__main__":
    main()
