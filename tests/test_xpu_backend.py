import gc
import sys
import tempfile

import pytest

torch = pytest.importorskip("torch")

from comfy_aimdo import control


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="requires an available Torch XPU device",
)


@pytest.fixture(scope="module", autouse=True)
def initialized_xpu_backend():
    assert control.detect_vendor() == "xpu"
    assert control.init(implementation="xpu", simple_vram_headroom=64 << 20)
    assert control.init_devices([torch.xpu.current_device()])
    yield
    torch.xpu.synchronize()
    control.deinit()


def _destroy_vbar(vbar):
    vbar.__del__()
    del vbar
    gc.collect()


def test_vbar_raw_tensor_hit_evict_and_refault_signature():
    from comfy_aimdo.model_vbar import ModelVBAR, vbar_signature_compare
    from comfy_aimdo.torch import aimdo_to_tensor

    device = torch.device("xpu", torch.xpu.current_device())
    stats_before = control.get_xpu_vmm_stats()
    vbar = ModelVBAR(96 << 20, device.index)
    allocation = vbar.alloc(8 << 20)

    first = vbar.fault(allocation[1], allocation[2])
    assert first is not None
    tensor = aimdo_to_tensor(allocation, device)
    tensor.fill_(37)
    torch.xpu.synchronize()
    assert tensor[:4096].cpu().eq(37).all()

    vbar.unpin(allocation[1], allocation[2])
    hit = vbar.fault(allocation[1], allocation[2])
    assert vbar_signature_compare(first, hit)
    assert tensor[:4096].cpu().eq(37).all()
    vbar.unpin(allocation[1], allocation[2])

    assert vbar.free_memory(32 << 20) == 32 << 20
    assert vbar.get_residency() == [0, 0, 0]
    stats_after_evict = control.get_xpu_vmm_stats()
    assert stats_after_evict["map_bytes"] - stats_before["map_bytes"] >= 32 << 20
    assert stats_after_evict["unmap_bytes"] - stats_before["unmap_bytes"] >= 32 << 20
    vbar.prioritize()
    refault = vbar.fault(allocation[1], allocation[2])
    assert refault is not None
    assert not vbar_signature_compare(first, refault)

    del tensor, allocation
    vbar.unpin(vbar.base_addr, 8 << 20)
    _destroy_vbar(vbar)


@pytest.mark.parametrize(
    "device_arg",
    ["xpu", torch.device("xpu")],
    ids=["string", "torch-device"],
)
def test_aimdo_to_tensor_resolves_indexless_xpu_to_current_device(
    device_arg,
):
    from comfy_aimdo.model_vbar import ModelVBAR
    from comfy_aimdo.torch import aimdo_to_tensor

    current_device = torch.xpu.current_device()
    concrete_device = torch.device("xpu", current_device)
    vbar = ModelVBAR(32 << 20, current_device)
    allocation = vbar.alloc(8 << 20)
    assert vbar.fault(allocation[1], allocation[2]) is not None

    tensor = aimdo_to_tensor(allocation, device_arg)
    assert tensor.device == concrete_device
    assert tensor.data_ptr() == allocation[1]
    tensor.fill_(53)
    torch.xpu.synchronize()
    assert tensor[:4096].cpu().eq(53).all()

    vbar.unpin(allocation[1], allocation[2])
    del tensor, allocation
    _destroy_vbar(vbar)


def test_unmap_waits_for_inflight_xpu_work():
    from comfy_aimdo.model_vbar import ModelVBAR
    from comfy_aimdo.torch import aimdo_to_tensor

    device = torch.device("xpu", torch.xpu.current_device())
    vbar = ModelVBAR(32 << 20, device.index)
    allocation = vbar.alloc(32 << 20)
    assert vbar.fault(allocation[1], allocation[2]) is not None
    tensor = aimdo_to_tensor(allocation, device)

    tensor.fill_(19)
    vbar.unpin(allocation[1], allocation[2])
    assert vbar.free_memory(32 << 20) == 32 << 20

    del tensor, allocation
    _destroy_vbar(vbar)
    probe = torch.arange(4096, device=device)
    assert probe.sum().cpu().item() == 4095 * 4096 // 2


def test_worker_stream_file_fill_rebinds_unmap_queue():
    from comfy_aimdo.host_buffer import cleanup_file_reader, read_file_to_device
    from comfy_aimdo.model_vbar import ModelVBAR, vbar_signature_compare
    from comfy_aimdo.torch import aimdo_to_tensor

    device = torch.device("xpu", torch.xpu.current_device())
    worker = torch.xpu.Stream(device=device)
    stats_before = control.get_xpu_vmm_stats()
    vbar = ModelVBAR(32 << 20, device.index)
    allocation = vbar.alloc(32 << 20)
    first = vbar.fault(allocation[1], allocation[2])
    tensor = aimdo_to_tensor(allocation, device)
    payload = bytes([7]) * (16 << 20)

    cleanup_file_reader()
    with tempfile.TemporaryFile() as handle:
        handle.write(payload)
        handle.flush()
        with torch.xpu.stream(worker):
            # Ten 16 MiB reads rotate and reuse the three 64 MiB file-reader
            # slots, covering the completion-token path as well as queue
            # rebinding across a worker stream.
            for _ in range(10):
                read_file_to_device(
                    handle, 0, len(payload), int(worker.sycl_queue),
                    tensor.data_ptr(), device.index, mark_cold=False)
            tensor[len(payload):].fill_(13)

    # No explicit worker synchronization: VBAR eviction must wait the queue
    # that submitted the file copy and tensor write, not the init thread queue.
    vbar.unpin(allocation[1], allocation[2])
    assert vbar.free_memory(32 << 20) == 32 << 20
    stats_after_evict = control.get_xpu_vmm_stats()
    assert stats_after_evict["queue_rebind_calls"] > stats_before["queue_rebind_calls"]
    assert (
        stats_after_evict["synchronous_host_to_device_calls"]
        - stats_before["synchronous_host_to_device_calls"]
        >= 10
    )
    assert (
        stats_after_evict["synchronous_host_to_device_calls"]
        == stats_after_evict["synchronous_host_to_device_completions"]
    )
    assert (
        stats_after_evict["context_sync_calls"]
        == stats_after_evict["context_sync_completions"]
    )
    assert stats_after_evict["event_sync_calls"] > stats_before["event_sync_calls"]
    assert (
        stats_after_evict["event_sync_calls"]
        == stats_after_evict["event_sync_completions"]
    )

    vbar.prioritize()
    refault = vbar.fault(allocation[1], allocation[2])
    assert refault is not None
    assert not vbar_signature_compare(first, refault)
    vbar.unpin(allocation[1], allocation[2])
    del tensor, allocation
    _destroy_vbar(vbar)


def test_torch_allocator_pressure_evicts_unpinned_vbar_pages():
    from comfy_aimdo.model_vbar import ModelVBAR

    device = torch.device("xpu", torch.xpu.current_device())
    assert control.empty_xpu_allocator_cache(wait=True)
    total = torch.xpu.get_device_properties(device).total_memory

    vbar = ModelVBAR(64 << 20, device.index)
    allocation = vbar.alloc(64 << 20)
    assert vbar.fault(allocation[1], allocation[2]) is not None
    vbar.unpin(allocation[1], allocation[2])
    assert vbar.get_residency() == [1, 1]
    recorded = control.get_total_vram_usage()
    control.init(simple_vram_headroom=total - recorded - (32 << 20))
    stats_before = control.get_xpu_vmm_stats()

    activation = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    activation.fill_(23)
    torch.xpu.synchronize()
    assert activation[:4096].cpu().eq(23).all()
    assert vbar.get_residency() == [1, 0]
    stats_after = control.get_xpu_vmm_stats()
    assert stats_after["unmap_bytes"] - stats_before["unmap_bytes"] == 32 << 20
    assert (
        stats_after["torch_allocator_physical_alloc_calls"]
        > stats_before["torch_allocator_physical_alloc_calls"]
    )

    del activation, allocation
    torch.xpu.empty_cache()
    control.init(simple_vram_headroom=64 << 20)
    _destroy_vbar(vbar)


def test_torch_allocator_reuses_completed_same_size_block():
    device = torch.device("xpu", torch.xpu.current_device())
    assert control.empty_xpu_allocator_cache(wait=True)
    stats_before = control.get_xpu_vmm_stats()
    active_before, reserved_before, _, _ = (
        control.get_xpu_allocator_memory_stats(device)
    )

    for index, value in enumerate((3, 5, 7)):
        tensor = torch.empty(4 << 20, dtype=torch.uint8, device=device)
        tensor.fill_(value)
        assert int(tensor[1024].cpu()) == value
        if index == 0:
            active, reserved, peak_active, peak_reserved = (
                control.get_xpu_allocator_memory_stats(device)
            )
            assert active - active_before == 4 << 20
            assert reserved - reserved_before == 4 << 20
            assert peak_active >= active
            assert peak_reserved >= reserved
        del tensor
    torch.xpu.synchronize()

    active_cached, reserved_cached, _, _ = (
        control.get_xpu_allocator_memory_stats(device)
    )
    assert active_cached == active_before
    assert reserved_cached - reserved_before == 4 << 20

    stats_after = control.get_xpu_vmm_stats()
    assert (
        stats_after["torch_allocator_physical_alloc_calls"]
        - stats_before["torch_allocator_physical_alloc_calls"]
        == 1
    )
    assert (
        stats_after["torch_allocator_cache_hits"]
        - stats_before["torch_allocator_cache_hits"]
        == 2
    )
    assert control.empty_xpu_allocator_cache(wait=True)
    active_empty, reserved_empty, _, _ = (
        control.get_xpu_allocator_memory_stats(device)
    )
    assert active_empty == active_before
    assert reserved_empty == reserved_before


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows keeps completed USM blocks out of the allocation hot path",
)
def test_windows_torch_allocator_defers_different_size_usm_release():
    device = torch.device("xpu", torch.xpu.current_device())
    assert control.empty_xpu_allocator_cache(wait=True)
    stats_before = control.get_xpu_vmm_stats()

    first = torch.empty(3 << 20, dtype=torch.uint8, device=device)
    first.fill_(11)
    del first
    torch.xpu.synchronize()

    second = None
    try:
        # A cache miss of a different size must not synchronously call
        # sycl::free from the allocator callback on Windows. The Level Zero
        # USM release can wait indefinitely under high residency while the
        # prompt worker holds the GIL and allocator mutex.
        second = torch.empty(5 << 20, dtype=torch.uint8, device=device)
        second.fill_(13)
        torch.xpu.synchronize()
        stats_after = control.get_xpu_vmm_stats()
        assert (
            stats_after["torch_allocator_physical_release_calls"]
            == stats_before["torch_allocator_physical_release_calls"]
        )
    finally:
        del second
        torch.xpu.synchronize()
        assert control.empty_xpu_allocator_cache(wait=True)


def test_priority_evicts_the_lowest_vbar_first():
    from comfy_aimdo.model_vbar import ModelVBAR

    device = torch.device("xpu", torch.xpu.current_device())
    assert control.empty_xpu_allocator_cache(wait=True)
    total = torch.xpu.get_device_properties(device).total_memory
    control.init(simple_vram_headroom=total - (128 << 20))

    low = ModelVBAR(32 << 20, device.index)
    middle = ModelVBAR(32 << 20, device.index)
    high = ModelVBAR(32 << 20, device.index)
    low_allocation = low.alloc(8 << 20)
    middle_allocation = middle.alloc(8 << 20)
    high_allocation = high.alloc(8 << 20)
    assert low.fault(low_allocation[1], low_allocation[2]) is not None
    low.unpin(low_allocation[1], low_allocation[2])
    assert middle.fault(middle_allocation[1], middle_allocation[2]) is not None
    middle.unpin(middle_allocation[1], middle_allocation[2])
    assert high.fault(high_allocation[1], high_allocation[2]) is not None
    high.unpin(high_allocation[1], high_allocation[2])

    newest = ModelVBAR(32 << 20, device.index)
    newest_allocation = newest.alloc(8 << 20)
    assert newest.fault(newest_allocation[1], newest_allocation[2]) is not None
    newest.unpin(newest_allocation[1], newest_allocation[2])

    assert low.get_residency() == [0]
    assert middle.get_residency() == [1]
    assert high.get_residency() == [1]
    assert newest.get_residency() == [1]

    control.init(simple_vram_headroom=64 << 20)
    del low_allocation, middle_allocation, high_allocation, newest_allocation
    _destroy_vbar(newest)
    _destroy_vbar(high)
    _destroy_vbar(middle)
    _destroy_vbar(low)


def test_vrambuffer_and_file_to_device_paths():
    from comfy_aimdo.host_buffer import (
        HostBuffer,
        cleanup_file_reader,
        read_file_to_device,
    )
    from comfy_aimdo.torch import aimdo_to_tensor, hostbuf_to_tensor
    from comfy_aimdo.vram_buffer import VRAMBuffer

    device = torch.device("xpu", torch.xpu.current_device())
    payload = bytes((index * 17 + 3) & 255 for index in range(1 << 20))
    with tempfile.TemporaryFile() as handle:
        handle.write(payload)
        handle.flush()

        direct = torch.empty(len(payload), dtype=torch.uint8, device=device)
        read_file_to_device(
            handle, 0, len(payload), 0, direct.data_ptr(), device.index,
            mark_cold=False)
        torch.xpu.synchronize()
        assert bytes(direct.cpu().numpy()) == payload

        host = HostBuffer(0, 0, 2 << 20, mark_cold=False)
        host.extend(len(payload), register=True)
        host.read_file_slice(handle, 0, len(payload), offset=0)
        assert bytes(hostbuf_to_tensor(host).numpy()) == payload

        staged = torch.empty(len(payload), dtype=torch.uint8, device=device)
        host.read_file_slice(
            handle, 0, len(payload), offset=0,
            device_ptr=staged.data_ptr(), device=device.index)
        torch.xpu.synchronize()
        assert bytes(staged.cpu().numpy()) == payload
        del direct, staged, host

    cleanup_file_reader()

    buffer = VRAMBuffer(64 << 20, device.index)
    allocation = buffer.get(4 << 20)
    tensor = aimdo_to_tensor(allocation, device)
    tensor.fill_(91)
    torch.xpu.synchronize()
    assert tensor[:4096].cpu().eq(91).all()
    del tensor, allocation
    buffer.__del__()
    del buffer
    gc.collect()
