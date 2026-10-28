"""Native XPU writeback/retention integration; run with the matching built DSO.

Run once per allocator mode in a fresh process. Uses only temporary backing.
"""
import os
import ctypes
import tempfile

import torch
from comfy_aimdo import control

mode = os.environ.get("AIMDO_XPU_ALLOCATOR_MODE", "global")
assert control.init("xpu") and control.init_device(0)
from comfy_aimdo.model_vbar import ModelVBAR, vbar_signature_compare
from comfy_aimdo.torch import get_tensor_from_raw_ptr
from comfy_aimdo.host_buffer import write_device_to_file, read_file_to_device, cleanup_file_reader

PAGE = 32 * 1024 ** 2
device = torch.device("xpu", 0)
size = PAGE + 123
stream = torch.xpu.Stream()

# Cross-page, partial-last-page, nonzero file offset and non-default source queue.
with tempfile.TemporaryFile() as backing:
    backing.write(bytes([177]) * (size + 22))
    backing.flush()
    calls = []
    saved = False

    def pageout(offset, length):
        global saved
        calls.append((offset, length))
        if not saved:
            write_device_to_file(backing, 11, size, stream.sycl_queue, vbar.base_addr, 0)
            saved = True

    vbar = ModelVBAR(size, 0, pageout_callback=pageout)
    signature = vbar.fault(vbar.base_addr, size)
    assert signature is not None
    with torch.xpu.stream(stream):
        value = get_tensor_from_raw_ptr(vbar.base_addr, size, device)
        value[:PAGE].fill_(31)
        value[PAGE:].fill_(79)
    vbar.set_watermark(0)
    assert not calls
    del value
    vbar.unpin(vbar.base_addr, size, stream=stream)
    assert calls == [(0, PAGE), (PAGE, 123)], calls
    backing.seek(0)
    actual = backing.read()
    assert actual[:11] == actual[-11:] == bytes([177]) * 11
    assert actual[11:11 + PAGE] == bytes([31]) * PAGE
    assert actual[11 + PAGE:-11] == bytes([79]) * 123
    assert vbar.loaded_size() == 0

    vbar.set_watermark(size)
    restored = vbar.fault(vbar.base_addr, size)
    assert restored is not None and not vbar_signature_compare(signature, restored)
    read_file_to_device(backing, 11, size, stream.sycl_queue, vbar.base_addr, 0)
    value = get_tensor_from_raw_ptr(vbar.base_addr, size, device)
    assert bool((value[:PAGE] == 31).all() & (value[PAGE:] == 79).all())
    del value
    vbar.unpin(vbar.base_addr, size, stream=stream)
    calls.clear()
    vbar.close()
    assert not calls
print("XPU_PAGEOUT_FILE_PASS", mode, flush=True)

headroom = control.get_simple_vram_headroom()
for operation in ("watermark", "free_memory", "unpin", "fault"):
    failed = True

    def pageout(offset, length):
        if failed:
            raise OSError("injected writeback failure")

    vbar = ModelVBAR(2 * PAGE, 0, pageout_callback=pageout)
    assert vbar.fault(vbar.base_addr, 2 * PAGE) is not None
    value = get_tensor_from_raw_ptr(vbar.base_addr, 2 * PAGE, device)
    value.fill_(79)
    torch.xpu.synchronize()
    target = None
    if operation == "unpin":
        vbar.set_watermark(0)
    else:
        vbar.unpin(vbar.base_addr, 2 * PAGE)
    if operation == "fault":
        control.set_simple_vram_headroom(1 << 60)
        target = ModelVBAR(PAGE, 0)
        action = lambda: target.fault(target.base_addr, PAGE)
    else:
        action = {
            "watermark": lambda: vbar.set_watermark(0),
            "free_memory": lambda: vbar.free_memory(2 * PAGE),
            "unpin": lambda: vbar.unpin(vbar.base_addr, 2 * PAGE),
        }[operation]
    try:
        try:
            action()
        except RuntimeError:
            pass
        else:
            raise AssertionError(operation + ": writeback failure was lost")
    finally:
        control.set_simple_vram_headroom(headroom)
    if target is not None:
        target.close()
    assert vbar.get_residency() == [1, 1], (operation, vbar.get_residency())
    assert bool((value == 79).all())
    del value
    failed = False
    vbar.set_watermark(2 * PAGE)
    assert vbar.free_memory(2 * PAGE) == 2 * PAGE
    assert vbar.loaded_size() == 0
    vbar.close()
    print("XPU_PAGEOUT_FAILURE_RECOVERED", mode, operation, flush=True)

# An allocator callback cannot re-enter Python while Torch/UR owns its locks.
# Writable pages remain available for a later model-owner pageout boundary.
calls = []
vbar = ModelVBAR(PAGE, 0, pageout_callback=lambda *args: calls.append(args))
assert vbar.fault(vbar.base_addr, PAGE) is not None
vbar.unpin(vbar.base_addr, PAGE)
control.lib.aimdo_xpu_prepare_allocation.argtypes = [ctypes.c_int, ctypes.c_size_t]
control.lib.aimdo_xpu_prepare_allocation.restype = ctypes.c_bool
control.set_simple_vram_headroom(1 << 60)
try:
    assert control.lib.aimdo_xpu_prepare_allocation(0, PAGE)
finally:
    control.set_simple_vram_headroom(headroom)
assert calls == [] and vbar.get_residency() == [1]
assert vbar.free_memory(PAGE) == PAGE
assert calls == [(0, PAGE)]
vbar.close()
print("XPU_PAGEOUT_ALLOCATOR_BOUNDARY_PASS", mode, flush=True)

# Invalid backing fails through the writer instead of claiming successful eviction.
with tempfile.TemporaryFile() as backing:
    pointer = torch.ones(1024, device=device, dtype=torch.uint8)
    try:
        write_device_to_file(-1, 0, pointer.numel(), torch.xpu.current_stream().sycl_queue,
                             pointer.data_ptr(), 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid file handle unexpectedly succeeded")
    del pointer
cleanup_file_reader()
torch.xpu.synchronize()
control.deinit()
