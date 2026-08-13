import torch
import ctypes
import platform

import logging

from . import control

_SMALL_VBAR_COPY_MAXIMUM = 2 * 1024 * 1024

def get_tensor_from_raw_ptr(ptr, size, device):
    device = torch.device(device)
    if device.type == "xpu":
        if device.index is None:
            device = torch.device("xpu", torch.xpu.current_device())
        # XPU virtual addresses may occupy the unsigned half of the address
        # space, while the private storage constructor accepts a signed
        # int64. Preserve the pointer bits across that boundary.
        signed_ptr = int(ptr)
        if signed_ptr >= 1 << 63:
            signed_ptr -= 1 << 64
        storage = torch._C._construct_storage_from_data_pointer(
            signed_ptr, device, int(size))
        return torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (int(size),), (1,))

    container = {
        "shape": (size,),
        "typestr": "|u1",
        "data": (ptr, False), #writable
        "version": 3,
    }

    class Holder:
        pass

    holder = Holder()
    holder.__cuda_array_interface__ = container

    return torch.as_tensor(holder, device=device)

def aimdo_to_tensor(alloc, device):
    _, ptr, size = alloc
    return get_tensor_from_raw_ptr(ptr, size, device)

def copy_to_vbar(destination, source, non_blocking=False):
    """Copy a tensor, avoiding the broken small VBAR path on Windows XPU."""
    size = destination.numel() * destination.element_size()
    if (
        control.implementation == "xpu"
        and platform.system() == "Windows"
        and control.lib is not None
        and destination.device.type == "xpu"
        and source.device.type == "cpu"
        and 0 < size <= _SMALL_VBAR_COPY_MAXIMUM
        and destination.dtype == source.dtype
        and destination.shape == source.shape
        and destination.is_contiguous()
        and source.is_contiguous()
    ):
        device = destination.device.index
        if device is None:
            device = torch.xpu.current_device()

        if (
            control.lib.aimdo_xpu_is_mapped_pinned_vbar(
                destination.data_ptr(), size
            )
            and control.lib.aimdo_xpu_needs_small_vbar_copy_workaround(device)
        ):
            if not control.lib.aimdo_xpu_copy_host_to_vbar(
                destination.data_ptr(), source.data_ptr(), size, device
            ):
                raise RuntimeError(
                    f"AIMDO failed to copy {size} bytes into XPU VBAR memory"
                )
            return destination

    return destination.copy_(source, non_blocking=non_blocking)

def hostbuf_to_tensor(hostbuf):
    byte_view = (ctypes.c_uint8 * hostbuf.size).from_address(hostbuf.get_raw_address())
    return torch.frombuffer(byte_view, dtype=torch.uint8)

#pytorch doesnt have an API for a CUDAPluggableAllocator from an already loaded
#library. Rather than force a second load that pytorch owns, construct these
#pytorch internals outselves as sperate CDLL loads is far too risky.

class CUDAPluggableAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self):
        alloc_fn = ctypes.cast(getattr(control.lib, "alloc_fn"), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(control.lib, "free_fn"), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)


class XPUPluggableAllocator(torch.xpu.memory.XPUPluggableAllocator):
    """Construct the XPU allocator from AIMDO's already-loaded library."""

    def __init__(self, raw_segments=False):
        alloc_name = "xpu_raw_alloc_fn" if raw_segments else "xpu_alloc_fn"
        free_name = "xpu_raw_free_fn" if raw_segments else "xpu_free_fn"
        alloc_fn = ctypes.cast(
            getattr(control.lib, alloc_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(
            getattr(control.lib, free_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._xpu_customAllocator(alloc_fn, free_fn)

def get_torch_allocator(raw_segments=False):
    # In native-pool mode the callbacks own only raw native segments. Pool
    # routing and pressure-lifecycle validation remain the caller's job.
    if control.implementation == "xpu":
        return (
            None
            if control.lib is None
            else XPUPluggableAllocator(raw_segments=raw_segments)
        )
    logging.warning(f"WARNING: Aimdo+CUDAPluggableAllocator is experimental and unsupported.")
    return None if control.lib is None else CUDAPluggableAllocator()
