import torch
import ctypes

import logging

from . import control

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

    def __init__(self):
        alloc_fn = ctypes.cast(
            getattr(control.lib, "xpu_alloc_fn"), ctypes.c_void_p).value
        free_fn = ctypes.cast(
            getattr(control.lib, "xpu_free_fn"), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._xpu_customAllocator(alloc_fn, free_fn)

def get_torch_allocator():
    if control.implementation == "xpu":
        return (
            None
            if control.lib is None
            else XPUPluggableAllocator()
        )
    logging.warning(f"WARNING: Aimdo+CUDAPluggableAllocator is experimental and unsupported.")
    return None if control.lib is None else CUDAPluggableAllocator()
