import ctypes
import itertools
import os
import sys

from . import control

lib = control.lib

_trace_enabled = os.environ.get("AIMDO_XPU_VBAR_TRACE") == "1"
_trace_calls = itertools.count(1)


def _trace_vbar(operation, phase, alloc, caller, result=None):
    if not _trace_enabled:
        return
    vbar, offset, size = alloc
    module = caller.f_locals.get("s") or caller.f_locals.get("m")
    module_name = getattr(module, "seed_key", None)
    module_type = type(module).__qualname__ if module is not None else None
    weight = getattr(module, "weight", None)
    weight_shape = tuple(weight.shape) if weight is not None else None
    weight_dtype = str(weight.dtype) if weight is not None else None
    print(
        "[AIMDO XPU VBAR] "
        f"call={next(_trace_calls)} op={operation} phase={phase} "
        f"vbar=0x{vbar.base_addr:x} offset={offset - vbar.base_addr} "
        f"size={size} module={module_name!r} type={module_type!r} "
        f"weight_shape={weight_shape!r} weight_dtype={weight_dtype!r} "
        f"result={result!r}",
        file=sys.stderr,
        flush=True,
    )

# Bindings
if lib is not None:
    lib.vbar_allocate.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    lib.vbar_allocate.restype = ctypes.c_void_p

    lib.vbar_set_watermark_limit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]

    lib.vbar_set_watermark.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]

    lib.vbars_reset_watermark_limits.argtypes = [ctypes.c_void_p]

    lib.vbar_prioritize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]

    lib.vbar_deprioritize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.vbar_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vbar_get.restype = ctypes.c_uint64

    lib.vbar_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.vbar_fault.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32)]
    lib.vbar_fault.restype = ctypes.c_int

    lib.vbar_unpin.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

    lib.vbar_loaded_size.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vbar_loaded_size.restype = ctypes.c_size_t

    lib.vbar_free_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.vbar_free_memory.restype = ctypes.c_uint64

    lib.vbars_analyze.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.vbars_analyze.restype = ctypes.c_uint64

    lib.vbar_get_nr_pages.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vbar_get_nr_pages.restype = ctypes.c_size_t

    lib.vbar_get_watermark.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vbar_get_watermark.restype = ctypes.c_size_t

    lib.vbar_get_residency.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

class ModelVBAR:
    def __init__(self, size, device):
        self._devctx = control.get_devctx(device)
        self._ptr = lib.vbar_allocate(self._devctx, int(size), device)
        if not self._ptr:
            raise MemoryError("VBAR allocation failed")
        self.device = device
        self.max_size = size
        self.offset = 0
        self.base_addr = lib.vbar_get(self._devctx, self._ptr)

    def prioritize(self, malloc_async_clamp=None):
        if malloc_async_clamp is None:
            malloc_async_clamp = ctypes.c_uint64(-1).value
        lib.vbar_prioritize(self._devctx, self._ptr, malloc_async_clamp)

    def deprioritize(self):
        lib.vbar_deprioritize(self._devctx, self._ptr)

    def alloc(self, num_bytes):
        self.offset = (self.offset + 511) & ~511

        if self.offset + num_bytes > self.max_size:
            raise MemoryError("VBAR OOM")

        alloc = self.base_addr + self.offset
        self.offset += num_bytes
        return (self, alloc, num_bytes)

    #define VBAR_PAGE_SIZE (32 << 20)

    #define VBAR_FAULT_SUCCESS      0
    #define VBAR_FAULT_OOM          1
    #define VBAR_FAULT_ERROR        2

    def fault(self, alloc, size):
        offset = alloc - self.base_addr
        # +2, one for misalignment and one for rounding
        signature = (ctypes.c_uint32 * (size // (32 * 1024 ** 2) + 2))()
        res = lib.vbar_fault(self._devctx, self._ptr, offset, size, signature)
        if res == 0:
            return signature
        elif res == 1:
            return None
        else:
            raise RuntimeError(f"Fault failed: {res}")

    def unpin(self, alloc, size):
        offset = alloc - self.base_addr
        lib.vbar_unpin(self._devctx, self._ptr, offset, size)

    def loaded_size(self):
        return lib.vbar_loaded_size(self._devctx, self._ptr)

    def set_watermark_limit(self, size_bytes):
        lib.vbar_set_watermark_limit(self._devctx, self._ptr, size_bytes)

    def set_watermark(self, size_bytes):
        lib.vbar_set_watermark(self._devctx, self._ptr, size_bytes)

    def free_memory(self, size_bytes):
        return lib.vbar_free_memory(self._devctx, self._ptr, int(size_bytes))

    def get_nr_pages(self):
        return lib.vbar_get_nr_pages(self._devctx, self._ptr)

    def get_watermark(self):
        return lib.vbar_get_watermark(self._devctx, self._ptr)

    def get_residency(self):
        """Returns a list of per-page status flags.
        Bit 0 (& 1): resident in VRAM
        Bit 1 (& 2): pinned
        """
        nr_pages = self.get_nr_pages()
        buf = (ctypes.c_uint8 * nr_pages)()
        lib.vbar_get_residency(self._devctx, self._ptr, buf, nr_pages)
        return list(buf)

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        aimdo_lib = getattr(control, "lib", None)
        if aimdo_lib is not None and ptr:
            # control.init() may create a new CDLL wrapper after a focused
            # test calls deinit().  That wrapper has not necessarily inherited
            # the model-vbar argtypes bound above, so preserve pointer width
            # explicitly when freeing through the current library handle.
            aimdo_lib.vbar_free(
                ctypes.c_void_p(self._devctx), ctypes.c_void_p(ptr)
            )
            self._ptr = None

def vbar_fault(alloc):
    caller = sys._getframe(1)
    _trace_vbar("fault", "begin", alloc, caller)
    vbar, offset, size = alloc
    result = vbar.fault(offset, size)
    _trace_vbar(
        "fault", "end", alloc, caller,
        "vbar" if result is not None else "fallback",
    )
    return result

def vbar_unpin(alloc):
    if alloc is not None:
        caller = sys._getframe(1)
        _trace_vbar("unpin", "begin", alloc, caller)
        vbar, offset, size = alloc
        vbar.unpin(offset, size)
        _trace_vbar("unpin", "end", alloc, caller)

def vbar_signature_compare(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        raise ValueError(f"Signatures of mismatched length {len(a)} != {len(b)}")
    return memoryview(a) == memoryview(b)

def vbars_reset_watermark_limits():
    for devctx in control.devctxs:
        lib.vbars_reset_watermark_limits(devctx)

def vbars_analyze(device=None):
    if lib is None or not control.devctxs:
        return 0

    devctx = control.devctxs[0] if device is None else control.get_devctx(device)

    return lib.vbars_analyze(devctx, False)
