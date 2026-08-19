import ctypes
import contextvars
import itertools
import os
import sys
import time
from contextlib import contextmanager

from . import control

lib = control.lib

_trace_enabled = os.environ.get("AIMDO_XPU_VBAR_TRACE") == "1"
_boundary_trace_enabled = (
    os.environ.get("AIMDO_XPU_BOUNDARY_TRACE") == "1"
)
_trace_calls = itertools.count(1)
_inference_memory_budgets = contextvars.ContextVar(
    "comfy_aimdo_inference_memory_budgets", default=None
)

# Torch's native XPU allocator keeps freed blocks cached, and that cache holds
# WDDM local memory that AIMDO cannot reclaim: a tensor that is freed does not
# reach zeMemFree, so the allocation hook never sees it come back.  Only
# empty_cache() returns it.  It cannot be called from the allocation hook -
# Torch holds its allocator lock across the driver call - but a VBAR fault that
# is about to give up is a queue-safe boundary and a real, measured shortage.
_native_cache_trim_enabled = (
    os.environ.get("AIMDO_XPU_NATIVE_CACHE_TRIM", "1") != "0"
)
# empty_cache() calls sycl::free(), which can stall in the Level Zero/UMF
# residency path under pressure, so this stays rate limited rather than
# becoming a per-weight operation.
_NATIVE_CACHE_TRIM_INTERVAL_SECONDS = 2.0
_native_cache_trim_last = 0.0
_unpin_stream_supported = (
    lib is not None
    and hasattr(lib, "vbar_unpin_stream")
    and sys.platform == "win32"
    and control.implementation == "xpu"
)

if _unpin_stream_supported:
    lib.vbar_unpin_stream.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64,
    ]


def _current_queue_ptr(device):
    """SYCL queue backing the caller's current Torch XPU stream."""
    try:
        import torch

        return int(torch.xpu.current_stream(
            torch.device("xpu", device)).sycl_queue)
    except Exception:
        return 0


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


@contextmanager
def inference_memory_budget(memory_required, devices):
    """Publish a scoped inference allocation budget for model activation."""
    budget = max(0, int(memory_required))
    current = _inference_memory_budgets.get()
    updated = {} if current is None else dict(current)
    for device in devices:
        device_index = getattr(device, "index", device)
        if device_index is not None:
            device_index = int(device_index)
            updated[device_index] = max(budget, updated.get(device_index, 0))

    token = _inference_memory_budgets.set(updated)
    try:
        yield
    finally:
        _inference_memory_budgets.reset(token)


def current_inference_memory_budget(device):
    budgets = _inference_memory_budgets.get()
    if budgets is None:
        return 0
    device_index = getattr(device, "index", device)
    if device_index is None:
        return 0
    return budgets.get(int(device_index), 0)

# Bindings
if lib is not None:
    lib.vbar_allocate.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    lib.vbar_allocate.restype = ctypes.c_void_p

    lib.vbar_set_watermark_limit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]

    lib.vbar_set_watermark.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]

    lib.vbars_reset_watermark_limits.argtypes = [ctypes.c_void_p]

    lib.vbars_prepare_allocation.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64
    ]

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

def _release_native_cache(device):
    """Return Torch's freed-but-cached device memory, if that is the shortage.

    Only worth doing when Torch is actually sitting on dead blocks; a VBAR miss
    caused by live tensors cannot be helped this way.  Returns True when
    something was released and a refault is worth attempting.
    """
    global _native_cache_trim_last

    if (not _native_cache_trim_enabled
            or sys.platform != "win32"
            or control.implementation != "xpu"):
        return False

    now = time.monotonic()
    if now - _native_cache_trim_last < _NATIVE_CACHE_TRIM_INTERVAL_SECONDS:
        return False
    # Throttle every attempt, including ones that find no cache or raise, and
    # time from completion: empty_cache() can itself take seconds under
    # pressure, and timing from the start would allow back-to-back calls.
    _native_cache_trim_last = now

    try:
        import torch

        stats = torch.xpu.memory_stats(device)
        cached = (int(stats.get("reserved_bytes.all.current", 0))
                  - int(stats.get("allocated_bytes.all.current", 0)))
        # The allocation hook needs this figure too, and cannot derive it: a
        # cached block never reaches the driver.
        control.publish_torch_cached_bytes(device, cached)
        if cached < 32 * 1024 ** 2:
            return False
        torch.xpu.empty_cache()
    except Exception:
        return False
    finally:
        _native_cache_trim_last = time.monotonic()
    return True


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
        self._prioritized_once = False

    def prioritize(self, malloc_async_clamp=None):
        if malloc_async_clamp is None:
            malloc_async_clamp = ctypes.c_uint64(-1).value
        was_prioritized = self._prioritized_once
        previous_watermark = None
        if sys.platform == "win32" and self._prioritized_once:
            # Record the prior working set for boundary diagnosis. It must not
            # become the next activation's hard ceiling: tiled models revisit
            # weights above a pressure-reduced watermark and would otherwise
            # stream those weights from host storage for every tile.
            previous_watermark = lib.vbar_get_watermark(
                self._devctx, self._ptr
            )
        lib.vbar_prioritize(self._devctx, self._ptr, malloc_async_clamp)
        if sys.platform == "win32":
            _, reserved, _, peak_reserved = (
                control.get_xpu_allocator_memory_stats(self.device)
            )
            historical_growth = max(0, peak_reserved - reserved)
            inference_budget = current_inference_memory_budget(self.device)
            anticipated_growth = max(historical_growth, inference_budget)
            prepared_allocation = False
            if anticipated_growth:
                # Linux's pluggable allocator can safely grow a newly
                # prioritized model under exact allocation-time pressure.
                # Windows only has a historical model-boundary estimate. Keep
                # the normal reset-to-full watermark so a tiled/repeated model
                # can fault its complete address range again, while excluding
                # the active VBAR from this speculative reclaim.
                # Level Zero allocation callbacks cannot safely wait on the
                # same SYCL queue. Reclaim at this model-switch boundary using
                # the native allocator's observed peak as the next-burst hint.
                lib.vbars_prepare_allocation(
                    self._devctx, self._ptr, anticipated_growth
                )
                prepared_allocation = True
            if _boundary_trace_enabled:
                current_watermark = lib.vbar_get_watermark(
                    self._devctx, self._ptr
                )
                print(
                    "[AIMDO XPU BOUNDARY] "
                    f"vbar=0x{self.base_addr:x} "
                    f"was_prioritized={was_prioritized} "
                    f"previous_watermark={previous_watermark} "
                    f"current_watermark={current_watermark} "
                    f"reserved={reserved} peak_reserved={peak_reserved} "
                    f"historical_growth={historical_growth} "
                    f"inference_budget={inference_budget} "
                    f"anticipated_growth={anticipated_growth} "
                    f"prepared_allocation={prepared_allocation}",
                    flush=True,
                )
        self._prioritized_once = True

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
        if res == 1 and _release_native_cache(self.device):
            # The shortage was at least partly Torch's own dead cache, which
            # AIMDO has no other way to reclaim. Retry once now that it is
            # back, rather than streaming this weight from host storage.
            res = lib.vbar_fault(
                self._devctx, self._ptr, offset, size, signature)
        if res == 0:
            return signature
        elif res == 1:
            return None
        else:
            raise RuntimeError(f"Fault failed: {res}")

    def unpin(self, alloc, size):
        offset = alloc - self.base_addr
        if _unpin_stream_supported:
            # VBAR map/unmap carry no stream, so this is the only point where
            # the queue that actually consumed the weight is visible. ComfyUI
            # may consume weights on a non-default stream, and a retirement
            # fence submitted only to the default queue does not order that
            # work: reclaiming on that proof caused DEVICE_LOST.
            lib.vbar_unpin_stream(
                self._devctx, self._ptr, offset, size,
                _current_queue_ptr(self.device))
            return
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
