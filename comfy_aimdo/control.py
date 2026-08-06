import os
import ctypes
import platform
from pathlib import Path
import logging
import importlib.util

lib = None
devctxs = []
_log_callback = None
implementation = None
_torch_allocator = None
_torch_allocator_library = None
_xpu_allocator_ready = False
_torch_xpu_empty_cache_original = None
_torch_xpu_memory_stats_original = None
_torch_xpu_reset_peak_stats_original = None

_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
_LOG_LEVELS = {
    1: logging.CRITICAL,
    2: logging.ERROR,
    3: logging.WARNING,
    4: logging.INFO,
    5: logging.DEBUG,
    6: logging.DEBUG,
    7: logging.DEBUG,
}


def _native_log(level, message):
    logging.log(_LOG_LEVELS.get(level, logging.DEBUG),
                message.decode("utf-8", errors="replace").rstrip())


def detect_vendor():
    version = ""
    try:
        torch_spec = importlib.util.find_spec("torch")
        for folder in torch_spec.submodule_search_locations:
            ver_file = Path(folder) / "version.py"
            if ver_file.is_file():
                spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                version = module.__version__
    except Exception as e:
        logging.warning("Failed to detect Torch version")
        pass

    if '+cu' in version:
        return "cuda"
    if '+rocm' in version:
        return "rocm"
    if '+xpu' in version:
        return "xpu"
    return None


def init(implementation: str | None = None, simple_vram_headroom: int | None = None, nvml_pressure: bool = False):
    global lib, _log_callback, _torch_allocator
    global _torch_allocator_library, _xpu_allocator_ready
    global _torch_xpu_empty_cache_original
    global _torch_xpu_memory_stats_original
    global _torch_xpu_reset_peak_stats_original

    if lib is not None:
        if simple_vram_headroom is not None:
            lib.set_simple_vram_headroom(int(simple_vram_headroom))
        lib.set_nvml_pressure(bool(nvml_pressure))
        return True

    if implementation is None:
        implementation = detect_vendor()

    if implementation is None:
        logging.warning("Could not autodetect AIMDO implementation, assuming Nvidia")
        implementation = "cuda"

    globals()["implementation"] = implementation

    impl = {
        "cuda": "aimdo",
        "rocm": "aimdo_rocm",
        "xpu": "aimdo_xpu",
    }[implementation]

    try:
        base_path = Path(__file__).parent.resolve()
        system = platform.system()
        if system == "Windows":
            ext = "dll"
            mode = 0
        elif system == "Linux":
            ext = "so"
            mode = 258
        else:
            logging.info(f"comfy-aimdo unsupported operating system: {system}")
            logging.info(f"NOTE: comfy-aimdo currently only supports Windows and Linux")
            return False
        lib = ctypes.CDLL(str(base_path / f"{impl}.{ext}"), mode=mode)
    except Exception as e:
        logging.info(f"comfy-aimdo failed to load: {e}")
        logging.info(f"NOTE: comfy-aimdo currently only supports Nvidia, AMD, and Intel XPU GPUs")
        return False

    lib.set_log_callback.argtypes = [_LOG_CALLBACK]
    lib.set_log_callback.restype = None
    _log_callback = _LOG_CALLBACK(_native_log)
    lib.set_log_callback(_log_callback)

    lib.get_total_vram_usage.argtypes = [ctypes.c_void_p]
    lib.get_total_vram_usage.restype = ctypes.c_uint64

    lib.aimdo_analyze.argtypes = [ctypes.c_void_p]

    lib.set_simple_vram_headroom.argtypes = [ctypes.c_int64]
    lib.set_simple_vram_headroom.restype = None

    lib.set_nvml_pressure.argtypes = [ctypes.c_bool]
    lib.set_nvml_pressure.restype = None

    lib.init.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.init.restype = ctypes.c_bool

    lib.get_devctx.argtypes = [ctypes.c_int]
    lib.get_devctx.restype = ctypes.c_void_p

    if implementation == "xpu":
        lib.xpu_set_queues.argtypes = [ctypes.POINTER(ctypes.c_int),
                                       ctypes.POINTER(ctypes.c_uint64),
                                       ctypes.c_size_t]
        lib.xpu_set_queues.restype = ctypes.c_bool
        lib.xpu_get_vmm_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                                          ctypes.c_size_t]
        lib.xpu_get_vmm_stats.restype = ctypes.c_bool
        lib.xpu_allocator_empty_cache.argtypes = [ctypes.c_bool]
        lib.xpu_allocator_empty_cache.restype = ctypes.c_bool
        lib.xpu_allocator_get_memory_stats.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.xpu_allocator_get_memory_stats.restype = ctypes.c_bool
        lib.xpu_allocator_reset_peak_stats.argtypes = [ctypes.c_int]
        lib.xpu_allocator_reset_peak_stats.restype = None

        try:
            if _torch_allocator is None:
                import torch
                from . import torch as aimdo_torch

                allocator = aimdo_torch.get_torch_allocator()
                if allocator is None:
                    raise RuntimeError("AIMDO XPU allocator is unavailable")
                torch.xpu.memory.change_current_allocator(allocator)
                _torch_allocator = allocator
                # The allocator owns function pointers into this CDLL. Keep a
                # process-lifetime reference even if control.deinit() is used
                # by a focused test and AIMDO is initialized again later.
                _torch_allocator_library = lib
                _torch_xpu_empty_cache_original = torch.xpu.empty_cache
                _torch_xpu_memory_stats_original = torch.xpu.memory_stats
                _torch_xpu_reset_peak_stats_original = (
                    torch.xpu.reset_peak_memory_stats
                )

                def aimdo_xpu_empty_cache():
                    empty_xpu_allocator_cache(wait=False)
                    try:
                        return _torch_xpu_empty_cache_original()
                    except RuntimeError as error:
                        if "does not yet support emptyCache" not in str(error):
                            raise
                        return None

                torch.xpu.empty_cache = aimdo_xpu_empty_cache

                def aimdo_xpu_memory_stats(device=None):
                    active, reserved, peak_active, peak_reserved = (
                        get_xpu_allocator_memory_stats(device)
                    )
                    return {
                        "active_bytes.all.current": active,
                        "active_bytes.all.peak": peak_active,
                        "allocated_bytes.all.current": active,
                        "allocated_bytes.all.peak": peak_active,
                        "reserved_bytes.all.current": reserved,
                        "reserved_bytes.all.peak": peak_reserved,
                    }

                def aimdo_xpu_reset_peak_memory_stats(device=None):
                    reset_xpu_allocator_peak_stats(device)
                    try:
                        return _torch_xpu_reset_peak_stats_original(device)
                    except RuntimeError as error:
                        if "does not yet support resetPeakStats" not in str(error):
                            raise
                        return None

                torch.xpu.memory.memory_stats = aimdo_xpu_memory_stats
                torch.xpu.memory_stats = aimdo_xpu_memory_stats
                torch.xpu.memory.reset_peak_memory_stats = (
                    aimdo_xpu_reset_peak_memory_stats
                )
                torch.xpu.reset_peak_memory_stats = (
                    aimdo_xpu_reset_peak_memory_stats
                )
            _xpu_allocator_ready = True
        except Exception as error:
            _xpu_allocator_ready = False
            logging.error(f"comfy-aimdo failed to install XPU allocator: {error}")
            return False

    if simple_vram_headroom is not None:
        lib.set_simple_vram_headroom(int(simple_vram_headroom))
    lib.set_nvml_pressure(bool(nvml_pressure))

    return True

def init_devices(device_ids):
    global devctxs

    if lib is None:
        return False
    if implementation == "xpu" and not _xpu_allocator_ready:
        return False

    requested = []
    headrooms = []
    for device_id in device_ids:
        if isinstance(device_id, tuple):
            if len(device_id) != 2:
                raise ValueError("device tuple must be (device_id, extra_vram_headroom)")
            device_id, headroom = device_id
        else:
            headroom = 0

        headroom = int(headroom)
        if headroom < 0:
            raise ValueError("extra_vram_headroom must be non-negative")
        requested.append(int(device_id))
        headrooms.append(headroom)

    if not requested:
        return False

    if implementation == "xpu":
        import torch

        queue_ptrs = []
        for device_id in requested:
            stream = torch.xpu.current_stream(torch.device("xpu", device_id))
            queue_ptr = int(stream.sycl_queue)
            if not queue_ptr:
                raise RuntimeError(f"XPU device {device_id} returned a null SYCL queue")
            queue_ptrs.append(queue_ptr)

        device_array = (ctypes.c_int * len(requested))(*requested)
        queue_array = (ctypes.c_uint64 * len(queue_ptrs))(*queue_ptrs)
        if not lib.xpu_set_queues(device_array, queue_array, len(requested)):
            return False

    if not lib.plat_init():
        return False

    device_array = (ctypes.c_int * len(requested))(*requested)
    headroom_array = (ctypes.c_uint64 * len(headrooms))(*headrooms)
    if lib.init(device_array, headroom_array, len(requested)):
        devctxs = [get_devctx(device_id) for device_id in requested]
        return True

    devctxs = []
    lib.plat_cleanup()
    return False

def init_device(device_id, extra_vram_headroom: int = 0):
    if extra_vram_headroom:
        device_id = (device_id, extra_vram_headroom)
    return init_devices([device_id])

def get_devctx(device_id: int):
    devctx = lib.get_devctx(int(device_id))
    if devctx:
        return devctx
    raise RuntimeError(f"comfy-aimdo device {device_id} is not initialized")

def deinit():
    global lib, devctxs, _log_callback, _xpu_allocator_ready
    if lib is not None:
        if implementation == "xpu" and _xpu_allocator_ready:
            lib.xpu_allocator_empty_cache(True)
        lib.cleanup()
        devctxs = []
        lib.plat_cleanup()
        lib.set_log_callback(ctypes.cast(None, _LOG_CALLBACK))
        _log_callback = None
    lib = None
    globals()["implementation"] = None
    _xpu_allocator_ready = False


def set_log_none(): lib.set_log_level_none()
def set_log_critical(): lib.set_log_level_critical()
def set_log_error(): lib.set_log_level_error()
def set_log_warning(): lib.set_log_level_warning()
def set_log_info(): lib.set_log_level_info()
def set_log_debug(): lib.set_log_level_debug()
def set_log_verbose(): lib.set_log_level_verbose()
def set_log_vverbose(): lib.set_log_level_vverbose()

def analyze():
    if lib is None:
        return
    for devctx in devctxs:
        lib.aimdo_analyze(devctx)

def get_total_vram_usage():
    if lib is None:
        return 0
    return sum(lib.get_total_vram_usage(devctx) for devctx in devctxs)


def get_xpu_vmm_stats():
    if lib is None or implementation != "xpu":
        return {}
    names = (
        "virtual_reserve_calls",
        "virtual_reserve_bytes",
        "physical_create_calls",
        "physical_create_bytes",
        "map_calls",
        "map_bytes",
        "unmap_calls",
        "unmap_bytes",
        "physical_release_calls",
        "host_to_device_bytes",
        "queue_rebind_calls",
        "context_sync_calls",
        "context_sync_completions",
        "event_sync_calls",
        "event_sync_completions",
        "synchronous_host_to_device_calls",
        "synchronous_host_to_device_completions",
        "torch_allocator_alloc_calls",
        "torch_allocator_free_calls",
        "torch_allocator_cache_hits",
        "torch_allocator_physical_alloc_calls",
        "torch_allocator_physical_alloc_bytes",
        "torch_allocator_physical_release_calls",
        "torch_allocator_physical_release_bytes",
    )
    values = (ctypes.c_uint64 * len(names))()
    if not lib.xpu_get_vmm_stats(values, len(names)):
        raise RuntimeError("failed to query XPU VMM statistics")
    return dict(zip(names, map(int, values)))


def empty_xpu_allocator_cache(wait=False):
    if lib is None or implementation != "xpu" or not _xpu_allocator_ready:
        return False
    return bool(lib.xpu_allocator_empty_cache(bool(wait)))


def _xpu_device_index(device=None):
    import torch

    if device is None:
        return int(torch.xpu.current_device())
    parsed = torch.device(device)
    if parsed.type != "xpu":
        raise ValueError(f"expected an XPU device, got {parsed}")
    return int(torch.xpu.current_device() if parsed.index is None else parsed.index)


def get_xpu_allocator_memory_stats(device=None):
    if lib is None or implementation != "xpu" or not _xpu_allocator_ready:
        return (0, 0, 0, 0)
    values = (ctypes.c_uint64 * 4)()
    if not lib.xpu_allocator_get_memory_stats(
        _xpu_device_index(device), values, len(values)
    ):
        raise RuntimeError("failed to query AIMDO XPU allocator memory statistics")
    return tuple(map(int, values))


def reset_xpu_allocator_peak_stats(device=None):
    if lib is None or implementation != "xpu" or not _xpu_allocator_ready:
        return
    lib.xpu_allocator_reset_peak_stats(_xpu_device_index(device))
