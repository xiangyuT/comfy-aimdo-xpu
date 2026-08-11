import os
import ctypes
import platform
import sys
from pathlib import Path
import logging
import importlib.util
import contextlib
import threading

lib = None
devctxs = []
_log_callback = None
implementation = None
_torch_allocator = None
_torch_allocator_library = None
_xpu_allocator_ready = False
_xpu_allocator_mode = None
_torch_xpu_memory_pools = {}
_torch_xpu_memory_pools_lock = threading.Lock()
_torch_xpu_empty_cache_original = None
_torch_xpu_memory_stats_original = None
_torch_xpu_reset_peak_stats_original = None
_windows_dll_directories = []
_windows_dll_directory_paths = set()

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

_XPU_ALLOCATOR_MODES = frozenset(("global", "native_hook", "native_pool"))


def _normalize_xpu_allocator_mode(mode):
    mode = (
        os.environ.get("AIMDO_XPU_ALLOCATOR_MODE", "global")
        if mode is None
        else str(mode)
    )
    if mode not in _XPU_ALLOCATOR_MODES:
        choices = ", ".join(sorted(_XPU_ALLOCATOR_MODES))
        raise ValueError(
            f"unsupported XPU allocator mode {mode!r}; expected one of {choices}"
        )
    return mode


def _install_xpu_allocator_backend(torch_module, aimdo_torch_module, mode):
    if mode == "native_hook":
        return None
    allocator = aimdo_torch_module.get_torch_allocator(
        raw_segments=mode == "native_pool"
    )
    if allocator is None:
        raise RuntimeError("AIMDO XPU allocator is unavailable")
    if mode == "global":
        torch_module.xpu.memory.change_current_allocator(allocator)
    return allocator


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


def _xpu_initialization_requested(implementation_was_explicit):
    """Keep ComfyUI XPU allocator installation explicitly opt-in."""
    if implementation_was_explicit:
        return True

    comfy_cli_args = sys.modules.get("comfy.cli_args")
    if comfy_cli_args is None:
        return True

    parsed_args = getattr(comfy_cli_args, "args", None)
    if parsed_args is None or not hasattr(parsed_args, "enable_dynamic_vram"):
        return True

    return bool(parsed_args.enable_dynamic_vram)


def _add_xpu_runtime_dll_directories():
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    candidates = (
        Path(sys.executable).resolve().parent / "Library" / "bin",
        Path(sys.prefix).resolve() / "Library" / "bin",
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir() and str(candidate) not in _windows_dll_directory_paths:
            _windows_dll_directories.append(os.add_dll_directory(str(candidate)))
            _windows_dll_directory_paths.add(str(candidate))


def init(
    implementation: str | None = None,
    simple_vram_headroom: int | None = None,
    nvml_pressure: bool = False,
    xpu_allocator_mode: str | None = None,
):
    global lib, _log_callback, _torch_allocator
    global _torch_allocator_library, _xpu_allocator_ready
    global _xpu_allocator_mode
    global _torch_xpu_empty_cache_original
    global _torch_xpu_memory_stats_original
    global _torch_xpu_reset_peak_stats_original

    if lib is not None:
        if xpu_allocator_mode is not None:
            requested_mode = _normalize_xpu_allocator_mode(xpu_allocator_mode)
            if implementation == "xpu" or globals()["implementation"] == "xpu":
                if requested_mode != _xpu_allocator_mode:
                    raise RuntimeError(
                        "AIMDO XPU allocator mode cannot change after initialization"
                    )
            else:
                raise ValueError(
                    "xpu_allocator_mode is valid only for the XPU implementation"
                )
        if simple_vram_headroom is not None:
            lib.set_simple_vram_headroom(int(simple_vram_headroom))
        lib.set_nvml_pressure(bool(nvml_pressure))
        return True

    implementation_was_explicit = implementation is not None
    if implementation is None:
        implementation = detect_vendor()

    if implementation is None:
        logging.warning("Could not autodetect AIMDO implementation, assuming Nvidia")
        implementation = "cuda"

    if implementation == "xpu":
        requested_xpu_allocator_mode = _normalize_xpu_allocator_mode(
            xpu_allocator_mode
        )
    elif xpu_allocator_mode is not None:
        raise ValueError(
            "xpu_allocator_mode is valid only for the XPU implementation"
        )
    else:
        requested_xpu_allocator_mode = None

    if (
        implementation == "xpu"
        and not _xpu_initialization_requested(implementation_was_explicit)
    ):
        logging.info(
            "comfy-aimdo XPU allocator was not explicitly enabled; "
            "using the native PyTorch XPU allocator"
        )
        return False

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
            if implementation == "xpu":
                _add_xpu_runtime_dll_directories()
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
        native_hook_symbols = (
            "xpu_ur_hook_is_interposed",
            "xpu_ur_hook_enable",
            "xpu_ur_hook_disable",
            "xpu_ur_hook_get_stats",
        )
        native_hook_available = (
            platform.system() == "Linux"
            and all(hasattr(lib, name) for name in native_hook_symbols)
        )
        if native_hook_available:
            lib.xpu_ur_hook_is_interposed.argtypes = []
            lib.xpu_ur_hook_is_interposed.restype = ctypes.c_bool
            lib.xpu_ur_hook_enable.argtypes = []
            lib.xpu_ur_hook_enable.restype = ctypes.c_bool
            lib.xpu_ur_hook_disable.argtypes = []
            lib.xpu_ur_hook_disable.restype = ctypes.c_bool
            lib.xpu_ur_hook_get_stats.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            lib.xpu_ur_hook_get_stats.restype = ctypes.c_bool

        if (
            requested_xpu_allocator_mode == "native_hook"
            and (
                not native_hook_available
                or not lib.xpu_ur_hook_is_interposed()
            )
        ):
            logging.error(
                "comfy-aimdo XPU native hook requires Linux and must be "
                "loaded through LD_PRELOAD before Python starts"
            )
            lib.set_log_callback(ctypes.cast(None, _LOG_CALLBACK))
            _log_callback = None
            lib = None
            globals()["implementation"] = None
            return False

        try:
            if system != "Windows" and _torch_allocator is None:
                import torch
                from . import torch as aimdo_torch

                allocator = _install_xpu_allocator_backend(
                    torch, aimdo_torch, requested_xpu_allocator_mode
                )
                _torch_allocator = allocator
                # The allocator owns function pointers into this CDLL. Keep a
                # process-lifetime reference even if control.deinit() is used
                # by a focused test and AIMDO is initialized again later.
                _torch_allocator_library = lib
                _xpu_allocator_mode = requested_xpu_allocator_mode

                if requested_xpu_allocator_mode == "global":
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
            elif requested_xpu_allocator_mode != _xpu_allocator_mode:
                raise RuntimeError(
                    "AIMDO XPU allocator mode cannot change after installation"
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
        if (
            implementation == "xpu"
            and _xpu_allocator_mode == "native_hook"
            and not lib.xpu_ur_hook_enable()
        ):
            lib.cleanup()
            devctxs = []
            lib.plat_cleanup()
            return False
        if implementation == "xpu" and _xpu_allocator_mode == "native_hook":
            logging.info(
                "comfy-aimdo XPU native allocator hook enabled; "
                "PyTorch caching allocator retained"
            )
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
            if _xpu_allocator_mode == "native_hook":
                import torch

                torch.xpu.empty_cache()
                torch.xpu.synchronize()
                if not lib.xpu_ur_hook_disable():
                    raise RuntimeError(
                        "cannot disable AIMDO XPU native hook while "
                        "tracked native segments remain live"
                    )
            elif platform.system() == "Windows":
                # Windows retains PyTorch's native XPU allocator even without
                # the native hook, so there is no AIMDO block cache to drain.
                import torch
                torch.xpu.empty_cache()
            else:
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


def get_xpu_ur_hook_stats():
    if (
        lib is None
        or implementation != "xpu"
        or not hasattr(lib, "xpu_ur_hook_get_stats")
    ):
        return {}
    names = (
        "alloc_calls",
        "free_calls",
        "pass_through_alloc_calls",
        "tracked_alloc_calls",
        "tracked_alloc_bytes",
        "tracked_free_calls",
        "tracked_free_bytes",
        "synthetic_oom_calls",
        "runtime_oom_calls",
        "native_reclaim_free_calls",
        "native_reclaim_free_bytes",
        "retry_eviction_calls",
        "retry_eviction_bytes",
        "unknown_device_calls",
        "unknown_free_calls",
        "dropped_metadata_calls",
        "direct_pressure_calls",
        "direct_pressure_bytes",
        "duplicate_pointer_calls",
    )
    values = (ctypes.c_uint64 * len(names))()
    if not lib.xpu_ur_hook_get_stats(values, len(names)):
        raise RuntimeError("failed to query AIMDO XPU UR hook statistics")
    return dict(zip(names, map(int, values)))


def empty_xpu_allocator_cache(wait=False):
    if lib is None or implementation != "xpu" or not _xpu_allocator_ready:
        return False
    if platform.system() == "Windows":
        import torch
        torch.xpu.empty_cache()
        return True
    return bool(lib.xpu_allocator_empty_cache(bool(wait)))


def get_xpu_allocator_mode():
    return _xpu_allocator_mode


def _xpu_device_index(device=None):
    import torch

    if device is None:
        return int(torch.xpu.current_device())
    if isinstance(device, int):
        return device
    parsed = torch.device(device)
    if parsed.type != "xpu":
        raise ValueError(f"expected an XPU device, got {parsed}")
    return int(torch.xpu.current_device() if parsed.index is None else parsed.index)


def get_xpu_allocator_pool(device=None):
    if (
        lib is None
        or implementation != "xpu"
        or not _xpu_allocator_ready
        or _xpu_allocator_mode != "native_pool"
    ):
        raise RuntimeError("AIMDO XPU native-pool mode is not initialized")

    import torch

    device_index = _xpu_device_index(device)
    with _torch_xpu_memory_pools_lock:
        pool = _torch_xpu_memory_pools.get(device_index)
        if pool is None:
            with torch.xpu.device(device_index):
                pool = torch.xpu.MemPool(_torch_allocator.allocator())
            _torch_xpu_memory_pools[device_index] = pool
        return pool


@contextlib.contextmanager
def use_xpu_allocator_pool(device=None):
    import torch

    device_index = _xpu_device_index(device)
    pool = get_xpu_allocator_pool(device_index)
    with torch.xpu.use_mem_pool(pool, device=device_index):
        yield pool


def release_xpu_allocator_pool(device=None):
    device_index = _xpu_device_index(device)
    with _torch_xpu_memory_pools_lock:
        pool = _torch_xpu_memory_pools.get(device_index)
        if pool is None:
            return False
        if pool.use_count() != 1:
            raise RuntimeError(
                f"AIMDO XPU pool for device {device_index} is still active"
            )
        del _torch_xpu_memory_pools[device_index]
    del pool
    return True


def get_xpu_allocator_memory_stats(device=None):
    if lib is None or implementation != "xpu" or not _xpu_allocator_ready:
        return (0, 0, 0, 0)
    if platform.system() == "Windows":
        import torch
        stats = torch.xpu.memory_stats(device)
        return (
            int(stats.get("active_bytes.all.current", 0)),
            int(stats.get("reserved_bytes.all.current", 0)),
            int(stats.get("active_bytes.all.peak", 0)),
            int(stats.get("reserved_bytes.all.peak", 0)),
        )
    values = (ctypes.c_uint64 * 4)()
    if not lib.xpu_allocator_get_memory_stats(
        _xpu_device_index(device), values, len(values)
    ):
        raise RuntimeError("failed to query AIMDO XPU allocator memory statistics")
    return tuple(map(int, values))


def reset_xpu_allocator_peak_stats(device=None):
    if lib is None or implementation != "xpu" or not _xpu_allocator_ready:
        return
    if platform.system() == "Windows":
        import torch
        torch.xpu.reset_peak_memory_stats(device)
        return
    lib.xpu_allocator_reset_peak_stats(_xpu_device_index(device))
