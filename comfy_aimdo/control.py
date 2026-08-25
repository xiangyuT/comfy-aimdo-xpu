import os
import ctypes
import platform
import sys
from pathlib import Path
import logging
import importlib.util
import time

lib = None
devctxs = []
_log_callback = None
implementation = None
_torch_allocator = None
_torch_allocator_library = None
_xpu_allocator_ready = False
_xpu_allocator_mode = None
_torch_xpu_empty_cache_original = None
_torch_xpu_memory_stats_original = None
_torch_xpu_reset_peak_stats_original = None
_windows_dll_directories = []
_windows_dll_directory_paths = set()
_xpu_oom_history = []
_XPU_OOM_HISTORY_LIMIT = 4
_XPU_OOM_SNAPSHOT_INTERVAL_SECONDS = 2.0
_xpu_oom_last_snapshot_monotonic = {}

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

_XPU_ALLOCATOR_MODES = frozenset(("global", "native_hook"))


def _normalize_xpu_allocator_mode(mode):
    # Windows keeps PyTorch's native XPU caching allocator and arbitrates at
    # the Unified Runtime allocation site instead of replacing the allocator,
    # so the native hook is the default there.
    default = "native_hook" if platform.system() == "Windows" else "global"
    mode = (
        os.environ.get("AIMDO_XPU_ALLOCATOR_MODE", default)
        if mode is None
        else str(mode)
    )
    if mode not in _XPU_ALLOCATOR_MODES:
        choices = ", ".join(sorted(_XPU_ALLOCATOR_MODES))
        raise ValueError(
            f"unsupported XPU allocator mode {mode!r}; expected one of {choices}"
        )
    return mode


def _expandable_segments_enabled():
    """Report whether PyTorch will bypass USM allocation entirely.

    With expandable segments PyTorch reserves virtual address space and backs
    it with physical memory objects instead of allocating USM, so
    urUSMDeviceAlloc is never called and the native hook observes nothing. That
    is silent blindness rather than degraded arbitration, so it is refused.
    """
    configuration = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get(
        "PYTORCH_XPU_ALLOC_CONF", ""
    )
    for entry in configuration.split(","):
        key, separator, value = entry.partition(":")
        if separator and key.strip() == "expandable_segments":
            return value.strip().lower() in ("true", "1")
    return False


def _install_xpu_allocator_backend(torch_module, aimdo_torch_module, mode):
    if mode == "native_hook":
        return None
    allocator = aimdo_torch_module.get_torch_allocator()
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
        and requested_xpu_allocator_mode == "native_hook"
        and os.environ.get("AIMDO_XPU_DISABLE_UR_HOOK") == "1"
    ):
        # The same switch the native side honours. Without this the hook would
        # still be attached through xpu_ur_hook_is_interposed() below and both
        # control loops would arbitrate the same pressure.
        logging.warning(
            "comfy-aimdo XPU native hook disabled by AIMDO_XPU_DISABLE_UR_HOOK; "
            "falling back to post-allocation Level Zero reclaim"
        )
        requested_xpu_allocator_mode = "global"

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
        if platform.system() == "Windows":
            lib.aimdo_xpu_is_mapped_pinned_vbar.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.aimdo_xpu_is_mapped_pinned_vbar.restype = ctypes.c_bool
            lib.aimdo_xpu_needs_small_vbar_copy_workaround.argtypes = [
                ctypes.c_int,
            ]
            lib.aimdo_xpu_needs_small_vbar_copy_workaround.restype = ctypes.c_bool
            lib.aimdo_xpu_copy_host_to_vbar.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.aimdo_xpu_copy_host_to_vbar.restype = ctypes.c_bool
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
            platform.system() in ("Linux", "Windows")
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

        if requested_xpu_allocator_mode == "native_hook":
            if _expandable_segments_enabled():
                logging.error(
                    "comfy-aimdo XPU native hook cannot arbitrate while "
                    "expandable_segments is enabled: PyTorch then allocates "
                    "physical memory directly and never calls "
                    "urUSMDeviceAlloc. Unset expandable_segments in "
                    "PYTORCH_ALLOC_CONF or select another allocator mode."
                )
                lib.set_log_callback(ctypes.cast(None, _LOG_CALLBACK))
                _log_callback = None
                lib = None
                globals()["implementation"] = None
                return False
            if not native_hook_available or not lib.xpu_ur_hook_is_interposed():
                if platform.system() == "Windows":
                    logging.error(
                        "comfy-aimdo XPU native hook could not attach to "
                        "ur_loader.dll; import torch before initializing AIMDO"
                    )
                else:
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
            if system == "Windows":
                # Windows never replaces PyTorch's allocator. Record the mode so
                # init_devices() enables the native hook and deinit() tears it
                # down through the same path as Linux.
                _xpu_allocator_mode = requested_xpu_allocator_mode
            elif _torch_allocator is None:
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
        "host_to_device_split_retries",
        "torch_allocator_alloc_calls",
        "torch_allocator_free_calls",
        "torch_allocator_cache_hits",
        "torch_allocator_physical_alloc_calls",
        "torch_allocator_physical_alloc_bytes",
        "torch_allocator_physical_release_calls",
        "torch_allocator_physical_release_bytes",
        "small_vbar_copy_fallback_calls",
        "small_vbar_copy_fallback_bytes",
        "small_vbar_copy_fallback_failures",
        "retire_token_calls",
        "retire_fence_submit_calls",
        "retire_fence_complete_calls",
        "retire_fence_submit_failures",
        "retire_force_polls",
        "retire_tracked_queues",
        "retire_queue_registration_failures",
        "retire_queue_identity_mismatches",
        "retire_fence_query_failures",
        "retire_shutdown_wait_failures",
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
    if platform.system() == "Windows":
        # Windows also counts the entry point PyTorch uses instead of USM when
        # expandable segments are enabled, so a blind hook is observable.
        names = names + ("physical_mem_create_calls", "cache_lever_skipped_calls")
    values = (ctypes.c_uint64 * len(names))()
    if not lib.xpu_ur_hook_get_stats(values, len(names)):
        raise RuntimeError("failed to query AIMDO XPU UR hook statistics")
    return dict(zip(names, map(int, values)))


def get_xpu_ur_hook_timing():
    """Hook and classification cost, Windows only.

    Reported separately from the shared statistics table so the Linux and
    Windows tables stay identical. ``hook_ns`` includes the driver call the
    hook wraps, which is what distinguishes a slow hook from slow consequences
    of the hook's decisions.
    """
    if (
        lib is None
        or implementation != "xpu"
        or not hasattr(lib, "xpu_ur_hook_get_hook_timing")
    ):
        return {}
    calls = ctypes.c_uint64()
    nanoseconds = ctypes.c_uint64()
    hits = ctypes.c_uint64()
    lib.xpu_ur_hook_get_hook_timing.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.xpu_ur_hook_get_hook_timing.restype = ctypes.c_bool
    lib.xpu_ur_hook_get_classify_timing.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.xpu_ur_hook_get_classify_timing.restype = ctypes.c_bool
    if not lib.xpu_ur_hook_get_hook_timing(
        ctypes.byref(calls), ctypes.byref(nanoseconds)
    ):
        return {}
    result = {"hook_calls": int(calls.value), "hook_ns": int(nanoseconds.value)}
    if lib.xpu_ur_hook_get_classify_timing(
        ctypes.byref(calls), ctypes.byref(nanoseconds), ctypes.byref(hits)
    ):
        result["classify_calls"] = int(calls.value)
        result["classify_ns"] = int(nanoseconds.value)
        result["classify_torch_hits"] = int(hits.value)
    return result


def publish_torch_cached_bytes(device, cached_bytes=None):
    """Tell the native hook how much PyTorch is holding in its own cache.

    The hook refuses an over-budget PyTorch allocation so that PyTorch releases
    its cache and retries, which is the only way AIMDO can reclaim those bytes.
    That is expensive - PyTorch discards the whole cache - so it must not be
    done when there is no cache to reclaim.

    The hook cannot work this out for itself. A cached block is one PyTorch
    freed without returning it to the driver, so it never reaches urUSMFree and
    the cache growing is invisible from there.
    """
    if (
        lib is None
        or implementation != "xpu"
        or not hasattr(lib, "xpu_ur_hook_set_torch_cached_bytes")
    ):
        return None
    if cached_bytes is None:
        try:
            import torch

            stats = torch.xpu.memory_stats(device)
            cached_bytes = int(stats.get("reserved_bytes.all.current", 0)) - int(
                stats.get("allocated_bytes.all.current", 0)
            )
        except Exception:
            return None
    cached_bytes = max(int(cached_bytes), 0)
    lib.xpu_ur_hook_set_torch_cached_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.xpu_ur_hook_set_torch_cached_bytes.restype = None
    lib.xpu_ur_hook_set_torch_cached_bytes(int(device), cached_bytes)
    return cached_bytes


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


def get_xpu_memory_snapshot(
    device=None, include_native_segments=False, include_vbar_pages=True
):
    """Capture native allocator and AIMDO state without changing ownership.

    Windows ``native_hook`` mode deliberately keeps activation/workspace
    blocks in PyTorch's XPU caching allocator.  VBAR pages remain a separate
    foreign-allocation domain, so the two snapshots are reported side by side
    rather than merged into a fictitious native block list.
    """
    if implementation != "xpu":
        return {}
    import torch

    device_index = _xpu_device_index(device)
    owner = (
        "torch_xpu_native"
        if platform.system() == "Windows" and
        _xpu_allocator_mode == "native_hook"
        else "aimdo_xpu_pluggable"
    )
    try:
        native_stats = {
            str(key): int(value)
            for key, value in torch.xpu.memory_stats(device_index).items()
        }
    except Exception as error:
        native_stats = {"snapshot_error": str(error)}
    native_segments = None
    if include_native_segments:
        try:
            native_segments = torch.xpu.memory_snapshot()
        except Exception as error:
            native_segments = {"snapshot_error": str(error)}
    try:
        from .model_vbar import vbars_snapshot

        vbars = vbars_snapshot(
            device_index, include_pages=include_vbar_pages
        )
    except Exception as error:
        vbars = [{"snapshot_error": str(error)}]
    result = {
        "timestamp": time.time(),
        "device": device_index,
        "allocator_owner": owner,
        "native_allocator": {"stats": native_stats},
        "aimdo": {
            "vmm": get_xpu_vmm_stats(),
            "ur_hook": get_xpu_ur_hook_stats(),
            "vbars": vbars,
        },
    }
    if include_native_segments:
        result["native_allocator"]["segments"] = native_segments
    return result


def capture_xpu_oom_snapshot(
    device=None, *, stage, request_bytes=None, error=None,
    include_native_segments=False,
):
    """Record an owner-boundary OOM snapshot; never called from a hook."""
    device_index = _xpu_device_index(device)
    now = time.monotonic()
    previous_time = _xpu_oom_last_snapshot_monotonic.get(device_index)
    if (
        _xpu_oom_history and previous_time is not None and
        now - previous_time < _XPU_OOM_SNAPSHOT_INTERVAL_SECONDS and
        _xpu_oom_history[-1].get("device") == device_index
    ):
        snapshot = _xpu_oom_history[-1]
        snapshot["timestamp"] = time.time()
        snapshot["oom"] = {
            "stage": str(stage),
            "request_bytes": (
                None if request_bytes is None else int(request_bytes)
            ),
            "error": None if error is None else str(error),
            "coalesced_events": int(
                snapshot.get("oom", {}).get("coalesced_events", 1)
            ) + 1,
        }
        return snapshot
    snapshot = get_xpu_memory_snapshot(
        device_index,
        include_native_segments=include_native_segments,
        include_vbar_pages=False,
    )
    if not snapshot:
        return snapshot
    snapshot["oom"] = {
        "stage": str(stage),
        "request_bytes": (
            None if request_bytes is None else int(request_bytes)
        ),
        "error": None if error is None else str(error),
        "coalesced_events": 1,
    }
    _xpu_oom_history.append(snapshot)
    del _xpu_oom_history[:-_XPU_OOM_HISTORY_LIMIT]
    _xpu_oom_last_snapshot_monotonic[device_index] = now
    return snapshot


def get_last_xpu_oom_snapshot(device=None):
    """Return the newest recorded OOM snapshot, optionally for one device."""
    if device is None:
        return _xpu_oom_history[-1] if _xpu_oom_history else None
    device_index = _xpu_device_index(device)
    for snapshot in reversed(_xpu_oom_history):
        if snapshot.get("device") == device_index:
            return snapshot
    return None


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
