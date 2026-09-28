"""Opt-in Linux Torch 2.14 native-owner diagnostic for the XPU provider.

The allocator proxy has process lifetime. This module is deliberately separate
from public malloc_graph.record(), whose XPU capability remains unavailable.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
from pathlib import Path
import platform


_TORCH_VERSION = "2.14.0+xpu"
_ENVIRONMENT_FLAG = "AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC"
_library = None


def requested() -> bool:
    value = os.environ.get(_ENVIRONMENT_FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_ENVIRONMENT_FLAG} must be 0 or 1")
    return value == "1"


def installed() -> bool:
    return bool(_library is not None and _library.aimdo_full_proxy_is_installed())


def install(torch_module) -> None:
    """Install once, before the first XPU device allocation."""
    global _library
    if platform.system() != "Linux":
        raise RuntimeError("native-owner diagnostic is Linux-only")
    if torch_module.__version__ != _TORCH_VERSION:
        raise RuntimeError(
            f"native-owner diagnostic requires Torch {_TORCH_VERSION}"
        )
    if _library is not None:
        if not installed():
            raise RuntimeError("native-owner diagnostic lost its process owner")
        return
    if torch_module.xpu.is_initialized():
        raise RuntimeError("native-owner diagnostic must install before XPU initialization")

    path = Path(__file__).resolve().parent / "aimdo_xpu_native_owner.so"
    if not path.is_file():
        raise RuntimeError("native-owner diagnostic DSO is missing from this provider")
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    library.aimdo_full_proxy_torch_version.argtypes = []
    library.aimdo_full_proxy_torch_version.restype = ctypes.c_char_p
    library.aimdo_full_proxy_is_installed.argtypes = []
    library.aimdo_full_proxy_is_installed.restype = ctypes.c_bool
    library.aimdo_full_proxy_install.argtypes = []
    library.aimdo_full_proxy_install.restype = ctypes.c_bool
    library.aimdo_full_proxy_test_fail_next_compiler_owner_insert.argtypes = []
    library.aimdo_full_proxy_test_fail_next_compiler_owner_insert.restype = ctypes.c_bool
    library.aimdo_full_proxy_test_duplicate_next_compiler_pointer.argtypes = [
        ctypes.c_void_p,
    ]
    library.aimdo_full_proxy_test_duplicate_next_compiler_pointer.restype = ctypes.c_bool
    library.aimdo_full_proxy_compiler_begin.argtypes = [
        ctypes.c_size_t, ctypes.c_uint64, ctypes.c_char_p,
    ]
    library.aimdo_full_proxy_compiler_begin.restype = ctypes.c_bool
    library.aimdo_full_proxy_compiler_end.argtypes = []
    library.aimdo_full_proxy_compiler_end.restype = ctypes.c_bool
    library.aimdo_full_proxy_snapshot.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
    ]
    library.aimdo_full_proxy_snapshot.restype = ctypes.c_bool

    if library.aimdo_full_proxy_torch_version() != _TORCH_VERSION.encode():
        raise RuntimeError("native-owner DSO Torch ABI does not match the runtime")
    if not library.aimdo_full_proxy_is_installed() and not library.aimdo_full_proxy_install():
        raise RuntimeError(
            "native-owner proxy installation failed; the process must exit"
        )
    if not library.aimdo_full_proxy_is_installed():
        raise RuntimeError("native-owner proxy installation was not retained")
    _library = library  # Never unload a library that owns Torch DataPtr deleters.


def snapshot() -> list[int]:
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    values = (ctypes.c_uint64 * 17)()
    if not _library.aimdo_full_proxy_snapshot(values, len(values)):
        raise RuntimeError("native-owner diagnostic snapshot failed")
    return list(map(int, values))


def inject_next_compiler_owner_insert_failure() -> None:
    """Diagnostic-only fault injection within an active selected scope."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    if not _library.aimdo_full_proxy_test_fail_next_compiler_owner_insert():
        raise RuntimeError("no active compiler scope for owner-insert injection")


def inject_duplicate_compiler_pointer(pointer: int) -> None:
    """Diagnostic-only collision with one currently live compiler owner."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    if not _library.aimdo_full_proxy_test_duplicate_next_compiler_pointer(
        ctypes.c_void_p(int(pointer))
    ):
        raise RuntimeError("no active scope or matching live compiler owner")


def inject_page_create_oom_attempts(attempts: int) -> None:
    """Fail one or both physical-page attempts in the active diagnostic graph."""
    if type(attempts) is not int or attempts not in (1, 2):
        raise ValueError("page-create OOM attempts must be 1 or 2")
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    if control.lib is None or control.implementation != "xpu" or \
            control.get_xpu_allocator_mode() != "native_hook":
        raise RuntimeError("page-create OOM injection requires XPU native_hook")
    try:
        inject = control.lib.malloc_graph_test_fail_next_page_creates
    except AttributeError as error:
        raise RuntimeError("page-create OOM diagnostic export is missing") from error
    inject.argtypes = [ctypes.c_uint]
    inject.restype = ctypes.c_bool
    if not inject(attempts):
        raise RuntimeError("no eligible active graph for page-create OOM injection")


def selected_scope(size: int, stream):
    """Route one exact-size tensor request within an active native graph.

    This limited diagnostic is not the public AIMDO memory compiler API.
    """
    size = int(size)
    if size <= 0:
        raise ValueError("native-owner selected scope size must be positive")
    return _compiler_scope(size, stream)


def compiler_scope(stream):
    """Route every positive-size Torch tensor request on this stream.

    This is an opt-in D2 diagnostic; public malloc_graph.record() is disabled.
    """
    return _compiler_scope(0, stream)


def record_diagnostic(stream, assert_graph_breaks: bool = False):
    """Create a memory-only XPU graph for the opt-in diagnostic route."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    if control.lib is None or control.implementation != "xpu" or \
            control.get_xpu_allocator_mode() != "native_hook":
        raise RuntimeError("native-owner graph requires active XPU native_hook")
    device = getattr(stream, "device", None)
    if getattr(device, "type", None) != "xpu":
        raise ValueError("native-owner graph requires an XPU stream")
    index = device.index
    if index is None:
        import torch
        index = torch.xpu.current_device()
    stream_pointer = int(stream.sycl_queue)
    if not stream_pointer:
        raise ValueError("native-owner graph requires a nonzero XPU queue")
    handle = control.lib.malloc_graph_create(
        control.get_devctx(index), ctypes.c_void_p(stream_pointer),
        bool(assert_graph_breaks),
    )
    if not handle:
        raise RuntimeError("AIMDO native-owner graph creation failed")
    from .malloc_graph import MallocGraph
    return MallocGraph(handle, stream)


@contextlib.contextmanager
def _compiler_scope(size: int, stream):
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    if control.lib is None or control.implementation != "xpu" or \
            control.get_xpu_allocator_mode() != "native_hook":
        raise RuntimeError("native-owner scope requires active XPU native_hook")
    if size < 0:
        raise ValueError("native-owner scope size cannot be negative")
    stream_pointer = int(stream.sycl_queue)
    source_revision = control.get_memory_compiler_capability()["source_revision"]
    if not stream_pointer or not source_revision:
        raise RuntimeError("native-owner scope lacks stream or D1 source identity")
    if not _library.aimdo_full_proxy_compiler_begin(
        size, stream_pointer, source_revision.encode("ascii")
    ):
        raise RuntimeError("native-owner compiler scope could not start")
    try:
        yield
    finally:
        if not _library.aimdo_full_proxy_compiler_end():
            raise RuntimeError("native-owner compiler scope could not end")
