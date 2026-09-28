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
import threading
import weakref


_TORCH_VERSION = "2.14.0+xpu"
_ENVIRONMENT_FLAG = "AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC"
_library = None
_scope_context = threading.local()
_scope_terminal = False
_graph_lock = threading.Lock()
_live_graphs = weakref.WeakSet()
_deferred_graphs = {}
_DEAD_OWNER_ERROR = "AIMDO graph owner thread exited; process must exit"


def _destroy_graph_checked(library, handle: int) -> bool:
    destroy = library.malloc_graph_destroy_checked
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_bool
    return bool(destroy(ctypes.c_void_p(int(handle))))


def _destroy_graph_terminal(library, handle: int) -> bool:
    terminal = getattr(library, "malloc_graph_destroy_terminal", None)
    if terminal is None:
        return False
    terminal.argtypes = [ctypes.c_void_p]
    terminal.restype = ctypes.c_bool
    return bool(terminal(ctypes.c_void_p(int(handle))))


def graph_destroy_terminal(graph) -> bool:
    """Report whether an owner-thread diagnostic graph must exit the process."""
    if not installed() or not getattr(graph, "_handle", None):
        raise RuntimeError("native-owner diagnostic graph is not live")
    if getattr(graph, "_owner_thread", None) is not threading.current_thread():
        raise RuntimeError("graph destroy status requires the owner thread")
    return _destroy_graph_terminal(graph._native_lib, graph._handle)


def _register_graph(graph) -> None:
    with _graph_lock:
        _live_graphs.add(graph)


def _defer_graph_destroy(owner: threading.Thread, library, handle: int) -> None:
    with _graph_lock:
        _deferred_graphs.setdefault(owner, []).append((library, int(handle)))


def dead_owner_graphs() -> int:
    """Count graph handles whose original Python owner thread has exited."""
    with _graph_lock:
        live = sum(
            bool(getattr(graph, "_handle", None))
            for graph in _live_graphs
            if (getattr(graph, "_owner_thread", None) is not None
                and not graph._owner_thread.is_alive())
        )
        deferred = sum(len(handles) for owner, handles in _deferred_graphs.items()
                       if not owner.is_alive())
    return live + deferred


def drain_deferred_graphs() -> int:
    """Complete queued tensor frees before graph closes on their owner thread."""
    if _scope_terminal:
        raise RuntimeError("native-owner compiler scope failed; process must exit")
    if dead_owner_graphs():
        raise RuntimeError(_DEAD_OWNER_ERROR)
    dead_frees = getattr(_library, "aimdo_full_proxy_dead_deferred_free_count", None)
    if dead_frees is not None and dead_frees():
        raise RuntimeError(_DEAD_OWNER_ERROR)
    drain_frees = getattr(_library, "aimdo_full_proxy_drain_deferred_frees", None)
    if drain_frees is not None and not drain_frees():
        raise RuntimeError("AIMDO deferred compiler free failed; process must exit")
    owner = threading.current_thread()
    with _graph_lock:
        pending = _deferred_graphs.pop(owner, [])
    for index, (library, handle) in enumerate(pending):
        terminal = False
        try:
            terminal = _destroy_graph_terminal(library, handle)
            if not terminal and _destroy_graph_checked(library, handle):
                continue
            terminal = terminal or _destroy_graph_terminal(library, handle)
        except Exception:
            pass
        with _graph_lock:
            _deferred_graphs.setdefault(owner, []).extend(pending[index:])
        if terminal:
            raise RuntimeError(
                "terminal AIMDO graph destroy error; process must exit"
            )
        raise RuntimeError("deferred AIMDO graph destroy failed on owner thread")
    return len(pending)


def graph_ownership_snapshot() -> dict[str, int]:
    """Count live diagnostic graph handles and deferred owner-thread closes."""
    with _graph_lock:
        return {
            "live": sum(bool(getattr(graph, "_handle", None)) for graph in _live_graphs),
            "deferred": sum(map(len, _deferred_graphs.values())),
        }


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
    library.aimdo_full_proxy_scoped_raw_snapshot.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
    ]
    library.aimdo_full_proxy_scoped_raw_snapshot.restype = ctypes.c_bool
    library.aimdo_full_proxy_is_compiler_owner.argtypes = [ctypes.c_void_p]
    library.aimdo_full_proxy_is_compiler_owner.restype = ctypes.c_bool
    library.aimdo_full_proxy_drain_deferred_frees.argtypes = []
    library.aimdo_full_proxy_drain_deferred_frees.restype = ctypes.c_bool
    library.aimdo_full_proxy_deferred_free_count.argtypes = []
    library.aimdo_full_proxy_deferred_free_count.restype = ctypes.c_uint64
    library.aimdo_full_proxy_dead_deferred_free_count.argtypes = []
    library.aimdo_full_proxy_dead_deferred_free_count.restype = ctypes.c_uint64

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


def scoped_raw_snapshot() -> list[int]:
    """Return live, allocation, release, failure and byte counts for raw workspaces."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    values = (ctypes.c_uint64 * 5)()
    if not _library.aimdo_full_proxy_scoped_raw_snapshot(values, len(values)):
        raise RuntimeError("scoped raw snapshot failed")
    return list(map(int, values))


def is_compiler_owner(pointer: int) -> bool:
    """Check whether a live XPU pointer is owned by the diagnostic compiler."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    return bool(_library.aimdo_full_proxy_is_compiler_owner(ctypes.c_void_p(int(pointer))))


def pending_compiler_frees() -> int:
    """Count compiler DataPtr releases awaiting their original graph thread."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    return int(_library.aimdo_full_proxy_deferred_free_count())


def dead_pending_compiler_frees() -> int:
    """Count pending compiler frees whose original native thread has exited."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    return int(_library.aimdo_full_proxy_dead_deferred_free_count())


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


def inject_destroy_release_failure(graph, stage: int) -> None:
    """Reject one mapped, physical or virtual release in a completed XPU graph."""
    if type(stage) is not int or stage not in (1, 2, 3):
        raise ValueError("destroy-release stage must be 1, 2 or 3")
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    if (control.lib is None or control.implementation != "xpu" or
            control.get_xpu_allocator_mode() != "native_hook" or
            getattr(graph, "_native_lib", None) is not control.lib or
            not getattr(graph, "_handle", None) or
            getattr(graph, "_owner_thread", None) is not threading.current_thread()):
        raise RuntimeError("destroy-release injection requires an owner-thread XPU graph")
    try:
        inject = control.lib.malloc_graph_test_fail_next_destroy_release
    except AttributeError as error:
        raise RuntimeError("destroy-release diagnostic export is missing") from error
    inject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    inject.restype = ctypes.c_bool
    if not inject(ctypes.c_void_p(int(graph._handle)), stage):
        raise RuntimeError("graph is not eligible for destroy-release injection")


def inject_driver_release_error(graph, stage: int, error_kind: int) -> None:
    """Return one OOM or device-lost error at a Level Zero release boundary."""
    if (type(stage) is not int or stage not in (1, 2, 3) or
            type(error_kind) is not int or error_kind not in (1, 2)):
        raise ValueError("driver-release stage must be 1-3 and error kind 1-2")
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    if (control.lib is None or control.implementation != "xpu" or
            control.get_xpu_allocator_mode() != "native_hook" or
            getattr(graph, "_native_lib", None) is not control.lib or
            not getattr(graph, "_handle", None) or
            getattr(graph, "_owner_thread", None) is not threading.current_thread()):
        raise RuntimeError("driver-release injection requires an owner-thread XPU graph")
    try:
        arm = control.lib.malloc_graph_test_arm_driver_release
    except AttributeError as error:
        raise RuntimeError("driver-release diagnostic export is missing") from error
    arm.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    arm.restype = ctypes.c_bool
    if not arm(ctypes.c_void_p(int(graph._handle)), stage, error_kind):
        raise RuntimeError("graph is not eligible for driver-release injection")


def pending_driver_release_error() -> int:
    """Return the unconsumed Level Zero release fault stage in this thread."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    pending = control.lib.aimdo_xpu_test_pending_vmm_release
    pending.argtypes = []
    pending.restype = ctypes.c_uint
    return int(pending())


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


@contextlib.contextmanager
def suspend_compiler_scope():
    """Route allocations through native Torch while an owner scope is suspended.

    This is private diagnostic state for a caller that pauses its memory graph.
    The caller must resume that graph before this context exits. Nested
    suspensions are allowed; nested compiler scopes remain unsupported.
    """
    global _scope_terminal
    if _scope_terminal:
        raise RuntimeError("native-owner compiler scope failed; process must exit")
    state = getattr(_scope_context, "active", None)
    if state is None or state["failed"]:
        raise RuntimeError("suspension requires an active native-owner compiler scope")
    if state["suspend_depth"] == 0:
        if not _library.aimdo_full_proxy_compiler_end():
            state["failed"] = True
            _scope_terminal = True
            raise RuntimeError("native-owner compiler suspension failed; process must exit")
        state["native_active"] = False
    state["suspend_depth"] += 1
    try:
        yield
    finally:
        state["suspend_depth"] -= 1
        if state["suspend_depth"] == 0:
            if _scope_terminal:
                state["failed"] = True
                raise RuntimeError("native-owner compiler scope failed; process must exit")
            if getattr(_scope_context, "active", None) is not state:
                state["failed"] = True
                _scope_terminal = True
                raise RuntimeError("native-owner compiler owner scope exited; process must exit")
            if not _library.aimdo_full_proxy_compiler_begin(
                state["size"], state["stream"], state["source_revision"]
            ):
                state["failed"] = True
                _scope_terminal = True
                raise RuntimeError("native-owner compiler resume failed; process must exit")
            state["native_active"] = True


@contextlib.contextmanager
def consumer_scope(tensor, stream):
    """Register a compiler tensor before queued work on another XPU stream.

    This opt-in diagnostic uses Torch's explicit record_stream contract. It
    covers only work submitted inside this scope or later on this registered
    stream; it cannot discover earlier unregistered consumers.
    """
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    drain_deferred_graphs()
    tensor_device = getattr(tensor, "device", None)
    stream_device = getattr(stream, "device", None)
    index = getattr(tensor_device, "index", None)
    if (getattr(tensor_device, "type", None) != "xpu" or
            getattr(stream_device, "type", None) != "xpu" or
            index is None or index != getattr(stream_device, "index", None) or
            not int(getattr(stream, "sycl_queue", 0))):
        raise ValueError("consumer scope requires the same indexed XPU device and queue")
    pointer = int(tensor.data_ptr())
    if not pointer or not is_compiler_owner(pointer):
        raise RuntimeError("consumer scope requires a live compiler-owned tensor")
    import torch

    tensor.record_stream(stream)  # Before yielding, so failures queue no consumer work.
    with torch.xpu.stream(stream):
        yield tensor


def record_diagnostic(stream, assert_graph_breaks: bool = False):
    """Create a memory-only XPU graph for the opt-in diagnostic route."""
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    from . import control

    if control.lib is None or control.implementation != "xpu" or \
            control.get_xpu_allocator_mode() != "native_hook":
        raise RuntimeError("native-owner graph requires active XPU native_hook")
    drain_deferred_graphs()
    if not hasattr(control.lib, "malloc_graph_destroy_checked"):
        raise RuntimeError("native-owner checked graph destroy export is missing")
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
    graph = None
    try:
        graph = MallocGraph(
            handle, stream, owner_thread=threading.current_thread(),
            native_lib=control.lib,
        )
        _register_graph(graph)
    except BaseException:
        if graph is not None:
            graph._handle = None
        if not _destroy_graph_checked(control.lib, handle):
            _defer_graph_destroy(threading.current_thread(), control.lib, handle)
        raise
    return graph


@contextlib.contextmanager
def _compiler_scope(size: int, stream):
    global _scope_terminal
    if _scope_terminal:
        raise RuntimeError("native-owner compiler scope failed; process must exit")
    if getattr(_scope_context, "active", None) is not None:
        raise RuntimeError("nested native-owner compiler scopes are unsupported")
    if not installed():
        raise RuntimeError("native-owner diagnostic is not installed")
    drain_deferred_graphs()
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
    revision_bytes = source_revision.encode("ascii")
    if not _library.aimdo_full_proxy_compiler_begin(
        size, stream_pointer, revision_bytes
    ):
        raise RuntimeError("native-owner compiler scope could not start")
    state = {"size": size, "stream": stream_pointer,
             "source_revision": revision_bytes, "suspend_depth": 0,
             "native_active": True, "failed": False}
    _scope_context.active = state
    try:
        yield
    finally:
        _scope_context.active = None
        if state["failed"] or state["suspend_depth"] or not state["native_active"]:
            _scope_terminal = True
            raise RuntimeError("native-owner compiler scope lost; process must exit")
        if not _library.aimdo_full_proxy_compiler_end():
            _scope_terminal = True
            raise RuntimeError("native-owner compiler scope could not end; process must exit")
