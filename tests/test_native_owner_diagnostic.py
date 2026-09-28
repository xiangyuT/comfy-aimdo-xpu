"""Fail-closed admission for the opt-in Torch 2.14 native-owner sidecar."""

import threading
from types import SimpleNamespace
import weakref

import pytest

from comfy_aimdo import control, malloc_graph, native_owner


def test_invalid_diagnostic_flag_rejected(monkeypatch):
    monkeypatch.setenv("AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        native_owner.requested()


def test_global_mode_cannot_install_native_owner(monkeypatch):
    monkeypatch.setenv("AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC", "1")
    monkeypatch.setattr(control, "lib", None)
    with pytest.raises(RuntimeError, match="requires the XPU native_hook mode"):
        control.init(implementation="xpu", xpu_allocator_mode="global")
    assert control.lib is None


def test_wrong_torch_and_late_xpu_init_rejected_before_dso_load(monkeypatch):
    monkeypatch.setattr(native_owner, "_library", None)
    monkeypatch.setattr(native_owner.ctypes, "CDLL", lambda *a, **k: pytest.fail("DSO loaded"))
    wrong = SimpleNamespace(__version__="2.13.0+xpu",
                            xpu=SimpleNamespace(is_initialized=lambda: False))
    with pytest.raises(RuntimeError, match="requires Torch 2.14.0"):
        native_owner.install(wrong)
    late = SimpleNamespace(__version__="2.14.0+xpu",
                           xpu=SimpleNamespace(is_initialized=lambda: True))
    with pytest.raises(RuntimeError, match="before XPU initialization"):
        native_owner.install(late)


def test_process_lifetime_proxy_cannot_be_disabled(monkeypatch):
    fake = SimpleNamespace(aimdo_full_proxy_is_installed=lambda: True)
    monkeypatch.setattr(native_owner, "_library", fake)
    monkeypatch.setenv("AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC", "0")
    monkeypatch.setattr(control, "lib", None)
    with pytest.raises(RuntimeError, match="cannot be disabled in this process"):
        control.init(implementation="xpu", xpu_allocator_mode="native_hook")
    assert control.lib is None


def test_unrestricted_scope_still_requires_opt_in_installation(monkeypatch):
    monkeypatch.setattr(native_owner, "_library", None)
    stream = SimpleNamespace(sycl_queue=1)
    with pytest.raises(RuntimeError, match="not installed"):
        with native_owner.compiler_scope(stream):
            pytest.fail("uninstalled proxy entered a compiler scope")
    with pytest.raises(ValueError, match="must be positive"):
        native_owner.selected_scope(0, stream)


def test_diagnostic_graph_requires_installed_proxy(monkeypatch):
    monkeypatch.setattr(native_owner, "_library", None)
    stream = SimpleNamespace(device=SimpleNamespace(type="xpu", index=0), sycl_queue=1234)
    with pytest.raises(RuntimeError, match="not installed"):
        native_owner.record_diagnostic(stream)


def test_page_create_oom_injection_is_bounded_and_requires_active_graph(monkeypatch):
    calls = []

    class Inject:
        def __call__(self, attempts):
            calls.append(attempts)
            return attempts == 1

    inject = Inject()
    monkeypatch.setattr(native_owner, "_library",
                        SimpleNamespace(aimdo_full_proxy_is_installed=lambda: True))
    monkeypatch.setattr(control, "lib",
                        SimpleNamespace(malloc_graph_test_fail_next_page_creates=inject))
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_allocator_mode", lambda: "native_hook")
    for value in (0, 3, True, 1.0):
        with pytest.raises(ValueError, match="must be 1 or 2"):
            native_owner.inject_page_create_oom_attempts(value)
    native_owner.inject_page_create_oom_attempts(1)
    with pytest.raises(RuntimeError, match="no eligible active graph"):
        native_owner.inject_page_create_oom_attempts(2)
    assert calls == [1, 2]
    assert inject.argtypes == [native_owner.ctypes.c_uint]
    assert inject.restype == native_owner.ctypes.c_bool


def test_destroy_release_injection_is_owner_thread_and_stage_bound(monkeypatch):
    calls = []

    def inject(handle, stage):
        calls.append((handle.value, stage))
        return stage == 1

    library = SimpleNamespace(malloc_graph_test_fail_next_destroy_release=inject)
    graph = SimpleNamespace(_native_lib=library, _handle=123,
                            _owner_thread=threading.current_thread())
    monkeypatch.setattr(native_owner, "_library",
                        SimpleNamespace(aimdo_full_proxy_is_installed=lambda: True))
    monkeypatch.setattr(control, "lib", library)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_allocator_mode", lambda: "native_hook")
    for value in (0, 4, True, 1.0):
        with pytest.raises(ValueError, match="must be 1, 2 or 3"):
            native_owner.inject_destroy_release_failure(graph, value)
    native_owner.inject_destroy_release_failure(graph, 1)
    with pytest.raises(RuntimeError, match="not eligible"):
        native_owner.inject_destroy_release_failure(graph, 2)
    graph._owner_thread = threading.Thread()
    with pytest.raises(RuntimeError, match="requires an owner-thread"):
        native_owner.inject_destroy_release_failure(graph, 1)
    assert calls == [(123, 1), (123, 2)]
    assert inject.argtypes == [native_owner.ctypes.c_void_p,
                               native_owner.ctypes.c_uint]


def test_driver_release_error_injection_requires_completed_owner_graph(monkeypatch):
    calls = []

    def arm(handle, stage, error_kind):
        calls.append((handle.value, stage, error_kind))
        return (stage, error_kind) == (1, 2)

    def pending():
        return 3

    library = SimpleNamespace(
        malloc_graph_test_arm_driver_release=arm,
        aimdo_xpu_test_pending_vmm_release=pending,
    )
    graph = SimpleNamespace(_native_lib=library, _handle=456,
                            _owner_thread=threading.current_thread())
    monkeypatch.setattr(native_owner, "_library",
                        SimpleNamespace(aimdo_full_proxy_is_installed=lambda: True))
    monkeypatch.setattr(control, "lib", library)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_allocator_mode", lambda: "native_hook")
    for stage, kind in ((0, 1), (4, 1), (1, 0), (1, 3), (True, 1), (1, True)):
        with pytest.raises(ValueError, match="stage must be 1-3"):
            native_owner.inject_driver_release_error(graph, stage, kind)
    native_owner.inject_driver_release_error(graph, 1, 2)
    with pytest.raises(RuntimeError, match="not eligible"):
        native_owner.inject_driver_release_error(graph, 2, 1)
    assert native_owner.pending_driver_release_error() == 3
    graph._owner_thread = threading.Thread()
    with pytest.raises(RuntimeError, match="requires an owner-thread"):
        native_owner.inject_driver_release_error(graph, 1, 2)
    assert calls == [(456, 1, 2), (456, 2, 1)]
    assert arm.argtypes == [native_owner.ctypes.c_void_p,
                            native_owner.ctypes.c_uint,
                            native_owner.ctypes.c_uint]
    assert pending.restype == native_owner.ctypes.c_uint


def test_scoped_raw_owner_diagnostics(monkeypatch):
    def snapshot(values, count):
        assert count == 5
        for index, value in enumerate((0, 2, 2, 0, 0)):
            values[index] = value
        return True

    fake = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_scoped_raw_snapshot=snapshot,
        aimdo_full_proxy_is_compiler_owner=lambda pointer: pointer.value == 0x1234,
    )
    monkeypatch.setattr(native_owner, "_library", fake)
    assert native_owner.scoped_raw_snapshot() == [0, 2, 2, 0, 0]
    assert native_owner.is_compiler_owner(0x1234)
    assert not native_owner.is_compiler_owner(0x5678)


def test_foreign_thread_graph_close_is_drained_by_owner(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    owner = threading.current_thread()
    calls = []

    def destroy(handle):
        calls.append((threading.current_thread(), handle.value))
        return True

    library = SimpleNamespace(malloc_graph_destroy_checked=destroy)
    graph = malloc_graph.MallocGraph(99, None, owner_thread=owner, native_lib=library)
    native_owner._register_graph(graph)
    results = []
    worker = threading.Thread(target=lambda: results.append(graph.close()))
    worker.start()
    worker.join()
    assert results == [False]
    assert calls == []
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 1}
    assert native_owner.drain_deferred_graphs() == 1
    assert calls == [(owner, 99)]
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 0}


def test_dead_owner_close_is_terminal_without_duplicate_queue_entry(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    dead = threading.Thread()
    assert not dead.is_alive()
    calls = []

    def destroy(handle):
        calls.append(handle.value)
        return True

    graph = malloc_graph.MallocGraph(
        321, None, owner_thread=dead,
        native_lib=SimpleNamespace(malloc_graph_destroy_checked=destroy),
    )
    native_owner._register_graph(graph)
    assert native_owner.dead_owner_graphs() == 1
    with pytest.raises(RuntimeError, match="owner thread exited; process must exit"):
        graph.close()
    assert graph._handle is None
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 1}
    graph.__del__()
    assert native_owner.dead_owner_graphs() == 1
    with pytest.raises(RuntimeError, match="owner thread exited; process must exit"):
        native_owner.drain_deferred_graphs()
    assert calls == []
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 1}


def test_dead_graph_owner_blocks_deinit_before_native_cleanup(monkeypatch):
    library = SimpleNamespace()
    monkeypatch.setattr(control, "lib", library)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_ready", False)
    monkeypatch.setattr(native_owner, "installed", lambda: True)
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs",
                        {threading.Thread(): [(library, 654)]})
    with pytest.raises(RuntimeError, match="owner thread exited; process must exit"):
        control.deinit()
    assert control.lib is library


def test_failed_checked_graph_close_keeps_handle_for_retry(monkeypatch):
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    outcomes = iter((False, True))

    def destroy(handle):
        assert handle.value == 123
        return next(outcomes)

    graph = malloc_graph.MallocGraph(
        123, None, owner_thread=threading.current_thread(),
        native_lib=SimpleNamespace(malloc_graph_destroy_checked=destroy),
    )
    with pytest.raises(RuntimeError, match="graph destroy failed"):
        graph.close()
    assert graph._handle == 123
    assert graph.close() is True
    assert graph._handle is None


def test_terminal_graph_close_preserves_handle_without_second_native_release(monkeypatch):
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    calls = []

    def destroy(handle):
        calls.append(handle.value)
        return False

    def terminal(handle):
        assert handle.value == 777
        return bool(calls)

    graph = malloc_graph.MallocGraph(
        777, None, owner_thread=threading.current_thread(),
        native_lib=SimpleNamespace(malloc_graph_destroy_checked=destroy,
                                   malloc_graph_destroy_terminal=terminal),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="terminal.*process must exit"):
            graph.close()
        assert graph._handle == 777
    assert calls == [777]
    graph._handle = None  # The fake native handle has no resource to release.


def test_terminal_deferred_destroy_is_retained_without_native_retry(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    calls = []

    def destroy(handle):
        calls.append(handle.value)
        return False

    def terminal(handle):
        assert handle.value == 888
        return True

    library = SimpleNamespace(malloc_graph_destroy_checked=destroy,
                              malloc_graph_destroy_terminal=terminal)
    native_owner._defer_graph_destroy(threading.current_thread(), library, 888)
    with pytest.raises(RuntimeError, match="terminal.*process must exit"):
        native_owner.drain_deferred_graphs()
    assert calls == []
    assert native_owner.graph_ownership_snapshot()["deferred"] == 1


def test_deinit_rejects_live_diagnostic_graph_before_native_cleanup(monkeypatch):
    library = SimpleNamespace()
    monkeypatch.setattr(control, "lib", library)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_ready", False)
    monkeypatch.setattr(native_owner, "installed", lambda: True)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(native_owner, "graph_ownership_snapshot",
                        lambda: {"live": 1, "deferred": 0})
    with pytest.raises(RuntimeError, match="diagnostic graphs remain live"):
        control.deinit()
    assert control.lib is library


def test_deinit_rejects_rogue_native_owner_after_graph_close(monkeypatch):
    library = SimpleNamespace()
    monkeypatch.setattr(control, "lib", library)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_ready", False)
    monkeypatch.setattr(native_owner, "installed", lambda: True)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(native_owner, "graph_ownership_snapshot",
                        lambda: {"live": 0, "deferred": 0})
    monkeypatch.setattr(native_owner, "snapshot", lambda: [1] + [0] * 16)
    monkeypatch.setattr(native_owner, "scoped_raw_snapshot", lambda: [0] * 5)
    with pytest.raises(RuntimeError, match="native-owner allocations remain live"):
        control.deinit()
    assert control.lib is library


def test_graph_stream_switch_uses_xpu_queue_pointer(monkeypatch):
    calls = []
    native = SimpleNamespace(
        malloc_graph_set_stream=lambda handle, queue: calls.append(
            (handle, queue.value)) or True,
        malloc_graph_destroy=lambda handle: calls.append(("destroy", handle)),
    )
    monkeypatch.setattr(control, "lib", native)
    monkeypatch.setattr(control, "implementation", "xpu")
    first = SimpleNamespace(sycl_queue=1234)
    second = SimpleNamespace(sycl_queue=5678)
    graph = malloc_graph.MallocGraph(99, first)
    with graph.use_stream(second):
        assert graph._stream is second
    assert graph._stream is first
    graph.__del__()
    assert calls == [(99, 5678), (99, 1234), ("destroy", 99)]
