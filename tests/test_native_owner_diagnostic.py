"""Ownership and lifecycle guards for the opt-in native-owner sidecar."""

import contextlib
import sys
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


def test_late_xpu_init_rejected_before_dso_load(monkeypatch):
    monkeypatch.setattr(native_owner, "_library", None)
    monkeypatch.setattr(native_owner.ctypes, "CDLL", lambda *a, **k: pytest.fail("DSO loaded"))
    late = SimpleNamespace(__version__="2.14.0+xpu",
                           xpu=SimpleNamespace(is_initialized=lambda: True))
    with pytest.raises(RuntimeError, match="before XPU initialization"):
        native_owner.install(late)


def test_release_policy_and_library_fingerprints_are_owned_by_the_caller(tmp_path, monkeypatch):
    (tmp_path / "aimdo_xpu_native_owner.so").write_bytes(b"fake-sidecar")
    monkeypatch.setattr(native_owner, "__file__", str(tmp_path / "native_owner.py"))
    monkeypatch.setattr(native_owner, "_library", None)
    monkeypatch.setattr(native_owner.platform, "system", lambda: "Linux")
    class ReachedLoader(Exception):
        pass
    def load(*args, **kwargs):
        raise ReachedLoader
    monkeypatch.setattr(native_owner.ctypes, "CDLL", load)
    # No JSON or Torch-library tree exists. AIMDO retains lifecycle guards;
    # its caller decides which declared release to support before calling it.
    torch_module = SimpleNamespace(__version__="2.13.0+xpu",
                                   xpu=SimpleNamespace(is_initialized=lambda: False))
    with pytest.raises(ReachedLoader):
        native_owner.install(torch_module)


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


def test_compiler_scope_suspension_restores_native_binding_after_nested_pause(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_compiler_begin=lambda size, stream, revision: (
            calls.append(("begin", size, stream, revision)) or True),
        aimdo_full_proxy_compiler_end=lambda: calls.append(("end",)) or True,
    )
    monkeypatch.setattr(native_owner, "_library", fake)
    monkeypatch.setattr(native_owner, "_scope_context", threading.local())
    monkeypatch.setattr(native_owner, "_scope_terminal", False)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(control, "lib", SimpleNamespace())
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_allocator_mode", lambda: "native_hook")
    monkeypatch.setattr(control, "_xpu_native_owner_context_ready", lambda: True)
    monkeypatch.setattr(control, "get_memory_compiler_capability",
                        lambda: {"source_revision": "exact-source"})
    stream = SimpleNamespace(sycl_queue=456)
    with pytest.raises(RuntimeError, match="requires an active"):
        with native_owner.suspend_compiler_scope():
            pytest.fail("suspended without a compiler scope")
    with native_owner.compiler_scope(stream):
        with native_owner.suspend_compiler_scope():
            with pytest.raises(RuntimeError, match="no live allocations or scopes"):
                with native_owner.allocator_transition():
                    pytest.fail("lifecycle changed during a suspended compiler scope")
            with native_owner.suspend_compiler_scope():
                with pytest.raises(RuntimeError, match="nested native-owner"):
                    with native_owner.compiler_scope(stream):
                        pytest.fail("nested compiler scope entered")
        with pytest.raises(RuntimeError, match="nested native-owner"):
            with native_owner.compiler_scope(stream):
                pytest.fail("nested compiler scope entered")
    assert calls == [("begin", 0, 456, b"exact-source"), ("end",),
                     ("begin", 0, 456, b"exact-source"), ("end",)]
    assert native_owner._scope_terminal is False


def test_compiler_scope_suspension_exception_restores_route(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_compiler_begin=lambda *args: calls.append(("begin", *args)) or True,
        aimdo_full_proxy_compiler_end=lambda: calls.append(("end",)) or True,
    )
    monkeypatch.setattr(native_owner, "_library", fake)
    monkeypatch.setattr(native_owner, "_scope_context", threading.local())
    monkeypatch.setattr(native_owner, "_scope_terminal", False)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(control, "lib", SimpleNamespace())
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_allocator_mode", lambda: "native_hook")
    monkeypatch.setattr(control, "_xpu_native_owner_context_ready", lambda: True)
    monkeypatch.setattr(control, "get_memory_compiler_capability",
                        lambda: {"source_revision": "exact-source"})
    with pytest.raises(KeyError, match="cancel"):
        with native_owner.selected_scope(4097, SimpleNamespace(sycl_queue=456)):
            with native_owner.suspend_compiler_scope():
                raise KeyError("cancel")
    assert calls == [("begin", 4097, 456, b"exact-source"), ("end",),
                     ("begin", 4097, 456, b"exact-source"), ("end",)]
    assert native_owner._scope_terminal is False


def test_failed_compiler_scope_resume_is_process_terminal(monkeypatch):
    calls = []
    real_drain = native_owner.drain_deferred_graphs

    def begin(*args):
        calls.append("begin")
        return len(calls) == 1

    fake = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_compiler_begin=begin,
        aimdo_full_proxy_compiler_end=lambda: calls.append("end") or True,
    )
    monkeypatch.setattr(native_owner, "_library", fake)
    monkeypatch.setattr(native_owner, "_scope_context", threading.local())
    monkeypatch.setattr(native_owner, "_scope_terminal", False)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(control, "lib", SimpleNamespace())
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_allocator_mode", lambda: "native_hook")
    monkeypatch.setattr(control, "_xpu_native_owner_context_ready", lambda: True)
    monkeypatch.setattr(control, "get_memory_compiler_capability",
                        lambda: {"source_revision": "exact-source"})
    stream = SimpleNamespace(sycl_queue=456)
    with pytest.raises(RuntimeError, match="process must exit"):
        with native_owner.compiler_scope(stream):
            with native_owner.suspend_compiler_scope():
                pass
    assert calls == ["begin", "end", "begin"]
    assert native_owner._scope_terminal is True
    with pytest.raises(RuntimeError, match="process must exit"):
        with native_owner.compiler_scope(stream):
            pytest.fail("terminal proxy scope was reused")
    with pytest.raises(RuntimeError, match="process must exit"):
        real_drain()


def test_paused_graph_scope_restores_graph_and_route_after_nested_cancellation(monkeypatch):
    calls = []
    proxy = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_compiler_begin=lambda *args: calls.append(("begin", *args)) or True,
        aimdo_full_proxy_compiler_end=lambda: calls.append(("end",)) or True,
    )
    core = SimpleNamespace()
    stream = SimpleNamespace(sycl_queue=456)
    graph = SimpleNamespace(
        _native_lib=core, _handle=123,
        _owner_thread=threading.current_thread(), _stream=stream,
        pause=lambda *, sync: calls.append(("pause", sync)),
        resume=lambda *, sync: calls.append(("resume", sync)),
    )
    other = SimpleNamespace(**{**vars(graph), "_handle": 789})
    monkeypatch.setattr(native_owner, "_library", proxy)
    monkeypatch.setattr(native_owner, "_scope_context", threading.local())
    monkeypatch.setattr(native_owner, "_scope_terminal", False)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(control, "lib", core)
    monkeypatch.setattr(control, "_xpu_native_owner_context_ready", lambda: True)
    monkeypatch.setattr(control, "get_memory_compiler_capability",
                        lambda: {"source_revision": "exact-source"})
    with pytest.raises(KeyError, match="cancel"):
        with native_owner.compiler_scope(stream):
            with native_owner.paused_graph_scope(graph, sync=True):
                with native_owner.paused_graph_scope(graph, sync=True):
                    calls.append(("ordinary-work",))
                with pytest.raises(RuntimeError, match="same graph and sync mode"):
                    with native_owner.paused_graph_scope(graph, sync=False):
                        pytest.fail("mismatched sync mode entered")
                with pytest.raises(RuntimeError, match="same graph and sync mode"):
                    with native_owner.paused_graph_scope(other, sync=True):
                        pytest.fail("different graph entered")
                raise KeyError("cancel")
    assert calls == [("begin", 0, 456, b"exact-source"), ("end",),
                     ("pause", True), ("ordinary-work",), ("resume", True),
                     ("begin", 0, 456, b"exact-source"), ("end",)]
    assert native_owner._scope_terminal is False


def test_paused_graph_scope_failed_resume_is_process_terminal(monkeypatch):
    calls = []
    proxy = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_compiler_begin=lambda *args: calls.append("begin") or True,
        aimdo_full_proxy_compiler_end=lambda: calls.append("end") or True,
    )
    core = SimpleNamespace()
    stream = SimpleNamespace(sycl_queue=456)

    def failed_resume(*, sync):
        calls.append("resume")
        raise RuntimeError("graph resume failed")

    graph = SimpleNamespace(
        _native_lib=core, _handle=123,
        _owner_thread=threading.current_thread(), _stream=stream,
        pause=lambda *, sync: calls.append("pause"), resume=failed_resume,
    )
    monkeypatch.setattr(native_owner, "_library", proxy)
    monkeypatch.setattr(native_owner, "_scope_context", threading.local())
    monkeypatch.setattr(native_owner, "_scope_terminal", False)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(control, "lib", core)
    monkeypatch.setattr(control, "_xpu_native_owner_context_ready", lambda: True)
    monkeypatch.setattr(control, "get_memory_compiler_capability",
                        lambda: {"source_revision": "exact-source"})
    with pytest.raises(RuntimeError, match="process must exit"):
        with native_owner.compiler_scope(stream):
            with native_owner.paused_graph_scope(graph):
                pass
    assert calls == ["begin", "end", "pause", "resume"]
    assert native_owner._scope_terminal is True


def test_consumer_scope_registers_before_work_and_restores_stream(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def select(stream):
        calls.append(("enter", stream.sycl_queue))
        try:
            yield
        finally:
            calls.append(("exit", stream.sycl_queue))

    device = SimpleNamespace(type="xpu", index=0)
    stream = SimpleNamespace(device=device, sycl_queue=456)
    tensor = SimpleNamespace(device=device, data_ptr=lambda: 127,
                             untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 123),
                             record_stream=lambda value: calls.append(
                                 ("record", value.sycl_queue)))
    monkeypatch.setattr(native_owner, "installed", lambda: True)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs",
                        lambda: calls.append(("drain", 0)))
    monkeypatch.setattr(native_owner, "is_compiler_owner",
                        lambda pointer: pointer == 123)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        xpu=SimpleNamespace(stream=select)))
    with native_owner.consumer_scope(tensor, stream) as selected:
        assert selected is tensor
        calls.append(("work", stream.sycl_queue))
    assert calls == [("drain", 0), ("record", 456), ("enter", 456),
                     ("work", 456), ("exit", 456)]
    calls.clear()
    with pytest.raises(KeyError):
        with native_owner.consumer_scope(tensor, stream):
            raise KeyError("cancel after work was queued")
    assert calls == [("drain", 0), ("record", 456), ("enter", 456),
                     ("exit", 456)]
    calls.clear()
    tensor.record_stream = lambda value: (_ for _ in ()).throw(
        RuntimeError("registration failed"))
    with pytest.raises(RuntimeError, match="registration failed"):
        with native_owner.consumer_scope(tensor, stream):
            pytest.fail("consumer work began before registration succeeded")
    assert calls == [("drain", 0)]


def test_consumer_scope_rejects_wrong_device_and_unowned_tensor(monkeypatch):
    calls = []
    monkeypatch.setattr(native_owner, "installed", lambda: True)
    monkeypatch.setattr(native_owner, "drain_deferred_graphs", lambda: 0)
    monkeypatch.setattr(native_owner, "is_compiler_owner",
                        lambda pointer: pointer == 123)
    tensor = SimpleNamespace(device=SimpleNamespace(type="xpu", index=0),
                             data_ptr=lambda: 999,
                             untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 999),
                             record_stream=lambda stream: calls.append(stream))
    wrong = SimpleNamespace(device=SimpleNamespace(type="xpu", index=1),
                            sycl_queue=456)
    with pytest.raises(ValueError, match="same indexed XPU device"):
        with native_owner.consumer_scope(tensor, wrong):
            pytest.fail("wrong-device consumer entered the scope")
    correct = SimpleNamespace(device=tensor.device, sycl_queue=456)
    with pytest.raises(RuntimeError, match="live compiler-owned tensor"):
        with native_owner.consumer_scope(tensor, correct):
            pytest.fail("unowned tensor entered the scope")
    tensor.untyped_storage = lambda: None
    with pytest.raises(ValueError, match="requires a tensor with XPU storage"):
        with native_owner.consumer_scope(tensor, correct):
            pytest.fail("missing storage entered the scope")
    assert calls == []


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


@pytest.mark.parametrize("owner_action", ("close", "abort", "pop"))
def test_owner_cleanup_after_foreign_close_drains_queued_handle(monkeypatch, owner_action):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    owner = threading.current_thread()
    released = []

    def destroy(handle):
        released.append((threading.current_thread(), handle.value))
        return True

    library = SimpleNamespace(
        malloc_graph_destroy_checked=destroy,
        malloc_graph_abort=lambda handle: pytest.fail("closed handle was aborted"),
        malloc_graph_pop=lambda handle: pytest.fail("closed handle was popped"),
    )
    monkeypatch.setattr(control, "lib", library)
    graph = malloc_graph.MallocGraph(4321, None, owner_thread=owner, native_lib=library)
    native_owner._register_graph(graph)
    graph._scopes.append("cancelled")
    worker = threading.Thread(target=graph.close)
    worker.start()
    worker.join()
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 1}
    assert released == []

    if owner_action == "pop":
        with pytest.raises(RuntimeError, match="graph is closed"):
            graph.pop()
    elif owner_action == "abort":
        assert graph.abort() is None
    else:
        assert graph.close() is True
    assert released == [(owner, 4321)]
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 0}
    if owner_action == "abort":
        assert graph._scopes == []


def test_owner_close_retry_retains_failed_deferred_graph(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    owner = threading.current_thread()
    calls = []

    def destroy(handle):
        calls.append((threading.current_thread(), handle.value))
        return len(calls) == 2

    graph = malloc_graph.MallocGraph(
        8765, None, owner_thread=owner,
        native_lib=SimpleNamespace(malloc_graph_destroy_checked=destroy),
    )
    native_owner._register_graph(graph)
    worker = threading.Thread(target=graph.close)
    worker.start()
    worker.join()
    with pytest.raises(RuntimeError, match="deferred AIMDO graph destroy failed"):
        graph.close()
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 1}
    assert graph.close() is True
    assert calls == [(owner, 8765), (owner, 8765)]
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 0}


def test_deferred_compiler_free_drains_before_graph_destroy(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    calls = []
    library = SimpleNamespace(malloc_graph_destroy_checked=lambda handle:
                              calls.append(("graph", handle.value)) or True)
    proxy = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_drain_deferred_frees=lambda: calls.append(("tensor", 1)) or True,
        aimdo_full_proxy_deferred_free_count=lambda: 1,
        aimdo_full_proxy_dead_deferred_free_count=lambda: 0,
    )
    monkeypatch.setattr(native_owner, "_library", proxy)
    native_owner._defer_graph_destroy(threading.current_thread(), library, 123)
    assert native_owner.pending_compiler_frees() == 1
    assert native_owner.drain_deferred_graphs() == 1
    assert calls == [("tensor", 1), ("graph", 123)]


def test_dead_native_owner_blocks_work_without_python_graph_handle(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    fake = SimpleNamespace(
        aimdo_full_proxy_is_installed=lambda: True,
        aimdo_full_proxy_dead_deferred_free_count=lambda: 1,
        aimdo_full_proxy_drain_deferred_frees=lambda: pytest.fail(
            "dead native owner was drained by a different thread"),
    )
    monkeypatch.setattr(native_owner, "_library", fake)
    assert native_owner.dead_owner_graphs() == 0
    assert native_owner.dead_pending_compiler_frees() == 1
    with pytest.raises(RuntimeError, match="owner thread exited; process must exit"):
        native_owner.drain_deferred_graphs()
    monkeypatch.setattr(control, "lib", SimpleNamespace())
    monkeypatch.setattr(control, "implementation", "xpu")
    with pytest.raises(RuntimeError, match="owner thread exited; process must exit"):
        control.deinit()


def test_failed_deferred_compiler_free_blocks_graph_destroy(monkeypatch):
    monkeypatch.setattr(native_owner, "_graph_lock", threading.Lock())
    monkeypatch.setattr(native_owner, "_live_graphs", weakref.WeakSet())
    monkeypatch.setattr(native_owner, "_deferred_graphs", {})
    calls = []
    library = SimpleNamespace(malloc_graph_destroy_checked=lambda handle:
                              calls.append(handle.value) or True)
    proxy = SimpleNamespace(aimdo_full_proxy_drain_deferred_frees=lambda: False)
    monkeypatch.setattr(native_owner, "_library", proxy)
    native_owner._defer_graph_destroy(threading.current_thread(), library, 456)
    with pytest.raises(RuntimeError, match="deferred compiler free failed; process must exit"):
        native_owner.drain_deferred_graphs()
    assert calls == []
    assert native_owner.graph_ownership_snapshot() == {"live": 0, "deferred": 1}


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
