"""Fail-closed admission for the opt-in Torch 2.14 native-owner sidecar."""

from types import SimpleNamespace

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
