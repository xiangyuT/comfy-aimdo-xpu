"""Native cache ownership and bounded recovery at model-owner boundaries."""
from types import SimpleNamespace

import pytest
import torch

from comfy_aimdo import control, model_vbar


@pytest.fixture
def native(monkeypatch):
    state = {"reserved": 64, "allocated": 32, "released": 32, "flushes": 0, "generation": 0}
    hints = []

    def stats(device=None):
        return {"reserved_bytes.all.current": state["reserved"],
                "allocated_bytes.all.current": state["allocated"],
                "active_bytes.all.current": state["allocated"],
                "active_bytes.all.peak": 96,
                "reserved_bytes.all.peak": 128, "allocated_bytes.all.peak": 96}

    def flush():
        state["flushes"] += 1
        state["reserved"] -= state["released"]

    monkeypatch.setattr(control, "lib", SimpleNamespace())
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_ready", True)
    monkeypatch.setattr(control, "_xpu_allocator_mode", "native_hook")
    monkeypatch.setattr(control.platform, "system", lambda: "Linux")
    monkeypatch.setattr(control, "publish_torch_cached_bytes", lambda dev, value=None: hints.append(value))
    monkeypatch.setattr(control, "get_xpu_ur_hook_stats", lambda: {"tracked_alloc_calls": state["generation"]})
    monkeypatch.setattr(model_vbar.sys, "platform", "linux")
    monkeypatch.setattr(model_vbar, "_native_cache_trim_enabled", True)
    monkeypatch.setattr(model_vbar, "_linux_native_cache_trim_signature", {})
    monkeypatch.setattr(torch.xpu, "memory_stats", stats)
    monkeypatch.setattr(torch.xpu, "empty_cache", flush)
    return state, hints


def test_helpers_follow_linux_native_owner(native, monkeypatch):
    state, _ = native
    assert control.get_xpu_allocator_memory_stats(0) == (32, 64, 96, 128)
    assert control.empty_xpu_allocator_cache(wait=True)
    assert state["reserved"] == 32
    peaks = []
    monkeypatch.setattr(torch.xpu, "reset_peak_memory_stats", lambda dev: peaks.append(dev))
    control.reset_xpu_allocator_peak_stats(0)
    assert peaks == [0]


def test_retry_only_after_actual_reserved_bytes_returned(native):
    state, hints = native
    assert model_vbar._release_native_cache(0)
    assert state["flushes"] == 1
    assert hints == [32, 0]
    assert not model_vbar._release_native_cache(0)
    assert state["flushes"] == 1


def test_unreleasable_cache_is_not_repeated_without_state_change(native):
    state, _ = native
    state["released"] = 0
    assert not model_vbar._release_native_cache(0)
    assert not model_vbar._release_native_cache(0)
    assert state["flushes"] == 1
    state["generation"] += 1
    assert not model_vbar._release_native_cache(0)
    assert state["flushes"] == 2


@pytest.mark.parametrize("mode,enabled", [("global", True), ("native_hook", False)])
def test_linux_trim_respects_owner_and_switch(native, monkeypatch, mode, enabled):
    state, _ = native
    monkeypatch.setattr(control, "_xpu_allocator_mode", mode)
    monkeypatch.setattr(model_vbar, "_native_cache_trim_enabled", enabled)
    assert not model_vbar._release_native_cache(0)
    assert state["flushes"] == 0


def test_failed_stats_do_not_replace_vbar_oom(native, monkeypatch):
    def fail(device):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(torch.xpu, "memory_stats", fail)
    assert not model_vbar._release_native_cache(0)


@pytest.mark.parametrize("watermark,released", [(1, 32), (0, 32), (1, 0)])
def test_fault_retries_with_original_watermark_only_after_release(native, monkeypatch, watermark, released):
    state, _ = native
    state["released"] = released
    calls = []
    current = [watermark]
    def fault(*args):
        calls.append(("fault", current[0]))
        if len(calls) == 1:
            current[0] = 0
            return 1
        return 0 if current[0] else 1
    def restore(ctx, ptr, size):
        calls.append(("restore", size))
        current[0] = size // (32 * 1024 ** 2)
        return True
    monkeypatch.setattr(model_vbar, "lib", SimpleNamespace(
        vbar_get_watermark=lambda *args: current[0], vbar_fault=fault,
        vbar_set_watermark=restore))
    monkeypatch.setattr(control, "capture_xpu_oom_snapshot", lambda *args, **kwargs: None)
    vbar = SimpleNamespace(base_addr=4096, device=0, _devctx=1, _ptr=2)
    vbar.set_watermark = lambda size: model_vbar.ModelVBAR.set_watermark(vbar, size)
    result = model_vbar.ModelVBAR.fault(vbar, 4096, 32 * 1024 ** 2)
    assert (result is not None) == bool(watermark and released)
    assert calls == ([("fault", watermark), ("restore", watermark * 32 * 1024 ** 2),
                      ("fault", watermark)] if released else [("fault", watermark)])


@pytest.mark.parametrize("registered", [True, False])
def test_linux_native_unpin_registers_queue_before_releasing_pin(native, monkeypatch, registered):
    calls = []
    def register(queue, device):
        calls.append(("register", queue, device))
        return registered
    lib = SimpleNamespace(aimdo_xpu_register_consumer_queue=register,
        vbar_unpin=lambda *args: (calls.append(("unpin", *args)) or True))
    monkeypatch.setattr(model_vbar, "lib", lib)
    monkeypatch.setattr(model_vbar, "_consumer_queue_ptr", lambda device, stream: 1234)
    vbar = SimpleNamespace(base_addr=4096, device=0, _devctx=1, _ptr=2)
    if registered:
        model_vbar.ModelVBAR.unpin(vbar, 4096, 512, stream=17)
        assert calls == [("register", 1234, 0), ("unpin", 1, 2, 0, 512)]
    else:
        with pytest.raises(RuntimeError, match="pin retained"):
            model_vbar.ModelVBAR.unpin(vbar, 4096, 512, stream=17)
        assert calls == [("register", 1234, 0)]
