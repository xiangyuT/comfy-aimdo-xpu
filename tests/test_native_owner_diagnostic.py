"""Fail-closed admission for the opt-in Torch 2.14 native-owner sidecar."""

from types import SimpleNamespace

import pytest

from comfy_aimdo import control, native_owner


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
