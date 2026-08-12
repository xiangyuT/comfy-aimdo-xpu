import sys
import types

import pytest

from comfy_aimdo import control


@pytest.mark.parametrize(
    ("implementation_was_explicit", "comfy_args", "expected"),
    [
        (True, types.SimpleNamespace(enable_dynamic_vram=False), True),
        (False, types.SimpleNamespace(enable_dynamic_vram=True), True),
        (False, types.SimpleNamespace(enable_dynamic_vram=False), False),
        (False, types.SimpleNamespace(), True),
        (False, None, True),
    ],
)
def test_xpu_initialization_request_contract(
    monkeypatch,
    implementation_was_explicit,
    comfy_args,
    expected,
):
    if comfy_args is None:
        monkeypatch.delitem(sys.modules, "comfy.cli_args", raising=False)
    else:
        monkeypatch.setitem(
            sys.modules,
            "comfy.cli_args",
            types.SimpleNamespace(args=comfy_args),
        )

    assert (
        control._xpu_initialization_requested(implementation_was_explicit)
        is expected
    )


def test_comfyui_xpu_opt_out_precedes_native_library_load(monkeypatch):
    monkeypatch.setattr(control, "lib", None)
    monkeypatch.setattr(control, "implementation", None)
    monkeypatch.setattr(control, "detect_vendor", lambda: "xpu")
    monkeypatch.setitem(
        sys.modules,
        "comfy.cli_args",
        types.SimpleNamespace(
            args=types.SimpleNamespace(enable_dynamic_vram=False)
        ),
    )

    def unexpected_library_load(*args, **kwargs):
        pytest.fail("XPU opt-out must not load the native AIMDO library")

    monkeypatch.setattr(control.ctypes, "CDLL", unexpected_library_load)

    assert control.init() is False
    assert control.lib is None
    assert control.implementation is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "global"),
        ("global", "global"),
        ("native_hook", "native_hook"),
    ],
)
def test_xpu_allocator_mode_contract(monkeypatch, value, expected):
    monkeypatch.delenv("AIMDO_XPU_ALLOCATOR_MODE", raising=False)
    assert control._normalize_xpu_allocator_mode(value) == expected


def test_xpu_allocator_mode_environment(monkeypatch):
    monkeypatch.setenv("AIMDO_XPU_ALLOCATOR_MODE", "native_hook")
    assert control._normalize_xpu_allocator_mode(None) == "native_hook"
    assert control._normalize_xpu_allocator_mode("global") == "global"


def test_xpu_allocator_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported XPU allocator mode"):
        control._normalize_xpu_allocator_mode("unknown")


def test_xpu_allocator_mode_rejects_retired_native_pool():
    with pytest.raises(ValueError, match="unsupported XPU allocator mode"):
        control._normalize_xpu_allocator_mode("native_pool")


@pytest.mark.parametrize(
    ("mode", "global_swaps", "has_allocator"),
    [
        ("global", 1, True),
        ("native_hook", 0, False),
    ],
)
def test_xpu_allocator_backend_selection(
    mode, global_swaps, has_allocator
):
    allocator = object()
    requests = []
    swaps = []
    aimdo_torch = types.SimpleNamespace(
        get_torch_allocator=lambda **kwargs: (
            requests.append(kwargs) or allocator
        )
    )
    torch_module = types.SimpleNamespace(
        xpu=types.SimpleNamespace(
            memory=types.SimpleNamespace(
                change_current_allocator=swaps.append
            )
        )
    )

    result = control._install_xpu_allocator_backend(
        torch_module, aimdo_torch, mode
    )

    assert (result is allocator) is has_allocator
    assert requests == ([{}] if has_allocator else [])
    assert swaps == ([allocator] if global_swaps else [])
