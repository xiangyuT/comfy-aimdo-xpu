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
