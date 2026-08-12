import sys
import types
from contextlib import nullcontext

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
        ("native_pool", "native_pool"),
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


@pytest.mark.parametrize(
    ("mode", "raw_segments", "global_swaps", "has_allocator"),
    [
        ("global", False, 1, True),
        ("native_hook", None, 0, False),
        ("native_pool", True, 0, True),
    ],
)
def test_xpu_allocator_backend_selection(
    mode, raw_segments, global_swaps, has_allocator
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
    assert requests == (
        [{"raw_segments": raw_segments}] if has_allocator else []
    )
    assert swaps == ([allocator] if global_swaps else [])


def test_native_pool_is_lazy_per_device_and_scoped(monkeypatch):
    allocator_handle = object()

    class FakeAllocator:
        def allocator(self):
            return allocator_handle

    class FakePool:
        def __init__(self, allocator):
            assert allocator is allocator_handle
            self.id = (0, 7)

        def use_count(self):
            return 1

    pool_scopes = []
    fake_xpu = types.SimpleNamespace(
        current_device=lambda: 1,
        device=lambda device: nullcontext(),
        MemPool=FakePool,
        use_mem_pool=lambda pool, device=None: (
            pool_scopes.append((pool, device)) or nullcontext()
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(xpu=fake_xpu),
    )
    monkeypatch.setattr(control, "lib", object())
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_ready", True)
    monkeypatch.setattr(control, "_xpu_allocator_mode", "native_pool")
    monkeypatch.setattr(control, "_torch_allocator", FakeAllocator())
    monkeypatch.setattr(control, "_torch_xpu_memory_pools", {})

    first = control.get_xpu_allocator_pool()
    second = control.get_xpu_allocator_pool()
    assert first is second

    with control.use_xpu_allocator_pool() as active:
        assert active is first

    assert pool_scopes == [(first, 1)]
    assert control.release_xpu_allocator_pool() is True
    assert control.release_xpu_allocator_pool() is False
