import platform
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
        (
            None,
            "native_hook" if platform.system() == "Windows" else "global",
        ),
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


def test_xpu_memory_snapshot_preserves_native_and_vbar_ownership(
    monkeypatch,
):
    from comfy_aimdo import model_vbar

    torch_module = types.SimpleNamespace(
        xpu=types.SimpleNamespace(
            memory_stats=lambda device: {
                "active_bytes.all.current": 1024,
                "reserved_bytes.all.current": 2048,
            },
            memory_snapshot=lambda: [{"address": 17}],
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_mode", "native_hook")
    monkeypatch.setattr(control.platform, "system", lambda: "Windows")
    monkeypatch.setattr(control, "_xpu_device_index", lambda device=None: 1)
    monkeypatch.setattr(
        control, "get_xpu_vmm_stats", lambda: {"unmap_calls": 3}
    )
    monkeypatch.setattr(
        control, "get_xpu_ur_hook_stats", lambda: {"runtime_oom_calls": 1}
    )
    monkeypatch.setattr(
        model_vbar,
        "vbars_snapshot",
        lambda device=None, include_pages=True: [
            {"device": device, "loaded_size": 4096}
        ],
    )

    snapshot = control.get_xpu_memory_snapshot(
        1, include_native_segments=True
    )

    assert snapshot["allocator_owner"] == "torch_xpu_native"
    assert snapshot["native_allocator"]["stats"][
        "reserved_bytes.all.current"
    ] == 2048
    assert snapshot["native_allocator"]["segments"] == [{"address": 17}]
    assert snapshot["aimdo"]["vbars"] == [
        {"device": 1, "loaded_size": 4096}
    ]
    assert snapshot["aimdo"]["vmm"] == {"unmap_calls": 3}


def test_xpu_oom_snapshot_records_owner_boundary_stage(monkeypatch):
    monkeypatch.setattr(control, "_xpu_oom_history", [])
    monkeypatch.setattr(control, "_xpu_oom_last_snapshot_monotonic", {})
    monkeypatch.setattr(control, "_xpu_device_index", lambda device=None: 0)
    monkeypatch.setattr(
        control,
        "get_xpu_memory_snapshot",
        lambda device=None, include_native_segments=False,
        include_vbar_pages=True: {
            "device": 0,
            "allocator_owner": "torch_xpu_native",
        },
    )

    snapshot = control.capture_xpu_oom_snapshot(
        0, stage="vbar_fault_host_offload", request_bytes=33554432
    )

    assert snapshot["oom"] == {
        "stage": "vbar_fault_host_offload",
        "request_bytes": 33554432,
        "error": None,
        "coalesced_events": 1,
    }
    assert control.get_last_xpu_oom_snapshot() is snapshot
