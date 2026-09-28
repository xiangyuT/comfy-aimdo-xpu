"""D1 ABI eligibility tests; no implicit device initialization or recording."""
import ctypes
import os
from pathlib import Path
import subprocess
import types

import pytest
from comfy_aimdo import control, malloc_graph, native_owner


class Function:
    def __init__(self, value=True):
        self.value = value
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.value


def library():
    functions = {name: Function() for name in control._MEMORY_COMPILER_SIGNATURES}
    functions.update(malloc_graph_abi_version=Function(1), malloc_graph_capabilities=Function(1),
        malloc_graph_source_revision=Function(b"c" * 40),
        malloc_graph_source_content_sha256=Function(b"a" * 64))
    return types.SimpleNamespace(**functions)


@pytest.mark.parametrize("missing", tuple(control._MEMORY_COMPILER_SIGNATURES))
def test_partial_native_abi_rejected_before_allocator_install(monkeypatch, missing):
    native = library()
    delattr(native, missing)
    monkeypatch.setattr(control, "lib", None)
    monkeypatch.setattr(control, "_memory_compiler_native", None)
    monkeypatch.setattr(control.ctypes, "CDLL", lambda *a, **k: native)
    monkeypatch.setattr(control, "_install_xpu_allocator_backend", lambda *a: pytest.fail("allocator must remain untouched"))
    assert control.init(implementation="xpu") is False
    assert control.lib is None
    assert control._memory_compiler_native is None


@pytest.mark.parametrize("symbol,value", [("malloc_graph_abi_version", 0),
                                         ("malloc_graph_abi_version", 2),
                                         ("malloc_graph_capabilities", 0),
                                         ("malloc_graph_capabilities", 3)])
def test_unreviewed_abi_or_router_bits_cannot_enable_recording(symbol, value):
    native = library()
    setattr(native, symbol, Function(value))
    with pytest.raises(RuntimeError, match="unsupported AIMDO"):
        control._bind_memory_compiler(native, "xpu")


def test_windows_xpu_legacy_payload_keeps_compiler_unavailable(monkeypatch):
    monkeypatch.setattr(control.platform, "system", lambda: "Windows")
    assert control._bind_memory_compiler(types.SimpleNamespace(), "xpu") is None


def test_core_build_does_not_claim_xpu_execution(monkeypatch):
    native = library()
    native._name = "/no-library-loaded-by-this-fixture"
    monkeypatch.setattr(control, "lib", native)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_memory_compiler_native", control._bind_memory_compiler(native, "xpu"))
    class XpuStream:
        device = types.SimpleNamespace(type="xpu", index=0)
        @property
        def cuda_stream(self): pytest.fail("XPU stream must be rejected before reading cuda_stream")
    for record in (control.record, malloc_graph.record):
        with pytest.raises(NotImplementedError, match="memory compiler.*not yet supported on XPU"):
            record(XpuStream())
    assert not native.malloc_graph_create.calls
    capability = control.get_memory_compiler_capability()
    assert capability["core_built"] and capability["native_symbols_complete"]
    assert capability["abi_revision"] == 1
    assert not capability["available"] and not capability["memory_only"]
    assert not capability["execution_graph"] and not capability["xpu_consumer_tracking"]
    capability["available"] = True
    assert not control.get_memory_compiler_capability()["available"]


def test_capability_query_before_init_is_read_only(monkeypatch):
    monkeypatch.setattr(control.platform, "system", lambda: "Linux")
    monkeypatch.setattr(control, "lib", None)
    monkeypatch.setattr(native_owner, "installed", lambda: False)
    monkeypatch.setattr(control.ctypes, "CDLL", lambda *a, **k: pytest.fail("read-only capability loaded a DSO"))
    capability = control.get_memory_compiler_capability()
    assert capability["reason"] == "not_initialized"
    assert not capability["core_built"] and not capability["available"]
    assert capability["native_owner_diagnostic"] == {
        "installed": False, "active": False, "entrypoint": None,
        "consumer_contract": None, "public_available": False,
        "reason": "not_installed",
    }


def test_native_owner_diagnostic_capability_is_separate_from_public_record(monkeypatch):
    native = library()
    native._name = "/no-library-loaded-by-this-fixture"
    monkeypatch.setattr(control.platform, "system", lambda: "Linux")
    monkeypatch.setattr(control, "lib", native)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_memory_compiler_native",
                        control._bind_memory_compiler(native, "xpu"))
    monkeypatch.setattr(control, "_xpu_allocator_mode", "native_hook")
    monkeypatch.setattr(control, "_xpu_allocator_ready", True)
    monkeypatch.setattr(native_owner, "installed", lambda: True)
    capability = control.get_memory_compiler_capability()
    assert capability["native_owner_diagnostic"] == {
        "installed": True, "active": True,
        "entrypoint": "native_owner.record_diagnostic",
        "consumer_contract": "explicit_record_stream",
        "public_available": False, "reason": "opt_in_component_only",
    }
    assert not capability["available"] and not capability["memory_only"]
    assert not capability["xpu_consumer_tracking"]
    stream = types.SimpleNamespace(device=types.SimpleNamespace(type="xpu", index=0))
    with pytest.raises(NotImplementedError, match="not yet supported on XPU"):
        malloc_graph.record(stream)
    assert not native.malloc_graph_create.calls
    monkeypatch.setattr(control, "_xpu_allocator_ready", False)
    diagnostic = control.get_memory_compiler_capability()["native_owner_diagnostic"]
    assert diagnostic["installed"] and not diagnostic["active"]
    assert diagnostic["entrypoint"] is None and diagnostic["consumer_contract"] is None
    assert diagnostic["reason"] == "installed_context_inactive"
    monkeypatch.setattr(control, "_xpu_allocator_ready", True)
    monkeypatch.setattr(control, "lib", None)
    diagnostic = control.get_memory_compiler_capability()["native_owner_diagnostic"]
    assert diagnostic["installed"] and not diagnostic["active"]
    assert diagnostic["reason"] == "installed_context_inactive"


def test_non_linux_capability_does_not_query_native_owner(monkeypatch):
    monkeypatch.setattr(control.platform, "system", lambda: "Windows")
    monkeypatch.setattr(control, "lib", None)
    monkeypatch.setattr(native_owner, "installed",
                        lambda: pytest.fail("Windows queried Linux native-owner sidecar"))
    diagnostic = control.get_memory_compiler_capability()["native_owner_diagnostic"]
    assert diagnostic["installed"] is False and diagnostic["active"] is False
    assert diagnostic["public_available"] is False and diagnostic["reason"] == "linux_only"


def test_legacy_cuda_abi_preserves_stream_and_pointer_contract(monkeypatch):
    native = types.SimpleNamespace(**{name: Function() for name in control._MEMORY_COMPILER_SIGNATURES})
    native._name = "/legacy-official-fixture"
    native.malloc_graph_create.value = 0x123456789AB
    monkeypatch.setattr(control, "lib", native)
    monkeypatch.setattr(control, "implementation", "cuda")
    monkeypatch.setattr(control, "get_devctx", lambda device: 0xABCDEF000 + device)
    monkeypatch.setattr(control, "_memory_compiler_native", control._bind_memory_compiler(native, "cuda"))
    stream = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=3), cuda_stream=0xFEDCBA987)
    graph = malloc_graph.record(stream)
    args = native.malloc_graph_create.calls[0]
    assert args[0] == 0xABCDEF003 and args[1].value == 0xFEDCBA987
    assert graph._handle == 0x123456789AB
    assert native.malloc_graph_create.restype is ctypes.c_void_p
    assert native.malloc_graph_stat.restype is ctypes.c_uint64
    graph.abort()
    graph.__del__()
    assert native.malloc_graph_abort.calls == [(0x123456789AB,)]
    assert native.malloc_graph_destroy.calls == [(0x123456789AB,)]


def test_duplicate_device_init_rejected_before_native_or_queue_use(monkeypatch):
    monkeypatch.setattr(control, "lib", object())
    monkeypatch.setattr(control, "devctxs", [0x1234])
    assert control.init_devices([0]) is False
    assert control.devctxs == [0x1234]


def test_linux_build_inputs_include_complete_core():
    root = Path(__file__).parents[1]
    linux = (root / "scripts/build-linux-xpu.sh").read_text()
    for unit in ("malloc-graph", "malloc-rogue", "vmm-ref"):
        assert unit + ".c" in linux
    assert "write-source-identity.py" in linux
    assert "zeVirtualMemSetAccessAttribute" in (root / "src-xpu/vmm-manager.h").read_text()


def test_actual_candidate_dso_binds_complete_abi_and_identity():
    import hashlib
    configured = os.environ.get("AIMDO_D1_LIBRARY")
    path = Path(configured) if configured else Path(control.__file__).parent / "aimdo_xpu.so"
    if configured:
        assert path.is_file(), "configured candidate DSO is missing"
    if not path.is_file():
        pytest.skip("requires an actual newly built AIMDO D1 XPU library")
    native = ctypes.CDLL(str(path))
    capability = control._bind_memory_compiler(native, "xpu")
    assert capability["abi_revision"] == 1 and capability["symbols_complete"]
    assert not capability["router_available"]
    revision = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(Path(__file__).parents[1]),
         "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    assert capability["source_revision"] == revision
    assert len(capability["source_content_sha256"]) == 64
    assert int(capability["source_content_sha256"], 16)
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()
