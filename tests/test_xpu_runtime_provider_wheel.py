from __future__ import annotations

import csv
import hashlib
import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest


_BUILDER = (
    Path(__file__).parents[1]
    / "packaging"
    / "xpu_runtime_provider"
    / "build_wheel.py"
)
_FAKE_TORCH_ABI_IDENTITY = (
    json.dumps({
        "schema_version": 1, "torch_version": "2.14.0+xpu", "cxx11_abi": True,
        "libraries": {name: "a" * 64 for name in
                      ("libc10.so", "libc10_xpu.so", "libtorch_xpu.so")},
    }, sort_keys=True) + "\n"
).encode()


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "comfy_aimdo_xpu_runtime_wheel_builder", _BUILDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_wheel(
    path: Path,
    *,
    distribution: str = "comfy-aimdo",
    include_native: bool = True,
    include_compiler_api: bool = False,
    include_native_owner: bool = False,
    include_native_owner_abi: bool = True,
    version: str = "0.5.5",
) -> Path:
    dist_info = f"comfy_aimdo-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.4\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
            "Requires-Python: >=3.9\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp39-abi3-linux_x86_64\n",
        )
        archive.writestr("comfy_aimdo/control.py", "lib = None\n")
        archive.writestr("comfy_aimdo/torch.py", "VALUE = 'xpu'\n")
        if include_compiler_api:
            for module in ("malloc_graph", "host_buffer", "model_vbar", "vram_buffer"):
                archive.writestr(f"comfy_aimdo/{module}.py", "VALUE = 'xpu'\n")
        if include_native_owner:
            archive.writestr("comfy_aimdo/native_owner.py", "VALUE = 'diagnostic'\n")
            archive.writestr("comfy_aimdo/aimdo_xpu_native_owner.so", b"fake-native-owner")
            if include_native_owner_abi:
                archive.writestr("comfy_aimdo/aimdo_xpu_native_owner_abi.json",
                                 _FAKE_TORCH_ABI_IDENTITY)
        if include_native:
            archive.writestr("comfy_aimdo/aimdo_xpu.so", b"fake-level-zero")
        archive.writestr(f"{dist_info}/RECORD", "")
    return path


def test_provider_wheel_has_disjoint_top_level_and_native_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl"
    )

    provider = builder.build_provider_wheel(
        source_wheel=source,
        output_directory=tmp_path / "dist",
        source_revision="a" * 40,
        torch_version="2.13.0+xpu",
        xpu_target="bmg",
    )

    assert provider.name == (
        "comfy_aimdo_xpu_runtime-0.5.5-cp39-abi3-linux_x86_64.whl"
    )
    if os.name == "posix":
        assert provider.stat().st_mode & 0o777 == 0o644
    with zipfile.ZipFile(provider) as archive:
        names = set(archive.namelist())
        assert not any(name.startswith("comfy_aimdo/") for name in names)
        vendored_control = (
            "comfy_aimdo_xpu_runtime/_vendor/comfy_aimdo/control.py"
        )
        vendored_native = (
            "comfy_aimdo_xpu_runtime/_vendor/comfy_aimdo/aimdo_xpu.so"
        )
        assert {vendored_control, vendored_native}.issubset(names)
        manifest = json.loads(
            archive.read("comfy_aimdo_xpu_runtime/provider.json")
        )
        assert manifest["provider_id"] == "comfy_aimdo.xpu"
        assert manifest["canonical_import"] == "comfy_aimdo"
        assert manifest["canonical_distribution"] == {
            "name": "comfy-aimdo",
            "compatible_versions": ["0.5.5"],
        }
        assert manifest["runtime"]["platforms"] == ["linux", "win32"]
        assert manifest["source"]["revision"] == "a" * 40
        assert manifest["source"]["wheel_sha256"] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        assert manifest["activation"] == {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {
                "linux": ["global", "native_hook"],
                "win32": ["native_hook"],
            },
            "default_allocator_modes": {"linux": "native_hook", "win32": "native_hook"},
        }
        assert manifest["native_artifacts"] == [
            {
                "path": vendored_native,
                "sha256": hashlib.sha256(b"fake-level-zero").hexdigest(),
            }
        ]

        wheel_metadata = archive.read(
            "comfy_aimdo_xpu_runtime-0.5.5.dist-info/WHEEL"
        ).decode()
        assert "Root-Is-Purelib: false" in wheel_metadata
        assert "Tag: cp39-abi3-linux_x86_64" in wheel_metadata
        entry_points = archive.read(
            "comfy_aimdo_xpu_runtime-0.5.5.dist-info/entry_points.txt"
        ).decode()
        assert "[comfyui_omnixpu.runtime_providers]" in entry_points
        assert (
            "comfy_aimdo.xpu = comfy_aimdo_xpu_runtime.provider:get_manifest"
            in entry_points
        )

        record = list(
            csv.reader(
                io.StringIO(
                    archive.read(
                    "comfy_aimdo_xpu_runtime-0.5.5.dist-info/RECORD"
                    ).decode()
                )
            )
        )
        assert {row[0] for row in record} == names


def test_provider_builder_requires_native_xpu_runtime(tmp_path):
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl",
        include_native=False,
    )

    with pytest.raises(RuntimeError, match="no AIMDO XPU native library"):
        builder.build_provider_wheel(
            source_wheel=source,
            output_directory=tmp_path / "dist",
            source_revision="b" * 40,
            torch_version="2.13.0+xpu",
            xpu_target="bmg",
        )


def test_provider_wheel_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl"
    )
    arguments = {
        "source_wheel": source,
        "source_revision": "c" * 40,
        "torch_version": "2.13.0+xpu",
        "xpu_target": "ptl-h",
    }

    first = builder.build_provider_wheel(
        output_directory=tmp_path / "first", **arguments
    )
    second = builder.build_provider_wheel(
        output_directory=tmp_path / "second", **arguments
    )

    assert first.read_bytes() == second.read_bytes()


def test_linux_compiler_api_manifest_requires_complete_055_exports(tmp_path, monkeypatch):
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl",
        include_compiler_api=True,
    )
    monkeypatch.setattr(builder, "shutil", types.SimpleNamespace(which=lambda _: "nm"))
    monkeypatch.setattr(
        builder,
        "subprocess",
        types.SimpleNamespace(
            run=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, stdout="\n".join(builder._COMPILER_SYMBOLS)
            )
        ),
    )

    provider = builder.build_provider_wheel(
        source_wheel=source,
        output_directory=tmp_path / "dist",
        source_revision="d" * 40,
        torch_version="2.13.0+xpu",
        xpu_target="bmg",
    )
    with zipfile.ZipFile(provider) as archive:
        manifest = json.loads(archive.read("comfy_aimdo_xpu_runtime/provider.json"))
    assert manifest["canonical_distribution"]["compatible_versions"] == ["0.5.5"]
    assert manifest["api_compatibility"]["xpu_recording_supported"] is False
    assert manifest["api_compatibility"]["upstream_reference_revision"] == (
        "3b8e8c162efeb9470d912609a7a6e7a2b1c693ec"
    )

    with pytest.raises(RuntimeError, match="reviewed AIMDO 0.5.5"):
        builder._compiler_api_contract("0.5.2", {
            "comfy_aimdo/" + module + ".py": b"x"
            for module in ("control", "torch", "malloc_graph", "host_buffer", "model_vbar", "vram_buffer")
        } | {"comfy_aimdo/aimdo_xpu.so": b"fake"})


def test_torch214_native_owner_sidecar_is_explicit_and_version_bound(tmp_path, monkeypatch):
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl",
        include_compiler_api=True, include_native_owner=True,
    )
    monkeypatch.setattr(builder, "shutil", types.SimpleNamespace(which=lambda _: "nm"))
    monkeypatch.setattr(
        builder, "subprocess",
        types.SimpleNamespace(run=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0,
            stdout="\n".join((*builder._COMPILER_SYMBOLS, *builder._NATIVE_OWNER_SYMBOLS,
                              *builder._NATIVE_OWNER_CORE_SYMBOLS)),
        )),
    )
    provider = builder.build_provider_wheel(
        source_wheel=source, output_directory=tmp_path / "provider",
        source_revision="e" * 40, torch_version="2.14.0+xpu", xpu_target="bmg",
    )
    with zipfile.ZipFile(provider) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("comfy_aimdo_xpu_runtime/provider.json"))
    sidecar = "comfy_aimdo_xpu_runtime/_vendor/comfy_aimdo/aimdo_xpu_native_owner.so"
    abi_identity = "comfy_aimdo_xpu_runtime/_vendor/comfy_aimdo/aimdo_xpu_native_owner_abi.json"
    diagnostic = manifest["native_owner_diagnostic"]
    assert sidecar in names and abi_identity in names
    assert diagnostic == {
        "enabled_by_default": False,
        "environment_flag": "AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC",
        "torch_version": "2.14.0+xpu",
        "path": sidecar,
        "sha256": hashlib.sha256(b"fake-native-owner").hexdigest(),
        "abi_identity_path": abi_identity,
        "abi_identity_sha256": hashlib.sha256(_FAKE_TORCH_ABI_IDENTITY).hexdigest(),
    }
    assert manifest["api_compatibility"]["xpu_recording_supported"] is False
    assert {item["path"] for item in manifest["native_artifacts"]} == {
        "comfy_aimdo_xpu_runtime/_vendor/comfy_aimdo/aimdo_xpu.so", sidecar,
    }
    (tmp_path / "missing-abi").mkdir()
    missing_abi = _source_wheel(
        tmp_path / "missing-abi/comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl",
        include_compiler_api=True, include_native_owner=True,
        include_native_owner_abi=False,
    )
    with pytest.raises(RuntimeError, match="Torch ABI identity is missing"):
        builder.build_provider_wheel(
            source_wheel=missing_abi, output_directory=tmp_path / "missing-abi-provider",
            source_revision="e" * 40, torch_version="2.14.0+xpu", xpu_target="bmg",
        )
    with pytest.raises(RuntimeError, match="requires AIMDO 0.5.5 and Torch 2.14.0"):
        builder.build_provider_wheel(
            source_wheel=source, output_directory=tmp_path / "wrong-torch",
            source_revision="e" * 40, torch_version="2.13.0+xpu", xpu_target="bmg",
        )


@pytest.mark.parametrize("payload", ("sidecar", "core"))
def test_native_owner_payload_rejects_each_missing_required_export(tmp_path, monkeypatch, payload):
    builder = _load_builder()
    required = builder._NATIVE_OWNER_SYMBOLS if payload == "sidecar" else builder._NATIVE_OWNER_CORE_SYMBOLS
    files = {
        "comfy_aimdo/native_owner.py": b"diagnostic",
        "comfy_aimdo/aimdo_xpu_native_owner.so": b"sidecar",
        "comfy_aimdo/aimdo_xpu.so": b"core",
        "comfy_aimdo/aimdo_xpu_native_owner_abi.json": _FAKE_TORCH_ABI_IDENTITY,
    }
    monkeypatch.setattr(builder.shutil, "which", lambda _: "nm")
    for missing in required:
        def inspect(command, **kwargs):
            selected = (builder._NATIVE_OWNER_SYMBOLS if command[-1].endswith("native_owner.so")
                        else builder._NATIVE_OWNER_CORE_SYMBOLS)
            return subprocess.CompletedProcess(command, 0, stdout="\n".join(set(selected) - {missing}))
        monkeypatch.setattr(builder.subprocess, "run", inspect)
        with pytest.raises(RuntimeError, match=missing):
            builder._native_owner_diagnostic_contract("0.5.5", "2.14.0+xpu", files)


def test_packaging_exports_cover_python_and_native_runtime_dependencies():
    builder = _load_builder()
    root = Path(__file__).parents[1]
    tree = ast.parse((root / "comfy_aimdo/native_owner.py").read_text())
    # Derive dependencies from the consumers, so an omitted declaration does
    # not also disappear from the test's expected ABI.
    python_symbols = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                      and node.attr.startswith("aimdo_full_proxy_")}
    python_symbols |= {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                       and isinstance(node.value, str) and node.value.startswith("aimdo_full_proxy_")}
    assert python_symbols <= set(builder._NATIVE_OWNER_SYMBOLS)
    native_source = (root / "src-xpu/native-owner-proxy.cpp").read_text()
    core_symbols = set(re.findall(r'dlsym\(\s*RTLD_DEFAULT,\s*"([A-Za-z0-9_]+)"', native_source))
    assert core_symbols <= set(builder._NATIVE_OWNER_CORE_SYMBOLS) | set(builder._COMPILER_SYMBOLS)
