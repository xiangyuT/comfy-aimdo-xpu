from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest


_BUILDER = (
    Path(__file__).parents[1]
    / "packaging"
    / "xpu_runtime_provider"
    / "build_wheel.py"
)


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
) -> Path:
    dist_info = "comfy_aimdo-0.4.13.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.4\n"
            f"Name: {distribution}\n"
            "Version: 0.4.13\n"
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
        tmp_path / "comfy_aimdo-0.4.13-cp39-abi3-linux_x86_64.whl"
    )

    provider = builder.build_provider_wheel(
        source_wheel=source,
        output_directory=tmp_path / "dist",
        source_revision="a" * 40,
        torch_version="2.13.0+xpu",
        xpu_target="bmg",
    )

    assert provider.name == (
        "comfy_aimdo_xpu_runtime-0.4.13-cp39-abi3-linux_x86_64.whl"
    )
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
            "compatible_versions": ["0.4.13"],
        }
        assert manifest["source"]["revision"] == "a" * 40
        assert manifest["source"]["wheel_sha256"] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        assert manifest["activation"] == {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {
                "linux": ["global"],
                "win32": ["native_hook"],
            },
        }
        assert manifest["native_artifacts"] == [
            {
                "path": vendored_native,
                "sha256": hashlib.sha256(b"fake-level-zero").hexdigest(),
            }
        ]

        wheel_metadata = archive.read(
            "comfy_aimdo_xpu_runtime-0.4.13.dist-info/WHEEL"
        ).decode()
        assert "Root-Is-Purelib: false" in wheel_metadata
        assert "Tag: cp39-abi3-linux_x86_64" in wheel_metadata
        entry_points = archive.read(
            "comfy_aimdo_xpu_runtime-0.4.13.dist-info/entry_points.txt"
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
                        "comfy_aimdo_xpu_runtime-0.4.13.dist-info/RECORD"
                    ).decode()
                )
            )
        )
        assert {row[0] for row in record} == names


def test_provider_builder_requires_native_xpu_runtime(tmp_path):
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "comfy_aimdo-0.4.13-cp39-abi3-linux_x86_64.whl",
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
        tmp_path / "comfy_aimdo-0.4.13-cp39-abi3-linux_x86_64.whl"
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
