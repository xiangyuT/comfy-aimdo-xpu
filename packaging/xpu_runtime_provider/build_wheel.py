#!/usr/bin/env python3
"""Build the co-installable AIMDO XPU runtime provider wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


CANONICAL_DISTRIBUTION = "comfy-aimdo"
CANONICAL_PACKAGE = "comfy_aimdo"
PROVIDER_DISTRIBUTION = "comfy-aimdo-xpu-runtime"
PROVIDER_PACKAGE = "comfy_aimdo_xpu_runtime"
PROVIDER_ID = "comfy_aimdo.xpu"
ENTRY_POINT_GROUP = "comfyui_omnixpu.runtime_providers"
SOURCE_REPOSITORY = "https://github.com/xiangyuT/comfy-aimdo-xpu.git"
SUPPORTED_PLATFORMS = ("linux", "win32")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_component(value: str) -> str:
    return re.sub(r"[^\w\d.]+", "_", value, flags=re.UNICODE)


def _zip_datetime() -> tuple[int, int, int, int, int, int]:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    return time.gmtime(max(epoch, 315532800))[:6]


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _zip_datetime())
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | 0o644) << 16
    info.create_system = 3
    return info


def _read_single_member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"source wheel must contain exactly one {suffix}, found {matches}"
        )
    return archive.read(matches[0])


def _source_wheel_contract(
    source_wheel: Path,
) -> tuple[str, str, str, tuple[str, ...], dict[str, bytes]]:
    with zipfile.ZipFile(source_wheel) as archive:
        metadata_text = _read_single_member(archive, ".dist-info/METADATA").decode(
            "utf-8"
        )
        wheel_text = _read_single_member(archive, ".dist-info/WHEEL").decode(
            "utf-8"
        )
        metadata = Parser().parsestr(metadata_text)
        distribution = metadata.get("Name", "")
        version = metadata.get("Version", "")
        requires_python = metadata.get("Requires-Python", ">=3.9")
        if _normalize_distribution(distribution) != CANONICAL_DISTRIBUTION:
            raise RuntimeError(
                f"expected a {CANONICAL_DISTRIBUTION} wheel, got {distribution!r}"
            )
        if not version:
            raise RuntimeError("source wheel metadata has no Version")

        wheel_metadata = Parser().parsestr(wheel_text)
        tags = tuple(wheel_metadata.get_all("Tag", ()))
        root_is_purelib = wheel_metadata.get("Root-Is-Purelib", "false").lower()
        if not tags:
            raise RuntimeError("source wheel metadata has no Tag")
        if root_is_purelib not in {"true", "false"}:
            raise RuntimeError("source wheel has an invalid Root-Is-Purelib value")

        prefix = f"{CANONICAL_PACKAGE}/"
        files: dict[str, bytes] = {}
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise RuntimeError(f"unsafe source wheel member: {member.filename!r}")
            if member.filename in files:
                raise RuntimeError(f"duplicate source wheel member: {member.filename}")
            files[member.filename] = archive.read(member)

    required = f"{CANONICAL_PACKAGE}/control.py"
    if required not in files:
        raise RuntimeError(f"source wheel does not contain {required}")
    if not any(
        PurePosixPath(name).name in {"aimdo_xpu.so", "aimdo_xpu.dll"}
        for name in files
    ):
        raise RuntimeError("source wheel has no AIMDO XPU native library")
    return version, requires_python, root_is_purelib, tags, files


def _provider_module_source() -> bytes:
    return (
        '"""Lightweight metadata entry point for the OmniXPU bootstrap."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "from importlib.resources import files\n"
        "\n"
        "\n"
        "def get_manifest():\n"
        "    path = files(__package__).joinpath(\"provider.json\")\n"
        "    return json.loads(path.read_text(encoding=\"utf-8\"))\n"
    ).encode("utf-8")


def _manifest(
    *,
    source_wheel: Path,
    source_revision: str,
    source_version: str,
    torch_version: str,
    xpu_target: str,
    vendored_files: dict[str, bytes],
) -> dict[str, object]:
    file_hashes = {
        name: _sha256(data) for name, data in sorted(vendored_files.items())
    }
    native_artifacts = [
        {"path": name, "sha256": digest}
        for name, digest in file_hashes.items()
        if PurePosixPath(name).suffix.lower() in {".dll", ".dylib", ".pyd", ".so"}
    ]
    return {
        "schema_version": 1,
        "provider_id": PROVIDER_ID,
        "provider_distribution": {
            "name": PROVIDER_DISTRIBUTION,
            "version": source_version,
        },
        "provider_package": PROVIDER_PACKAGE,
        "canonical_distribution": {
            "name": CANONICAL_DISTRIBUTION,
            "compatible_versions": [source_version],
        },
        "canonical_import": CANONICAL_PACKAGE,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "distribution": CANONICAL_DISTRIBUTION,
            "version": source_version,
            "wheel_sha256": _sha256(source_wheel.read_bytes()),
        },
        "runtime": {
            "torch_version": torch_version,
            "torch_build": "xpu",
            "xpu_targets": [xpu_target],
            "platforms": list(SUPPORTED_PLATFORMS),
        },
        "activation": {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {
                "linux": ["global"],
                "win32": ["native_hook"],
            },
        },
        "vendor_root": f"{PROVIDER_PACKAGE}/_vendor",
        "vendored_files": file_hashes,
        "native_artifacts": native_artifacts,
    }


def build_provider_wheel(
    *,
    source_wheel: Path,
    output_directory: Path,
    source_revision: str,
    torch_version: str,
    xpu_target: str,
) -> Path:
    """Re-home the canonical AIMDO wheel below a provider-owned package."""

    source_wheel = source_wheel.resolve(strict=True)
    if not _REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source revision must be a lowercase 40-character Git SHA")
    if not torch_version.endswith("+xpu"):
        raise ValueError("torch version must identify an XPU build with +xpu")
    if xpu_target not in {"bmg", "ptl-h"}:
        raise ValueError("xpu target must be bmg or ptl-h")

    (
        source_version,
        requires_python,
        root_is_purelib,
        tags,
        source_files,
    ) = _source_wheel_contract(source_wheel)
    vendored_files = {
        f"{PROVIDER_PACKAGE}/_vendor/{name}": data
        for name, data in source_files.items()
    }
    manifest = _manifest(
        source_wheel=source_wheel,
        source_revision=source_revision,
        source_version=source_version,
        torch_version=torch_version,
        xpu_target=xpu_target,
        vendored_files=vendored_files,
    )

    dist_info = (
        f"{_wheel_component(PROVIDER_DISTRIBUTION)}-"
        f"{_wheel_component(source_version)}.dist-info"
    )
    contents = dict(vendored_files)
    contents[f"{PROVIDER_PACKAGE}/__init__.py"] = (
        '"""Co-installable AIMDO XPU runtime provider."""\n'
        "\n"
        "from .provider import get_manifest\n"
        "\n"
        '__all__ = ["get_manifest"]\n'
    ).encode("utf-8")
    contents[f"{PROVIDER_PACKAGE}/provider.py"] = _provider_module_source()
    contents[f"{PROVIDER_PACKAGE}/provider.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    contents[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.4\n"
        f"Name: {PROVIDER_DISTRIBUTION}\n"
        f"Version: {source_version}\n"
        "Summary: Co-installable AIMDO Intel XPU runtime provider\n"
        f"Requires-Python: {requires_python}\n"
        "\n"
    ).encode("utf-8")
    contents[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: comfy-aimdo-xpu-runtime-provider\n"
        f"Root-Is-Purelib: {root_is_purelib}\n"
        + "".join(f"Tag: {tag}\n" for tag in tags)
        + "\n"
    ).encode("utf-8")
    contents[f"{dist_info}/entry_points.txt"] = (
        f"[{ENTRY_POINT_GROUP}]\n"
        f"{PROVIDER_ID} = {PROVIDER_PACKAGE}.provider:get_manifest\n"
    ).encode("utf-8")
    contents[f"{dist_info}/top_level.txt"] = f"{PROVIDER_PACKAGE}\n".encode(
        "utf-8"
    )

    record_path = f"{dist_info}/RECORD"
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name, data in sorted(contents.items()):
        writer.writerow((name, _record_hash(data), str(len(data))))
    writer.writerow((record_path, "", ""))
    contents[record_path] = record_buffer.getvalue().encode("utf-8")

    filename = (
        f"{_wheel_component(PROVIDER_DISTRIBUTION)}-"
        f"{_wheel_component(source_version)}-{tags[0]}.whl"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory.resolve() / filename
    with tempfile.NamedTemporaryFile(
        prefix=f".{filename}.", suffix=".tmp", dir=output_directory, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(_zip_info(name), data)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--xpu-target", choices=("bmg", "ptl-h"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    wheel = build_provider_wheel(
        source_wheel=args.source_wheel,
        output_directory=args.output_dir,
        source_revision=args.source_revision,
        torch_version=args.torch_version,
        xpu_target=args.xpu_target,
    )
    print(
        json.dumps(
            {"path": str(wheel), "sha256": _sha256(wheel.read_bytes())},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
