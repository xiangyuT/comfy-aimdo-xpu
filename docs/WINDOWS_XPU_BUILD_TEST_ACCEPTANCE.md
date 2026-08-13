# Windows XPU build and validation

This document defines the current reproducible build and validation entry
points. It is a contract, not a record: machine-specific package snapshots,
artifact hashes, and individual performance runs do not belong here.

## Build prerequisites

The Windows native backend requires:

- Visual Studio Build Tools 2022 with the x64 C/C++ toolchain;
- Intel oneAPI DPC++/C++ Compiler 2025.3 or a compatible compiler;
- Level Zero headers, by default below `build/level-zero-src/include`;
- Microsoft Detours, by default below `build/detours-src`;
- a Windows SDK, either from Visual Studio or the project-local NuGet layout
  recognized by the build scripts.

Override the default locations with `LEVEL_ZERO_INCLUDE`, `DETOURS_ROOT`,
`DETOURS_INCLUDE`, `DETOURS_LIB_DIR`, `WINDOWS_SDK_NUGET`, or
`WINDOWS_SDK_VERSION` when necessary.

Provision Detours once when it is not already available:

```powershell
git clone --depth 1 https://github.com/microsoft/Detours.git build\detours-src
cmd /d /c scripts\build-windows-detours.cmd
```

## Native build

From the repository root:

```powershell
cmd /d /c scripts\build-windows-xpu.cmd
```

The result is `comfy_aimdo/aimdo_xpu.dll`. The script discovers Visual Studio,
activates oneAPI, applies the optional project-local Windows SDK, compiles the
Level Zero and Unified Runtime interception code, and links Detours, DXGI, and
the SYCL runtime.

The runtime provides `ze_loader.dll`; it is not bundled into the AIMDO wheel.
Portable environments must also make their SYCL and Unified Runtime DLL
directory visible before AIMDO loads. `comfy_aimdo.control` adds the embedded
Python `Library/bin` directory on Windows.

## Package version

The version is derived by `setuptools-scm` from the git tag history and must
never be supplied by hand. `SETUPTOOLS_SCM_PRETEND_VERSION` produces a number
that is attached to no commit, so two different trees can install as the same
version and a measurement cannot be traced back to a build.

Deriving it requires the release tags to be present in the clone. A fork
created without them yields `0.1.dev<count>`, which is what makes the override
tempting. Fetch them once:

```powershell
git fetch https://github.com/Comfy-Org/comfy-aimdo.git "refs/tags/*:refs/tags/*"
```

`git describe --tags` then names the build, and a wheel built from a clean tree
carries the matching version: `v0.4.13-40-g1a17d3e` builds `0.4.14.dev40`.
Because the configured `local_scheme` drops the local segment, the version
identifies a commit only when the tree is clean; do not record a result from a
wheel built over uncommitted changes.

## Focused validation

Run the hook unit test, which needs no XPU:

```powershell
cmd /d /c scripts\test-windows-xpu-hook.cmd
```

Run the platform/source tests from an isolated environment containing pytest:

```powershell
build\python-venv\Scripts\python.exe -m pytest `
    tests\test_windows_pressure_source.py `
    tests\test_xpu_comfyui_opt_in.py -q
```

Real-XPU checks must use the same Python, Torch, driver, and native DLL that
will run ComfyUI. Useful focused entry points include:

```powershell
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_smoke.py
python tests\repro_xpu_platform_memory_policy.py
```

A component reproducer proves only the path it exercises. A memory-policy
change also requires a real workload that reaches pressure, demonstrated by
non-zero pressure/reclaim counters in the same run.

## Runtime identity

The embeddable Portable Python uses a `._pth` file and ignores `PYTHONPATH`.
Do not assume a repository checkout is being imported. Before accepting a
result, record the installed package version and resolved module path:

```powershell
<portable>\python_embeded\python.exe -s -c `
    "import comfy_aimdo, comfy_aimdo.control as c; print(comfy_aimdo.__file__); print(c.__file__)"
```

When validating a local wheel, install it explicitly with `--force-reinstall
--no-deps`, then restart ComfyUI. The installed native DLL should hash
identically to the DLL packaged in the wheel.

## ComfyUI acceptance

Launch ComfyUI with `--enable-dynamic-vram` and record the selected physical
adapter, `ZE_AFFINITY_MASK`, reserve, package revisions, and server log window.

For changes to pressure, reclaim, cache interaction, or residency, a short
two-step workflow is only a regression smoke test. Run a configuration that
actually reaches pressure, normally a long video generation at full sampler
step count, and require all of the following in the same window:

1. output/media correctness;
2. forward progress without a long unchanged interval;
3. no Level Zero, Unified Runtime, copy, OOM, device-loss, or driver-reset
   error;
4. bounded WDDM local/non-local usage relative to the configured reserve;
5. pressure and reclaim counters proving the changed path executed;
6. repeated-prompt timing without a systematic regression.

Do not promote a cached execution as a sample. Change the seed or another
sampler input for every prompt while allowing static loader nodes to remain
cached.
