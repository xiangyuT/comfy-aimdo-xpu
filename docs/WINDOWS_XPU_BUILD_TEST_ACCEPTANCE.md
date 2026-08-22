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

The version is derived by `setuptools-scm` and must never be supplied by hand.
`SETUPTOOLS_SCM_PRETEND_VERSION` produces a number attached to no commit, so
two different trees install as the same version and a measurement cannot be
traced back to a build.

This repository is not a branch of the upstream release line; it is upstream
plus a Windows/XPU port under the same distribution name. The version therefore
uses a PEP 440 local identifier, which is the construct for a modification of
an upstream release:

```text
0.4.13+xpu.g7acd727
```

The base is the newest upstream release the tree contains, `xpu` marks the
port, and the commit follows. A tree with uncommitted changes appends `.dirty`,
so a wheel built over local edits says so and must not be used to record a
result. The schemes are defined in `setup.py`, because `setuptools-scm`
resolves a scheme named in `pyproject.toml` through installed entry points and
cannot import a module from an unbuilt source tree.

Deriving the base requires the upstream release tags, which a fork is created
without. Fetch them once, otherwise the base degrades to `0.0`:

```powershell
git fetch https://github.com/Comfy-Org/comfy-aimdo.git "refs/tags/*:refs/tags/*"
```

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

Retirement changes must first pass both the premature-unmap oracle and the
same-model resident-growth gate. Run each mode in a fresh process because the
native mode selection is cached:

```powershell
<portable>\python_embeded\python.exe -s `
    tests\run_xpu_vbar_retirement_safety.py `
    --device <index> --mode default --matrix-size 12288
<portable>\python_embeded\python.exe -s `
    tests\run_xpu_vbar_retirement_safety.py `
    --device <index> --mode async --matrix-size 12288 --cycles 100
<portable>\python_embeded\python.exe -s `
    tests\run_xpu_vbar_resident_growth.py `
    --device <index> --mode default --pages 16
<portable>\python_embeded\python.exe -s `
    tests\run_xpu_vbar_resident_growth.py `
    --device <index> --mode reference --pages 16
```

The default 16-page capacity gate maps at most 512 MiB. Default/async must
remain bounded at one resident page in this deterministic pressure window and
finish at zero; reference is expected to grow from one through sixteen and is
therefore a correctness oracle, not a product-performance candidate. Do not
start a workflow benchmark when the default capacity gate or the default
multi-queue safety oracle fails.

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

The workflow ladder is strictly staged: deterministic component gates above,
then a bounded two-step pressure canary, then one full-step-count sample, and
only then a repeated duration/resolution cycle. A canary that exceeds the
matched baseline by more than 10% is a promotion failure and must not be
allowed to continue into the long cycle.

Do not promote a cached execution as a sample. Change the seed or another
sampler input for every prompt while allowing static loader nodes to remain
cached.
