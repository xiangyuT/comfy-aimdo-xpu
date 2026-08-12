# Windows XPU build, test, and acceptance environment

This document records the environment actually used for the Windows AIMDO
investigation on 2026-08-11. It distinguishes the isolated build environment
from the ComfyUI Portable runtime. They must not be treated as one Python
environment.

The Windows DynamicVRAM issue is still unresolved. Reproducing this environment
does not imply that the formal acceptance workload passes.

## 1. Source baseline

| Component | Revision or version used |
| --- | --- |
| `comfy-aimdo-xpu` runtime code | branch `dev/windows-xpu-usm-free-hang`, code commit `323c0e0`, plus the uncommitted allocation-time interception described in the liveness analysis (installed as `0.4.14.dev3`) |
| Analysis-only follow-up | `ce1120f` and later documentation commits; no runtime-code change |
| `llm-scaler` / `ComfyUI-OmniXPU` source | branch `feature/omni-0.1.0b9-preview`, commit `9da0b75a9472bde58db2f9238b61619c76126690` |
| ComfyUI | detached commit `b1693ecba9f5b65f8c80ab36b195ab963ec92413`, reports version `0.30.0` |
| Level Zero build headers | commit `5cc079a` |

The Custom Node installed below `ComfyUI/custom_nodes/ComfyUI-OmniXPU` is a
copied deployment, not an independent Git checkout. Its source files matched
the milestone source above when line-ending-only differences were ignored.
Record a content hash or install from a clean source commit when producing a
release archive.

The Portable ComfyUI checkout intentionally has one tracked modification:
`comfy-kitchen==0.2.26` is removed from `requirements.txt` with a comment that
the XPU Kitchen fork is managed separately. This prevents a later generic
requirements update from replacing the XPU wheel with the upstream wheel.

## 2. Build host and native toolchain

The wheel currently installed in Portable was built with:

| Item | Captured value |
| --- | --- |
| Host OS | Windows 11 Pro, 64-bit, version `10.0.26200`, build `26200` |
| Visual Studio | Visual Studio Build Tools 2022 `17.14.36` |
| MSVC C/C++ compiler | `19.42.34444` for x64 |
| Intel DPC++ compiler | oneAPI DPC++/C++ `2025.3.3` (`2025.3.3.20260319`) |
| Windows SDK | NuGet SDK `10.0.26100.0` |
| Level Zero headers | repository commit `5cc079a` |
| Build Python | CPython `3.13.12`, MSC `v.1944`, 64-bit |
| `build` | `1.3.0` |
| `setuptools` | `78.1.0` |
| `setuptools-scm` | `9.2.2` |
| `wheel` | `0.47.0` |
| `pytest` | `9.1.1` |

The Python build environment is repository-local at
`build/python-venv`. It is not the Portable `python_embeded` environment and
must not install or upgrade packages in Portable.

### Native build inputs

The native build is driven by `scripts/build-windows-xpu.cmd`. The script:

1. discovers Visual Studio with `vswhere`;
2. activates the x64 Visual Studio command environment;
3. activates Intel oneAPI;
4. uses the NuGet Windows SDK stored below `build/windows-sdk-nuget`;
5. uses Level Zero headers below `build/level-zero-src/include`;
6. uses Detours below `build/detours-src`;
7. compiles C sources with `/O2 /MD /DAIMDO_XPU`;
8. compiles the SYCL dispatcher with `icx-cl`, C++17, `/O2`, `/MD`, and
   `-fsycl`;
9. links against `ze_loader`, `detours`, `dxgi`, `dxguid`, and `onecore`.

The resulting `aimdo_xpu.dll` imports at least `ze_loader.dll`, `sycl8.dll`,
`libmmd.dll`, the MSVC runtime, DXGI, and Windows system APIs. The build does
not bundle the Level Zero loader.

### Two environment defects that must be worked around

Both are properties of this recorded installation, not of the project. The
scripts handle them so a build does not depend on repairing the machine.

oneAPI's `setvars.bat` reports `'vars.bat' is not recognized` for every
component and leaves `icx-cl` off PATH, so the SYCL step fails with
`'icx-cl.exe' is not recognized`. `scripts/setup-oneapi-env.cmd` sets the
compiler PATH/INCLUDE/LIB directly from the installation layout, and
`build-windows-xpu.cmd` falls back to it automatically.

The Visual Studio Build Tools installation contains no Windows SDK, so Detours'
own `nmake` fails on `windows.h`. `scripts/build-windows-detours.cmd` applies
the same NuGet SDK environment used for AIMDO and builds only `src`, because
the sample tree additionally requires the .NET strong-name tool `sn`.

### Provisioning Detours

```powershell
git clone --depth 1 https://github.com/microsoft/Detours.git build\detours-src
cmd /d /c scripts\build-windows-detours.cmd
```

This produces `build\detours-src\lib.X64\detours.lib`. Set `DETOURS_LIB_DIR`
and `DETOURS_INCLUDE` to use an existing installation instead.

### Validating a locally built DLL

The Portable interpreter is an embeddable build whose `._pth` file disables
`PYTHONPATH`. Setting `PYTHONPATH` to the repository therefore does **not**
load a local build; the installed wheel is used instead, silently. A test
script must call `sys.path.insert()` itself, and a run whose conclusion depends
on which build was loaded should print the resolved DLL first:

```powershell
<portable>\python_embeded\python.exe -s -c ^
  "import comfy_aimdo.control as c, pathlib; print(pathlib.Path(c.__file__).parent)"
```

The alternative, used for the recorded result, is to install the wheel over the
previous one:

```powershell
<portable>\python_embeded\python.exe -s -m pip install ^
    --force-reinstall --no-deps build\wheel-detour-dev3\comfy_aimdo-*.whl
```

### Reproducible build commands

From the AIMDO repository root:

```powershell
cmd /d /c scripts\build-windows-xpu.cmd

$env:SETUPTOOLS_SCM_PRETEND_VERSION = "0.4.14.dev1"
build\python-venv\Scripts\python.exe -m build `
    --wheel `
    --no-isolation `
    --outdir build\wheel-windows-dev1
Remove-Item Env:SETUPTOOLS_SCM_PRETEND_VERSION
```

`--no-isolation` is intentional for the captured local build environment. The
version override is also intentional: without it, `setuptools-scm` derives a
repository-development version rather than the Portable milestone version.

The captured artifact was:

```text
comfy_aimdo-0.4.14.dev1-cp39-abi3-win_amd64.whl
SHA256 CD505BA225726B273553D918ECC81A93F01A58B7DE80BC6D05C40D97B004012F
```

Its `aimdo_xpu.dll` SHA256 was
`AC3D1D40E271032E50C83F313E9C3E9892D8DCEDEABAC2B704E8984218C91541`.
The installed Portable DLL was checked byte-for-byte against this build.

## 3. Portable test environment

### Hardware and driver

| Item | Captured value |
| --- | --- |
| GPU | two Intel Arc Pro B70 devices |
| Torch-reported memory | `33,454,690,304` bytes per device (about 31.15 GiB) |
| Windows display driver | `32.0.101.8515` on both devices |
| Default test device | `xpu:0`; explicitly record any `ZE_AFFINITY_MASK` override |

WMI `AdapterRAM` is not reliable for this device and must not be used as the
capacity value. Use Torch device properties and WDDM/DXGI Budget instead.

### Portable Python and packages

| Package | Captured version |
| --- | --- |
| Python | `3.13.12`, MSC `v.1944`, embedded/Portable |
| Torch | `2.12.0+xpu` |
| TorchVision | `0.27.0+xpu` |
| TorchAudio | `2.11.0+xpu` |
| `comfy-aimdo` | `0.4.14.dev3`, rebuilt from the source baseline above and installed over `0.4.14.dev1` |
| `comfy-kitchen` | `0.2.26`, XPU-managed installation |
| `omni_xpu_kernel` | `0.1.0b9.dev1+torch212.bmg` |
| `intel-sycl-rt` | `2025.3.2` |
| `intel-cmplr-lib-ur` | `2025.3.2` |
| `intel-opencl-rt` | `2025.3.2` |
| `aiohttp` | `3.14.1` |
| PyAV | `18.0.0` |
| NumPy | `2.4.4` |
| safetensors | `0.8.0` |
| psutil | `7.2.2` |
| ComfyUI frontend | `1.47.12` |
| ComfyUI embedded docs | `0.5.9` |
| ComfyUI workflow templates | `0.11.28` |

The build compiler is oneAPI `2025.3.3`, while the Portable SYCL/UR Python
runtime distributions are `2025.3.2`. This exact combination built and loaded,
but it is a recorded compatibility dependency, not proof that arbitrary
compiler/runtime combinations are supported.

The Portable runtime resolves:

* `sycl8.dll` from `python_embeded/Library/bin`, file version `2025.3.0.0`;
* `ur_loader.dll` from the same directory, file version `0.12.0`;
* Level Zero UR adapters from the same directory, file version `2025.3.0.0`;
* `ze_loader.dll` from the Windows installation/driver, version `1.26.1`.

### Model files used by the acceptance workflow

| Model file | Captured size |
| --- | ---: |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 bytes |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 bytes |
| `minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 bytes |
| `minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 bytes |

Hashes should be added before publishing a redistributable validation image;
filenames and sizes alone are not content identity.

### Launch policy

The tested `run_intel_gpu.bat` policy is:

```text
OMNIXPU_ENABLE=1
OMNI_XPU_REQUIRE_CUTE=0
OMNI_ATTN_BACKEND=torch
OMNIXPU_INTERPOLATE_FIX=0
OMNI_COMFYUI_RESERVE_VRAM_GB=4
--windows-standalone-build
--enable-dynamic-vram
--reserve-vram 4
```

The default attention backend is Torch/SDPA. ESIMD is an explicit diagnostic
option, not the default acceptance configuration. Do not enable high-volume
allocation, VBAR, or synchronization tracing during a performance run; trace
overhead invalidates timing comparisons.

To select one physical XPU, set `ZE_AFFINITY_MASK` before starting the process
and record its value with the result. Never change it between prompts in a
three-run series.

## 4. Validation hierarchy

Validation proceeds in this order. Passing a later-looking but smaller test
does not waive an earlier failure.

### Gate A: source and build checks

1. Native DLL and wheel build complete without an error.
2. The installed DLL hash matches the wheel/source artifact.
3. Portable imports `comfy_aimdo.control`, `comfy_kitchen`, and
   `omni_xpu_kernel`.
4. `torch.xpu.is_available()` is true and the intended device count/name is
   correct.
5. Focused tests run from the isolated build environment:

   ```powershell
   build\python-venv\Scripts\python.exe -m pytest `
       tests\test_windows_pressure_source.py `
       tests\test_xpu_backend.py
   ```

The captured result was `7 passed, 1 skipped`. The skipped test was
`tests/test_xpu_backend.py` because the isolated build environment deliberately
does not contain Torch. This is static/source coverage only; it is not an XPU
runtime pass. A skipped XPU runtime test must always be reported as skipped,
not passed.

### Gate B: component behavior

Run `tests/repro_xpu_platform_memory_policy.py` with the Portable Python. This
is a component check for the retained pressure policy, not proof of swap
liveness.

Two further reproducers cover the Windows allocation path. They run in seconds
and need no ComfyUI, workflow, or H3 model:

```powershell
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_malloc_pressure.py --mode budget --max-gib 40
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_malloc_pressure.py --mode thrash --working-set-gib 36
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_malloc_pressure.py --mode cache --working-set-gib 12
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_vbar_vs_torch.py --vbar-gib 6 --torch-gib 30
```

The first characterises the platform: a Torch USM allocation spills instead of
failing, an over-budget working set costs about nine times its bandwidth, and a
freed-but-cached Torch block keeps holding WDDM local memory.

The second is the regression gate for allocation-time arbitration. It must show
VBAR residency falling *during* the Torch ramp while the watermark stays at its
full value, with zero Torch allocation failures. A run where `unmap_calls`
stays 0 for the whole ramp means the hook is not intercepting; check which DLL
was loaded before drawing any other conclusion.

Neither reproducer replaces Gate C or Gate D, and neither exercises repeated
model activation, tiled VAE decode, or sustained multi-prompt residency. A
native test that performs real H2D, XPU consumption, retirement, eviction,
refault, and data verification under competing Torch allocations is still
missing.

**Component reproducers must not gate delivery on their own.** Three builds
passed every reproducer here and then failed the real workload within four
minutes, because none of these scripts exercise the file-streaming path that
loads a missed weight. A memory-policy change is validated only by a workload
that demonstrably reaches pressure, evidenced by a non-zero reclaim count and a
peak local usage that approaches the configured target in the same log.

The 2-step profile does not reach pressure on this hardware; the 20-step
profile does. Use `--steps 20` when validating anything that touches pressure,
reclaim, or residency:

```powershell
<portable>\python_embeded\python.exe -s `
    tests\run_windows_h3_acceptance.py --server http://127.0.0.1:8189 `
    --output-root <portable>\ComfyUI\output --runs 3 --steps 20
```

### Gate C: development workload

Use MiniMax H3 T2V at 864x480, 124 frames at 24 fps (about five seconds), two
sampler steps, and three sequential prompts with distinct seeds in one ComfyUI
process. Do not restart ComfyUI or clear caches between runs.

This gate is fast feedback only. It cannot replace the formal workload.

### Gate C2: pressure gradient, mandatory before any root-cause claim

Gate C at two sampler steps never reaches memory pressure, and neither do the
smaller resolutions at twenty steps. A memory-policy change that is validated
only there has not been validated at all. Three separate builds in this project
were declared fixed on the strength of a passing component reproducer and a
passing small prompt, and each then failed the real workload.

Run the ladder in order, at `--steps 20`, one run per step, and stop at the
first failure:

| Step | Command arguments |
| --- | --- |
| 1 | `--width 864 --height 480 --frames 124 --steps 20 --runs 1` |
| 2 | `--width 1280 --height 736 --frames 124 --steps 20 --runs 1` |
| 3 | `--width 1280 --height 736 --frames 243 --steps 20 --runs 1` |
| 4 | `--width 1280 --height 736 --frames 362 --steps 20 --runs 3` |

Step 3 is the lowest configuration observed to reach real pressure on this
device with this model, so it is the earliest point at which a result carries
any weight. Steps 1 and 2 are regression guards; passing them proves only that
nothing obvious broke.

MiniMax H3 requires frame counts of the form 17k+5, which gives 124 frames for
five seconds, 243 for ten, and 362 for fifteen. Other values are rejected.

Two failure modes must both be treated as failures. An error is obvious. A
stall is not: check that the sampler progress bar advances, and confirm that
`host_to_device_bytes` grows while `map_bytes` and `unmap_bytes` do not grow at
the same rate as each other. Equal map and unmap rates mean pages are being
mapped and immediately released, which presents as a hang at step 0 with the
process at full CPU.

### Gate D: formal workload

Use `tests/run_windows_h3_acceptance.py`:

```powershell
<portable>\python_embeded\python.exe -s `
    tests\run_windows_h3_acceptance.py `
    --server http://127.0.0.1:8188 `
    --output-root <portable>\ComfyUI\output `
    --runs 3
```

The fixed workload is:

* MiniMax H3 text-to-video;
* 1280x736;
* 362 frames at 24 fps (15.083 seconds);
* two sampler steps;
* three distinct seeds;
* one unchanged ComfyUI process for all runs.

The script's four-hour per-prompt timeout is only a dead-process safety limit.
A prompt that eventually finishes after a large performance collapse has
failed acceptance.

## 5. Formal pass/fail criteria

All four groups below must pass.

### Output correctness

1. All prompts report `execution_success`.
2. Every output is a non-empty MP4 at 1280x736, 24 fps, 362 frames, and
   15.083 seconds within one frame of duration tolerance.
3. The changed seed forces the sampler to execute for every prompt. Static
   loader nodes may remain cached.

`ACCEPTANCE_PASS` from the client proves only this automated execution/media
gate. It does not prove the remaining groups.

### Performance stability

1. Record total prompt time and sampler-only seconds per step for every run.
2. Run 2 and run 3 must not show a systematic slowdown. The working threshold
   is at most 10% sampler-time regression relative to the fastest completed
   run in the same three-run series.
3. Model initialization and VAE encode/decode time are recorded separately so
   they cannot hide sampler degradation.
4. A long idle interval or a multi-minute step is a failure even if the prompt
   later completes.

If a different variance threshold is adopted, it must be agreed before the
run and recorded with the result; it must not be relaxed after seeing a
failure.

### WDDM memory stability

1. Sample the selected adapter's local `CurrentUsage` and `Budget`, plus
   non-local usage, for the complete three-run window.
2. Local usage must not remain above WDDM Budget. The configured operating
   target is `Budget - 4 GiB` for this launch policy.
3. Non-local usage must not grow monotonically across prompts as a substitute
   for local device memory. Treat more than 512 MiB sustained growth above the
   pre-run baseline as a failure requiring diagnosis.
4. After each prompt, memory must return to a bounded recurring state; run 3
   must not retain another prompt-sized tranche over run 2.

The output client does not collect these values. They require the same-window
server trace or an external DXGI monitor.

### Liveness and error freedom

1. No `result=999`, null-copy error, `Fault failed: 2`, OOM,
   `UR_RESULT_ERROR_DEVICE_LOST`, traceback, or driver reset.
2. No interval of at least 60 seconds where all available progress indicators
   are unchanged: sampler progress, GPU activity, VBAR/copy counters, and log
   position.
3. Every traced wait/fault/copy begin record has a matching end record. A
   timeout or unmatched record is a failed run.
4. Windows System events for the test window contain no display-driver reset
   or device-removal event for the selected adapter.

When a stall is suspected, capture py-spy and WDDM state before terminating the
process. The capture diagnoses a failed run; it does not convert it into a
pass.

## 6. Result record

Each result should include:

* all revisions and package versions listed above;
* selected device and `ZE_AFFINITY_MASK`;
* launch environment and command-line arguments;
* prompt seeds and output media metadata;
* total, initialization, sampler, and VAE times per run;
* local usage/Budget and non-local baseline/peak/final values;
* relevant VBAR, copy, physical allocation/release, and synchronization
  counters;
* the exact server log interval;
* a final status for each of the four acceptance groups.

Do not summarize a series as passed when only the output/media group passed.
