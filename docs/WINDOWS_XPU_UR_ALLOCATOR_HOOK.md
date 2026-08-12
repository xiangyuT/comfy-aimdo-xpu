# Windows XPU Unified Runtime allocator hook

Status: **implemented, component-verified, and blocked by a performance
regression** on branch `dev/windows-xpu-native-allocator-hook`, which is based
on the Linux `dev/xpu-native-allocator-hook` work with the Windows platform
series replayed on top. The interception mechanism, the PyTorch retry contract
and the production control loop are each verified on the recorded Windows
environment. The gradient ladder is **suspended**: step 1 completes 56 % slower
than the recorded baseline and step 2 regressed and was stopped, for reasons
that are not yet explained; see section 10. This branch is not a candidate for
delivery in its current state. Per the maintenance rule in
[the liveness analysis](WINDOWS_XPU_VBAR_LIVENESS_ANALYSIS.md), that document's
unresolved 720p/10s failure is *not* erased by anything recorded here.

This record answers one question: can the Linux
`dev/xpu-native-allocator-hook` design, which interposes Unified Runtime USM
allocation entry points, be applied to Windows?

The answer is yes, with one mechanism substitution, one newly discovered hard
constraint, and one structural advantage over the Windows design currently in
tree.

## 1. What the Linux branch does

`src-xpu/ur-usm-hook.cpp` on `dev/xpu-native-allocator-hook` builds a shared
object that exports `urUSMDeviceAlloc` and `urUSMFree` in the
`LIBUR_LOADER_0.12` version namespace, is injected with `LD_PRELOAD`, and
recovers the real entry points with `dlvsym(RTLD_NEXT, ...)`. PyTorch keeps its
native caching allocator. The hook:

1. classifies the caller by walking the stack for `libc10_xpu.so`, so it can
   tell a PyTorch caching-allocator segment request from a direct SYCL
   allocation;
2. asks AIMDO for a budget deficit before the allocation is placed;
3. when a PyTorch request is over budget, returns a **synthetic**
   `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` without calling the real allocator;
4. lets PyTorch respond by releasing its cached blocks and retrying, then
   performs the eviction and allows the retry through.

Step 3 and 4 are the whole point. They convert PyTorch's own cache into
reclaimable memory at the exact moment of pressure.

## 2. The Windows call path is the same, the injection method is not

Windows has no `LD_PRELOAD`, and PE has no symbol versioning. What matters is
whether the *same function* is still the one PyTorch's allocations flow
through. It is.

Evidence, taken from the binaries in the recorded Portable environment:

| Fact | Method | Result |
| --- | --- | --- |
| `ur_loader.dll` exports flat `urUSMDeviceAlloc`, `urUSMFree` | PE export table | present, among 627 exports |
| `sycl8.dll` statically imports any `ur*` symbol | PE import table | **no** |
| `sycl8.dll` imports from `ur_win_proxy_loader.dll` | PE import table | exactly one symbol, `?getPreloadedURLib@@YAPEAXXZ` |
| `sycl8.dll` contains the string `urUSMDeviceAlloc` | binary scan | yes, 1 occurrence |
| `sycl8.dll` contains the string `urGetUSMProcAddrTable` | binary scan | **no**, 0 occurrences |

So on Windows the SYCL runtime obtains the `ur_loader.dll` module handle from
`ur_win_proxy_loader.dll!getPreloadedURLib()` and then resolves the **flat
exported symbol by name with `GetProcAddress`**. It does not consume the DDI
proc-address tables. The pointer it holds points directly into
`ur_loader.dll`'s code section.

This rules out the failure mode that would have killed the port: if SYCL had
consumed a DDI table populated with the loader's internal functions, patching
the exported symbol would have intercepted nothing.

### Injection: Detours, and it is strictly easier than LD_PRELOAD

The correct substitution is Microsoft Detours, patching the prologue of
`ur_loader.dll!urUSMDeviceAlloc` in the loaded image. `src-xpu/ze-detour.c`
already does exactly this for `ze_loader.dll`, so no new tooling is required.

Detours is not merely an adequate replacement, it is a better one here:

* Inline patching rewrites the function body, so a pointer SYCL captured
  *before* the patch still lands in the hook. The Linux prototype must be
  loaded before the process starts and `control.py` currently rejects the
  configuration otherwise; on Windows the hook can be installed from
  `control.init()` at any point before the allocations that matter.
* Nothing has to be exported from `aimdo_xpu.dll`, so `src-xpu/ur-usm-hook.map`
  needs no Windows equivalent and there is no risk of AIMDO's exports colliding
  with the loader's.

### Resolving the right module

`GetModuleHandleA("ur_win_proxy_loader.dll")` returns NULL in a live PyTorch
XPU process even though the module *is* loaded; the proxy must be found by
walking the module list. `GetModuleHandleA("ur_loader.dll")` does resolve.

The probe therefore does both and compares them. Measured result:
`getPreloadedURLib()` returns `0x7ffc1a760000`, and `GetModuleHandleA(
"ur_loader.dll")` returns the same value. The by-name lookup is correct here,
and the proxy call is retained as the authoritative cross-check because a
process may in principle hold more than one `ur_loader.dll`.

## 3. The experiment

Built by `scripts/build-windows-ur-probe.cmd` from
`tests/ur_usm_detour_probe.c`; driven by `tests/run_windows_ur_hook_probe.py`.
The probe only counts, classifies and can inject one synthetic failure. It
never evicts and never calls into AIMDO, so a pass says nothing about memory
policy - only about mechanism.

```powershell
scripts\build-windows-ur-probe.cmd
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_probe.py
```

Environment: Intel Arc Pro B70, driver `32.0.101.8515`, ComfyUI Portable with
`torch 2.12.0+xpu`, Python 3.13, `ur_loader.dll` and `sycl8.dll` from
`python_embeded\Library\bin`, `ze_loader.dll` from the driver.

### Result: 12/12 checks pass, reproduced on two consecutive runs

| Check | Result | Evidence |
| --- | --- | --- |
| 1. Detours attaches in a live process | PASS | hooked `python_embeded\Library\bin\ur_loader.dll` after `import torch` |
| 1b. hooked module is the one SYCL uses | PASS | `resolved_via_proxy=1`, `proxy_agrees_with_byname=1` |
| 2. torch allocations reach the entry point | PASS | `alloc_calls` advanced |
| 3. caller classified via `c10_xpu.dll` | PASS | stack below |
| 3b. request size matches the tensor | PASS | `last_alloc_size=536870912` |
| 3c. a non-torch caller is not misread | PASS | replayed allocation counted as `other`, not `torch_native` |
| 4a. torch holds cached blocks | PASS | 384 MiB cached |
| 4b. synthetic OOM delivered | PASS | `synthetic_oom_calls=1` |
| 4c. torch released its cache in response | PASS | `frees_between_oom_and_retry=6`, 384 MiB |
| 4d. torch retried the same request | PASS | `retry_alloc_calls=1` |
| 4e. the allocation succeeded | PASS | no exception raised |
| 4f. the memory is usable | PASS | write/read back verified |

The intercepted stack, which is the direct Windows analogue of the Linux
`libc10_xpu.so` check:

```text
ur_usm_detour_probe.dll <- sycl8.dll <- sycl8.dll <- sycl8.dll
  <- c10_xpu.dll <- c10_xpu.dll <- c10_xpu.dll <- c10_xpu.dll
  <- c10.dll <- torch_cpu.dll <- ... <- torch_python.dll <- python313.dll
```

`RtlCaptureStackBackTrace` plus `GetModuleHandleExW(
GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, ...)` is the working equivalent of
`backtrace()`/`dladdr()`. It uses x64 unwind tables, not frame pointers, so it
does not depend on how the intervening modules were optimised.

The negative control matters: the probe replays a device allocation through the
public entry point from its own module, using the context and device handles
captured from a real PyTorch request. That call is classified `other`, so check
3 is discriminating rather than vacuous.

### Claim 4 is the load-bearing result

`XPUCachingAllocator::malloc` reacts to a failed `alloc_block` by calling
`release_cached_blocks()` and retrying once before raising. The measurement
confirms every step on Windows: one synthetic failure, six real `urUSMFree`
calls totalling 384 MiB in between, one retry of the identical size, success,
and usable memory.

This is significant beyond porting, but its scope is narrower than it first
appears, and section 9 records the measurement that establishes the limit.
`XPU_PLATFORM_MEMORY_POLICY.md` records that PyTorch's freed-but-cached blocks
"never reach `zeMemFree`, so the allocation hook cannot see them", which is why
Windows needs `AIMDO_XPU_NATIVE_CACHE_TRIM` to call `empty_cache()` from the
fault boundary. The synthetic OOM addresses that **only while PyTorch is the
allocator**. It does not help when AIMDO's own VBAR fault needs memory, because
a fault allocates no USM and the hook never runs.

## 4. Newly discovered hard constraint: expandable segments

`PYTORCH_ALLOC_CONF=expandable_segments:True` was tested because it was
identified as a risk to the retry contract. The measured result is more severe
than a broken retry.

| Configuration | `urUSMDeviceAlloc` | `urPhysicalMemCreate` | `urVirtualMemMap` |
| --- | ---: | ---: | ---: |
| default | 11 calls, 10 of them PyTorch | 0 | 0 |
| `expandable_segments:True` | **0 calls** | 53 calls, 1.02 GiB | 53 calls |

With expandable segments enabled, PyTorch does not allocate USM at all. It
reserves virtual address space and backs it with physical memory objects, the
same primitive family AIMDO's own VBAR uses. A USM hook is not merely degraded,
it is **completely blind**, and every check that depends on seeing an
allocation fails.

Consequences for the port:

* The design is valid for the default configuration, which is expandable
  segments off.
* AIMDO must detect the setting and refuse to claim arbitration rather than
  silently observing nothing. Silent blindness is the failure mode this
  investigation has already been burned by twice.
* If expandable segments must ever be supported, `urPhysicalMemCreate` is the
  equivalent interception point and it carries the size argument.

## 5. Why this addresses the two recorded Windows root causes

### Re-entrant reclaim, the cause of `DEVICE_LOST`

`ze-detour.c` documents that releasing a VBAR page from inside
`zeMemAllocDevice` re-enters Level Zero's own memory management and corrupts
driver state, which is why reclaim was moved to *after* the driver call
returns. That ordering compromise is the reason arbitration is late.

A UR hook sits above the driver:

```text
ur_loader.dll!urUSMDeviceAlloc      <- AIMDO decides here
  -> ur_adapter_level_zero.dll
    -> ze_loader.dll!zeMemAllocDevice   <- driver allocation happens here
```

When the hook returns a synthetic failure it has not called the adapter at all,
and when it evicts on the retry it does so between two separate UR calls with
no driver allocation in flight on that thread. `zeVirtualMemUnmap` and
`zePhysicalMemDestroy` are then ordinary calls, not re-entrant ones. The
ordering gap closes without reintroducing the corruption.

### Blocking reclaim, the cause of the removed allocator experiment's failure

The earlier "removed allocation-time allocator experiment" fixed the ordering
and then failed because it reclaimed by synchronising the whole queue
(`cuCtxSynchronize` -> `sycl::queue::wait_and_throw`), which cannot complete a
formal prompt.

That is already solved in tree, separately: `vbars_free_retired()` releases only
epoch-retired pages and returns immediately. The two halves have never been
combined, because they live on different branches. This is the first design in
which allocation-time ordering and non-blocking reclaim can both hold.

## 6. Branch strategy

The two lines of work did not share history:

| Work | Where it was |
| --- | --- |
| UR USM hook, synthetic OOM, retry contract | `origin/dev/xpu-native-allocator-hook`, based on `origin/master`, containing **no** Windows code |
| Windows Detours, epoch retirement, WDDM reserve, VBAR reopen | `dev/windows-xpu-usm-free-hang`, 13 commits on the same base |

`origin/master` is exactly the merge base of both, so the fork is clean. The
chosen strategy is to treat the Linux branch as the new base and replay the
Windows platform series onto it:

```text
origin/dev/xpu-native-allocator-hook   (Linux UR hook, new base)
  + 13 replayed Windows commits        (platform support, unchanged intent)
  + Windows UR arbitration             (this design)
= dev/windows-xpu-native-allocator-hook
```

Only four files needed conflict resolution: `README.md`,
`comfy_aimdo/control.py`, `src-xpu/dispatch.cpp` and `src-xpu/stubs.c`. The
two real conflicts were both additive - the native-hook teardown next to the
Windows `torch.xpu.empty_cache()` teardown, and `device_from_native_handle()`
next to `aimdo_xpu_note_queue()` - and both sides were kept.

The old `dev/windows-xpu-usm-free-hang` branch is reference material only. It
is not merged and should not be.

## 7. What was implemented

`src-xpu/ur-usm-detour.c` is the Windows counterpart of
`src-xpu/ur-usm-hook.cpp`. It exports the same four control symbols, so
`comfy_aimdo/control.py` drives both platforms through one interface:

| Symbol | Linux meaning | Windows meaning |
| --- | --- | --- |
| `xpu_ur_hook_is_interposed` | the dynamic linker bound the symbol to AIMDO | Detours patched the entry point, attaching on demand |
| `xpu_ur_hook_enable` | begin arbitrating | begin arbitrating |
| `xpu_ur_hook_disable` | stop, refusing while tracked segments are live | same |
| `xpu_ur_hook_get_stats` | 19 counters | the same 19, plus `physical_mem_create_calls` |

The hook body mirrors the Linux logic: pass through when disabled, resolve the
AIMDO device through `urDeviceGetNativeHandle`, ask
`aimdo_xpu_allocation_deficit()`, classify the caller, return a synthetic
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` for an over-budget PyTorch request, and
evict on the matching retry. The shared C ABI
(`aimdo_xpu_allocation_deficit`, `aimdo_xpu_evict_for_allocation`,
`aimdo_xpu_account_allocation`, `xpu_device_from_native_handle`) is used
unchanged.

Three Windows-specific decisions:

* `aimdo_xpu_evict_for_allocation()` routes to `vbars_free_retired()` on
  Windows. Nothing on an allocation path may wait on the compute queue, and
  that queue may be blocked behind work needing the residency being requested.
  Linux keeps its exact `vbars_free()` because its arbitration must satisfy the
  request.
* `ze-detour.c` hands arbitration over. When the UR hook attaches, the
  `zeMemAllocDevice` detour stops sampling and reclaiming and only accounts, so
  two control loops never run against the same pressure. If the UR hook fails
  to attach, the old post-allocation behaviour remains as the fallback, and
  `AIMDO_XPU_DISABLE_UR_HOOK=1` forces it.
* `control.py` defaults Windows to `native_hook` mode and **refuses to start**
  when `expandable_segments` is enabled, rather than attaching a hook that
  would observe nothing.

## 8. Production verification

`tests/run_windows_ur_hook_smoke.py` exercises the real
`comfy_aimdo.control` path rather than a standalone probe.

Without pressure, with a 512 MiB reserve and three 512 MiB tensors:

| Check | Result |
| --- | --- |
| Windows selects `native_hook` | PASS |
| torch allocations reach AIMDO's hook | PASS, 3 calls |
| tracked and accounted, not passed through | PASS, 1536 MiB, `pass_through_alloc_calls=0` |
| device resolved for every request | PASS, `unknown_device_calls=0` |
| releases observed and credited | PASS, 1536 MiB |
| accounting balances after `empty_cache()` | PASS, alloc bytes == free bytes exactly |
| clean detach | PASS |

Under real budget pressure, with a 28 GiB reserve on a 31 GiB device and three
2 GiB tensors, the complete arbitration loop runs:

| Counter | Value |
| --- | ---: |
| `alloc_calls` | 5 |
| `synthetic_oom_calls` | 2 |
| `retry_eviction_calls` | 2 |
| `retry_eviction_bytes` | 8.91 GiB |
| `tracked_alloc_calls` | 3 |
| `pass_through_alloc_calls` | 0 |

Two over-budget requests were refused, PyTorch retried both, eviction ran on
the retry, and all three allocations ultimately succeeded with balanced
accounting. That is the full path - deficit detection, caller classification,
synthetic failure, PyTorch retry, eviction, allocation - running in production
code.

## 9. Real pressure test, and the claim it refuted

The verification above creates pressure with an inflated reserve on a device
holding no VBAR working set, so it proves the control loop executes and nothing
more. `tests/repro_windows_xpu_vbar_vs_torch.py` is the real test: a 6 GiB VBAR
working set of 192 pages competing with a 30 GiB native Torch ramp on a 31 GiB
device.

```powershell
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_vbar_vs_torch.py --vbar-gib 6 --torch-gib 30
```

| Phase | VBAR resident | Torch cached | unmaps | synthetic OOM | retry evictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| idle device | 6144 MiB | 0 | 0 | 0 | 0 |
| Torch ramp applied | 0 MiB | 30720 MiB | - | 71 | 71 |
| VBAR faults while Torch holds memory | 32 MiB | 30720 MiB | 383 | 71 | 71 |
| after `del blocks`, no `empty_cache()` | 32 MiB | 30720 MiB | 575 | 71 | 71 |
| after `torch.xpu.empty_cache()` | 6144 MiB | 256 MiB | 575 | 71 | 71 |

### What passed

Allocation-time arbitration works under real competition. The hook fired 71
times during the Torch ramp and evicted on every retry, releasing the whole
6 GiB VBAR ahead of Torch's growth. Torch recorded **zero** allocation
failures, and there was no `DEVICE_LOST` and no driver instability - which is
the specific failure that the `zeMemAllocDevice` arbitration could not avoid.

### What failed: the synthetic OOM does not subsume the cache trim

Section 3 of an earlier revision of this document claimed the synthetic OOM
"removes that entire workaround" and that `AIMDO_XPU_NATIVE_CACHE_TRIM` could
therefore be retired. **That claim is false, and this measurement refutes it.**

In the fourth phase Torch holds 30,720 MiB in its own cache and AIMDO's VBAR
faults recover only 32 MiB. The synthetic OOM counter does not move at all
between the third and fourth phases: it stays at 71. `empty_cache()` then
recovers the full 6144 MiB immediately.

The reason is structural rather than a tuning problem. The hook lives inside
`urUSMDeviceAlloc`, so it only runs when **PyTorch** is allocating. A VBAR
fault allocates no USM at all - it creates Level Zero physical memory and maps
it - so when AIMDO is the party that needs memory and Torch is the party
holding it, the hook is never entered and Torch is never asked to release
anything. `AIMDO_XPU_NATIVE_CACHE_TRIM` must therefore be **retained**; it
covers the exact direction the hook cannot.

`native_reclaim_free_calls` also stayed at 0 for the whole run. During the ramp
PyTorch was allocating fresh blocks and had no cache to release, so every
synthetic failure was answered by AIMDO evicting VBAR pages rather than by
Torch returning memory. The cache-return benefit demonstrated by the standalone
probe did not occur even once in this workload.

### What is unchanged: the map/unmap thrash

The third phase issues 383 unmaps for 192 pages in a single pass and settles at
32 MiB resident. That is the same signature as the livelock recorded in the
liveness analysis, and this branch does not change it. Moving the interception
point does not address admission or watermark policy, exactly as predicted.

### A defect this test found

`xpu_ur_hook_disable()` originally refused while tracked segments were live,
copying the Linux invariant. That invariant does not transfer: on Linux AIMDO
*is* Torch's allocator, so a live segment means its own accounting would be
lost, while on Windows the hook merely observes allocations PyTorch owns and
legitimately retains across `empty_cache()`. The result was that
`control.deinit()` raised in any process that had ever allocated. Windows now
drops the tracking table and reports the count instead.

## 10. Performance regression across the gradient ladder

Recorded as an open defect that blocks delivery, not explained.

Gate C2 step 1 (`--width 864 --height 480 --frames 124 --steps 20 --runs 1`)
passed and produced a valid 124 frame video, but it is materially slower than
the recorded baseline for the same configuration. Step 2
(`--width 1280 --height 736 --frames 124 --steps 20 --runs 1`) showed the same
regression on observation and was stopped before completion, so it has no
recorded time.

| Build | Gate C2 step 1 | Gate C2 step 2 | Source |
| --- | ---: | ---: | --- |
| `1e14a29`, Level Zero arbitration | 169 s | 706 s | liveness analysis, "Why the earlier gradient steps passed" |
| this branch, UR arbitration, `0.4.15.dev1` | 264 s | stopped, regressed | `build/ur-integration/ladder1.log` |

Step 1 is roughly +56 % on a configuration that does not even reach the
pressure point. Both runs used `--reserve-vram 4` on the same device, driver and
model, so the configuration matches; the whole stack differs by more than one
change, however, so the comparison bounds the regression rather than
attributing it.

The ladder was abandoned at step 2. Continuing it would only produce more
slow passes, and a slow pass at step 3 could not be distinguished from the
stall this branch is supposed to address.

Two candidate causes, neither yet confirmed:

1. **`release_cached_blocks()` is all-or-nothing.** PyTorch answers a failed
   `alloc_block` by releasing *every* cached block, not the amount that was
   short, and only then retries. Each synthetic failure therefore flushes the
   whole caching allocator, and every later tensor must go back to the driver.
   The VBAR reproducer recorded 71 synthetic failures during a single 30 GiB
   ramp, which on this theory is 71 full cache flushes. This is a property of
   the design, not a tuning constant, and it is the more likely of the two.
2. **The caller classification is expensive.**
   `is_torch_native_segment_request()` runs `RtlCaptureStackBackTrace` over up
   to 48 frames and calls `GetModuleHandleExW` plus `GetModuleFileNameW` on
   each one, formatting a path per frame. It runs whenever the deficit is
   positive, which is exactly the hot path under pressure. The Linux hook has
   the same shape but `dladdr()` is considerably cheaper than resolving and
   formatting a module path. This one has an obvious fix: resolve the
   `c10_xpu.dll` image range once and compare frame addresses against it,
   removing every per-frame Win32 call.

Neither has been measured. The next step is to separate them, because they
imply different responses: (2) is an implementation cost that can simply be
removed, while (1) would mean the synthetic-OOM design needs a way to bound how
much cache PyTorch discards, or a policy that only fails requests when the
deficit is large enough to be worth a full flush.

Until this is understood, no timing on this branch should be compared against
the historical figures, and a later gradient step that "passes slowly" must not
be read as a pass.

## 11. Remaining plan

Phase A - **done, and it changed the design.** See section 9.

Phase B - `AIMDO_XPU_NATIVE_CACHE_TRIM` is **retained**, not retired. The two
mechanisms cover opposite directions and both are required: the synthetic OOM
when PyTorch allocates, the fault-boundary trim when AIMDO faults. What remains
is to check they do not oscillate against each other under sustained pressure.

Phase C - separate the two regression candidates in section 10. Remove the
per-frame module lookups from the classification first, since that is cheap and
unambiguous, then re-measure step 1. Whatever remains is attributable to the
cache-flush behaviour.

Phase D - investigate whether the fault path needs its own way to demand memory
from PyTorch, since the trim is rate limited and best effort. The measured
32 MiB against 30 GiB of cache suggests the current trim is far too weak at
that boundary.

Phase E - the gradient ladder, in full. Step 1 passes but is 56 % slower than
baseline, step 2 regressed and was stopped, and the ladder is suspended until
Phase C explains the regression. Only step 3 or above may be reported as a fix.

## 12. What this does not establish

1. **Gate C2 is incomplete and step 1 shows a 56 % regression** (section 10).
   The 720p/10s livelock recorded in the liveness analysis is an admission and
   watermark policy problem, and section 9 shows the map/unmap thrash is still
   present on this branch. Presenting any of this as a fix for that failure
   would be the third occurrence of the method failure that document records -
   and section 9 is itself a fourth occurrence caught early, because this
   document did claim the cache trim was subsumed before it was measured.
2. `native_reclaim_free_calls` has never been observed nonzero outside the
   standalone probe. The mechanism is proven to exist; its usefulness in a real
   allocation pattern is not.
3. Eviction was exercised against a synthetic VBAR working set of uniform
   32 MiB pages, not a real model with real access order.
4. Multi-threaded attach was not exercised. `DetourUpdateThread` was called for
   the installing thread only, which is safe when hooks are installed before the
   worker threads allocate but is not a validated policy for attaching to a busy
   process.
5. Only one device was measured, and only the default allocator configuration.
6. Interaction with other software that hooks the same UR or Level Zero entry
   points was not tested.
7. The Linux build was not compiled on this machine. The change to
   `aimdo_xpu_evict_for_allocation()` is guarded so Linux keeps `vbars_free()`,
   but that has not been verified by a Linux build.

## 12. Reproducing

```powershell
scripts\build-windows-xpu.cmd
scripts\build-windows-ur-probe.cmd

rem mechanism, standalone probe, no AIMDO involved
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_probe.py

rem production control loop
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_smoke.py
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_smoke.py `
    --reserve-mib 28000 --alloc-mib 2048

rem real competition between a VBAR working set and a native Torch ramp
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_vbar_vs_torch.py --vbar-gib 6 --torch-gib 30

rem integration: install into Portable first, then run the ladder
$env:SETUPTOOLS_SCM_PRETEND_VERSION = "0.4.15.dev1"
build\python-venv\Scripts\python.exe -m build --wheel --no-isolation `
    --outdir build\wheel-ur-hook-dev1
Remove-Item Env:SETUPTOOLS_SCM_PRETEND_VERSION
<portable>\python_embeded\python.exe -s -m pip install `
    --force-reinstall --no-deps build\wheel-ur-hook-dev1\comfy_aimdo-*.whl
<portable>\python_embeded\python.exe -s tests\run_windows_h3_acceptance.py `
    --server http://127.0.0.1:8188 --output-root <portable>\ComfyUI\output `
    --width 864 --height 480 --frames 124 --steps 20 --runs 1

rem the expandable segments constraint, which must fail
$env:PYTORCH_ALLOC_CONF = "expandable_segments:True"
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_probe.py
```

The installed DLL must hash identically to `comfy_aimdo\aimdo_xpu.dll` from the
build, and the server log must contain `arbitrating Unified Runtime USM
allocations` together with the expected version, or the run is testing a
different build. Both were confirmed for the measurements recorded here.

The first three must print `RESULT: PASS`. The reproducer must complete without
a fault error and reproduce the table in section 9, including the fourth phase
in which VBAR residency does *not* recover. The last must fail checks 2, 3, 3c
and 4b-4d while reporting nonzero `physical_mem_create_calls`; that failure is
the expected, documented behaviour, and `control.init()` refuses the
configuration outright.

## Maintenance rule

The same rule as the liveness analysis applies. Append revision-specific
evidence; do not let a component pass overwrite a workload failure.
