# Windows XPU Unified Runtime allocator hook

Windows uses AIMDO's `native_hook` mode by default. PyTorch retains its native
XPU caching allocator; AIMDO attaches Detours to the Unified Runtime loader
used by SYCL and arbitrates physical USM growth at `urUSMDeviceAlloc` while
accounting releases at `urUSMFree`.

Linux and Windows expose the same control interface, but attach differently:

- Linux `native_hook` is an explicit mode and must be interposed before Python
  starts;
- Windows resolves the `ur_loader.dll` selected by
  `ur_win_proxy_loader.dll` and attaches Detours during AIMDO initialization;
- Linux `global` remains the default Linux mode and replaces PyTorch's XPU
  allocator with AIMDO's pluggable allocator;
- Windows never replaces PyTorch's allocator, including when the hook is
  disabled for diagnosis.

## Initialization contract

Import Torch before initializing AIMDO so the Unified Runtime loader exists.
ComfyUI must explicitly enable DynamicVRAM on XPU:

```text
--enable-dynamic-vram --reserve-vram <GiB>
```

Standalone callers may select the mode explicitly:

```python
from comfy_aimdo import control

control.init(
    implementation="xpu",
    simple_vram_headroom=4 << 30,
    xpu_allocator_mode="native_hook",
)
control.init_devices([0])
```

`init()` attaches or verifies the hook. `init_devices()` publishes the active
SYCL queues and enables arbitration only after the device contexts are ready.
`deinit()` drains Torch's cache, synchronizes, disables arbitration, and then
tears down AIMDO state. PyTorch remains the owner of its allocations for the
whole process.

`expandable_segments` is rejected in `native_hook` mode. That PyTorch path
backs virtual reservations with `urPhysicalMemCreate` and bypasses
`urUSMDeviceAlloc`, so accepting it would silently disable arbitration.

## Allocation policy

The hook distinguishes PyTorch cache growth from direct SYCL callers.

For a PyTorch allocation that would exceed the current budget:

```text
urUSMDeviceAlloc request
  -> sample deficit without waiting
  -> if Torch has reclaimable cached bytes, return one synthetic OOM
  -> PyTorch releases its cache and retries
  -> reclaim only the residual deficit from unpinned VBAR pages
  -> call the real Unified Runtime allocation
  -> account the physical segment
```

For a direct non-PyTorch SYCL allocation, AIMDO may reclaim eligible VBAR
pages but does not inject a synthetic failure into an unknown caller.

The hook body must never wait on a SYCL queue and must not perform re-entrant
driver memory management. Reclaim is best effort and limited to pages whose
retirement is already proven by previously submitted queue barriers. Windows
USM can spill to WDDM non-local memory; explicit VBAR physical pages cannot,
so VBAR yields first when pressure requires it.

The hook covers the direction where PyTorch requests memory. It does not run
when an AIMDO VBAR fault needs memory while Torch holds freed blocks in its
cache. The rate-limited fault-boundary native-cache trim remains necessary for
that reverse direction.

## Configuration

| Setting | Meaning |
| --- | --- |
| `AIMDO_XPU_ALLOCATOR_MODE=native_hook` | Select the retained-native-allocator path explicitly |
| `AIMDO_XPU_DISABLE_UR_HOOK=1` | Disable the Windows UR hook and use the diagnostic fallback path |
| `AIMDO_XPU_NATIVE_CACHE_TRIM=0` | Disable fault-boundary `torch.xpu.empty_cache()` recovery |
| `AIMDO_XPU_ENABLE_ALLOCATION_TRACING=1` | Enable the heavier Level Zero tracing path for diagnosis only |
| `AIMDO_XPU_ALLOCATION_TRACE=1` | Log native allocation activity |

Do not enable allocation tracing for performance measurements.

## Counters and timing

`control.get_xpu_ur_hook_stats()` reports allocation/free calls, tracked and
pass-through bytes, synthetic/runtime OOMs, native-cache reclaim, residual
VBAR eviction, unknown devices/frees, direct pressure, and metadata failures.
Windows additionally reports physical-memory-create calls and cache-lever
skips.

`control.get_xpu_ur_hook_timing()` reports total time inside the hook and
caller-classification time separately. Hook time includes the real driver call
being wrapped; use the decision counters to distinguish hook overhead from the
downstream cost of cache release or VBAR eviction.

## Validation

The no-device unit test covers attach lifecycle, caller classification,
synthetic retry, accounting, multi-thread interception, and timing:

```powershell
cmd /d /c scripts\test-windows-xpu-hook.cmd
```

The production smoke and VBAR competition checks require a Portable XPU
runtime:

```powershell
<portable>\python_embeded\python.exe -s tests\run_windows_ur_hook_smoke.py
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_vbar_vs_torch.py --vbar-gib 6 --torch-gib 30
```

Verify the loaded package/DLL identity before interpreting the counters. A
real workload is still required for any end-to-end performance or liveness
claim.

## Development history

The prototype sequence, rejected designs, measured counter tables, invalidated
performance attributions, and remaining hypotheses are preserved in
`omni-xpu-kernel-tuning/docs/results/bmg/2026-08-12/`. This file is only the
current implementation contract.
