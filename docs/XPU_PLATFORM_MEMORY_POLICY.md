# XPU platform memory policy

Intel XPU uses one VBAR priority/fault model on Linux and Windows, but regular
PyTorch memory has different ownership on the two platforms. Pressure and
reclaim must preserve that ownership boundary.

## Allocator modes

| Platform | Default mode | Regular allocation owner | Pressure point |
| --- | --- | --- | --- |
| Linux | `global` | AIMDO XPU pluggable allocator | Exact cache-miss allocation size before `sycl::malloc_device()` |
| Windows | `native_hook` | PyTorch native XPU caching allocator | Unified Runtime physical USM growth at `urUSMDeviceAlloc` |

Linux may opt into `native_hook` when AIMDO is interposed before Python starts.
Windows does not replace PyTorch's allocator in either mode.

## Shared invariants

1. VBAR pages are reclaimed only when unpinned and safe to retire.
2. The active model has higher priority than older VBARs.
3. Actual allocation pressure may reduce active-model residency; speculative
   model-boundary reclaim must not.
4. Pressure uses pending physical growth, not logical tensor size or virtual
   reservation size.
5. A component probe cannot establish workload liveness or performance unless
   the changed pressure path executes in the same run.

## Linux allocation-time pressure

In `global` mode, an AIMDO allocator cache miss supplies the exact request to
the budget policy before allocating:

```text
PyTorch cache miss
  -> budget_deficit(requested bytes)
  -> reclaim eligible VBAR pages only when required
  -> sycl::malloc_device()
```

A cache hit creates no physical growth and performs no speculative reclaim.
Freed blocks remain cached per device and queue, and their completion state
protects reuse.

## Windows allocation-time pressure

Windows keeps PyTorch's splitting, coalescing, stream ordering, retry,
statistics, and `empty_cache()` behavior. AIMDO observes physical USM segments
at the Unified Runtime boundary.

The allocation hook follows two rules:

- it never waits on a SYCL queue or performs re-entrant driver memory
  management;
- it injects a synthetic allocation failure only for a PyTorch request when
  PyTorch has cached bytes that its normal retry can release.

After PyTorch releases its cache and retries, AIMDO may reclaim the residual
deficit from retirement-proven VBAR pages. Direct non-PyTorch SYCL callers are
never given the PyTorch-specific synthetic failure.

The hook handles the direction where PyTorch requests memory. A VBAR fault that
needs memory while PyTorch holds freed blocks does not call `urUSMDeviceAlloc`.
The rate-limited fault-boundary cache trim covers that reverse direction and
must run outside the hook.

See [Windows Unified Runtime allocator hook](WINDOWS_XPU_UR_ALLOCATOR_HOOK.md)
for lifecycle, configuration, and counters.

## Windows pressure accounting

DXGI/WDDM local-memory `CurrentUsage` and `Budget` are the sampled process-wide
baseline. This includes SYCL, oneDNN, driver, and other allocations not owned
by AIMDO. Between rate-limited DXGI samples, AIMDO applies the delta from
native allocations it observes.

When DXGI/LUID mapping is unavailable, the backend falls back to Level Zero
free-memory information. WDDM non-local usage is considered fallback pressure
only after the platform safety margin, so ordinary bookkeeping does not cause
continuous VBAR eviction.

## Retirement and model boundaries

Windows reclaim cannot create a queue fence at the moment memory is needed and
then wait for it. Pages are tagged when they become unpinned, and previously
submitted completion barriers advance retirement epochs. The allocation path
only compares completed epochs and returns immediately.

At `prioritize()`, Windows may use PyTorch's observed peak-minus-current
reserved memory as a hint for expected native growth. This is speculative:
older VBARs may be reclaimed, but the newly active VBAR is preserved. Its full
fault range is reopened so a transient pressure window does not become a
permanent streaming ceiling.

Individual faults revalidate current pressure and can lower residency when the
requested pages do not fit. If pressure later clears, a request above the old
watermark may reopen the range and fault pages normally.

## Reserve policy

ComfyUI passes `--reserve-vram` to AIMDO as `simple_vram_headroom`. On Windows,
AIMDO keeps at least:

```text
max(simple_vram_headroom, 512 MiB)
```

between complete-process DXGI `CurrentUsage` and `Budget`. The usage basis and
reserve target must come from the same accounting domain; AIMDO-only usage is
not sufficient on Windows.

On Linux, the reserve is evaluated at the exact allocator request against the
allocator/VBAR accounting available to the backend.

## Physical-page allocation retry

A VBAR physical-page request first reclaims the full value returned by
`budget_deficit()`, which may be larger than one page. If Level Zero still
reports physical allocation OOM, Windows performs one final 512 MiB reclaim
and retries once. This is an error-recovery margin, not the primary pressure
control loop.

## Small VBAR copy fallback

Some Windows multi-adapter configurations reject small host-to-VBAR copies on a
non-primary Level Zero node. `comfy_aimdo.torch.copy_to_vbar()` is the
supported way to write host data into AIMDO VBAR storage; integrations must use
it instead of calling `Tensor.copy_` directly.

The helper takes the fallback route only when the request is Windows XPU,
CPU source to XPU destination, contiguous with identical shape and dtype,
1 byte through 2 MiB, inside a fully mapped and pinned VBAR range, and on an
adapter whose Level Zero node mask is not `0x1`. It then stages through private
device memory padded past the affected size and writes exactly the requested
range with a kernel, so padding never touches VBAR. Every other request uses
the normal copy path.

`AIMDO_XPU_SMALL_VBAR_COPY_FALLBACK` forces the route on with `1` or off with
`0`. `control.get_xpu_vmm_stats()` reports `small_vbar_copy_fallback_calls`,
`small_vbar_copy_fallback_bytes`, and `small_vbar_copy_fallback_failures`.

## Minimal regression check

Run from an environment with the XPU backend built and Torch XPU available:

```powershell
python tests\repro_xpu_platform_memory_policy.py
```

The script creates a lower-priority and an active VBAR, applies controlled
pressure, and reports residency around the platform policy. For Windows hook
behavior under native Torch growth, also run:

```powershell
<portable>\python_embeded\python.exe -s `
    tests\repro_windows_xpu_vbar_vs_torch.py --vbar-gib 6 --torch-gib 30
```
