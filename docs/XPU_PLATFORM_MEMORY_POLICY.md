# XPU platform memory policy

Intel XPU uses the same VBAR priority and fault machinery on Linux and Windows,
but the regular PyTorch allocation path differs. This distinction matters when
ComfyUI passes `--reserve-vram` to both its model loader and AIMDO's
`simple_vram_headroom`.

## Linux: exact allocator-time pressure

On Linux, AIMDO installs its XPU pluggable allocator before the first XPU
allocation. A cache miss supplies the exact pending allocation size to
`budget_deficit()` before `sycl::malloc_device()` runs. AIMDO may reclaim any
unpinned VBAR pages, including pages from the active model if real pressure
requires it. A cache hit does not trigger physical growth or speculative VBAR
reclaim.

This is an allocation-time decision:

```text
PyTorch cache miss
  -> exact requested bytes
  -> budget_deficit(requested bytes)
  -> reclaim only if required
  -> sycl::malloc_device()
```

## Windows: allocation-time interception

On Windows, PyTorch keeps its native XPU caching allocator, with all of its
block splitting, coalescing, stream ordering and `empty_cache()` behaviour.
AIMDO instead intercepts the driver entry point that actually grows physical
device memory, `zeMemAllocDevice`, using Detours. This mirrors the production
CUDA design, which hooks `cuMemAlloc_v2` rather than replacing the allocator.

Two alternatives were measured and rejected:

* Replacing Torch's allocator with an AIMDO pluggable allocator degraded block
  management to an exact-size cache, and its release path had to call
  `sycl::free()`, which stalls inside the Level Zero/UMF residency path under
  WDDM pressure. That stall is why `AIMDO_XPU_EAGER_USM_RELEASE` defaults off.
* The Level Zero tracing layer observes the same calls but is process-wide and
  progressively slows long command streams. It is retained for diagnosis behind
  `AIMDO_XPU_ENABLE_ALLOCATION_TRACING=1`, never for performance work.

### The allocation path must never wait

A hook body runs inside the driver's allocation call. The compute queue it
would wait on may already be blocked behind work that needs the very residency
being requested, so any wait there can stop forward progress entirely. An
earlier revision reclaimed with `cuCtxSynchronize()`, which on this backend is
`sycl::queue::wait_and_throw()`, and could not complete one formal prompt.

Nothing forces AIMDO to wait, because a Windows Torch allocation does not fail:
40 GiB was retained on a 31.9 GiB device with zero allocation failures, WDDM
demoting the excess to non-local memory. There is therefore no correctness
requirement to free anything before returning. Reclaim is best effort.

The two allocation kinds are not symmetric, which is why AIMDO always yields
first without this arbitration. A Torch USM allocation can be demoted; a VBAR
page is an explicit Level Zero physical object that is mapped and made
resident, and has no equivalent fallback.

### Retirement epochs

Torch's XPU queues are in order, so a barrier submitted at reclaim time waits
for everything already queued. A fence created when memory is needed is
therefore useless: it has already lost the property it was meant to buy.

A page is instead tagged with the current epoch when it is unpinned, which
costs one atomic read and submits nothing. A small ring of fences, at most one
new barrier per completion interval, publishes how far retirement has
progressed. Reclaim compares two integers and never blocks.

The ring matters. With a single outstanding fence, a page tagged while the
queue was busy has to wait for a later fence stuck behind unrelated work, and
measured reclaim during an allocation ramp was zero.

```text
weight unpinned
  -> tag page with current epoch (no barrier)
  -> poll the fence ring, submitting at most one barrier

native allocation (zeMemAllocDevice hook)
  -> sample WDDM pressure
  -> release VBAR pages whose epoch has retired
  -> proceed regardless of how much was reclaimed
```

### Reclaim does not move the watermark

The non-blocking reclaim scans every page rather than stopping at the first one
it cannot release: pages are tagged as they are unpinned, so the most recently
used sit at the top of the range and retire last, which is exactly where a
top-down scan would stall.

It also leaves the watermark alone. Lowering it denies future faults until the
next model activation, turning a momentary allocation spike into a working set
that never recovers. Releasing a page without lowering the watermark lets a
later fault bring it back when memory is genuinely available, and that fault
still re-checks pressure, so it cannot reintroduce an overcommit by itself.

Measured on an Arc Pro B70 with a 6 GiB VBAR and a 30 GiB Torch ramp:

| Torch retained | VBAR resident | Watermark |
| ---: | ---: | ---: |
| 20,480 MiB | 6,144 MiB | 192 |
| 25,600 MiB | 5,504 MiB | 192 |
| 30,720 MiB | 384 MiB | 192 |

Reclaim tracks the pressure while it is being created, and the model's declared
working set survives it.

## Windows: model-boundary reclaim

* Lower-priority VBARs may be reclaimed.
* The newly prioritized active VBAR is excluded from speculative reclaim and
  its full virtual address range is reopened for the new activation.
* Later VBAR faults use current pressure and may reclaim active, unpinned pages
  when memory is actually required.

```text
Level Zero allocation callback
  -> record native growth only

next active_model.prioritize()
  -> reopen its full virtual address range
  -> estimate peak_reserved - reserved
  -> reclaim lower-priority VBARs
  -> preserve active VBAR during speculative reclaim

later active-model fault
  -> enforce exact current pressure
```

The pressure-reduced watermark from a completed activation cannot be reused as
the next activation's hard ceiling. In particular, the MiniMax H3 Video VAE
revisits the same decoder layers for every spatial and temporal tile. If its
previous low watermark is restored, every layer above that boundary receives a
VBAR OOM fallback even when the current activation could reclaim room. The
roughly 5 GiB decoder is then streamed from host storage once per tile, causing
hundreds of GiB of repeated host-to-device copies and severe second-run
slowdown. Reopening the range allows normal faults to rebuild a working set;
those faults still apply current pressure and can lower the watermark again.
Linux keeps its existing reset behavior and exact allocation-time arbitration.

The fault path revalidates the watermark against live pressure. A request above
the watermark reopens the range when the pages it would have to allocate
actually fit right now, so a spike absorbed by a fault no longer costs the
activation its residency until the next model boundary. Reopening still runs
the normal per-page allocation checks and cannot reintroduce an overcommit by
itself. `tests/repro_windows_xpu_vbar_vs_torch.py` shows the effect: after
Torch releases 30 GiB the next pass faults all 192 pages again, with no
`prioritize()`.

A fault that would still fail first asks Torch to return its freed-but-cached
blocks. Those never reach `zeMemFree`, so the allocation hook cannot see them
and AIMDO has no other way to reclaim them. The call is made from the fault
boundary, never from the hook, because Torch holds its allocator lock across
the driver call; it is rate limited and skipped unless Torch is actually
holding cached blocks. `AIMDO_XPU_NATIVE_CACHE_TRIM=0` disables it.

Sampler latency changes must not be attributed to VBAR churn without checking
the VMM counters. A sampler interval with unchanged `map_bytes`, `unmap_bytes`,
and recorded VBAR usage did not fault or evict model pages, even if its per-step
time changed.

## Physical-page allocation retry

A VBAR physical page is 32 MiB. The fallback path must reclaim the complete
value returned by `budget_deficit(32 MiB)`, not just one page: the current WDDM
deficit can be much larger than the page being requested. If a real Level Zero
physical allocation still reports OOM after that exact reclaim, Windows XPU
performs one additional 512 MiB reclaim before its final retry. This second
margin matches the WDDM safety margin and is reached only after an actual
allocation failure; it is not used for Linux or for speculative model-boundary
pressure.

That retry is a last resort, not the control loop. A Torch USM allocation is
demoted by WDDM rather than failing, so the pressure-generating side normally
never emits the error this path waits for. Arbitration has to happen in the
allocation hook, before the driver places the request.

## `--reserve-vram`

ComfyUI uses `--reserve-vram` to limit model loading and passes the same byte
value to AIMDO as `simple_vram_headroom`. On Windows AIMDO enforces
`max(simple_vram_headroom, 512 MiB)` as the distance between DXGI's
complete-process `CurrentUsage` and `Budget`.

This document previously described that policy while the code applied only the
512 MiB floor to the DXGI signal, and applied the configured reserve solely to
AIMDO's own `total_vram_usage`. Because that figure excludes SYCL, oneDNN and
driver allocations, enforcement stopped a fixed distance above the target - a
measured 0.77-0.78 GiB, unchanged whether the reserve was 4 or 8 GiB - and a
later streaming copy failed with `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. The
basis and the target must come from the same accounting; see the
[liveness analysis](WINDOWS_XPU_VBAR_LIVENESS_ANALYSIS.md).

This is not a literal double subtraction with ComfyUI's own reserve because the
two components have different accounting, but both controllers can request
recovery near the same threshold. On Windows, using a historical peak as if it
were guaranteed future growth used to amplify that interaction and
progressively remove active-model residency across repeated workflows.

The Windows active-VBAR exclusion fixes that policy error without changing the
configured headroom or Linux's exact allocator-time behavior.

## Minimal reproducer

Run the script from an environment where the XPU backend has already been
built and `torch.xpu.is_available()` is true:

```bash
python tests/repro_xpu_platform_memory_policy.py
```

The script creates one lower-priority VBAR and one active VBAR, forces a small
controlled pressure window, and prints residency before and after the platform
pressure path.

Expected result:

| Platform | Pressure source | Expected active VBAR result |
| --- | --- | --- |
| Linux | Exact live pluggable-allocator request | May lose an unpinned page when real pressure exceeds lower-priority residency |
| Windows | Historical peak at model boundary | Must preserve active residency during speculative reclaim and reopen the full fault range |

The Windows assertion fails with the old global speculative-reclaim behavior,
where the active VBAR was used after lower-priority residency was exhausted.

For focused model-boundary diagnosis, set `AIMDO_XPU_BOUNDARY_TRACE=1` before
starting the process. It logs each VBAR's prior/current watermark and native
allocator peak gap without enabling the much higher-volume per-weight VBAR
trace.

To replay an exported ComfyUI history prompt three times in the same process,
changing its noise seed and output prefix on each run, use:

```powershell
python tests/run_saved_prompt_repeated.py `
    --history-json <history-export.json> `
    --server http://127.0.0.1:8188 `
    --runs 3
```
