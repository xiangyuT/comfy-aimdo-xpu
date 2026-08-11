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

## Windows: deferred and speculative pressure

On Windows, PyTorch retains its native XPU caching allocator. AIMDO observes
physical Level Zero allocations and releases with tracing callbacks. It cannot
unmap VBAR pages from inside `zeMemAllocDevice()`: VBAR unmap synchronizes the
SYCL queue, and waiting re-entrantly on that same queue can deadlock.

The callback therefore records pressure without performing VBAR eviction. At
the next model-prioritization boundary, AIMDO estimates possible native growth
as `peak_reserved - reserved` and performs a queue-safe speculative reclaim.
Because that value is allocator history rather than a live allocation request,
the reclaim follows an important boundary:

* Lower-priority VBARs may be reclaimed.
* On repeated activation with deferred native growth pending, the newly
  prioritized active VBAR is excluded from speculative reclaim and retains the
  working-set watermark established by its earlier sampler faults.
* Later VBAR faults use current pressure and may reclaim active, unpinned pages
  when memory is actually required.

```text
Level Zero allocation callback
  -> record native growth only

next active_model.prioritize()
  -> retain its previous exact-pressure watermark
  -> estimate peak_reserved - reserved
  -> reclaim lower-priority VBARs
  -> preserve active VBAR at that watermark

later active-model fault
  -> enforce exact current pressure
```

Resetting the active watermark to the full virtual range on every Windows model
switch is also unsafe. Pages above a watermark were removed by real pressure;
reopening that range causes repeated workflows to fault those pages again while
evicting other pages from the same fixed-size working set. This can repeatedly
reload a model boundary and eventually exhaust physical-page allocation.
Windows reprioritization therefore retains the last exact-pressure watermark
only after that VBAR has already been activated once and it must also handle
deferred native growth. First activation has no previous sampler working set;
it keeps the original reset-to-full behavior. Calls with no deferred growth do
the same, which explicit free/refault callers require. Linux always keeps the
original reset behavior because its allocator can arbitrate growth at the
actual request.

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

## `--reserve-vram`

ComfyUI uses `--reserve-vram` to limit model loading and passes the same byte
value to AIMDO as `simple_vram_headroom`. This is not a literal double
subtraction because the two components have different accounting, but both
controllers can request recovery near the same threshold. On Windows, using a
historical peak as if it were guaranteed future growth used to amplify that
interaction and progressively remove active-model residency across repeated
workflows.

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
| Windows | Historical peak at model boundary | Must retain its established working set during speculative reclaim |

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
