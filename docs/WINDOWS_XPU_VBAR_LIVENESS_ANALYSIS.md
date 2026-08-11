# Windows XPU DynamicVRAM mechanism analysis

Status: unresolved. This document records confirmed evidence and failed local
experiments; it is not a claim that MiniMax H3 acceptance passes on Windows.
Last consolidated: 2026-08-11.

The exact build toolchain, Portable runtime, model set, launch policy, and
four-part acceptance gate are recorded in
`WINDOWS_XPU_BUILD_TEST_ACCEPTANCE.md`.

## Source state retained after the investigation

The worktree was returned to commit `323c0e0`. All later uncommitted allocator,
page-size, handle-pool, tracing, acceptance-harness, and residency experiments
were removed because they did not pass the formal three-run workload.

The retained Windows changes have narrower, independently observed effects:

* `4698184` avoids eager USM release on Windows. This removed a severe
  `sycl::free()`/UMF stall from the VBAR fault path.
* `32d26d0` adds the Windows pressure-observation baseline and keeps the native
  Torch XPU allocator. It does not guarantee bounded WDDM usage or repeated-run
  performance.
* `323c0e0` reopens the complete VBAR address range on model reactivation. It
  prevents a pressure-reduced watermark from becoming the next activation's
  hard address limit. It does not solve allocation-time arbitration.

These commits are useful fixes, but their presence must not be interpreted as
resolution of the complete Windows DynamicVRAM problem.

## Acceptance status

The required acceptance remains:

1. Development gate: MiniMax H3 T2V, 864x480, 124 frames, 24 fps, two sampler
   steps, repeated three times in one Portable ComfyUI process.
2. Formal gate: MiniMax H3 T2V, 1280x736, 362 frames, 24 fps, two sampler
   steps, repeated three times in the same process.
3. No systematic run-2/run-3 slowdown, OOM, device loss, copy failure, or
   forward-progress stall.
4. WDDM local usage remains bounded by the configured target, and non-local
   growth does not increase monotonically because of device-memory spill.

The retained source state has not passed this complete acceptance. A small
three-run result or one completed large prompt is only diagnostic evidence.

## Confirmed observations

### Eager physical release was one real fault, not the root cause

Before `4698184`, Windows could spend a long time in eager `sycl::free()` while
Level Zero/UMF was already under residency pressure. Deferring that release
restored millisecond-scale VBAR faults in the reported case. Afterward, WDDM
local usage could still grow beyond Budget with no AIMDO eviction, so removing
the free stall did not close the memory-control loop.

One reported 12 GiB trace grew from approximately 5.69 GiB local usage to
11.71 GiB, 12.15 GiB, and 14.97 GiB while Budget remained approximately
11.75-11.82 GiB. Successful Level Zero allocation and map calls therefore do
not prove that the process is staying within its WDDM budget.

### The production Windows ordering is late

The retained Windows implementation keeps Torch's native XPU caching
allocator. AIMDO does not receive every exact pending Torch allocation at a
safe pre-allocation point. WDDM can place excess allocations in non-local
memory instead of failing the allocation. Consequently, a policy that evicts
only after VBAR fault failure may never run before performance collapses.

The missing control loop is:

```text
exact pending physical allocation
  -> fresh complete-process WDDM usage and Budget
  -> requested-byte-aware deficit
  -> reclaim completed VBAR victims
  -> retry or fail before submitting dependent work
```

The retained Windows path observes pieces of this state, but it does not own
the complete sequence.

### The observed apparent hang is a queue/residency wait

Repeated py-spy samples from a failed large workflow placed the prompt worker
in ComfyUI's `cast_to_gathered()` / XPU `copy_()` path, with the native stack
in `urEventWait`. GPU execution showed short bursts separated by long idle
intervals. At that time local usage was approximately 25.9 GiB and non-local
usage approximately 4.5 GiB on the selected GPU; the second GPU was idle.

This evidence rules out a Python sampler loop as the immediate blocking site.
It does not identify which earlier in-order command the event was waiting for.
The failure class is consistent with residency starvation: the host waits for
queued work, while the next AIMDO boundary that could reclaim memory cannot be
reached.

### Repeated-run slowdown is real

Observed same-process series included:

* about 7.06 s/step for run 1, followed by about 15.27 s/step during run 2;
* about 9.81 s/step for run 1, 14.14 s/step for run 2, and continued degraded
  timing in run 3;
* later OOM, `Fault failed: 2`, and
  `UR_RESULT_ERROR_DEVICE_LOST` outcomes on repeated execution.

Completion alone is not acceptance when each repeat faults or copies more data
and becomes materially slower.

## Removed allocation-time allocator experiment

A later uncommitted experiment installed an AIMDO XPU pluggable allocator on
Windows and moved pressure arbitration in front of `sycl::malloc_device()`.
It was removed during cleanup because it did not pass the formal workload.

The experiment established several useful facts:

1. A fresh-process same-queue test performed 1,000 allocate/free operations of
   the same size with one physical allocation, 999 cache hits, and no per-free
   event synchronization. This only validated that narrow cache-reuse path.
2. The development workload completed three times in approximately 44.25,
   32.13, and 32.14 seconds, while sampled WDDM local usage remained below the
   configured target.
3. The formal run was still in model initialization after 112.6 seconds and
   was stopped; it did not produce one complete formal prompt, let alone three.
4. A py-spy sample caught the prompt worker in the experimental path:

   ```text
   xpu_alloc_fn
     -> allocation pressure reclaim
     -> cuCtxSynchronize
     -> sycl::queue::wait_and_throw
   ```

   Allocation-time ownership fixed the ordering gap but reclaimed VBAR pages
   by synchronizing the complete queue. That converts pressure-bearing cache
   misses into global execution barriers and is not a valid forward-progress
   design.

The experiment also used an exact-size cache rather than a production-quality
splitting/coalescing allocator. A small same-size microbenchmark could not
validate fragmentation, diverse temporary sizes, stream ordering, or
`empty_cache()` behavior in a real workflow.

## Removed persistent handle-ceiling experiment

Another uncommitted design attempted to learn a permanent physical-handle
ceiling after releasing VBAR backing. A deterministic four-page reproducer
showed the state error:

```text
initial:                         live=4, ceiling=unbounded
explicitly reclaim two pages:   live=2, ceiling=2
empty Torch cache + reactivate: live=2, ceiling=2
```

The pressure source in a real workflow can be a temporary SDPA workspace.
After that workspace is released, the VBAR working set must be allowed to grow
again if the live WDDM budget permits it. A monotonic ceiling preserves the
temporary reduction across prompts and creates a one-way performance decline.

The attempted activation-local correction was not built or tested and was
removed. No handle-ceiling implementation from this experiment is retained.

## Why 192 or 766 handles is not a proven limit

The earlier values were calculated from resident bytes divided by a chosen
VBAR page size:

* 24 GiB / 128 MiB = 192 pages;
* approximately the same bytes / 32 MiB = approximately 768 pages.

Those counts include only AIMDO VBAR physical objects. They exclude Torch,
oneDNN, Unified Runtime, driver allocations, queues, events, and other process
resources. No capability query or isolated first-failure sweep established a
hardware, OS, or driver limit of 192 handles. An asynchronous
`OUT_OF_RESOURCES` reported at a later synchronization is also not proof that
the failed resource was the VBAR physical-handle count.

Larger pages may reduce object count, but they also make eviction coarser and
increase edge waste. Page size is a diagnostic variable, not a memory-policy
solution.

## Mechanism that remains unresolved

Three requirements currently conflict:

1. Reclaim must happen before unmanaged Torch/oneDNN growth pushes WDDM into
   non-local memory.
2. A VBAR page cannot be unmapped while queued kernels or copies can still use
   it.
3. Waiting for the whole in-order queue inside a pressure-bearing allocation
   or copy path can block behind work that itself needs residency to progress.

The implementation lacks an exact retirement token for the last consumer of a
VBAR page or reclaim batch. A raw queue is insufficient: `queue.wait()` waits
for all submitted work, not the last command that used a selected page.

A future design must represent the page lifecycle explicitly:

```text
absent
  -> mapped and resident
  -> pinned/in use
  -> retired by an exact completion event
  -> event complete and evictable
  -> unmapped/released or recycled
  -> absent
```

Victim selection must first use already-complete retirement events. If enough
safe victims are unavailable, the allocator must return a controlled failure
before submitting new dependent work; it must not enter an unbounded
whole-queue wait.

## Required next validation

Do not resume end-to-end tuning before a native functional reproducer proves
the following:

1. Fill a VBAR working set beyond a controlled residency target.
2. Copy distinct host data into each faulted range and execute an XPU consumer
   kernel.
3. Record the exact last-use completion token, evict only completed ranges,
   refault them, restore their data, and verify the result.
4. Rotate the complete set multiple times while competing Torch allocations
   grow and shrink.
5. Record allocation request bytes, WDDM local usage/Budget, non-local
   baseline/growth, selected victim bytes, physical create/release counts, and
   every wait duration.
6. Fail on a data mismatch, unmatched wait begin/end, OOM, device loss,
   monotonic non-local growth, or missing forward progress.

Only after that component test passes should the development and formal
three-prompt gates run. A single hot run, a 480p-only pass, or a process that
eventually completes after severe slowdown must not be promoted to a fix.

## Evidence artifacts

Relevant diagnostic output remains under ignored `build/` directories,
including the handle-cap comparison, residency traces, process-wide Level Zero
tracing experiment, py-spy dumps, and the incomplete allocation-time formal
run. These artifacts belong to different revisions and must not be combined as
one benchmark series.

## References

* Microsoft `IDXGIAdapter3::QueryVideoMemoryInfo`:
  <https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo>
* Level Zero core API:
  <https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html>
* SYCL 2020 queue and event semantics:
  <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>
* Windows over-budget report:
  <https://github.com/xiangyuT/comfy-aimdo-xpu/issues/3>

## Maintenance rule

Future experiments must append revision-specific evidence to this record. Do
not erase a failed large/repeated workload with a later small pass, and do not
promote arithmetic estimates or asynchronous error codes into hardware limits
without an isolated reproducer.
