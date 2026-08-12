# Windows XPU DynamicVRAM mechanism analysis

Status: the formal workload passed for the first time on 2026-08-12 (see below);
remaining gaps are recorded there. Earlier sections keep the failed experiments.
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

Current standing as of 2026-08-12, against build `0.4.14.dev17`:

| Step | Configuration | Result |
| --- | --- | --- |
| Static | `tests/test_windows_pressure_source.py` | pass, 8/8 |
| Component | `repro_windows_xpu_vbar_vs_torch` | pass, 192/192 recovered |
| 1 | 864x480, 124 frames, 20 steps | pass, 168.9 s |
| 2 | 1280x736, 124 frames, 20 steps | pass, 705.7 s |
| 3 | 1280x736, 243 frames, 20 steps | **fail**, stalls at sampler step 0 |
| 4 | 1280x736, 362 frames, 20 steps, x3 | not reached |

Steps 1 and 2 do not reach memory pressure and must not be reported as
evidence that a memory-policy change works.

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

## Windows allocation-time interception, 2026-08-12

Status: implemented, component-verified, and carried through one passing formal
workload on the recorded Portable environment. Installed as
`comfy_aimdo 0.4.14.dev3`.

This supersedes the "Removed allocation-time allocator experiment" above. That
experiment replaced Torch's allocator; this one does not.

### What was built

`src-xpu/ze-detour.c` detours `zeMemAllocDevice`, `zeMemFree` and
`zeMemFreeExt` in `ze_loader.dll`, mirroring the production CUDA design in
`src-win/cuda-detour.c`. Torch keeps its native caching allocator. The Level
Zero tracing layer is retained for diagnosis behind
`AIMDO_XPU_ENABLE_ALLOCATION_TRACING=1`; `AIMDO_XPU_DISABLE_ALLOCATION_HOOKS=1`
disables interception entirely.

Reclaim reached from a hook never waits. `src/model-vbar.c` gains
`vbars_free_retired()`, and `src-xpu/dispatch.cpp` gains a retirement-epoch
ring: a page is tagged with the current epoch when unpinned (one atomic read,
no submission), and a bounded set of fences publishes how far retirement has
progressed.

### Platform facts this rests on

Measured with `tests/repro_windows_xpu_malloc_pressure.py` on an Arc Pro B70
(31,904 MiB, driver 32.0.101.8515, Torch 2.12.0+xpu):

1. A Torch USM allocation does not fail past the budget. 40,960 MiB was
   retained with zero failures; WDDM ended at 31,216 MiB local and 10,076 MiB
   non-local. Write latency rose from 0.3 ms to 39 ms with a knee exactly at
   local capacity. Any design waiting for an allocation error is waiting for a
   signal this platform does not send, and equally, AIMDO never has to block to
   keep an allocation correct.
2. Spilling is graceful; blocking is not. A 24 GiB working set ran at
   269-549 GiB/s, a 36 GiB one at 51.6-58.9 GiB/s, stable across iterations
   with no hang or device loss. A controlled spill is a better failure mode
   than an unbounded wait.
3. A freed-but-cached Torch block keeps holding WDDM local memory. After
   freeing every tensor, reserved stayed 12,290 MiB and WDDM local 12,530 MiB;
   `empty_cache()` returned it. AIMDO cannot reclaim that memory, which is a
   real cost of retaining the native allocator.
4. Torch XPU queues are in order (`c10/xpu/XPUEvent.h`), with 32 round-robin
   queues per pool per device (`c10/xpu/XPUStream.h`).

### Two design corrections forced by measurement

A single outstanding fence was not enough. A page tagged while the queue was
busy must wait for a later fence that is itself stuck behind unrelated work, so
measured reclaim during an allocation ramp was zero. A small ring fixed it.

A top-down scan that stops at the first unreclaimable page also reclaimed
nothing. Pages are tagged as they are unpinned, so the most recently used sit
at the top and retire last, which is precisely where such a scan stalls. The
scan is now position independent and does not move the watermark.

### Verified result

`tests/repro_windows_xpu_vbar_vs_torch.py`, 6 GiB VBAR against a 30 GiB Torch
ramp, run from the installed wheel:

| Phase | VBAR faulted | VBAR resident | Watermark |
| --- | ---: | ---: | ---: |
| idle device | 192/192 | 6,144 MiB | 192 |
| ramp at 20,480 MiB | - | 6,144 MiB | 192 |
| ramp at 25,600 MiB | - | 5,504 MiB | 192 |
| ramp at 30,720 MiB | - | 384 MiB | 192 |
| after Torch `empty_cache()` | 11/192 | 352 MiB | 11 |
| after `prioritize()` | 192/192 | 6,144 MiB | 192 |

Reclaim now tracks pressure while it is being created, with zero Torch
allocation failures and no queue wait. On the previous baseline the same ramp
reclaimed nothing until the next fault pass.

`tests/test_windows_pressure_source.py` passes 8/8; two assertions that encoded
the old deferred behaviour were rewritten to assert the new invariants, and one
new test pins that the non-blocking reclaim neither synchronizes nor moves the
watermark.

### Watermark revalidation and native cache reclaim

Two defects in the same causal chain were fixed with the interception work,
because leaving them would have made a Gate C run fail for an already
understood reason.

The watermark recorded that pressure was seen earlier in the activation, not
that memory was still short, and only `prioritize()` could raise it again. On
Windows the pressure is frequently transient, so one spike cost the model its
working set for the rest of the run - the mechanism behind a tiled VAE decoder
streaming from host storage on every tile. A fault above the watermark now
revalidates against live pressure and reopens the range when the pages it would
have to allocate actually fit. Reopening still allocates through the normal
per-page checks, so it cannot reintroduce an overcommit by itself.

Torch's freed-but-cached blocks never reach `zeMemFree`, so the allocation hook
cannot see them return and AIMDO cannot reclaim them. `empty_cache()` can, but
it must not be called from the hook, where Torch holds its allocator lock
across the driver call. A VBAR fault that is about to give up is a queue-safe
boundary and a measured shortage, so that is where AIMDO now releases the dead
cache and retries once. It is rate limited to one attempt every two seconds and
only runs when Torch is actually holding cached blocks, because `empty_cache()`
calls `sycl::free()`, which can stall under pressure.
`AIMDO_XPU_NATIVE_CACHE_TRIM=0` disables it.

Measured with the same 6 GiB VBAR against a 30 GiB Torch ramp:

| Phase | Before | After |
| --- | ---: | ---: |
| Torch holds 30 GiB | 13/192 | 13/192 |
| Torch tensors freed, cache retained | 13/192 | 192/192 |
| after `empty_cache()` | 11/192 | 192/192 |
| after `prioritize()` | 192/192 | 192/192 |

Residency now recovers inside a running activation, without waiting for a model
boundary. The middle row is the direct A/B for the cache reclaim: with
`AIMDO_XPU_NATIVE_CACHE_TRIM=0` the same run stays at 13/192.

### Formal workload result, 2026-08-12

The formal Gate D workload passed for the first time on this configuration:
MiniMax H3 T2V, 1280x736, 362 frames at 24 fps, two sampler steps, three
sequential prompts with distinct seeds in one unmodified Portable process
running `comfy_aimdo 0.4.14.dev3` with `--enable-dynamic-vram --reserve-vram 4`.

| Run | Elapsed | Cached nodes | Result |
| ---: | ---: | --- | --- |
| 1 | 578.18 s | none | `execution_success` |
| 2 | 612.99 s | static loaders only | `execution_success` |
| 3 | 572.28 s | static loaders only | `execution_success` |

Every output was 1280x736, 24 fps, 362 frames, 15.083 s. The client printed
`ACCEPTANCE_PASS`.

WDDM over the complete three-run window, sampled every 20 s (69 samples):
peak local 28,099 MiB against a target of `Budget - 4 GiB` = 27,691 MiB, so the
largest excess was 408 MiB and local never approached the 31,787 MiB budget.
Non-local started at 168 MiB, peaked at 1,607 MiB and ended at 247 MiB. No
`result=999`, `Fault failed`, `DEVICE_LOST`, `OUT_OF_RESOURCES`, traceback or
`CUDA API FAILED` appeared in the server log, and the Windows System log
contains no display-driver reset or device-removal event for the window.

The decisive comparison is with the recorded history, where the same repeated
workload regressed from about 7.06 s/step to 15.27 s/step, or degraded further
on run 3, and eventually produced OOM, `Fault failed: 2` or `DEVICE_LOST`. Run 3
is now the *fastest* of the three, and run 2 is 7.1% slower than run 3, inside
the 10% working threshold. There is no monotonic degradation across prompts.

### What this result does not establish

1. The client records only end-to-end prompt time. Model initialization,
   sampler seconds per step and VAE encode/decode are not separated, and the
   acceptance criteria require that so a sampler regression cannot hide behind
   a shorter load. Run 1 includes model loading and still landed between runs 2
   and 3, which is consistent with steady-state compute being somewhat slower
   than run 1's compute; that cannot be confirmed or refuted from this data.
2. Non-local memory peaked 1,439 MiB above its baseline, above the 512 MiB
   gate, before returning to 247 MiB. It is transient rather than sustained or
   monotonic, so it does not meet the failure condition as written, but it was
   not diagnosed.
3. Five of 69 samples exceeded the operating target. The criterion is written
   per consecutive sample against the server-side WDDM trace, not against a
   20 s external sampler, so this needs the in-process trace to adjudicate
   properly.
4. One passing series is not a stability claim. The known gaps below,
   especially the single-queue fence assumption, are unchanged by this run.



### Root cause of DEVICE_LOST: re-entrant reclaim, 2026-08-12

A production 20-step run on `0.4.14.dev3` reproduced the historical failure
shape exactly: prompt 1 in 10:23, prompt 2 in 18:43 (54.91 s/it), prompt 3
aborting at model initialization with `UR_RESULT_ERROR_DEVICE_LOST`.

The cause is a defect introduced by the interception work, not by the baseline.
`aimdo_xpu_prepare_allocation()` was called *before* `true_zeMemAllocDevice`,
and it reclaims by issuing `zeVirtualMemUnmap` and `zePhysicalMemDestroy`. That
mutates Level Zero physical memory while the driver is inside its own
allocation call. The earlier concern about this call site was that waiting on
the compute queue could deadlock; the stronger hazard is that re-entering the
driver's memory management at all corrupts its state, which surfaces later as
progressive slowdown and then device loss.

Nothing required reclaim to happen first. A Windows device allocation does not
fail - WDDM demotes the excess - so reclaiming immediately *after* the driver
call returns bounds steady-state usage just as well and is not re-entrant. The
hook now samples pressure before the call, which is read-only, and reclaims
after it returns. The diagnostic Level Zero tracer was changed the same way.

Isolation that identified it, each case repeated:

| Case | Before | After |
| --- | --- | --- |
| Cross-queue consume, no pressure | ok | ok |
| Pressure only, no VBAR | ok | ok |
| Single stream + pressure + reclaim | ok | ok |
| Cross-queue + pressure + reclaim | **device_lost** | **ok** (4/4) |
| Torch-only control, AIMDO uninvolved | ok | ok |
| Full synchronize, then explicit reclaim | ok | ok |

The cross-queue case was the trigger only because it widens the window in which
the driver is servicing an allocation while AIMDO unmaps; the defect is the
re-entrancy, not the second queue. That is also why per-queue retirement fences
did not fix it: the retirement proof was never the problem. Reclaim during an
allocation ramp still works and now engages earlier, with VBAR residency
falling at 15 GiB of Torch pressure rather than 25 GiB, while the watermark
stays at its full extent.

Installed as `0.4.14.dev4`.

### Gate C did not exercise this

The formal pass recorded above ran with `reopening watermark`, `free_retired`,
`CACHE TRIM` and `Deficit > 0` all at zero occurrences across 1,016 WDDM
samples. The two-step workload never entered pressure, so it never executed the
paths this work added. Its value was as a no-regression check, not as
validation of the new reclaim. A 20-step workload does create pressure and
found the defect immediately.

Acceptance for a memory-policy change must therefore require evidence that the
workload reached pressure, in the form of non-zero reclaim counters in the same
log. A pass with all-zero counters proves only that nothing regressed.

### Reserve enforcement gap behind `result=999` copy failures, 2026-08-12

> Correction, same day: this section originally called the reserve gap the root
> cause of the copy failures. It is a real defect and the fix is retained, but
> it was not sufficient. A second defect - the pin ordering in `vbar_fault` -
> was found afterwards, and even with both fixed the 720p/10s workload still
> stalls. See "Pin-on-map removed the copy failure but not the stall" below.

A 20-step run fails deterministically with a host-to-device copy returning
Level Zero error 39, `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, reported as
`hostbuf_file_reader_read: device copy failed result=999`. Two builds failed
identically: same 28,573,696-byte chunk, same 27.82 GiB local usage, at 205 s
and 207 s. That determinism ruled out the race conditions previously assumed.

A control experiment settled it. Only `--reserve-vram` was changed:

| reserve | target (Budget − reserve) | peak local | overshoot | result |
| ---: | ---: | ---: | ---: | --- |
| 4 GiB | 27.04 GiB | 27.82 GiB | 0.77 GiB | error 39 at 207 s |
| 8 GiB | 23.04 GiB | 23.82 GiB | 0.78 GiB | no error, steady 186.9 s/it |

The overshoot is constant and independent of the reserve, so the reserve was
not merely set too low: a fixed quantity was unaccounted for. Comparing AIMDO's
own accounting with DXGI confirms it:

| | AIMDO `total_vram_usage` | DXGI `CurrentUsage` | unaccounted |
| --- | ---: | ---: | ---: |
| reserve 4 | 27.13 GiB | 27.82 GiB | 0.69 GiB |
| reserve 8 | 23.16 GiB | 23.82 GiB | 0.66 GiB |

The two pressure signals in `budget_deficit()` each had one half of the
problem:

* `deficit_sync` (`src-win/shmem-detect.c`) used the right basis - DXGI
  `CurrentUsage` covers SYCL, oneDNN and driver allocations that never reach
  AIMDO - but a hardcoded 512 MiB margin instead of the configured reserve. At
  the failure point it computed `27.82 + 0.5 − 31.04 = −2.72 GiB`: no deficit.
* `deficit_simple` used the configured reserve but against
  `total_vram_usage`, which undercounts by exactly the amount above.

Neither combined complete-process usage with the configured reserve, so
enforcement halted at target plus the undercount. Substituting the reserve into
the DXGI expression reproduces the measurement exactly:
`27.82 + 4 − 31.04 = +0.78 GiB`.

Why an over-target process fails rather than merely slowing down: Torch's USM
allocations are demotable, and an earlier probe retained 40 GiB on a 31.9 GiB
device with zero failures because WDDM moved the excess to non-local memory.
VBAR pages are `zePhysicalMemCreate` objects and have no such fallback. At the
failure the split was 17.9 GiB Torch against 11.3 GiB VBAR on a 31.16 GiB
device, with non-local flat at 0.16 GiB: WDDM was not demoting anything, and
the copy had nowhere to land.

Two fixes follow directly:

1. `deficit_sync` now enforces `max(configured reserve, 512 MiB)` against DXGI
   `CurrentUsage`, which is the policy this document already described.
2. `hostbuf_file_reader_read()` and `hostbuf.c`'s streaming copy sample
   pressure and run the non-blocking reclaim before each chunk. Streaming a
   missed weight is the largest consumer of device memory that never allocates
   through a hooked entry point, so nothing applied pressure there at all -
   both files contained zero pressure checks, against three in `model-vbar.c`
   and two each in `vrambuf.c` and the CUDA allocator shim.

### Method failure worth recording

Three builds were shipped against this failure with the wrong cause: a
re-entrancy fix, a leak fix, and a retirement-ordering fix. All three passed
the component reproducers and all three failed the real workload, because the
reproducers never exercise the file-streaming path. The defect was reached only
by enumerating every site that consumes device memory and checking which ones
apply pressure - a five-minute audit that should have preceded the first fix.

Component reproducers are necessary but must not gate delivery on their own. A
memory-policy change is only validated by a workload that demonstrably reaches
pressure.

### Known gaps

1. Only one passing three-run series, without per-phase timing. See the caveats above.
2. The retirement fence covers every queue AIMDO has seen, registered when a
   page consumed on that queue is unpinned. A queue that AIMDO never observes
   is still uncovered, and the tracked-queue table is bounded at 64 entries;
   overflow stops retirement from advancing rather than releasing unprovable
   pages. Only ComfyUI's normal single-stream path was measured end to end.
3. Retirement can only advance as fast as the queue drains. Under sustained
   submission, pages that are idle in reality stay unprovable and reclaim
   yields less than the deficit. The allocation is allowed to spill, which is
   bounded but not free.
4. Detours attaches at `plat_init()`, so any allocation made before AIMDO
   initializes is neither arbitrated nor accounted.
5. `zeMemAllocHost` and shared-memory allocations are not intercepted; only
   device allocations are.
6. Detours is a process-wide inline hook on `ze_loader.dll` exports. Coexistence
   with other components that hook Level Zero was not tested.
7. Watermark revalidation calls `budget_deficit()` on a fault that would
   otherwise have returned immediately. Under genuine sustained pressure this
   adds a rate-limited WDDM sample per missing weight rather than a fast reject.


### Pin-on-map removed the copy failure but not the stall, 2026-08-12

The pinned-but-unmapped page described above is a real defect and the fix for
it is retained. It is **not** the root cause of the workload failure. Claiming
it was is the second time in this investigation that a partial fix was
presented as a root cause, and the mechanism of that mistake is recorded here
because it is more useful than the fix.

What the fix changed. `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on host-to-device
copies stopped occurring. The range diagnostic no longer reports `ALL_MAPPED=0`
with a nonzero pin count. Those are genuine improvements and they are why the
copy path now survives past the point where every earlier build died.

What it did not change. The 720p/10s workload still does not complete. It
reaches sampler step 0 and stops making useful progress there, in the same
weight-streaming path as before:

```
Thread (active): "Thread-1 (prompt_worker)"
    read_file_to_device (host_buffer.py:81)
    read_tensor_file_slice_into (memory_management.py:59)
    cast_to_gathered (model_management.py:1500)
    ...
```

Twelve consecutive py-spy samples put the top Python frame on
`host_buffer.py:81` every time, that is, inside native
`hostbuf_file_reader_read`.

This is a livelock, not a deadlock. Over a 45-second window the counters moved:

| Counter | Delta over 45 s |
| --- | --- |
| `host_to_device_bytes` | +368 MiB (≈8.5 MiB/s) |
| `map_bytes` | +352 MiB |
| `unmap_bytes` | +352 MiB |

Map and unmap advance at the same rate. The system is mapping pages and
immediately releasing them again, so forward progress costs a full
map/unmap cycle per page and effective streaming bandwidth collapses to
roughly 8.5 MiB/s against a device that sustains hundreds of GiB/s. A run left
in this state for 45 minutes never produced a single sampler iteration.

Why the earlier gradient steps passed. 480p/5s completed in 169 s and
720p/5s in 706 s. Neither reaches the pressure point. The failure needs both
the 19.5 GiB model resident and the larger activation working set of a 243
frame prompt. Reporting those two passes as evidence that the root cause was
fixed repeated exactly the error this document already warns about: a small
pass does not erase a large failure.

A relationship to the reserve fix is suspected and unproven. With
`--reserve-vram 4` the WDDM trace shows `local_budget=31.04 GiB` and
`local_usage` held at 27.05 GiB, which is the budget minus the configured
reserve, so the reserve is now being enforced as intended. The 19.5 GiB model
plus the 720p/10s activation set does not fit in what remains, and the VBAR
falls back to streaming for the shortfall. That is the designed degradation
path, but at 8.5 MiB/s it is indistinguishable from a hang. Whether the
correct answer is a lower reserve, a streaming path that does not thrash, or a
watermark that stops re-admitting pages it cannot keep, is not yet determined.
The A/B run against `--reserve-vram 0.5` that would separate these was
prepared and then stopped; it has not been performed.

Open questions, in the order they should be answered:

1. Is the map/unmap cycling driven by the watermark re-admitting pages that
   pressure then immediately reclaims? If so the watermark revalidation added
   here is oscillating rather than converging, and it needs hysteresis.
2. Does `AIMDO_XPU_NO_STREAM_RECLAIM=1` change the 8.5 MiB/s figure? That
   isolates whether the streaming-path reclaim is the thing releasing what the
   stream just mapped.
3. Does a reserve small enough to fit the working set eliminate the stall
   entirely? If yes, the defect is admission policy, not reclaim.
4. Is there a size at which the streaming fallback is expected to be viable at
   all? A model that must stream a large fraction of its weights every step may
   simply have no acceptable configuration on this device, and that would be a
   documented limit rather than a bug.

### Method failure worth recording, second occurrence

The first occurrence in this document was shipping on component-reproducer
evidence. This is the same failure with a different substitute: the component
reproducer reported 192/192 recovered pages and the two smallest gradient steps
passed, and that was treated as confirmation. The reproducer has now passed on
several builds that failed the real workload; it establishes a floor and
nothing more.

The gradient ladder exists to prevent this and was not run to completion before
the conclusion was drawn. The rule is that no root-cause claim is made until
the ladder passes at the step that actually reaches memory pressure, which for
this model and device is 720p/10s and above.



The Portable interpreter is an embeddable build. Its `._pth` file disables
`PYTHONPATH`, so `set PYTHONPATH=<repo>` silently loads the *installed* wheel
instead of a local build. Several runs were interpreted before this was found:
they showed no interception at all while the same logic in a script that called
`sys.path.insert()` reclaimed correctly. Any script used to validate a local
build must insert the repository path itself, and a run should confirm which
DLL it loaded before its result is trusted.

Two toolchain problems in the recorded environment are worked around by
`scripts/`, not by changing the environment: oneAPI's `setvars.bat` fails to
invoke any component `vars.bat` and leaves `icx-cl` off PATH, and Detours must
be built against the project-local NuGet Windows SDK because the Visual Studio
Build Tools installation has no SDK.

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
