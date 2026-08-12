# Windows XPU second-GPU copy failure

Status: **unresolved.** The failure is isolated to a one-call reproducer and
its boundary is measured precisely, but it is not fixed. Five attempted fixes
were each built, run, and observed to fail; they are recorded here so nobody
repeats them.

## Symptom

With AIMDO DynamicVRAM enabled on the second GPU (`ONEAPI_DEVICE_SELECTOR=level_zero:1`),
the MiniMax H3 workflow fails about two seconds in, while loading the text
encoder:

```text
Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged.
RuntimeError: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
  comfy/ops.py:217 in cast_maybe_lowvram_patch
  comfy/model_management.py:1500 in cast_to_gathered
  comfy_kitchen/tensor/base.py:86 in copy_from -> Tensor.copy_
```

The same workflow completes on the same GPU with `--disable-dynamic-vram`
(199.4 s), so AIMDO is involved, but see the measurement below: the failing
operation is a plain `Tensor.copy_` into a VBAR mapping, and no AIMDO
allocation path is entered.

## Reproducer

`tests/repro_windows_xpu_vbar_copy.py` reduces a four-minute failure that needs
a 19.5 GiB model to a single 594 KB copy that fails on the first VBAR page:

```powershell
$env:ONEAPI_DEVICE_SELECTOR = "level_zero:1"
<portable>\python_embeded\python.exe -s tests\repro_windows_xpu_vbar_copy.py `
    --pages 5 --copy-bytes 607744
```

## What was measured

The transfer size decides the outcome. Nothing else does.

| Copy size | Result |
| ---: | --- |
| 4 KiB | fails |
| 64 KiB | fails |
| 512 KiB | fails |
| 607,744 B (the size ComfyUI uses) | fails |
| 1 MiB | fails |
| 1.5 MiB | fails |
| 2 MiB exactly | fails |
| **2 MiB + 1 byte** | **succeeds** |
| 2.5 / 3 / 4 / 32 MiB | succeeds |

Alignment is not the variable: 607,744 is neither 4 KiB nor 64 KiB aligned and
fails, while 2,097,153 is unaligned and succeeds. The boundary is 2 MiB, which
is the Level Zero large-page granularity.

At the moment of failure every operand is valid:

| Property | Observed |
| --- | --- |
| destination in a VBAR range | yes |
| destination fully mapped | yes |
| destination pinned | yes |
| destination type from `zeMemGetAllocProperties` | `ZE_MEMORY_TYPE_DEVICE`, query succeeded |
| source type | host USM, query succeeded |
| queue context vs AIMDO's Level Zero context | identical |
| device free memory | 31,407 MiB |
| transfer size | 594 KB |

`zeCommandListAppendMemoryCopy` called directly on an immediate command list
returns `0x70000003`, which is Level Zero's own `OUT_OF_DEVICE_MEMORY`. The
refusal therefore comes from the driver, not from Unified Runtime's wrapper,
and the error code does not describe a real shortage.

## Hypotheses refuted by measurement

Each of these was proposed and then disproved. They are listed so the evidence
against them is not lost.

| Hypothesis | Evidence against |
| --- | --- |
| VBAR page not mapped | `all_mapped=1 pin=1` at the failure |
| Driver staging buffer exhausted | a blocking retry fails identically |
| Device memory exhausted | 594 KB copy with 31,407 MiB free |
| Destination is not USM, so the driver cannot resolve it | `zeMemGetAllocProperties` succeeds and reports device memory |
| Queue context differs from AIMDO's mapping context | both native handles compared equal |
| Device index mis-plumbed to the second GPU | under the selector only one device exists and every index is 0; still fails |
| Blocking reclaim in `vrambuf.c` | fixed and re-run; still fails |
| VBAR pages never made resident | `zeContextMakeMemoryResident` added; still fails |

## Fixes attempted and rejected

None of these are in the tree. Each was built, installed, and run against the
reproducer.

1. **Blocking retry of the same copy.** Fails with the same code.
2. **Stage through a USM device buffer** (host -> USM -> VBAR). The second leg
   fails, which proves the destination is the problem and the source is not.
3. **Copy with `zeCommandListAppendMemoryCopy` on an immediate command list.**
   Fails with raw `0x70000003`, which is what established that the refusal is
   the driver's own.
4. **`zeContextMakeMemoryResident` after `zeVirtualMemMap`,** with a matching
   `zeContextEvictMemory` before unmap. No effect.
5. **Copy with a SYCL kernel** (`parallel_for` over bytes), since a kernel
   `fill_` on the same range does work. Still failed.

## What is not yet known

Why the first GPU does not hit this. The same code, the same model and the same
sizes complete there. That difference is the most likely route to the cause and
it has not been investigated: the first GPU was in use for unrelated work, and
a background driver component was measured holding 40-47 % of its 3D engine,
so it is not a clean comparison either.

The next step is to determine what differs between the two adapters for a
sub-2 MiB copy into a `zeVirtualMemMap` range - copy-engine availability,
peer/host visibility flags, or the page size reported by
`zeVirtualMemQueryPageSize` - rather than to try another fix.

## Separate defect found while investigating

`src/vrambuf.c` called the blocking `vbars_free()` on an allocation path in
three places. That function calls `cuCtxSynchronize()`, which on the XPU
backend is `sycl::queue::wait_and_throw()`, and waiting there can block behind
work that itself needs the residency being requested. Every equivalent site in
`src/model-vbar.c` had already been converted to the non-blocking
`vbars_free_retired()` on Windows XPU; `vrambuf.c` was missed.

This is a real defect, it is **not** the cause of the failure above (fixing it
changed nothing), and it is **not** currently in the tree: the working state was
restored to the build used for the stress test. The change is preserved in
`build/trace/diagnostics-and-wip.patch` and should be re-applied on its own
merits, ideally behind a `vbars_reclaim_for_allocation()` helper so the
platform rule lives in one place.
