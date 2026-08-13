# Windows XPU second-GPU small VBAR copy failure

Status: **workaround implemented and end-to-end validated on 2026-08-13.**

## Result

The second Intel Arc Pro B70 rejects a host-to-VMM memory copy when the copy is
2 MiB or smaller. The first B70 accepts the same operation. The error is the
driver's `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (`0x70000003`), not an actual
memory shortage and not AIMDO's allocation-pressure policy.

AIMDO now avoids that driver path on the affected Windows adapter. It copies a
2 MiB + 1 byte padded host buffer into private device staging, then launches a
kernel that writes exactly the requested byte range into VBAR. Padding never
touches VBAR, so the workaround cannot overwrite a neighboring allocation.

ComfyUI routes both ordinary gathered weights and quantized layout-parameter
tensors through this operation. The fallback is deliberately limited to:

* Windows XPU;
* a CPU source and contiguous, same-shape/same-dtype XPU destination;
* a fully mapped and pinned AIMDO VBAR range;
* a transfer of 1 through 2 MiB;
* the adapter reporting a Level Zero node mask other than `0x1`.

`AIMDO_XPU_SMALL_VBAR_COPY_FALLBACK=0` disables the route for a fixed driver;
setting it to `1` forces the route for diagnostics.

## Original symptom

With DynamicVRAM on the second physical GPU, MiniMax H3 failed about two
seconds into text-encoder loading:

```text
RuntimeError: level_zero backend failed with error: 39
  comfy/ops.py in cast_maybe_lowvram_patch
  comfy/model_management.py in cast_to_gathered
  comfy_kitchen/tensor/base.py in copy_from -> Tensor.copy_
```

The same GPU completed with DynamicVRAM disabled because that path did not copy
into a Level Zero virtual-memory mapping.

## Isolation evidence

`tests/repro_windows_xpu_vbar_copy.py` reduces the failure to one 607,744-byte
copy. The transfer-size boundary is exact:

| Copy size | Physical GPU 0 | Physical GPU 1 |
| ---: | --- | --- |
| 607,744 B | pass | fail |
| 2 MiB | pass | fail |
| 2 MiB + 1 B | pass | pass |

At failure, all VBAR pages were mapped and pinned, the source was host USM, the
queue and mapping shared a Level Zero context, and about 31 GiB remained free.
Direct immediate Level Zero command lists failed on both the compute and copy
queue groups with the same raw result, which excludes Unified Runtime and
PyTorch as the origin of the rejection.

`tests/diagnose_windows_xpu_devices.cpp` found the two adapters equivalent in
memory properties, command-queue groups, D3D12 node count, and VMM page-size
queries. Their Level Zero node masks differed:

| Adapter | PCI | LUID | Node mask |
| --- | --- | --- | ---: |
| physical GPU 0 | `04:00.0` | `00000000:00016072` | `0x1` |
| physical GPU 1 | `31:00.0` | `00000000:000163ad` | `0x2` |

The node mask is a reliable selector on this system, not a claim about the
driver's internal root cause.

## Why the workaround has two stages

The standalone `tests/diagnose_windows_xpu_vmm_copy.cpp` established:

* direct host-to-VBAR copy at or below 2 MiB fails;
* changing the VMM backing page size does not help;
* direct Level Zero copies on compute and copy engines both fail;
* a kernel can write the VBAR mapping;
* host-to-private-device staging also fails when its transfer is at or below
  2 MiB on the affected adapter;
* padding only the private staging transfer to 2 MiB + 1 and then kernel-copying
  the exact requested range passes with byte verification.

A padded read-modify-write directly against VBAR also passed, but was rejected
for production because it would race with adjacent allocations. The selected
algorithm never writes outside the caller's destination range.

An attempted `urEnqueueUSMMemcpy` detour was also rejected. Calling SYCL again
inside the intercepted UR frame returned `device or resource busy`; moving the
call to a joined worker deadlocked on the runtime lock. The working route is
therefore invoked above PyTorch/UR at ComfyUI's AIMDO transfer boundary.

## Validation

Build the two standalone diagnostics:

```powershell
scripts\build-windows-xpu-device-diagnostic.cmd
scripts\build-windows-xpu-vmm-copy-diagnostic.cmd
```

Run the production reproducer against physical GPU 1:

```powershell
<portable>\python_embeded\python.exe -s tests\repro_windows_xpu_vbar_copy.py `
    --device 1 --pages 5 --copy-bytes 607744 --workaround
```

The verified result was five successful copies, five fallback calls,
3,038,720 copied bytes, and zero fallback failures. A 30-case matrix (two
adapters, three boundary sizes, five VBAR pages) passed with byte verification.

A fresh 864x480 H3 run with the final wheel recorded 651 fallback calls,
45,562,744 requested bytes, and zero fallback failures inside the ComfyUI
process. This confirms the end-to-end pass exercised the workaround heavily.

The formal MiniMax H3 workload and its GPU-specific gates then passed on
physical GPU 1 in one fresh ComfyUI process:

* three sequential prompts with distinct seeds;
* 1280x736, 362 frames, 24 fps, two steps;
* all three `execution_success`;
* server times 467.93 s, 456.23 s, and 456.47 s;
* three valid 15.083-second MP4 files;
* no UR error 39, AIMDO copy failure, `result=999`, or device reset.

Windows asyncio logged `ConnectionResetError [WinError 10054]` after the first
two client connections closed. The server continued immediately and all three
prompts passed; this is a separate HTTP transport log issue, not an XPU error.
It still means the broader acceptance document's literal no-traceback log gate
is not clean and should be tracked independently.

## Remaining driver gap

The workaround restores correctness but does not explain why the driver binds
the small-copy behavior to the second adapter. A future driver qualification
should rerun the direct diagnostic. If direct copies pass, disable the route
with the environment override, repeat the formal gate, then remove the
workaround rather than carrying it indefinitely.
