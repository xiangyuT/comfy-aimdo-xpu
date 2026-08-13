# Windows XPU second-GPU small VBAR copy workaround

Some Windows multi-adapter Intel XPU systems reject host-to-VBAR copies of
2 MiB or less on a non-primary Level Zero node with
`ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, even when the device has ample free
memory. AIMDO provides a bounded correctness workaround for that driver path.

## Current route

`comfy_aimdo.torch.copy_to_vbar()` checks whether the request is:

- Windows XPU;
- CPU source to XPU destination;
- contiguous, with identical shape and dtype;
- 1 through 2 MiB;
- inside a fully mapped and pinned AIMDO VBAR range;
- on an adapter whose Level Zero node mask is not `0x1`.

When all conditions hold, AIMDO pads the host-to-private-device staging copy
to 2 MiB + 1 byte, then launches a kernel that writes exactly the requested
range into VBAR. Padding never touches VBAR and cannot overwrite an adjacent
allocation. All other requests use the normal `Tensor.copy_` path.

ComfyUI integrations that write ordinary gathered weights or quantized layout
parameters into AIMDO VBAR storage must call this helper rather than invoking
`Tensor.copy_` directly.

`AIMDO_XPU_SMALL_VBAR_COPY_FALLBACK=0` disables the workaround for driver
qualification. Setting it to `1` forces the route for focused diagnosis.

## Diagnostics

Build the adapter and direct-copy probes:

```powershell
cmd /d /c scripts\build-windows-xpu-device-diagnostic.cmd
cmd /d /c scripts\build-windows-xpu-vmm-copy-diagnostic.cmd
```

Run the production reproducer on the affected adapter:

```powershell
<portable>\python_embeded\python.exe -s tests\repro_windows_xpu_vbar_copy.py `
    --device 1 --pages 5 --copy-bytes 607744 --workaround
```

Validation requires byte-for-byte correctness and zero fallback failures.
`control.get_xpu_vmm_stats()` exposes
`small_vbar_copy_fallback_calls`, `small_vbar_copy_fallback_bytes`, and
`small_vbar_copy_fallback_failures` for an end-to-end run.

The node mask is a selector for the observed affected adapter, not a claim
about the driver's internal cause. Re-run the direct diagnostic after a driver
update. If direct copies at 1 byte, the workload-specific size, and 2 MiB all
pass on every adapter, disable the fallback, repeat the real workload, and only
then remove the workaround.

## Development history

The original failure, copy-size matrix, rejected UR-detour and read-modify-write
approaches, adapter identity, and 2026-08-13 formal workload evidence are
preserved in
`omni-xpu-kernel-tuning/docs/results/bmg/2026-08-13/`. This file describes only
the maintained route and its removal gate.
