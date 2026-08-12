"""Windows XPU VBAR versus native Torch allocation competition reproducer.

`repro_windows_xpu_malloc_pressure.py` establishes that Torch's USM device
allocations never fail on Windows: WDDM demotes pages to non-local memory
instead. AIMDO's VBAR pages are not USM allocations. They are explicit Level
Zero physical objects that are mapped and made resident, so they cannot be
demoted the same way.

This reproducer measures what that asymmetry does when both compete:

* phase 1 faults VBAR pages on an idle device and records the achieved
  residency;
* phase 2 grows a native Torch working set that is deliberately allowed to
  exceed local memory;
* phase 3 faults more VBAR pages while Torch holds that memory and records
  whether the VBAR fault fails while Torch allocation keeps succeeding;
* phase 4 flushes Torch's cache and refaults to see whether VBAR residency
  recovers.

AIMDO must be importable and an XPU device must be available. Run it under the
Portable interpreter with -s.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The Portable interpreter is an embeddable build whose ._pth file ignores
# PYTHONPATH, so an explicit insert is the only way to test a locally built
# comfy_aimdo instead of the installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIB = 1024 ** 2
GIB = 1024 ** 3

# src/model-vbar.c: #define VBAR_PAGE_SIZE (32 << 20). The baseline package
# does not export it, so keep the reproducer independent of that detail.
VBAR_PAGE_SIZE = 32 * MIB


def emit(**fields):
    fields["t"] = round(time.time(), 3)
    print(json.dumps(fields), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vbar-gib", type=float, default=6.0)
    parser.add_argument("--torch-gib", type=float, default=30.0)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--debug", action="store_true",
                        help="enable AIMDO's own debug logging")
    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="aimdo: %(message)s",
                            stream=sys.stderr)

    from comfy_aimdo import control

    if not control.init(implementation="xpu",
                        simple_vram_headroom=args.reserve_mib * MIB):
        raise SystemExit("comfy_aimdo control.init() failed")

    import torch

    if not control.init_devices([args.device]):
        raise SystemExit("comfy_aimdo control.init_devices() failed")

    from comfy_aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin
    from comfy_aimdo.torch import aimdo_to_tensor

    device = torch.device("xpu", args.device)
    properties = torch.xpu.get_device_properties(args.device)
    emit(
        event="start", pid=os.getpid(), device=args.device,
        name=properties.name,
        total_memory_mib=int(properties.total_memory) // MIB,
        page_mib=VBAR_PAGE_SIZE // MIB, torch=torch.__version__,
        vbar_gib=args.vbar_gib, torch_gib=args.torch_gib,
        reserve_mib=args.reserve_mib,
    )

    page_count = int(args.vbar_gib * GIB) // VBAR_PAGE_SIZE
    vbar = ModelVBAR(page_count * VBAR_PAGE_SIZE, device=args.device)
    allocations = [vbar.alloc(VBAR_PAGE_SIZE) for _ in range(page_count)]

    def fault_all(phase):
        """One model pass: fault, use, then unpin every weight in order."""
        faulted = 0
        missed = 0
        errors = 0
        for allocation in allocations:
            try:
                if vbar_fault(allocation) is not None:
                    tensor = aimdo_to_tensor(allocation, device)
                    tensor.fill_(1)
                    # ComfyUI unpins as soon as the operator is submitted, so
                    # the page becomes a legal eviction victim afterwards.
                    vbar_unpin(allocation)
                    faulted += 1
                else:
                    missed += 1
            except Exception as exception:
                errors += 1
                emit(event="fault_error", phase=phase,
                     error=f"{type(exception).__name__}: {exception}")
                break
        torch.xpu.synchronize(args.device)
        stats = torch.xpu.memory_stats(args.device)
        vmm = control.get_xpu_vmm_stats()
        # The UR hook only runs inside urUSMDeviceAlloc, so these counters show
        # whether a VBAR fault can ask PyTorch for its cache back. A fault does
        # not allocate USM, so it cannot.
        hook = control.get_xpu_ur_hook_stats()
        emit(
            event="fault_pass", phase=phase, pages=len(allocations),
            faulted=faulted, missed=missed, errors=errors,
            vbar_resident_mib=vbar.loaded_size() // MIB,
            watermark=vbar.get_watermark(),
            unmap_calls=vmm.get("unmap_calls", 0),
            physical_release_calls=vmm.get("physical_release_calls", 0),
            hook_synthetic_oom=hook.get("synthetic_oom_calls", 0),
            hook_retry_eviction=hook.get("retry_eviction_calls", 0),
            hook_native_reclaim_free=hook.get("native_reclaim_free_calls", 0),
            torch_reserved_mib=int(
                stats.get("reserved_bytes.all.current", 0)) // MIB,
        )
        return faulted

    idle_faulted = fault_all("idle_device")

    if args.debug:
        control.set_log_debug()

    blocks = []
    block_mib = 256
    torch_failures = 0
    block_count = int(args.torch_gib * 1024) // block_mib
    for index in range(block_count):
        try:
            block = torch.empty(block_mib * MIB // 2, dtype=torch.float16,
                                device=device)
            block.fill_(1.0)
            blocks.append(block)
        except Exception as exception:
            torch_failures += 1
            emit(event="torch_alloc_failed",
                 error=f"{type(exception).__name__}: {exception}")
            break
        if (index + 1) % 20 == 0 or index + 1 == block_count:
            # Sampled without synchronizing: this is what the allocation hook
            # managed to release while Torch was still submitting work.
            emit(event="pressure_progress",
                 torch_mib=len(blocks) * block_mib,
                 vbar_resident_mib=vbar.loaded_size() // MIB,
                 watermark=vbar.get_watermark(),
                 unmap_calls=control.get_xpu_vmm_stats().get("unmap_calls", 0))
    torch.xpu.synchronize(args.device)
    stats = torch.xpu.memory_stats(args.device)
    emit(
        event="torch_pressure_applied", blocks=len(blocks),
        torch_mib=len(blocks) * block_mib, torch_alloc_failures=torch_failures,
        torch_reserved_mib=int(
            stats.get("reserved_bytes.all.current", 0)) // MIB,
        vbar_resident_mib=vbar.loaded_size() // MIB,
    )

    under_pressure_faulted = fault_all("torch_holds_memory")

    del blocks
    torch.xpu.synchronize(args.device)
    # No empty_cache() here: Torch still holds every block in its own cache,
    # which is exactly the memory AIMDO cannot reclaim through the allocation
    # hook. A fault must be able to recover it.
    dead_cache_faulted = fault_all("after_free_without_empty_cache")

    torch.xpu.empty_cache()
    torch.xpu.synchronize(args.device)
    recovered_faulted = fault_all("after_torch_empty_cache")

    # A model reactivation is the only boundary that reopens the address
    # range, so measure recovery with and without it.
    vbar.prioritize()
    reprioritized_faulted = fault_all("after_prioritize")

    emit(
        event="verdict",
        idle_faulted=idle_faulted,
        under_pressure_faulted=under_pressure_faulted,
        dead_cache_faulted=dead_cache_faulted,
        recovered_faulted=recovered_faulted,
        reprioritized_faulted=reprioritized_faulted,
        torch_alloc_failures=torch_failures,
        vmm=control.get_xpu_vmm_stats(),
    )

    del allocations
    vbar.__del__()
    control.deinit()


if __name__ == "__main__":
    main()
