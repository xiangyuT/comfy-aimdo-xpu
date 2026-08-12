"""Windows Unified Runtime USM interception experiment.

This is the falsifiable component test that must pass before any decision is
made about porting the Linux ``dev/xpu-native-allocator-hook`` design to
Windows. It proves, or disproves, four independent claims:

1. ``ur_loader.dll!urUSMDeviceAlloc`` can be intercepted in a live PyTorch XPU
   process using Detours, without preloading anything before the interpreter
   starts.
2. PyTorch XPU tensor allocations actually reach that entry point.
3. The caller can be classified as PyTorch's native caching allocator by
   walking the stack for ``c10_xpu.dll``, which is the Windows equivalent of
   the Linux prototype's ``libc10_xpu.so`` check.
4. A synthetic ``UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`` makes
   ``XPUCachingAllocator::malloc`` release its cached blocks and retry the same
   request, rather than raising ``torch.OutOfMemoryError``.

Claim 4 is the load-bearing one. The whole Linux design depends on it, and it
is also the only mechanism by which AIMDO can ever see PyTorch's cached blocks:
those blocks never reach ``zeMemFree``, which is why the existing Windows
``zeMemAllocDevice`` detour cannot reclaim them.

Run it with the Portable interpreter, which is the recorded acceptance
environment::

    <portable>\\python_embeded\\python.exe -s tests\\run_windows_ur_hook_probe.py
"""

import argparse
import ctypes
import json
import os
import pathlib
import sys

MIB = 1024 * 1024


def _probe_path(explicit):
    if explicit:
        return pathlib.Path(explicit)
    root = pathlib.Path(__file__).resolve().parent.parent
    return root / "build" / "ur-probe" / "ur_usm_detour_probe.dll"


class Probe:
    def __init__(self, path):
        self._library = ctypes.CDLL(str(path))
        self._library.probe_install.restype = ctypes.c_int
        self._library.probe_remove.restype = ctypes.c_int
        self._library.probe_stat_count.restype = ctypes.c_size_t
        self._library.probe_stat_name.argtypes = [ctypes.c_size_t]
        self._library.probe_stat_name.restype = ctypes.c_char_p
        self._library.probe_get_stats.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_size_t,
        ]
        self._library.probe_get_stats.restype = ctypes.c_int
        self._library.probe_arm_synthetic_oom.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_int,
        ]
        self._library.probe_set_trace.argtypes = [ctypes.c_int]
        self._library.probe_last_stack.restype = ctypes.c_char_p
        self._library.probe_module_path.restype = ctypes.c_char_p
        self._library.probe_direct_alloc.argtypes = [
            ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.probe_direct_alloc.restype = ctypes.c_int
        self._library.probe_direct_free.argtypes = [ctypes.c_void_p]
        self._library.probe_direct_free.restype = ctypes.c_int
        self._count = int(self._library.probe_stat_count())
        self._names = [
            self._library.probe_stat_name(i).decode() for i in range(self._count)
        ]

    def install(self):
        return bool(self._library.probe_install())

    def remove(self):
        return bool(self._library.probe_remove())

    def stats(self):
        values = (ctypes.c_ulonglong * self._count)()
        if not self._library.probe_get_stats(values, self._count):
            raise RuntimeError("probe_get_stats failed")
        return dict(zip(self._names, [int(v) for v in values]))

    def arm_synthetic_oom(self, size, torch_only=True):
        self._library.probe_arm_synthetic_oom(
            ctypes.c_ulonglong(size), ctypes.c_int(1 if torch_only else 0)
        )

    def set_trace(self, enabled):
        self._library.probe_set_trace(ctypes.c_int(1 if enabled else 0))

    def last_stack(self):
        value = self._library.probe_last_stack()
        return value.decode() if value else ""

    def module_path(self):
        value = self._library.probe_module_path()
        return value.decode() if value else ""

    def direct_alloc(self, size):
        pointer = ctypes.c_void_p()
        result = self._library.probe_direct_alloc(
            ctypes.c_ulonglong(size), ctypes.byref(pointer)
        )
        return result, pointer

    def direct_free(self, pointer):
        return self._library.probe_direct_free(pointer)


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
        return passed

    @property
    def ok(self):
        return all(entry["passed"] for entry in self.checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--target-mib",
        type=int,
        default=512,
        help="size of the allocation used to trigger the synthetic OOM",
    )
    parser.add_argument(
        "--cache-block-mib",
        type=int,
        default=64,
        help="size of the cached blocks that must be released by the retry",
    )
    parser.add_argument("--cache-blocks", type=int, default=6)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    probe_path = _probe_path(arguments.probe)
    if not probe_path.exists():
        print(f"probe not built: {probe_path}", file=sys.stderr)
        print("run scripts\\build-windows-ur-probe.cmd first", file=sys.stderr)
        return 2

    import torch

    report = Report()
    print(f"torch {torch.__version__}")
    print(f"alloc conf {os.environ.get('PYTORCH_ALLOC_CONF', '<unset>')}")
    if not torch.xpu.is_available():
        print("XPU is not available", file=sys.stderr)
        return 2

    device = f"xpu:{arguments.device}"

    # Hook before torch initialises XPU. This matches how AIMDO installs its
    # hooks from control.init(), and it exposes runtime allocations that do not
    # come from the caching allocator, which is what makes the caller
    # classification check falsifiable rather than vacuous.
    probe = Probe(probe_path)
    installed = probe.install()
    report.check(
        "1. detours attach to ur_loader.dll!urUSMDeviceAlloc in a live process",
        installed,
        probe.module_path(),
    )
    if not installed:
        return 1
    probe.set_trace(arguments.trace)

    stats = probe.stats()
    report.check(
        "1b. hooked module is the one the SYCL runtime itself uses",
        stats["resolved_via_proxy"] == 1 and stats["proxy_agrees_with_byname"] == 1,
        f"resolved_via_proxy={stats['resolved_via_proxy']} "
        f"proxy_agrees_with_byname={stats['proxy_agrees_with_byname']}",
    )

    torch.xpu.init()
    warmup = torch.empty(1024, dtype=torch.uint8, device=device)
    del warmup
    torch.xpu.synchronize()

    # --- Claim 2 and 3 -----------------------------------------------------
    torch.xpu.empty_cache()
    before = probe.stats()
    probed = torch.empty(arguments.target_mib * MIB, dtype=torch.uint8, device=device)
    torch.xpu.synchronize()
    after = probe.stats()
    interception_stack = probe.last_stack()

    report.check(
        "2. torch xpu allocation reaches the intercepted UR entry point",
        after["alloc_calls"] > before["alloc_calls"],
        f"alloc_calls {before['alloc_calls']} -> {after['alloc_calls']}",
    )
    report.check(
        "3. caller classified as the torch native allocator via c10_xpu.dll",
        after["torch_native_alloc_calls"] > before["torch_native_alloc_calls"],
        interception_stack,
    )
    observed_size = after["last_alloc_size"]
    report.check(
        "3b. intercepted request size matches the tensor",
        observed_size >= arguments.target_mib * MIB,
        f"last_alloc_size={observed_size}",
    )

    del probed
    torch.xpu.empty_cache()
    torch.xpu.synchronize()

    # Negative control for the caller classification: replay a device
    # allocation from the probe module itself, using the context and device
    # handles captured from the PyTorch request above.
    before_direct = probe.stats()
    direct_result, direct_pointer = probe.direct_alloc(1 * MIB)
    direct_after = probe.stats()
    direct_stack = probe.last_stack()
    if direct_result == 0 and direct_pointer:
        probe.direct_free(direct_pointer)

    report.check(
        "3c. classification discriminates: a non-torch caller is not misread",
        direct_after["other_alloc_calls"] == before_direct["other_alloc_calls"] + 1
        and direct_after["torch_native_alloc_calls"]
        == before_direct["torch_native_alloc_calls"],
        f"direct_result={direct_result} "
        f"other {before_direct['other_alloc_calls']}->{direct_after['other_alloc_calls']} "
        f"torch_native {before_direct['torch_native_alloc_calls']}"
        f"->{direct_after['torch_native_alloc_calls']} :: {direct_stack}",
    )

    # --- Claim 4 -----------------------------------------------------------
    # Populate the cache with blocks that cannot serve the target request, so
    # the retry can only succeed if release_cached_blocks() actually returned
    # them to the driver.
    cache_tensors = [
        torch.empty(arguments.cache_block_mib * MIB, dtype=torch.uint8, device=device)
        for _ in range(arguments.cache_blocks)
    ]
    torch.xpu.synchronize()
    for tensor in cache_tensors:
        del tensor
    cache_tensors.clear()
    torch.xpu.synchronize()

    cached_bytes = torch.xpu.memory_reserved(arguments.device) - torch.xpu.memory_allocated(
        arguments.device
    )
    report.check(
        "4a. torch is holding freed-but-cached blocks the driver cannot see",
        cached_bytes > 0,
        f"cached={cached_bytes / MIB:.0f} MiB",
    )

    armed = probe.stats()
    probe.arm_synthetic_oom(observed_size, torch_only=True)

    oom_raised = None
    try:
        retried = torch.empty(
            arguments.target_mib * MIB, dtype=torch.uint8, device=device
        )
        torch.xpu.synchronize()
    except Exception as error:  # noqa: BLE001 - the exception type is the result
        oom_raised = f"{type(error).__name__}: {error}"
        retried = None

    final = probe.stats()

    report.check(
        "4b. the synthetic OUT_OF_DEVICE_MEMORY was delivered to torch",
        final["synthetic_oom_calls"] == armed["synthetic_oom_calls"] + 1,
        f"synthetic_oom_calls={final['synthetic_oom_calls']}",
    )
    report.check(
        "4c. torch released cached blocks in response",
        final["frees_between_oom_and_retry"] > 0,
        "frees_between_oom_and_retry="
        f"{final['frees_between_oom_and_retry']} "
        f"bytes={final['bytes_freed_between_oom_and_retry'] / MIB:.0f} MiB",
    )
    report.check(
        "4d. torch retried the same request instead of raising",
        final["retry_alloc_calls"] == armed["retry_alloc_calls"] + 1,
        f"retry_alloc_calls={final['retry_alloc_calls']}",
    )
    report.check(
        "4e. the allocation ultimately succeeded",
        retried is not None and oom_raised is None,
        oom_raised or "no exception",
    )

    if retried is not None:
        retried.fill_(7)
        torch.xpu.synchronize()
        value = int(retried[-1].item())
        report.check(
            "4f. the retried allocation is usable memory",
            value == 7,
            f"last byte={value}",
        )
        del retried

    torch.xpu.empty_cache()
    probe.remove()

    summary = {
        "torch": torch.__version__,
        "device": device,
        "ur_loader": probe.module_path(),
        "interception_stack": interception_stack,
        "stats": final,
        "checks": report.checks,
        "passed": report.ok,
    }
    print()
    print("stack observed at the intercepted allocation:")
    print(f"  {interception_stack}")
    print()
    print("final probe counters:")
    for key, value in final.items():
        print(f"  {key:34s} {value}")

    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(summary, indent=2))

    print()
    print("RESULT:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
