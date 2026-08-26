# AI Model Dynamic Offloader

This project is a pytorch VRAM allocator that implements on-demand offloading of model weights when the primary pytorch VRAM allocator comes under pressure.

## Support:

* **Nvidia CUDA GPUs** and **AMD GPUs** through ROCm/HIP
* **Intel XPU GPUs on Linux and Windows** through the Level Zero backend
* **PyTorch 2.8+** for the CUDA and ROCm backends
* A PyTorch XPU build exposing the current stream's SYCL queue for the Intel
  backend. Linux additionally requires
  `torch.xpu.memory.XPUPluggableAllocator`.
* **CUDA 12.8+** (Nvidia) / **ROCm 7+** (AMD)
* **Windows 11+** / **Linux** as per python ManyLinux support

---

## How it works:

* The pytorch application creates a Virtual Base Address Register (**VBAR**) for a model. Creating a VBAR doesn't cost any VRAM, only GPU virtual address space (which is pretty much free).
* The pytorch application allocates tensors for model weights within the VBAR. These tensors are initially un-allocated and will segfault if touched.
* The pytorch application faults in the tensors using the `fault()` API at the time the tensor is needed. This is where VRAM actually gets allocated.

##### If the `fault()` is successful (sufficient VRAM for this tensor):
1.  **If the fault() resultant signature is changed or unknown:**
    * The application uses `tensor::_copy()` to populate the weight data on the GPU.
    * The application saves the returned signature against this weight for future comparison
2.  The layer uses the weight tensor.
3.  The application calls `unpin()` on the tensor to allow it to be freed under pressure later if needed.

##### If the `fault()` is unsuccessful (offloaded weight):
1.  The application allocates a temporary regular GPU tensor.
2.  Uses `_copy` to populate weight data on the GPU.
3.  The layer uses the temporary as the weight.
4.  Pytorch garbage collects the temp when the layer is finished.

see examples/example.py

---

## Priorities:

* The most recent VBARs are the highest priority and lower addresses in the VBAR take priority over higher addresses.
* Applications should order their tensor allocations in the VBAR in load-priority order with the lowest addresses for the highest priority weights.
* Calling `fault()` on a weight that is higher priority than other weights will cause those lower priority weights to get freed to make space.
* Having a weight evicted sets that VBAR's watermark to that weight's level. Any weights in the same VBAR above the watermark automatically fail the `fault()` API. This avoids constantly faulting in all weights each model iteration while allowing the application to just blindly call `fault()` every layer and check the results. There is no need for the application to manage any VRAM quotas or watermarks.
* Existing VBARs can be pushed to top priority with the `prioritize()` API. This allows use of an already loaded or partially model (e.g. using the same model twice in a complex workflow). Using `prioritize` resets the offload watermark of that model to no offloading, giving its weights priority over any other currently loaded models.

---

## Backend:

* VBAR allocation is done with `cuMemAddressReserve()`, faulting with `cuMemCreate()` and `cuMemMap()` and all frees done with appropriate converse APIs.
* For consistency with VBAR memory management, main pytorch allocator plugin is also implemented with `cuMemAddressReserve` -> `cuMemCreate` -> `cuMemMap`. This also behaves a lot better on Windows systems with System Memory fallback.
* The Intel XPU backend shares PyTorch's current SYCL queue and Level Zero
  context, and implements VBAR allocation with `zeVirtualMemReserve()`,
  `zePhysicalMemCreate()`, `zeVirtualMemMap()`, and their converse APIs.
* On Windows, PyTorch retains ownership of its native XPU caching allocator.
  AIMDO's default `native_hook` mode intercepts the Unified Runtime physical
  USM growth/release boundary with Detours while preserving PyTorch's block
  splitting, coalescing, stream ordering, retry, statistics, and
  `empty_cache()` behavior. The hook never waits or performs re-entrant driver
  memory management. It lets PyTorch release its own cache on retry, then
  reclaims only a residual deficit from retirement-proven VBAR pages. Model
  prioritization preserves the active VBAR during speculative reclaim, and
  individual faults apply current pressure when residency really must fall.
  On Windows, pages publish the actual consuming SYCL queue before becoming
  idle. Non-blocking two-phase retirement is the default: pressure may unmap a
  page only after every registered queue marker completes and the page
  identity is revalidated. `AIMDO_XPU_ASYNC_VBAR_RECLAIM=0` selects the
  synchronized model-boundary reference oracle; it is diagnostic-only because
  pages touched within one activation otherwise accumulate until a model
  switch and can force WDDM paging.
  Custom or external kernels that can outlive the model pin must use the
  pre-submission `vbar_external_consumer()` lease; post-submission
  `vbar_register_consumer()` is valid only while the original model pin is
  still active. Captured graphs use `vbar_capture_begin()` and keep that lease
  across every replay. Ending graph construction is not a release boundary.
  `control.get_xpu_memory_snapshot()` reports native allocator state and the
  separate foreign VBAR ownership domain side by side.
* On Linux, AIMDO continues to install its XPU pluggable allocator backed by
  `sycl::malloc_device()`. Freed regular allocations are cached per device and
  SYCL queue, with completion barriers protecting reuse.

The allocator and `--reserve-vram` interaction differs between Linux and
Windows. See the [documentation index](docs/README.md) and
[XPU platform memory policy](docs/XPU_PLATFORM_MEMORY_POLICY.md) for the
maintained contracts and regression entry points.

### Build the Intel XPU backend on Linux

Build in a Linux environment containing PyTorch XPU, the oneAPI DPC++ compiler,
and Level Zero development files:

```bash
./scripts/build-linux-xpu.sh
```

The command creates `comfy_aimdo/aimdo_xpu.so`. Vendor detection selects it
automatically for a PyTorch build whose version ends in `+xpu`; callers may
also request `control.init(implementation="xpu")` explicitly. ComfyUI must be
started with DynamicVRAM enabled so its model patcher creates and faults VBAR
weights. The XPU allocator must be installed before the first XPU stream or
allocation is initialized; `control.init()` performs that installation.
Within ComfyUI, installing this package does not enable the XPU allocator by
itself: launch ComfyUI with `--enable-dynamic-vram` to opt in. Starting without
that option, or with `--disable-dynamic-vram`, preserves PyTorch's native XPU
allocator and the legacy ComfyUI model patcher. Standalone callers can opt in
programmatically with `control.init(implementation="xpu")`.

### Build the Intel XPU backend on Windows

Install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the Desktop C++ workload and the
[Intel oneAPI DPC++/C++ Compiler 2025.3](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html).
Obtain Level Zero headers from the official
[oneapi-src/level-zero](https://github.com/oneapi-src/level-zero) repository,
then run from an ordinary command prompt:

```bat
git init build\level-zero-src
git -C build\level-zero-src remote add origin https://github.com/oneapi-src/level-zero.git
git -C build\level-zero-src fetch --depth 1 origin 5cc079af8b1e07329922bea1fde954221ebfd0a3
git -C build\level-zero-src checkout --detach FETCH_HEAD
scripts\build-windows-xpu.cmd
```

The command creates `comfy_aimdo\aimdo_xpu.dll`. The build generates a small
import library for the Level Zero and tracing entry points used by AIMDO; at
runtime the DLL uses `ze_loader.dll` installed by the Intel graphics driver.
Set
`LEVEL_ZERO_INCLUDE` before running the script when the headers are stored
elsewhere. On Windows, the XPU device LUID is matched to its DXGI adapter so
AIMDO can use both WDDM local-memory `CurrentUsage` and `Budget` as its sampled
pressure baseline. This includes driver, SYCL, oneDNN, and other allocations
that do not pass through AIMDO's own accounting; AIMDO applies its recorded
allocation delta between rate-limited DXGI samples. Older runtimes without the
LUID properties fall back to Level Zero free-memory.
ComfyUI must be launched with `--enable-dynamic-vram` on XPU until
upstream ComfyUI enables DynamicVRAM by default for that device type.
On Windows, do not replace Torch's XPU allocator with AIMDO's pluggable
allocator. The native Unified Runtime hook arbitrates physical USM growth;
completed tensor blocks remain under Torch's native cache policy and
`torch.xpu.empty_cache()` remains the native implementation. WDDM non-local
usage is treated as fallback pressure only after it exceeds a 512 MiB margin,
so normal runtime bookkeeping does not cause repeated VBAR eviction. Linux
retains the existing pluggable-allocator and immediate-pressure policy.
For focused Windows diagnosis, set `AIMDO_XPU_WDDM_TRACE=1` to log DXGI
local/non-local usage and budgets, and `AIMDO_XPU_ALLOCATION_TRACE=1` to log
physical Level Zero allocation requests. Both traces are disabled by default.

When the Visual Studio installation does not include the Windows SDK/UCRT, the
script also recognizes the project-local Microsoft Windows SDK NuGet packages
`10.0.26100.3916` extracted below `build\windows-sdk-nuget` (their internal SDK
directory is `10.0.26100.0`):

* [Microsoft.Windows.SDK.CPP](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP/10.0.26100.3916)
* [Microsoft.Windows.SDK.CPP.x64](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP.x64/10.0.26100.3916)
* [Microsoft.Windows.SDK.BuildTools](https://www.nuget.org/packages/Microsoft.Windows.SDK.BuildTools/10.0.26100.3916)

On AMD, the equivalent HIP APIs (`hipMemAddressReserve` -> `hipMemCreate` -> `hipMemMap`, and their converse calls) are used throughout via the same flow.

## Caveats:

* There is no real way for this allocator to tell the difference between high usage and bad fragmentation in the pytorch caching allocator. As we always return success to the pytorch caching allocator it experiences no pressure while weights are being offloaded which means it can run in an extremely fragmented mode. The assumption is model weight access patterns are reasonably regular over blocks or iterations and it finds a good set of sizes to cache. What you should generally do though, is completely flush the pytorch caching allocator before each new model run, which avoids completely un-used reservations from taking priority over the next models weights.
