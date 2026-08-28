# AI Model Dynamic Offloader XPU

> [!IMPORTANT]
> **Experimental Intel XPU fork — limited validation scope**
>
> The XPU changes in this fork are developed and performance-tuned for work
> related to [Intel llm-scaler](https://github.com/intel/llm-scaler), primarily
> on Ubuntu 24.04 LTS with Intel Arc Pro B70.
>
> Other operating systems, GPU models, software environments, workloads, and
> configurations are outside the validated scope. Their compatibility,
> correctness, output quality, stability, and performance are not guaranteed.
>
> If you have a specific requirement outside the current scope, you may open a
> relevant issue with the complete environment, expected use case, and a
> reproducible example. Opening an issue provides a way to document and discuss
> the request. Whether it can be explored or supported will depend on its
> relevance, reproducibility, and available capacity. The validated scope will
> be updated only after any additional support has been implemented and
> validated.
>
> This work is experimental. Ongoing support, maintenance, compatibility
> updates, and future development depend on available time, capacity, and
> project priorities.

## Intel XPU integration

This fork adds an Intel XPU backend to
[Comfy-Org/comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo), using
PyTorch XPU, SYCL, and Level Zero for DynamicVRAM model-weight management.

Linux and Windows use different allocator ownership models:

- Linux can install AIMDO's XPU pluggable allocator and manages VBAR storage
  through SYCL and Level Zero.
- Windows retains PyTorch's native XPU caching allocator while AIMDO
  coordinates physical Unified Runtime allocation pressure and manages Level
  Zero VBAR pages.

Within ComfyUI, installing this package does not enable the XPU allocator by
itself. DynamicVRAM must be selected explicitly:

```bash
python main.py --enable-dynamic-vram
```

Standalone callers can initialize the backend with
`control.init(implementation="xpu")`. Initialization must occur before the
first XPU stream or allocation is created.

### Build

Build the Linux backend in an environment containing PyTorch XPU, the oneAPI
DPC++ compiler, and Level Zero development files:

```bash
./scripts/build-linux-xpu.sh
```

For Windows prerequisites and the native build entry point, see
[Windows XPU build and validation](docs/WINDOWS_XPU_BUILD_TEST_ACCEPTANCE.md).

The [documentation index](docs/README.md) links the maintained allocator
ownership, pressure, reclaim, runtime-hook, build, and validation contracts.

---

## Upstream documentation

> The following is the original README from
> [Comfy-Org/comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo) at revision
> [`84b79a6c950dfe51f84294fce13232f4de2aa5a1`](https://github.com/Comfy-Org/comfy-aimdo/commit/84b79a6c950dfe51f84294fce13232f4de2aa5a1)
> (upstream `v0.4.15`), retained unchanged for reference.

# AI Model Dynamic Offloader

This project is a pytorch VRAM allocator that implements on-demand offloading of model weights when the primary pytorch VRAM allocator comes under pressure.

## Support:

* **Nvidia GPUs** (CUDA) **and AMD GPUs** (ROCm/HIP)
* **PyTorch 2.8+**
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

On AMD, the equivalent HIP APIs (`hipMemAddressReserve` -> `hipMemCreate` -> `hipMemMap`, and their converse calls) are used throughout via the same flow.

## Caveats:

* There is no real way for this allocator to tell the difference between high usage and bad fragmentation in the pytorch caching allocator. As we always return success to the pytorch caching allocator it experiences no pressure while weights are being offloaded which means it can run in an extremely fragmented mode. The assumption is model weight access patterns are reasonably regular over blocks or iterations and it finds a good set of sizes to cache. What you should generally do though, is completely flush the pytorch caching allocator before each new model run, which avoids completely un-used reservations from taking priority over the next models weights.
