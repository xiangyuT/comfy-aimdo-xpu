# AI Model Dynamic Offloader

This project is a pytorch VRAM allocator that implements on-demand offloading of model weights when the primary pytorch VRAM allocator comes under pressure.

## Support:

* **Nvidia CUDA GPUs**
* **Intel XPU GPUs on Linux** through the Level Zero backend
* **Pytorch 2.8+** for the CUDA backend
* A PyTorch XPU build exposing the current stream's SYCL queue for the Intel
  backend and providing `torch.xpu.memory.XPUPluggableAllocator`
* **Cuda 12.8+** for the Nvidia backend
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
  `zePhysicalMemCreate()`, `zeVirtualMemMap()`, and their converse APIs. It
  installs an XPU allocator backed by `sycl::malloc_device()` for regular
  tensors. Before that allocator grows physically, it evicts unpinned VBAR
  pages according to the same AIMDO budget. Freed regular allocations are
  cached per device and SYCL queue, with completion barriers protecting reuse.
  `torch.xpu.empty_cache()` releases completed cached blocks, and the allocator
  reports its live and reserved bytes through PyTorch's XPU memory-stat APIs.

### Build the Intel XPU backend

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

## Caveats:

* There is no real way for this allocator to tell the difference between high usage and bad fragmentation in the pytorch caching allocator. As we always return success to the pytorch caching allocator it experiences no pressure while weights are being offloaded which means it can run in an extremely fragmented mode. The assumption is model weight access patterns are reasonably regular over blocks or iterations and it finds a good set of sizes to cache. What you should generally do though, is completely flush the pytorch caching allocator before each new model run, which avoids completely un-used reservations from taking priority over the next models weights.
