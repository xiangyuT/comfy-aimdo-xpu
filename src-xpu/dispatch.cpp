extern "C" {
#include "gpu_dispatch.h"
}

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" bool aimdo_xpu_prepare_allocation(int device, size_t size);
extern "C" bool aimdo_xpu_retry_allocation(int device, size_t size);
extern "C" bool aimdo_xpu_account_allocation(int device, int64_t delta);

struct CUevent_st {
    sycl::event event;
    bool recorded = false;
};

namespace {

constexpr CUresult kCudaErrorUnknown = 999;

struct XpuDeviceState {
    int id;
    sycl::queue *queue;
    ze_context_handle_t context;
    ze_device_handle_t device;
};

std::mutex g_devices_mutex;
std::vector<XpuDeviceState> g_devices;

enum XpuStat : size_t {
    kVirtualReserveCalls,
    kVirtualReserveBytes,
    kPhysicalCreateCalls,
    kPhysicalCreateBytes,
    kMapCalls,
    kMapBytes,
    kUnmapCalls,
    kUnmapBytes,
    kPhysicalReleaseCalls,
    kHostToDeviceBytes,
    kQueueRebindCalls,
    kContextSyncCalls,
    kContextSyncCompletions,
    kEventSyncCalls,
    kEventSyncCompletions,
    kSynchronousHostToDeviceCalls,
    kSynchronousHostToDeviceCompletions,
    kTorchAllocatorAllocCalls,
    kTorchAllocatorFreeCalls,
    kTorchAllocatorCacheHits,
    kTorchAllocatorPhysicalAllocCalls,
    kTorchAllocatorPhysicalAllocBytes,
    kTorchAllocatorPhysicalReleaseCalls,
    kTorchAllocatorPhysicalReleaseBytes,
    kXpuStatCount,
};

std::atomic<uint64_t> g_stats[kXpuStatCount];

struct XpuTorchBlock {
    void *pointer;
    size_t size;
    int device;
    sycl::queue queue;
    sycl::event ready;
    bool has_ready = false;

    XpuTorchBlock(void *pointer_value, size_t size_value, int device_value,
                  const sycl::queue &queue_value)
        : pointer(pointer_value),
          size(size_value),
          device(device_value),
          queue(queue_value) {}
};

std::mutex g_torch_allocator_mutex;
std::unordered_map<void *, std::unique_ptr<XpuTorchBlock>>
    g_torch_live_blocks;
std::vector<std::unique_ptr<XpuTorchBlock>> g_torch_cached_blocks;
std::unordered_map<int, uint64_t> g_torch_active_bytes;
std::unordered_map<int, uint64_t> g_torch_reserved_bytes;
std::unordered_map<int, uint64_t> g_torch_peak_active_bytes;
std::unordered_map<int, uint64_t> g_torch_peak_reserved_bytes;

void increase_torch_bytes(std::unordered_map<int, uint64_t> &current,
                          std::unordered_map<int, uint64_t> &peak,
                          int device, uint64_t size) {
    current[device] += size;
    peak[device] = std::max(peak[device], current[device]);
}

void decrease_torch_bytes(std::unordered_map<int, uint64_t> &current,
                          int device, uint64_t size) {
    uint64_t &value = current[device];
    value = size < value ? value - size : 0;
}

bool sync_trace_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("AIMDO_XPU_SYNC_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void trace_sync(const char *operation, const char *phase, uint64_t call,
                const void *queue, size_t size = 0) {
    if (!sync_trace_enabled()) {
        return;
    }
    std::fprintf(stderr,
                 "[AIMDO XPU SYNC] op=%s phase=%s call=%llu queue=%p size=%zu\n",
                 operation, phase, static_cast<unsigned long long>(call),
                 queue, size);
    std::fflush(stderr);
}

extern "C" int aimdo_xpu_current_device(void);

XpuDeviceState *find_device(int id) {
    auto found = std::find_if(
        g_devices.begin(), g_devices.end(),
        [id](const XpuDeviceState &state) { return state.id == id; });
    return found == g_devices.end() ? nullptr : &*found;
}

XpuDeviceState *current_device() {
    return find_device(aimdo_xpu_current_device());
}

sycl::queue *resolve_queue(CUstream stream) {
    if (stream) {
        auto *queue = reinterpret_cast<sycl::queue *>(stream);
        // Torch's current XPU stream is thread-local. ComfyUI initializes
        // AIMDO on the server thread but faults and consumes model weights on
        // a worker thread, so the queue seen by file-to-device copies is the
        // authoritative owner for subsequent VBAR synchronization.
        auto *state = current_device();
        if (state && state->queue != queue) {
            state->queue = queue;
            g_stats[kQueueRebindCalls].fetch_add(1, std::memory_order_relaxed);
        }
        return queue;
    }
    auto *state = current_device();
    return state ? state->queue : nullptr;
}

CUresult from_ze(ze_result_t result) {
    if (result == ZE_RESULT_SUCCESS) {
        return CUDA_SUCCESS;
    }
    if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
        result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return kCudaErrorUnknown;
}

CUresult xpu_init(unsigned int) {
    return from_ze(zeInit(0));
}

CUresult xpu_get_error_string(CUresult error, const char **description) {
    if (!description) {
        return kCudaErrorUnknown;
    }
    switch (error) {
        case CUDA_SUCCESS:
            *description = "success";
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            *description = "Level Zero out of memory";
            break;
        default:
            *description = "Level Zero or SYCL error";
            break;
    }
    return CUDA_SUCCESS;
}

CUresult xpu_context_get_device(CUdevice *device) {
    if (!device || !current_device()) {
        return kCudaErrorUnknown;
    }
    *device = current_device()->id;
    return CUDA_SUCCESS;
}

CUresult xpu_context_synchronize() {
    auto *state = current_device();
    if (!state) {
        return kCudaErrorUnknown;
    }
    const uint64_t call =
        g_stats[kContextSyncCalls].fetch_add(1, std::memory_order_relaxed) + 1;
    trace_sync("context", "begin", call, state->queue);
    try {
        state->queue->wait_and_throw();
        g_stats[kContextSyncCompletions].fetch_add(
            1, std::memory_order_relaxed);
        trace_sync("context", "end", call, state->queue);
        return CUDA_SUCCESS;
    } catch (...) {
        trace_sync("context", "error", call, state->queue);
        return kCudaErrorUnknown;
    }
}

CUresult xpu_device_get(CUdevice *device, int ordinal) {
    if (!device || !find_device(ordinal)) {
        return kCudaErrorUnknown;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult xpu_device_get_attribute(int *value, CUdevice_attribute attribute,
                                  CUdevice device) {
    auto *state = find_device(device);
    if (!value || !state || attribute != CU_DEVICE_ATTRIBUTE_INTEGRATED) {
        return kCudaErrorUnknown;
    }
    *value = 0;
    return CUDA_SUCCESS;
}

CUresult xpu_device_total_memory(size_t *bytes, CUdevice device) {
    auto *state = find_device(device);
    if (!bytes || !state) {
        return kCudaErrorUnknown;
    }
    try {
        *bytes = state->queue->get_device()
                     .get_info<sycl::info::device::global_mem_size>();
        return CUDA_SUCCESS;
    } catch (...) {
        return kCudaErrorUnknown;
    }
}

CUresult xpu_device_get_name(char *name, int length, CUdevice device) {
    auto *state = find_device(device);
    if (!name || length <= 0 || !state) {
        return kCudaErrorUnknown;
    }
    try {
        const std::string value = state->queue->get_device()
                                      .get_info<sycl::info::device::name>();
        std::strncpy(name, value.c_str(), static_cast<size_t>(length) - 1);
        name[length - 1] = '\0';
        return CUDA_SUCCESS;
    } catch (...) {
        return kCudaErrorUnknown;
    }
}

CUresult xpu_device_get_uuid(CUuuid *uuid, CUdevice device) {
    if (!uuid || !find_device(device)) {
        return kCudaErrorUnknown;
    }
    std::memset(uuid, 0, sizeof(*uuid));
    std::memcpy(uuid->bytes, &device,
                std::min(sizeof(device), sizeof(uuid->bytes)));
    return CUDA_SUCCESS;
}

CUresult xpu_memory_info(size_t *free_bytes, size_t *total_bytes) {
    auto *state = current_device();
    if (!free_bytes || !total_bytes || !state) {
        return kCudaErrorUnknown;
    }
    try {
        const sycl::device device = state->queue->get_device();
        *total_bytes = device.get_info<sycl::info::device::global_mem_size>();
        if (!device.has(sycl::aspect::ext_intel_free_memory)) {
            return kCudaErrorUnknown;
        }
        *free_bytes = static_cast<size_t>(
            device.get_info<sycl::ext::intel::info::device::free_memory>());
        return CUDA_SUCCESS;
    } catch (...) {
        return kCudaErrorUnknown;
    }
}

CUresult xpu_malloc(CUdeviceptr *pointer, size_t size) {
    auto *state = current_device();
    if (!pointer || !state) {
        return kCudaErrorUnknown;
    }
    try {
        void *allocation = sycl::malloc_device(size, *state->queue);
        if (!allocation) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        *pointer = reinterpret_cast<CUdeviceptr>(allocation);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult xpu_free(CUdeviceptr pointer) {
    auto *state = current_device();
    if (!state) {
        return kCudaErrorUnknown;
    }
    try {
        sycl::free(reinterpret_cast<void *>(pointer), state->queue->get_context());
        return CUDA_SUCCESS;
    } catch (...) {
        return kCudaErrorUnknown;
    }
}

CUresult xpu_malloc_async(CUdeviceptr *pointer, size_t size, CUstream stream) {
    sycl::queue *queue = resolve_queue(stream);
    if (!pointer || !queue) {
        return kCudaErrorUnknown;
    }
    try {
        void *allocation = sycl::malloc_device(size, *queue);
        if (!allocation) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        *pointer = reinterpret_cast<CUdeviceptr>(allocation);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult xpu_free_async(CUdeviceptr pointer, CUstream stream) {
    sycl::queue *queue = resolve_queue(stream);
    if (!queue) {
        return kCudaErrorUnknown;
    }
    try {
        queue->wait_and_throw();
        sycl::free(reinterpret_cast<void *>(pointer), queue->get_context());
        return CUDA_SUCCESS;
    } catch (...) {
        return kCudaErrorUnknown;
    }
}

CUresult xpu_host_alloc(void **pointer, size_t size) {
    if (!pointer) {
        return kCudaErrorUnknown;
    }
    *pointer = std::malloc(size);
    return *pointer ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult xpu_host_free(void *pointer) {
    std::free(pointer);
    return CUDA_SUCCESS;
}

CUresult xpu_host_register(void *, size_t, unsigned int) {
    // XPU phase 1 keeps ComfyUI host pinning disabled. Existing HostBuffer
    // registration calls remain valid no-ops and copies use the SYCL queue.
    return CUDA_SUCCESS;
}

CUresult xpu_host_unregister(void *) {
    return CUDA_SUCCESS;
}

CUresult xpu_virtual_reserve(CUdeviceptr *pointer, size_t size, size_t,
                             CUdeviceptr requested, unsigned long long) {
    auto *state = current_device();
    if (!pointer || !state) {
        return kCudaErrorUnknown;
    }
    void *address = reinterpret_cast<void *>(requested);
    const ze_result_t result = zeVirtualMemReserve(
        state->context, address, size, &address);
    if (result == ZE_RESULT_SUCCESS) {
        *pointer = reinterpret_cast<CUdeviceptr>(address);
        g_stats[kVirtualReserveCalls].fetch_add(1, std::memory_order_relaxed);
        g_stats[kVirtualReserveBytes].fetch_add(size, std::memory_order_relaxed);
    }
    return from_ze(result);
}

CUresult xpu_virtual_free(CUdeviceptr pointer, size_t size) {
    auto *state = current_device();
    return state
               ? from_ze(zeVirtualMemFree(
                     state->context, reinterpret_cast<void *>(pointer), size))
               : kCudaErrorUnknown;
}

CUresult xpu_physical_create(CUmemGenericAllocationHandle *handle, size_t size,
                             const CUmemAllocationProp *properties,
                             unsigned long long) {
    const int device_id = properties ? properties->location.id
                                     : aimdo_xpu_current_device();
    auto *state = find_device(device_id);
    if (!handle || !state) {
        return kCudaErrorUnknown;
    }

    size_t page_size = 0;
    ze_result_t result = zeVirtualMemQueryPageSize(
        state->context, state->device, size, &page_size);
    if (result != ZE_RESULT_SUCCESS || !page_size || size % page_size != 0) {
        return result == ZE_RESULT_SUCCESS ? kCudaErrorUnknown : from_ze(result);
    }

    ze_physical_mem_desc_t description{
        ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
        nullptr,
        ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE,
        size,
    };
    ze_physical_mem_handle_t physical = nullptr;
    result = zePhysicalMemCreate(
        state->context, state->device, &description, &physical);
    if (result == ZE_RESULT_SUCCESS) {
        *handle = static_cast<CUmemGenericAllocationHandle>(
            reinterpret_cast<uintptr_t>(physical));
        g_stats[kPhysicalCreateCalls].fetch_add(1, std::memory_order_relaxed);
        g_stats[kPhysicalCreateBytes].fetch_add(size, std::memory_order_relaxed);
    }
    return from_ze(result);
}

CUresult xpu_virtual_map(CUdeviceptr pointer, size_t size, size_t offset,
                         CUmemGenericAllocationHandle handle,
                         unsigned long long) {
    auto *state = current_device();
    if (!state) {
        return kCudaErrorUnknown;
    }
    const ze_result_t result = zeVirtualMemMap(
        state->context, reinterpret_cast<void *>(pointer), size,
        reinterpret_cast<ze_physical_mem_handle_t>(
            static_cast<uintptr_t>(handle)),
        offset, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    if (result == ZE_RESULT_SUCCESS) {
        g_stats[kMapCalls].fetch_add(1, std::memory_order_relaxed);
        g_stats[kMapBytes].fetch_add(size, std::memory_order_relaxed);
    }
    return from_ze(result);
}

CUresult xpu_virtual_set_access(CUdeviceptr, size_t,
                                const CUmemAccessDesc *, size_t) {
    // Access is selected atomically with zeVirtualMemMap above.
    return CUDA_SUCCESS;
}

CUresult xpu_virtual_unmap(CUdeviceptr pointer, size_t size) {
    auto *state = current_device();
    if (!state) {
        return kCudaErrorUnknown;
    }
    const ze_result_t result = zeVirtualMemUnmap(
        state->context, reinterpret_cast<void *>(pointer), size);
    if (result == ZE_RESULT_SUCCESS) {
        g_stats[kUnmapCalls].fetch_add(1, std::memory_order_relaxed);
        g_stats[kUnmapBytes].fetch_add(size, std::memory_order_relaxed);
    }
    return from_ze(result);
}

CUresult xpu_physical_release(CUmemGenericAllocationHandle handle) {
    auto *state = current_device();
    if (!state) {
        return kCudaErrorUnknown;
    }
    const ze_result_t result = zePhysicalMemDestroy(
        state->context,
        reinterpret_cast<ze_physical_mem_handle_t>(
            static_cast<uintptr_t>(handle)));
    if (result == ZE_RESULT_SUCCESS) {
        g_stats[kPhysicalReleaseCalls].fetch_add(1, std::memory_order_relaxed);
    }
    return from_ze(result);
}

CUresult xpu_memcpy_host_to_device(CUdeviceptr destination, const void *source,
                                   size_t size, CUstream stream) {
    sycl::queue *queue = resolve_queue(stream);
    if (!queue) {
        return kCudaErrorUnknown;
    }
    const uint64_t call = g_stats[kSynchronousHostToDeviceCalls].fetch_add(
                              1, std::memory_order_relaxed) +
                          1;
    trace_sync("h2d", "begin", call, queue, size);
    try {
        // XPU phase 1 uses ordinary malloc-backed host buffers rather than
        // pinned host allocations. Keep their lifetime unambiguous across
        // ComfyUI workflow boundaries: complete the copy before the common
        // file-reader ring can retire or reuse its slot.
        queue->memcpy(reinterpret_cast<void *>(destination), source, size)
            .wait_and_throw();
        g_stats[kHostToDeviceBytes].fetch_add(size, std::memory_order_relaxed);
        g_stats[kSynchronousHostToDeviceCompletions].fetch_add(
            1, std::memory_order_relaxed);
        trace_sync("h2d", "end", call, queue, size);
        return CUDA_SUCCESS;
    } catch (...) {
        trace_sync("h2d", "error", call, queue, size);
        return kCudaErrorUnknown;
    }
}

CUresult xpu_event_create(CUevent *event, unsigned int) {
    if (!event) {
        return kCudaErrorUnknown;
    }
    try {
        *event = new CUevent_st();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult xpu_event_destroy(CUevent event) {
    delete event;
    return CUDA_SUCCESS;
}

CUresult xpu_event_record(CUevent event, CUstream stream) {
    sycl::queue *queue = resolve_queue(stream);
    if (!event || !queue) {
        return kCudaErrorUnknown;
    }
    // All XPU host-to-device copies are completed synchronously above. The
    // CUDA-shaped event retained by the common file-reader is therefore an
    // already-completed token; submitting a long-lived SYCL barrier here can
    // retain a worker queue beyond one ComfyUI workflow.
    event->recorded = true;
    return CUDA_SUCCESS;
}

CUresult xpu_event_synchronize(CUevent event) {
    if (!event || !event->recorded) {
        return kCudaErrorUnknown;
    }
    const uint64_t call =
        g_stats[kEventSyncCalls].fetch_add(1, std::memory_order_relaxed) + 1;
    trace_sync("event", "begin", call, nullptr);
    g_stats[kEventSyncCompletions].fetch_add(
        1, std::memory_order_relaxed);
    trace_sync("event", "end", call, nullptr);
    return CUDA_SUCCESS;
}

bool event_is_complete(const sycl::event &event) {
    try {
        return event
                   .get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (...) {
        return false;
    }
}

bool release_torch_block(XpuTorchBlock &block, bool wait) {
    try {
        if (block.has_ready) {
            if (!wait && !event_is_complete(block.ready)) {
                return false;
            }
            block.ready.wait_and_throw();
        }
        sycl::free(block.pointer, block.queue.get_context());
        aimdo_xpu_account_allocation(
            block.device, -static_cast<int64_t>(block.size));
        decrease_torch_bytes(
            g_torch_reserved_bytes, block.device, block.size);
        g_stats[kTorchAllocatorPhysicalReleaseCalls].fetch_add(
            1, std::memory_order_relaxed);
        g_stats[kTorchAllocatorPhysicalReleaseBytes].fetch_add(
            block.size, std::memory_order_relaxed);
        return true;
    } catch (...) {
        return false;
    }
}

void release_cached_torch_blocks(int device, bool wait) {
    for (auto block = g_torch_cached_blocks.begin();
         block != g_torch_cached_blocks.end();) {
        if ((*block)->device == device && release_torch_block(**block, wait)) {
            block = g_torch_cached_blocks.erase(block);
        } else {
            ++block;
        }
    }
}

void *allocate_torch_block(size_t size, int device, sycl::queue *queue) {
    if (!queue || size == 0) {
        return nullptr;
    }
    resolve_queue(reinterpret_cast<CUstream>(queue));
    g_stats[kTorchAllocatorAllocCalls].fetch_add(
        1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    for (auto block = g_torch_cached_blocks.begin();
         block != g_torch_cached_blocks.end(); ++block) {
        if ((*block)->device != device || (*block)->size != size) {
            continue;
        }
        try {
            if ((*block)->has_ready && (*block)->queue != *queue) {
                (*block)->ready.wait_and_throw();
            }
            (*block)->queue = *queue;
            (*block)->has_ready = false;
            void *pointer = (*block)->pointer;
            g_torch_live_blocks.emplace(pointer, std::move(*block));
            g_torch_cached_blocks.erase(block);
            increase_torch_bytes(
                g_torch_active_bytes, g_torch_peak_active_bytes,
                device, size);
            g_stats[kTorchAllocatorCacheHits].fetch_add(
                1, std::memory_order_relaxed);
            return pointer;
        } catch (...) {
            return nullptr;
        }
    }

    // Retain exact-size blocks for the next iteration, but return completed
    // blocks of other sizes to Level Zero before physically growing the pool.
    release_cached_torch_blocks(device, false);
    if (!aimdo_xpu_prepare_allocation(device, size)) {
        return nullptr;
    }

    void *pointer = nullptr;
    try {
        pointer = sycl::malloc_device(size, *queue);
    } catch (...) {
        pointer = nullptr;
    }
    if (!pointer) {
        // A pending cached block is safer to wait here than letting the XPU
        // runtime enter an asynchronous OOM/device-lost state. Retry once
        // after returning every idle block and evicting another request-size
        // tranche of unpinned model pages.
        release_cached_torch_blocks(device, true);
        if (!aimdo_xpu_retry_allocation(device, size)) {
            return nullptr;
        }
        try {
            pointer = sycl::malloc_device(size, *queue);
        } catch (...) {
            pointer = nullptr;
        }
    }
    if (!pointer) {
        return nullptr;
    }

    try {
        auto block = std::make_unique<XpuTorchBlock>(
            pointer, size, device, *queue);
        g_torch_live_blocks.emplace(pointer, std::move(block));
    } catch (...) {
        sycl::free(pointer, queue->get_context());
        return nullptr;
    }
    aimdo_xpu_account_allocation(device, static_cast<int64_t>(size));
    increase_torch_bytes(
        g_torch_active_bytes, g_torch_peak_active_bytes, device, size);
    increase_torch_bytes(
        g_torch_reserved_bytes, g_torch_peak_reserved_bytes, device, size);
    g_stats[kTorchAllocatorPhysicalAllocCalls].fetch_add(
        1, std::memory_order_relaxed);
    g_stats[kTorchAllocatorPhysicalAllocBytes].fetch_add(
        size, std::memory_order_relaxed);
    return pointer;
}

void free_torch_block(void *pointer, sycl::queue *queue) {
    if (!pointer) {
        return;
    }
    g_stats[kTorchAllocatorFreeCalls].fetch_add(
        1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    auto found = g_torch_live_blocks.find(pointer);
    if (found == g_torch_live_blocks.end()) {
        return;
    }
    try {
        if (queue) {
            found->second->queue = *queue;
        }
        found->second->ready =
            found->second->queue.ext_oneapi_submit_barrier();
        found->second->has_ready = true;
        const int device = found->second->device;
        const size_t size = found->second->size;
        g_torch_cached_blocks.emplace_back(std::move(found->second));
        decrease_torch_bytes(g_torch_active_bytes, device, size);
        g_torch_live_blocks.erase(found);
    } catch (...) {
        // Keep the block live rather than freeing storage that may still be in
        // use by an asynchronous XPU command.
    }
}

CUresult xpu_device_get_luid(char *, unsigned int *, CUdevice) {
    return kCudaErrorUnknown;
}

}  // namespace

extern "C" {

AimdoCudaDispatch g_cuda{};
PFN_deviceGetProperties g_device_get_properties = nullptr;

__attribute__((visibility("default"))) void *xpu_alloc_fn(
    size_t size, int device, sycl::queue *queue) {
    return allocate_torch_block(size, device, queue);
}

__attribute__((visibility("default"))) void xpu_free_fn(
    void *pointer, size_t, int, sycl::queue *queue) {
    free_torch_block(pointer, queue);
}

__attribute__((visibility("default"))) bool xpu_allocator_empty_cache(
    bool wait) {
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    for (auto block = g_torch_cached_blocks.begin();
         block != g_torch_cached_blocks.end();) {
        if (release_torch_block(**block, wait)) {
            block = g_torch_cached_blocks.erase(block);
        } else {
            ++block;
        }
    }
    return g_torch_cached_blocks.empty();
}

__attribute__((visibility("default"))) bool xpu_allocator_get_memory_stats(
    int device, uint64_t *values, size_t count) {
    if (!values || count < 4) {
        return false;
    }
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    values[0] = g_torch_active_bytes[device];
    values[1] = g_torch_reserved_bytes[device];
    values[2] = g_torch_peak_active_bytes[device];
    values[3] = g_torch_peak_reserved_bytes[device];
    return true;
}

__attribute__((visibility("default"))) void xpu_allocator_reset_peak_stats(
    int device) {
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    g_torch_peak_active_bytes[device] = g_torch_active_bytes[device];
    g_torch_peak_reserved_bytes[device] = g_torch_reserved_bytes[device];
}

__attribute__((visibility("default"))) bool xpu_set_queues(
    const int *device_ids, const uint64_t *queue_pointers, size_t count) {
    if (!device_ids || !queue_pointers || count == 0) {
        return false;
    }
    try {
        std::lock_guard<std::mutex> guard(g_devices_mutex);
        g_devices.clear();
        for (auto &stat : g_stats) {
            stat.store(0, std::memory_order_relaxed);
        }
        g_devices.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (!queue_pointers[i] || find_device(device_ids[i])) {
                g_devices.clear();
                return false;
            }
            auto *queue = reinterpret_cast<sycl::queue *>(queue_pointers[i]);
            if (queue->get_backend() != sycl::backend::ext_oneapi_level_zero) {
                g_devices.clear();
                return false;
            }
            auto context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                queue->get_context());
            auto device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                queue->get_device());
            size_t page_size = 0;
            if (zeVirtualMemQueryPageSize(
                    context, device, 32ULL << 20, &page_size) !=
                    ZE_RESULT_SUCCESS ||
                !page_size || ((32ULL << 20) % page_size) != 0 ||
                ((2ULL << 20) % page_size) != 0) {
                g_devices.clear();
                return false;
            }
            g_devices.push_back(
                XpuDeviceState{device_ids[i], queue, context, device});
        }
        return true;
    } catch (...) {
        g_devices.clear();
        return false;
    }
}

__attribute__((visibility("default"))) bool xpu_get_vmm_stats(
    uint64_t *values, size_t count) {
    if (!values || count < kXpuStatCount) {
        return false;
    }
    for (size_t i = 0; i < kXpuStatCount; ++i) {
        values[i] = g_stats[i].load(std::memory_order_relaxed);
    }
    return true;
}

bool aimdo_cuda_runtime_init(void) {
    std::lock_guard<std::mutex> guard(g_devices_mutex);
    if (g_devices.empty() || zeInit(0) != ZE_RESULT_SUCCESS) {
        return false;
    }
    g_cuda.p_cuInit = xpu_init;
    g_cuda.p_cuGetErrorString = xpu_get_error_string;
    g_cuda.p_cuCtxGetDevice = xpu_context_get_device;
    g_cuda.p_cuCtxSynchronize = xpu_context_synchronize;
    g_cuda.p_cuDeviceGet = xpu_device_get;
    g_cuda.p_cuDeviceGetAttribute = xpu_device_get_attribute;
    g_cuda.p_cuDeviceTotalMem = xpu_device_total_memory;
    g_cuda.p_cuDeviceGetName = xpu_device_get_name;
    g_cuda.p_cuDeviceGetUuid = xpu_device_get_uuid;
    g_cuda.p_cuMemGetInfo = xpu_memory_info;
    g_cuda.p_cuMemAlloc_v2 = xpu_malloc;
    g_cuda.p_cuMemFree_v2 = xpu_free;
    g_cuda.p_cuMemAllocAsync = xpu_malloc_async;
    g_cuda.p_cuMemAllocAsync_ptsz = xpu_malloc_async;
    g_cuda.p_cuMemFreeAsync = xpu_free_async;
    g_cuda.p_cuMemFreeAsync_ptsz = xpu_free_async;
    g_cuda.p_cuMemAllocHost = xpu_host_alloc;
    g_cuda.p_cuMemFreeHost = xpu_host_free;
    g_cuda.p_cuMemHostRegister = xpu_host_register;
    g_cuda.p_cuMemHostUnregister = xpu_host_unregister;
    g_cuda.p_cuMemAddressReserve = xpu_virtual_reserve;
    g_cuda.p_cuMemAddressFree = xpu_virtual_free;
    g_cuda.p_cuMemCreate = xpu_physical_create;
    g_cuda.p_cuMemMap = xpu_virtual_map;
    g_cuda.p_cuMemSetAccess = xpu_virtual_set_access;
    g_cuda.p_cuMemUnmap = xpu_virtual_unmap;
    g_cuda.p_cuMemRelease = xpu_physical_release;
    g_cuda.p_cuMemcpyHtoDAsync = xpu_memcpy_host_to_device;
    g_cuda.p_cuEventCreate = xpu_event_create;
    g_cuda.p_cuEventDestroy = xpu_event_destroy;
    g_cuda.p_cuEventRecord = xpu_event_record;
    g_cuda.p_cuEventSynchronize = xpu_event_synchronize;
    g_cuda.p_cuDeviceGetLuid = xpu_device_get_luid;
    return true;
}

void aimdo_cuda_runtime_cleanup(void) {
    std::memset(&g_cuda, 0, sizeof(g_cuda));
}

}  // extern "C"
