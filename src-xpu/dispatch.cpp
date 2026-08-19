extern "C" {
#include "gpu_dispatch.h"
}

#if __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#elif __has_include(<ze_api.h>)
#include <ze_api.h>
#else
#error "Level Zero headers were not found"
#endif
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" bool aimdo_xpu_prepare_allocation(int device, size_t size);
extern "C" bool aimdo_xpu_retry_allocation(int device, size_t size);
extern "C" bool aimdo_xpu_account_allocation(int device, int64_t delta);
extern "C" int aimdo_vbar_describe_range(uint64_t address, uint64_t size, int *mapped, unsigned *pin, uint64_t *page_index, uint64_t *unmapped_page, uint64_t *pages_spanned);

#if defined(_WIN32) || defined(_WIN64)
#define AIMDO_XPU_EXPORT __declspec(dllexport)
#else
#define AIMDO_XPU_EXPORT __attribute__((visibility("default")))
#endif

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
    kHostToDeviceSplitRetries,
    kTorchAllocatorAllocCalls,
    kTorchAllocatorFreeCalls,
    kTorchAllocatorCacheHits,
    kTorchAllocatorPhysicalAllocCalls,
    kTorchAllocatorPhysicalAllocBytes,
    kTorchAllocatorPhysicalReleaseCalls,
    kTorchAllocatorPhysicalReleaseBytes,
    kSmallVbarCopyFallbackCalls,
    kSmallVbarCopyFallbackBytes,
    kSmallVbarCopyFallbackFailures,
    kRetireTokenCalls,
    kRetireFenceSubmitCalls,
    kRetireFenceCompleteCalls,
    kRetireFenceSubmitFailures,
    kRetireForcePolls,
    kRetireTrackedQueues,
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

// VBAR retirement follows the same ownership rule as a caching allocator's
// recordStream(): a page remembers only the queues that actually consumed it,
// and each queue advances independently.  This avoids the old process-global
// minimum epoch, where one unrelated/hidden queue both delayed every page and
// forced a barrier onto every queue for every completion interval.
std::mutex g_retire_mutex;

// Fences are deliberately batched.  Submitting one barrier per weight caused
// thousands of live Level Zero events during long H3 cycles.  Sixty-four uses
// amortizes submission while the force poll at real pressure closes a partial
// batch, so reclaim does not depend on reaching the batch size.
constexpr size_t kRetireFenceSlots = 8;
constexpr size_t kRetireBatchUses = 64;

// The token format is shared with model-vbar.c through plat.h.  Queue tags are
// one based so token zero remains the fail-closed "unknown queue" value.
constexpr size_t kMaxTrackedQueues = AIMDO_XPU_RETIRE_MAX_QUEUES;

struct RetireFence {
    sycl::event event;
    uint64_t generation = 0;
    bool valid = false;
};

struct RetireQueue {
    sycl::queue *queue = nullptr;
    uint64_t open_generation = 1;
    uint64_t retired_generation = 0;
    size_t pending_uses = 0;
    RetireFence fences[kRetireFenceSlots];
};

RetireQueue g_retire_queues[kMaxTrackedQueues];
size_t g_retire_queue_count = 0;
std::atomic<bool> g_retire_tracking_overflow{false};

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

bool proactive_cache_release_enabled() {
#if defined(_WIN32) || defined(_WIN64)
    static const bool enabled = [] {
        const char *value = std::getenv("AIMDO_XPU_EAGER_USM_RELEASE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
#else
    return true;
#endif
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
void aimdo_xpu_note_queue(sycl::queue *queue);

XpuDeviceState *find_device(int id) {
    auto found = std::find_if(
        g_devices.begin(), g_devices.end(),
        [id](const XpuDeviceState &state) { return state.id == id; });
    return found == g_devices.end() ? nullptr : &*found;
}

XpuDeviceState *current_device() {
    return find_device(aimdo_xpu_current_device());
}

int device_from_native_handle(uintptr_t native_handle) {
    std::lock_guard<std::mutex> guard(g_devices_mutex);
    auto found = std::find_if(
        g_devices.begin(), g_devices.end(),
        [native_handle](const XpuDeviceState &state) {
            return reinterpret_cast<uintptr_t>(state.device) == native_handle;
        });
    return found == g_devices.end() ? -1 : found->id;
}

size_t aimdo_xpu_note_queue_locked(sycl::queue *queue) {
    if (!queue) {
        return kMaxTrackedQueues;
    }
    for (size_t index = 0; index < g_retire_queue_count; ++index) {
        if (g_retire_queues[index].queue == queue) {
            return index;
        }
    }
    if (g_retire_queue_count >= kMaxTrackedQueues) {
        // Cannot prove retirement on an untracked queue, so stop advancing
        // pages that name it rather than guessing from a different queue.
        if (!g_retire_tracking_overflow.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                         "[AIMDO XPU] tracked queue table full (%zu); VBAR "
                         "pages on additional queues are non-reclaimable\n",
                         kMaxTrackedQueues);
            std::fflush(stderr);
        }
        return kMaxTrackedQueues;
    }
    const size_t index = g_retire_queue_count++;
    g_retire_queues[index].queue = queue;
    g_stats[kRetireTrackedQueues].store(
        g_retire_queue_count, std::memory_order_relaxed);
    return index;
}

/* Register a queue AIMDO has seen.  Page retirement itself is scoped to the
 * queue token returned at unpin, not to this process-wide registry. */
void aimdo_xpu_note_queue(sycl::queue *queue) {
    std::lock_guard<std::mutex> guard(g_retire_mutex);
    (void)aimdo_xpu_note_queue_locked(queue);
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
        aimdo_xpu_note_queue(queue);
        return queue;
    }
    auto *state = current_device();
    if (state && state->queue) {
        aimdo_xpu_note_queue(state->queue);
    }
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
        // Callers use this as "all device work has completed" before unmapping
        // VBAR pages. Torch hands out many queues per device, so waiting only
        // the current one can release a page still running elsewhere. Wait
        // every queue AIMDO has seen.
        state->queue->wait_and_throw();
        {
            std::lock_guard<std::mutex> guard(g_retire_mutex);
            for (size_t index = 0; index < g_retire_queue_count; ++index) {
                sycl::queue *queue = g_retire_queues[index].queue;
                if (queue && queue != state->queue) {
                    queue->wait_and_throw();
                }
            }
        }
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
    /* Reverted: backing this with sycl::malloc_host made the failure worse,
     * turning a 64 MiB copy's OUT_OF_DEVICE_MEMORY into DEVICE_LOST and
     * moving it 56 seconds earlier. Keep pageable staging until the copy
     * failure itself is understood. */
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
    } catch (const sycl::exception &error) {
        const std::error_code code = error.code();
        /* A large copy from pageable host memory makes the runtime stage the
         * transfer itself, and that internal work is not covered by the WDDM
         * budget: 64 MiB copies failed with OUT_OF_DEVICE_MEMORY while both
         * DXGI and the driver still reported ~4 GiB free. Splitting the
         * transfer shrinks the staging requirement, so retry once in smaller
         * pieces before giving up. */
        constexpr size_t kSplitChunk = 8ULL * 1024 * 1024;

        if (size > kSplitChunk) {
            try {
                const auto *bytes = static_cast<const unsigned char *>(source);
                for (size_t done = 0; done < size; done += kSplitChunk) {
                    const size_t piece = std::min(kSplitChunk, size - done);
                    queue->memcpy(
                        reinterpret_cast<void *>(destination + done),
                        bytes + done, piece).wait_and_throw();
                }
                g_stats[kHostToDeviceBytes].fetch_add(
                    size, std::memory_order_relaxed);
                g_stats[kSynchronousHostToDeviceCompletions].fetch_add(
                    1, std::memory_order_relaxed);
                g_stats[kHostToDeviceSplitRetries].fetch_add(
                    1, std::memory_order_relaxed);
                trace_sync("h2d", "end_split", call, queue, size);
                return CUDA_SUCCESS;
            } catch (...) {
                // Fall through and report the original failure.
            }
        }
        /* Report what the driver itself believes at the moment of failure.
         * DXGI Budget said roughly 4 GiB was still available when a 64 MiB
         * copy reported OUT_OF_DEVICE_MEMORY, so the two accountings disagree
         * and only the driver's own view can settle it. The pointer type
         * distinguishes a VBAR VMM mapping (not USM, reported as unknown)
         * from a Torch USM device allocation. */
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const char *destination_kind = "unavailable";
        int vbar_hit = 0;
        int vbar_mapped = 0;
        unsigned vbar_pin = 0;
        uint64_t vbar_page = 0;
        uint64_t vbar_unmapped = 0;
        uint64_t vbar_span = 0;

        vbar_hit = aimdo_vbar_describe_range(
            static_cast<uint64_t>(destination), size, &vbar_mapped, &vbar_pin,
            &vbar_page, &vbar_unmapped, &vbar_span);
        try {
            const sycl::device device = queue->get_device();
            if (device.has(sycl::aspect::ext_intel_free_memory)) {
                free_bytes = static_cast<size_t>(
                    device.get_info<
                        sycl::ext::intel::info::device::free_memory>());
            }
            total_bytes =
                device.get_info<sycl::info::device::global_mem_size>();
            switch (sycl::get_pointer_type(
                        reinterpret_cast<void *>(destination),
                        queue->get_context())) {
            case sycl::usm::alloc::device: destination_kind = "usm_device"; break;
            case sycl::usm::alloc::host:   destination_kind = "usm_host"; break;
            case sycl::usm::alloc::shared: destination_kind = "usm_shared"; break;
            default:                       destination_kind = "not_usm_vmm"; break;
            }
        } catch (...) {
        }
        std::fprintf(
            stderr,
            "[AIMDO XPU ERROR] op=h2d queue=%p destination=%p size=%zu "
            "sycl_code=%d category=%s driver_free=%zu driver_total=%zu "
            "dest_kind=%s vbar=%d all_mapped=%d min_pin=%u first_page=%llu "
            "span=%llu first_unmapped=%lld message=%s\n",
            static_cast<void *>(queue), reinterpret_cast<void *>(destination),
            size, code.value(), code.category().name(),
            free_bytes, total_bytes, destination_kind, vbar_hit, vbar_mapped,
            vbar_pin, static_cast<unsigned long long>(vbar_page),
            static_cast<unsigned long long>(vbar_span),
            vbar_unmapped == UINT64_MAX ? -1LL
                                        : static_cast<long long>(vbar_unmapped),
            error.what());
        std::fflush(stderr);
        trace_sync("h2d", "error", call, queue, size);
        return kCudaErrorUnknown;
    } catch (const std::exception &error) {
        std::fprintf(
            stderr,
            "[AIMDO XPU ERROR] op=h2d queue=%p destination=%p size=%zu "
            "exception=%s\n",
            static_cast<void *>(queue), reinterpret_cast<void *>(destination),
            size, error.what());
        std::fflush(stderr);
        trace_sync("h2d", "error", call, queue, size);
        return kCudaErrorUnknown;
    } catch (...) {
        std::fprintf(
            stderr,
            "[AIMDO XPU ERROR] op=h2d queue=%p destination=%p size=%zu "
            "exception=<non-standard>\n",
            static_cast<void *>(queue), reinterpret_cast<void *>(destination),
            size);
        std::fflush(stderr);
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
    const uint64_t call =
        g_stats[kTorchAllocatorPhysicalReleaseCalls].load(
            std::memory_order_relaxed) + 1;
    trace_sync("torch_usm_free", "begin", call, &block.queue, block.size);
    try {
        if (block.has_ready) {
            if (!wait && !event_is_complete(block.ready)) {
                trace_sync("torch_usm_free", "deferred", call, &block.queue,
                           block.size);
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
        trace_sync("torch_usm_free", "end", call, &block.queue, block.size);
        return true;
    } catch (...) {
        trace_sync("torch_usm_free", "error", call, &block.queue, block.size);
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

    // Retain exact-size blocks for the next iteration. Linux also returns
    // completed blocks of other sizes before physically growing the pool.
    // On Windows, sycl::free can wait indefinitely in the Level Zero/UMF
    // residency path under high WDDM usage. Keep it out of the allocator hot
    // path by default; explicit empty_cache and the allocation retry path
    // remain able to release cached storage.
    if (proactive_cache_release_enabled()) {
        release_cached_torch_blocks(device, false);
    }
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
        // after returning every idle block and publishing another request-size
        // tranche for owner-side VBAR reclaim.
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

void *allocate_raw_torch_segment(
    size_t size, int device, sycl::queue *queue) {
    if (!queue || size == 0) {
        return nullptr;
    }
    resolve_queue(reinterpret_cast<CUstream>(queue));
    g_stats[kTorchAllocatorAllocCalls].fetch_add(
        1, std::memory_order_relaxed);

    // The native XPU caching allocator owns block reuse, stream events,
    // splitting, and coalescing in this mode. AIMDO serializes only physical
    // segment growth and VBAR pressure; it must not retain a second cache.
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
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

void free_raw_torch_segment(
    void *pointer, size_t size, int device, sycl::queue *queue) {
    if (!pointer || !queue) {
        return;
    }
    g_stats[kTorchAllocatorFreeCalls].fetch_add(
        1, std::memory_order_relaxed);

    // XPUCachingAllocator calls raw_delete only after the segment is no longer
    // active and all recorded stream uses are safe. Free the physical segment
    // directly instead of moving it into AIMDO's legacy exact-size cache.
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    try {
        sycl::free(pointer, queue->get_context());
    } catch (...) {
        return;
    }
    aimdo_xpu_account_allocation(device, -static_cast<int64_t>(size));
    decrease_torch_bytes(g_torch_active_bytes, device, size);
    decrease_torch_bytes(g_torch_reserved_bytes, device, size);
    g_stats[kTorchAllocatorPhysicalReleaseCalls].fetch_add(
        1, std::memory_order_relaxed);
    g_stats[kTorchAllocatorPhysicalReleaseBytes].fetch_add(
        size, std::memory_order_relaxed);
}

CUresult xpu_device_get_luid(char *luid, unsigned int *node_mask,
                             CUdevice device) {
#if defined(_WIN32) || defined(_WIN64)
    auto *state = find_device(device);
    if (!luid || !node_mask || !state) {
        return kCudaErrorUnknown;
    }
    try {
        const sycl::device sycl_device = state->queue->get_device();
        if (!sycl_device.has(sycl::aspect::ext_intel_device_info_luid) ||
            !sycl_device.has(
                sycl::aspect::ext_intel_device_info_node_mask)) {
            return kCudaErrorUnknown;
        }
        const auto luid_value = sycl_device.get_info<
            sycl::ext::intel::info::device::luid>();
        const uint32_t node_mask_value = sycl_device.get_info<
            sycl::ext::intel::info::device::node_mask>();
        constexpr size_t kWindowsLuidSize = 8;
        if (luid_value.size() != kWindowsLuidSize ||
            node_mask_value == 0 ||
            (node_mask_value & (node_mask_value - 1)) != 0) {
            return kCudaErrorUnknown;
        }
        std::memcpy(luid, luid_value.data(), kWindowsLuidSize);
        *node_mask = node_mask_value;
        return CUDA_SUCCESS;
    } catch (...) {
        return kCudaErrorUnknown;
    }
#else
    (void)luid;
    (void)node_mask;
    (void)device;
    return kCudaErrorUnknown;
#endif
}

void poll_retire_queue_locked(RetireQueue &retire_queue) {
    for (size_t slot = 0; slot < kRetireFenceSlots; ++slot) {
        RetireFence &fence = retire_queue.fences[slot];
        if (!fence.valid || !event_is_complete(fence.event)) {
            continue;
        }
        retire_queue.retired_generation = std::max(
            retire_queue.retired_generation, fence.generation);
        fence.valid = false;
        fence.event = sycl::event();
        fence.generation = 0;
        g_stats[kRetireFenceCompleteCalls].fetch_add(
            1, std::memory_order_relaxed);
    }
}

bool submit_retire_fence_locked(RetireQueue &retire_queue) {
    if (!retire_queue.queue || retire_queue.pending_uses == 0) {
        return true;
    }

    size_t free_slot = kRetireFenceSlots;
    for (size_t slot = 0; slot < kRetireFenceSlots; ++slot) {
        if (!retire_queue.fences[slot].valid) {
            free_slot = slot;
            break;
        }
    }
    if (free_slot == kRetireFenceSlots) {
        return false;
    }

    // Hold g_retire_mutex while submitting and advance the generation only
    // after the barrier exists.  A concurrent unpin can therefore receive
    // either the generation ordered by this barrier or the next generation,
    // never a token whose fence was already submitted before its operator.
    try {
        sycl::event event =
            retire_queue.queue->ext_oneapi_submit_barrier();
        RetireFence &fence = retire_queue.fences[free_slot];
        fence.event = std::move(event);
        fence.generation = retire_queue.open_generation;
        fence.valid = true;
        retire_queue.open_generation++;
        retire_queue.pending_uses = 0;
        g_stats[kRetireFenceSubmitCalls].fetch_add(
            1, std::memory_order_relaxed);
        return true;
    } catch (...) {
        // Leave the generation open and all pages carrying it
        // non-reclaimable. A later pressure poll may retry safely.
        g_stats[kRetireFenceSubmitFailures].fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
}

}  // namespace

extern "C" {

AimdoCudaDispatch g_cuda{};
PFN_deviceGetProperties g_device_get_properties = nullptr;

int aimdo_xpu_device_index_from_native(void *native_device) {
    std::lock_guard<std::mutex> guard(g_devices_mutex);
    for (const auto &state : g_devices) {
        if (reinterpret_cast<void *>(state.device) == native_device) {
            return state.id;
        }
    }
    return -1;
}

void aimdo_xpu_record_native_allocation(size_t size) {
    g_stats[kTorchAllocatorPhysicalAllocCalls].fetch_add(
        1, std::memory_order_relaxed);
    g_stats[kTorchAllocatorPhysicalAllocBytes].fetch_add(
        size, std::memory_order_relaxed);
}

void aimdo_xpu_record_native_release(size_t size) {
    g_stats[kTorchAllocatorPhysicalReleaseCalls].fetch_add(
        1, std::memory_order_relaxed);
    g_stats[kTorchAllocatorPhysicalReleaseBytes].fetch_add(
        size, std::memory_order_relaxed);
}

/* Return a token for the actual queue that consumed a VBAR page.  Tokens from
 * the same queue share one generation until a bounded batch is closed. */
uint64_t aimdo_xpu_retire_token_current(void *queue_pointer) {
    auto *queue = reinterpret_cast<sycl::queue *>(queue_pointer);
    if (!queue) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(g_retire_mutex);
    const size_t index = aimdo_xpu_note_queue_locked(queue);
    if (index >= kMaxTrackedQueues) {
        return 0;
    }

    RetireQueue &retire_queue = g_retire_queues[index];
    poll_retire_queue_locked(retire_queue);
    const uint64_t token =
        (retire_queue.open_generation <<
         AIMDO_XPU_RETIRE_TOKEN_QUEUE_BITS) |
        (index + 1);
    if (retire_queue.pending_uses < kRetireBatchUses) {
        retire_queue.pending_uses++;
    }
    g_stats[kRetireTokenCalls].fetch_add(1, std::memory_order_relaxed);
    if (retire_queue.pending_uses >= kRetireBatchUses) {
        (void)submit_retire_fence_locked(retire_queue);
    }
    return token;
}

/* Snapshot independently completed generations.  Normal unpin batching never
 * waits. A force poll is used only by an owner-side pressure scan and closes
 * partial batches so a later retry can reclaim them. */
size_t aimdo_xpu_retire_snapshot(uint64_t *completed, size_t count,
                                 bool force_submit) {
    if (!completed || count == 0) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(g_retire_mutex);
    if (force_submit) {
        g_stats[kRetireForcePolls].fetch_add(1, std::memory_order_relaxed);
    }
    const size_t copied = std::min(count, g_retire_queue_count);
    for (size_t index = 0; index < g_retire_queue_count; ++index) {
        RetireQueue &retire_queue = g_retire_queues[index];
        poll_retire_queue_locked(retire_queue);
        if (force_submit && retire_queue.pending_uses) {
            (void)submit_retire_fence_locked(retire_queue);
        }
        if (index < copied) {
            completed[index] = retire_queue.retired_generation;
        }
    }
    for (size_t index = copied; index < count; ++index) {
        completed[index] = 0;
    }
    return copied;
}

/* Drop every outstanding fence before the SYCL context is torn down. */
void aimdo_xpu_retire_reset(void) {
    std::lock_guard<std::mutex> guard(g_retire_mutex);
    for (size_t index = 0; index < kMaxTrackedQueues; ++index) {
        RetireQueue &retire_queue = g_retire_queues[index];
        for (size_t slot = 0; slot < kRetireFenceSlots; ++slot) {
            retire_queue.fences[slot].valid = false;
            retire_queue.fences[slot].event = sycl::event();
            retire_queue.fences[slot].generation = 0;
        }
        retire_queue.queue = nullptr;
        retire_queue.open_generation = 1;
        retire_queue.retired_generation = 0;
        retire_queue.pending_uses = 0;
    }
    g_retire_queue_count = 0;
    g_retire_tracking_overflow.store(false, std::memory_order_relaxed);
    g_stats[kRetireTrackedQueues].store(0, std::memory_order_relaxed);
}

AIMDO_XPU_EXPORT void *xpu_alloc_fn(
    size_t size, int device, sycl::queue *queue) {
    return allocate_torch_block(size, device, queue);
}

AIMDO_XPU_EXPORT void xpu_free_fn(
    void *pointer, size_t, int, sycl::queue *queue) {
    free_torch_block(pointer, queue);
}

AIMDO_XPU_EXPORT void *xpu_raw_alloc_fn(
    size_t size, int device, sycl::queue *queue) {
    return allocate_raw_torch_segment(size, device, queue);
}

AIMDO_XPU_EXPORT void xpu_raw_free_fn(
    void *pointer, size_t size, int device, sycl::queue *queue) {
    free_raw_torch_segment(pointer, size, device, queue);
}

AIMDO_XPU_EXPORT bool xpu_allocator_empty_cache(
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

AIMDO_XPU_EXPORT bool xpu_allocator_get_memory_stats(
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

AIMDO_XPU_EXPORT void xpu_allocator_reset_peak_stats(
    int device) {
    std::lock_guard<std::mutex> guard(g_torch_allocator_mutex);
    g_torch_peak_active_bytes[device] = g_torch_active_bytes[device];
    g_torch_peak_reserved_bytes[device] = g_torch_reserved_bytes[device];
}

AIMDO_XPU_EXPORT bool xpu_set_queues(
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

/* Work around a Windows driver failure on the second B70 adapter.
 *
 * A host-to-VMM copy of 2 MiB or less is rejected with
 * ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY even when the destination is mapped,
 * pinned and more than 31 GiB is free. The same transfer at 2 MiB + 1 byte
 * succeeds, and a kernel can write the VMM mapping. Pad only the transfer into
 * a private device staging allocation, then have a kernel write exactly the
 * requested destination range. Unlike padding the destination copy, this
 * cannot overwrite an adjacent VBAR allocation.
 *
 * Call this above the failing torch/UR copy, after the source is ready and the
 * destination range has been faulted and pinned. The operation is synchronous
 * so the temporary allocations and source can be released on return. */
extern "C" AIMDO_XPU_EXPORT bool aimdo_xpu_copy_host_to_vbar(
    void *destination, const void *source, size_t size, int device) {
    constexpr size_t kBrokenCopyMaximum = 2ULL * 1024 * 1024;
    constexpr size_t kSafeStagingSize = kBrokenCopyMaximum + 1;
    auto *state = find_device(device);
    unsigned char *host_staging = nullptr;
    unsigned char *device_staging = nullptr;
    bool copied = false;
    const char *stage = "validate";

    if (!state || !destination || !source || size == 0 ||
        size > kBrokenCopyMaximum) {
        return false;
    }
    g_stats[kSmallVbarCopyFallbackCalls].fetch_add(
        1, std::memory_order_relaxed);
    try {
        stage = "host_alloc";
        host_staging = sycl::malloc_host<unsigned char>(
            kSafeStagingSize, *state->queue);
        stage = "device_alloc";
        device_staging = sycl::malloc_device<unsigned char>(
            kSafeStagingSize, *state->queue);
        if (!host_staging || !device_staging) {
            throw std::bad_alloc();
        }

        std::memset(host_staging, 0, kSafeStagingSize);
        std::memcpy(host_staging, source, size);
        stage = "padded_h2d";
        state->queue->memcpy(device_staging, host_staging,
                             kSafeStagingSize).wait_and_throw();

        auto *bytes_out = static_cast<unsigned char *>(destination);
        const auto *bytes_in = device_staging;
        stage = "exact_kernel";
        state->queue->parallel_for(
            sycl::range<1>(size),
            [=](sycl::id<1> index) {
                bytes_out[index] = bytes_in[index];
            }).wait_and_throw();
        copied = true;
    } catch (const sycl::exception &error) {
        std::fprintf(stderr,
                     "[AIMDO XPU ERROR] small VBAR copy fallback failed: "
                     "stage=%s destination=%p size=%zu message=%s\n",
                     stage, destination, size, error.what());
        std::fflush(stderr);
    } catch (const std::exception &error) {
        std::fprintf(stderr,
                     "[AIMDO XPU ERROR] small VBAR copy fallback failed: "
                     "stage=%s destination=%p size=%zu message=%s\n",
                     stage, destination, size, error.what());
        std::fflush(stderr);
    }

    try {
        if (device_staging) {
            sycl::free(device_staging, state->queue->get_context());
        }
        if (host_staging) {
            sycl::free(host_staging, state->queue->get_context());
        }
    } catch (...) {
        /* The copy has already completed synchronously. A cleanup failure
         * is not allowed to turn a correct tensor write back into the
         * driver's spurious OOM. */
    }
    if (copied) {
        g_stats[kSmallVbarCopyFallbackBytes].fetch_add(
            size, std::memory_order_relaxed);
    } else {
        g_stats[kSmallVbarCopyFallbackFailures].fetch_add(
            1, std::memory_order_relaxed);
    }
    return copied;
}

extern "C" AIMDO_XPU_EXPORT bool aimdo_xpu_is_mapped_pinned_vbar(
    void *address, size_t size) {
    int mapped = 0;
    unsigned pin = 0;
    uint64_t page_index = 0;
    uint64_t unmapped_page = UINT64_MAX;
    uint64_t pages_spanned = 0;

    return address && size && aimdo_vbar_describe_range(
        reinterpret_cast<uint64_t>(address), size, &mapped, &pin,
        &page_index, &unmapped_page, &pages_spanned) && mapped && pin;
}

/* The failing adapter reports node mask 0x2 even though its matching DXGI
 * adapter exposes one D3D12 node; the working adapter reports 0x1. Route the
 * copy before calling the poisoned driver path, because after the first
 * rejected small copy even an otherwise safe padded H2D reports
 * "device or resource busy". The override makes the policy reversible on a
 * fixed driver and lets diagnostics force it on another adapter. */
extern "C" AIMDO_XPU_EXPORT bool
aimdo_xpu_needs_small_vbar_copy_workaround(int device) {
#if defined(_WIN32) || defined(_WIN64)
    const char *override_value =
        std::getenv("AIMDO_XPU_SMALL_VBAR_COPY_FALLBACK");
    auto *state = find_device(device);

    if (override_value) {
        if (std::strcmp(override_value, "0") == 0) {
            return false;
        }
        if (std::strcmp(override_value, "1") == 0) {
            return true;
        }
    }
    if (!state) {
        return false;
    }
    try {
        const sycl::device sycl_device = state->queue->get_device();
        if (!sycl_device.has(
                sycl::aspect::ext_intel_device_info_node_mask)) {
            return false;
        }
        return sycl_device.get_info<
                   sycl::ext::intel::info::device::node_mask>() != 1;
    } catch (...) {
        return false;
    }
#else
    (void)device;
    return false;
#endif
}

AIMDO_XPU_EXPORT int xpu_device_from_native_handle(
    uintptr_t native_handle) {
    return device_from_native_handle(native_handle);
}

AIMDO_XPU_EXPORT bool xpu_get_vmm_stats(
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
    aimdo_xpu_retire_reset();
    std::memset(&g_cuda, 0, sizeof(g_cuda));
}

}  // extern "C"
