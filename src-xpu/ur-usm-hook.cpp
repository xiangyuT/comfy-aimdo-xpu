#include <ur_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>

#if defined(_WIN32) || defined(_WIN64)
#error "The UR USM interposer is currently a Linux-only prototype"
#endif

#define AIMDO_XPU_EXPORT __attribute__((visibility("default")))

extern "C" bool aimdo_xpu_allocation_deficit(
    int device, size_t size, int64_t *deficit);
extern "C" bool aimdo_xpu_evict_for_allocation(
    int device, int64_t deficit);
extern "C" bool aimdo_xpu_account_allocation(int device, int64_t delta);
extern "C" int xpu_device_from_native_handle(uintptr_t native_handle);

namespace {

constexpr const char *kUrLoader = "libur_loader.so.0";
constexpr const char *kUrVersion = "LIBUR_LOADER_0.12";

using DeviceAllocFn = ur_result_t (*)(
    ur_context_handle_t,
    ur_device_handle_t,
    const ur_usm_desc_t *,
    ur_usm_pool_handle_t,
    size_t,
    void **);
using FreeFn = ur_result_t (*)(ur_context_handle_t, void *);
using DeviceGetNativeHandleFn = ur_result_t (*)(
    ur_device_handle_t, ur_native_handle_t *);

enum HookStat : size_t {
    kAllocCalls,
    kFreeCalls,
    kPassThroughAllocCalls,
    kTrackedAllocCalls,
    kTrackedAllocBytes,
    kTrackedFreeCalls,
    kTrackedFreeBytes,
    kSyntheticOomCalls,
    kRuntimeOomCalls,
    kNativeReclaimFreeCalls,
    kNativeReclaimFreeBytes,
    kRetryEvictionCalls,
    kRetryEvictionBytes,
    kUnknownDeviceCalls,
    kUnknownFreeCalls,
    kDroppedMetadataCalls,
    kHookStatCount,
};

struct Allocation {
    size_t size;
    int device;
};

enum class RetryReason {
    kNone,
    kBudgetDeficit,
    kRuntimeOom,
};

struct RetryState {
    ur_context_handle_t context = nullptr;
    ur_device_handle_t device = nullptr;
    ur_usm_pool_handle_t pool = nullptr;
    size_t size = 0;
    RetryReason reason = RetryReason::kNone;
};

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_stats[kHookStatCount];
std::mutex g_hook_mutex;
std::unordered_map<void *, Allocation> g_allocations;
thread_local RetryState g_retry;

void *open_ur_loader() {
    static void *loader = dlopen(kUrLoader, RTLD_NOW | RTLD_LOCAL);
    return loader;
}

template <typename Function>
Function resolve_real(const char *name) {
    dlerror();
    void *symbol = dlvsym(RTLD_NEXT, name, kUrVersion);
    if (!symbol) {
        dlerror();
        void *loader = open_ur_loader();
        if (loader) {
            symbol = dlvsym(loader, name, kUrVersion);
        }
    }
    return reinterpret_cast<Function>(symbol);
}

DeviceAllocFn real_device_alloc() {
    static DeviceAllocFn function = resolve_real<DeviceAllocFn>(
        "urUSMDeviceAlloc");
    return function;
}

FreeFn real_free() {
    static FreeFn function = resolve_real<FreeFn>("urUSMFree");
    return function;
}

DeviceGetNativeHandleFn real_device_get_native_handle() {
    static DeviceGetNativeHandleFn function =
        resolve_real<DeviceGetNativeHandleFn>("urDeviceGetNativeHandle");
    return function;
}

bool is_oom(ur_result_t result) {
    return result == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
           result == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
}

bool retry_matches(
    ur_context_handle_t context,
    ur_device_handle_t device,
    ur_usm_pool_handle_t pool,
    size_t size) {
    return g_retry.reason != RetryReason::kNone &&
           g_retry.context == context && g_retry.device == device &&
           g_retry.pool == pool && g_retry.size == size;
}

void arm_retry(
    ur_context_handle_t context,
    ur_device_handle_t device,
    ur_usm_pool_handle_t pool,
    size_t size,
    RetryReason reason) {
    g_retry = RetryState{context, device, pool, size, reason};
}

void clear_retry() {
    g_retry = RetryState{};
}

int resolve_device(ur_device_handle_t device) {
    DeviceGetNativeHandleFn get_native = real_device_get_native_handle();
    if (!get_native) {
        return -1;
    }
    ur_native_handle_t native_handle = 0;
    if (get_native(device, &native_handle) != UR_RESULT_SUCCESS) {
        return -1;
    }
    return xpu_device_from_native_handle(
        static_cast<uintptr_t>(native_handle));
}

bool account_success(void *pointer, size_t size, int device) {
    if (!pointer) {
        return false;
    }
    try {
        g_allocations.emplace(pointer, Allocation{size, device});
    } catch (...) {
        g_stats[kDroppedMetadataCalls].fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    aimdo_xpu_account_allocation(device, static_cast<int64_t>(size));
    g_stats[kTrackedAllocCalls].fetch_add(1, std::memory_order_relaxed);
    g_stats[kTrackedAllocBytes].fetch_add(size, std::memory_order_relaxed);
    return true;
}

bool same_file(const char *left, const char *right) {
    struct stat left_stat {};
    struct stat right_stat {};
    return left && right && stat(left, &left_stat) == 0 &&
           stat(right, &right_stat) == 0 &&
           left_stat.st_dev == right_stat.st_dev &&
           left_stat.st_ino == right_stat.st_ino;
}

bool loaded_through_ld_preload() {
    const char *preload = std::getenv("LD_PRELOAD");
    Dl_info self {};
    if (!preload || !*preload ||
        dladdr(reinterpret_cast<void *>(&urUSMDeviceAlloc), &self) == 0 ||
        !self.dli_fname) {
        return false;
    }

    std::string entries(preload);
    size_t begin = 0;
    while (begin < entries.size()) {
        begin = entries.find_first_not_of(" :\t\n", begin);
        if (begin == std::string::npos) {
            break;
        }
        const size_t end = entries.find_first_of(" :\t\n", begin);
        const std::string entry = entries.substr(begin, end - begin);
        if (same_file(entry.c_str(), self.dli_fname)) {
            return true;
        }
        begin = end;
    }
    return false;
}

}  // namespace

extern "C" ur_result_t urUSMDeviceAlloc(
    ur_context_handle_t context,
    ur_device_handle_t device,
    const ur_usm_desc_t *description,
    ur_usm_pool_handle_t pool,
    size_t size,
    void **pointer) {
    DeviceAllocFn real = real_device_alloc();
    if (!real) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    g_stats[kAllocCalls].fetch_add(1, std::memory_order_relaxed);
    if (!g_enabled.load(std::memory_order_acquire)) {
        g_stats[kPassThroughAllocCalls].fetch_add(
            1, std::memory_order_relaxed);
        return real(context, device, description, pool, size, pointer);
    }

    std::lock_guard<std::mutex> guard(g_hook_mutex);
    const int aimdo_device = resolve_device(device);
    if (aimdo_device < 0) {
        clear_retry();
        g_stats[kUnknownDeviceCalls].fetch_add(
            1, std::memory_order_relaxed);
        return real(context, device, description, pool, size, pointer);
    }

    int64_t deficit = 0;
    if (!aimdo_xpu_allocation_deficit(aimdo_device, size, &deficit)) {
        clear_retry();
        return real(context, device, description, pool, size, pointer);
    }

    // First account every physical segment that the native allocator may
    // later release. If metadata allocation fails, return OOM before touching
    // the real allocator so no untracked pointer can escape to PyTorch.
    try {
        g_allocations.reserve(g_allocations.size() + 1);
    } catch (...) {
        clear_retry();
        g_stats[kDroppedMetadataCalls].fetch_add(
            1, std::memory_order_relaxed);
        if (pointer) {
            *pointer = nullptr;
        }
        return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    if (retry_matches(context, device, pool, size)) {
        const RetryReason reason = g_retry.reason;
        clear_retry();
        int64_t eviction = std::max<int64_t>(deficit, 0);
        if (reason == RetryReason::kRuntimeOom) {
            eviction = std::max<int64_t>(
                eviction, static_cast<int64_t>(size));
        }
        if (eviction > 0) {
            aimdo_xpu_evict_for_allocation(aimdo_device, eviction);
            g_stats[kRetryEvictionCalls].fetch_add(
                1, std::memory_order_relaxed);
            g_stats[kRetryEvictionBytes].fetch_add(
                static_cast<uint64_t>(eviction),
                std::memory_order_relaxed);
        }
        ur_result_t result =
            real(context, device, description, pool, size, pointer);
        if (is_oom(result)) {
            g_stats[kRuntimeOomCalls].fetch_add(
                1, std::memory_order_relaxed);
        } else if (result == UR_RESULT_SUCCESS && pointer) {
            if (!account_success(*pointer, size, aimdo_device)) {
                real_free()(context, *pointer);
                *pointer = nullptr;
                return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
            }
        }
        return result;
    }

    clear_retry();
    if (deficit > 0) {
        arm_retry(
            context, device, pool, size, RetryReason::kBudgetDeficit);
        if (pointer) {
            *pointer = nullptr;
        }
        g_stats[kSyntheticOomCalls].fetch_add(
            1, std::memory_order_relaxed);
        return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    ur_result_t result = real(
        context, device, description, pool, size, pointer);
    if (is_oom(result)) {
        arm_retry(context, device, pool, size, RetryReason::kRuntimeOom);
        g_stats[kRuntimeOomCalls].fetch_add(1, std::memory_order_relaxed);
    } else if (result == UR_RESULT_SUCCESS && pointer) {
        if (!account_success(*pointer, size, aimdo_device)) {
            real_free()(context, *pointer);
            *pointer = nullptr;
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    return result;
}

extern "C" ur_result_t urUSMFree(
    ur_context_handle_t context, void *pointer) {
    FreeFn real = real_free();
    if (!real) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    g_stats[kFreeCalls].fetch_add(1, std::memory_order_relaxed);
    if (!g_enabled.load(std::memory_order_acquire)) {
        return real(context, pointer);
    }

    std::lock_guard<std::mutex> guard(g_hook_mutex);
    ur_result_t result = real(context, pointer);
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    auto found = g_allocations.find(pointer);
    if (found == g_allocations.end()) {
        g_stats[kUnknownFreeCalls].fetch_add(
            1, std::memory_order_relaxed);
        return result;
    }
    aimdo_xpu_account_allocation(
        found->second.device, -static_cast<int64_t>(found->second.size));
    if (g_retry.reason != RetryReason::kNone) {
        g_stats[kNativeReclaimFreeCalls].fetch_add(
            1, std::memory_order_relaxed);
        g_stats[kNativeReclaimFreeBytes].fetch_add(
            found->second.size, std::memory_order_relaxed);
    }
    g_stats[kTrackedFreeCalls].fetch_add(1, std::memory_order_relaxed);
    g_stats[kTrackedFreeBytes].fetch_add(
        found->second.size, std::memory_order_relaxed);
    g_allocations.erase(found);
    return result;
}

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_is_interposed() {
    try {
        if (!loaded_through_ld_preload()) {
            return false;
        }
    } catch (...) {
        return false;
    }
    dlerror();
    void *active = dlvsym(
        RTLD_DEFAULT, "urUSMDeviceAlloc", kUrVersion);
    return active == reinterpret_cast<void *>(&urUSMDeviceAlloc);
}

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_enable() {
    if (!xpu_ur_hook_is_interposed() || !real_device_alloc() ||
        !real_free() || !real_device_get_native_handle()) {
        return false;
    }
    g_enabled.store(true, std::memory_order_release);
    return true;
}

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_disable() {
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    if (!g_allocations.empty()) {
        return false;
    }
    g_enabled.store(false, std::memory_order_release);
    clear_retry();
    return true;
}

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_get_stats(
    uint64_t *values, size_t count) {
    if (!values || count < kHookStatCount) {
        return false;
    }
    for (size_t i = 0; i < kHookStatCount; ++i) {
        values[i] = g_stats[i].load(std::memory_order_relaxed);
    }
    return true;
}
