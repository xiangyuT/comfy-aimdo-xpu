#include <ur_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <execinfo.h>
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
    kDirectPressureCalls,
    kDirectPressureBytes,
    kDuplicatePointerCalls,
    kHookStatCount,
};

struct Allocation {
    size_t size;
    int device;
    bool accounted = true;
    bool retired = false;
    bool aimdo_owned = false;
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
    uint64_t generation = 0;
    int aimdo_device = -1;
    uint64_t returned_bytes = 0;
};

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_generation{0};
std::atomic<uint64_t> g_stats[kHookStatCount];
// Retired caller-owned USM can be freed during Torch's process teardown.
// Keep this non-owning ledger and its lock alive for those late callbacks.
std::mutex &g_hook_mutex = *new std::mutex();
std::unordered_map<void *, Allocation> &g_allocations =
    *new std::unordered_map<void *, Allocation>();
std::atomic<bool> g_owned_shutdown{false};
thread_local unsigned g_owned_usm_depth = 0;
std::atomic<uint64_t> g_retirement_stats[5];
std::atomic<uint64_t> g_retired_live_count{0};
// Published only at Python/model-owner boundaries, never by re-entering the
// native allocator from its UR callback. A hint is not a releasability proof.
std::unordered_map<int, uint64_t> g_torch_cached_bytes;
std::atomic<uint64_t> g_cache_lever_skipped_calls{0};
thread_local RetryState g_retry;

#ifdef AIMDO_XPU_TESTING
enum class TestRequestKind {
    kAutomatic,
    kDirect,
    kTorchNative,
};

DeviceAllocFn g_test_device_alloc = nullptr;
FreeFn g_test_free = nullptr;
DeviceGetNativeHandleFn g_test_device_get_native_handle = nullptr;
std::atomic<TestRequestKind> g_test_request_kind{TestRequestKind::kAutomatic};
void (*g_test_after_fast_enabled_check)() = nullptr;
bool g_test_retirement_ready = false;
#endif

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
#ifdef AIMDO_XPU_TESTING
    if (g_test_device_alloc) {
        return g_test_device_alloc;
    }
#endif
    static DeviceAllocFn function = resolve_real<DeviceAllocFn>(
        "urUSMDeviceAlloc");
    return function;
}

FreeFn real_free() {
#ifdef AIMDO_XPU_TESTING
    if (g_test_free) {
        return g_test_free;
    }
#endif
    static FreeFn function = resolve_real<FreeFn>("urUSMFree");
    return function;
}

DeviceGetNativeHandleFn real_device_get_native_handle() {
#ifdef AIMDO_XPU_TESTING
    if (g_test_device_get_native_handle) {
        return g_test_device_get_native_handle;
    }
#endif
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
    size_t size,
    uint64_t generation) {
    return g_retry.reason != RetryReason::kNone &&
           g_retry.context == context && g_retry.device == device &&
           g_retry.pool == pool && g_retry.size == size &&
           g_retry.generation == generation;
}

void arm_retry(
    ur_context_handle_t context,
    ur_device_handle_t device,
    ur_usm_pool_handle_t pool,
    size_t size,
    RetryReason reason,
    uint64_t generation, int aimdo_device = -1) {
    g_retry = RetryState{
        context, device, pool, size, reason, generation, aimdo_device, 0};
}

void clear_retry() {
    g_retry = RetryState{};
}

bool consume_cache_hint(int device) {
    auto found = g_torch_cached_bytes.find(device);
    if (found == g_torch_cached_bytes.end() || found->second == 0) {
        return false;
    }
    // Consume the hint when requesting a flush, even if Torch subsequently
    // cannot release a split/pending block. A stale estimate cannot repeatedly
    // request the same unsuccessful flush without another owner publication.
    found->second = 0;
    return true;
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

bool is_torch_native_segment_request() {
#ifdef AIMDO_XPU_TESTING
    const TestRequestKind kind =
        g_test_request_kind.load(std::memory_order_relaxed);
    if (kind != TestRequestKind::kAutomatic) {
        return kind == TestRequestKind::kTorchNative;
    }
#endif
    void *frames[48];
    const int count = backtrace(frames, 48);
    for (int index = 1; index < count; ++index) {
        Dl_info info {};
        if (dladdr(frames[index], &info) == 0 || !info.dli_fname) {
            continue;
        }
        const char *slash = std::strrchr(info.dli_fname, '/');
        const char *base = slash ? slash + 1 : info.dli_fname;
        if (std::strcmp(base, "libc10_xpu.so") == 0) {
            return true;
        }
    }
    return false;
}

enum class AccountResult {
    kSuccess,
    kInvalidPointer,
    kDuplicatePointer,
    kMetadataFailure,
};

AccountResult account_success(void *pointer, size_t size, int device) {
    if (!pointer) {
        return AccountResult::kInvalidPointer;
    }
    try {
        const auto inserted =
            g_allocations.emplace(pointer, Allocation{size, device, true, false,
                                                       g_owned_usm_depth != 0});
        if (!inserted.second) {
            g_stats[kDuplicatePointerCalls].fetch_add(
                1, std::memory_order_relaxed);
            return AccountResult::kDuplicatePointer;
        }
    } catch (...) {
        g_stats[kDroppedMetadataCalls].fetch_add(
            1, std::memory_order_relaxed);
        return AccountResult::kMetadataFailure;
    }
    if (!aimdo_xpu_account_allocation(
            device, static_cast<int64_t>(size))) {
        g_allocations.erase(pointer);
        g_stats[kDroppedMetadataCalls].fetch_add(
            1, std::memory_order_relaxed);
        return AccountResult::kMetadataFailure;
    }
    g_stats[kTrackedAllocCalls].fetch_add(1, std::memory_order_relaxed);
    g_stats[kTrackedAllocBytes].fetch_add(size, std::memory_order_relaxed);
    return AccountResult::kSuccess;
}

ur_result_t real_allocate_and_account(
    DeviceAllocFn real,
    FreeFn free,
    ur_context_handle_t context,
    ur_device_handle_t device,
    const ur_usm_desc_t *description,
    ur_usm_pool_handle_t pool,
    size_t size,
    void **pointer,
    int aimdo_device) {
    ur_result_t result =
        real(context, device, description, pool, size, pointer);
    if (is_oom(result)) {
        g_stats[kRuntimeOomCalls].fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    const AccountResult account =
        account_success(pointer ? *pointer : nullptr, size, aimdo_device);
    if (account == AccountResult::kSuccess) {
        return result;
    }
    // A duplicate pointer identifies a broken lower-allocation contract. Do
    // not free it here: the pointer may still belong to the first tracked
    // allocation. Other failures own a unique new allocation and can roll it
    // back safely.
    if (account != AccountResult::kDuplicatePointer && pointer && *pointer) {
        free(context, *pointer);
    }
    if (pointer) {
        *pointer = nullptr;
    }
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    if (g_owned_usm_depth && g_owned_shutdown.load(std::memory_order_acquire)) {
        if (pointer) *pointer = nullptr;
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (!g_enabled.load(std::memory_order_acquire)) {
        if (g_owned_usm_depth && g_owned_shutdown.load(std::memory_order_acquire)) {
            if (pointer) *pointer = nullptr;
            return UR_RESULT_ERROR_UNINITIALIZED;
        }
        g_stats[kPassThroughAllocCalls].fetch_add(
            1, std::memory_order_relaxed);
        return real(context, device, description, pool, size, pointer);
    }
#ifdef AIMDO_XPU_TESTING
    if (g_test_after_fast_enabled_check) {
        g_test_after_fast_enabled_check();
    }
#endif

    std::lock_guard<std::mutex> guard(g_hook_mutex);
    if (g_owned_usm_depth && g_owned_shutdown.load(std::memory_order_relaxed)) {
        if (pointer) *pointer = nullptr;
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (!g_enabled.load(std::memory_order_relaxed)) {
        clear_retry();
        g_stats[kPassThroughAllocCalls].fetch_add(
            1, std::memory_order_relaxed);
        return real(context, device, description, pool, size, pointer);
    }
    const uint64_t generation =
        g_generation.load(std::memory_order_relaxed);
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

    const bool caller_classified =
        deficit > 0 || g_retry.reason != RetryReason::kNone;
    bool torch_native_request =
        caller_classified && is_torch_native_segment_request();
    if (torch_native_request &&
        retry_matches(context, device, pool, size, generation)) {
        const RetryReason reason = g_retry.reason;
        const uint64_t returned_bytes = g_retry.returned_bytes;
        clear_retry();
        int64_t eviction = std::max<int64_t>(deficit, 0);
        if (reason == RetryReason::kRuntimeOom) {
            const size_t residual = size - std::min<size_t>(size, returned_bytes);
            eviction = std::max<int64_t>(eviction, static_cast<int64_t>(residual));
        }
        if (eviction > 0) {
            aimdo_xpu_evict_for_allocation(aimdo_device, eviction);
            g_stats[kRetryEvictionCalls].fetch_add(
                1, std::memory_order_relaxed);
            g_stats[kRetryEvictionBytes].fetch_add(
                static_cast<uint64_t>(eviction),
                std::memory_order_relaxed);
        }
        return real_allocate_and_account(
            real, real_free(), context, device, description, pool, size,
            pointer, aimdo_device);
    }

    clear_retry();
    if (deficit > 0) {
        if (torch_native_request && consume_cache_hint(aimdo_device)) {
            arm_retry(
                context, device, pool, size,
                RetryReason::kBudgetDeficit, generation, aimdo_device);
            if (pointer) {
                *pointer = nullptr;
            }
            g_stats[kSyntheticOomCalls].fetch_add(
                1, std::memory_order_relaxed);
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        if (torch_native_request) {
            g_cache_lever_skipped_calls.fetch_add(1, std::memory_order_relaxed);
        }
        const bool reclaimed = aimdo_xpu_evict_for_allocation(aimdo_device, deficit);
        if (!reclaimed && !torch_native_request) {
            if (pointer) {
                *pointer = nullptr;
            }
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        if (!torch_native_request) {
            g_stats[kDirectPressureCalls].fetch_add(
                1, std::memory_order_relaxed);
            g_stats[kDirectPressureBytes].fetch_add(
                static_cast<uint64_t>(deficit), std::memory_order_relaxed);
        }
    }

    ur_result_t result = real_allocate_and_account(
        real, real_free(), context, device, description, pool, size, pointer,
        aimdo_device);
    if (is_oom(result) && !caller_classified) {
        torch_native_request = is_torch_native_segment_request();
    }
    if (torch_native_request && is_oom(result)) {
        arm_retry(
            context, device, pool, size,
            RetryReason::kRuntimeOom, generation, aimdo_device);
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
    if (!g_enabled.load(std::memory_order_acquire) &&
        !g_retired_live_count.load(std::memory_order_acquire)) {
        return real(context, pointer);
    }

    std::lock_guard<std::mutex> guard(g_hook_mutex);
    const bool enabled = g_enabled.load(std::memory_order_relaxed);
    if (!enabled) {
        clear_retry();
    }
    ur_result_t result = real(context, pointer);
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    auto found = g_allocations.find(pointer);
    if (found == g_allocations.end()) {
        if (enabled) {
            g_stats[kUnknownFreeCalls].fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    if (found->second.accounted) {
        aimdo_xpu_account_allocation(
            found->second.device, -static_cast<int64_t>(found->second.size));
    }
    if (found->second.retired) {
        g_retirement_stats[3].fetch_add(1, std::memory_order_relaxed);
        g_retirement_stats[4].fetch_add(found->second.size, std::memory_order_relaxed);
        g_retired_live_count.fetch_sub(1, std::memory_order_release);
    } else if (g_retry.reason != RetryReason::kNone &&
        g_retry.context == context &&
        g_retry.aimdo_device == found->second.device &&
        g_retry.generation ==
            g_generation.load(std::memory_order_relaxed)) {
        g_retry.returned_bytes += found->second.size;
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

namespace {
void enable_locked() {
    g_generation.fetch_add(1, std::memory_order_relaxed);
    // A new context borrows the accounting of surviving external buffers.
    // Their later free is charged only if this adoption succeeded.
    for (auto &[pointer, allocation] : g_allocations) {
        (void)pointer;
        if (allocation.retired && !allocation.accounted) {
            allocation.accounted = aimdo_xpu_account_allocation(
                allocation.device, static_cast<int64_t>(allocation.size));
        }
    }
    g_owned_shutdown.store(false, std::memory_order_release);
    g_torch_cached_bytes.clear();
    clear_retry();
    g_enabled.store(true, std::memory_order_release);
}
}  // namespace

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_enable() {
    if (!xpu_ur_hook_is_interposed() || !real_device_alloc() ||
        !real_free() || !real_device_get_native_handle()) {
        return false;
    }
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    if (g_enabled.load(std::memory_order_relaxed)) {
        return true;
    }
    enable_locked();
    return true;
}

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_disable() {
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    if (!g_allocations.empty()) {
        return false;
    }
    g_enabled.store(false, std::memory_order_release);
    g_generation.fetch_add(1, std::memory_order_relaxed);
    g_torch_cached_bytes.clear();
    clear_retry();
    return true;
}

namespace {
bool retirement_ready() {
#ifdef AIMDO_XPU_TESTING
    return g_test_retirement_ready;
#else
    using Ready = bool (*)();
    auto ready = reinterpret_cast<Ready>(dlsym(
        RTLD_DEFAULT, "aimdo_full_proxy_transition_ready"));
    // The sidecar checks ownership/cache while holding its allocation gate.
    // Do not call it while holding the UR lock (native allocator lock order).
    return ready && ready();
#endif
}

bool retire_borrowed_locked() {
    for (const auto &[pointer, allocation] : g_allocations) {
        (void)pointer;
        if (allocation.aimdo_owned) return false;
    }
    g_owned_shutdown.store(true, std::memory_order_release);
    g_enabled.store(false, std::memory_order_release);
    g_generation.fetch_add(1, std::memory_order_relaxed);
    g_torch_cached_bytes.clear();
    clear_retry();
    for (auto &[pointer, allocation] : g_allocations) {
        (void)pointer;
        if (!allocation.retired) {
            allocation.retired = true;
            g_retirement_stats[1].fetch_add(1, std::memory_order_relaxed);
            g_retirement_stats[2].fetch_add(allocation.size, std::memory_order_relaxed);
            g_retired_live_count.fetch_add(1, std::memory_order_release);
        }
        // The old context is closed next. Keep physical-free counts honest.
        allocation.accounted = false;
    }
    g_retirement_stats[0].fetch_add(1, std::memory_order_relaxed);
    return true;
}
}  // namespace

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_retire_borrowed() {
    if (!retirement_ready()) return false;
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    return retire_borrowed_locked();
}

extern "C" AIMDO_XPU_EXPORT void aimdo_xpu_owned_usm_enter() {
    ++g_owned_usm_depth;
}

extern "C" AIMDO_XPU_EXPORT void aimdo_xpu_owned_usm_leave() {
    if (g_owned_usm_depth) --g_owned_usm_depth;
}

extern "C" AIMDO_XPU_EXPORT bool xpu_ur_hook_get_retirement_stats(
    uint64_t *values, size_t count) {
    if (!values || count != 7) return false;
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    for (size_t index = 0; index != 5; ++index) {
        values[index] = g_retirement_stats[index].load(std::memory_order_relaxed);
    }
    values[5] = g_retired_live_count.load(std::memory_order_relaxed);
    values[6] = 0;
    for (const auto &[pointer, allocation] : g_allocations) {
        (void)pointer;
        if (allocation.retired) values[6] += allocation.size;
    }
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

extern "C" AIMDO_XPU_EXPORT void xpu_ur_hook_set_torch_cached_bytes(
    int device, uint64_t bytes) {
    if (device < 0) {
        return;
    }
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    if (!g_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    try {
        g_torch_cached_bytes.insert_or_assign(device, bytes);
    } catch (...) {
        // Missing advisory metadata skips the optional cache-flush request.
        // It must never throw through the C ABI or lose allocation accounting.
        g_torch_cached_bytes.erase(device);
    }
}

extern "C" AIMDO_XPU_EXPORT uint64_t xpu_ur_hook_get_cache_lever_skipped_calls() {
    return g_cache_lever_skipped_calls.load(std::memory_order_relaxed);
}
