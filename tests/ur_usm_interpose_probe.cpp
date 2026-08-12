#include <ur_api.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace {

using DeviceAllocFn = ur_result_t (*)(
    ur_context_handle_t,
    ur_device_handle_t,
    const ur_usm_desc_t *,
    ur_usm_pool_handle_t,
    size_t,
    void **);
using FreeFn = ur_result_t (*)(ur_context_handle_t, void *);

std::atomic<uint64_t> g_alloc_calls{0};
std::atomic<uint64_t> g_alloc_bytes{0};
std::atomic<uint64_t> g_free_calls{0};
std::atomic<bool> g_failed_once{false};

template <typename Function>
Function resolve_next(const char *name) {
    dlerror();
    void *symbol = dlvsym(RTLD_NEXT, name, "LIBUR_LOADER_0.12");
    if (!symbol) {
        dlerror();
        static void *loader =
            dlopen("libur_loader.so.0", RTLD_NOW | RTLD_LOCAL);
        if (loader) {
            symbol = dlvsym(loader, name, "LIBUR_LOADER_0.12");
        }
    }
    const char *error = dlerror();
    if (error || !symbol) {
        std::fprintf(
            stderr, "[AIMDO UR PROBE] resolve_failed symbol=%s error=%s\n",
            name, error ? error : "unknown");
        std::abort();
    }
    return reinterpret_cast<Function>(symbol);
}

bool trace_calls() {
    static const bool enabled = [] {
        const char *value = std::getenv("AIMDO_UR_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

size_t fail_once_size() {
    static const size_t size = [] {
        const char *value = std::getenv("AIMDO_UR_FAIL_ONCE_SIZE");
        if (!value || !value[0]) {
            return static_cast<size_t>(0);
        }
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end && *end == '\0' ? static_cast<size_t>(parsed) : 0;
    }();
    return size;
}

}  // namespace

extern "C" ur_result_t urUSMDeviceAlloc(
    ur_context_handle_t context,
    ur_device_handle_t device,
    const ur_usm_desc_t *description,
    ur_usm_pool_handle_t pool,
    size_t size,
    void **pointer) {
    static DeviceAllocFn real =
        resolve_next<DeviceAllocFn>("urUSMDeviceAlloc");
    const uint64_t call =
        g_alloc_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    g_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    if (trace_calls()) {
        std::fprintf(
            stderr, "[AIMDO UR PROBE] alloc call=%llu size=%zu\n",
            static_cast<unsigned long long>(call), size);
    }
    if (size == fail_once_size() &&
        !g_failed_once.exchange(true, std::memory_order_relaxed)) {
        if (pointer) {
            *pointer = nullptr;
        }
        std::fprintf(
            stderr,
            "[AIMDO UR PROBE] synthetic_oom call=%llu size=%zu\n",
            static_cast<unsigned long long>(call), size);
        return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return real(context, device, description, pool, size, pointer);
}

extern "C" ur_result_t urUSMFree(
    ur_context_handle_t context, void *pointer) {
    static FreeFn real = resolve_next<FreeFn>("urUSMFree");
    const uint64_t call =
        g_free_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (trace_calls()) {
        std::fprintf(
            stderr, "[AIMDO UR PROBE] free call=%llu\n",
            static_cast<unsigned long long>(call));
    }
    return real(context, pointer);
}

__attribute__((destructor)) static void report_probe_counts() {
    std::fprintf(
        stderr,
        "[AIMDO UR PROBE] summary alloc_calls=%llu alloc_bytes=%llu "
        "free_calls=%llu\n",
        static_cast<unsigned long long>(
            g_alloc_calls.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            g_alloc_bytes.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            g_free_calls.load(std::memory_order_relaxed)));
}
