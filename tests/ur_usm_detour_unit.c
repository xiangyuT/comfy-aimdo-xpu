/* Unit test for the Windows Unified Runtime allocation hook.
 *
 * This is the Windows counterpart of tests/ur_usm_hook_unit.cpp. It includes
 * the hook source directly so the state machine can be driven without a
 * device, a Unified Runtime loader, or Detours, and it supplies its own AIMDO
 * side so budget decisions are scripted rather than sampled.
 *
 * It also measures the caller classification, because that runs on the hot
 * path whenever the budget is short and its cost was a candidate cause of the
 * gradient-ladder regression recorded in
 * docs/WINDOWS_XPU_UR_ALLOCATOR_HOOK.md. Both the retained range-compare form
 * and the original per-frame GetModuleFileNameW form are timed here so the
 * comparison is reproducible rather than asserted.
 */

#include <stdio.h>

#define AIMDO_XPU_TESTING 1

/* Scripted AIMDO side. Declared before the hook source so its extern
 * declarations match, and defined after it so the hook sees prototypes only. */
static long long g_script_deficit;
static long long g_evicted_bytes;
static long long g_accounted_bytes;
static int g_evict_calls;
static int g_real_alloc_calls;
static int g_real_free_calls;
static int g_fail_next_real_alloc;

#include "../src-xpu/ur-usm-detour.c"

static int g_failures;

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            printf("[FAIL] %s:%d %s\n", __func__, __LINE__, #condition);      \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

void aimdo_log(int level, const char *file, int line, const char *format, ...) {
    (void)level;
    (void)file;
    (void)line;
    (void)format;
}

bool aimdo_xpu_allocation_deficit(int device, size_t size, int64_t *deficit) {
    (void)device;
    (void)size;
    if (!deficit) {
        return false;
    }
    *deficit = (int64_t)g_script_deficit;
    return true;
}

bool aimdo_xpu_evict_for_allocation(int device, int64_t deficit) {
    (void)device;
    g_evicted_bytes += deficit;
    ++g_evict_calls;
    return true;
}

bool aimdo_xpu_account_allocation(int device, int64_t delta) {
    (void)device;
    g_accounted_bytes += delta;
    return true;
}

int xpu_device_from_native_handle(uintptr_t native_handle) {
    return native_handle == 0x1234 ? 0 : -1;
}

static ur_result_t __cdecl fake_alloc(ur_context_handle_t context,
                                      ur_device_handle_t device,
                                      const ur_usm_desc_t *description,
                                      ur_usm_pool_handle_t pool, size_t size,
                                      void **pointer) {
    static char storage[8];

    (void)context;
    (void)device;
    (void)description;
    (void)pool;
    ++g_real_alloc_calls;
    if (g_fail_next_real_alloc) {
        g_fail_next_real_alloc = 0;
        if (pointer) {
            *pointer = NULL;
        }
        return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (pointer) {
        /* A distinct address per size keeps the tracking table meaningful
         * without allocating anything real. */
        *pointer = storage + (size % 4);
    }
    return UR_RESULT_SUCCESS;
}

static ur_result_t __cdecl fake_free(ur_context_handle_t context,
                                     void *pointer) {
    (void)context;
    (void)pointer;
    ++g_real_free_calls;
    return UR_RESULT_SUCCESS;
}

static ur_result_t __cdecl fake_native_handle(ur_device_handle_t device,
                                              ur_native_handle_t *handle) {
    (void)device;
    if (handle) {
        *handle = (ur_native_handle_t)(uintptr_t)0x1234;
    }
    return UR_RESULT_SUCCESS;
}

static void reset(void) {
    g_script_deficit = 0;
    g_evicted_bytes = 0;
    g_accounted_bytes = 0;
    g_evict_calls = 0;
    g_real_alloc_calls = 0;
    g_real_free_calls = 0;
    g_fail_next_real_alloc = 0;
    test_reset_state();
    true_urUSMDeviceAlloc = fake_alloc;
    true_urUSMFree = fake_free;
    real_urDeviceGetNativeHandle = fake_native_handle;
    test_force_enable();
}

/* A request that fits the budget must reach the real allocator untouched, and
 * releasing it must return the accounting to zero. */
static void test_within_budget_passes_through(void) {
    void *pointer = NULL;

    reset();
    InterlockedExchange(&g_test_request_kind, kTestRequestTorchNative);
    g_script_deficit = 0;

    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 4096, &pointer) ==
          UR_RESULT_SUCCESS);
    CHECK(pointer != NULL);
    CHECK(g_real_alloc_calls == 1);
    CHECK(g_evict_calls == 0);
    CHECK(g_stats[kSyntheticOomCalls] == 0);
    CHECK(g_accounted_bytes == 4096);

    CHECK(aimdo_urUSMFree(NULL, pointer) == UR_RESULT_SUCCESS);
    CHECK(g_accounted_bytes == 0);
    CHECK(g_stats[kTrackedFreeCalls] == 1);
}

/* An over-budget PyTorch request must be refused without reaching the real
 * allocator, and the retry of the identical request must evict and succeed. */
static void test_torch_request_fails_then_retry_evicts(void) {
    void *pointer = (void *)(uintptr_t)0xdeadbeef;

    reset();
    InterlockedExchange(&g_test_request_kind, kTestRequestTorchNative);
    g_script_deficit = 4096;

    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 8192, &pointer) ==
          UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(pointer == NULL);
    CHECK(g_real_alloc_calls == 0);
    CHECK(g_evict_calls == 0);
    CHECK(g_stats[kSyntheticOomCalls] == 1);

    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 8192, &pointer) ==
          UR_RESULT_SUCCESS);
    CHECK(pointer != NULL);
    CHECK(g_real_alloc_calls == 1);
    CHECK(g_evict_calls == 1);
    CHECK(g_evicted_bytes == 4096);
    CHECK(g_stats[kRetryEvictionCalls] == 1);
}

/* A caller that is not PyTorch's allocator has no cache to surrender and no
 * retry to expect, so it must be evicted for directly rather than failed. */
static void test_direct_request_evicts_without_failing(void) {
    void *pointer = NULL;

    reset();
    InterlockedExchange(&g_test_request_kind, kTestRequestDirect);
    g_script_deficit = 2048;

    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 4096, &pointer) ==
          UR_RESULT_SUCCESS);
    CHECK(pointer != NULL);
    CHECK(g_stats[kSyntheticOomCalls] == 0);
    CHECK(g_stats[kDirectPressureCalls] == 1);
    CHECK(g_evict_calls == 1);
    CHECK(g_evicted_bytes == 2048);
    CHECK(g_real_alloc_calls == 1);
}

/* A retry must only match the request that was refused. A different size is a
 * new decision, not the continuation of the old one. */
static void test_retry_does_not_match_a_different_request(void) {
    void *pointer = NULL;

    reset();
    InterlockedExchange(&g_test_request_kind, kTestRequestTorchNative);
    g_script_deficit = 4096;

    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 8192, &pointer) ==
          UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(g_stats[kSyntheticOomCalls] == 1);

    /* Different size: must be refused on its own merits, not treated as the
     * armed retry. */
    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 16384, &pointer) ==
          UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(g_stats[kSyntheticOomCalls] == 2);
    CHECK(g_evict_calls == 0);
}

/* Disabling must succeed even while PyTorch still owns tracked segments;
 * requiring an empty table made control.deinit() raise on Windows. */
static void test_disable_succeeds_with_live_segments(void) {
    void *pointer = NULL;

    reset();
    InterlockedExchange(&g_test_request_kind, kTestRequestTorchNative);
    g_script_deficit = 0;
    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 4096, &pointer) ==
          UR_RESULT_SUCCESS);
    CHECK(g_allocation_count == 1);

    CHECK(xpu_ur_hook_disable());
    CHECK(g_allocation_count == 0);
    CHECK(!InterlockedCompareExchange(&g_enabled, 0, 0));

    /* Once disabled the hook must not arbitrate at all. */
    g_script_deficit = 1 << 20;
    CHECK(aimdo_urUSMDeviceAlloc(NULL, NULL, NULL, NULL, 4096, &pointer) ==
          UR_RESULT_SUCCESS);
    CHECK(g_stats[kPassThroughAllocCalls] == 1);
    CHECK(g_evict_calls == 0);
}

/* The tracking table must survive deletion churn. An earlier revision cleared
 * removed slots instead of tombstoning them, which truncated the probe chain
 * and lost the size of every pointer behind a deleted one. */
static void test_tracking_survives_deletion_churn(void) {
    enum { kCount = 512 };
    void *pointers[kCount];
    int index;
    int recovered = 0;

    reset();
    for (index = 0; index < kCount; ++index) {
        /* 2 MiB apart, which is the alignment that collided before. */
        pointers[index] = (void *)(uintptr_t)(0x100000000ull +
                                              (unsigned long long)index * 0x200000ull);
        CHECK(allocation_insert(pointers[index], 4096, 0));
    }
    for (index = 0; index < kCount; ++index) {
        size_t size = 0;
        int device = -1;

        if (allocation_remove(pointers[index], &size, &device) && size == 4096) {
            ++recovered;
        }
    }
    CHECK(recovered == kCount);
}

/* Correctness of the range classifier: an address inside a loaded module's
 * image must be recognised and one outside it must not. */
static void test_module_range_resolution(void) {
    uintptr_t base = 0;
    uintptr_t end = 0;
    HMODULE self = GetModuleHandleA("kernel32.dll");

    CHECK(module_image_range(L"kernel32.dll", &base, &end));
    CHECK(base != 0 && end > base);
    CHECK((uintptr_t)self >= base && (uintptr_t)self < end);
    CHECK((uintptr_t)(void *)&test_module_range_resolution < base ||
          (uintptr_t)(void *)&test_module_range_resolution >= end);
    CHECK(!module_image_range(L"definitely_not_loaded_xyz.dll", &base, &end));
}

/* The original per-frame form, kept only so its cost can be compared. */
static bool classify_by_module_path(void) {
    void *frames[48];
    USHORT count;
    USHORT index;

    count = RtlCaptureStackBackTrace(1, ARRAYSIZE(frames), frames, NULL);
    for (index = 0; index < count; ++index) {
        HMODULE module = NULL;
        wchar_t path[MAX_PATH];

        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR)frames[index], &module)) {
            continue;
        }
        if (!GetModuleFileNameW(module, path, ARRAYSIZE(path))) {
            continue;
        }
        if (_wcsicmp(base_name(path), L"kernel32.dll") == 0) {
            return true;
        }
    }
    return false;
}

static bool classify_by_range(uintptr_t base, uintptr_t end) {
    void *frames[CLASSIFY_MAX_FRAMES];
    USHORT count = RtlCaptureStackBackTrace(1, CLASSIFY_MAX_FRAMES, frames,
                                            NULL);
    USHORT index;

    for (index = 0; index < count; ++index) {
        const uintptr_t address = (uintptr_t)frames[index];

        if (address >= base && address < end) {
            return true;
        }
    }
    return false;
}

static double elapsed_ns(LARGE_INTEGER start, LARGE_INTEGER stop,
                         LARGE_INTEGER frequency, int iterations) {
    return (double)(stop.QuadPart - start.QuadPart) * 1e9 /
           ((double)frequency.QuadPart * iterations);
}

/* Stack depth matters more than anything else here. The per-frame form calls
 * GetModuleFileNameW once per frame, so a shallow microbenchmark flatters it
 * badly: the real PyTorch stack measured by tests/run_windows_ur_hook_probe.py
 * is 42 frames deep. These helpers recurse to a chosen depth first, and the
 * volatile accumulator keeps the recursion from being turned into a loop. */
static volatile int g_recursion_sink;

static bool recurse_path(int depth) {
    bool result;

    if (depth <= 0) {
        result = classify_by_module_path();
    } else {
        result = recurse_path(depth - 1);
    }
    g_recursion_sink += result ? 1 : 0;
    return result;
}

static bool recurse_range(int depth, uintptr_t base, uintptr_t end) {
    bool result;

    if (depth <= 0) {
        result = classify_by_range(base, end);
    } else {
        result = recurse_range(depth - 1, base, end);
    }
    g_recursion_sink += result ? 1 : 0;
    return result;
}

static void benchmark_at_depth(int depth, uintptr_t base, uintptr_t end,
                               double *path_ns, double *range_ns) {
    enum { kIterations = 20000 };
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
    int index;

    QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&start);
    for (index = 0; index < kIterations; ++index) {
        recurse_path(depth);
    }
    QueryPerformanceCounter(&stop);
    *path_ns = elapsed_ns(start, stop, frequency, kIterations);

    QueryPerformanceCounter(&start);
    for (index = 0; index < kIterations; ++index) {
        recurse_range(depth, base, end);
    }
    QueryPerformanceCounter(&stop);
    *range_ns = elapsed_ns(start, stop, frequency, kIterations);
}

static void benchmark_classification(void) {
    uintptr_t base = 0;
    uintptr_t end = 0;
    double shallow_path = 0.0;
    double shallow_range = 0.0;
    double deep_path = 0.0;
    double deep_range = 0.0;

    if (!module_image_range(L"kernel32.dll", &base, &end)) {
        printf("[SKIP] benchmark: kernel32.dll range unavailable\n");
        return;
    }

    benchmark_at_depth(2, base, end, &shallow_path, &shallow_range);
    benchmark_at_depth(36, base, end, &deep_path, &deep_range);

    printf("[TIME] depth=4   module_path=%7.0f ns  range=%7.0f ns  %.1fx\n",
           shallow_path, shallow_range, shallow_path / shallow_range);
    printf("[TIME] depth=38  module_path=%7.0f ns  range=%7.0f ns  %.1fx\n",
           deep_path, deep_range, deep_path / deep_range);
    printf("[TIME] a PyTorch allocation stack is 42 frames deep in practice\n");

    /* The range form exists only because it is much cheaper at the depth that
     * actually occurs. If that stops being true the change has lost its
     * justification. */
    CHECK(deep_range * 2.0 < deep_path);
}

int main(void) {
    test_within_budget_passes_through();
    test_torch_request_fails_then_retry_evicts();
    test_direct_request_evicts_without_failing();
    test_retry_does_not_match_a_different_request();
    test_disable_succeeds_with_live_segments();
    test_tracking_survives_deletion_churn();
    test_module_range_resolution();
    benchmark_classification();

    if (g_failures) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
