/* Windows Unified Runtime USM allocation interception for AIMDO.
 *
 * This is the Windows counterpart of src-xpu/ur-usm-hook.cpp. Both arbitrate
 * the same entry points for the same reason: PyTorch keeps its native XPU
 * caching allocator, with all of its block splitting, coalescing and
 * empty_cache() behaviour, and AIMDO intercepts the Unified Runtime call that
 * grows device memory so it can apply the AIMDO budget before the request is
 * placed.
 *
 * Only the injection method differs. Linux exports urUSMDeviceAlloc in the
 * LIBUR_LOADER_0.12 version namespace and is injected with LD_PRELOAD. Windows
 * has neither mechanism, but it does not need them: the SYCL runtime obtains
 * the ur_loader.dll module handle from ur_win_proxy_loader.dll and resolves the
 * flat exports by name with GetProcAddress, so the pointer it calls points
 * directly into ur_loader.dll's code. Patching that function with Detours
 * intercepts every caller no matter when the pointer was captured, which is why
 * this hook can be installed from control.init() instead of before the process
 * starts.
 *
 * Why this supersedes arbitrating at zeMemAllocDevice (see ze-detour.c):
 *
 *   ur_loader.dll!urUSMDeviceAlloc      <- AIMDO decides here
 *     -> ur_adapter_level_zero.dll
 *       -> ze_loader.dll!zeMemAllocDevice   <- the driver allocates here
 *
 * A hook at the driver entry point cannot reclaim before the allocation
 * without re-entering Level Zero's own memory management from inside its
 * allocation call, which was observed to corrupt driver state and surface as
 * UR_RESULT_ERROR_DEVICE_LOST. Arbitrating one layer up removes that hazard:
 * when this hook returns a synthetic failure it has not called the adapter at
 * all, and when it evicts on the retry there is no driver allocation in flight
 * on the thread, so zeVirtualMemUnmap and zePhysicalMemDestroy are ordinary
 * calls.
 *
 * Reclaim on this path must still never wait. aimdo_xpu_evict_for_allocation()
 * routes to vbars_free_retired() on Windows, which releases only provably idle
 * pages and returns immediately otherwise.
 */

#include <windows.h>
#include <psapi.h>
#include <detours.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Minimal Unified Runtime declarations. ur_api.h is not required to build
 * AIMDO and is not assumed to be installed; only opaque handles and three
 * result codes are needed, and those are part of the frozen UR ABI. */
typedef int ur_result_t;

#define UR_RESULT_SUCCESS 0
#define UR_RESULT_ERROR_UNINITIALIZED 37
#define UR_RESULT_ERROR_OUT_OF_HOST_MEMORY 38
#define UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY 39

typedef void *ur_context_handle_t;
typedef void *ur_device_handle_t;
typedef void *ur_usm_pool_handle_t;
typedef void *ur_native_handle_t;
typedef struct ur_usm_desc_t ur_usm_desc_t;

typedef ur_result_t(__cdecl *PFN_urUSMDeviceAlloc)(
    ur_context_handle_t, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t, void **);
typedef ur_result_t(__cdecl *PFN_urUSMFree)(ur_context_handle_t, void *);
typedef ur_result_t(__cdecl *PFN_urDeviceGetNativeHandle)(ur_device_handle_t,
                                                          ur_native_handle_t *);
typedef ur_result_t(__cdecl *PFN_urPhysicalMemCreate)(ur_context_handle_t,
                                                      ur_device_handle_t, size_t,
                                                      const void *, void **);
typedef void *(__cdecl *PFN_getPreloadedURLib)(void);

#if defined(__cplusplus)
extern "C" {
#endif

enum AimdoUrLogLevel {
    kAimdoUrLogError = 2,
    kAimdoUrLogWarning = 3,
    kAimdoUrLogInfo = 4,
    kAimdoUrLogDebug = 5,
};

extern void aimdo_log(
    int level, const char *file, int line, const char *format, ...);
extern bool aimdo_xpu_allocation_deficit(int device, size_t size,
                                         int64_t *deficit);
extern bool aimdo_xpu_evict_for_allocation(int device, int64_t deficit);
extern bool aimdo_xpu_account_allocation(int device, int64_t delta);
extern int xpu_device_from_native_handle(uintptr_t native_handle);

/* Statistic order must match src-xpu/ur-usm-hook.cpp so control.py can report
 * both platforms with one table. */
enum UrHookStat {
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
    kPhysicalMemCreateCalls,
    kHookStatCount
};

enum RetryReason {
    kRetryNone = 0,
    kRetryBudgetDeficit,
    kRetryRuntimeOom
};

#ifdef AIMDO_XPU_TESTING
/* Seams used by tests/ur_usm_detour_unit.c. They mirror the ones
 * src-xpu/ur-usm-hook.cpp exposes for the Linux unit test, so both platforms
 * verify the same state machine without a device or a Unified Runtime loader. */
enum TestRequestKind {
    kTestRequestAutomatic = 0,
    kTestRequestDirect,
    kTestRequestTorchNative
};
static volatile LONG g_test_request_kind;
#endif

static PFN_urUSMDeviceAlloc true_urUSMDeviceAlloc;
static PFN_urUSMFree true_urUSMFree;
static PFN_urPhysicalMemCreate true_urPhysicalMemCreate;
static PFN_urDeviceGetNativeHandle real_urDeviceGetNativeHandle;

static volatile LONG g_attached;
static volatile LONG g_enabled;
static volatile LONG64 g_generation;
static volatile LONG64 g_stats[kHookStatCount];
static CRITICAL_SECTION g_hook_lock;
static volatile LONG g_lock_ready;

static __declspec(thread) ur_context_handle_t t_retry_context;
static __declspec(thread) ur_device_handle_t t_retry_device;
static __declspec(thread) ur_usm_pool_handle_t t_retry_pool;
static __declspec(thread) size_t t_retry_size;
static __declspec(thread) int t_retry_reason;
static __declspec(thread) LONG64 t_retry_generation;

/* Live tracked allocations. Open addressed with tombstones: USM device
 * allocations are highly aligned and collide heavily, and clearing a slot on
 * removal would truncate the probe chain of every pointer behind it. */
#define POINTER_TABLE_SIZE 262144
#define POINTER_TOMBSTONE ((void *)(uintptr_t)1)

struct TrackedAllocation {
    void *pointer;
    size_t size;
    int device;
};

static struct TrackedAllocation g_allocations[POINTER_TABLE_SIZE];
static volatile LONG64 g_allocation_count;

static void stat_add(enum UrHookStat stat, LONG64 delta) {
    InterlockedAdd64(&g_stats[stat], delta);
}

static size_t pointer_slot(void *pointer) {
    uint64_t value = (uint64_t)(uintptr_t)pointer;

    /* Fold the high bits down before mixing: multiplication only propagates
     * bits upwards, so hashing the low bits alone would map every 2 MiB
     * aligned allocation to one slot. */
    value ^= value >> 33;
    value *= 11400714819323198485ull;
    return (size_t)((value >> 44) & (POINTER_TABLE_SIZE - 1));
}

static bool allocation_insert(void *pointer, size_t size, int device) {
    size_t slot = pointer_slot(pointer);
    size_t probe;

    for (probe = 0; probe < POINTER_TABLE_SIZE; ++probe) {
        size_t index = (slot + probe) & (POINTER_TABLE_SIZE - 1);

        if (g_allocations[index].pointer == pointer) {
            return false;
        }
        if (!g_allocations[index].pointer ||
            g_allocations[index].pointer == POINTER_TOMBSTONE) {
            g_allocations[index].pointer = pointer;
            g_allocations[index].size = size;
            g_allocations[index].device = device;
            InterlockedIncrement64(&g_allocation_count);
            return true;
        }
    }
    return false;
}

static bool allocation_remove(void *pointer, size_t *size, int *device) {
    size_t slot = pointer_slot(pointer);
    size_t probe;

    for (probe = 0; probe < POINTER_TABLE_SIZE; ++probe) {
        size_t index = (slot + probe) & (POINTER_TABLE_SIZE - 1);

        if (!g_allocations[index].pointer) {
            return false;
        }
        if (g_allocations[index].pointer == pointer) {
            *size = g_allocations[index].size;
            *device = g_allocations[index].device;
            g_allocations[index].pointer = POINTER_TOMBSTONE;
            g_allocations[index].size = 0;
            InterlockedDecrement64(&g_allocation_count);
            return true;
        }
    }
    return false;
}

static const wchar_t *base_name(const wchar_t *path) {
    const wchar_t *slash = wcsrchr(path, L'\\');

    return slash ? slash + 1 : path;
}

typedef BOOL(WINAPI *PFN_EnumProcessModules)(HANDLE, HMODULE *, DWORD, LPDWORD);
typedef BOOL(WINAPI *PFN_GetModuleInformation)(HANDLE, HMODULE, LPMODULEINFO,
                                               DWORD);

/* Caller classification.
 *
 * ur-usm-hook.cpp walks the stack with backtrace()/dladdr() looking for
 * libc10_xpu.so. The direct Windows transliteration - GetModuleHandleExW plus
 * GetModuleFileNameW on every frame - resolves and formats a module path up to
 * 48 times per allocation, and it runs precisely when the deficit is positive,
 * which is the hot path under pressure. dladdr() is cheap enough for that
 * shape; the Win32 pair is not.
 *
 * c10_xpu.dll occupies one contiguous image range that cannot move while it is
 * loaded, so the range is resolved once and each frame becomes a pair of
 * integer comparisons. See docs/WINDOWS_XPU_UR_ALLOCATOR_HOOK.md for the
 * measured cost of both forms. */
#define CLASSIFY_MAX_FRAMES 32

static uintptr_t g_torch_module_base;
static uintptr_t g_torch_module_end;
static volatile LONG g_torch_module_state; /* 0 unknown, 1 resolved */

static volatile LONG64 g_classify_calls;
static volatile LONG64 g_classify_ticks;
static volatile LONG64 g_classify_torch_hits;
static volatile LONG64 g_hook_calls;
static volatile LONG64 g_hook_ticks;

static HMODULE find_module_by_name(const wchar_t *wanted) {
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    PFN_EnumProcessModules enumerate;
    HMODULE modules[1024];
    DWORD needed = 0;
    DWORD count;
    DWORD index;

    if (!kernel) {
        return NULL;
    }
    enumerate = (PFN_EnumProcessModules)(void *)GetProcAddress(
        kernel, "K32EnumProcessModules");
    if (!enumerate ||
        !enumerate(GetCurrentProcess(), modules, sizeof(modules), &needed)) {
        return NULL;
    }
    count = needed / (DWORD)sizeof(HMODULE);
    if (count > ARRAYSIZE(modules)) {
        count = ARRAYSIZE(modules);
    }
    for (index = 0; index < count; ++index) {
        wchar_t path[MAX_PATH];

        if (!GetModuleFileNameW(modules[index], path, ARRAYSIZE(path))) {
            continue;
        }
        if (_wcsicmp(base_name(path), wanted) == 0) {
            return modules[index];
        }
    }
    return NULL;
}

/* Resolve a loaded module's image range. Returns false while the module is not
 * loaded yet, which is normal before PyTorch initializes XPU, so the result is
 * only cached once it succeeds. */
static bool module_image_range(const wchar_t *name, uintptr_t *base,
                               uintptr_t *end) {
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    PFN_GetModuleInformation get_information;
    MODULEINFO information;
    HMODULE module = GetModuleHandleW(name);

    if (!module) {
        module = find_module_by_name(name);
    }
    if (!module || !kernel) {
        return false;
    }
    get_information = (PFN_GetModuleInformation)(void *)GetProcAddress(
        kernel, "K32GetModuleInformation");
    if (!get_information ||
        !get_information(GetCurrentProcess(), module, &information,
                         (DWORD)sizeof(information))) {
        return false;
    }
    *base = (uintptr_t)information.lpBaseOfDll;
    *end = *base + information.SizeOfImage;
    return true;
}

static bool ensure_torch_module(void) {
    uintptr_t base;
    uintptr_t end;

    if (InterlockedCompareExchange(&g_torch_module_state, 0, 0)) {
        return true;
    }
    if (!module_image_range(L"c10_xpu.dll", &base, &end)) {
        return false;
    }
    g_torch_module_base = base;
    g_torch_module_end = end;
    InterlockedExchange(&g_torch_module_state, 1);
    return true;
}

static bool is_torch_native_segment_request(void) {
    void *frames[CLASSIFY_MAX_FRAMES];
    LARGE_INTEGER started;
    LARGE_INTEGER finished;
    USHORT count;
    USHORT index;
    bool found = false;

#ifdef AIMDO_XPU_TESTING
    if (g_test_request_kind != kTestRequestAutomatic) {
        return g_test_request_kind == kTestRequestTorchNative;
    }
#endif
    if (!ensure_torch_module()) {
        return false;
    }

    QueryPerformanceCounter(&started);
    count = RtlCaptureStackBackTrace(1, CLASSIFY_MAX_FRAMES, frames, NULL);
    for (index = 0; index < count; ++index) {
        const uintptr_t address = (uintptr_t)frames[index];

        if (address >= g_torch_module_base && address < g_torch_module_end) {
            found = true;
            break;
        }
    }
    QueryPerformanceCounter(&finished);

    InterlockedAdd64(&g_classify_ticks, finished.QuadPart - started.QuadPart);
    InterlockedIncrement64(&g_classify_calls);
    if (found) {
        InterlockedIncrement64(&g_classify_torch_hits);
    }
    return found;
}

static int resolve_device(ur_device_handle_t device) {
    ur_native_handle_t native = NULL;

    if (!real_urDeviceGetNativeHandle) {
        return -1;
    }
    if (real_urDeviceGetNativeHandle(device, &native) != UR_RESULT_SUCCESS) {
        return -1;
    }
    return xpu_device_from_native_handle((uintptr_t)native);
}

static void clear_retry(void) {
    t_retry_context = NULL;
    t_retry_device = NULL;
    t_retry_pool = NULL;
    t_retry_size = 0;
    t_retry_reason = kRetryNone;
    t_retry_generation = 0;
}

static bool retry_matches(ur_context_handle_t context, ur_device_handle_t device,
                          ur_usm_pool_handle_t pool, size_t size,
                          LONG64 generation) {
    return t_retry_reason != kRetryNone && t_retry_context == context &&
           t_retry_device == device && t_retry_pool == pool &&
           t_retry_size == size && t_retry_generation == generation;
}

static void arm_retry(ur_context_handle_t context, ur_device_handle_t device,
                      ur_usm_pool_handle_t pool, size_t size, int reason,
                      LONG64 generation) {
    t_retry_context = context;
    t_retry_device = device;
    t_retry_pool = pool;
    t_retry_size = size;
    t_retry_reason = reason;
    t_retry_generation = generation;
}

static bool is_oom(ur_result_t result) {
    return result == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
           result == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
}

static ur_result_t allocate_and_account(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer, int aimdo_device) {
    ur_result_t result =
        true_urUSMDeviceAlloc(context, device, description, pool, size, pointer);

    if (is_oom(result)) {
        stat_add(kRuntimeOomCalls, 1);
        return result;
    }
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    if (!pointer || !*pointer) {
        return result;
    }
    if (!allocation_insert(*pointer, size, aimdo_device)) {
        /* A duplicate pointer means the layer below broke its contract. Do not
         * free it: it may still belong to the first tracked allocation. */
        stat_add(kDuplicatePointerCalls, 1);
        stat_add(kDroppedMetadataCalls, 1);
        return result;
    }
    aimdo_xpu_account_allocation(aimdo_device, (int64_t)size);
    stat_add(kTrackedAllocCalls, 1);
    stat_add(kTrackedAllocBytes, (LONG64)size);
    return result;
}

static ur_result_t aimdo_urUSMDeviceAlloc_body(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer) {
    LONG64 generation;
    int aimdo_device;
    int64_t deficit = 0;
    bool caller_classified;
    bool torch_native_request;
    ur_result_t result;

    if (!true_urUSMDeviceAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        stat_add(kPassThroughAllocCalls, 1);
        return true_urUSMDeviceAlloc(context, device, description, pool, size,
                                     pointer);
    }

    EnterCriticalSection(&g_hook_lock);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        clear_retry();
        LeaveCriticalSection(&g_hook_lock);
        stat_add(kPassThroughAllocCalls, 1);
        return true_urUSMDeviceAlloc(context, device, description, pool, size,
                                     pointer);
    }

    generation = InterlockedCompareExchange64(&g_generation, 0, 0);
    aimdo_device = resolve_device(device);
    if (aimdo_device < 0) {
        clear_retry();
        LeaveCriticalSection(&g_hook_lock);
        stat_add(kUnknownDeviceCalls, 1);
        return true_urUSMDeviceAlloc(context, device, description, pool, size,
                                     pointer);
    }

    if (!aimdo_xpu_allocation_deficit(aimdo_device, size, &deficit)) {
        clear_retry();
        LeaveCriticalSection(&g_hook_lock);
        return true_urUSMDeviceAlloc(context, device, description, pool, size,
                                     pointer);
    }

    caller_classified = deficit > 0 || t_retry_reason != kRetryNone;
    torch_native_request =
        caller_classified && is_torch_native_segment_request();

    if (torch_native_request &&
        retry_matches(context, device, pool, size, generation)) {
        /* PyTorch is retrying the request this hook failed. It has just called
         * release_cached_blocks(), so its cache has already come back through
         * urUSMFree and been credited. Evict what is still missing and let the
         * allocation through. */
        int reason = t_retry_reason;
        int64_t eviction = deficit > 0 ? deficit : 0;

        clear_retry();
        if (reason == kRetryRuntimeOom && eviction < (int64_t)size) {
            eviction = (int64_t)size;
        }
        if (eviction > 0) {
            aimdo_xpu_evict_for_allocation(aimdo_device, eviction);
            stat_add(kRetryEvictionCalls, 1);
            stat_add(kRetryEvictionBytes, eviction);
        }
        result = allocate_and_account(context, device, description, pool, size,
                                      pointer, aimdo_device);
        LeaveCriticalSection(&g_hook_lock);
        return result;
    }

    clear_retry();
    if (deficit > 0) {
        if (torch_native_request) {
            /* Fail the request instead of evicting immediately. PyTorch
             * responds by releasing its cached blocks, which no hook below
             * this one can see because they never reach the driver, and then
             * retries the identical request. That converts Torch's own cache
             * into the first source of reclaim and keeps AIMDO's VBAR pages
             * as the second. */
            arm_retry(context, device, pool, size, kRetryBudgetDeficit,
                      generation);
            if (pointer) {
                *pointer = NULL;
            }
            stat_add(kSyntheticOomCalls, 1);
            LeaveCriticalSection(&g_hook_lock);
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        /* Not a PyTorch segment request, so there is no cache to ask for and
         * no retry to expect. Evict directly; this never waits on Windows. */
        if (!aimdo_xpu_evict_for_allocation(aimdo_device, deficit)) {
            if (pointer) {
                *pointer = NULL;
            }
            LeaveCriticalSection(&g_hook_lock);
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        stat_add(kDirectPressureCalls, 1);
        stat_add(kDirectPressureBytes, deficit);
    }

    result = allocate_and_account(context, device, description, pool, size,
                                  pointer, aimdo_device);
    if (is_oom(result)) {
        if (!caller_classified) {
            torch_native_request = is_torch_native_segment_request();
        }
        if (torch_native_request) {
            arm_retry(context, device, pool, size, kRetryRuntimeOom, generation);
        }
    }
    LeaveCriticalSection(&g_hook_lock);
    return result;
}

/* Wrapper that times the whole hook, including the driver call it wraps. */
static ur_result_t __cdecl aimdo_urUSMDeviceAlloc(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer) {
    LARGE_INTEGER started;
    LARGE_INTEGER finished;
    ur_result_t result;

    QueryPerformanceCounter(&started);
    result = aimdo_urUSMDeviceAlloc_body(context, device, description, pool,
                                         size, pointer);
    QueryPerformanceCounter(&finished);
    stat_add(kAllocCalls, 1);
    InterlockedAdd64(&g_hook_ticks, finished.QuadPart - started.QuadPart);
    InterlockedIncrement64(&g_hook_calls);
    return result;
}

static ur_result_t __cdecl aimdo_urUSMFree(ur_context_handle_t context,
                                           void *pointer) {
    ur_result_t result;
    size_t size = 0;
    int device = -1;

    if (!true_urUSMFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    stat_add(kFreeCalls, 1);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        return true_urUSMFree(context, pointer);
    }

    EnterCriticalSection(&g_hook_lock);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        LeaveCriticalSection(&g_hook_lock);
        return true_urUSMFree(context, pointer);
    }
    result = true_urUSMFree(context, pointer);
    if (result != UR_RESULT_SUCCESS) {
        LeaveCriticalSection(&g_hook_lock);
        return result;
    }
    if (!allocation_remove(pointer, &size, &device)) {
        stat_add(kUnknownFreeCalls, 1);
        LeaveCriticalSection(&g_hook_lock);
        return result;
    }
    aimdo_xpu_account_allocation(device, -(int64_t)size);
    if (t_retry_reason != kRetryNone &&
        t_retry_generation == InterlockedCompareExchange64(&g_generation, 0, 0)) {
        /* Released while a synthetic failure is outstanding, so this is
         * PyTorch returning its cache in response to it. */
        stat_add(kNativeReclaimFreeCalls, 1);
        stat_add(kNativeReclaimFreeBytes, (LONG64)size);
    }
    stat_add(kTrackedFreeCalls, 1);
    stat_add(kTrackedFreeBytes, (LONG64)size);
    LeaveCriticalSection(&g_hook_lock);
    return result;
}

/* Counted, never arbitrated. PyTorch allocates through this entry point
 * instead of USM when expandable segments are enabled, in which case
 * urUSMDeviceAlloc is never called and AIMDO would otherwise be silently
 * blind. control.py refuses that configuration; this counter is what proves
 * the refusal is warranted. */
static ur_result_t __cdecl aimdo_urPhysicalMemCreate(
    ur_context_handle_t context, ur_device_handle_t device, size_t size,
    const void *properties, void **handle) {
    stat_add(kPhysicalMemCreateCalls, 1);
    return true_urPhysicalMemCreate(context, device, size, properties, handle);
}



/* Resolve the loader SYCL itself uses.
 *
 * GetModuleHandleA("ur_loader.dll") is correct in every environment measured
 * so far, but a process may hold more than one copy, so the authoritative
 * handle is the one ur_win_proxy_loader.dll hands to the SYCL runtime. When
 * both are available and disagree, the proxy wins and the mismatch is logged,
 * because hooking the wrong copy would silently arbitrate nothing.
 *
 * Note that GetModuleHandleA cannot find the proxy itself in a live PyTorch
 * process even though it is loaded, hence the module walk. */
static HMODULE resolve_ur_loader(void) {
    HMODULE proxy = find_module_by_name(L"ur_win_proxy_loader.dll");
    HMODULE by_name = GetModuleHandleA("ur_loader.dll");
    HMODULE from_proxy = NULL;

    if (proxy) {
        PFN_getPreloadedURLib get_lib = (PFN_getPreloadedURLib)(void *)
            GetProcAddress(proxy, "?getPreloadedURLib@@YAPEAXXZ");

        if (get_lib) {
            from_proxy = (HMODULE)get_lib();
        }
    }
    if (from_proxy && by_name && from_proxy != by_name) {
        aimdo_log(kAimdoUrLogWarning, __FILE__, __LINE__,
                  "%s: more than one ur_loader.dll is loaded; hooking the copy "
                  "used by the SYCL runtime\n", __func__);
    }
    return from_proxy ? from_proxy : by_name;
}

#ifdef AIMDO_XPU_TESTING
/* Bring the hook up without Detours or a Unified Runtime loader. The unit test
 * installs its own entry points into the true_* pointers first. */
static void test_force_enable(void) {
    if (!InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
        InitializeCriticalSection(&g_hook_lock);
        InterlockedExchange(&g_lock_ready, 1);
    }
    InterlockedExchange(&g_attached, 1);
    InterlockedIncrement64(&g_generation);
    clear_retry();
    InterlockedExchange(&g_enabled, 1);
}

static void test_reset_state(void) {
    size_t index;

    for (index = 0; index < (size_t)kHookStatCount; ++index) {
        InterlockedExchange64(&g_stats[index], 0);
    }
    InterlockedExchange64(&g_allocation_count, 0);
    memset(g_allocations, 0, sizeof(g_allocations));
    clear_retry();
}
#endif

bool aimdo_xpu_ur_hook_install(void) {
    HMODULE loader;
    LONG status;

    if (InterlockedCompareExchange(&g_attached, 0, 0)) {
        return true;
    }
    if (!InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
        InitializeCriticalSection(&g_hook_lock);
        InterlockedExchange(&g_lock_ready, 1);
    }

    loader = resolve_ur_loader();
    if (!loader) {
        aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                  "%s: ur_loader.dll is not loaded; import torch before "
                  "initializing AIMDO\n", __func__);
        return false;
    }

    true_urUSMDeviceAlloc =
        (PFN_urUSMDeviceAlloc)(void *)GetProcAddress(loader, "urUSMDeviceAlloc");
    true_urUSMFree = (PFN_urUSMFree)(void *)GetProcAddress(loader, "urUSMFree");
    true_urPhysicalMemCreate = (PFN_urPhysicalMemCreate)(void *)GetProcAddress(
        loader, "urPhysicalMemCreate");
    real_urDeviceGetNativeHandle =
        (PFN_urDeviceGetNativeHandle)(void *)GetProcAddress(
            loader, "urDeviceGetNativeHandle");
    if (!true_urUSMDeviceAlloc || !true_urUSMFree ||
        !real_urDeviceGetNativeHandle) {
        aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                  "%s: Unified Runtime USM entry points were not found\n",
                  __func__);
        return false;
    }

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    status = DetourAttach((void **)&true_urUSMDeviceAlloc, aimdo_urUSMDeviceAlloc);
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&true_urUSMFree, aimdo_urUSMFree);
    }
    if (status == NO_ERROR && true_urPhysicalMemCreate) {
        status = DetourAttach((void **)&true_urPhysicalMemCreate,
                              aimdo_urPhysicalMemCreate);
    }
    if (status != NO_ERROR) {
        DetourTransactionAbort();
        aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                  "%s: DetourAttach failed: %ld\n", __func__, (long)status);
        true_urUSMDeviceAlloc = NULL;
        true_urUSMFree = NULL;
        true_urPhysicalMemCreate = NULL;
        return false;
    }
    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                  "%s: DetourTransactionCommit failed: %ld\n", __func__,
                  (long)status);
        return false;
    }

    InterlockedExchange(&g_attached, 1);
    aimdo_log(kAimdoUrLogInfo, __FILE__, __LINE__,
              "%s: arbitrating Unified Runtime USM allocations; PyTorch "
              "caching allocator retained\n", __func__);
    return true;
}

void aimdo_xpu_ur_hook_remove(void) {
    LONG status;

    if (!InterlockedCompareExchange(&g_attached, 0, 0)) {
        return;
    }
    InterlockedExchange(&g_enabled, 0);
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourDetach((void **)&true_urUSMDeviceAlloc, aimdo_urUSMDeviceAlloc);
    DetourDetach((void **)&true_urUSMFree, aimdo_urUSMFree);
    if (true_urPhysicalMemCreate) {
        DetourDetach((void **)&true_urPhysicalMemCreate,
                     aimdo_urPhysicalMemCreate);
    }
    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        /* The hooks are still live; keeping the trampolines is the only way
         * they can still reach the real entry points. */
        aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                  "%s: DetourDetach failed: %ld; hooks remain installed\n",
                  __func__, (long)status);
        return;
    }
    true_urUSMDeviceAlloc = NULL;
    true_urUSMFree = NULL;
    true_urPhysicalMemCreate = NULL;
    InterlockedExchange(&g_attached, 0);
}

/* The four entry points below carry the same names and meanings as the Linux
 * hook in src-xpu/ur-usm-hook.cpp, so comfy_aimdo/control.py drives both
 * platforms through one interface. On Linux "interposed" means the dynamic
 * linker bound the symbol to AIMDO; here it means Detours patched it. */
__declspec(dllexport) bool xpu_ur_hook_is_interposed(void) {
    if (InterlockedCompareExchange(&g_attached, 0, 0)) {
        return true;
    }
    return aimdo_xpu_ur_hook_install();
}

__declspec(dllexport) bool xpu_ur_hook_enable(void) {
    if (!xpu_ur_hook_is_interposed()) {
        return false;
    }
    EnterCriticalSection(&g_hook_lock);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        InterlockedIncrement64(&g_generation);
        clear_retry();
        InterlockedExchange(&g_enabled, 1);
    }
    LeaveCriticalSection(&g_hook_lock);
    return true;
}

/* Stop arbitrating.
 *
 * The Linux hook refuses while tracked segments are live, because there AIMDO
 * *is* Torch's allocator and a live segment means its own accounting would be
 * dropped. Windows is not symmetric: the hook only observes allocations that
 * PyTorch owns, and PyTorch legitimately retains some of them across
 * empty_cache(). Refusing here made control.deinit() raise in any process that
 * had ever allocated, so the table is dropped instead and the count reported.
 * The memory itself is unaffected: PyTorch frees it through the unhooked entry
 * point, and AIMDO's accounting is torn down immediately afterwards. */
__declspec(dllexport) bool xpu_ur_hook_disable(void) {
    LONG64 live;

    if (!InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
        return true;
    }
    EnterCriticalSection(&g_hook_lock);
    InterlockedExchange(&g_enabled, 0);
    InterlockedIncrement64(&g_generation);
    clear_retry();
    live = InterlockedExchange64(&g_allocation_count, 0);
    memset(g_allocations, 0, sizeof(g_allocations));
    LeaveCriticalSection(&g_hook_lock);

    if (live > 0) {
        aimdo_log(kAimdoUrLogInfo, __FILE__, __LINE__,
                  "%s: stopped arbitrating with %lld PyTorch segments still "
                  "live; they remain owned by PyTorch\n", __func__,
                  (long long)live);
    }
    return true;
}

__declspec(dllexport) bool xpu_ur_hook_get_stats(uint64_t *values,
                                                 size_t count) {
    size_t index;

    if (!values || count < (size_t)kHookStatCount) {
        return false;
    }
    for (index = 0; index < (size_t)kHookStatCount; ++index) {
        values[index] =
            (uint64_t)InterlockedCompareExchange64(&g_stats[index], 0, 0);
    }
    return true;
}

/* Caller classification cost, reported separately from the shared statistics
 * table so the Linux and Windows tables stay identical. Nanoseconds are
 * derived here because QueryPerformanceFrequency is fixed for the process. */
__declspec(dllexport) bool xpu_ur_hook_get_classify_timing(
    uint64_t *calls, uint64_t *nanoseconds, uint64_t *torch_hits) {
    LARGE_INTEGER frequency;
    LONG64 ticks;

    if (!calls || !nanoseconds || !torch_hits) {
        return false;
    }
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart == 0) {
        return false;
    }
    ticks = InterlockedCompareExchange64(&g_classify_ticks, 0, 0);
    *calls = (uint64_t)InterlockedCompareExchange64(&g_classify_calls, 0, 0);
    *torch_hits =
        (uint64_t)InterlockedCompareExchange64(&g_classify_torch_hits, 0, 0);
    *nanoseconds =
        (uint64_t)((double)ticks * 1e9 / (double)frequency.QuadPart);
    return true;
}

/* Total wall time spent inside the allocation hook, including the real driver
 * call it wraps. This is what decides whether a workload regression belongs to
 * the hook itself or to the consequences of its decisions, such as PyTorch
 * discarding its entire cache after a synthetic failure. */
__declspec(dllexport) bool xpu_ur_hook_get_hook_timing(uint64_t *calls,
                                                       uint64_t *nanoseconds) {
    LARGE_INTEGER frequency;
    LONG64 ticks;

    if (!calls || !nanoseconds) {
        return false;
    }
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart == 0) {
        return false;
    }
    ticks = InterlockedCompareExchange64(&g_hook_ticks, 0, 0);
    *calls = (uint64_t)InterlockedCompareExchange64(&g_hook_calls, 0, 0);
    *nanoseconds =
        (uint64_t)((double)ticks * 1e9 / (double)frequency.QuadPart);
    return true;
}

#if defined(__cplusplus)
}
#endif
