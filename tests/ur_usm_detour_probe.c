/* Windows Unified Runtime USM interception probe.
 *
 * This is the Windows counterpart of tests/ur_usm_interpose_probe.cpp from the
 * Linux dev/xpu-native-allocator-hook branch. It answers one question and
 * nothing else: can AIMDO intercept the same Unified Runtime USM entry points
 * on Windows that the Linux prototype interposes with LD_PRELOAD, and does
 * PyTorch's XPU caching allocator honour the synthetic-OOM retry contract that
 * the Linux design depends on?
 *
 * Windows has no LD_PRELOAD and PE has no symbol versioning, so the Linux
 * mechanism (a shared object exporting urUSMDeviceAlloc under
 * LIBUR_LOADER_0.12, resolved with dlvsym(RTLD_NEXT, ...)) has no direct
 * equivalent. It is also not needed. On Windows the SYCL runtime resolves the
 * flat loader exports by name:
 *
 *   sycl8.dll  --imports-->  ur_win_proxy_loader.dll!getPreloadedURLib()
 *              --GetProcAddress("urUSMDeviceAlloc")-->  ur_loader.dll
 *
 * so the function pointer SYCL holds points directly into ur_loader.dll's code
 * section. Patching that function's prologue with Microsoft Detours therefore
 * intercepts every caller regardless of when the pointer was captured, which
 * is why this probe does not have to be loaded before the process starts.
 *
 * The probe never evicts anything and never calls back into AIMDO. It counts,
 * classifies, and can inject one synthetic out-of-memory result.
 */

#include <windows.h>
#include <detours.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Minimal Unified Runtime declarations.
 *
 * ur_api.h is deliberately not included: the probe must be buildable with
 * plain cl.exe against whatever oneAPI happens to be installed, and it only
 * needs opaque handles plus three result codes. Values are taken from
 * ur_api.h of oneAPI 2025.3 and are part of the frozen UR ABI. */
typedef int ur_result_t;

#define UR_RESULT_SUCCESS 0
#define UR_RESULT_ERROR_OUT_OF_HOST_MEMORY 38
#define UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY 39

typedef void *ur_context_handle_t;
typedef void *ur_device_handle_t;
typedef void *ur_usm_pool_handle_t;
typedef struct ur_usm_desc_t ur_usm_desc_t;

typedef ur_result_t(__cdecl *PFN_urUSMDeviceAlloc)(
    ur_context_handle_t, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t, void **);
typedef ur_result_t(__cdecl *PFN_urUSMFree)(ur_context_handle_t, void *);
typedef void *(__cdecl *PFN_getPreloadedURLib)(void);

/* Expandable segments do not allocate USM. They reserve virtual address space
 * and back it with physical memory objects, so these two entry points are
 * hooked purely to record where those allocations go when
 * PYTORCH_ALLOC_CONF=expandable_segments:True is set. */
typedef void *ur_physical_mem_handle_t;
typedef struct ur_physical_mem_properties_t ur_physical_mem_properties_t;
typedef uint32_t ur_virtual_mem_access_flags_t;

typedef ur_result_t(__cdecl *PFN_urPhysicalMemCreate)(
    ur_context_handle_t, ur_device_handle_t, size_t,
    const ur_physical_mem_properties_t *, ur_physical_mem_handle_t *);
typedef ur_result_t(__cdecl *PFN_urVirtualMemMap)(ur_context_handle_t,
                                                  const void *, size_t,
                                                  ur_physical_mem_handle_t,
                                                  size_t,
                                                  ur_virtual_mem_access_flags_t);

enum ProbeStat {
    kAllocCalls,
    kAllocBytes,
    kFreeCalls,
    kFreeCallsTorchNative,
    kTorchNativeAllocCalls,
    kOtherAllocCalls,
    kUnknownCallerAllocCalls,
    kSyntheticOomCalls,
    kRetryAllocCalls,
    kFreesBetweenOomAndRetry,
    kBytesFreedBetweenOomAndRetry,
    kRealAllocCalls,
    kRealAllocFailures,
    kLastAllocSize,
    kPhysicalMemCreateCalls,
    kPhysicalMemCreateBytes,
    kVirtualMemMapCalls,
    kResolvedViaProxy,
    kProxyAgreesWithByName,
    kHooked,
    kProbeStatCount
};

static const char *const kStatNames[kProbeStatCount] = {
    "alloc_calls",
    "alloc_bytes",
    "free_calls",
    "free_calls_torch_native",
    "torch_native_alloc_calls",
    "other_alloc_calls",
    "unknown_caller_alloc_calls",
    "synthetic_oom_calls",
    "retry_alloc_calls",
    "frees_between_oom_and_retry",
    "bytes_freed_between_oom_and_retry",
    "real_alloc_calls",
    "real_alloc_failures",
    "last_alloc_size",
    "physical_mem_create_calls",
    "physical_mem_create_bytes",
    "virtual_mem_map_calls",
    "resolved_via_proxy",
    "proxy_agrees_with_byname",
    "hooked",
};

static PFN_urUSMDeviceAlloc true_urUSMDeviceAlloc;
static PFN_urUSMFree true_urUSMFree;
static PFN_urPhysicalMemCreate true_urPhysicalMemCreate;
static PFN_urVirtualMemMap true_urVirtualMemMap;

static volatile LONG64 g_stats[kProbeStatCount];
static volatile LONG g_installed;
static volatile LONG g_trace;

/* Synthetic OOM control. Armed for one exact request size; fires once. */
static volatile LONG64 g_armed_size;
static volatile LONG g_armed_torch_only = 1;

/* Retry tracking is per thread, matching the Linux prototype: PyTorch retries
 * the failed allocation on the same thread that observed the failure. */
static __declspec(thread) size_t t_retry_size;
static __declspec(thread) LONG64 t_retry_free_calls;
static __declspec(thread) LONG64 t_retry_free_bytes;

static CRITICAL_SECTION g_stack_lock;
static char g_last_stack[2048];
static char g_module_path[MAX_PATH];

/* Handles captured from a real PyTorch allocation. They let the probe replay a
 * device allocation through the public entry point from its own module, which
 * is the only way to produce a caller that is genuinely not the Torch caching
 * allocator without building a separate SYCL program. */
static HMODULE g_loader_module;
static ur_context_handle_t g_seen_context;
static ur_device_handle_t g_seen_device;
static volatile LONG g_seen_handles;

/* Allocation sizes are recorded so the free path can report the bytes PyTorch
 * actually returned to the driver between the synthetic failure and its retry.
 * A tiny open-addressed table avoids pulling in a C++ container. Deleted slots
 * must be tombstoned rather than cleared: USM device allocations of the same
 * size are highly aligned, so they collide heavily and clearing a slot would
 * truncate the probe chain of every pointer behind it. */
#define POINTER_TABLE_SIZE 65536
#define POINTER_TOMBSTONE ((void *)(uintptr_t)1)
static struct {
    void *pointer;
    size_t size;
} g_pointers[POINTER_TABLE_SIZE];
static CRITICAL_SECTION g_pointer_lock;

static size_t pointer_slot(void *pointer) {
    uint64_t value = (uint64_t)(uintptr_t)pointer;

    /* Take the high bits of the product. Multiplication only propagates bits
     * upwards, so masking the low bits of the product would ignore everything
     * above bit 21 of the pointer and hash every 2 MiB-aligned allocation to
     * the same slot. */
    value ^= value >> 33;
    value *= 11400714819323198485ull;
    return (size_t)((value >> 48) & (POINTER_TABLE_SIZE - 1));
}

static void pointer_insert(void *pointer, size_t size) {
    size_t slot = pointer_slot(pointer);
    size_t probe;

    EnterCriticalSection(&g_pointer_lock);
    for (probe = 0; probe < POINTER_TABLE_SIZE; ++probe) {
        size_t index = (slot + probe) & (POINTER_TABLE_SIZE - 1);

        if (!g_pointers[index].pointer ||
            g_pointers[index].pointer == POINTER_TOMBSTONE ||
            g_pointers[index].pointer == pointer) {
            g_pointers[index].pointer = pointer;
            g_pointers[index].size = size;
            break;
        }
    }
    LeaveCriticalSection(&g_pointer_lock);
}

static size_t pointer_remove(void *pointer) {
    size_t slot = pointer_slot(pointer);
    size_t probe;
    size_t size = 0;

    EnterCriticalSection(&g_pointer_lock);
    for (probe = 0; probe < POINTER_TABLE_SIZE; ++probe) {
        size_t index = (slot + probe) & (POINTER_TABLE_SIZE - 1);

        if (!g_pointers[index].pointer) {
            break;
        }
        if (g_pointers[index].pointer == pointer) {
            size = g_pointers[index].size;
            g_pointers[index].pointer = POINTER_TOMBSTONE;
            g_pointers[index].size = 0;
            break;
        }
    }
    LeaveCriticalSection(&g_pointer_lock);
    return size;
}

static void stat_add(enum ProbeStat stat, LONG64 delta) {
    InterlockedAdd64(&g_stats[stat], delta);
}

static void stat_set(enum ProbeStat stat, LONG64 value) {
    InterlockedExchange64(&g_stats[stat], value);
}

static const wchar_t *base_name(const wchar_t *path) {
    const wchar_t *slash = wcsrchr(path, L'\\');

    return slash ? slash + 1 : path;
}

/* Caller classification.
 *
 * The Linux prototype walks the stack with backtrace()/dladdr() and looks for
 * libc10_xpu.so. RtlCaptureStackBackTrace plus GetModuleHandleExW with
 * GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS is the documented Windows equivalent;
 * it uses the x64 unwind tables rather than frame pointers, so it does not
 * depend on the optimisation settings of the modules being walked. */
static int classify_caller(char *trace, size_t trace_size) {
    void *frames[62];
    USHORT count;
    USHORT index;
    int is_torch_native = 0;
    size_t used = 0;

    count = RtlCaptureStackBackTrace(1, ARRAYSIZE(frames), frames, NULL);
    for (index = 0; index < count; ++index) {
        HMODULE module = NULL;
        wchar_t path[MAX_PATH];
        const wchar_t *name;

        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR)frames[index], &module)) {
            continue;
        }
        if (!GetModuleFileNameW(module, path, ARRAYSIZE(path))) {
            continue;
        }
        name = base_name(path);
        if (_wcsicmp(name, L"c10_xpu.dll") == 0) {
            is_torch_native = 1;
        }
        if (trace && used + 1 < trace_size) {
            int written = _snprintf_s(trace + used, trace_size - used, _TRUNCATE,
                                      "%s%ls", used ? " <- " : "", name);

            if (written > 0) {
                used += (size_t)written;
            }
        }
    }
    return is_torch_native;
}

static ur_result_t __cdecl probe_urUSMDeviceAlloc(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer) {
    char trace[1024];
    int is_torch_native;
    LONG64 armed;
    ur_result_t result;

    trace[0] = '\0';
    is_torch_native = classify_caller(trace, sizeof(trace));

    stat_add(kAllocCalls, 1);
    stat_add(kAllocBytes, (LONG64)size);
    stat_set(kLastAllocSize, (LONG64)size);
    if (is_torch_native) {
        stat_add(kTorchNativeAllocCalls, 1);
    } else {
        stat_add(kOtherAllocCalls, 1);
    }
    if (trace[0] == '\0') {
        stat_add(kUnknownCallerAllocCalls, 1);
    }

    EnterCriticalSection(&g_stack_lock);
    strncpy_s(g_last_stack, sizeof(g_last_stack), trace, _TRUNCATE);
    LeaveCriticalSection(&g_stack_lock);

    if (InterlockedCompareExchange(&g_trace, 0, 0)) {
        fprintf(stderr, "[UR PROBE] alloc size=%zu torch_native=%d stack=%s\n",
                size, is_torch_native, trace);
        fflush(stderr);
    }

    /* Retry detection. PyTorch's XPUCachingAllocator::malloc reacts to a failed
     * alloc_block by calling release_cached_blocks() and then retrying the very
     * same request. Both halves of that contract are measured here: the frees
     * observed in between prove the cache was actually returned to the driver,
     * and the matching request proves the retry happened. */
    if (t_retry_size != 0 && t_retry_size == size) {
        LONG64 frees = InterlockedCompareExchange64(&g_stats[kFreeCalls], 0, 0);
        LONG64 bytes =
            InterlockedCompareExchange64(&g_stats[kBytesFreedBetweenOomAndRetry], 0, 0);

        (void)bytes;
        stat_add(kRetryAllocCalls, 1);
        stat_set(kFreesBetweenOomAndRetry, frees - t_retry_free_calls);
        t_retry_size = 0;
        if (InterlockedCompareExchange(&g_trace, 0, 0)) {
            fprintf(stderr, "[UR PROBE] retry observed size=%zu frees_between=%lld\n",
                    size, (long long)(frees - t_retry_free_calls));
            fflush(stderr);
        }
    }

    armed = InterlockedCompareExchange64(&g_armed_size, 0, 0);
    if (armed != 0 && (size_t)armed == size &&
        (is_torch_native || !InterlockedCompareExchange(&g_armed_torch_only, 0, 0))) {
        if (InterlockedCompareExchange64(&g_armed_size, 0, armed) == armed) {
            t_retry_size = size;
            t_retry_free_calls =
                InterlockedCompareExchange64(&g_stats[kFreeCalls], 0, 0);
            t_retry_free_bytes = 0;
            stat_add(kSyntheticOomCalls, 1);
            stat_set(kBytesFreedBetweenOomAndRetry, 0);
            if (pointer) {
                *pointer = NULL;
            }
            fprintf(stderr, "[UR PROBE] synthetic_oom size=%zu torch_native=%d\n",
                    size, is_torch_native);
            fflush(stderr);
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    stat_add(kRealAllocCalls, 1);
    result = true_urUSMDeviceAlloc(context, device, description, pool, size,
                                   pointer);
    if (result != UR_RESULT_SUCCESS) {
        stat_add(kRealAllocFailures, 1);
        return result;
    }
    if (pointer && *pointer) {
        pointer_insert(*pointer, size);
    }
    if (is_torch_native && !InterlockedCompareExchange(&g_seen_handles, 0, 0)) {
        g_seen_context = context;
        g_seen_device = device;
        InterlockedExchange(&g_seen_handles, 1);
    }
    return result;
}

static ur_result_t __cdecl probe_urUSMFree(ur_context_handle_t context,
                                           void *pointer) {
    size_t size = pointer ? pointer_remove(pointer) : 0;
    ur_result_t result;

    stat_add(kFreeCalls, 1);
    if (t_retry_size != 0) {
        /* Frees seen between the synthetic failure and the retry are exactly
         * PyTorch's release_cached_blocks() returning its cache. */
        stat_add(kBytesFreedBetweenOomAndRetry, (LONG64)size);
        t_retry_free_bytes += (LONG64)size;
    }
    if (classify_caller(NULL, 0)) {
        stat_add(kFreeCallsTorchNative, 1);
    }
    if (InterlockedCompareExchange(&g_trace, 0, 0)) {
        fprintf(stderr, "[UR PROBE] free size=%zu\n", size);
        fflush(stderr);
    }
    result = true_urUSMFree(context, pointer);
    return result;
}

typedef BOOL(WINAPI *PFN_EnumProcessModules)(HANDLE, HMODULE *, DWORD, LPDWORD);

static ur_result_t __cdecl probe_urPhysicalMemCreate(
    ur_context_handle_t context, ur_device_handle_t device, size_t size,
    const ur_physical_mem_properties_t *properties,
    ur_physical_mem_handle_t *handle) {
    stat_add(kPhysicalMemCreateCalls, 1);
    stat_add(kPhysicalMemCreateBytes, (LONG64)size);
    if (InterlockedCompareExchange(&g_trace, 0, 0)) {
        fprintf(stderr, "[UR PROBE] physical_mem_create size=%zu\n", size);
        fflush(stderr);
    }
    return true_urPhysicalMemCreate(context, device, size, properties, handle);
}

static ur_result_t __cdecl probe_urVirtualMemMap(
    ur_context_handle_t context, const void *start, size_t size,
    ur_physical_mem_handle_t physical, size_t offset,
    ur_virtual_mem_access_flags_t flags) {
    stat_add(kVirtualMemMapCalls, 1);
    return true_urVirtualMemMap(context, start, size, physical, offset, flags);
}

/* GetModuleHandleA("ur_win_proxy_loader.dll") does not resolve in a live
 * PyTorch XPU process even though the module is loaded, so the proxy has to be
 * found by walking the module list. The proxy is worth finding: it returns the
 * exact ur_loader.dll handle the SYCL runtime itself uses, which is the only
 * way to be certain the right module was hooked when several copies of
 * ur_loader.dll could be present in one process image. */
static HMODULE find_proxy_loader(void) {
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    PFN_EnumProcessModules enumerate;
    HMODULE modules[1024];
    DWORD needed = 0;
    DWORD count;
    DWORD index;

    if (!kernel) {
        return NULL;
    }
    enumerate =
        (PFN_EnumProcessModules)(void *)GetProcAddress(kernel, "K32EnumProcessModules");
    if (!enumerate) {
        return NULL;
    }
    if (!enumerate(GetCurrentProcess(), modules, sizeof(modules), &needed)) {
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
        if (_wcsicmp(base_name(path), L"ur_win_proxy_loader.dll") == 0) {
            return modules[index];
        }
    }
    return NULL;
}

static HMODULE resolve_ur_loader(int *via_proxy, int *agrees) {
    HMODULE proxy;
    HMODULE by_name;
    HMODULE from_proxy = NULL;

    *via_proxy = 0;
    *agrees = 0;

    by_name = GetModuleHandleA("ur_loader.dll");

    proxy = find_proxy_loader();
    if (proxy) {
        PFN_getPreloadedURLib get_lib = (PFN_getPreloadedURLib)(void *)
            GetProcAddress(proxy, "?getPreloadedURLib@@YAPEAXXZ");

        if (get_lib) {
            from_proxy = (HMODULE)get_lib();
        }
    }
    if (from_proxy) {
        *via_proxy = 1;
        *agrees = (by_name == from_proxy) ? 1 : 0;
        return from_proxy;
    }
    return by_name;
}

__declspec(dllexport) int probe_install(void) {
    HMODULE loader;
    int via_proxy = 0;
    int agrees = 0;
    LONG status;
    wchar_t path[MAX_PATH];

    if (InterlockedCompareExchange(&g_installed, 0, 0)) {
        return 1;
    }

    loader = resolve_ur_loader(&via_proxy, &agrees);
    if (!loader) {
        fprintf(stderr, "[UR PROBE] ur_loader.dll is not loaded in this process\n");
        return 0;
    }
    if (GetModuleFileNameW(loader, path, ARRAYSIZE(path))) {
        _snprintf_s(g_module_path, sizeof(g_module_path), _TRUNCATE, "%ls", path);
    }
    g_loader_module = loader;

    true_urUSMDeviceAlloc =
        (PFN_urUSMDeviceAlloc)(void *)GetProcAddress(loader, "urUSMDeviceAlloc");
    true_urUSMFree = (PFN_urUSMFree)(void *)GetProcAddress(loader, "urUSMFree");
    true_urPhysicalMemCreate = (PFN_urPhysicalMemCreate)(void *)GetProcAddress(
        loader, "urPhysicalMemCreate");
    true_urVirtualMemMap =
        (PFN_urVirtualMemMap)(void *)GetProcAddress(loader, "urVirtualMemMap");
    if (!true_urUSMDeviceAlloc || !true_urUSMFree) {
        fprintf(stderr, "[UR PROBE] flat USM exports were not found\n");
        return 0;
    }

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    status = DetourAttach((void **)&true_urUSMDeviceAlloc, probe_urUSMDeviceAlloc);
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&true_urUSMFree, probe_urUSMFree);
    }
    if (status == NO_ERROR && true_urPhysicalMemCreate) {
        status = DetourAttach((void **)&true_urPhysicalMemCreate,
                              probe_urPhysicalMemCreate);
    }
    if (status == NO_ERROR && true_urVirtualMemMap) {
        status =
            DetourAttach((void **)&true_urVirtualMemMap, probe_urVirtualMemMap);
    }
    if (status != NO_ERROR) {
        DetourTransactionAbort();
        fprintf(stderr, "[UR PROBE] DetourAttach failed: %ld\n", (long)status);
        return 0;
    }
    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        fprintf(stderr, "[UR PROBE] DetourTransactionCommit failed: %ld\n",
                (long)status);
        return 0;
    }

    stat_set(kResolvedViaProxy, via_proxy);
    stat_set(kProxyAgreesWithByName, agrees);
    stat_set(kHooked, 1);
    InterlockedExchange(&g_installed, 1);
    return 1;
}

__declspec(dllexport) int probe_remove(void) {
    LONG status;

    if (!InterlockedCompareExchange(&g_installed, 0, 0)) {
        return 1;
    }
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourDetach((void **)&true_urUSMDeviceAlloc, probe_urUSMDeviceAlloc);
    DetourDetach((void **)&true_urUSMFree, probe_urUSMFree);
    if (true_urPhysicalMemCreate) {
        DetourDetach((void **)&true_urPhysicalMemCreate,
                     probe_urPhysicalMemCreate);
    }
    if (true_urVirtualMemMap) {
        DetourDetach((void **)&true_urVirtualMemMap, probe_urVirtualMemMap);
    }
    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        fprintf(stderr, "[UR PROBE] DetourDetach failed: %ld\n", (long)status);
        return 0;
    }
    InterlockedExchange(&g_installed, 0);
    stat_set(kHooked, 0);
    return 1;
}

__declspec(dllexport) int probe_get_stats(unsigned long long *values,
                                          size_t count) {
    size_t index;

    if (!values || count < (size_t)kProbeStatCount) {
        return 0;
    }
    for (index = 0; index < (size_t)kProbeStatCount; ++index) {
        values[index] =
            (unsigned long long)InterlockedCompareExchange64(&g_stats[index], 0, 0);
    }
    return 1;
}

__declspec(dllexport) size_t probe_stat_count(void) {
    return (size_t)kProbeStatCount;
}

__declspec(dllexport) const char *probe_stat_name(size_t index) {
    return index < (size_t)kProbeStatCount ? kStatNames[index] : NULL;
}

__declspec(dllexport) void probe_arm_synthetic_oom(unsigned long long size,
                                                   int torch_only) {
    InterlockedExchange(&g_armed_torch_only, torch_only ? 1 : 0);
    InterlockedExchange64(&g_armed_size, (LONG64)size);
}

__declspec(dllexport) void probe_set_trace(int enabled) {
    InterlockedExchange(&g_trace, enabled ? 1 : 0);
}

__declspec(dllexport) const char *probe_last_stack(void) {
    return g_last_stack;
}

__declspec(dllexport) const char *probe_module_path(void) {
    return g_module_path;
}

/* Replay an allocation through the public, patched entry point using handles
 * captured from PyTorch. The call originates in this module, so the stack walk
 * must not find c10_xpu.dll and the request must be classified as "other".
 * This is the negative control for the caller classification. */
__declspec(dllexport) int probe_direct_alloc(unsigned long long size,
                                             void **pointer) {
    PFN_urUSMDeviceAlloc entry;

    if (!InterlockedCompareExchange(&g_seen_handles, 0, 0) || !g_loader_module) {
        return -1;
    }
    entry = (PFN_urUSMDeviceAlloc)(void *)GetProcAddress(g_loader_module,
                                                         "urUSMDeviceAlloc");
    if (!entry) {
        return -2;
    }
    return entry(g_seen_context, g_seen_device, NULL, NULL, (size_t)size, pointer);
}

__declspec(dllexport) int probe_direct_free(void *pointer) {
    PFN_urUSMFree entry;

    if (!InterlockedCompareExchange(&g_seen_handles, 0, 0) || !g_loader_module) {
        return -1;
    }
    entry = (PFN_urUSMFree)(void *)GetProcAddress(g_loader_module, "urUSMFree");
    if (!entry) {
        return -2;
    }
    return entry(g_seen_context, pointer);
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance;
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        InitializeCriticalSection(&g_stack_lock);
        InitializeCriticalSection(&g_pointer_lock);
    } else if (reason == DLL_PROCESS_DETACH) {
        DeleteCriticalSection(&g_stack_lock);
        DeleteCriticalSection(&g_pointer_lock);
    }
    return TRUE;
}
