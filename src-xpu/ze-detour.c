/* Windows Level Zero allocation interception for AIMDO.
 *
 * This mirrors the production CUDA design in src-win/cuda-detour.c: PyTorch
 * keeps its own native caching allocator, with all of its block splitting,
 * coalescing, stream ordering and empty_cache() behaviour, and AIMDO instead
 * intercepts the driver entry point that actually grows physical device
 * memory. Replacing Torch's allocator was tried and rejected: it degraded
 * block management to an exact-size cache, and its release path had to call
 * sycl::free(), which is itself known to stall inside the Level Zero/UMF
 * residency path under WDDM pressure.
 *
 * The Level Zero tracing layer can observe the same calls and is retained for
 * diagnosis, but it is process-wide and measurably slows long command streams,
 * so it is no longer the default.
 *
 * Everything reached from these hooks must be non-blocking. A hook body runs
 * inside the driver's allocation call, and the compute queue it would wait on
 * may be blocked behind work that needs the very residency being requested.
 * aimdo_xpu_prepare_allocation() therefore reclaims only VBAR pages that are
 * provably retired and returns immediately otherwise.
 */

#include <ze_api.h>

#include <windows.h>
#include <detours.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

enum AimdoDetourLogLevel {
    kAimdoDetourLogError = 2,
    kAimdoDetourLogWarning = 3,
    kAimdoDetourLogInfo = 4,
    kAimdoDetourLogDebug = 5,
};

extern void aimdo_log(
    int level, const char *file, int line, const char *format, ...);
extern int aimdo_xpu_device_index_from_native(void *native_device);
extern bool aimdo_xpu_sample_pressure(int device, size_t size);
extern bool aimdo_xpu_prepare_allocation(int device, size_t size);
extern bool aimdo_xpu_retry_allocation(int device, size_t size);
extern void aimdo_xpu_note_native_allocation(void *ptr, size_t size, int device);
extern void aimdo_xpu_note_native_release(void *ptr);
extern bool aimdo_xpu_native_accounting_init(void);
extern void aimdo_xpu_native_accounting_cleanup(void);
extern bool aimdo_xpu_tracer_install(void);
extern void aimdo_xpu_tracer_remove(void);
extern bool aimdo_xpu_ur_hook_install(void);
extern void aimdo_xpu_ur_hook_remove(void);

typedef ze_result_t (ZE_APICALL *PFN_zeMemAllocDevice)(
    ze_context_handle_t, const ze_device_mem_alloc_desc_t *, size_t, size_t,
    ze_device_handle_t, void **);
typedef ze_result_t (ZE_APICALL *PFN_zeMemFree)(ze_context_handle_t, void *);
typedef ze_result_t (ZE_APICALL *PFN_zeMemFreeExt)(
    ze_context_handle_t, const ze_memory_free_ext_desc_t *, void *);

static PFN_zeMemAllocDevice true_zeMemAllocDevice;
static PFN_zeMemFree true_zeMemFree;
static PFN_zeMemFreeExt true_zeMemFreeExt;

static bool g_hooks_installed;
static bool g_tracer_owns_hooks;
static bool g_ur_hook_owns_arbitration;

static bool env_flag_enabled(const char *name) {
    char value[8];
    DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));

    return length > 0 && length < sizeof(value) && value[0] == '1';
}

static ze_result_t ZE_APICALL aimdo_zeMemAllocDevice(
    ze_context_handle_t context, const ze_device_mem_alloc_desc_t *descriptor,
    size_t size, size_t alignment, ze_device_handle_t device_handle,
    void **pointer) {
    ze_result_t result;
    int device;

    if (!true_zeMemAllocDevice) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }

    device = aimdo_xpu_device_index_from_native((void *)device_handle);
    if (device < 0) {
        /* Not a device AIMDO manages: stay completely out of the way. */
        return true_zeMemAllocDevice(context, descriptor, size, alignment,
                                     device_handle, pointer);
    }

    if (g_ur_hook_owns_arbitration) {
        /* The Unified Runtime hook already applied the AIMDO budget to this
         * request one layer above, before the driver was entered. Running a
         * second control loop here would sample and reclaim against pressure
         * that has just been arbitrated, so this path only accounts. */
        result = true_zeMemAllocDevice(context, descriptor, size, alignment,
                                       device_handle, pointer);
        if (result == ZE_RESULT_SUCCESS && pointer && *pointer) {
            aimdo_xpu_note_native_allocation(*pointer, size, device);
        }
        return result;
    }

    /* Sampling pressure before the allocation is read-only and safe. Reclaim
     * is not: releasing a VBAR page calls zeVirtualMemUnmap and
     * zePhysicalMemDestroy, and issuing those while the driver is inside
     * zeMemAllocDevice re-enters Level Zero's own memory management. That was
     * observed to corrupt driver state and surface later as
     * UR_RESULT_ERROR_DEVICE_LOST, with progressive slowdown beforehand.
     *
     * Nothing requires reclaim to complete first: a Windows device allocation
     * does not fail, WDDM demotes the excess instead. Reclaiming immediately
     * after the driver call returns keeps steady-state usage bounded just as
     * well, without ever re-entering the driver. */
    aimdo_xpu_sample_pressure(device, size);

    result = true_zeMemAllocDevice(context, descriptor, size, alignment,
                                   device_handle, pointer);

    if (result == ZE_RESULT_SUCCESS && pointer && *pointer) {
        aimdo_xpu_note_native_allocation(*pointer, size, device);
        aimdo_xpu_prepare_allocation(device, 0);
        return result;
    }

    if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
        result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
        /* WDDM normally demotes an over-budget allocation instead of failing,
         * so reaching this point means the driver could not place the request
         * at all. The driver call has returned, so reclaiming here is not
         * re-entrant. */
        aimdo_xpu_retry_allocation(device, size);
        result = true_zeMemAllocDevice(context, descriptor, size, alignment,
                                       device_handle, pointer);
        if (result == ZE_RESULT_SUCCESS && pointer && *pointer) {
            aimdo_xpu_note_native_allocation(*pointer, size, device);
        }
    }
    return result;
}

static ze_result_t ZE_APICALL aimdo_zeMemFree(
    ze_context_handle_t context, void *pointer) {
    ze_result_t result;

    if (!true_zeMemFree) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    result = true_zeMemFree(context, pointer);
    if (result == ZE_RESULT_SUCCESS) {
        aimdo_xpu_note_native_release(pointer);
    }
    return result;
}

static ze_result_t ZE_APICALL aimdo_zeMemFreeExt(
    ze_context_handle_t context, const ze_memory_free_ext_desc_t *descriptor,
    void *pointer) {
    ze_result_t result;

    if (!true_zeMemFreeExt) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    result = true_zeMemFreeExt(context, descriptor, pointer);
    if (result == ZE_RESULT_SUCCESS) {
        aimdo_xpu_note_native_release(pointer);
    }
    return result;
}

static void *resolve_loader_entry(const char *name) {
    HMODULE loader = GetModuleHandleA("ze_loader.dll");

    if (!loader) {
        loader = LoadLibraryA("ze_loader.dll");
    }
    return loader ? (void *)GetProcAddress(loader, name) : NULL;
}

static bool install_detours(void) {
    LONG status;

    true_zeMemAllocDevice =
        (PFN_zeMemAllocDevice)resolve_loader_entry("zeMemAllocDevice");
    true_zeMemFree = (PFN_zeMemFree)resolve_loader_entry("zeMemFree");
    true_zeMemFreeExt = (PFN_zeMemFreeExt)resolve_loader_entry("zeMemFreeExt");

    if (!true_zeMemAllocDevice || !true_zeMemFree) {
        aimdo_log(kAimdoDetourLogError, __FILE__, __LINE__,
                  "%s: could not resolve Level Zero allocation entry points\n",
                  __func__);
        return false;
    }

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    status = DetourAttach((void **)&true_zeMemAllocDevice, aimdo_zeMemAllocDevice);
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&true_zeMemFree, aimdo_zeMemFree);
    }
    if (status == NO_ERROR && true_zeMemFreeExt) {
        status = DetourAttach((void **)&true_zeMemFreeExt, aimdo_zeMemFreeExt);
    }
    if (status != NO_ERROR) {
        aimdo_log(kAimdoDetourLogError, __FILE__, __LINE__,
                  "%s: DetourAttach failed: %ld\n", __func__, (long)status);
        DetourTransactionAbort();
        true_zeMemAllocDevice = NULL;
        true_zeMemFree = NULL;
        true_zeMemFreeExt = NULL;
        return false;
    }

    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        aimdo_log(kAimdoDetourLogError, __FILE__, __LINE__,
                  "%s: DetourTransactionCommit failed: %ld\n", __func__,
                  (long)status);
        true_zeMemAllocDevice = NULL;
        true_zeMemFree = NULL;
        true_zeMemFreeExt = NULL;
        return false;
    }
    return true;
}

static void remove_detours(void) {
    LONG status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    if (true_zeMemAllocDevice) {
        DetourDetach((void **)&true_zeMemAllocDevice, aimdo_zeMemAllocDevice);
    }
    if (true_zeMemFree) {
        DetourDetach((void **)&true_zeMemFree, aimdo_zeMemFree);
    }
    if (true_zeMemFreeExt) {
        DetourDetach((void **)&true_zeMemFreeExt, aimdo_zeMemFreeExt);
    }
    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        /* The hooks are still live. Keeping the trampolines is the only way
         * they can still reach the real driver entry points. */
        aimdo_log(kAimdoDetourLogError, __FILE__, __LINE__,
                  "%s: DetourDetach failed: %ld; hooks remain installed\n",
                  __func__, (long)status);
        return;
    }
    true_zeMemAllocDevice = NULL;
    true_zeMemFree = NULL;
    true_zeMemFreeExt = NULL;
}

bool aimdo_setup_hooks(void) {
    if (g_hooks_installed) {
        return true;
    }

    if (env_flag_enabled("AIMDO_XPU_DISABLE_ALLOCATION_HOOKS")) {
        aimdo_log(kAimdoDetourLogWarning, __FILE__, __LINE__,
                  "%s: allocation interception disabled by request; AIMDO "
                  "cannot arbitrate native Torch allocations\n", __func__);
        g_hooks_installed = true;
        return true;
    }

    if (env_flag_enabled("AIMDO_XPU_ENABLE_ALLOCATION_TRACING")) {
        /* Diagnostic path. The Level Zero tracing layer is process-wide and
         * progressively slows long command streams; do not use it for
         * performance measurement. */
        if (!aimdo_xpu_tracer_install()) {
            return false;
        }
        g_tracer_owns_hooks = true;
        g_hooks_installed = true;
        return true;
    }

    aimdo_xpu_native_accounting_init();
    if (!install_detours()) {
        aimdo_xpu_native_accounting_cleanup();
        return false;
    }

    /* Preferred arbitration point. It sits above the Level Zero driver, so it
     * can decide before the allocation is placed without re-entering Level
     * Zero memory management from inside its own allocation call. Failing to
     * attach is not fatal: the Level Zero detour above remains as the
     * post-allocation fallback. */
    if (env_flag_enabled("AIMDO_XPU_DISABLE_UR_HOOK")) {
        aimdo_log(kAimdoDetourLogWarning, __FILE__, __LINE__,
                  "%s: Unified Runtime arbitration disabled by request; using "
                  "post-allocation Level Zero reclaim\n", __func__);
    } else {
        g_ur_hook_owns_arbitration = aimdo_xpu_ur_hook_install();
        if (!g_ur_hook_owns_arbitration) {
            aimdo_log(kAimdoDetourLogWarning, __FILE__, __LINE__,
                      "%s: Unified Runtime arbitration unavailable; falling "
                      "back to post-allocation Level Zero reclaim\n", __func__);
        }
    }

    aimdo_log(kAimdoDetourLogInfo, __FILE__, __LINE__,
              "%s: native Torch XPU allocator retained; arbitrating %s\n",
              __func__,
              g_ur_hook_owns_arbitration ? "Unified Runtime USM allocations"
                                         : "Level Zero physical allocations");
    g_hooks_installed = true;
    return true;
}

void aimdo_teardown_hooks(void) {
    if (!g_hooks_installed) {
        return;
    }
    if (g_tracer_owns_hooks) {
        aimdo_xpu_tracer_remove();
        g_tracer_owns_hooks = false;
        g_hooks_installed = false;
        return;
    }
    if (true_zeMemAllocDevice) {
        remove_detours();
        aimdo_xpu_native_accounting_cleanup();
    }
    if (g_ur_hook_owns_arbitration) {
        aimdo_xpu_ur_hook_remove();
        g_ur_hook_owns_arbitration = false;
    }
    g_hooks_installed = false;
}

#if defined(__cplusplus)
}
#endif
