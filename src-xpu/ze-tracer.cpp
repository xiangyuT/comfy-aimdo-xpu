#include <ze_api.h>
#include <layers/zel_tracing_api.h>
#include <layers/zel_tracing_register_cb.h>
#include <loader/ze_loader.h>

#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define XPU_ALLOCATION_HASH_SIZE 4096

enum AimdoTracerLogLevel {
    kAimdoTracerLogError = 2,
    kAimdoTracerLogInfo = 4,
};

typedef struct XpuNativeAllocation {
    void *ptr;
    size_t size;
    int device;
    struct XpuNativeAllocation *next;
} XpuNativeAllocation;

static zel_tracer_handle_t g_xpu_tracer;
static CRITICAL_SECTION g_xpu_allocation_lock;
static bool g_xpu_allocation_lock_initialized;
static XpuNativeAllocation *g_xpu_allocations[XPU_ALLOCATION_HASH_SIZE];
static int g_xpu_tracer_user_data;

static bool xpu_allocation_trace_enabled(void) {
    static int enabled = -1;
    char value[2];

    if (enabled < 0) {
        enabled = GetEnvironmentVariableA(
            "AIMDO_XPU_ALLOCATION_TRACE", value, sizeof(value)) > 0 &&
            value[0] == '1';
    }
    return enabled != 0;
}

extern void aimdo_log(
    int level, const char *file, int line, const char *format, ...);
extern int aimdo_xpu_device_index_from_native(void *native_device);
extern bool aimdo_xpu_sample_pressure(int device, size_t size);
extern bool aimdo_xpu_prepare_allocation(int device, size_t size);
extern bool aimdo_xpu_retry_allocation(int device, size_t size);
extern bool aimdo_xpu_account_allocation(int device, int64_t delta);
extern void aimdo_xpu_record_native_allocation(size_t size);
extern void aimdo_xpu_record_native_release(size_t size);

static size_t xpu_allocation_hash(void *ptr) {
    uintptr_t value = (uintptr_t)ptr;
    return ((value >> 12) ^ (value >> 25)) % XPU_ALLOCATION_HASH_SIZE;
}

void aimdo_xpu_note_native_allocation(void *ptr, size_t size, int device) {
    XpuNativeAllocation *entry;
    size_t bucket;

    if (!ptr || size == 0 || device < 0 || !g_xpu_allocation_lock_initialized) {
        return;
    }
    entry = (XpuNativeAllocation *)malloc(sizeof(*entry));
    if (!entry) {
        return;
    }
    entry->ptr = ptr;
    entry->size = size;
    entry->device = device;
    bucket = xpu_allocation_hash(ptr);

    EnterCriticalSection(&g_xpu_allocation_lock);
    entry->next = g_xpu_allocations[bucket];
    g_xpu_allocations[bucket] = entry;
    LeaveCriticalSection(&g_xpu_allocation_lock);

    aimdo_xpu_account_allocation(device, (int64_t)size);
    aimdo_xpu_record_native_allocation(size);
}

void aimdo_xpu_note_native_release(void *ptr) {
    XpuNativeAllocation **previous;
    XpuNativeAllocation *entry;
    size_t bucket;

    if (!ptr || !g_xpu_allocation_lock_initialized) {
        return;
    }
    bucket = xpu_allocation_hash(ptr);
    EnterCriticalSection(&g_xpu_allocation_lock);
    previous = &g_xpu_allocations[bucket];
    entry = *previous;
    while (entry && entry->ptr != ptr) {
        previous = &entry->next;
        entry = entry->next;
    }
    if (entry) {
        *previous = entry->next;
    }
    LeaveCriticalSection(&g_xpu_allocation_lock);

    if (!entry) {
        return;
    }
    aimdo_xpu_account_allocation(entry->device, -(int64_t)entry->size);
    aimdo_xpu_record_native_release(entry->size);
    free(entry);
}

static void ZE_APICALL xpu_mem_alloc_device_prologue(
    ze_mem_alloc_device_params_t *params, ze_result_t result,
    void *tracer_user_data, void **instance_user_data) {
    int device;

    (void)result;
    (void)tracer_user_data;
    if (!params || !params->phDevice || !params->psize ||
        !instance_user_data) {
        return;
    }
    device = aimdo_xpu_device_index_from_native(
        (void *)*params->phDevice);
    if (xpu_allocation_trace_enabled()) {
        aimdo_log(kAimdoTracerLogInfo, __FILE__, __LINE__,
                  "AIMDO XPU allocation begin device=%d size=%zu\n",
                  device, *params->psize);
    }
    if (device < 0 ||
        !aimdo_xpu_sample_pressure(device, *params->psize)) {
        return;
    }
    if (xpu_allocation_trace_enabled()) {
        aimdo_log(kAimdoTracerLogInfo, __FILE__, __LINE__,
                  "AIMDO XPU allocation dispatch device=%d size=%zu\n",
                  device, *params->psize);
    }
    *instance_user_data = (void *)(intptr_t)(device + 1);
}

static void ZE_APICALL xpu_mem_alloc_device_epilogue(
    ze_mem_alloc_device_params_t *params, ze_result_t result,
    void *tracer_user_data, void **instance_user_data) {
    int device;

    (void)tracer_user_data;
    if (!params || !params->psize || !params->ppptr ||
        !instance_user_data || !*instance_user_data) {
        return;
    }
    device = (int)(intptr_t)*instance_user_data - 1;
    if (xpu_allocation_trace_enabled()) {
        aimdo_log(kAimdoTracerLogInfo, __FILE__, __LINE__,
                  "AIMDO XPU allocation end device=%d size=%zu result=%d ptr=%p\n",
                  device, *params->psize, (int)result,
                  params->ppptr ? **params->ppptr : NULL);
    }
    if (result == ZE_RESULT_SUCCESS && **params->ppptr) {
        aimdo_xpu_note_native_allocation(**params->ppptr, *params->psize, device);
        /* Record pressure after the driver call; owner-side VBAR code reclaims. */
        aimdo_xpu_prepare_allocation(device, 0);
    } else if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
        // PyTorch's native caching allocator releases idle blocks and retries
        // the physical allocation after an OOM. Record one request-sized
        // tranche for the next owner-side VBAR reclaim boundary.
        aimdo_xpu_retry_allocation(device, *params->psize);
    }
}

static void ZE_APICALL xpu_mem_free_epilogue(
    ze_mem_free_params_t *params, ze_result_t result,
    void *tracer_user_data, void **instance_user_data) {
    (void)tracer_user_data;
    (void)instance_user_data;
    if (result == ZE_RESULT_SUCCESS && params && params->pptr) {
        aimdo_xpu_note_native_release(*params->pptr);
    }
}

static void ZE_APICALL xpu_mem_free_ext_epilogue(
    ze_mem_free_ext_params_t *params, ze_result_t result,
    void *tracer_user_data, void **instance_user_data) {
    (void)tracer_user_data;
    (void)instance_user_data;
    if (result == ZE_RESULT_SUCCESS && params && params->pptr) {
        aimdo_xpu_note_native_release(*params->pptr);
    }
}

static bool xpu_register_tracer_callbacks(void) {
    return
        zelTracerMemAllocDeviceRegisterCallback(
            g_xpu_tracer, ZEL_REGISTER_PROLOGUE,
            xpu_mem_alloc_device_prologue) == ZE_RESULT_SUCCESS &&
        zelTracerMemAllocDeviceRegisterCallback(
            g_xpu_tracer, ZEL_REGISTER_EPILOGUE,
            xpu_mem_alloc_device_epilogue) == ZE_RESULT_SUCCESS &&
        zelTracerMemFreeRegisterCallback(
            g_xpu_tracer, ZEL_REGISTER_EPILOGUE,
            xpu_mem_free_epilogue) == ZE_RESULT_SUCCESS &&
        zelTracerMemFreeExtRegisterCallback(
            g_xpu_tracer, ZEL_REGISTER_EPILOGUE,
            xpu_mem_free_ext_epilogue) == ZE_RESULT_SUCCESS;
}

/* The native allocation table is shared by both interception backends: the
 * Detours hooks used in production and the Level Zero tracing layer kept for
 * diagnosis. Both are idempotent so either may own the lifecycle. */
bool aimdo_xpu_native_accounting_init(void) {
    if (!g_xpu_allocation_lock_initialized) {
        InitializeCriticalSection(&g_xpu_allocation_lock);
        g_xpu_allocation_lock_initialized = true;
    }
    return true;
}

void aimdo_xpu_native_accounting_cleanup(void) {
    size_t bucket;

    if (!g_xpu_allocation_lock_initialized) {
        return;
    }
    EnterCriticalSection(&g_xpu_allocation_lock);
    for (bucket = 0; bucket < XPU_ALLOCATION_HASH_SIZE; ++bucket) {
        XpuNativeAllocation *entry = g_xpu_allocations[bucket];
        while (entry) {
            XpuNativeAllocation *next = entry->next;
            free(entry);
            entry = next;
        }
        g_xpu_allocations[bucket] = NULL;
    }
    LeaveCriticalSection(&g_xpu_allocation_lock);
    DeleteCriticalSection(&g_xpu_allocation_lock);
    g_xpu_allocation_lock_initialized = false;
}

bool aimdo_xpu_tracer_install(void) {
    zel_tracer_desc_t descriptor = {
        ZEL_STRUCTURE_TYPE_TRACER_DESC,
        NULL,
        &g_xpu_tracer_user_data,
    };

    if (g_xpu_tracer) {
        return true;
    }
    aimdo_xpu_native_accounting_init();

    if (zelEnableTracingLayer() != ZE_RESULT_SUCCESS ||
        zelTracerCreate(&descriptor, &g_xpu_tracer) != ZE_RESULT_SUCCESS ||
        !xpu_register_tracer_callbacks() ||
        zelTracerSetEnabled(g_xpu_tracer, true) != ZE_RESULT_SUCCESS) {
        aimdo_log(kAimdoTracerLogError, __FILE__, __LINE__,
                  "%s: failed to enable Level Zero allocation tracing\n",
                  __func__);
        if (g_xpu_tracer) {
            zelTracerDestroy(g_xpu_tracer);
            g_xpu_tracer = NULL;
        }
        zelDisableTracingLayer();
        aimdo_xpu_native_accounting_cleanup();
        return false;
    }
    aimdo_log(kAimdoTracerLogInfo, __FILE__, __LINE__,
              "%s: native Torch XPU allocator retained; tracing Level Zero physical allocations\n",
              __func__);
    return true;
}

void aimdo_xpu_tracer_remove(void) {
    if (g_xpu_tracer) {
        zelTracerSetEnabled(g_xpu_tracer, false);
        zelTracerDestroy(g_xpu_tracer);
        g_xpu_tracer = NULL;
        zelDisableTracingLayer();
    }
    aimdo_xpu_native_accounting_cleanup();
}

#if defined(__cplusplus)
}
#endif
