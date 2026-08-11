#include "plat.h"
#include "aimdo-time.h"

#include <windows.h>
#include <dxgi1_4.h>

#if defined(__HIP_PLATFORM_AMD__)
typedef union {
    struct {
        char name[256];
        char uuid[16];
        char luid[8];
    };
    void *ptr;
    unsigned long long ull;
    unsigned char bytes[4096];
} AimdoHipDeviceProp;
#endif

bool aimdo_wddm_init(CUdevice dev)
{
    int fail_code = 1;
    LUID cuda_luid;
    char adapter_name[256];
    IDXGIFactory4 *factory;
    IDXGIAdapter1 *adapter;
    UINT i;

    factory = NULL;
    adapter = NULL;
    adapter_name[0] = '\0';
    if (g_wddm_adapter) {
        g_wddm_adapter->lpVtbl->Release(g_wddm_adapter);
        g_wddm_adapter = NULL;
    }

#if defined(__HIP_PLATFORM_AMD__)
    AimdoHipDeviceProp hip_props = {0};

    if (!g_device_get_properties ||
        !CHECK_CU(g_device_get_properties(hip_props.bytes, dev))) {
        goto fail;
    }
    memcpy(&cuda_luid, hip_props.luid, sizeof(cuda_luid));
#else
    unsigned int node_mask;
    if (!CHECK_CU(cuDeviceGetLuid((char *)&cuda_luid, &node_mask, dev))) {
#if defined(AIMDO_XPU)
        /* Keep AIMDO functional on older runtimes or drivers that do not
         * expose the SYCL/Level Zero device LUID properties. */
        log(INFO,
            "comfy-aimdo XPU WDDM LUID unavailable; using Level Zero memory pressure\n");
        return true;
#else
        goto fail;
#endif
    }
#endif

    fail_code++;

    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&factory))) {
        goto fail;
    }

    for (i = 0; factory->lpVtbl->EnumAdapters1(factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->lpVtbl->GetDesc1(adapter, &desc);

        if (desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
            desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
            if (!WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, adapter_name,
                                     sizeof(adapter_name), NULL, NULL)) {
                strcpy(adapter_name, "<unknown>");
            }

            if (FAILED(adapter->lpVtbl->QueryInterface(adapter, &IID_IDXGIAdapter3, (void **)&g_wddm_adapter))) {
                adapter->lpVtbl->Release(adapter);
                break;
            }

            log(INFO,
                "comfy-aimdo WDDM adapter match: %s runtime_luid=%08lx:%08lx dxgi_luid=%08lx:%08lx\n",
                adapter_name,
                (unsigned long)(unsigned int)cuda_luid.HighPart,
                (unsigned long)cuda_luid.LowPart,
                (unsigned long)(unsigned int)desc.AdapterLuid.HighPart,
                (unsigned long)desc.AdapterLuid.LowPart);

#if defined(AIMDO_XPU)
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info;
                if (SUCCEEDED(g_wddm_adapter->lpVtbl->QueryVideoMemoryInfo(
                        g_wddm_adapter, 0,
                        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                        &nonlocal_info))) {
                    wddm_nonlocal_usage_baseline = nonlocal_info.CurrentUsage;
                }
            }
#endif

            adapter->lpVtbl->Release(adapter);
            factory->lpVtbl->Release(factory);
            return true;
        }
        adapter->lpVtbl->Release(adapter);
    }

fail:
    g_wddm_adapter = NULL;
    if (factory) {
        factory->lpVtbl->Release(factory);
    }
    log(WARNING, "comfy-aimdo WDDM init failed (%d). aimdo is blind to the CUDA Sysmem Fallback Policy\n", fail_code);
    return false;
}

/* Apparently this is still too small for all common graphics VRAM spikes.
 * However we can't pad too much on the smaller cards, and its not the end
 * of the world if we page out a little bit because it will adapt and correct
 * quickly.
 */

/* FIXME: This should be 0 if sysmem fallback is disabled by the user */
#define WDDM_BUDGET_HEADROOM (512 * 1024 * 1024)
#define CUDA_BUDGET_HEADROOM (192 * 1024 * 1024)

static bool wddm_trace_enabled(void)
{
    static int enabled = -1;
    char value[2];

    if (enabled < 0) {
        enabled = GetEnvironmentVariableA(
            "AIMDO_XPU_WDDM_TRACE", value, sizeof(value)) > 0 &&
            value[0] == '1';
    }
    return enabled != 0;
}

void aimdo_wddm_force_poll(void)
{
    wddm_timestamp_last_check = 0;
}

bool poll_budget_deficit(const char **prevailing_deficit_method)
{
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info;
    uint64_t effective_usage = total_vram_usage;
    uint64_t effective_budget = vram_capacity;
    uint64_t nonlocal_excess = 0;
    size_t free_vram = 0, total_vram = 0;
    bool used_nvml = false;

    uint64_t now = GET_TICK();

    if (now - wddm_timestamp_last_check < 2000) {
        return true;
    }
    wddm_timestamp_last_check = now;
    total_vram_last_check = total_vram_usage;

    if (g_wddm_adapter) {
        if (SUCCEEDED(g_wddm_adapter->lpVtbl->QueryVideoMemoryInfo(g_wddm_adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            /*
             * CurrentUsage is WDDM's complete local-memory accounting for
             * this process. Unlike total_vram_usage it includes allocations
             * made directly by SYCL, oneDNN, the driver, and other libraries
             * that do not pass through AIMDO. Use it as the sampled baseline;
             * budget_deficit() applies AIMDO's recorded usage delta until the
             * next rate-limited DXGI query.
             */
            effective_usage = info.CurrentUsage;
            effective_budget = info.Budget;
            log(DEBUG,
                "%s: WDDM budget=%zu MB usage=%zu MB recorded=%zu MB reservation=%zu MB available=%zu MB\n",
                __func__, (size_t)(info.Budget / M), (size_t)(info.CurrentUsage / M),
                (size_t)(total_vram_usage / M),
                (size_t)(info.CurrentReservation / M),
                (size_t)(info.AvailableForReservation / M));
#if defined(AIMDO_XPU)
            if (SUCCEEDED(g_wddm_adapter->lpVtbl->QueryVideoMemoryInfo(
                    g_wddm_adapter, 0,
                    DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                    &nonlocal_info))) {
                if (!wddm_nonlocal_usage_baseline) {
                    wddm_nonlocal_usage_baseline = nonlocal_info.CurrentUsage;
                }
                if (nonlocal_info.CurrentUsage >
                    wddm_nonlocal_usage_baseline) {
                    nonlocal_excess = nonlocal_info.CurrentUsage -
                                      wddm_nonlocal_usage_baseline;
                }
                if (wddm_trace_enabled()) {
                    log(INFO,
                        "AIMDO XPU WDDM local_budget=%zu local_usage=%zu nonlocal_budget=%zu nonlocal_usage=%zu nonlocal_baseline=%zu recorded=%zu\n",
                        (size_t)info.Budget, (size_t)info.CurrentUsage,
                        (size_t)nonlocal_info.Budget,
                        (size_t)nonlocal_info.CurrentUsage,
                        (size_t)wddm_nonlocal_usage_baseline,
                        (size_t)total_vram_usage);
                }
            }
#endif
        } else {
            log(WARNING, "comfy-aimdo WDDM VRAM query failed. Using physical capacity as fallback\n");
        }
    }

    deficit_sync = (ssize_t)effective_usage + (ssize_t)WDDM_BUDGET_HEADROOM -
                   (ssize_t)effective_budget;
    *prevailing_deficit_method = g_wddm_adapter
        ? "WDDM budget"
        : "physical capacity";
#if defined(AIMDO_XPU)
    /* Runtime bookkeeping can move a small amount of memory into the
     * non-local segment without indicating device-memory fallback. Only
     * react once the excess clears the same 512 MiB safety margin used for
     * the local WDDM budget. */
    if (nonlocal_excess > WDDM_BUDGET_HEADROOM &&
        (ssize_t)nonlocal_excess > deficit_sync) {
        deficit_sync = (ssize_t)nonlocal_excess;
        *prevailing_deficit_method = "WDDM non-local usage";
    }
#endif

#if defined(AIMDO_CUDA)
    used_nvml = nvml_device && aimdo_nvml_memory_info(nvml_device, &free_vram, &total_vram);
#endif
    if (used_nvml || CHECK_CU(cuMemGetInfo(&free_vram, &total_vram))) {
        ssize_t deficit_cuda = (ssize_t)(CUDA_BUDGET_HEADROOM / 2) - (ssize_t)free_vram;

        log(DEBUG,
            "%s: device memory free=%zu MB total=%zu MB deficit_cuda=%zd MB\n",
            __func__, free_vram / M, total_vram / M, deficit_cuda / (ssize_t)M);

        if (deficit_cuda > deficit_sync) {
            deficit_sync = deficit_cuda;
            *prevailing_deficit_method = used_nvml ? "NVML (Windows)" : "cuMemGetInfo (Windows)";
        }
    }

    log(DEBUG, "%s: prevailing method %s\n", __func__, *prevailing_deficit_method);
    return true;
}

void aimdo_wddm_cleanup()
{
    if (g_wddm_adapter) {
        g_wddm_adapter->lpVtbl->Release(g_wddm_adapter);
        g_wddm_adapter = NULL;
    }
}
