#include "plat.h"

#if !defined(_WIN32) && !defined(_WIN64)
bool aimdo_setup_hooks(void) {
    log(DEBUG, "%s: XPU keeps the native Torch allocator; no allocator hooks installed\n",
        __func__);
    return true;
}

void aimdo_teardown_hooks(void) {
}
#endif

int aimdo_xpu_current_device(void) {
    return g_devctx ? g_devctx->_device_id : -1;
}

uint64_t aimdo_xpu_recorded_usage(void) {
    return g_devctx ? g_devctx->_total_vram_usage : 0;
}

bool aimdo_xpu_sample_pressure(int device, size_t size) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    const char *deficit_method = "unknown";

    /* Read-only. Safe to call while the driver is servicing an allocation,
     * unlike reclaim, which mutates Level Zero physical memory. */
    if (size >= (size_t)1 << 30) {
        aimdo_wddm_force_poll();
    }
    poll_budget_deficit(&deficit_method);
#else
    (void)size;
#endif
    return true;
}

bool aimdo_xpu_prepare_allocation(int device, size_t size) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    ssize_t deficit;

    /* Called after the driver's allocation call has returned, never during
     * it: releasing a VBAR page issues zeVirtualMemUnmap and
     * zePhysicalMemDestroy, and re-entering Level Zero memory management from
     * inside zeMemAllocDevice corrupts driver state.
     *
     * It also must not wait on the compute queue: that queue may be blocked
     * behind work which itself needs the residency being requested.
     * vbars_free_retired() releases only pages that are provably idle and
     * returns immediately otherwise. Any remaining shortage is absorbed by
     * WDDM, and the next fault or unpin boundary reclaims it. */
    deficit = budget_deficit(size);
    log(VVERBOSE, "%s: device=%d size=%zuk recorded=%zuk deficit=%zdk\n", __func__,
        device, size / K, (size_t)total_vram_usage / K, deficit / (ssize_t)K);
    if (deficit > 0) {
        vbars_free_retired(deficit);
    }
#else
    vbars_free(budget_deficit(size));
#endif
    return true;
}

bool aimdo_xpu_retry_allocation(int device, size_t size) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    /* Reached only after a real Level Zero allocation failure, which on this
     * platform means the driver could not place the request even in non-local
     * memory. Re-sample and reclaim again, still without waiting. */
    aimdo_wddm_force_poll();
    vbars_free_retired((ssize_t)size);
#else
    vbars_free((ssize_t)size);
#endif
    return true;
}

bool aimdo_xpu_allocation_deficit(int device, size_t size, int64_t *deficit) {
    if (!deficit || !set_devctx_for_device(device)) {
        return false;
    }
    *deficit = (int64_t)budget_deficit(size);
    return true;
}

bool aimdo_xpu_evict_for_allocation(int device, int64_t deficit) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
    if (deficit > 0) {
#if defined(_WIN32) || defined(_WIN64)
        /* Called from the Unified Runtime allocation hook. Nothing on this
         * path may wait on the compute queue: that queue can be blocked behind
         * work which itself needs the residency being requested.
         * vbars_free_retired() releases only pages that are provably idle and
         * returns immediately otherwise. Linux keeps the exact reclaim because
         * its allocator-time arbitration must satisfy the request. */
        vbars_free_retired((ssize_t)deficit);
#else
        vbars_free((ssize_t)deficit);
#endif
    }
    return true;
}

bool aimdo_xpu_account_allocation(int device, int64_t delta) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
    if (delta >= 0) {
        total_vram_usage += (uint64_t)delta;
    } else {
        uint64_t released = (uint64_t)(-delta);
        total_vram_usage = released < total_vram_usage
            ? total_vram_usage - released
            : 0;
    }
    return true;
}
