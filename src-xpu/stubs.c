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

    /* This can run under PyTorch's allocator/UMF locks.  Re-entering Level
     * Zero virtual-memory management here is enough to destabilize WDDM even
     * though the lower driver allocation call has returned.  Record pressure
     * only; the next VBAR/model-owner boundary performs the actual unmap. */
    deficit = budget_deficit(size);
    log(VVERBOSE, "%s: device=%d size=%zuk recorded=%zuk deficit=%zdk\n", __func__,
        device, size / K, (size_t)total_vram_usage / K, deficit / (ssize_t)K);
    if (deficit > 0) {
        vbars_request_reclaim(deficit);
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
    /* Even the retry is still inside the allocator's call stack.  Force a
     * fresh pressure sample, but leave VBAR mutation to its owner. */
    aimdo_wddm_force_poll();
    vbars_request_reclaim((ssize_t)size);
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
        /* The UR hook is above the driver call but remains inside the native
         * allocation stack.  Publish the shortage and let WDDM place this
         * request; the next VBAR fault drains it outside allocator locks. */
        vbars_request_reclaim((ssize_t)deficit);
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
