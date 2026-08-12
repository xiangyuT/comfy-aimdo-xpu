#include "plat.h"

bool aimdo_setup_hooks(void) {
    log(DEBUG, "%s: XPU keeps the native Torch allocator; no allocator hooks installed\n",
        __func__);
    return true;
}

void aimdo_teardown_hooks(void) {
}

int aimdo_xpu_current_device(void) {
    return g_devctx ? g_devctx->_device_id : -1;
}

uint64_t aimdo_xpu_recorded_usage(void) {
    return g_devctx ? g_devctx->_total_vram_usage : 0;
}

bool aimdo_xpu_prepare_allocation(int device, size_t size) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
    vbars_free(budget_deficit(size));
    return true;
}

bool aimdo_xpu_retry_allocation(int device, size_t size) {
    if (!set_devctx_for_device(device)) {
        return false;
    }
    vbars_free((ssize_t)size);
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
        vbars_free((ssize_t)deficit);
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
