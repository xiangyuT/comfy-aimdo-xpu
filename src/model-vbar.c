#include "plat.h"
#include "thread-plat.h"

#define VBAR_PAGE_SIZE (32 << 20)

/* A WDDM physical allocation can still fail after the sampled budget says a
 * single page fits.  Keep this second-stage margin Windows/XPU-only: Linux
 * receives exact allocator-time pressure and must retain its existing
 * behavior. */
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
#define VBAR_WDDM_RETRY_RECLAIM (512 << 20)
#endif

#define VBAR_GET_PAGE_NR(x) ((x) / VBAR_PAGE_SIZE)
#define VBAR_GET_PAGE_NR_UP(x) VBAR_GET_PAGE_NR((x) + VBAR_PAGE_SIZE - 1)

typedef struct ResidentPage {
    CUmemGenericAllocationHandle handle;
    uint32_t pin_count;
    size_t serial;
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* Epoch at which this page stopped being used. Compared against the
     * published retired epoch so a reclaim can prove the page is idle without
     * waiting on the queue. */
    uint64_t retire_epoch;
#endif
} ResidentPage;

typedef struct ModelVBAR {
    CUdeviceptr vbar;
    size_t nr_pages;
    size_t watermark;
    size_t watermark_limit;

    int device;

    void *higher;
    void *lower;

    size_t resident_count;

    ResidentPage residency_map[1]; /* Must be last! */
} ModelVBAR;

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
/* VBAR mappings are visible to the Python model thread and to allocations
 * intercepted on arbitrary runtime threads.  The retirement fence proves GPU
 * completion, but it cannot make the CPU-side handle/pin/epoch transition
 * atomic.  Serialize that metadata exactly as a caching allocator serializes
 * its block state.  Allocation-path reclaim uses try-lock below, so this lock
 * never makes a driver allocation wait behind the model thread. */
static inline void vbar_state_lock(void) {
    mutex_lock((Mutex)vbar_lock);
}

static inline bool vbar_state_try_lock(void) {
    return mutex_try_lock((Mutex)vbar_lock);
}

static inline void vbar_state_unlock(void) {
    mutex_unlock((Mutex)vbar_lock);
}

/* Diagnostic/correctness escape hatch.  A disabled asynchronous path leaves
 * reclaim to synchronized model boundaries and normal host offload.  Keep the
 * decision in vbars_free_retired() so every caller, including allocator hooks
 * and both host streaming implementations, is covered. */
static bool vbar_async_reclaim_enabled(void) {
    static volatile LONG cached = -1;
    LONG enabled = InterlockedCompareExchange(&cached, -1, -1);

    if (enabled < 0) {
        char value[8];
        DWORD length = GetEnvironmentVariableA(
            "AIMDO_XPU_ASYNC_VBAR_RECLAIM", value, sizeof(value));
        LONG detected = !(length > 0 && length < sizeof(value) &&
                          value[0] == '0');

        InterlockedCompareExchange(&cached, detected, -1);
        enabled = InterlockedCompareExchange(&cached, -1, -1);
    }
    return enabled != 0;
}
#else
static inline void vbar_state_lock(void) {}
static inline bool vbar_state_try_lock(void) { return true; }
static inline void vbar_state_unlock(void) {}
#endif

static inline void one_time_setup() {
    if (!highest_priority_p) {
        highest_priority_p = calloc(2, sizeof(*highest_priority_p));
        if (!highest_priority_p) {
            log(CRITICAL, "Host OOM\n");
            return;
        }
        lowest_priority_p = highest_priority_p + 1;
    }
    if (!highest_priority.lower) {
        assert(!lowest_priority.higher);
        highest_priority.lower = &lowest_priority;
        lowest_priority.higher = &highest_priority;
    }
}

SHARED_EXPORT
uint64_t vbars_analyze(void *devctx, bool only_dirty) {
    size_t calculated_total_vram = 0;

    set_devctx((AimdoContext *)devctx);
    vbar_state_lock();

    one_time_setup();
    if (only_dirty && !vbars_dirty) {
        vbar_state_unlock();
        return 0;
    }
    vbars_dirty = false;
    log(DEBUG, "---------------- VBAR Usage ---------------\n")

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        size_t actual_resident_count = 0;

        for (size_t p = 0; p < i->nr_pages; p++) {
            ResidentPage *rp = &i->residency_map[p];

            if (rp->handle) {
                actual_resident_count++;

                if (p >= i->watermark) {
                    log(WARNING, "VBAR %p: Resident page %zu is ABOVE watermark %zu\n",
                        (void*)i, p, i->watermark);
                }

                if (rp->pin_count) {
                    log(WARNING, "VBAR %p: Page %zu pin_count=%u\n", (void*)i, p, rp->pin_count);
                }
            }
        }

        if (actual_resident_count != i->resident_count) {
            log(WARNING, "VBAR %p: resident_count sync error! Struct: %zu, Actual: %zu\n",
                (void*)i, i->resident_count, actual_resident_count);
        }

        calculated_total_vram += (actual_resident_count * VBAR_PAGE_SIZE);

        log(DEBUG, "VBAR %p: Actual Resident VRAM = %zu MB\n",
            (void*)i, (actual_resident_count * VBAR_PAGE_SIZE) / M);
    }

    log(DEBUG, "Total VRAM for VBARs: %zu MB\n", calculated_total_vram / M);
    vbar_state_unlock();
    return (uint64_t)calculated_total_vram;
}

/* Describe a device address range for diagnostics: is it inside a VBAR, and is
 * every page it spans mapped and pinned? A copy only needs one page in the
 * range to be missing its physical backing to fail, so checking the first
 * address alone is not enough. */
int aimdo_vbar_describe_address(uint64_t address, int *mapped, unsigned *pin,
                                uint64_t *page_index) {
    return aimdo_vbar_describe_range(address, VBAR_PAGE_SIZE, mapped, pin,
                                     page_index, NULL, NULL);
}

int aimdo_vbar_describe_range(uint64_t address, uint64_t size, int *mapped,
                              unsigned *pin, uint64_t *page_index,
                              uint64_t *unmapped_page, uint64_t *pages_spanned) {
    if (!g_devctx) {
        return 0;
    }
    vbar_state_lock();
    one_time_setup();
    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority;
         i = i->higher) {
        uint64_t base = (uint64_t)i->vbar;
        uint64_t span = (uint64_t)i->nr_pages * VBAR_PAGE_SIZE;

        if (address >= base && address < base + span) {
            uint64_t offset = address - base;
            size_t first = (size_t)(offset / VBAR_PAGE_SIZE);
            size_t last = (size_t)((offset + (size ? size - 1 : 0)) /
                                   VBAR_PAGE_SIZE);
            int all_mapped = 1;
            unsigned min_pin = 0xffffffffu;

            if (last >= i->nr_pages) {
                last = i->nr_pages - 1;
            }
            if (unmapped_page) {
                *unmapped_page = UINT64_MAX;
            }
            for (size_t p = first; p <= last; p++) {
                if (!i->residency_map[p].handle) {
                    all_mapped = 0;
                    if (unmapped_page && *unmapped_page == UINT64_MAX) {
                        *unmapped_page = (uint64_t)p;
                    }
                }
                if (i->residency_map[p].pin_count < min_pin) {
                    min_pin = i->residency_map[p].pin_count;
                }
            }
            if (mapped) {
                *mapped = all_mapped;
            }
            if (pin) {
                *pin = min_pin == 0xffffffffu ? 0 : min_pin;
            }
            if (page_index) {
                *page_index = (uint64_t)first;
            }
            if (pages_spanned) {
                *pages_spanned = (uint64_t)(last - first + 1);
            }
            vbar_state_unlock();
            return 1;
        }
    }
    vbar_state_unlock();
    return 0;
}

/* Undo the pins this fault has taken so far. A fault that gives up partway
 * must not leave pins behind: a pinned page is excluded from every reclaim,
 * so a leaked pin permanently removes that page from the pool. */
static inline void vbar_unpin_range(ModelVBAR *mv, size_t first, size_t last) {
    for (size_t page_nr = first; page_nr < last; page_nr++) {
        ResidentPage *rp = &mv->residency_map[page_nr];

        if (rp->pin_count) {
            rp->pin_count--;
        }
    }
}

static inline bool mod1(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;

    do_free = do_free && rp->handle && (do_unpin || rp->pin_count == 0);
    if (do_free) {
        CHECK_CU(cuMemUnmap(vaddr, VBAR_PAGE_SIZE));
        unmap_workaround(vaddr, VBAR_PAGE_SIZE);
        CHECK_CU(cuMemRelease(rp->handle));
        total_vram_usage -= VBAR_PAGE_SIZE;
        rp->handle = 0;
        mv->resident_count--;
    }
    if (do_unpin) {
        rp->pin_count = 0;
    }
    return do_free;
}

static size_t vbars_free_except(ssize_t size, ModelVBAR *preserved) {
    size_t pages_needed;
    bool dirty = false;

    if (size <= 0) {
        return 0;
    }

    vbar_state_lock();
    one_time_setup();
    vbars_dirty = true;

    pages_needed = VBAR_GET_PAGE_NR_UP((size_t)size);

    for (ModelVBAR *i = lowest_priority.higher; pages_needed && i != &highest_priority;
         i = i->higher) {
        if (i == preserved) {
            continue;
        }
        for (;pages_needed && i->watermark > i->watermark_limit; i->watermark--) {
            if (!dirty) {
                CHECK_CU(cuCtxSynchronize());
                dirty = true;
            }
            if (mod1(i, i->watermark - 1, true, false)) {
                pages_needed--;
            }
        }
    }

    if (dirty) {
        CHECK_CU(cuCtxSynchronize());
    }

    vbar_state_unlock();
    return pages_needed;
}

size_t vbars_free(ssize_t size) {
    return vbars_free_except(size, NULL);
}

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
/* Reclaim without ever waiting on the compute queue.
 *
 * This is the only reclaim that is legal inside a native allocation path. A
 * Windows Torch allocation cannot fail - WDDM demotes the excess to non-local
 * memory instead - so there is no correctness requirement to free anything
 * before returning. Reclaim is therefore best effort: it releases every page
 * that is provably idle and skips the rest rather than waiting for one.
 *
 * Two differences from vbars_free_except() matter:
 *
 * The scan does not stop at the first page it cannot release. Pages are
 * tagged as they are unpinned, so the most recently used ones sit at the top
 * of the range and are the last to retire; stopping there would block reclaim
 * on exactly the page least likely to be ready.
 *
 * The watermark is left alone. Lowering it is a policy decision that denies
 * future faults until the next model activation, which turns a momentary
 * allocation spike into a working set that never recovers. Releasing a page
 * without lowering the watermark lets a later fault bring it back if the
 * memory is genuinely available again, and that fault still re-checks
 * pressure, so it cannot reintroduce an overcommit on its own.
 *
 * Returns the number of pages that could not be reclaimed.
 */
size_t vbars_free_retired(ssize_t size) {
    size_t pages_needed;
    uint64_t retired;

    if (size <= 0) {
        return 0;
    }

    pages_needed = VBAR_GET_PAGE_NR_UP((size_t)size);
    if (!vbar_async_reclaim_enabled()) {
        return pages_needed;
    }
    retired = aimdo_xpu_retired_epoch();

    /* The Unified Runtime hook may arrive while fault/unpin owns the metadata
     * lock.  Waiting here would move the old allocation-path deadlock from a
     * GPU queue wait to a CPU lock wait, so fail closed and retry later. */
    if (!vbar_state_try_lock()) {
        log(VVERBOSE,
            "%s: VBAR metadata busy; skipped %zu pages without waiting\n",
            __func__, pages_needed);
        return pages_needed;
    }

    one_time_setup();
    vbars_dirty = true;

    for (ModelVBAR *i = lowest_priority.higher; pages_needed && i != &highest_priority;
         i = i->higher) {
        /* Start above the watermark, not at it. Pages there are outside the
         * model's allowed working set: unpin no longer frees them on Windows
         * (that path synchronized the queue on every weight), so this reclaim
         * is the only thing that can, and scanning from the watermark down
         * left them mapped forever. They are also the correct first victims.
         *
         * The lower bound is min(watermark, watermark_limit): a page can be
         * left above a watermark that was later lowered past the limit, and
         * stopping at the limit alone would strand it. */
        size_t floor = i->watermark < i->watermark_limit
            ? i->watermark : i->watermark_limit;
        size_t page_nr = i->nr_pages;

        while (pages_needed && page_nr > floor) {
            ResidentPage *rp = &i->residency_map[--page_nr];

            if (!rp->handle || rp->pin_count || rp->retire_epoch > retired) {
                continue;
            }
            if (mod1(i, page_nr, true, false)) {
                pages_needed--;
            }
        }
    }

    if (pages_needed) {
        log(DEBUG, "%s: %zu pages still required after non-blocking reclaim\n",
            __func__, pages_needed);
    }

    vbar_state_unlock();
    return pages_needed;
}
#endif

static inline size_t move_cursor_to_absent(ModelVBAR *mv, size_t cursor) {
    while (cursor < mv->watermark && mv->residency_map[cursor].handle) {
        cursor++;
    }
    return cursor;
}

static inline size_t spend_surplus_on_cursor(ModelVBAR *mv, size_t target, size_t cursor,
                                             ssize_t *surplus) {
    while (*surplus >= (ssize_t)VBAR_PAGE_SIZE && cursor < target && cursor < mv->watermark) {
        *surplus -= (ssize_t)VBAR_PAGE_SIZE;
        cursor = move_cursor_to_absent(mv, cursor + 1);
    }
    return cursor;
}

static void vbars_free_for_vbar(ModelVBAR *mv, size_t target, ssize_t surplus) {
    size_t cursor = move_cursor_to_absent(mv, 0);
    bool synced = false;

    cursor = spend_surplus_on_cursor(mv, target, cursor, &surplus);

    for (ModelVBAR *i = lowest_priority.higher;
         ((cursor < target && cursor < mv->watermark) || surplus < 0) && i != &highest_priority;
         i = i->higher) {
        for (; ((cursor < target && cursor < mv->watermark) || surplus < 0) &&
               i->watermark > i->watermark_limit;
             i->watermark--) {
            ResidentPage *rp = &i->residency_map[i->watermark - 1];

            if (!synced && rp->handle && rp->pin_count == 0) {
                CHECK_CU(cuCtxSynchronize());
                synced = true;
            }
            if (mod1(i, i->watermark - 1, true, false)) {
                surplus += (ssize_t)VBAR_PAGE_SIZE;
                cursor = spend_surplus_on_cursor(mv, target, cursor, &surplus);
            }
        }
    }

    if (synced) {
        CHECK_CU(cuCtxSynchronize());
    }
}

static inline void remove_vbar(ModelVBAR *mv) {
    ((ModelVBAR *)mv->lower)->higher = mv->higher;
    ((ModelVBAR *)mv->higher)->lower = mv->lower;
}

static inline void insert_vbar(ModelVBAR *mv) {
    mv->lower = highest_priority.lower;
    ((ModelVBAR *)highest_priority.lower)->higher = mv;
    mv->higher = &highest_priority;
    highest_priority.lower = mv;
}

static inline void insert_vbar_last(ModelVBAR *mv) {
    mv->higher = lowest_priority.higher;
    ((ModelVBAR *)lowest_priority.higher)->lower = mv;
    mv->lower = &lowest_priority;
    lowest_priority.higher = mv;
}

SHARED_EXPORT
void *vbar_allocate(void *devctx, uint64_t size, int device) {
    ModelVBAR *mv;

    set_devctx((AimdoContext *)devctx);

    log_reset_shots();
    log(DEBUG, "%s (start): size=%zuM, device=%d\n", __func__, size / M, device);

    size_t nr_pages = VBAR_GET_PAGE_NR_UP(size);
    size_t nr_pages_max = VBAR_GET_PAGE_NR(vram_capacity);
    if (nr_pages_max < nr_pages) {
        nr_pages = nr_pages_max;
    }
    size = (uint64_t)nr_pages * VBAR_PAGE_SIZE;

    if (!(mv = calloc(1, sizeof(*mv) + nr_pages * sizeof(mv->residency_map[0])))) {
        log(CRITICAL, "Host OOM\n");
        return NULL;
    }

    /* FIXME: Do I care about alignment? Does Cuda just look after itself? */
    if (!CHECK_CU(cuMemAddressReserve(&mv->vbar, size, 0, 0, 0))) {
        log(AIMDO_LOG_ERROR, "Could not reseve Virtual Address space for VBAR\n");
        free(mv);
        return NULL;
    }

    mv->device = device;
    mv->nr_pages = mv->watermark = nr_pages;

    vbar_state_lock();
    one_time_setup();
    vbars_dirty = true;
    insert_vbar(mv);
    vbar_state_unlock();

    log(DEBUG, "%s (return): vbar=%p\n", __func__, (void *)mv);
    return mv;
}

SHARED_EXPORT
void vbar_set_watermark_limit(void *devctx, void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    vbar_state_lock();
    mv->watermark_limit = VBAR_GET_PAGE_NR_UP(size);
    vbar_state_unlock();
}

SHARED_EXPORT
void vbar_set_watermark(void *devctx, void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t watermark = VBAR_GET_PAGE_NR_UP(size);

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    vbar_state_lock();
    vbars_dirty = true;

    if (watermark > mv->nr_pages) {
        watermark = mv->nr_pages;
    }

    if (watermark < mv->watermark) {
        CHECK_CU(cuCtxSynchronize());
        for (size_t page_nr = watermark; page_nr < mv->watermark; page_nr++) {
            mod1(mv, page_nr, true, false);
        }
    }

    mv->watermark = watermark;
    vbar_state_unlock();
}

SHARED_EXPORT
void vbars_reset_watermark_limits(void *devctx) {
    set_devctx((AimdoContext *)devctx);
    vbar_state_lock();
    one_time_setup();
    log(VERBOSE, "%s\n", __func__);

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        i->watermark_limit = 0;
    }
    vbar_state_unlock();
}

SHARED_EXPORT
void vbars_prepare_allocation(void *devctx, void *vbar, uint64_t size) {
    set_devctx((AimdoContext *)devctx);
    one_time_setup();
    /*
     * Windows cannot evict from the Level Zero allocation callback because
     * doing so waits re-entrantly on the same SYCL queue.  Its model-boundary
     * reclaim is therefore only a prediction based on native allocator
     * history.  It may discard lower-priority models, but it must not discard
     * pages from the active model on the strength of that prediction alone.
     * A later vbar_fault() still applies exact live pressure to every VBAR.
     */
    vbars_free_except(budget_deficit((size_t)size), (ModelVBAR *)vbar);
}

SHARED_EXPORT
void vbar_prioritize(void *devctx, void *vbar, uint64_t clamp) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);
    malloc_async_clamp = clamp;

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    vbar_state_lock();
    vbars_dirty = true;

    log_reset_shots();

    remove_vbar(mv);
    insert_vbar(mv);

    mv->watermark = mv->nr_pages;
    vbar_state_unlock();
}

SHARED_EXPORT
void vbar_deprioritize(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    vbar_state_lock();
    vbars_dirty = true;

    log_reset_shots();

    remove_vbar(mv);
    insert_vbar_last(mv);
    vbar_state_unlock();
}

SHARED_EXPORT
uint64_t vbar_get(void *devctx, void *vbar) {
    set_devctx((AimdoContext *)devctx);
    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    return (uint64_t)((ModelVBAR *)vbar)->vbar;
}

#define VBAR_FAULT_SUCCESS           0
#define VBAR_FAULT_OOM               1
#define VBAR_FAULT_ERROR             2

static int vbar_fault_locked(void *devctx, void *vbar, uint64_t offset,
                             uint64_t size, uint32_t *signature) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    int ret = VBAR_FAULT_SUCCESS;
    size_t signature_index = 0;
    bool miss_alloc_checked = false;

    set_devctx((AimdoContext *)devctx);

    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));
    vbars_dirty = true;

    /* Stopgap. If the we get a bad shared memory spike, collect it here on the next layer
     * as the allocator is unreliable as it may not actually be called reliably when you
     * really need to know you have spilled.
     */
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* vbars_free() synchronizes the compute queue, and this runs on every
     * fault. Under pressure that turns each weight into a full queue drain:
     * py-spy caught the sampler thread inside
     * vbar_fault -> vbars_analyze -> queue::wait_and_throw -> urQueueFinish,
     * with per-step time degrading run over run as the resident set shrank and
     * misses multiplied. Use the non-blocking reclaim: it releases every page
     * that is provably idle and skips the rest. */
    vbars_free_retired(budget_deficit(0));
#else
    vbars_free(budget_deficit(0));
#endif

    if (page_end > mv->watermark) {
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        /* The watermark records that pressure was seen earlier in this
         * activation, not that memory is still short. On Windows the pressure
         * is often transient - a workspace tensor or a Torch cache block - so
         * latching it until the next prioritize() makes one spike cost the
         * model its working set for the rest of the run. That is what streams
         * a tiled VAE decoder from host storage on every tile.
         *
         * Revalidate against live pressure instead: reopen the range only when
         * the pages this fault would have to allocate actually fit right now.
         * A reopened range still allocates through the normal per-page checks
         * below, so it cannot reintroduce an overcommit on its own. */
        size_t absent = 0;

        for (size_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
            if (!mv->residency_map[page_nr].handle) {
                absent++;
            }
        }
        if (budget_deficit(absent * VBAR_PAGE_SIZE) > 0) {
            log(VVERBOSE, "VBAR Allocation is above watermark\n");
            return VBAR_FAULT_OOM;
        }
        log(DEBUG, "VBAR reopening watermark %zu -> %zu; %zu absent pages fit\n",
            mv->watermark, page_end, absent);
        mv->watermark = page_end;
#else
        log(VVERBOSE, "VBAR Allocation is above watermark\n");
        return VBAR_FAULT_OOM;
#endif
    }

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
        CUresult err = CUDA_ERROR_OUT_OF_MEMORY;
        CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;
        ResidentPage *rp = &mv->residency_map[page_nr];
        ssize_t allocation_deficit;

        if (rp->handle) {
            /* Pin before anything else in this fault can reclaim. The pin loop
             * used to run only after every page was mapped, which left the
             * pages this fault had already mapped unpinned and therefore
             * eligible for the reclaim calls further down. Those calls then
             * released a page that the trailing loop still marked as pinned,
             * producing a pinned page with no physical backing: the copy into
             * it failed with OUT_OF_DEVICE_MEMORY while the device still had
             * gigabytes free. */
            rp->pin_count++;
            signature[signature_index++] = rp->serial;
            continue;
        }

        if (!miss_alloc_checked) {
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
            /* vbars_free_for_vbar() synchronizes the queue twice. Reclaim only
             * provably idle pages instead; a shortfall is absorbed by WDDM and
             * revisited at the next boundary. */
            {
                ssize_t shortfall =
                    (ssize_t)(total_vram_usage +
                              (page_end - page_nr) * VBAR_PAGE_SIZE) +
                    (ssize_t)simple_vram_headroom - (ssize_t)vram_capacity;
                if (shortfall > 0) {
                    vbars_free_retired(shortfall);
                }
            }
#else
            vbars_free_for_vbar(mv, page_end,
                                (ssize_t)vram_capacity -
                                ((ssize_t)(total_vram_usage +
                                           (page_end - page_nr) * VBAR_PAGE_SIZE) +
                                 (ssize_t)simple_vram_headroom));
#endif
            miss_alloc_checked = true;

            if (page_end > mv->watermark) {
                log(DEBUG, "VBAR allocation cancelled due to allocation-check watermark reduction\n");
                vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                return VBAR_FAULT_OOM;
            }
        }

        log(VERBOSE, "VBAR needs to allocate VRAM for page %d\n", (int)page_nr);

        allocation_deficit = budget_deficit(VBAR_PAGE_SIZE);
        if (allocation_deficit > 0 ||
            (err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
            size_t retry_reclaim = allocation_deficit > (ssize_t)VBAR_PAGE_SIZE
                ? (size_t)allocation_deficit
                : (size_t)VBAR_PAGE_SIZE;

            if (err != CUDA_ERROR_OUT_OF_MEMORY) {
                log(AIMDO_LOG_ERROR, "VRAM Allocation failed (non OOM)\n");
                vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                return VBAR_FAULT_ERROR;
            }
            log(DEBUG,
                "VBAR allocator attempt exceeds available VRAM; reclaiming %zu MB ...\n",
                retry_reclaim / M);
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
            vbars_free_retired((ssize_t)retry_reclaim);
#else
            vbars_free((ssize_t)retry_reclaim);
#endif
            if (page_end > mv->watermark) {
                log(DEBUG, "VBAR allocation cancelled due to backup-free watermark reduction\n");
                vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                return VBAR_FAULT_OOM;
            }
            if ((err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
                if (err == CUDA_ERROR_OUT_OF_MEMORY) {
                    /* The DXGI budget is sampled and Level Zero may need more
                     * contiguous physical headroom than one VBAR page.  This
                     * path is reached only after a real allocation failure,
                     * so reclaiming the WDDM safety margin is not speculative.
                     * Retry only with pages whose completion events have
                     * already retired. Waiting for the compute queue inside a
                     * fault can deadlock behind work that needs this page; if
                     * no retired page is ready, report OOM so the caller can
                     * use its host-streaming fallback. */
                    log(DEBUG,
                        "VBAR Windows XPU retry reclaiming an additional %zu MB ...\n",
                        (size_t)VBAR_WDDM_RETRY_RECLAIM / M);
                    (void)vbars_free_retired(VBAR_WDDM_RETRY_RECLAIM);
                    if (page_end > mv->watermark) {
                        log(DEBUG,
                            "VBAR allocation cancelled after Windows XPU retry reclaim\n");
                        vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                        return VBAR_FAULT_OOM;
                    }
                    err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device,
                                        &rp->handle);
                }
#endif
            }
            if (err != CUDA_SUCCESS) {
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
                if (err == CUDA_ERROR_OUT_OF_MEMORY) {
                    /* A genuine physical OOM is a memory shortage, not a
                     * malfunction. Reporting it as an error made the caller
                     * raise "Fault failed: 2" instead of taking the offload
                     * path, and skipped the native-cache recovery that only
                     * runs for OOM. */
                    log(INFO, "VRAM Allocation OOM; weight will be offloaded\n");
                    vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                    return VBAR_FAULT_OOM;
                }
#endif
                log(AIMDO_LOG_ERROR, "VRAM Allocation failed\n");
                vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                return VBAR_FAULT_ERROR;
            }
        }
        rp->serial++;
        rp->pin_count++;
        signature[signature_index++] = rp->serial;
        mv->resident_count++;
    }

    /* Every page in the range was pinned as it was confirmed mapped, so no
     * reclaim inside this fault could take one back. */

    log(VVERBOSE, "%s (return) %d\n", __func__, ret);
    return ret;
}

SHARED_EXPORT
int vbar_fault(void *devctx, void *vbar, uint64_t offset, uint64_t size,
               uint32_t *signature) {
    int result;

    set_devctx((AimdoContext *)devctx);
    vbar_state_lock();
    result = vbar_fault_locked(devctx, vbar, offset, size, signature);
    vbar_state_unlock();
    return result;
}

SHARED_EXPORT
void vbar_unpin(void *devctx, void *vbar, uint64_t offset, uint64_t size) {
    vbar_unpin_stream(devctx, vbar, offset, size, 0);
}

SHARED_EXPORT
void vbar_unpin_stream(void *devctx, void *vbar, uint64_t offset, uint64_t size,
                       uint64_t stream) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    uint64_t retirement_epoch;
#endif

    set_devctx((AimdoContext *)devctx);

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* VBAR map/unmap use the Level Zero virtual-memory calls, which carry no
     * stream, so this is the only point where the queue that actually consumed
     * the weight is visible. Register it before tagging the pages: a
     * retirement fence submitted only to the default queue does not order work
     * queued elsewhere, and reclaiming on that proof released pages that were
     * still in use. */
    if (stream) {
        aimdo_xpu_register_queue((void *)(uintptr_t)stream);
    }
    /* Snapshot the epoch after the operator was submitted but before taking
     * the page-state lock.  This preserves the retire-lock -> VBAR-lock order
     * used by reclaim and avoids a lock inversion.  Publishing the epoch and
     * the final pin transition happens atomically under vbar_state_lock(). */
    retirement_epoch = aimdo_xpu_retire_epoch_current();
#else
    (void)stream;
#endif
    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));
    vbar_state_lock();
    vbars_dirty = true;
    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* Windows never frees a page from here. Pages above the watermark are
     * tagged with a retirement epoch and released later by the non-blocking
     * reclaim, so neither synchronize below is needed - and both ran on every
     * unpinned weight, which is the hot path. */
    const bool free_above_watermark = false;
#else
    const bool free_above_watermark = true;

    if (page_end > mv->watermark) {
        CHECK_CU(cuCtxSynchronize());
    }
#endif

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end && page_nr < mv->nr_pages; page_nr++) {
        ResidentPage *rp = &mv->residency_map[page_nr];
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        /* Publish the retirement proof before making the page idle. Because
         * reclaim holds the same lock, it can never observe pin_count == 0
         * paired with the previous use's retire_epoch. */
        if (rp->pin_count == 1) {
            rp->retire_epoch = retirement_epoch;
            rp->pin_count = 0;
        } else if (rp->pin_count) {
            rp->pin_count--;
        }
#else
        if (rp->pin_count) {
            rp->pin_count--;
        }
#endif
        mod1(mv, page_nr, free_above_watermark && page_nr >= mv->watermark, false);
    }

#if !defined(AIMDO_XPU) || (!defined(_WIN32) && !defined(_WIN64))
    if (page_end > mv->watermark) {
        CHECK_CU(cuCtxSynchronize());
    }
#endif

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* Advance retirement while the model runs, not only once something is
     * already under pressure. Polling here keeps at most one fence in flight
     * and costs a single barrier per completion interval, but it means the
     * published retired epoch is current by the time an allocation hook needs
     * to reclaim. Waiting until first pressure to submit a fence would leave
     * every page unprovable exactly when the memory is needed. */
    vbar_state_unlock();
    (void)aimdo_xpu_retired_epoch();
#else
    vbar_state_unlock();
#endif
}

SHARED_EXPORT
void vbar_free(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: vbar=%p\n", __func__, vbar);
    vbar_state_lock();
    vbars_dirty = true;

    CHECK_CU(cuCtxSynchronize());

    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, true, true);
    }
    remove_vbar(mv);
    CHECK_CU(cuMemAddressFree(mv->vbar, (size_t)mv->nr_pages * VBAR_PAGE_SIZE));
    CHECK_CU(cuCtxSynchronize());
    vbar_state_unlock();
    free(mv);
}

SHARED_EXPORT
size_t vbar_loaded_size(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t loaded;

    set_devctx((AimdoContext *)devctx);

    vbar_state_lock();
    loaded = mv->resident_count * VBAR_PAGE_SIZE;
    vbar_state_unlock();
    return loaded;
}

SHARED_EXPORT
size_t vbar_get_nr_pages(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t pages;
    set_devctx((AimdoContext *)devctx);
    vbar_state_lock();
    pages = mv->nr_pages;
    vbar_state_unlock();
    return pages;
}

SHARED_EXPORT
size_t vbar_get_watermark(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t watermark;
    set_devctx((AimdoContext *)devctx);
    vbar_state_lock();
    watermark = mv->watermark;
    vbar_state_unlock();
    return watermark;
}

SHARED_EXPORT
void vbar_get_residency(void *devctx, void *vbar, uint8_t *out, size_t max_pages) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t n = mv->nr_pages < max_pages ? mv->nr_pages : max_pages;

    set_devctx((AimdoContext *)devctx);
    vbar_state_lock();
    for (size_t i = 0; i < n; i++) {
        ResidentPage *rp = &mv->residency_map[i];
        /* bit 0: resident, bit 1: pinned */
        out[i] = (rp->handle ? 1 : 0) | (rp->pin_count ? 2 : 0);
    }
    vbar_state_unlock();
}

SHARED_EXPORT
uint64_t vbar_free_memory(void *devctx, void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t pages_to_free = VBAR_GET_PAGE_NR_UP(size);
    size_t pages_freed = 0;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s (start): size=%lldk\n", __func__, (ull)size);
    vbar_state_lock();
    vbars_dirty = true;

    CHECK_CU(cuCtxSynchronize());

    for (;pages_to_free && mv->watermark > mv->watermark_limit; mv->watermark--) {
        /* In theory we should never have pins here, but
         * respect pins if it really comes up.
         */
        if (mod1(mv, mv->watermark - 1, true, false)) {
            pages_to_free--;
            pages_freed++;
        }
    }

    CHECK_CU(cuCtxSynchronize());

    vbar_state_unlock();
    return (uint64_t)pages_freed * VBAR_PAGE_SIZE;
}
