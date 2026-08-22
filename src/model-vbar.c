#include "plat.h"
#include "thread-plat.h"

#define VBAR_PAGE_SIZE (32 << 20)

/* A WDDM physical allocation can still fail after the sampled budget says a
 * single page fits.  Keep this second-stage margin Windows/XPU-only: Linux
 * receives exact allocator-time pressure and must retain its existing
 * behavior. */
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
#define VBAR_CUDA_ERROR_UNKNOWN ((CUresult)999)
#endif

#define VBAR_GET_PAGE_NR(x) ((x) / VBAR_PAGE_SIZE)
#define VBAR_GET_PAGE_NR_UP(x) VBAR_GET_PAGE_NR((x) + VBAR_PAGE_SIZE - 1)

typedef struct ResidentPage {
    CUmemGenericAllocationHandle handle;
    uint32_t pin_count;
    size_t serial;
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* One dependency per queue that actually consumed this page. The token
     * names both the queue and its fence generation. Sixty-four entries cost
     * at most 512 bytes per 32 MiB VBAR page and avoid lossy fixed-size
     * overflow or a heap allocation in the hot path. */
    uint64_t retire_tokens[AIMDO_XPU_RETIRE_MAX_QUEUES];
    uint64_t eviction_generation;
    uint8_t retire_token_count;
    uint8_t retire_unknown;
    uint8_t mapped;
    uint8_t evicting;
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

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    uint64_t identity;
    uint8_t closing;
#endif

    ResidentPage residency_map[1]; /* Must be last! */
} ModelVBAR;

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
static uint64_t vbar_identity_counter;

enum {
    VBAR_MAPPING_UNMAPPED = 0,
    VBAR_MAPPING_MAPPED = 1,
    /* A map succeeded but cleanup after an access-setting failure could not
     * prove whether the driver removed it. Never use or reclaim this page. */
    VBAR_MAPPING_UNKNOWN = 2,
};

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
 * reclaim to synchronized model boundaries and normal host offload. */
static bool vbar_async_reclaim_enabled(void) {
    static volatile LONG cached = -1;
    LONG enabled = InterlockedCompareExchange(&cached, -1, -1);

    if (enabled < 0) {
        char value[8];
        DWORD length = GetEnvironmentVariableA(
            "AIMDO_XPU_ASYNC_VBAR_RECLAIM", value, sizeof(value));
        /* Non-blocking two-phase retirement is the product path.  The
         * synchronized owner-boundary implementation remains available as an
         * explicit correctness oracle, but cannot be the default: it retains
         * every page touched inside one model activation and forces WDDM to
         * page model weights and activations together. */
        LONG detected = 1;
        if (length == 1 && value[0] == '0') {
            detected = 0;
        }

        InterlockedCompareExchange(&cached, detected, -1);
        enabled = InterlockedCompareExchange(&cached, -1, -1);
    }
    return enabled != 0;
}

static inline void vbar_retire_reset(ResidentPage *rp) {
    rp->retire_token_count = 0;
    rp->retire_unknown = 0;
}

static inline bool vbar_page_is_mapped(const ResidentPage *rp) {
    return rp->handle && rp->mapped == VBAR_MAPPING_MAPPED;
}

static inline void vbar_page_cancel_eviction(ResidentPage *rp) {
    if (rp->evicting) {
        rp->evicting = 0;
        rp->eviction_generation++;
    }
}

static inline void vbar_retire_record(ResidentPage *rp, uint64_t token) {
    uint64_t queue_tag;

    if (!token) {
        rp->retire_unknown = 1;
        return;
    }
    queue_tag = token & AIMDO_XPU_RETIRE_TOKEN_QUEUE_MASK;
    for (uint8_t index = 0; index < rp->retire_token_count; ++index) {
        uint64_t existing = rp->retire_tokens[index];

        if ((existing & AIMDO_XPU_RETIRE_TOKEN_QUEUE_MASK) == queue_tag) {
            if (token > existing) {
                rp->retire_tokens[index] = token;
            }
            return;
        }
    }
    if (rp->retire_token_count >= AIMDO_XPU_RETIRE_MAX_QUEUES) {
        rp->retire_unknown = 1;
        return;
    }
    rp->retire_tokens[rp->retire_token_count++] = token;
}

static inline bool vbar_retire_complete(
    const ResidentPage *rp, const uint64_t *completed, size_t count) {
    if (rp->retire_unknown) {
        return false;
    }
    for (uint8_t index = 0; index < rp->retire_token_count; ++index) {
        uint64_t token = rp->retire_tokens[index];
        uint64_t queue_tag = token & AIMDO_XPU_RETIRE_TOKEN_QUEUE_MASK;
        size_t queue_index;
        uint64_t stamp;
        uint64_t generation;
        uint64_t incarnation;
        uint64_t completed_generation;
        uint64_t completed_incarnation;

        if (!queue_tag) {
            return false;
        }
        queue_index = (size_t)(queue_tag - 1);
        stamp = token >> AIMDO_XPU_RETIRE_TOKEN_QUEUE_BITS;
        generation = stamp & AIMDO_XPU_RETIRE_GENERATION_MASK;
        incarnation = stamp >> AIMDO_XPU_RETIRE_GENERATION_BITS;
        if (queue_index >= count) {
            return false;
        }
        completed_generation =
            completed[queue_index] & AIMDO_XPU_RETIRE_GENERATION_MASK;
        completed_incarnation =
            completed[queue_index] >> AIMDO_XPU_RETIRE_GENERATION_BITS;
        if (completed_incarnation != incarnation ||
            completed_generation < generation) {
            return false;
        }
    }
    return true;
}
#else
static inline bool vbar_page_is_mapped(const ResidentPage *rp) {
    return rp->handle != 0;
}
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

            if (vbar_page_is_mapped(rp)) {
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
                if (!vbar_page_is_mapped(&i->residency_map[p])) {
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

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
static uint8_t vbar_restore_page_mapping(ModelVBAR *mv, size_t page_nr,
                                         bool restore_resident_count) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;
    CUmemAccessDesc access_desc = {
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = mv->device,
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    if (!rp->handle) {
        return VBAR_MAPPING_UNMAPPED;
    }
    if (!CHECK_CU_ERROR(cuMemMap(
            vaddr, VBAR_PAGE_SIZE, 0, rp->handle, 0))) {
        rp->mapped = VBAR_MAPPING_UNMAPPED;
        return rp->mapped;
    }
    rp->mapped = VBAR_MAPPING_UNKNOWN;
    if (!CHECK_CU_ERROR(cuMemSetAccess(
            vaddr, VBAR_PAGE_SIZE, &access_desc, 1))) {
        if (CHECK_CU_ERROR(cuMemUnmap(vaddr, VBAR_PAGE_SIZE))) {
            rp->mapped = VBAR_MAPPING_UNMAPPED;
        } else {
            rp->retire_unknown = 1;
        }
        return rp->mapped;
    }
    rp->mapped = VBAR_MAPPING_MAPPED;
    if (restore_resident_count) {
        mv->resident_count++;
    }
    return rp->mapped;
}

static CUresult vbar_map_new_page(ModelVBAR *mv, size_t page_nr) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;
    CUmemAllocationProp prop = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = mv->device,
    };
    CUmemAccessDesc access_desc = {
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = mv->device,
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    CUresult err;

    /* Recover a physical handle retained after a prior cleanup failure before
     * creating another one. Never overwrite a handle AIMDO still owns. */
    if (rp->handle) {
        if (rp->mapped == VBAR_MAPPING_UNKNOWN) {
            return VBAR_CUDA_ERROR_UNKNOWN;
        }
        return vbar_restore_page_mapping(mv, page_nr, false) ==
                       VBAR_MAPPING_MAPPED
            ? CUDA_SUCCESS : VBAR_CUDA_ERROR_UNKNOWN;
    }

    if (!CHECK_CU_ERROR(err = cuMemCreate(
            &rp->handle, VBAR_PAGE_SIZE, &prop, 0))) {
        return err;
    }
    if (!CHECK_CU_ERROR(err = cuMemMap(
            vaddr, VBAR_PAGE_SIZE, 0, rp->handle, 0))) {
        if (CHECK_CU_ERROR(cuMemRelease(rp->handle))) {
            rp->handle = 0;
        } else {
            /* The physical allocation still exists even though it is not
             * mapped. Account it once and retry this handle on a later fault. */
            total_vram_usage += VBAR_PAGE_SIZE;
        }
        return err;
    }
    rp->mapped = VBAR_MAPPING_UNKNOWN;
    if (!CHECK_CU_ERROR(err = cuMemSetAccess(
            vaddr, VBAR_PAGE_SIZE, &access_desc, 1))) {
        if (CHECK_CU_ERROR(cuMemUnmap(vaddr, VBAR_PAGE_SIZE))) {
            rp->mapped = VBAR_MAPPING_UNMAPPED;
            if (CHECK_CU_ERROR(cuMemRelease(rp->handle))) {
                rp->handle = 0;
            } else {
                total_vram_usage += VBAR_PAGE_SIZE;
            }
        } else {
            /* The mapping may still exist but its access contract is unknown.
             * Retain both VA and handle permanently rather than guessing. */
            rp->retire_unknown = 1;
            total_vram_usage += VBAR_PAGE_SIZE;
        }
        return err;
    }

    rp->mapped = VBAR_MAPPING_MAPPED;
    total_vram_usage += VBAR_PAGE_SIZE;
    return CUDA_SUCCESS;
}
#endif

static inline bool mod1(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;

    do_free = do_free && vbar_page_is_mapped(rp) &&
              (do_unpin || (rp->pin_count == 0 && !rp->evicting));
    if (do_free) {
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        if (!CHECK_CU_ERROR(cuMemUnmap(vaddr, VBAR_PAGE_SIZE))) {
            rp->evicting = 0;
            return false;
        }
        rp->mapped = VBAR_MAPPING_UNMAPPED;
        if (!CHECK_CU_ERROR(cuMemRelease(rp->handle))) {
            /* Keep metadata truthful if physical destroy fails after unmap.
             * First restore the old mapping.  If the driver rejects that too,
             * retain the handle and total physical accounting but mark the
             * page unmapped/non-reclaimable for a later recovery attempt. */
            if (vbar_restore_page_mapping(mv, page_nr, false) !=
                VBAR_MAPPING_MAPPED) {
                rp->retire_unknown = 1;
                if (mv->resident_count) {
                    mv->resident_count--;
                }
            }
            rp->evicting = 0;
            return false;
        }
        total_vram_usage -= VBAR_PAGE_SIZE;
        rp->handle = 0;
        rp->mapped = VBAR_MAPPING_UNMAPPED;
        rp->evicting = 0;
        vbar_retire_reset(rp);
        mv->resident_count--;
#else
        CHECK_CU(cuMemUnmap(vaddr, VBAR_PAGE_SIZE));
        unmap_workaround(vaddr, VBAR_PAGE_SIZE);
        CHECK_CU(cuMemRelease(rp->handle));
        total_vram_usage -= VBAR_PAGE_SIZE;
        rp->handle = 0;
        mv->resident_count--;
#endif
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
/* Publish pressure from a runtime allocation/copy callback without touching
 * Level Zero virtual memory from that callback's stack.  Deficits describe a
 * target shortage, rather than independent allocations, so keep the maximum
 * instead of adding concurrent requests. */
void vbars_request_reclaim(ssize_t size) {
    volatile LONG64 *requested;
    LONG64 current;

    if (!g_devctx || size <= 0) {
        return;
    }
    requested = (volatile LONG64 *)&vbar_reclaim_requested;
    current = InterlockedCompareExchange64(requested, 0, 0);
    while (current < (LONG64)size) {
        LONG64 observed =
            InterlockedCompareExchange64(requested, (LONG64)size, current);

        if (observed == current) {
            return;
        }
        current = observed;
    }
}

static ssize_t vbars_take_reclaim_request(void) {
    return g_devctx
        ? (ssize_t)InterlockedExchange64(
              (volatile LONG64 *)&vbar_reclaim_requested, 0)
        : 0;
}

static size_t vbars_reclaim_at_owner_boundary(ssize_t live_deficit) {
    ssize_t requested;

    /* Reference-safe mode consumes deferred pressure only at the explicit
     * model-switch boundary in vbars_prepare_allocation().  A per-weight fault
     * must not turn the kill switch into a hidden queue synchronize/unmap. */
    if (!vbar_async_reclaim_enabled()) {
        return live_deficit > 0
            ? VBAR_GET_PAGE_NR_UP((size_t)live_deficit) : 0;
    }
    requested = vbars_take_reclaim_request();

    if (requested > 0) {
        log(DEBUG,
            "%s: deferred=%zu MB live=%zu MB\n", __func__,
            (size_t)requested / M,
            live_deficit > 0 ? (size_t)live_deficit / M : 0);
    }
    if (requested > live_deficit) {
        live_deficit = requested;
    }
    return vbars_free_retired(live_deficit);
}

#define VBAR_EVICTION_BATCH_PAGES 64

typedef struct VbarEvictionCandidate {
    uint64_t vbar_identity;
    size_t page_nr;
    CUmemGenericAllocationHandle handle;
    size_t serial;
    uint64_t eviction_generation;
} VbarEvictionCandidate;

static ModelVBAR *vbar_find_identity_locked(uint64_t identity) {
    for (ModelVBAR *mv = lowest_priority.higher;
         mv && mv != &highest_priority; mv = mv->higher) {
        if (mv->identity == identity) {
            return mv;
        }
    }
    return NULL;
}

static void vbar_cancel_candidates(VbarEvictionCandidate *candidates,
                                   size_t count) {
    vbar_state_lock();
    for (size_t index = 0; index < count; ++index) {
        VbarEvictionCandidate *candidate = &candidates[index];
        ModelVBAR *mv = vbar_find_identity_locked(candidate->vbar_identity);

        if (!mv || candidate->page_nr >= mv->nr_pages) {
            continue;
        }
        ResidentPage *rp = &mv->residency_map[candidate->page_nr];
        if (rp->evicting &&
            rp->eviction_generation == candidate->eviction_generation) {
            vbar_page_cancel_eviction(rp);
        }
    }
    vbar_state_unlock();
}

static size_t vbar_commit_candidates(VbarEvictionCandidate *candidates,
                                     size_t count) {
    size_t freed = 0;

    vbar_state_lock();
    for (size_t index = 0; index < count; ++index) {
        VbarEvictionCandidate *candidate = &candidates[index];
        ModelVBAR *mv = vbar_find_identity_locked(candidate->vbar_identity);

        if (!mv || candidate->page_nr >= mv->nr_pages) {
            continue;
        }
        ResidentPage *rp = &mv->residency_map[candidate->page_nr];
        if (!rp->evicting || rp->pin_count ||
            !vbar_page_is_mapped(rp) ||
            rp->handle != candidate->handle ||
            rp->serial != candidate->serial ||
            rp->eviction_generation != candidate->eviction_generation) {
            if (rp->evicting &&
                rp->eviction_generation == candidate->eviction_generation) {
                vbar_page_cancel_eviction(rp);
            }
            continue;
        }

        /* The metadata lock now freezes fault/register/unpin across final
         * validation and physical unmap. */
        rp->evicting = 0;
        if (mod1(mv, candidate->page_nr, true, false)) {
            freed++;
        }
    }
    vbar_state_unlock();
    return freed;
}

static size_t vbar_freeze_retired_candidates(
    size_t pages_needed, const uint64_t *completed, size_t completed_count,
    ModelVBAR *preserved, VbarEvictionCandidate *candidates,
    size_t capacity) {
    size_t count = 0;

    if (!vbar_state_try_lock()) {
        return 0;
    }
    one_time_setup();
    vbars_dirty = true;
    for (ModelVBAR *mv = lowest_priority.higher;
         count < capacity && count < pages_needed && mv != &highest_priority;
         mv = mv->higher) {
        size_t floor;
        size_t page_nr;

        if (mv == preserved) {
            continue;
        }
        floor = mv->watermark < mv->watermark_limit
            ? mv->watermark : mv->watermark_limit;
        page_nr = mv->nr_pages;
        while (count < capacity && count < pages_needed && page_nr > floor) {
            ResidentPage *rp = &mv->residency_map[--page_nr];

            if (!vbar_page_is_mapped(rp) || rp->pin_count || rp->evicting ||
                !vbar_retire_complete(rp, completed, completed_count)) {
                continue;
            }
            rp->evicting = 1;
            rp->eviction_generation++;
            candidates[count++] = (VbarEvictionCandidate){
                .vbar_identity = mv->identity,
                .page_nr = page_nr,
                .handle = rp->handle,
                .serial = rp->serial,
                .eviction_generation = rp->eviction_generation,
            };
        }
    }
    vbar_state_unlock();
    return count;
}

static size_t vbar_freeze_reference_candidates(
    size_t pages_needed, ModelVBAR *preserved,
    VbarEvictionCandidate *candidates, size_t capacity) {
    size_t count = 0;

    vbar_state_lock();
    one_time_setup();
    vbars_dirty = true;
    for (ModelVBAR *mv = lowest_priority.higher;
         count < capacity && count < pages_needed && mv != &highest_priority;
         mv = mv->higher) {
        size_t page_nr = mv->nr_pages;

        if (mv == preserved) {
            continue;
        }
        while (count < capacity && count < pages_needed &&
               page_nr > mv->watermark_limit) {
            ResidentPage *rp = &mv->residency_map[--page_nr];

            /* Unknown includes graph capture and a failed/unregistered queue.
             * Even the reference sync cannot enumerate such a consumer. */
            if (!vbar_page_is_mapped(rp) || rp->pin_count || rp->evicting ||
                rp->retire_unknown) {
                continue;
            }
            rp->evicting = 1;
            rp->eviction_generation++;
            candidates[count++] = (VbarEvictionCandidate){
                .vbar_identity = mv->identity,
                .page_nr = page_nr,
                .handle = rp->handle,
                .serial = rp->serial,
                .eviction_generation = rp->eviction_generation,
            };
        }
    }
    vbar_state_unlock();
    return count;
}

static size_t vbars_free_synchronized_except(ssize_t size,
                                             ModelVBAR *preserved) {
    size_t pages_needed;

    if (size <= 0) {
        return 0;
    }
    pages_needed = VBAR_GET_PAGE_NR_UP((size_t)size);
    while (pages_needed) {
        VbarEvictionCandidate candidates[VBAR_EVICTION_BATCH_PAGES];
        size_t count = vbar_freeze_reference_candidates(
            pages_needed, preserved, candidates,
            VBAR_EVICTION_BATCH_PAGES);
        size_t freed;

        if (!count) {
            break;
        }
        if (!CHECK_CU_ERROR(cuCtxSynchronize())) {
            vbar_cancel_candidates(candidates, count);
            break;
        }
        freed = vbar_commit_candidates(candidates, count);
        if (!freed) {
            break;
        }
        pages_needed -= freed;
    }
    return pages_needed;
}

static size_t vbar_freeze_model_candidates(
    uint64_t identity, size_t first, size_t last, size_t pages_needed,
    VbarEvictionCandidate *candidates, size_t capacity) {
    size_t count = 0;

    vbar_state_lock();
    {
        ModelVBAR *mv = vbar_find_identity_locked(identity);
        if (mv && !mv->closing) {
            if (last > mv->nr_pages) {
                last = mv->nr_pages;
            }
            while (count < capacity && count < pages_needed && last > first) {
                ResidentPage *rp = &mv->residency_map[--last];

                if (!vbar_page_is_mapped(rp) || rp->pin_count ||
                    rp->evicting || rp->retire_unknown) {
                    continue;
                }
                rp->evicting = 1;
                rp->eviction_generation++;
                candidates[count++] = (VbarEvictionCandidate){
                    .vbar_identity = mv->identity,
                    .page_nr = last,
                    .handle = rp->handle,
                    .serial = rp->serial,
                    .eviction_generation = rp->eviction_generation,
                };
            }
        }
    }
    vbar_state_unlock();
    return count;
}

static size_t vbar_free_model_range_synchronized(
    uint64_t identity, size_t first, size_t last, size_t pages_needed) {
    size_t freed_total = 0;

    while (pages_needed) {
        VbarEvictionCandidate candidates[VBAR_EVICTION_BATCH_PAGES];
        size_t count = vbar_freeze_model_candidates(
            identity, first, last, pages_needed, candidates,
            VBAR_EVICTION_BATCH_PAGES);
        size_t freed;

        if (!count) {
            break;
        }
        if (!CHECK_CU_ERROR(cuCtxSynchronize())) {
            vbar_cancel_candidates(candidates, count);
            break;
        }
        freed = vbar_commit_candidates(candidates, count);
        if (!freed) {
            break;
        }
        pages_needed -= freed;
        freed_total += freed;
    }
    return freed_total;
}

/* Reclaim without ever waiting on the compute queue.
 *
 * This runs only from a VBAR/model-owner boundary or immediately before the
 * direct file reader submits a host copy. It never runs from a native
 * allocation or driver callback. It is still best effort: release every page
 * that is provably idle and skip the rest rather than waiting for one.
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
static size_t vbars_free_retired_except(ssize_t size,
                                        ModelVBAR *preserved) {
    size_t pages_needed;
    uint64_t completed[AIMDO_XPU_RETIRE_MAX_QUEUES] = {0};
    size_t completed_count;

    if (size <= 0) {
        return 0;
    }

    pages_needed = VBAR_GET_PAGE_NR_UP((size_t)size);
    if (!vbar_async_reclaim_enabled()) {
        return pages_needed;
    }
    completed_count = aimdo_xpu_retire_snapshot(
        completed, AIMDO_XPU_RETIRE_MAX_QUEUES, true);
    while (pages_needed) {
        VbarEvictionCandidate candidates[VBAR_EVICTION_BATCH_PAGES];
        size_t count = vbar_freeze_retired_candidates(
            pages_needed, completed, completed_count, preserved, candidates,
            VBAR_EVICTION_BATCH_PAGES);
        size_t freed;

        if (!count) {
            break;
        }
        /* Event queries above proved the registered consumers complete.  The
         * separate commit phase still revalidates all mutable page identity
         * before physical unmap. */
        freed = vbar_commit_candidates(candidates, count);
        if (!freed) {
            break;
        }
        pages_needed -= freed;
    }

    if (pages_needed) {
        log(DEBUG, "%s: %zu pages still required after non-blocking reclaim\n",
            __func__, pages_needed);
    }

    return pages_needed;
}

size_t vbars_free_retired(ssize_t size) {
    return vbars_free_retired_except(size, NULL);
}

/* Pressure/OOM recovery follows the same rule as a caching allocator cache
 * flush: once a real shortage exists, release every block whose recorded
 * consumers have completed.  Unlike vbars_free_retired(), this is not an
 * arbitrary byte target.  It still never waits and still fails closed for a
 * pinned, unknown or incomplete page. */
size_t vbars_free_all_retired(void) {
    uint64_t completed[AIMDO_XPU_RETIRE_MAX_QUEUES] = {0};
    size_t completed_count;
    size_t freed_total = 0;

    if (!vbar_async_reclaim_enabled()) {
        return 0;
    }
    completed_count = aimdo_xpu_retire_snapshot(
        completed, AIMDO_XPU_RETIRE_MAX_QUEUES, true);
    for (;;) {
        VbarEvictionCandidate candidates[VBAR_EVICTION_BATCH_PAGES];
        size_t count = vbar_freeze_retired_candidates(
            VBAR_EVICTION_BATCH_PAGES, completed, completed_count, NULL,
            candidates, VBAR_EVICTION_BATCH_PAGES);
        size_t freed;

        if (!count) {
            break;
        }
        freed = vbar_commit_candidates(candidates, count);
        freed_total += freed;
        if (!freed) {
            break;
        }
    }
    return freed_total;
}
#endif

static inline size_t move_cursor_to_absent(ModelVBAR *mv, size_t cursor) {
    while (cursor < mv->watermark &&
           vbar_page_is_mapped(&mv->residency_map[cursor])) {
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

            if (!synced && vbar_page_is_mapped(rp) && rp->pin_count == 0) {
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
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    mv->identity = ++vbar_identity_counter;
#endif
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
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    size_t old_watermark;
    uint64_t identity;
#endif

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    vbar_state_lock();
    vbars_dirty = true;

    if (watermark > mv->nr_pages) {
        watermark = mv->nr_pages;
    }

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    old_watermark = mv->watermark;
    identity = mv->identity;
    mv->watermark = watermark;
    vbar_state_unlock();
    if (watermark < old_watermark) {
        (void)vbar_free_model_range_synchronized(
            identity, watermark, old_watermark,
            old_watermark - watermark);
    }
#else
    if (watermark < mv->watermark) {
        CHECK_CU(cuCtxSynchronize());
        for (size_t page_nr = watermark; page_nr < mv->watermark; page_nr++) {
            mod1(mv, page_nr, true, false);
        }
    }

    mv->watermark = watermark;
    vbar_state_unlock();
#endif
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
    ssize_t reclaim;

    set_devctx((AimdoContext *)devctx);
    /*
     * Windows cannot evict from the Level Zero allocation callback because
     * doing so waits re-entrantly on the same SYCL queue.  Its model-boundary
     * reclaim is therefore only a prediction based on native allocator
     * history.  It may discard lower-priority models, but it must not discard
     * pages from the active model on the strength of that prediction alone.
     * A later vbar_fault() still applies exact live pressure to every VBAR.
     */
    reclaim = budget_deficit((size_t)size);
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    {
        ssize_t requested = vbars_take_reclaim_request();

        if (requested > reclaim) {
            reclaim = requested;
        }
    }
    if (vbar_async_reclaim_enabled()) {
        (void)vbars_free_retired_except(reclaim, (ModelVBAR *)vbar);
    } else {
        (void)vbars_free_synchronized_except(reclaim, (ModelVBAR *)vbar);
    }
#else
    vbars_free_except(reclaim, (ModelVBAR *)vbar);
#endif
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

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    if (mv->closing) {
        return VBAR_FAULT_ERROR;
    }
#endif

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
     * misses multiplied. Merge live pressure with requests recorded by native
     * allocator/copy callbacks, then reclaim here: this is the model owner's
     * call stack, outside the allocator/UMF critical section. */
    /* Windows consumes this pressure before taking the VBAR metadata lock in
     * vbar_fault(); doing it here would retain the outer recursive lock across
     * both phases and defeat fault/eviction revalidation. */
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
            if (!vbar_page_is_mapped(&mv->residency_map[page_nr])) {
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

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        if (rp->handle && rp->mapped == VBAR_MAPPING_UNKNOWN) {
            log(AIMDO_LOG_ERROR,
                "VBAR page %zu has an indeterminate mapping state\n",
                (size_t)page_nr);
            vbar_unpin_range(
                mv, VBAR_GET_PAGE_NR(offset), (size_t)page_nr);
            return VBAR_FAULT_ERROR;
        }
        if (rp->handle && rp->mapped == VBAR_MAPPING_UNMAPPED) {
            /* A prior unmap succeeded but physical destroy and immediate
             * remap both failed.  Reuse that still-owned physical handle
             * before considering a new allocation; never overwrite/leak it. */
            if (vbar_restore_page_mapping(mv, (size_t)page_nr, true) !=
                VBAR_MAPPING_MAPPED) {
                log(AIMDO_LOG_ERROR,
                    "VBAR page %zu has an unrecoverable unmapped physical handle\n",
                    (size_t)page_nr);
                vbar_unpin_range(
                    mv, VBAR_GET_PAGE_NR(offset), (size_t)page_nr);
                return VBAR_FAULT_ERROR;
            }
        }
#endif
        if (vbar_page_is_mapped(rp)) {
            /* Pin before anything else in this fault can reclaim. The pin loop
             * used to run only after every page was mapped, which left the
             * pages this fault had already mapped unpinned and therefore
             * eligible for the reclaim calls further down. Those calls then
             * released a page that the trailing loop still marked as pinned,
             * producing a pinned page with no physical backing: the copy into
             * it failed with OUT_OF_DEVICE_MEMORY while the device still had
             * gigabytes free. */
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
            /* A fault is allowed to cancel a frozen candidate until physical
             * unmap begins.  The eviction generation makes the later
             * revalidation reject its stale candidate. */
            vbar_page_cancel_eviction(rp);
#endif
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
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
            (err = vbar_map_new_page(mv, (size_t)page_nr)) != CUDA_SUCCESS) {
#else
            (err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device,
                                 &rp->handle)) != CUDA_SUCCESS) {
#endif
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
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
            if ((err = vbar_map_new_page(mv, (size_t)page_nr)) !=
                CUDA_SUCCESS) {
#else
            if ((err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device,
                                     &rp->handle)) != CUDA_SUCCESS) {
#endif
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
                        (size_t)AIMDO_XPU_WDDM_RECLAIM_FLOOR / M);
                    (void)vbars_free_retired(AIMDO_XPU_WDDM_RECLAIM_FLOOR);
                    if (page_end > mv->watermark) {
                        log(DEBUG,
                            "VBAR allocation cancelled after Windows XPU retry reclaim\n");
                        vbar_unpin_range(mv, VBAR_GET_PAGE_NR(offset), page_nr);
                        return VBAR_FAULT_OOM;
                    }
                    err = vbar_map_new_page(mv, (size_t)page_nr);
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
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        rp->mapped = VBAR_MAPPING_MAPPED;
        rp->evicting = 0;
#endif
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
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    (void)vbars_reclaim_at_owner_boundary(budget_deficit(0));
#endif
    vbar_state_lock();
    result = vbar_fault_locked(devctx, vbar, offset, size, signature);
    vbar_state_unlock();
    return result;
}

SHARED_EXPORT
void vbar_unpin(void *devctx, void *vbar, uint64_t offset, uint64_t size) {
    vbar_unpin_stream(devctx, vbar, offset, size, 0);
}

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
static uint64_t vbar_consumer_dependency(uint64_t stream, int device,
                                         bool *known) {
    void *queue = (void *)(uintptr_t)stream;

    *known = false;
    if (!queue) {
        return 0;
    }
    if (!vbar_async_reclaim_enabled()) {
        /* Reference-safe mode still owns a stable queue copy so its explicit
         * model-boundary synchronize can drain the consumer.  It must create
         * no token, barrier or event query. */
        *known = aimdo_xpu_register_consumer_queue(queue, device);
        return 0;
    }
    {
        uint64_t token = aimdo_xpu_retire_token_current(queue, device);
        *known = token != 0;
        return token;
    }
}

SHARED_EXPORT
void vbar_register_consumer_stream(void *devctx, void *vbar, uint64_t offset,
                                   uint64_t size, uint64_t stream) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    bool known;
    uint64_t retirement_token;
    size_t page_end;

    set_devctx((AimdoContext *)devctx);
    retirement_token = vbar_consumer_dependency(stream, mv->device, &known);
    vbar_state_lock();
    page_end = VBAR_GET_PAGE_NR_UP(offset + size);
    for (size_t page_nr = VBAR_GET_PAGE_NR(offset);
         page_nr < page_end && page_nr < mv->nr_pages; ++page_nr) {
        ResidentPage *rp = &mv->residency_map[page_nr];

        if (!vbar_page_is_mapped(rp) || !rp->pin_count) {
            continue;
        }
        vbar_page_cancel_eviction(rp);
        if (!known) {
            rp->retire_unknown = 1;
        } else if (vbar_async_reclaim_enabled()) {
            vbar_retire_record(rp, retirement_token);
        }
    }
    vbar_state_unlock();
}
#endif

SHARED_EXPORT
void vbar_unpin_stream(void *devctx, void *vbar, uint64_t offset, uint64_t size,
                       uint64_t stream) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    uint64_t retirement_token;
    bool consumer_known;
#endif

    set_devctx((AimdoContext *)devctx);

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* Capture the actual consuming queue after its operator was submitted.
     * The token call owns only the retire lock; taking it before the page lock
     * preserves the retire-lock -> VBAR-lock order used by pressure scans. A
     * missing/overflowed queue returns zero and makes the page fail closed. */
    retirement_token = vbar_consumer_dependency(
        stream, mv->device, &consumer_known);
#else
    (void)stream;
#endif
    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));
    vbar_state_lock();
    vbars_dirty = true;
    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    /* Windows never frees a page from here. Pages above the watermark retain
     * per-queue completion tokens and are released later by non-blocking
     * reclaim, so neither synchronize below is needed. */
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
        /* Record every concurrent consumer, not just the final unpin. The last
         * caller to drop pin_count cannot prove that earlier queues have also
         * completed. Publishing dependencies before decrementing makes the
         * transition to idle atomic to the reclaim scan. */
        if (rp->pin_count) {
            vbar_page_cancel_eviction(rp);
            if (!consumer_known) {
                rp->retire_unknown = 1;
            } else if (vbar_async_reclaim_enabled()) {
                vbar_retire_record(rp, retirement_token);
            }
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

    vbar_state_unlock();
}

SHARED_EXPORT
void vbar_free(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: vbar=%p\n", __func__, vbar);
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    vbar_state_lock();
    if (mv->closing) {
        vbar_state_unlock();
        return;
    }
    mv->closing = 1;
    vbar_state_unlock();

    /* Closing prevents new faults.  The first drain covers existing submitted
     * work; the pin check proves every owner published its dependency; the
     * second drain covers a consumer that submitted immediately before its
     * final unpin raced with the first drain. */
    if (!CHECK_CU_ERROR(cuCtxSynchronize())) {
        return;
    }
    vbar_state_lock();
    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        ResidentPage *rp = &mv->residency_map[page_nr];
        if (rp->pin_count || rp->retire_unknown) {
            log(AIMDO_LOG_ERROR,
                "%s: VBAR %p page %zu still has an unknown/live consumer; "
                "keeping its mapping fail-closed\n",
                __func__, vbar, (size_t)page_nr);
            vbar_state_unlock();
            return;
        }
    }
    vbar_state_unlock();
    if (!CHECK_CU_ERROR(cuCtxSynchronize())) {
        return;
    }

    vbar_state_lock();
    vbars_dirty = true;
    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        ResidentPage *rp = &mv->residency_map[page_nr];
        if (rp->pin_count || rp->retire_unknown) {
            log(AIMDO_LOG_ERROR,
                "%s: VBAR %p page %zu changed during teardown; keeping it "
                "fail-closed\n",
                __func__, vbar, (size_t)page_nr);
            vbar_state_unlock();
            return;
        }
        vbar_page_cancel_eviction(rp);
        (void)mod1(mv, page_nr, true, true);
        if (rp->handle) {
            log(AIMDO_LOG_ERROR,
                "%s: VBAR %p page %zu physical release failed; preserving "
                "the virtual range\n",
                __func__, vbar, (size_t)page_nr);
            vbar_state_unlock();
            return;
        }
    }
    if (!CHECK_CU_ERROR(cuMemAddressFree(
            mv->vbar, (size_t)mv->nr_pages * VBAR_PAGE_SIZE))) {
        vbar_state_unlock();
        return;
    }
    remove_vbar(mv);
    vbar_state_unlock();
    free(mv);
#else
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
#endif
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
        out[i] = (vbar_page_is_mapped(rp) ? 1 : 0) |
                 (rp->pin_count ? 2 : 0);
    }
    vbar_state_unlock();
}

SHARED_EXPORT
uint64_t vbar_free_memory(void *devctx, void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t pages_to_free = VBAR_GET_PAGE_NR_UP(size);
    size_t pages_freed = 0;
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    size_t old_watermark;
    size_t target_watermark;
    size_t available_pages;
    uint64_t identity;
#endif

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s (start): size=%lldk\n", __func__, (ull)size);
    vbar_state_lock();
    vbars_dirty = true;

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
    old_watermark = mv->watermark;
    available_pages = 0;
    target_watermark = old_watermark;
    /* ``size`` requests resident bytes, not address-range bytes. Scan through
     * already-absent or non-reclaimable high pages just as the original
     * watermark loop did, and stop only after enough actual candidates were
     * found or the limit was reached. */
    while (target_watermark > mv->watermark_limit &&
           available_pages < pages_to_free) {
        ResidentPage *rp = &mv->residency_map[--target_watermark];

        if (vbar_page_is_mapped(rp) && !rp->pin_count && !rp->evicting &&
            !rp->retire_unknown) {
            available_pages++;
        }
    }
    identity = mv->identity;
    mv->watermark = target_watermark;
    vbar_state_unlock();
    pages_freed = vbar_free_model_range_synchronized(
        identity, target_watermark, old_watermark, pages_to_free);
#else
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
#endif
    return (uint64_t)pages_freed * VBAR_PAGE_SIZE;
}
