/* Exercise the production core/VMM references with injected driver callbacks.
 * No SYCL device or real GPU allocation is used by this CPU state-machine test. */
#include "../src/malloc-graph.c"
#include "../src/vmm-ref.c"

AimdoCudaDispatch g_cuda;
_Thread_local AimdoContext *g_devctx;
int64_t simple_vram_headroom;
static AimdoContext context;
static CUdeviceptr next_address = 0x1000000000ULL;
static uint64_t next_handle = 1;
static int sync_calls, mutations;
static CUresult sync_error;
static bool queued_work;

bool set_devctx_for_device(int device) {
    if (device != context._device_id) return false;
    g_devctx = &context;
    return true;
}
bool cuda_budget_deficit(const char **method) { return false; }
size_t vbars_free(ssize_t bytes) { return 0; }
void allocations_lock(void) {}
void allocations_unlock(void) {}
bool rogue_exists(CUdeviceptr ptr) { return false; }
bool free_rogue(CUdeviceptr ptr, int *result) { return false; }
bool rogue_candidate_freed(CUdeviceptr ptr) { return false; }
bool register_rogue_candidate(CUdeviceptr ptr) { return true; }
void unregister_rogue_candidate(CUdeviceptr ptr) {}
RogueHandoff handoff_rogue(VirtualRange *range, CUdeviceptr ptr,
                         PhysicalPage **pages, size_t count) {
    assert(!"unexpected escaped owner in CPU replay-free fixture");
    return ROGUE_HANDOFF_ERROR;
}

static CUresult CUDAAPI sync_context(void) {
    assert(g_devctx == &context);
    sync_calls++;
    if (sync_error) return sync_error;
    queued_work = false;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI reserve_address(CUdeviceptr *ptr, size_t bytes,
        size_t alignment, CUdeviceptr hint, unsigned long long flags) {
    if (!alignment) alignment = MG_PAGE;
    next_address = ALIGN_UP(next_address, alignment);
    *ptr = next_address;
    next_address += bytes + MG_PAGE;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI free_address(CUdeviceptr address, size_t bytes) {
    assert(!queued_work);
    mutations++;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI create_physical(CUmemGenericAllocationHandle *handle,
        size_t bytes, const CUmemAllocationProp *prop, unsigned long long flags) {
    assert(!queued_work);
    *handle = next_handle++;
    mutations++;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI release_physical(CUmemGenericAllocationHandle handle) {
    assert(!queued_work);
    mutations++;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI map_physical(CUdeviceptr address, size_t bytes,
        size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    assert(!queued_work);
    mutations++;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI unmap_physical(CUdeviceptr address, size_t bytes) {
    assert(!queued_work);
    mutations++;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI set_access(CUdeviceptr address, size_t bytes,
        const CUmemAccessDesc *descs, size_t count) { return CUDA_SUCCESS; }

static MallocGraph *capture(void) {
    MallocGraph *g = malloc_graph_create(&context, (CUstream)1, false);
    assert(g);
    CUdeviceptr small, large;
    int result = 73;
    assert(malloc_graph_alloc(&small, 64, (CUstream)1) && small);
    assert(!malloc_graph_free_owned_replay(small, (CUstream)1, &result) && result == 73);
    assert(malloc_graph_alloc(&large, MG_PAGE, (CUstream)1) && large);
    assert(malloc_graph_free_owned(large, (CUstream)1, &result) && result == 0);
    assert(malloc_graph_free_owned(small, (CUstream)1, &result) && result == 0);
    assert(malloc_graph_pop(g) == 1 && !g->owner_work_pending);
    return g;
}
static void replay_and_retire(MallocGraph *g) {
    assert(malloc_graph_push(g, NULL));
    CUdeviceptr small, large;
    assert(malloc_graph_alloc(&small, 64, (CUstream)1));
    assert(malloc_graph_alloc(&large, MG_PAGE, (CUstream)1));
    Event *cursor = g->state->cursor;
    int result = 73;
    assert(!malloc_graph_free_owned_replay(large, (CUstream)2, &result));
    assert(!malloc_graph_free_owned_replay(large + 1, (CUstream)1, &result));
    assert(!malloc_graph_free_owned_replay(small, (CUstream)1, &result));
    g->paused = true;
    assert(!malloc_graph_free_owned_replay(large, (CUstream)1, &result));
    g->paused = false;
    assert(cursor == g->state->cursor && result == 73);
    int before_sync = sync_calls, before_mutations = mutations;
    queued_work = true;
    assert(malloc_graph_free_owned_replay(large, (CUstream)1, &result) && result == 0);
    assert(malloc_graph_free_owned_replay(small, (CUstream)1, &result) && result == 0);
    assert(g->owner_work_pending && sync_calls == before_sync && mutations == before_mutations);
}
int main(void) {
    context._device_id = 0;
    context._vram_capacity = 1ULL << 50;
    g_cuda.p_cuCtxSynchronize = sync_context;
    g_cuda.p_cuMemAddressReserve = reserve_address;
    g_cuda.p_cuMemAddressFree = free_address;
    g_cuda.p_cuMemCreate = create_physical;
    g_cuda.p_cuMemRelease = release_physical;
    g_cuda.p_cuMemMap = map_physical;
    g_cuda.p_cuMemUnmap = unmap_physical;
    g_cuda.p_cuMemSetAccess = set_access;

    MallocGraph *g = capture();
    replay_and_retire(g);
    assert(malloc_graph_pop(g) == 1 && queued_work);
    int before = sync_calls;
    assert(malloc_graph_destroy_checked(g));
    assert(sync_calls == before + 1 && !queued_work && context._total_vram_usage == 0);

    g = capture();
    replay_and_retire(g);
    /* A replay miss must drain work before returning to recording/mapping. */
    CUdeviceptr changed;
    before = sync_calls;
    assert(malloc_graph_alloc(&changed, MG_PAGE * 2, (CUstream)1) && changed);
    assert(sync_calls == before + 1 && !queued_work);
    int result;
    assert(malloc_graph_free_owned(changed, (CUstream)1, &result) && result == 0);
    assert(malloc_graph_pop(g) == 2);
    assert(malloc_graph_destroy_checked(g) && context._total_vram_usage == 0);

    g = capture();
    replay_and_retire(g);
    before = sync_calls;
    assert(malloc_graph_abort(g) && sync_calls > before && !queued_work);
    assert(malloc_graph_destroy_checked(g) && context._total_vram_usage == 0);

    g = capture();
    replay_and_retire(g);
    assert(malloc_graph_pop(g) == 1);
    sync_error = CUDA_ERROR_UNKNOWN;
    before = sync_calls;
    int before_mutations = mutations;
    assert(!malloc_graph_destroy_checked(g));
    assert(g->owner_work_pending && g->destroy_terminal && queued_work);
    assert(sync_calls == before + 1 && mutations == before_mutations);
    assert(!malloc_graph_destroy_checked(g) && sync_calls == before + 1);
    assert(malloc_graph_live_handles() == 1); /* Intentional terminal retention. */
    puts("PASS replay-free eligibility, coalesced drain, mismatch, abort, failed fence retention");
    return 0;
}
