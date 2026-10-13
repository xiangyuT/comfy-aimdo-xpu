/* Execute the real core with a deterministic VMM dispatch, without a GPU. */
#include "../src/model-vbar.c"

_Thread_local AimdoContext *g_devctx;
AimdoCudaDispatch g_cuda;
int log_level = -1;
uint64_t log_shot_counter;
int64_t simple_vram_headroom;
static size_t creates, physical, mappings, next_handle;
static size_t allowed = SIZE_MAX;
static int fail_release, fail_unmap, fail_sync, fail_map;

void aimdo_log(int level, const char *file, int line, const char *format, ...) {}
void log_reset_shots(void) {}
bool cuda_budget_deficit(const char **method) {
    deficit_sync = -(ssize_t)vram_capacity;
    total_vram_last_check = total_vram_usage;
    return false;
}
bool malloc_graph_sync_paused(void) { return false; }
static CUresult error_string(CUresult code, const char **out) { *out = "injected"; return 0; }
static CUresult sync_context(void) { return fail_sync ? 999 : 0; }
static CUresult create(CUmemGenericAllocationHandle *out, size_t bytes,
                        const CUmemAllocationProp *prop, unsigned long long flags) {
    if (creates++ >= allowed) return CUDA_ERROR_OUT_OF_MEMORY;
    *out = ++next_handle;
    physical++;
    return 0;
}
static CUresult map(CUdeviceptr address, size_t bytes, size_t offset,
                     CUmemGenericAllocationHandle handle, unsigned long long flags) {
    if (fail_map) return 999;
    mappings++;
    return 0;
}
static CUresult access_page(CUdeviceptr address, size_t bytes,
                            const CUmemAccessDesc *desc, size_t count) { return 0; }
static CUresult unmap(CUdeviceptr address, size_t bytes) {
    if (fail_unmap) { fail_unmap--; return CUDA_ERROR_OUT_OF_MEMORY; }
    assert(mappings);
    mappings--;
    return 0;
}
static CUresult release(CUmemGenericAllocationHandle handle) {
    if (fail_release) { fail_release--; return CUDA_ERROR_OUT_OF_MEMORY; }
    assert(physical);
    physical--;
    return 0;
}
static CUresult free_va(CUdeviceptr address, size_t bytes) { assert(!mappings); return 0; }

int main(int argc, char **argv) {
    assert(argc == 2);
    int scenario = atoi(argv[1]);
    AimdoContext context = { ._vram_capacity = 1ull << 40 };
    g_devctx = &context;
    g_cuda = (AimdoCudaDispatch){
        .p_cuGetErrorString = error_string, .p_cuCtxSynchronize = sync_context,
        .p_cuMemCreate = create, .p_cuMemMap = map, .p_cuMemSetAccess = access_page,
        .p_cuMemUnmap = unmap, .p_cuMemRelease = release, .p_cuMemAddressFree = free_va,
    };
    one_time_setup();
    assert(vbars_can_deinit(&context));
    ModelVBAR *vbar = calloc(1, sizeof(*vbar) + 2 * sizeof(ResidentPage));
    vbar->vbar = 0x10000000;
    vbar->nr_pages = vbar->watermark = 2;
    insert_vbar(vbar);
    assert(!vbars_can_deinit(&context));
    uint32_t signature[2];
    bool preexisting = scenario == 1;
    if (preexisting) assert(vbar_fault(&context, vbar, 0, VBAR_PAGE_SIZE, signature) == 0);
    allowed = 1;
    if (scenario == 2) vbar->watermark_limit = 2;  /* Retry still physically OOM. */
    if (scenario == 3) fail_unmap = 1;
    if (scenario == 4 || scenario == 5) fail_release = 1;
    if (scenario == 5) fail_map = 1;  /* Set after the initial first-page map below. */
    if (scenario == 5 || scenario == 6) {
        fail_map = 0;
        assert(vbar_fault(&context, vbar, 0, VBAR_PAGE_SIZE, signature) == 0);
        vbar_unpin(&context, vbar, 0, VBAR_PAGE_SIZE);
        if (scenario == 5) fail_map = 1;
        if (scenario == 6) fail_sync = 1;
    }
    int result = vbar_fault(&context, vbar, 0, 2 * VBAR_PAGE_SIZE, signature);
    ResidentPage *first = &vbar->residency_map[0];
    if (scenario == 0) {
        assert(result == 1 && !first->handle && !physical && !mappings);
        assert(!total_vram_usage && !vbar->resident_count && !vbar->watermark);
    } else if (scenario == 1) {
        assert(result == 1 && first->pin_count == 1 && first->handle);
        assert(physical == 1 && mappings == 1);
        vbar_unpin(&context, vbar, 0, VBAR_PAGE_SIZE);
    } else if (scenario == 2) {
        assert(result == 1 && first->handle && !first->pin_count);
        assert(vbar->watermark == 2 && creates == 3);
    } else if (scenario == 3 || scenario == 4) {
        assert(result == 2 && first->handle && !first->pin_count);
        assert(first->mapped == VBAR_MAPPING_MAPPED && !first->retire_unknown);
        assert(physical == 1 && mappings == 1 && total_vram_usage == VBAR_PAGE_SIZE);
    } else {
        assert(result == 2 && first->handle && first->retire_unknown);
        assert(physical == 1 && total_vram_usage == VBAR_PAGE_SIZE);
        vbar_free(&context, vbar);
        assert(!vbars_can_deinit(&context));  /* Uncertain owners are never discarded. */
        if (scenario == 5) assert(!mappings && first->mapped == VBAR_MAPPING_UNMAPPED);
        if (scenario == 6) assert(mappings == 1);
        return 0;  /* Process exit is the required terminal cleanup boundary. */
    }
    allowed = SIZE_MAX;
    vbar_prioritize(&context, vbar, 0);
    assert(vbar_fault(&context, vbar, 0, 2 * VBAR_PAGE_SIZE, signature) == 0);
    assert(physical == 2 && mappings == 2 && total_vram_usage == 2 * VBAR_PAGE_SIZE);
    vbar_unpin(&context, vbar, 0, 2 * VBAR_PAGE_SIZE);
    vbar_free(&context, vbar);
    assert(!physical && !mappings && !total_vram_usage && vbars_can_deinit(&context));
    free(highest_priority_p);
    return 0;
}
