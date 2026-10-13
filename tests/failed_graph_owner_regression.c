/* Direct tests of the actual core's failed-owner bookkeeping, without a GPU. */
#include "../src/malloc-graph.c"

int main(void) {
    MallocGraph *g = calloc(1, sizeof(*g));
    VirtualRange base = {.address = 0x2000000000ull};
    VirtualRange small_base = {.address = 0x10000000ull};
    State state = {0};
    Event small_event = {.type = EV_ALLOC_SMALL};
    Event large_event = {.type = EV_ALLOC};
    SmallRange range = {.offset = 4096, .allocation = &small_event};
    g->base = &base;
    g->small_base = &small_base;
    g->state = &state;
    g->small_ranges = &range;
    g->allocations.virtual_pages[3] = (VirtualPage){.va_span = 2, .allocation = &large_event};
    int result = -1;
    assert(!retire_failed_graph_owner(g, small_base.address + 4096, &result));
    g->failed = true;
    assert(!retire_failed_graph_owner(g, small_base.address + 4097, &result));
    assert(!retire_failed_graph_owner(g, base.address + 4 * MG_PAGE, &result));
    assert(!retire_failed_graph_owner(g, base.address + 3 * MG_PAGE + 1, &result));
    assert(result == -1 && !small_event.freed_out_of_scope && !large_event.freed_out_of_scope);
    assert(retire_failed_graph_owner(g, small_base.address + 4096, &result));
    assert(result == 0 && small_event.freed_out_of_scope);
    assert(!retire_failed_graph_owner(g, small_base.address + 4096, &result));
    assert(retire_failed_graph_owner(g, base.address + 3 * MG_PAGE, &result));
    assert(large_event.freed_out_of_scope && g->allocations.virtual_pages[3].va_span == 2);
    assert(g->failed && g->state == &state && g->small_ranges == &range);
    small_event.freed_out_of_scope = false;
    g->handoff_attempted = true;
    assert(!retire_failed_graph_owner(g, small_base.address + 4096, &result));
    g->handoff_attempted = false;
    g->state = NULL;
    assert(!retire_failed_graph_owner(g, small_base.address + 4096, &result));
    free(g);
    return 0;
}
