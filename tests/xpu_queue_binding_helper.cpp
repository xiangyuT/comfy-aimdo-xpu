extern "C" {
#include "gpu_dispatch.h"
}
#include <dlfcn.h>
#include <sycl/sycl.hpp>

namespace {
AimdoCudaDispatch *dispatch = nullptr;
bool (*select_device)(int) = nullptr;
}

extern "C" bool queue_test_init(const char *path) {
    void *library = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!library) return false;
    dispatch = static_cast<AimdoCudaDispatch *>(dlsym(library, "g_cuda"));
    select_device = reinterpret_cast<bool (*)(int)>(
        dlsym(library, "set_devctx_for_device"));
    return dispatch && select_device;
}

extern "C" void *queue_test_clone(void *source) {
    return new sycl::queue(*static_cast<sycl::queue *>(source));
}

extern "C" void *queue_test_foreign_context(void *source) {
    const auto device = static_cast<sycl::queue *>(source)->get_device();
    return new sycl::queue(sycl::context(device), device,
                          sycl::property::queue::in_order{});
}

extern "C" bool queue_test_same_context(void *a, void *b) {
    return static_cast<sycl::queue *>(a)->get_context() ==
           static_cast<sycl::queue *>(b)->get_context();
}

extern "C" void queue_test_destroy(void *pointer) {
    auto *queue = static_cast<sycl::queue *>(pointer);
    queue->wait_and_throw();
    delete queue;
}

extern "C" int queue_test_event_bind(void *queue) {
    if (!dispatch || !select_device || !select_device(0)) return 999;
    CUevent event = nullptr;
    int result = dispatch->p_cuEventCreate(&event, 0);
    if (result) return result;
    result = dispatch->p_cuEventRecord(event, reinterpret_cast<CUstream>(queue));
    const auto destroyed = dispatch->p_cuEventDestroy(event);
    return result ? result : destroyed;
}

extern "C" int queue_test_sync() {
    if (!dispatch || !select_device || !select_device(0)) return 999;
    return dispatch->p_cuCtxSynchronize();
}

extern "C" int queue_test_copy(void *queue, uint64_t dest, void *source, size_t size) {
    if (!dispatch || !select_device || !select_device(0)) return 999;
    return dispatch->p_cuMemcpyHtoDAsync(dest, source, size,
                                        reinterpret_cast<CUstream>(queue));
}
