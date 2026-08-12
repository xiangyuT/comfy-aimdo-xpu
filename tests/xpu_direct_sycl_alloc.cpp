#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

extern "C" __attribute__((visibility("default"))) bool
aimdo_test_direct_sycl_alloc(uint64_t queue_pointer, size_t size) {
    if (!queue_pointer || !size) {
        return false;
    }
    auto *queue = reinterpret_cast<sycl::queue *>(queue_pointer);
    void *pointer = nullptr;
    try {
        pointer = sycl::malloc_device(size, *queue);
        if (!pointer) {
            return false;
        }
        queue->memset(pointer, 0x5a, size).wait_and_throw();
        sycl::free(pointer, queue->get_context());
        return true;
    } catch (...) {
        if (pointer) {
            try {
                sycl::free(pointer, queue->get_context());
            } catch (...) {
            }
        }
        return false;
    }
}
