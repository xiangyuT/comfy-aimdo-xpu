#include <c10/xpu/XPUCachingAllocator.h>
#include <sycl/sycl.hpp>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
struct Buffer { size_t bytes; std::shared_ptr<sycl::queue> queue; bool aimdo_owned; };
std::mutex mutex;
std::unordered_map<void *, Buffer> buffers;
}

extern "C" void *retirement_allocate(size_t bytes, uint64_t stream, bool aimdo_owned) {
    try {
        auto queue = std::make_shared<sycl::queue>(*reinterpret_cast<sycl::queue *>(stream));
        void *pointer = nullptr;
        if (aimdo_owned) {
            using Alloc = void *(*)(size_t, int, sycl::queue *);
            auto alloc = reinterpret_cast<Alloc>(dlsym(RTLD_DEFAULT, "xpu_alloc_fn"));
            if (!alloc) return nullptr;
            pointer = alloc(bytes, 0, queue.get());
        } else {
            pointer = sycl::malloc_device(bytes, *queue);
        }
        if (!pointer) return nullptr;
        std::lock_guard<std::mutex> guard(mutex);
        buffers.emplace(pointer, Buffer{bytes, queue, aimdo_owned});
        return pointer;
    } catch (...) { return nullptr; }
}

extern "C" bool retirement_write(void *pointer, const void *data, size_t bytes) {
    try {
        std::lock_guard<std::mutex> guard(mutex);
        const auto &buffer = buffers.at(pointer);
        if (bytes != buffer.bytes) return false;
        buffer.queue->memcpy(pointer, data, bytes).wait_and_throw();
        return true;
    } catch (...) { return false; }
}

extern "C" bool retirement_read(void *pointer, void *data, size_t bytes) {
    try {
        std::lock_guard<std::mutex> guard(mutex);
        const auto &buffer = buffers.at(pointer);
        if (bytes != buffer.bytes) return false;
        buffer.queue->memcpy(data, pointer, bytes).wait_and_throw();
        return true;
    } catch (...) { return false; }
}

extern "C" bool retirement_free(void *pointer) {
    try {
        std::lock_guard<std::mutex> guard(mutex);
        const auto &buffer = buffers.at(pointer);
        buffer.queue->wait_and_throw();
        if (buffer.aimdo_owned) {
            using Free = void (*)(void *, size_t, int, sycl::queue *);
            using Empty = bool (*)(bool);
            auto free = reinterpret_cast<Free>(dlsym(RTLD_DEFAULT, "xpu_free_fn"));
            auto empty = reinterpret_cast<Empty>(dlsym(RTLD_DEFAULT, "xpu_allocator_empty_cache"));
            if (!free || !empty) return false;
            free(pointer, buffer.bytes, 0, buffer.queue.get());
            if (!empty(true)) return false;
        } else {
            sycl::free(pointer, buffer.queue->get_context());
        }
        buffers.erase(pointer);
        return true;
    } catch (...) { return false; }
}

extern "C" void *retirement_raw_allocate(size_t bytes) {
    try { return c10::xpu::XPUCachingAllocator::get()->raw_alloc(bytes); }
    catch (...) { return nullptr; }
}

extern "C" bool retirement_raw_free(void *pointer) {
    try { c10::xpu::XPUCachingAllocator::get()->raw_delete(pointer); return true; }
    catch (...) { return false; }
}
