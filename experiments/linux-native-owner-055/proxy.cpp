#include <c10/core/Allocator.h>
#include <c10/xpu/XPUCachingAllocator.h>

// Experimental, process-local bridge for a bounded D2 component probe. This
// is not linked into the AIMDO provider and does not enable public record().

#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace {

struct Owner {
    explicit Owner(c10::DataPtr &&value) : kind(Kind::Native), native(std::move(value)) {}
    Owner(void *pointer, size_t bytes, const sycl::queue &queue)
        : kind(Kind::Custom), custom_pointer(pointer), custom_bytes(bytes),
          custom_queue(queue) {}
    Owner(uint64_t pointer, size_t bytes, const sycl::queue &queue,
          uint64_t stream, int device)
        : kind(Kind::Compiler), custom_pointer(reinterpret_cast<void *>(pointer)),
          custom_bytes(bytes), custom_queue(queue), compiler_stream(stream),
          compiler_device(device) {}
    ~Owner();
    enum class Kind { Native, Custom, Compiler } kind;
    c10::DataPtr native;
    void *custom_pointer = nullptr;
    size_t custom_bytes = 0;
    std::optional<sycl::queue> custom_queue;
    std::mutex consumer_mutex;
    std::vector<sycl::queue> consumers;
    bool accounted = false;
    uint64_t compiler_stream = 0;
    int compiler_device = -1;
};

using CompilerAlloc = bool (*)(uint64_t *, size_t, void *);
using CompilerFree = bool (*)(uint64_t, void *, int *);
using CompilerRogue = bool (*)(uint64_t, int *);
using SetDevice = bool (*)(int);
CompilerAlloc g_compiler_alloc = nullptr;
CompilerFree g_compiler_free = nullptr;
CompilerRogue g_compiler_rogue = nullptr;
SetDevice g_set_device = nullptr;

std::mutex g_owner_mutex;
std::unordered_map<void *, std::shared_ptr<Owner>> g_owners;
std::atomic<uint64_t> g_allocations{0};
std::atomic<uint64_t> g_releases{0};
std::atomic<uint64_t> g_record_stream{0};
std::atomic<uint64_t> g_raw_allocations{0};
std::atomic<uint64_t> g_raw_releases{0};
std::atomic<uint64_t> g_unknown_releases{0};
std::atomic<uint64_t> g_custom_allocations{0};
std::atomic<uint64_t> g_custom_releases{0};
std::atomic<uint64_t> g_custom_record_stream{0};
std::atomic<uint64_t> g_custom_failures{0};
std::atomic<uint64_t> g_custom_live_bytes{0};
std::atomic<uint64_t> g_compiler_allocations{0};
std::atomic<uint64_t> g_compiler_releases{0};
std::atomic<uint64_t> g_compiler_record_stream{0};
std::atomic<uint64_t> g_compiler_failures{0};
std::atomic<uint64_t> g_compiler_live_bytes{0};
thread_local bool g_custom_scope = false;
thread_local size_t g_custom_scope_bytes = 0;
thread_local bool g_compiler_scope = false;
thread_local size_t g_compiler_scope_bytes = 0;
thread_local uint64_t g_compiler_scope_stream = 0;

Owner::~Owner() {
    if (kind == Kind::Native || !custom_pointer || !custom_queue) return;
    try {
        for (auto &queue : consumers) {
            queue.ext_oneapi_submit_barrier().wait_and_throw();
        }
        custom_queue->ext_oneapi_submit_barrier().wait_and_throw();
        if (kind == Kind::Compiler) {
            if (!g_set_device || !g_set_device(compiler_device) ||
                !g_compiler_free || !g_compiler_rogue) {
                g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            int result = -1;
            const uint64_t pointer = reinterpret_cast<uintptr_t>(custom_pointer);
            bool handled = g_compiler_rogue(pointer, &result);
            if (!handled) {
                handled = g_compiler_free(pointer,
                    reinterpret_cast<void *>(compiler_stream), &result);
            }
            if (!handled || result != 0) {
                g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (accounted) {
                g_compiler_live_bytes.fetch_sub(custom_bytes, std::memory_order_relaxed);
                g_compiler_releases.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        sycl::free(custom_pointer, custom_queue->get_context());
        if (accounted) {
            g_custom_live_bytes.fetch_sub(custom_bytes, std::memory_order_relaxed);
            g_custom_releases.fetch_add(1, std::memory_order_relaxed);
        }
    } catch (...) {
        // A failed fence keeps the device pointer unreleased in this process.
        g_custom_failures.fetch_add(1, std::memory_order_relaxed);
    }
}

void proxy_delete(void *pointer) {
    if (!pointer) return;
    std::shared_ptr<Owner> owner;
    {
        std::lock_guard<std::mutex> guard(g_owner_mutex);
        const auto found = g_owners.find(pointer);
        if (found == g_owners.end()) {
            g_unknown_releases.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        owner = std::move(found->second);
        g_owners.erase(found);
    }
    owner.reset(); // Run the original native DataPtr deleter outside our mutex.
    g_releases.fetch_add(1, std::memory_order_relaxed);
}

class ForwardingXpuAllocator final : public c10::xpu::XPUCachingAllocator::XPUAllocator {
public:
    explicit ForwardingXpuAllocator(c10::xpu::XPUCachingAllocator::XPUAllocator *native)
        : native_(native) {}

    void init(c10::DeviceIndex count) override { native_->init(count); }
    bool initialized() override { return native_->initialized(); }
    void emptyCache(c10::MempoolId_t id = {0, 0}) override { native_->emptyCache(id); }
    c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
        return native_->getDeviceStats(device);
    }
    void resetAccumulatedStats(c10::DeviceIndex device) override {
        native_->resetAccumulatedStats(device);
    }
    void resetPeakStats(c10::DeviceIndex device) override {
        native_->resetPeakStats(device);
    }
    std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
        return native_->getMemoryInfo(device);
    }
    void *raw_alloc(size_t bytes) override {
        if (g_custom_scope || g_compiler_scope) {
            throw std::runtime_error("raw allocation is unsupported inside compiler scope");
        }
        g_raw_allocations.fetch_add(1, std::memory_order_relaxed);
        return native_->raw_alloc(bytes);
    }
    void raw_delete(void *pointer) override {
        if (!pointer) return;
        bool tracked;
        {
            std::lock_guard<std::mutex> guard(g_owner_mutex);
            tracked = g_owners.find(pointer) != g_owners.end();
        }
        if (tracked) {
            proxy_delete(pointer);
        } else {
            native_->raw_delete(pointer);
        }
        g_raw_releases.fetch_add(1, std::memory_order_relaxed);
    }

    c10::DataPtr allocate(size_t bytes) override {
        if (g_compiler_scope) {
            if (!bytes || bytes != g_compiler_scope_bytes || !g_compiler_scope_stream ||
                !g_compiler_alloc || !g_set_device) {
                throw std::runtime_error("invalid compiler scope allocation request");
            }
            const auto device = c10::xpu::current_device();
            sycl::queue queue = c10::xpu::getCurrentXPUStream(device).queue();
            auto *supplied = reinterpret_cast<sycl::queue *>(g_compiler_scope_stream);
            if (!supplied || *supplied != queue || !g_set_device(device)) {
                throw std::runtime_error("compiler queue or device differs");
            }
            uint64_t address = 0;
            if (!g_compiler_alloc(&address, bytes,
                                  reinterpret_cast<void *>(g_compiler_scope_stream)) ||
                !address) {
                throw std::runtime_error("AIMDO compiler allocation was not handled");
            }
            std::shared_ptr<Owner> owner;
            try {
                owner = std::make_shared<Owner>(address, bytes, queue,
                                                g_compiler_scope_stream, device);
            } catch (...) {
                int status = -1;
                (void)g_compiler_free(address,
                    reinterpret_cast<void *>(g_compiler_scope_stream), &status);
                throw;
            }
            void *pointer = reinterpret_cast<void *>(address);
            {
                std::lock_guard<std::mutex> guard(g_owner_mutex);
                if (!g_owners.emplace(pointer, owner).second) {
                    throw std::runtime_error("duplicate compiler owner pointer");
                }
            }
            g_compiler_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
            g_compiler_allocations.fetch_add(1, std::memory_order_relaxed);
            owner->accounted = true;
            g_allocations.fetch_add(1, std::memory_order_relaxed);
            return {pointer, pointer, &proxy_delete,
                    c10::Device(c10::DeviceType::XPU, device)};
        }
        if (g_custom_scope) {
            if (!bytes || bytes != g_custom_scope_bytes) {
                throw std::runtime_error("unexpected tensor size inside selected custom scope");
            }
            const auto device = c10::xpu::current_device();
            sycl::queue queue = c10::xpu::getCurrentXPUStream(device).queue();
            void *pointer = sycl::malloc_device(bytes, queue);
            if (!pointer) throw std::bad_alloc();
            std::shared_ptr<Owner> owner;
            try {
                owner = std::make_shared<Owner>(pointer, bytes, queue);
            } catch (...) {
                sycl::free(pointer, queue.get_context());
                throw;
            }
            {
                std::lock_guard<std::mutex> guard(g_owner_mutex);
                if (!g_owners.emplace(pointer, owner).second) {
                    throw std::runtime_error("duplicate custom owner pointer");
                }
            }
            g_custom_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
            g_custom_allocations.fetch_add(1, std::memory_order_relaxed);
            owner->accounted = true;
            g_allocations.fetch_add(1, std::memory_order_relaxed);
            return {pointer, pointer, &proxy_delete,
                    c10::Device(c10::DeviceType::XPU, device)};
        }
        c10::DataPtr native = native_->allocate(bytes);
        void *pointer = native.get();
        if (!pointer) return native;
        const auto device = native.device();
        auto owner = std::make_shared<Owner>(std::move(native));
        {
            std::lock_guard<std::mutex> guard(g_owner_mutex);
            if (!g_owners.emplace(pointer, owner).second) {
                throw std::runtime_error("duplicate native owner pointer");
            }
        }
        g_allocations.fetch_add(1, std::memory_order_relaxed);
        return {pointer, pointer, &proxy_delete, device};
    }
    c10::DeleterFnPtr raw_deleter() const override { return &proxy_delete; }
    void copy_data(void *dest, const void *src, size_t count) const override {
        native_->copy_data(dest, src, count);
    }
    void recordStream(const c10::DataPtr &pointer, c10::Stream stream) override {
        if (!pointer.get()) return;
        if (pointer.get_deleter() != &proxy_delete) {
            native_->recordStream(pointer, stream);
            return;
        }
        std::shared_ptr<Owner> owner;
        {
            std::lock_guard<std::mutex> guard(g_owner_mutex);
            const auto found = g_owners.find(pointer.get());
            if (found == g_owners.end()) {
                throw std::runtime_error("recordStream lost owner");
            }
            owner = found->second;
        }
        if (owner->kind == Owner::Kind::Native) {
            native_->recordStream(owner->native, stream);
        } else {
            c10::xpu::XPUStream selected{stream};
            sycl::queue consumer = selected.queue();
            if (!owner->custom_queue ||
                consumer.get_context() != owner->custom_queue->get_context() ||
                consumer.get_device() != owner->custom_queue->get_device()) {
                throw std::runtime_error("custom owner consumer queue differs");
            }
            std::lock_guard<std::mutex> guard(owner->consumer_mutex);
            bool found = false;
            for (const auto &existing : owner->consumers) {
                if (existing == consumer) { found = true; break; }
            }
            if (!found) owner->consumers.emplace_back(std::move(consumer));
            if (owner->kind == Owner::Kind::Compiler) {
                g_compiler_record_stream.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_custom_record_stream.fetch_add(1, std::memory_order_relaxed);
            }
        }
        g_record_stream.fetch_add(1, std::memory_order_relaxed);
    }
private:
    c10::xpu::XPUCachingAllocator::XPUAllocator *native_;
};

ForwardingXpuAllocator *g_proxy = nullptr;

} // namespace

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_install() {
    if (g_proxy) return false;
    try {
        auto *native = c10::xpu::XPUCachingAllocator::get();
        if (!native || c10::GetAllocator(c10::DeviceType::XPU) != native) return false;
        auto *proxy = new (std::nothrow) ForwardingXpuAllocator(native);
        if (!proxy) return false;
        c10::SetAllocator(c10::DeviceType::XPU, proxy, 255);
        c10::xpu::XPUCachingAllocator::allocator.store(proxy);
        if (c10::GetAllocator(c10::DeviceType::XPU) != proxy ||
            c10::xpu::XPUCachingAllocator::get() != proxy) return false;
        g_proxy = proxy; // Process-lifetime owner for this disposable probe.
        return true;
    } catch (...) {
        return false;
    }
}

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_snapshot(
    uint64_t *values, size_t count) {
    if (!g_proxy || !values || count != 17) return false;
    {
        std::lock_guard<std::mutex> guard(g_owner_mutex);
        values[0] = g_owners.size();
    }
    values[1] = g_allocations.load(std::memory_order_relaxed);
    values[2] = g_releases.load(std::memory_order_relaxed);
    values[3] = g_record_stream.load(std::memory_order_relaxed);
    values[4] = g_raw_allocations.load(std::memory_order_relaxed);
    values[5] = g_raw_releases.load(std::memory_order_relaxed);
    values[6] = g_unknown_releases.load(std::memory_order_relaxed);
    values[7] = g_custom_allocations.load(std::memory_order_relaxed);
    values[8] = g_custom_releases.load(std::memory_order_relaxed);
    values[9] = g_custom_record_stream.load(std::memory_order_relaxed);
    values[10] = g_custom_failures.load(std::memory_order_relaxed);
    values[11] = g_custom_live_bytes.load(std::memory_order_relaxed);
    values[12] = g_compiler_allocations.load(std::memory_order_relaxed);
    values[13] = g_compiler_releases.load(std::memory_order_relaxed);
    values[14] = g_compiler_record_stream.load(std::memory_order_relaxed);
    values[15] = g_compiler_failures.load(std::memory_order_relaxed);
    values[16] = g_compiler_live_bytes.load(std::memory_order_relaxed);
    return true;
}

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_scope_begin(
    size_t bytes) {
    if (!g_proxy || g_custom_scope || !bytes) return false;
    g_custom_scope_bytes = bytes;
    g_custom_scope = true;
    return true;
}

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_scope_end() {
    if (!g_custom_scope) return false;
    g_custom_scope = false;
    g_custom_scope_bytes = 0;
    return true;
}

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_compiler_begin(
    size_t bytes, uint64_t stream, const char *expected_revision) {
    if (!g_proxy || g_custom_scope || g_compiler_scope || !bytes ||
        !stream || !expected_revision) return false;
    using SourceRevision = const char *(*)();
    auto source = reinterpret_cast<SourceRevision>(
        dlsym(RTLD_DEFAULT, "malloc_graph_source_revision"));
    auto alloc = reinterpret_cast<CompilerAlloc>(
        dlsym(RTLD_DEFAULT, "malloc_graph_alloc"));
    auto free = reinterpret_cast<CompilerFree>(
        dlsym(RTLD_DEFAULT, "malloc_graph_free"));
    auto rogue = reinterpret_cast<CompilerRogue>(
        dlsym(RTLD_DEFAULT, "free_rogue"));
    auto set_device = reinterpret_cast<SetDevice>(
        dlsym(RTLD_DEFAULT, "set_devctx_for_device"));
    if (!source || !alloc || !free || !rogue || !set_device ||
        std::strcmp(source(), expected_revision) != 0) return false;
    g_compiler_alloc = alloc;
    g_compiler_free = free;
    g_compiler_rogue = rogue;
    g_set_device = set_device;
    g_compiler_scope_bytes = bytes;
    g_compiler_scope_stream = stream;
    g_compiler_scope = true;
    return true;
}

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_compiler_end() {
    if (!g_compiler_scope) return false;
    g_compiler_scope = false;
    g_compiler_scope_bytes = 0;
    g_compiler_scope_stream = 0;
    return true;
}
