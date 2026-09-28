#include <c10/core/Allocator.h>
#include <c10/xpu/XPUCachingAllocator.h>

// Optional Linux Torch 2.14 diagnostic sidecar. The normal provider never
// installs it. Public XPU record() and compiler capability remain disabled.

#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace {

uint64_t current_thread_token();
struct ThreadState {
    explicit ThreadState(uint64_t value) : token(value) {}
    uint64_t token;
    std::atomic<bool> alive{true};
    std::atomic<uint64_t> pending{0};
};
std::shared_ptr<ThreadState> current_thread_state();

struct Owner {
    explicit Owner(c10::DataPtr &&value) : kind(Kind::Native), native(std::move(value)) {}
    Owner(void *pointer, size_t bytes, const sycl::queue &queue)
        : kind(Kind::Custom), custom_pointer(pointer), custom_bytes(bytes),
          custom_queue(queue) {}
    Owner(uint64_t pointer, size_t bytes, const sycl::queue &queue,
          uint64_t stream, int device)
        : kind(Kind::Compiler), custom_pointer(reinterpret_cast<void *>(pointer)),
          custom_bytes(bytes), custom_queue(queue), compiler_stream(stream),
          compiler_device(device), creator_state(current_thread_state()),
          creator_token(creator_state->token) {}
    ~Owner();
    bool wait_consumers();
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
    std::shared_ptr<ThreadState> creator_state;
    uint64_t creator_token = 0;
    bool fenced = false;
};

struct ScopedRaw {
    size_t bytes;
    bool deleting = false;
};

using CompilerAlloc = bool (*)(uint64_t *, size_t, void *);
using CompilerFree = bool (*)(uint64_t, void *, int *);
using CompilerRogue = bool (*)(uint64_t, int *);
using SetDevice = bool (*)(int);
CompilerAlloc g_compiler_alloc = nullptr;
CompilerFree g_compiler_free = nullptr;
CompilerFree g_compiler_free_owned = nullptr;
CompilerRogue g_compiler_rogue = nullptr;
SetDevice g_set_device = nullptr;

std::mutex g_owner_mutex;
std::unordered_map<void *, std::shared_ptr<Owner>> g_owners;
std::atomic<uint64_t> g_next_thread_token{1};
thread_local uint64_t g_thread_token =
    g_next_thread_token.fetch_add(1, std::memory_order_relaxed);
uint64_t current_thread_token() { return g_thread_token; }
std::mutex g_deferred_mutex;
using DeferredFrees =
    std::unordered_map<uint64_t, std::deque<std::shared_ptr<Owner>>>;
DeferredFrees *g_deferred_frees = new DeferredFrees(); // Never drain on process teardown.
std::atomic<uint64_t> g_deferred_free_count{0};
std::atomic<bool> g_compiler_terminal{false};
struct ThreadExitGuard {
    std::shared_ptr<ThreadState> state;
    ~ThreadExitGuard() {
        if (!state) return;
        state->alive.store(false);
        if (state->pending.load()) {
            g_compiler_terminal.store(true, std::memory_order_release);
        }
    }
};
thread_local ThreadExitGuard g_thread_exit_guard;
std::shared_ptr<ThreadState> current_thread_state() {
    if (!g_thread_exit_guard.state) {
        g_thread_exit_guard.state =
            std::make_shared<ThreadState>(current_thread_token());
    }
    return g_thread_exit_guard.state;
}
std::mutex g_scoped_raw_mutex;
std::unordered_map<void *, ScopedRaw> g_scoped_raw;
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
std::atomic<uint64_t> g_scoped_raw_allocations{0};
std::atomic<uint64_t> g_scoped_raw_releases{0};
std::atomic<uint64_t> g_scoped_raw_failures{0};
std::atomic<uint64_t> g_scoped_raw_live_bytes{0};
thread_local bool g_custom_scope = false;
thread_local size_t g_custom_scope_bytes = 0;
thread_local bool g_compiler_scope = false;
thread_local size_t g_compiler_scope_bytes = 0;
thread_local uint64_t g_compiler_scope_stream = 0;
thread_local bool g_fail_next_compiler_owner_insert = false;
thread_local void *g_duplicate_next_compiler_pointer = nullptr;

bool Owner::wait_consumers() {
    if (fenced) return true;
    if (!custom_queue) return false;
    try {
        for (auto &queue : consumers) {
            queue.ext_oneapi_submit_barrier().wait_and_throw();
        }
        custom_queue->ext_oneapi_submit_barrier().wait_and_throw();
        fenced = true;
        return true;
    } catch (...) {
        return false;
    }
}

Owner::~Owner() {
    if (kind == Kind::Native || !custom_pointer || !custom_queue) return;
    if (!wait_consumers()) {
        if (kind == Kind::Compiler) {
            g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
            g_compiler_terminal.store(true, std::memory_order_release);
        } else {
            g_custom_failures.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    try {
        if (kind == Kind::Compiler) {
            if (!g_set_device || !g_set_device(compiler_device) ||
                !g_compiler_free_owned || !g_compiler_rogue) {
                g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
                g_compiler_terminal.store(true, std::memory_order_release);
                return;
            }
            int result = -1;
            const uint64_t pointer = reinterpret_cast<uintptr_t>(custom_pointer);
            bool handled = g_compiler_rogue(pointer, &result);
            if (!handled) {
                handled = g_compiler_free_owned(pointer,
                    reinterpret_cast<void *>(compiler_stream), &result);
            }
            if (!handled || result != 0) {
                g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
                g_compiler_terminal.store(true, std::memory_order_release);
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
        if (kind == Kind::Compiler) {
            g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
            g_compiler_terminal.store(true, std::memory_order_release);
        } else {
            g_custom_failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void terminalize_compiler_owner(std::shared_ptr<Owner> &owner) {
    // The owner cannot safely be freed from this thread after a failed fence,
    // graph release or deferred-queue insertion. Keep its backing until exit.
    owner->custom_pointer = nullptr;
    g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
    g_compiler_terminal.store(true, std::memory_order_release);
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
    if (owner->kind == Owner::Kind::Compiler &&
        owner->creator_token != current_thread_token()) {
        if (g_compiler_terminal.load(std::memory_order_acquire) ||
            !owner->wait_consumers() || !g_set_device ||
            !g_set_device(owner->compiler_device) || !g_compiler_rogue) {
            terminalize_compiler_owner(owner);
        } else {
            int result = -1;
            bool handled = g_compiler_rogue(
                reinterpret_cast<uintptr_t>(pointer), &result);
            if (handled) {
                if (result == 0) {
                    owner->custom_pointer = nullptr;
                    if (owner->accounted) {
                        g_compiler_live_bytes.fetch_sub(owner->custom_bytes,
                            std::memory_order_relaxed);
                        g_compiler_releases.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    terminalize_compiler_owner(owner);
                }
            } else {
                try {
                    std::lock_guard<std::mutex> guard(g_deferred_mutex);
                    (*g_deferred_frees)[owner->creator_token].push_back(owner);
                    g_deferred_free_count.fetch_add(1, std::memory_order_relaxed);
                    owner->creator_state->pending.fetch_add(1);
                    if (!owner->creator_state->alive.load()) {
                        g_compiler_terminal.store(true, std::memory_order_release);
                    }
                } catch (...) {
                    terminalize_compiler_owner(owner);
                }
            }
        }
        owner.reset();
        g_releases.fetch_add(1, std::memory_order_relaxed);
        return;
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
        if (g_custom_scope) {
            throw std::runtime_error("raw allocation is unsupported inside custom scope");
        }
        void *pointer = nullptr;
        try {
            pointer = native_->raw_alloc(bytes);
        } catch (...) {
            if (g_compiler_scope) {
                g_scoped_raw_failures.fetch_add(1, std::memory_order_relaxed);
            }
            throw;
        }
        if (!pointer) return pointer;
        g_raw_allocations.fetch_add(1, std::memory_order_relaxed);
        if (g_compiler_scope) {
            bool inserted = false;
            try {
                std::lock_guard<std::mutex> guard(g_scoped_raw_mutex);
                inserted = g_scoped_raw.emplace(pointer, ScopedRaw{bytes}).second;
            } catch (...) {
                g_scoped_raw_failures.fetch_add(1, std::memory_order_relaxed);
                native_->raw_delete(pointer);
                g_raw_releases.fetch_add(1, std::memory_order_relaxed);
                throw;
            }
            if (!inserted) {
                g_scoped_raw_failures.fetch_add(1, std::memory_order_relaxed);
                throw std::runtime_error(
                    "duplicate scoped raw pointer; process must exit");
            }
            g_scoped_raw_allocations.fetch_add(1, std::memory_order_relaxed);
            g_scoped_raw_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
        return pointer;
    }
    void raw_delete(void *pointer) override {
        if (!pointer) return;
        bool tracked;
        {
            std::lock_guard<std::mutex> guard(g_owner_mutex);
            tracked = g_owners.find(pointer) != g_owners.end();
        }
        size_t scoped_bytes = 0;
        bool scoped = false;
        {
            std::lock_guard<std::mutex> guard(g_scoped_raw_mutex);
            const auto found = g_scoped_raw.find(pointer);
            if (found != g_scoped_raw.end()) {
                if (found->second.deleting || tracked) {
                    g_scoped_raw_failures.fetch_add(1, std::memory_order_relaxed);
                    throw std::runtime_error(
                        "ambiguous scoped raw release; process must exit");
                }
                found->second.deleting = true;
                scoped_bytes = found->second.bytes;
                scoped = true;
            }
        }
        if (tracked) {
            proxy_delete(pointer);
        } else {
            try {
                native_->raw_delete(pointer);
            } catch (...) {
                if (scoped) {
                    g_scoped_raw_failures.fetch_add(1, std::memory_order_relaxed);
                }
                throw;
            }
            if (scoped) {
                std::lock_guard<std::mutex> guard(g_scoped_raw_mutex);
                g_scoped_raw.erase(pointer);
                g_scoped_raw_releases.fetch_add(1, std::memory_order_relaxed);
                g_scoped_raw_live_bytes.fetch_sub(scoped_bytes, std::memory_order_relaxed);
            }
        }
        g_raw_releases.fetch_add(1, std::memory_order_relaxed);
    }

    c10::DataPtr allocate(size_t bytes) override {
        if (g_compiler_scope) {
            if (g_compiler_terminal.load(std::memory_order_acquire)) {
                throw std::runtime_error(
                    "terminal compiler owner error; process must exit");
            }
            if (!bytes) return native_->allocate(0);
            if ((g_compiler_scope_bytes && bytes != g_compiler_scope_bytes) ||
                !g_compiler_scope_stream || !g_compiler_alloc || !g_set_device) {
                throw std::runtime_error("invalid compiler scope allocation request");
            }
            const auto device = c10::xpu::current_device();
            sycl::queue queue = c10::xpu::getCurrentXPUStream(device).queue();
            auto *supplied = reinterpret_cast<sycl::queue *>(g_compiler_scope_stream);
            if (!supplied || *supplied != queue || !g_set_device(device)) {
                throw std::runtime_error("compiler queue or device differs");
            }
            uint64_t address = 0;
            void *synthetic_duplicate = std::exchange(
                g_duplicate_next_compiler_pointer, nullptr);
            if (synthetic_duplicate) {
                address = reinterpret_cast<uintptr_t>(synthetic_duplicate);
            } else {
                if (!g_compiler_alloc(&address, bytes,
                                      reinterpret_cast<void *>(g_compiler_scope_stream)) ||
                    !address) {
                    throw std::runtime_error("AIMDO compiler allocation was not handled");
                }
            }
            std::shared_ptr<Owner> owner;
            try {
                owner = std::make_shared<Owner>(address, bytes, queue,
                                                g_compiler_scope_stream, device);
            } catch (...) {
                if (!synthetic_duplicate) {
                    int status = -1;
                    (void)g_compiler_free(address,
                        reinterpret_cast<void *>(g_compiler_scope_stream), &status);
                }
                throw;
            }
            void *pointer = reinterpret_cast<void *>(address);
            // Exercise the same unwind/owner rollback as an emplace failure.
            if (std::exchange(g_fail_next_compiler_owner_insert, false)) {
                throw std::bad_alloc();
            }
            {
                std::lock_guard<std::mutex> guard(g_owner_mutex);
                if (!g_owners.emplace(pointer, owner).second) {
                    // Another live owner has this VA. Releasing the new owner
                    // could free the old tensor, so leak the ambiguous claim
                    // and require the process to exit.
                    owner->custom_pointer = nullptr;
                    g_compiler_failures.fetch_add(1, std::memory_order_relaxed);
                    g_compiler_terminal.store(true, std::memory_order_release);
                    throw std::runtime_error(
                        "duplicate compiler owner pointer; process must exit");
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
                    owner->custom_pointer = nullptr;
                    g_custom_failures.fetch_add(1, std::memory_order_relaxed);
                    throw std::runtime_error(
                        "duplicate custom owner pointer; process must exit");
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
                (void)owner->native.release_context();
                throw std::runtime_error(
                    "duplicate native owner pointer; process must exit");
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

extern "C" __attribute__((visibility("default"))) const char *
aimdo_full_proxy_torch_version() {
    return "2.14.0+xpu";
}

extern "C" __attribute__((visibility("default"))) bool
aimdo_full_proxy_is_installed() {
    return g_proxy != nullptr;
}

extern "C" __attribute__((visibility("default"))) bool
aimdo_full_proxy_drain_deferred_frees() {
    if (!g_proxy || g_compiler_terminal.load(std::memory_order_acquire)) return false;
    const uint64_t token = current_thread_token();
    try {
        for (;;) {
            std::shared_ptr<Owner> owner;
            {
                std::lock_guard<std::mutex> guard(g_deferred_mutex);
                auto found = g_deferred_frees->find(token);
                if (found == g_deferred_frees->end() || found->second.empty()) break;
                owner = std::move(found->second.front());
                found->second.pop_front();
                if (found->second.empty()) g_deferred_frees->erase(found);
                g_deferred_free_count.fetch_sub(1, std::memory_order_relaxed);
                owner->creator_state->pending.fetch_sub(1);
            }
            const uint64_t failures =
                g_compiler_failures.load(std::memory_order_acquire);
            owner.reset(); // Native graph free runs on the original owner thread.
            if (g_compiler_terminal.load(std::memory_order_acquire) ||
                g_compiler_failures.load(std::memory_order_acquire) != failures) {
                return false;
            }
        }
    } catch (...) {
        g_compiler_terminal.store(true, std::memory_order_release);
        return false;
    }
    return !g_compiler_terminal.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t
aimdo_full_proxy_deferred_free_count() {
    return g_deferred_free_count.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t
aimdo_full_proxy_dead_deferred_free_count() {
    uint64_t count = 0;
    std::lock_guard<std::mutex> guard(g_deferred_mutex);
    for (const auto &[token, pending] : *g_deferred_frees) {
        (void)token;
        for (const auto &owner : pending) {
            count += !owner->creator_state->alive.load();
        }
    }
    return count;
}

extern "C" __attribute__((visibility("default"))) bool
aimdo_full_proxy_test_fail_next_compiler_owner_insert() {
    if (!g_proxy || !g_compiler_scope || g_fail_next_compiler_owner_insert) {
        return false;
    }
    g_fail_next_compiler_owner_insert = true;
    return true;
}

extern "C" __attribute__((visibility("default"))) bool
aimdo_full_proxy_test_duplicate_next_compiler_pointer(void *pointer) {
    if (!g_proxy || !g_compiler_scope || !pointer ||
        g_duplicate_next_compiler_pointer || g_fail_next_compiler_owner_insert) {
        return false;
    }
    std::lock_guard<std::mutex> guard(g_owner_mutex);
    const auto found = g_owners.find(pointer);
    if (found == g_owners.end() || found->second->kind != Owner::Kind::Compiler) {
        return false;
    }
    g_duplicate_next_compiler_pointer = pointer;
    return true;
}

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
        g_proxy = proxy; // Process-lifetime owner; never unload this DSO.
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
        values[0] = g_owners.size() +
            g_deferred_free_count.load(std::memory_order_acquire);
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

extern "C" __attribute__((visibility("default"))) bool
aimdo_full_proxy_scoped_raw_snapshot(uint64_t *values, size_t count) {
    if (!g_proxy || !values || count != 5) return false;
    {
        std::lock_guard<std::mutex> guard(g_scoped_raw_mutex);
        values[0] = g_scoped_raw.size();
    }
    values[1] = g_scoped_raw_allocations.load(std::memory_order_relaxed);
    values[2] = g_scoped_raw_releases.load(std::memory_order_relaxed);
    values[3] = g_scoped_raw_failures.load(std::memory_order_relaxed);
    values[4] = g_scoped_raw_live_bytes.load(std::memory_order_relaxed);
    return true;
}

extern "C" __attribute__((visibility("default"))) bool
aimdo_full_proxy_is_compiler_owner(void *pointer) {
    if (!g_proxy || !pointer) return false;
    std::lock_guard<std::mutex> guard(g_owner_mutex);
    const auto found = g_owners.find(pointer);
    return found != g_owners.end() && found->second->kind == Owner::Kind::Compiler;
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
    // Zero selects every positive-size Torch tensor request in this scope.
    if (!g_proxy || g_custom_scope || g_compiler_scope ||
        g_compiler_terminal.load(std::memory_order_acquire) ||
        !stream || !expected_revision) return false;
    using SourceRevision = const char *(*)();
    auto source = reinterpret_cast<SourceRevision>(
        dlsym(RTLD_DEFAULT, "malloc_graph_source_revision"));
    auto alloc = reinterpret_cast<CompilerAlloc>(
        dlsym(RTLD_DEFAULT, "malloc_graph_alloc"));
    auto free = reinterpret_cast<CompilerFree>(
        dlsym(RTLD_DEFAULT, "malloc_graph_free"));
    auto free_owned = reinterpret_cast<CompilerFree>(
        dlsym(RTLD_DEFAULT, "malloc_graph_free_owned"));
    auto rogue = reinterpret_cast<CompilerRogue>(
        dlsym(RTLD_DEFAULT, "free_rogue"));
    auto set_device = reinterpret_cast<SetDevice>(
        dlsym(RTLD_DEFAULT, "set_devctx_for_device"));
    if (!source || !alloc || !free || !free_owned || !rogue || !set_device ||
        std::strcmp(source(), expected_revision) != 0) return false;
    g_compiler_alloc = alloc;
    g_compiler_free = free;
    g_compiler_free_owned = free_owned;
    g_compiler_rogue = rogue;
    g_set_device = set_device;
    g_compiler_scope_bytes = bytes;
    g_compiler_scope_stream = stream;
    g_compiler_scope = true;
    return true;
}

extern "C" __attribute__((visibility("default"))) bool aimdo_full_proxy_compiler_end() {
    if (!g_compiler_scope) return false;
    g_fail_next_compiler_owner_insert = false;
    g_duplicate_next_compiler_pointer = nullptr;
    g_compiler_scope = false;
    g_compiler_scope_bytes = 0;
    g_compiler_scope_stream = 0;
    return true;
}
