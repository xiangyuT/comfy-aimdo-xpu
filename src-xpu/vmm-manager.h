#pragma once

// Device VMM ownership, independent of SYCL and of the tensor allocator.
// Metadata is allocated before a driver mutation. Failed releases retain the
// owner and accounting; callers may retry cleanup after resolving the error.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <new>

namespace aimdo_xpu {

struct VmmOwner {
    int id;
    ze_context_handle_t context;
    ze_device_handle_t device;
    bool operator==(const VmmOwner &other) const {
        return id == other.id && context == other.context && device == other.device;
    }
};

struct VmmApi {
    decltype(&zeVirtualMemQueryPageSize) page_size = zeVirtualMemQueryPageSize;
    decltype(&zeVirtualMemReserve) reserve = zeVirtualMemReserve;
    decltype(&zeVirtualMemFree) free = zeVirtualMemFree;
    decltype(&zePhysicalMemCreate) create = zePhysicalMemCreate;
    decltype(&zePhysicalMemDestroy) destroy = zePhysicalMemDestroy;
    decltype(&zeVirtualMemMap) map = zeVirtualMemMap;
    decltype(&zeVirtualMemUnmap) unmap = zeVirtualMemUnmap;
    decltype(&zeVirtualMemSetAccessAttribute) access = zeVirtualMemSetAccessAttribute;
};

class VmmManager {
    struct Reservation {
        VmmOwner owner;
        uintptr_t address = 0;
        size_t bytes = 0;
        void *outer = nullptr;
        size_t outer_bytes = 0;
        size_t page = 0;
    };
    struct Physical {
        VmmOwner owner;
        ze_physical_mem_handle_t handle = nullptr;
        size_t bytes = 0;
        size_t page = 0;
    };
    struct Mapping {
        VmmOwner owner;
        uintptr_t address;
        size_t bytes;
        ze_physical_mem_handle_t handle;
        size_t offset;
        uintptr_t cleanup_address = 0;
        size_t cleanup_bytes = 0;
    };
    VmmApi api;
    std::mutex mutex;
    std::list<Reservation> reservations;
    std::list<Physical> physical;
    std::list<Mapping> mappings;

    static bool power_of_two(size_t n) { return n && !(n & (n - 1)); }
    static bool valid_range(uintptr_t p, size_t n) {
        return n && n <= std::numeric_limits<uintptr_t>::max() - p;
    }
    static bool contains(uintptr_t p, size_t n, uintptr_t q, size_t m) {
        return valid_range(p, n) && valid_range(q, m) && q >= p && m <= n && q - p <= n - m;
    }
    static bool overlaps(uintptr_t p, size_t n, uintptr_t q, size_t m) {
        return p < q + m && q < p + n;
    }
    auto reservation(const VmmOwner &owner, uintptr_t address, size_t bytes) {
        return std::find_if(reservations.begin(), reservations.end(), [&](const auto &r) {
            return r.owner == owner && contains(r.address, r.bytes, address, bytes);
        });
    }
    bool covered(const VmmOwner &owner, uintptr_t address, size_t bytes) {
        size_t total = 0;
        for (const auto &m : mappings) {
            if (!(m.owner == owner) || !overlaps(address, bytes, m.address, m.bytes)) continue;
            if (!contains(address, bytes, m.address, m.bytes) || m.cleanup_bytes) return false;
            total += m.bytes;
        }
        return total == bytes;
    }

public:
    explicit VmmManager(VmmApi calls = {}) : api(calls) {}

    ze_result_t reserve(const VmmOwner &owner, uintptr_t *out, size_t bytes,
                        size_t alignment, uintptr_t requested, uint64_t flags) {
        if (!out || !bytes || requested || flags || (alignment && !power_of_two(alignment)))
            return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(mutex);
        size_t page = 0;
        auto result = api.page_size(owner.context, owner.device, bytes, &page);
        if (result != ZE_RESULT_SUCCESS) return result;
        if (!power_of_two(page) || bytes % page) return ZE_RESULT_ERROR_INVALID_SIZE;
        alignment = std::max(alignment, page);
        const size_t padding = alignment - page;
        if (bytes > std::numeric_limits<size_t>::max() - padding)
            return ZE_RESULT_ERROR_INVALID_SIZE;
        try { reservations.push_back({owner, 0, bytes, nullptr, bytes + padding, page}); }
        catch (const std::bad_alloc &) { return ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY; }
        auto &r = reservations.back();
        result = api.reserve(owner.context, nullptr, r.outer_bytes, &r.outer);
        if (result != ZE_RESULT_SUCCESS) { reservations.pop_back(); return result; }
        // The native reservation remains whole: partial zeVirtualMemFree is
        // unsupported. Its aligned interior is the only exposed range.
        r.address = (reinterpret_cast<uintptr_t>(r.outer) + alignment - 1) & ~(alignment - 1);
        *out = r.address;
        return ZE_RESULT_SUCCESS;
    }

    ze_result_t free(const VmmOwner &owner, uintptr_t address, size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex);
        auto r = reservation(owner, address, bytes);
        if (r == reservations.end() || r->address != address || r->bytes != bytes)
            return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        for (const auto &m : mappings)
            if (overlaps(address, bytes, m.address, m.bytes)) return ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE;
        auto result = api.free(owner.context, r->outer, r->outer_bytes);
        if (result == ZE_RESULT_SUCCESS) reservations.erase(r);
        return result;
    }

    ze_result_t create(const VmmOwner &owner, ze_physical_mem_handle_t *out, size_t bytes) {
        if (!out || !bytes) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(mutex);
        size_t page = 0;
        auto result = api.page_size(owner.context, owner.device, bytes, &page);
        if (result != ZE_RESULT_SUCCESS) return result;
        if (!power_of_two(page) || bytes % page) return ZE_RESULT_ERROR_INVALID_SIZE;
        try { physical.push_back({owner, nullptr, bytes, page}); }
        catch (const std::bad_alloc &) { return ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY; }
        ze_physical_mem_desc_t desc{ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr,
                                  ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE, bytes};
        result = api.create(owner.context, owner.device, &desc, &physical.back().handle);
        if (result != ZE_RESULT_SUCCESS) { physical.pop_back(); return result; }
        *out = physical.back().handle;
        return ZE_RESULT_SUCCESS;
    }

    ze_result_t map(const VmmOwner &owner, uintptr_t address, size_t bytes,
                    ze_physical_mem_handle_t handle, size_t offset, uint64_t flags) {
        if (flags || !valid_range(address, bytes)) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(mutex);
        auto r = reservation(owner, address, bytes);
        auto p = std::find_if(physical.begin(), physical.end(), [&](const auto &p) {
            return p.handle == handle && p.owner == owner;
        });
        if (r == reservations.end() || p == physical.end() || bytes > p->bytes ||
            offset > p->bytes - bytes || address % p->page || bytes % p->page || offset % p->page)
            return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        for (const auto &m : mappings) {
            if (overlaps(address, bytes, m.address, m.bytes) ||
                (m.cleanup_bytes && overlaps(address, bytes, m.cleanup_address, m.cleanup_bytes)))
                return ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE;
        }
        try { mappings.push_back({owner, address, bytes, handle, offset}); }
        catch (const std::bad_alloc &) { return ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY; }
        auto result = api.map(owner.context, reinterpret_cast<void *>(address), bytes,
                              handle, offset, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
        if (result != ZE_RESULT_SUCCESS) mappings.pop_back();
        return result;
    }

    ze_result_t access(const VmmOwner &owner, uintptr_t address, size_t bytes) {
        if (!valid_range(address, bytes)) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(mutex);
        if (!covered(owner, address, bytes)) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        // Only READWRITE is admitted by dispatch. Do not claim protection or
        // peer access that this adapter has not implemented.
        return api.access(owner.context, reinterpret_cast<void *>(address), bytes,
                          ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    }

    ze_result_t unmap(const VmmOwner &owner, uintptr_t address, size_t bytes) {
        if (!valid_range(address, bytes)) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(mutex);
        if (reservation(owner, address, bytes) == reservations.end()) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        bool retry = false;
        for (const auto &m : mappings) {
            if (m.owner == owner && m.cleanup_address == address && m.cleanup_bytes == bytes) retry = true;
            if (overlaps(address, bytes, m.address, m.bytes) &&
                (!(m.owner == owner) || !contains(address, bytes, m.address, m.bytes) ||
                 (m.cleanup_bytes && (m.cleanup_address != address || m.cleanup_bytes != bytes))))
                return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (!retry && !covered(owner, address, bytes)) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        for (auto &m : mappings) {
            if (m.owner == owner && contains(address, bytes, m.address, m.bytes)) {
                m.cleanup_address = address;
                m.cleanup_bytes = bytes;
            }
        }
        for (auto m = mappings.begin(); m != mappings.end();) {
            if (!(m->owner == owner) || m->cleanup_address != address || m->cleanup_bytes != bytes) { ++m; continue; }
            auto result = api.unmap(owner.context, reinterpret_cast<void *>(m->address), m->bytes);
            if (result != ZE_RESULT_SUCCESS) return result;
            m = mappings.erase(m);
        }
        return ZE_RESULT_SUCCESS;
    }

    ze_result_t destroy(const VmmOwner &owner, ze_physical_mem_handle_t handle) {
        std::lock_guard<std::mutex> guard(mutex);
        auto p = std::find_if(physical.begin(), physical.end(), [&](const auto &p) {
            return p.handle == handle && p.owner == owner;
        });
        if (p == physical.end()) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        for (const auto &m : mappings)
            if (m.handle == handle) return ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE;
        auto result = api.destroy(owner.context, handle);
        if (result == ZE_RESULT_SUCCESS) physical.erase(p);
        return result;
    }

    bool snapshot(uint64_t *out, size_t count) {
        if (!out || count != 7) return false;
        std::lock_guard<std::mutex> guard(mutex);
        std::fill(out, out + count, 0);
        out[0] = reservations.size(); out[1] = physical.size(); out[2] = mappings.size();
        for (const auto &r : reservations) { out[3] += r.bytes; out[4] += r.outer_bytes; }
        for (const auto &p : physical) out[5] += p.bytes;
        for (const auto &m : mappings) out[6] += m.bytes;
        return true;
    }
};

} // namespace aimdo_xpu
