#include <level_zero/ze_api.h>
#include "../src-xpu/vmm-manager.h"
#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <thread>

using aimdo_xpu::VmmManager;
using aimdo_xpu::VmmOwner;
constexpr size_t page = 2ULL << 20;
constexpr size_t block = 8ULL << 20;
const VmmOwner owner{0, reinterpret_cast<ze_context_handle_t>(1), reinterpret_cast<ze_device_handle_t>(2)};
const VmmOwner other_context{0, reinterpret_cast<ze_context_handle_t>(3), owner.device};
const VmmOwner other_device{1, owner.context, reinterpret_cast<ze_device_handle_t>(4)};

struct Driver {
    uintptr_t next_va = 0x20200000;
    uintptr_t next_handle = 32;
    std::map<uintptr_t, size_t> reservations;
    std::map<ze_physical_mem_handle_t, size_t> backing;
    std::map<uintptr_t, std::pair<size_t, ze_physical_mem_handle_t>> mappings;
    std::string fail;
    int fail_after = 0;
    ze_result_t error = ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    size_t calls = 0;
    ze_result_t call(const char *name) {
        ++calls;
        if (fail == name && fail_after-- == 0) { fail.clear(); return error; }
        return ZE_RESULT_SUCCESS;
    }
} driver;

ze_result_t ZE_APICALL query(ze_context_handle_t c, ze_device_handle_t d, size_t, size_t *out) {
    assert(c == owner.context && d == owner.device);
    auto r = driver.call("page"); if (r) return r;
    *out = page; return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL reserve(ze_context_handle_t c, const void *start, size_t bytes, void **out) {
    assert(c == owner.context && !start && bytes % page == 0);
    auto r = driver.call("reserve"); if (r) return r;
    *out = reinterpret_cast<void *>(driver.next_va);
    driver.reservations[driver.next_va] = bytes;
    driver.next_va += bytes + page;
    return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL release(ze_context_handle_t c, const void *address, size_t bytes) {
    assert(c == owner.context);
    const auto key = reinterpret_cast<uintptr_t>(address);
    assert(driver.reservations.count(key) && driver.reservations.at(key) == bytes);
    for (auto [p, m] : driver.mappings) assert(p < key || p >= key + bytes);
    auto r = driver.call("free"); if (r) return r;
    driver.reservations.erase(key); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL create(ze_context_handle_t c, ze_device_handle_t d,
                              ze_physical_mem_desc_t *desc, ze_physical_mem_handle_t *out) {
    assert(c == owner.context && d == owner.device && desc->size % page == 0);
    auto r = driver.call("create"); if (r) return r;
    *out = reinterpret_cast<ze_physical_mem_handle_t>(driver.next_handle++);
    driver.backing[*out] = desc->size; return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL destroy(ze_context_handle_t c, ze_physical_mem_handle_t h) {
    assert(c == owner.context && driver.backing.count(h));
    for (auto [p, m] : driver.mappings) assert(m.second != h);
    auto r = driver.call("destroy"); if (r) return r;
    driver.backing.erase(h); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL map(ze_context_handle_t c, const void *p, size_t n,
                           ze_physical_mem_handle_t h, size_t offset, ze_memory_access_attribute_t access) {
    assert(c == owner.context && driver.backing.count(h) && n + offset <= driver.backing.at(h));
    assert(access == ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    auto r = driver.call("map"); if (r) return r;
    assert(driver.mappings.emplace(reinterpret_cast<uintptr_t>(p), std::make_pair(n, h)).second);
    return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL unmap(ze_context_handle_t c, const void *p, size_t n) {
    auto key = reinterpret_cast<uintptr_t>(p);
    assert(c == owner.context && driver.mappings.at(key).first == n);
    auto r = driver.call("unmap"); if (r) return r;
    driver.mappings.erase(key); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL access(ze_context_handle_t c, const void *, size_t,
                              ze_memory_access_attribute_t flags) {
    assert(c == owner.context && flags == ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    return driver.call("access");
}
aimdo_xpu::VmmApi api{query, reserve, release, create, destroy, map, unmap, access};

void check(VmmManager &manager, size_t reservations, size_t physical, size_t mappings) {
    uint64_t values[7]{};
    assert(manager.snapshot(values, 7));
    assert(values[0] == reservations && values[1] == physical && values[2] == mappings);
    assert(driver.reservations.size() == reservations && driver.backing.size() == physical && driver.mappings.size() == mappings);
    size_t outer = 0, backing = 0, mapped = 0;
    for (auto [p, n] : driver.reservations) outer += n;
    for (auto [h, n] : driver.backing) backing += n;
    for (auto [p, m] : driver.mappings) mapped += m.first;
    assert(values[4] == outer && values[5] == backing && values[6] == mapped);
}

void failure_case(const std::string &operation, ze_result_t error) {
    driver = {};
    VmmManager manager(api);
    uintptr_t a = 0, b = 0;
    ze_physical_mem_handle_t h = nullptr;
    auto fail = [&] { driver.fail = operation; driver.error = error; };
    if (operation == "page" || operation == "reserve") {
        fail();
        assert(manager.reserve(owner, &a, block, block, 0, 0) == error);
        assert(a == 0); check(manager, 0, 0, 0);
    }
    assert(manager.reserve(owner, &a, block, block, 0, 0) == ZE_RESULT_SUCCESS);
    assert(a % block == 0);
    if (operation == "create") {
        fail(); assert(manager.create(owner, &h, block) == error);
        assert(h == nullptr); check(manager, 1, 0, 0);
    }
    assert(manager.create(owner, &h, block) == ZE_RESULT_SUCCESS);
    if (operation == "map") {
        fail(); assert(manager.map(owner, a, block, h, 0, 0) == error);
        check(manager, 1, 1, 0);
    }
    assert(manager.map(owner, a, block, h, 0, 0) == ZE_RESULT_SUCCESS);
    if (operation == "access") {
        fail(); assert(manager.access(owner, a, block) == error);
        check(manager, 1, 1, 1);
    }
    assert(manager.access(owner, a, block) == ZE_RESULT_SUCCESS);
    assert(manager.reserve(owner, &b, block, block, 0, 0) == ZE_RESULT_SUCCESS);
    assert(manager.map(owner, b, block, h, 0, 0) == ZE_RESULT_SUCCESS);
    // Live aliases own the backing independently; premature release must not
    // reach the driver, even after a failed cleanup request.
    auto calls = driver.calls;
    assert(manager.destroy(owner, h) == ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
    assert(manager.free(owner, a, block) == ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
    assert(driver.calls == calls);
    if (operation == "unmap") {
        fail(); assert(manager.unmap(owner, a, block) == error);
        check(manager, 2, 1, 2);
    }
    assert(manager.unmap(owner, a, block) == ZE_RESULT_SUCCESS);
    assert(manager.destroy(owner, h) == ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
    assert(manager.access(owner, b, block) == ZE_RESULT_SUCCESS);
    assert(manager.unmap(owner, b, block) == ZE_RESULT_SUCCESS);
    if (operation == "destroy") {
        fail(); assert(manager.destroy(owner, h) == error);
        check(manager, 2, 1, 0);
    }
    assert(manager.destroy(owner, h) == ZE_RESULT_SUCCESS);
    if (operation == "free") {
        fail(); assert(manager.free(owner, a, block) == error);
        check(manager, 2, 0, 0);
    }
    assert(manager.free(owner, a, block) == ZE_RESULT_SUCCESS);
    assert(manager.free(owner, b, block) == ZE_RESULT_SUCCESS);
    check(manager, 0, 0, 0);
    std::cout << "PASS failure_" << operation << "_" << static_cast<uint32_t>(error) << '\n';
}

void bounds_and_identity() {
    driver = {}; VmmManager manager(api);
    uintptr_t a = 0; ze_physical_mem_handle_t h = nullptr;
    assert(manager.reserve(owner, &a, 2 * block, block, 0, 0) == ZE_RESULT_SUCCESS);
    assert(manager.create(owner, &h, 2 * block) == ZE_RESULT_SUCCESS);
    const auto calls = driver.calls;
    assert(manager.map(other_context, a, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(other_device, a, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.destroy(other_context, h) != ZE_RESULT_SUCCESS);
    assert(manager.free(other_context, a, 2 * block) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, a + 4096, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, a, block + 1, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, a, block, h, 4096, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, a, block, h, 2 * block, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, a + 2 * block, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, UINTPTR_MAX - block + 1, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.map(owner, a, block, h, 0, 1) != ZE_RESULT_SUCCESS);
    assert(manager.access(owner, a, block) != ZE_RESULT_SUCCESS);
    assert(driver.calls == calls);
    assert(manager.map(owner, a, block, h, block, 0) == ZE_RESULT_SUCCESS);
    assert(manager.unmap(owner, a, page) != ZE_RESULT_SUCCESS); // partial mapping is unsupported
    assert(manager.map(owner, a, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.access(other_context, a, block) != ZE_RESULT_SUCCESS);
    assert(manager.unmap(other_context, a, block) != ZE_RESULT_SUCCESS);
    assert(manager.unmap(owner, a, block) == ZE_RESULT_SUCCESS);
    assert(manager.destroy(owner, h) == ZE_RESULT_SUCCESS);
    assert(manager.free(owner, a, 2 * block) == ZE_RESULT_SUCCESS);
    assert(manager.reserve(owner, &a, block, 123, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.reserve(owner, &a, block, block, 1, 0) != ZE_RESULT_SUCCESS);
    assert(manager.reserve(owner, &a, block, block, 0, 1) != ZE_RESULT_SUCCESS);
    assert(manager.reserve(owner, &a, 0, block, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.reserve(owner, &a, SIZE_MAX - page + 1, block, 0, 0) != ZE_RESULT_SUCCESS);
    check(manager, 0, 0, 0);
    std::cout << "PASS bounds_context_device_access_offset\n";
}

void partial_cleanup() {
    driver = {}; VmmManager manager(api);
    uintptr_t a; ze_physical_mem_handle_t h;
    assert(manager.reserve(owner, &a, 3 * block, block, 0, 0) == ZE_RESULT_SUCCESS);
    assert(manager.create(owner, &h, 3 * block) == ZE_RESULT_SUCCESS);
    for (int i = 0; i < 3; ++i)
        assert(manager.map(owner, a + i * block, block, h, i * block, 0) == ZE_RESULT_SUCCESS);
    driver.fail = "unmap"; driver.fail_after = 1;
    assert(manager.unmap(owner, a, 3 * block) == driver.error);
    check(manager, 1, 1, 2);
    assert(manager.map(owner, a, block, h, 0, 0) != ZE_RESULT_SUCCESS);
    assert(manager.unmap(owner, a + block, block) != ZE_RESULT_SUCCESS);
    assert(manager.access(owner, a + block, block) != ZE_RESULT_SUCCESS);
    assert(manager.unmap(owner, a, 3 * block) == ZE_RESULT_SUCCESS);
    assert(manager.destroy(owner, h) == ZE_RESULT_SUCCESS);
    assert(manager.free(owner, a, 3 * block) == ZE_RESULT_SUCCESS);
    check(manager, 0, 0, 0);
    std::cout << "PASS partial_unmap_cleanup_retry\n";
}

void concurrency() {
    driver = {}; VmmManager manager(api);
    auto work = [&] {
        for (int i = 0; i < 32; ++i) {
            uintptr_t a; ze_physical_mem_handle_t h;
            assert(manager.reserve(owner, &a, block, block, 0, 0) == ZE_RESULT_SUCCESS);
            assert(manager.create(owner, &h, block) == ZE_RESULT_SUCCESS);
            assert(manager.map(owner, a, block, h, 0, 0) == ZE_RESULT_SUCCESS);
            assert(manager.access(owner, a, block) == ZE_RESULT_SUCCESS);
            assert(manager.unmap(owner, a, block) == ZE_RESULT_SUCCESS);
            assert(manager.destroy(owner, h) == ZE_RESULT_SUCCESS);
            assert(manager.free(owner, a, block) == ZE_RESULT_SUCCESS);
        }
    };
    std::thread first(work), second(work), third(work), fourth(work);
    first.join(); second.join(); third.join(); fourth.join();
    check(manager, 0, 0, 0);
    std::cout << "PASS concurrent_independent_owners\n";
}

int main() {
    for (auto operation : {"page", "reserve", "create", "map", "access", "unmap", "destroy", "free"})
        for (auto error : {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, ZE_RESULT_ERROR_DEVICE_LOST})
            failure_case(operation, error);
    bounds_and_identity(); partial_cleanup(); concurrency();
    std::cout << "19 cases passed\n";
}
