#include <ze_api.h>

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr size_t kVbarPageSize = 32ULL * 1024 * 1024;

bool ze_ok(const char *operation, ze_result_t result) {
    std::printf("%s=0x%08x\n", operation, static_cast<unsigned>(result));
    return result == ZE_RESULT_SUCCESS;
}

bool verify(sycl::queue &queue, unsigned char *vmm,
            const unsigned char *expected, size_t size) {
    auto *device_copy = sycl::malloc_device<unsigned char>(size, queue);
    auto *host_copy = sycl::malloc_host<unsigned char>(size, queue);
    if (!device_copy || !host_copy) {
        return false;
    }

    try {
        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> index) {
            device_copy[index] = vmm[index];
        });
        queue.memcpy(host_copy, device_copy, size).wait_and_throw();
    } catch (const sycl::exception &error) {
        std::fprintf(stderr, "verify_exception=%s\n", error.what());
        sycl::free(device_copy, queue);
        sycl::free(host_copy, queue);
        return false;
    }

    const bool equal = std::memcmp(host_copy, expected, size) == 0;
    std::printf("verify=%s\n", equal ? "pass" : "fail");
    sycl::free(device_copy, queue);
    sycl::free(host_copy, queue);
    return equal;
}

bool run_ze_copy(ze_context_handle_t context, ze_device_handle_t device,
                 uint32_t ordinal, void *destination, const void *source,
                 size_t size) {
    ze_command_queue_desc_t description{
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        ordinal,
        0,
        0,
        ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    ze_command_list_handle_t command_list = nullptr;
    if (!ze_ok("ze_immediate_create",
               zeCommandListCreateImmediate(context, device, &description,
                                            &command_list))) {
        return false;
    }
    const ze_result_t append = zeCommandListAppendMemoryCopy(
        command_list, destination, source, size, nullptr, 0, nullptr);
    const bool appended = ze_ok("ze_copy_append", append);
    bool synchronized = false;
    if (appended) {
        synchronized = ze_ok("ze_copy_sync", zeCommandListHostSynchronize(
            command_list, UINT64_MAX));
    }
    ze_ok("ze_immediate_destroy", zeCommandListDestroy(command_list));
    return appended && synchronized;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr,
                     "usage: diagnose-windows-xpu-vmm-copy "
                     "<direct|fill-kernel|host-kernel|staged-kernel|"
                     "padded-staged-kernel|padded-rmw|ze-compute|ze-copy> "
                     "<bytes> [backing-bytes]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const size_t size = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
    const size_t backing_size = argc == 4
        ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10))
        : kVbarPageSize;
    if (size == 0 || backing_size < size) {
        std::fprintf(stderr, "invalid size\n");
        return 2;
    }

    try {
        sycl::queue queue{sycl::gpu_selector_v,
                          sycl::property::queue::in_order{}};
        const auto sycl_device = queue.get_device();
        const std::string name =
            sycl_device.get_info<sycl::info::device::name>();
        const auto context =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                queue.get_context());
        const auto device =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                sycl_device);
        size_t queried_page_size = 0;
        const ze_result_t page_query = zeVirtualMemQueryPageSize(
            context, device, backing_size, &queried_page_size);
        std::printf("mode=%s size=%zu backing=%zu page_query=0x%08x "
                    "page_size=%zu device=%s context=%p native_device=%p\n",
                    mode.c_str(), size, backing_size,
                    static_cast<unsigned>(page_query), queried_page_size,
                    name.c_str(),
                    static_cast<void *>(context), static_cast<void *>(device));

        void *address = nullptr;
        if (!ze_ok("vmm_reserve", zeVirtualMemReserve(
                context, nullptr, backing_size, &address))) {
            return 1;
        }

        ze_physical_mem_desc_t physical_description{
            ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
            nullptr,
            ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE,
            backing_size,
        };
        ze_physical_mem_handle_t physical = nullptr;
        if (!ze_ok("physical_create", zePhysicalMemCreate(
                context, device, &physical_description, &physical))) {
            zeVirtualMemFree(context, address, backing_size);
            return 1;
        }
        if (!ze_ok("vmm_map", zeVirtualMemMap(
                context, address, backing_size, physical, 0,
                ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE))) {
            zePhysicalMemDestroy(context, physical);
            zeVirtualMemFree(context, address, backing_size);
            return 1;
        }

        auto *source = sycl::malloc_host<unsigned char>(size, queue);
        if (!source) {
            return 1;
        }
        for (size_t index = 0; index < size; ++index) {
            source[index] = static_cast<unsigned char>((index * 131 + 17) & 0xff);
        }
        if (mode == "fill-kernel") {
            std::memset(source, 7, size);
        }

        bool copied = false;
        bool verification_done = false;
        bool verification_passed = false;
        auto *destination = static_cast<unsigned char *>(address);
        try {
            if (mode == "direct") {
                queue.memcpy(destination, source, size).wait_and_throw();
                std::puts("stage=direct_complete");
                copied = true;
            } else if (mode == "fill-kernel") {
                queue.parallel_for(sycl::range<1>(size),
                                   [=](sycl::id<1> index) {
                    destination[index] = 7;
                }).wait_and_throw();
                std::puts("stage=fill_kernel_complete");
                copied = true;
            } else if (mode == "host-kernel") {
                queue.parallel_for(sycl::range<1>(size),
                                   [=](sycl::id<1> index) {
                    destination[index] = source[index];
                }).wait_and_throw();
                std::puts("stage=host_kernel_complete");
                copied = true;
            } else if (mode == "staged-kernel") {
                auto *staging = sycl::malloc_device<unsigned char>(size, queue);
                std::printf("stage=device_alloc pointer=%p\n",
                            static_cast<void *>(staging));
                if (staging) {
                    queue.memcpy(staging, source, size).wait_and_throw();
                    std::puts("stage=host_to_staging_complete");
                    queue.parallel_for(sycl::range<1>(size),
                                       [=](sycl::id<1> index) {
                        destination[index] = staging[index];
                    }).wait_and_throw();
                    std::puts("stage=staging_kernel_complete");
                    sycl::free(staging, queue);
                    copied = true;
                }
            } else if (mode == "padded-staged-kernel") {
                constexpr size_t safe_size = 2ULL * 1024 * 1024 + 1;
                auto *host_staging = sycl::malloc_host<unsigned char>(
                    safe_size, queue);
                auto *device_staging = sycl::malloc_device<unsigned char>(
                    safe_size, queue);
                auto *readback = sycl::malloc_host<unsigned char>(
                    safe_size, queue);
                if (host_staging && device_staging && readback &&
                    backing_size >= safe_size) {
                    std::memset(host_staging, 0, safe_size);
                    std::memcpy(host_staging, source, size);
                    queue.memcpy(device_staging, host_staging, safe_size)
                        .wait_and_throw();
                    std::puts("stage=padded_host_to_staging_complete");
                    queue.parallel_for(sycl::range<1>(size),
                                       [=](sycl::id<1> index) {
                        destination[index] = device_staging[index];
                    }).wait_and_throw();
                    std::puts("stage=padded_staging_kernel_complete");
                    queue.memcpy(readback, destination, safe_size)
                        .wait_and_throw();
                    std::puts("stage=padded_destination_readback_complete");
                    verification_passed =
                        std::memcmp(readback, source, size) == 0;
                    verification_done = true;
                    copied = true;
                }
                if (host_staging) {
                    sycl::free(host_staging, queue);
                }
                if (device_staging) {
                    sycl::free(device_staging, queue);
                }
                if (readback) {
                    sycl::free(readback, queue);
                }
            } else if (mode == "padded-rmw") {
                constexpr size_t safe_size = 2ULL * 1024 * 1024 + 1;
                if (backing_size < safe_size) {
                    std::fprintf(stderr, "backing is too small for padded-rmw\n");
                } else {
                    auto *window = sycl::malloc_host<unsigned char>(
                        safe_size, queue);
                    auto *readback = sycl::malloc_host<unsigned char>(
                        safe_size, queue);
                    if (window && readback) {
                        queue.parallel_for(
                            sycl::range<1>(safe_size),
                            [=](sycl::id<1> index) {
                                destination[index] = 7;
                            }).wait_and_throw();
                        std::puts("stage=padded_initialize_complete");
                        queue.memcpy(window, destination, safe_size)
                            .wait_and_throw();
                        std::puts("stage=padded_read_complete");
                        std::memcpy(window, source, size);
                        queue.memcpy(destination, window, safe_size)
                            .wait_and_throw();
                        std::puts("stage=padded_write_complete");
                        queue.memcpy(readback, destination, safe_size)
                            .wait_and_throw();
                        std::puts("stage=padded_readback_complete");
                        verification_passed =
                            std::memcmp(readback, source, size) == 0 &&
                            std::all_of(readback + size,
                                        readback + safe_size,
                                        [](unsigned char value) {
                                            return value == 7;
                                        });
                        verification_done = true;
                        copied = true;
                    }
                    if (window) {
                        sycl::free(window, queue);
                    }
                    if (readback) {
                        sycl::free(readback, queue);
                    }
                }
            } else if (mode == "ze-compute") {
                copied = run_ze_copy(context, device, 0, destination, source,
                                     size);
            } else if (mode == "ze-copy") {
                copied = run_ze_copy(context, device, 1, destination, source,
                                     size);
            } else {
                std::fprintf(stderr, "unknown mode\n");
            }
        } catch (const sycl::exception &error) {
            std::fprintf(stderr, "copy_exception=%s\n", error.what());
        }

        std::printf("copy=%s\n", copied ? "pass" : "fail");
        const bool verified = verification_done
            ? verification_passed
            : copied && verify(queue, destination, source, size);
        if (verification_done) {
            std::printf("verify=%s\n", verified ? "pass" : "fail");
        }
        sycl::free(source, queue);
        ze_ok("vmm_unmap", zeVirtualMemUnmap(
            context, address, backing_size));
        ze_ok("physical_destroy", zePhysicalMemDestroy(context, physical));
        ze_ok("vmm_free", zeVirtualMemFree(
            context, address, backing_size));
        return verified ? 0 : 1;
    } catch (const sycl::exception &error) {
        std::fprintf(stderr, "setup_exception=%s\n", error.what());
        return 1;
    }
}
