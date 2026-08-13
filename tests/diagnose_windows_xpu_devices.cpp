#include <ze_api.h>

#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void describe_dxgi_adapters() {
    IDXGIFactory4 *factory = nullptr;
    const HRESULT factory_result = CreateDXGIFactory1(
        __uuidof(IDXGIFactory4), reinterpret_cast<void **>(&factory));
    std::printf("dxgi_factory_result=0x%08x\n",
                static_cast<unsigned>(factory_result));
    if (FAILED(factory_result)) {
        return;
    }

    for (UINT index = 0;; ++index) {
        IDXGIAdapter1 *adapter = nullptr;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 description{};
        adapter->GetDesc1(&description);
        char name[256]{};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1,
                            name, sizeof(name), nullptr, nullptr);
        std::printf("dxgi_adapter=%u name=%s luid=%08x:%08x "
                    "vendor=0x%04x device=0x%04x flags=0x%x "
                    "dedicated=%zu shared=%zu\n",
                    index, name,
                    static_cast<unsigned>(description.AdapterLuid.HighPart),
                    description.AdapterLuid.LowPart, description.VendorId,
                    description.DeviceId, description.Flags,
                    description.DedicatedVideoMemory,
                    description.SharedSystemMemory);

        IDXGIAdapter3 *adapter3 = nullptr;
        if (SUCCEEDED(adapter->QueryInterface(
                __uuidof(IDXGIAdapter3),
                reinterpret_cast<void **>(&adapter3)))) {
            for (UINT segment = 0; segment < 2; ++segment) {
                DXGI_QUERY_VIDEO_MEMORY_INFO info{};
                const DXGI_MEMORY_SEGMENT_GROUP group = segment == 0
                    ? DXGI_MEMORY_SEGMENT_GROUP_LOCAL
                    : DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL;
                const HRESULT query_result = adapter3->QueryVideoMemoryInfo(
                    0, group, &info);
                std::printf("  dxgi_memory segment=%s result=0x%08x "
                            "budget=%" PRIu64 " usage=%" PRIu64
                            " reservation=%" PRIu64 " available=%" PRIu64
                            "\n",
                            segment == 0 ? "local" : "nonlocal",
                            static_cast<unsigned>(query_result), info.Budget,
                            info.CurrentUsage, info.CurrentReservation,
                            info.AvailableForReservation);
            }
            adapter3->Release();
        }

        ID3D12Device *d3d_device = nullptr;
        const HRESULT d3d_result = D3D12CreateDevice(
            adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
            reinterpret_cast<void **>(&d3d_device));
        std::printf("  d3d12_result=0x%08x node_count=%u\n",
                    static_cast<unsigned>(d3d_result),
                    d3d_device ? d3d_device->GetNodeCount() : 0);
        if (d3d_device) {
            d3d_device->Release();
        }
        adapter->Release();
    }
    factory->Release();
}

const char *result_name(ze_result_t result) {
    switch (result) {
    case ZE_RESULT_SUCCESS: return "success";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "unsupported_feature";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "invalid_argument";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "uninitialized";
    default: return "error";
    }
}

void print_result(const char *operation, ze_result_t result) {
    std::printf("  %s_result=0x%08x (%s)\n", operation,
                static_cast<unsigned>(result), result_name(result));
}

void print_uuid(const ze_device_uuid_t &uuid) {
    for (unsigned char byte : uuid.id) {
        std::printf("%02x", static_cast<unsigned>(byte));
    }
}

void print_luid(const ze_device_luid_ext_t &luid) {
    for (unsigned char byte : luid.id) {
        std::printf("%02x", static_cast<unsigned>(byte));
    }
}

void describe_device(ze_driver_handle_t driver, ze_context_handle_t context,
                     ze_device_handle_t device, uint32_t index) {
    ze_device_luid_ext_properties_t luid{
        ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES, nullptr, {}, 0};
    ze_device_properties_t properties{
        ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &luid};
    ze_result_t result = zeDeviceGetProperties(device, &properties);

    std::printf("device=%u\n", index);
    print_result("properties", result);
    if (result == ZE_RESULT_SUCCESS) {
        std::printf("  name=%s\n", properties.name);
        std::printf("  vendor_id=0x%04x device_id=0x%04x flags=0x%x\n",
                    properties.vendorId, properties.deviceId,
                    static_cast<unsigned>(properties.flags));
        std::printf("  uuid=");
        print_uuid(properties.uuid);
        std::printf("\n  luid=");
        print_luid(luid.luid);
        std::printf(" node_mask=0x%x\n", luid.nodeMask);
        std::printf("  core_clock_mhz=%u max_alloc=%" PRIu64
                    " slices=%u subslices_per_slice=%u eus_per_subslice=%u\n",
                    properties.coreClockRate, properties.maxMemAllocSize,
                    properties.numSlices, properties.numSubslicesPerSlice,
                    properties.numEUsPerSubslice);
    }

    ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES, nullptr};
    result = zeDevicePciGetPropertiesExt(device, &pci);
    print_result("pci", result);
    if (result == ZE_RESULT_SUCCESS) {
        std::printf("  pci=%04x:%02x:%02x.%x max_gen=%d max_width=%d "
                    "max_bandwidth=%" PRId64 "\n",
                    pci.address.domain, pci.address.bus, pci.address.device,
                    pci.address.function, pci.maxSpeed.genVersion,
                    pci.maxSpeed.width, pci.maxSpeed.maxBandwidth);
    }

    uint32_t memory_count = 0;
    result = zeDeviceGetMemoryProperties(device, &memory_count, nullptr);
    print_result("memory_count", result);
    if (result == ZE_RESULT_SUCCESS) {
        std::vector<ze_device_memory_properties_t> memories(memory_count);
        for (auto &memory : memories) {
            memory.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        }
        result = zeDeviceGetMemoryProperties(device, &memory_count,
                                             memories.data());
        print_result("memory_properties", result);
        if (result == ZE_RESULT_SUCCESS) {
            for (uint32_t ordinal = 0; ordinal < memory_count; ++ordinal) {
                const auto &memory = memories[ordinal];
                std::printf("  memory[%u] name=%s total=%" PRIu64
                            " clock=%u bus_width=%u flags=0x%x\n",
                            ordinal, memory.name, memory.totalSize,
                            memory.maxClockRate, memory.maxBusWidth,
                            static_cast<unsigned>(memory.flags));
            }
        }
    }

    ze_device_memory_access_properties_t access{
        ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES, nullptr};
    result = zeDeviceGetMemoryAccessProperties(device, &access);
    print_result("memory_access", result);
    if (result == ZE_RESULT_SUCCESS) {
        std::printf("  access host=0x%x device=0x%x shared_single=0x%x "
                    "shared_cross=0x%x shared_system=0x%x\n",
                    static_cast<unsigned>(access.hostAllocCapabilities),
                    static_cast<unsigned>(access.deviceAllocCapabilities),
                    static_cast<unsigned>(
                        access.sharedSingleDeviceAllocCapabilities),
                    static_cast<unsigned>(
                        access.sharedCrossDeviceAllocCapabilities),
                    static_cast<unsigned>(access.sharedSystemAllocCapabilities));
    }

    uint32_t group_count = 0;
    result = zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                     nullptr);
    print_result("queue_group_count", result);
    if (result == ZE_RESULT_SUCCESS) {
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto &group : groups) {
            group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        result = zeDeviceGetCommandQueueGroupProperties(
            device, &group_count, groups.data());
        print_result("queue_groups", result);
        if (result == ZE_RESULT_SUCCESS) {
            for (uint32_t ordinal = 0; ordinal < group_count; ++ordinal) {
                const auto &group = groups[ordinal];
                std::printf("  queue_group[%u] flags=0x%x queues=%u "
                            "max_fill_pattern=%zu\n",
                            ordinal, static_cast<unsigned>(group.flags),
                            group.numQueues, group.maxMemoryFillPatternSize);
            }
        }
    }

    constexpr std::array<size_t, 6> query_sizes{
        4096, 607744, 2ULL * 1024 * 1024,
        2ULL * 1024 * 1024 + 1, 32ULL * 1024 * 1024,
        64ULL * 1024 * 1024};
    for (size_t query_size : query_sizes) {
        size_t page_size = 0;
        result = zeVirtualMemQueryPageSize(context, device, query_size,
                                           &page_size);
        std::printf("  vmm_page_size query=%zu result=0x%08x page=%zu\n",
                    query_size, static_cast<unsigned>(result), page_size);
    }
}

} // namespace

int main() {
    describe_dxgi_adapters();

    ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (result != ZE_RESULT_SUCCESS) {
        print_result("init", result);
        return 1;
    }

    uint32_t driver_count = 0;
    result = zeDriverGet(&driver_count, nullptr);
    if (result != ZE_RESULT_SUCCESS || driver_count == 0) {
        print_result("driver_count", result);
        return 1;
    }

    std::vector<ze_driver_handle_t> drivers(driver_count);
    result = zeDriverGet(&driver_count, drivers.data());
    if (result != ZE_RESULT_SUCCESS) {
        print_result("drivers", result);
        return 1;
    }

    std::printf("driver_count=%u\n", driver_count);
    uint32_t global_device_index = 0;
    for (uint32_t driver_index = 0; driver_index < driver_count;
         ++driver_index) {
        ze_api_version_t version = ZE_API_VERSION_1_0;
        zeDriverGetApiVersion(drivers[driver_index], &version);
        std::printf("driver=%u api=%u.%u\n", driver_index,
                    ZE_MAJOR_VERSION(version), ZE_MINOR_VERSION(version));

        ze_context_desc_t context_desc{
            ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        ze_context_handle_t context = nullptr;
        result = zeContextCreate(drivers[driver_index], &context_desc,
                                 &context);
        if (result != ZE_RESULT_SUCCESS) {
            print_result("context_create", result);
            continue;
        }

        uint32_t device_count = 0;
        result = zeDeviceGet(drivers[driver_index], &device_count, nullptr);
        if (result != ZE_RESULT_SUCCESS) {
            print_result("device_count", result);
            zeContextDestroy(context);
            continue;
        }
        std::vector<ze_device_handle_t> devices(device_count);
        result = zeDeviceGet(drivers[driver_index], &device_count,
                             devices.data());
        if (result != ZE_RESULT_SUCCESS) {
            print_result("devices", result);
            zeContextDestroy(context);
            continue;
        }
        for (ze_device_handle_t device : devices) {
            describe_device(drivers[driver_index], context, device,
                            global_device_index++);
        }
        zeContextDestroy(context);
    }

    return 0;
}
