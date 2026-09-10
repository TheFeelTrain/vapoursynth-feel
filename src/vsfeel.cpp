#include <immintrin.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <vulkan/vulkan.h>

#include <VapourSynth4.h>

#include "vsfeel.h"

using namespace std::string_literals;

const char * vk_result_string(VkResult result) {
    switch (result) {
        case VK_SUCCESS:                      return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY:     return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:   return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_DEVICE_LOST:            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_INCOMPATIBLE_DRIVER:    return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_EXTENSION_NOT_PRESENT:  return "VK_ERROR_EXTENSION_NOT_PRESENT";
        default:
            return "<unknown VkResult>";
    }
}

// ---------------------------------------------------------------------------
// Shared, reference-counted VkDevice (one per physical device)
// ---------------------------------------------------------------------------

static std::mutex g_device_lock;
static std::map<int, std::shared_ptr<VK_Device>> g_devices;

std::variant<std::shared_ptr<VK_Device>, std::string> get_device(int device_id) {
    std::lock_guard lock(g_device_lock);

    if (auto it = g_devices.find(device_id); it != g_devices.end()) {
        ++it->second->refcount;
        return it->second;
    }

    auto dev = std::make_shared<VK_Device>();

    {
        VkApplicationInfo app_info {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "vsfeel",
            .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .pEngineName = "vsfeel",
            .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .apiVersion = VK_API_VERSION_1_4
        };

        VkInstanceCreateInfo instance_info {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = nullptr
        };

        VkResult result = vkCreateInstance(&instance_info, nullptr, &dev->instance);
        if (result != VK_SUCCESS) {
            return "vkCreateInstance failed: "s + vk_result_string(result);
        }
    }

    uint32_t device_count = 0;
    VkResult result = vkEnumeratePhysicalDevices(dev->instance, &device_count, nullptr);
    if (result != VK_SUCCESS) {
        vkDestroyInstance(dev->instance, nullptr);
        return "vkEnumeratePhysicalDevices failed: "s + vk_result_string(result);
    }

    if (device_id < 0 || static_cast<uint32_t>(device_id) >= device_count) {
        vkDestroyInstance(dev->instance, nullptr);
        return "invalid device ID (" + std::to_string(device_id) + ")";
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    result = vkEnumeratePhysicalDevices(
        dev->instance, &device_count, physical_devices.data());
    if (result != VK_SUCCESS) {
        vkDestroyInstance(dev->instance, nullptr);
        return "vkEnumeratePhysicalDevices failed: "s + vk_result_string(result);
    }

    dev->physical_device = physical_devices[device_id];

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(dev->physical_device, &props);
    dev->limits = props.limits;

    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_props {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
        .pNext = nullptr
    };
    VkPhysicalDeviceProperties2 props2 {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_props
    };
    vkGetPhysicalDeviceProperties2(dev->physical_device, &props2);
    if (subgroup_props.minSubgroupSize && subgroup_props.maxSubgroupSize) {
        dev->min_subgroup_size = subgroup_props.minSubgroupSize;
        dev->max_subgroup_size = subgroup_props.maxSubgroupSize;
    }
    dev->subgroup_size_control =
        dev->min_subgroup_size <= 32 && 32 <= dev->max_subgroup_size;
    if (trace_on("VSFEEL_DBG")) {
        fprintf(stderr, "[vsfeel] subgroup_size_control=%d min=%u max=%u\n",
            dev->subgroup_size_control, dev->min_subgroup_size, dev->max_subgroup_size);
    }

    vkGetPhysicalDeviceMemoryProperties(dev->physical_device, &dev->mem_props);

    // Host-pointer import (VK_EXT_external_memory_host): lets a kernel write
    // straight into a VapourSynth frame's plane memory, removing a host
    // staging round trip. Optional; query the alignment requirement and the
    // extension's presence before anyone relies on it.
    dev->host_import = false;
    {
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(dev->physical_device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(dev->physical_device, nullptr, &ext_count, exts.data());
        bool present = false;
        for (const auto & e : exts) {
            if (std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
                present = true;
                break;
            }
        }
        if (present) {
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
                .pNext = nullptr
            };
            VkPhysicalDeviceProperties2 p2 {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &host_props
            };
            vkGetPhysicalDeviceProperties2(dev->physical_device, &p2);
            dev->host_pointer_alignment = host_props.minImportedHostPointerAlignment;
            dev->host_import = true;
        }
        if (trace_on("VSFEEL_DBG")) {
            fprintf(stderr, "[vsfeel] host_import=%d min_align=%llu\n",
                dev->host_import,
                (unsigned long long)dev->host_pointer_alignment);
        }
    }

    // Pick a compute-capable queue family, prefer the one with the most queues
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        dev->physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> family_props(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        dev->physical_device, &family_count, family_props.data());

    uint32_t best_family = ~0u;
    uint32_t best_queues = 0;
    for (uint32_t i = 0; i < family_count; ++i) {
        if (family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (best_family == ~0u || family_props[i].queueCount > best_queues) {
                best_family = i;
                best_queues = family_props[i].queueCount;
            }
        }
    }
    if (best_family == ~0u) {
        vkDestroyInstance(dev->instance, nullptr);
        return "no compute-capable queue family found";
    }
    dev->queue_family = best_family;
    dev->queue_count = best_queues;

    // One priority per requested queue: the specification (and every driver)
    // reads pQueuePriorities[0 .. queueCount), so the array must be that long.
    // A single float here made the driver read past the object.
    std::vector<float> queue_priorities(best_queues, 1.0f);
    VkDeviceQueueCreateInfo queue_info {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = best_family,
        .queueCount = best_queues,
        .pQueuePriorities = queue_priorities.data()
    };

    // Query the physical device for every feature the shipped shaders need
    // before asking for it. The answers are recorded in VK_Device so a filter
    // whose shaders require a missing feature can report a precise creation
    // error instead of relying on the driver accepting the pipeline anyway.
    // The extension list comes first: an extension feature struct must only be
    // chained when the extension itself is supported.
    bool atomic_float_ext = false;
    {
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(dev->physical_device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        if (ext_count) {
            vkEnumerateDeviceExtensionProperties(dev->physical_device, nullptr, &ext_count, exts.data());
            for (const auto & e : exts) {
                if (std::strcmp(e.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0) {
                    atomic_float_ext = true;
                    break;
                }
            }
        }
    }

    VkPhysicalDeviceVulkan11Features supported_11 {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = nullptr
    };
    VkPhysicalDeviceVulkan12Features supported_12 {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &supported_11
    };
    VkPhysicalDeviceVulkan13Features supported_13 {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &supported_12
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_float {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &supported_13
    };
    VkPhysicalDeviceSubgroupSizeControlFeatures supported_subgroup {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .pNext = atomic_float_ext ? static_cast<void *>(&supported_atomic_float)
                                  : static_cast<void *>(&supported_13)
    };
    VkPhysicalDeviceFeatures2 supported_features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &supported_subgroup,
        .features = {}
    };
    vkGetPhysicalDeviceFeatures2(dev->physical_device, &supported_features);

    dev->feat_float64 = supported_features.features.shaderFloat64 == VK_TRUE;
    dev->feat_16bit_storage = supported_11.storageBuffer16BitAccess == VK_TRUE;
    dev->feat_8bit_storage = supported_12.storageBuffer8BitAccess == VK_TRUE &&
                             supported_12.shaderInt8 == VK_TRUE;
    dev->feat_vulkan_memory_model = supported_12.vulkanMemoryModel == VK_TRUE;
    dev->feat_maintenance4 = supported_13.maintenance4 == VK_TRUE;
    dev->feat_atomic_float32_add = atomic_float_ext &&
        supported_atomic_float.shaderBufferFloat32AtomicAdd == VK_TRUE;
    dev->feat_compute_full_subgroups =
        supported_subgroup.computeFullSubgroups == VK_TRUE;

    VkPhysicalDeviceFeatures features {};
    features.shaderFloat64 = dev->feat_float64 ? VK_TRUE : VK_FALSE;

    // EEDI3 stores its int8 predecessor/direction/mask buffers in SSBOs and
    // does int8 ALU on them; its row/vcheck shaders use
    // GL_KHR_memory_scope_semantics (VulkanMemoryModel, SPIR-V `OpMemoryModel
    // Logical Vulkan`) and `local_size_x_id` (LocalSizeId, which needs
    // maintenance4). BM3D accumulates into float SSBOs with atomicAdd, which
    // needs VK_EXT_shader_atomic_float. Timeline semaphores (core 1.2) back
    // the DFTTest/BM3D frame caches. All of these live in the 1.1/1.2/1.3
    // feature structs; chaining the individual promoted structs alongside
    // them is what VUID-VkDeviceCreateInfo-pNext-02830 forbids.
    VkPhysicalDeviceVulkan12Features vulkan12_features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = nullptr,
        .storageBuffer8BitAccess = supported_12.storageBuffer8BitAccess,
        .uniformAndStorageBuffer8BitAccess = supported_12.uniformAndStorageBuffer8BitAccess,
        .storagePushConstant8 = VK_FALSE,
        .shaderFloat16 = VK_FALSE,
        .shaderInt8 = supported_12.shaderInt8,
        .descriptorIndexing = VK_FALSE,
        .timelineSemaphore = supported_12.timelineSemaphore,
        .vulkanMemoryModel = supported_12.vulkanMemoryModel,
        .vulkanMemoryModelDeviceScope = supported_12.vulkanMemoryModelDeviceScope
    };

    VkPhysicalDeviceVulkan11Features vulkan11_features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &vulkan12_features,
        .storageBuffer16BitAccess = supported_11.storageBuffer16BitAccess,
        .uniformAndStorageBuffer16BitAccess = supported_11.uniformAndStorageBuffer16BitAccess,
        .storagePushConstant16 = VK_FALSE,
        .storageInputOutput16 = VK_FALSE
    };

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &vulkan11_features,
        .shaderBufferFloat32AtomicAdd =
            dev->feat_atomic_float32_add ? VK_TRUE : VK_FALSE
    };

    VkPhysicalDeviceVulkan13Features vulkan13_features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &atomic_float_features,
        .subgroupSizeControl = supported_13.subgroupSizeControl,
        .computeFullSubgroups = supported_13.computeFullSubgroups,
        .maintenance4 = supported_13.maintenance4
    };

    // VK_EXT_shader_atomic_float carries the atomicAdd(float) feature; it is
    // not promoted to core, so the extension has to be enabled for the
    // feature struct to be meaningful.
    std::vector<const char *> device_exts { VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME };
    if (dev->host_import) {
        device_exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    }
    if (atomic_float_ext) {
        device_exts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }

    VkDeviceCreateInfo device_info {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vulkan13_features,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(device_exts.size()),
        .ppEnabledExtensionNames = device_exts.data(),
        .pEnabledFeatures = &features
    };

    result = vkCreateDevice(dev->physical_device, &device_info, nullptr, &dev->device);
    if (result != VK_SUCCESS) {
        vkDestroyInstance(dev->instance, nullptr);
        return "vkCreateDevice failed: "s + vk_result_string(result);
    }

    dev->queues.reserve(dev->queue_count);
    for (uint32_t i = 0; i < dev->queue_count; ++i) {
        VkQueue queue;
        vkGetDeviceQueue(dev->device, best_family, i, &queue);
        dev->queues.push_back(VK_Queue { queue, std::make_unique<std::mutex>() });
    }
    ++dev->refcount;
    g_devices.emplace(device_id, dev);

    return dev;
}

void release_device(const std::shared_ptr<VK_Device> & dev) {
    std::lock_guard lock(g_device_lock);
    if (--dev->refcount == 0) {
        vkDeviceWaitIdle(dev->device);
        vkDestroyDevice(dev->device, nullptr);
        vkDestroyInstance(dev->instance, nullptr);
        for (auto it = g_devices.begin(); it != g_devices.end(); ++it) {
            if (it->second == dev) {
                g_devices.erase(it);
                break;
            }
        }
    }
}

void copy_stream_out(void * dst, const void * src, size_t bytes) {
    const uint8_t * s = static_cast<const uint8_t *>(src);
    uint8_t * d = static_cast<uint8_t *>(dst);
    const uintptr_t align = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
    if (align > bytes) {
        memcpy(d, s, bytes);
        return;
    }
    memcpy(d, s, align);
    s += align;
    d += align;
    bytes -= align;
    size_t i = 0;
    for (; i + 64 <= bytes; i += 64) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i),
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i)));
        _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i + 32),
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i + 32)));
    }
    if (i < bytes) {
        memcpy(d + i, s + i, bytes - i);
    }
}

void copy_stream_read(void * dst, const void * src, size_t bytes) {
    const uint8_t * s = static_cast<const uint8_t *>(src);
    uint8_t * d = static_cast<uint8_t *>(dst);
    const uintptr_t align = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
    if (align > bytes) {
        memcpy(d, s, bytes);
        return;
    }
    memcpy(d, s, align);
    s += align;
    d += align;
    bytes -= align;
    // movntdqa faults on a misaligned source, and rows inside a pitched plane
    // are routinely misaligned (any visible row width not a multiple of 32
    // bytes, e.g. a 630-pixel float32 plane). Keep the non-temporal load only
    // when the source actually is 32-byte aligned; the streaming *stores*
    // (which only need the destination, aligned above) are kept either way.
    const bool aligned_src = (reinterpret_cast<uintptr_t>(s) & 31) == 0;
    size_t i = 0;
    if (aligned_src) {
        for (; i + 64 <= bytes; i += 64) {
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i),
                _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(s + i)));
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i + 32),
                _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(s + i + 32)));
        }
    } else {
        for (; i + 64 <= bytes; i += 64) {
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i)));
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i + 32),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i + 32)));
        }
    }
    if (i < bytes) {
        memcpy(d + i, s + i, bytes - i);
    }
}

void copy_plane_out(void * dst, ptrdiff_t dst_pitch, const void * src,
                    ptrdiff_t src_pitch, size_t row_bytes, int height, bool nt) {
    if (height <= 0 || row_bytes == 0) {
        return;
    }
    const auto copy_row = [nt](void * d, const void * s, size_t n) {
        if (nt) {
            copy_stream_out(d, s, n);
        } else {
            memcpy(d, s, n);
        }
    };
    if (dst_pitch == static_cast<ptrdiff_t>(row_bytes) &&
        src_pitch == static_cast<ptrdiff_t>(row_bytes)) {
        copy_row(dst, src, row_bytes * static_cast<size_t>(height));
        return;
    }
    for (int y = 0; y < height; ++y) {
        copy_row(static_cast<uint8_t *>(dst) + static_cast<ptrdiff_t>(y) * dst_pitch,
                 static_cast<const uint8_t *>(src) + static_cast<ptrdiff_t>(y) * src_pitch,
                 row_bytes);
    }
}

void copy_plane_read(void * dst, ptrdiff_t dst_pitch, const void * src,
                     ptrdiff_t src_pitch, size_t row_bytes, int height) {
    if (height <= 0 || row_bytes == 0) {
        return;
    }
    if (dst_pitch == static_cast<ptrdiff_t>(row_bytes) &&
        src_pitch == static_cast<ptrdiff_t>(row_bytes)) {
        copy_stream_read(dst, src, row_bytes * static_cast<size_t>(height));
        return;
    }
    for (int y = 0; y < height; ++y) {
        copy_stream_read(static_cast<uint8_t *>(dst) + static_cast<ptrdiff_t>(y) * dst_pitch,
                         static_cast<const uint8_t *>(src) + static_cast<ptrdiff_t>(y) * src_pitch,
                         row_bytes);
    }
}

static std::optional<uint32_t> find_memory_type(
    const VK_Device & dev, uint32_t type_bits, VkMemoryPropertyFlags required) {

    for (uint32_t i = 0; i < dev.mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (dev.mem_props.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return std::nullopt;
}

std::variant<AllocatedMemory, std::string> allocate_memory(
    const VK_Device & dev, VkBuffer buffer, VkMemoryPropertyFlags required) {

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(dev.device, buffer, &requirements);

    // Try the requested flags, then relaxed variants, so that e.g. a staging
    // buffer prefers host-cached memory (fast CPU access) but still works on
    // drivers that only expose write-combined host memory.
    const VkMemoryPropertyFlags candidates[] {
        required,
        required & ~VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        required & ~VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    };

    auto type_index = std::optional<uint32_t> {};
    for (auto candidate : candidates) {
        type_index = find_memory_type(dev, requirements.memoryTypeBits, candidate);
        if (type_index) {
            break;
        }
    }
    if (!type_index) {
        return "no suitable memory type found";
    }

    VkMemoryAllocateInfo allocate_info {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = *type_index
    };

    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(dev.device, &allocate_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        return "vkAllocateMemory failed: "s + vk_result_string(result);
    }

    result = vkBindBufferMemory(dev.device, buffer, memory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(dev.device, memory, nullptr);
        return "vkBindBufferMemory failed: "s + vk_result_string(result);
    }

    return AllocatedMemory { memory, *type_index };
}

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        "com.thefeeltrain.vsfeel",
        "vsfeel",
        "GPU-accelerated VapourSynth filters (Vulkan)",
        VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION, 0, plugin
    );

    vsfeel_register_bilateral(vspapi, plugin);
    vsfeel_register_bm3dv2(vspapi, plugin);
    vsfeel_register_gaussblur(vspapi, plugin);
    vsfeel_register_dfttest(vspapi, plugin);
    vsfeel_register_nlmeans(vspapi, plugin);
    vsfeel_register_eedi3(vspapi, plugin);
    vsfeel_register_nnedi3(vspapi, plugin);
}
