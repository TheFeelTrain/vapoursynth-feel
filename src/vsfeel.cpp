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
            .apiVersion = VK_API_VERSION_1_2
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

    vkGetPhysicalDeviceMemoryProperties(dev->physical_device, &dev->mem_props);

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

    const float queue_priority { 1.0f };
    VkDeviceQueueCreateInfo queue_info {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = best_family,
        .queueCount = best_queues,
        .pQueuePriorities = &queue_priority
    };

    VkPhysicalDeviceFeatures features {};
    features.shaderFloat64 = VK_TRUE;

    VkDeviceCreateInfo device_info {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
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
    size_t i = 0;
    for (; i + 64 <= bytes; i += 64) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i),
            _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(s + i)));
        _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i + 32),
            _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(s + i + 32)));
    }
    if (i < bytes) {
        memcpy(d + i, s + i, bytes - i);
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
}
