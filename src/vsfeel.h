#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include <vulkan/vulkan.h>

#include <VapourSynth4.h>

#define checkVK(expr) do {                                                          \
    if (VkResult __result = (expr); __result != VK_SUCCESS) [[unlikely]] {          \
        return set_error("'"s + #expr + "' failed: " + vk_result_string(__result));\
    }                                                                               \
} while(0)

struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

const char * vk_result_string(VkResult result);

// ---------------------------------------------------------------------------
// Shared, reference-counted VkDevice (one per physical device)
// ---------------------------------------------------------------------------

struct VK_Queue {
    VkQueue queue {};
    std::unique_ptr<std::mutex> lock { std::make_unique<std::mutex>() };
};

struct VK_Device {
    VkInstance instance {};
    VkPhysicalDevice physical_device {};
    VkDevice device {};
    VkPhysicalDeviceMemoryProperties mem_props {};
    VkPhysicalDeviceLimits limits {};
    uint32_t queue_family {};
    uint32_t queue_count {};
    std::vector<VK_Queue> queues {};
    std::atomic<intptr_t> refcount { 0 };
};

std::variant<std::shared_ptr<VK_Device>, std::string> get_device(int device_id);
void release_device(const std::shared_ptr<VK_Device> & dev);

// Streaming copy: non-temporal stores bypass the CPU cache so the freshly
// written lines sit clean in DRAM; the GPU can then read them over PCIe
// without snoop/writeback stalls.
void copy_stream_out(void * dst, const void * src, size_t bytes);

// Streaming copy variant that reads the GPU-written staging without caching
// it (non-temporal loads) before writing the destination with streaming stores.
// The source must be 32-byte aligned (staging download offsets always are).
void copy_stream_read(void * dst, const void * src, size_t bytes);

struct AllocatedMemory {
    VkDeviceMemory memory;
    uint32_t type_index;
};

std::variant<AllocatedMemory, std::string> allocate_memory(
    const VK_Device & dev, VkBuffer buffer, VkMemoryPropertyFlags required);

// ---------------------------------------------------------------------------
// Filter registration
// ---------------------------------------------------------------------------

void vsfeel_register_bilateral(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_bm3dv2(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_gaussblur(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
