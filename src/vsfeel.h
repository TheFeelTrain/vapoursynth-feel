#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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
    // VK_EXT_external_memory_host: whether a host pointer can be imported as a
    // Vulkan buffer, and the alignment it must satisfy.
    bool host_import {};
    VkDeviceSize host_pointer_alignment {};
    uint32_t min_subgroup_size { 64 };
    uint32_t max_subgroup_size { 64 };
    bool subgroup_size_control { false };
    // Feature availability as queried from the physical device and enabled at
    // device creation. A filter whose shaders need one of these reports a
    // precise creation error instead of relying on the driver accepting the
    // pipeline anyway.
    bool feat_float64 { false };
    bool feat_atomic_float32_add { false };
    bool feat_vulkan_memory_model { false };
    bool feat_maintenance4 { false };
    bool feat_compute_full_subgroups { false };
    bool feat_8bit_storage { false };
    bool feat_16bit_storage { false };
    // Persistent pipeline cache: compiling the compute shaders from SPIR-V is
    // by far the most expensive part of filter creation (seconds per variant
    // on RADV), and every new filter instance would otherwise pay it again in
    // every process. A VkPipelineCache seeded from and flushed to a file makes
    // it a one-time cost per (device, driver) pair.
    VkPipelineCache pipeline_cache {};
    std::mutex pipeline_cache_lock;      // serializes cache data extraction
    std::string pipeline_cache_path;     // empty = cache disabled
    std::vector<VK_Queue> queues {};
    std::atomic<intptr_t> refcount { 0 };
};

std::variant<std::shared_ptr<VK_Device>, std::string> get_device(int device_id);
void release_device(const std::shared_ptr<VK_Device> & dev);

// Flush the device's compiled-pipeline cache to its file (best effort, no-op
// when the cache is disabled or nothing was compiled). Called when the last
// instance releases the device and once at process exit, so subprocess-based
// runs (vspipe, test comparisons) still contribute to the cache.
void save_pipeline_cache(VK_Device & dev);

// Streaming copy: non-temporal stores bypass the CPU cache so the freshly
// written lines sit clean in DRAM; the GPU can then read them over PCIe
// without snoop/writeback stalls.
void copy_stream_out(void * dst, const void * src, size_t bytes);

// Streaming copy variant that reads the GPU-written staging without caching
// it (non-temporal loads) before writing the destination with streaming stores.
// The 32-byte-aligned source fast path is used only when the source actually
// is aligned; anything else falls back to unaligned loads (rows inside a
// pitched plane are frequently misaligned).
void copy_stream_read(void * dst, const void * src, size_t bytes);

// Copy `height` visible rows of `row_bytes` bytes from `src` (row pitch
// `src_pitch`) to `dst` (row pitch `dst_pitch`). When every pitch equals
// `row_bytes` the whole plane is one contiguous copy; otherwise the rows are
// copied one at a time, so padded VapourSynth frame strides and GPU-aligned
// plane pitches are both handled. `nt` selects streaming stores (right for the
// GTT staging buffers) instead of ordinary cached stores (right for the
// write-combined VRAM BAR window).
void copy_plane_out(void * dst, ptrdiff_t dst_pitch, const void * src,
                    ptrdiff_t src_pitch, size_t row_bytes, int height, bool nt);

// Row-wise counterpart of copy_stream_read with independent source and
// destination pitches (GPU plane pitch in, frame stride out).
void copy_plane_read(void * dst, ptrdiff_t dst_pitch, const void * src,
                     ptrdiff_t src_pitch, size_t row_bytes, int height);

struct AllocatedMemory {
    VkDeviceMemory memory;
    uint32_t type_index;
};

std::variant<AllocatedMemory, std::string> allocate_memory(
    const VK_Device & dev, VkBuffer buffer, VkMemoryPropertyFlags required);

// ---------------------------------------------------------------------------
// Shared per-frame plumbing (zero-overhead inline helpers)
// ---------------------------------------------------------------------------

// Pool of per-frame resources guarded by a ticket semaphore. `take()` blocks
// on the semaphore (only `current` frames in flight per instance), then pops
// the most recently released resource under the pool lock; `give_back()`
// returns the resource and releases one ticket. All five filters initialize
// `pool.semaphore.current` to `num_streams - 1` before first use, so the
// in-flight depth is num_streams regardless of how the device's queues are
// shared between instances.
template <typename T>
struct FramePool {
    ticket_semaphore semaphore;
    std::vector<T> items;
    std::mutex lock;

    void reserve(size_t n) {
        items.reserve(n);
    }

    void push(T && r) {
        std::lock_guard guard(lock);
        items.push_back(std::move(r));
    }

    T take() {
        semaphore.acquire();
        std::lock_guard guard(lock);
        T r = std::move(items.back());
        items.pop_back();
        return r;
    }

    void give_back(T && r) {
        {
            std::lock_guard guard(lock);
            items.push_back(std::move(r));
        }
        semaphore.release();
    }
};

// Destroy the per-frame fields every filter's resource shares: the command
// buffer (from its pool), the command pool, the fence, and the staging
// buffer with its memory. The filter itself unmaps mapped pointers *before*
// this call (while the memory is still alive) and destroys its
// filter-specific buffers (device-local temporaries, extra command buffers,
// timeline semaphores, query pools) on either side of it.
template <typename T>
void destroy_common(VkDevice dev, T & r) {
    if (r.cmd) {
        vkFreeCommandBuffers(dev, r.pool, 1, &r.cmd);
    }
    if (r.pool) {
        vkDestroyCommandPool(dev, r.pool, nullptr);
    }
    if (r.fence) {
        vkDestroyFence(dev, r.fence, nullptr);
    }
    if (r.staging_mem) {
        vkFreeMemory(dev, r.staging_mem, nullptr);
    }
    if (r.staging) {
        vkDestroyBuffer(dev, r.staging, nullptr);
    }
}

// Retire this instance's own GPU work without idling a shared device.
// VapourSynth destroys a filter instance only after every frame callback has
// returned, and each frame path waits for its own fence before returning, so
// this instance's submissions have already completed by teardown. What
// remains is Vulkan's external synchronization: drain every queue the
// instance submitted on, holding that queue's lock. A device-wide
// vkDeviceWaitIdle here would stall (and, without the queue locks, race)
// unrelated filters sharing the same device; the device-wide idle stays in
// release_device(), where the refcount proves no other filter exists.
template <typename T>
inline void retire_instance(const FramePool<T> & pool) {
    std::vector<std::pair<VkQueue, std::mutex *>> queues;
    for (const auto & r : pool.items) {
        if (!r.queue) {
            continue;
        }
        bool seen = false;
        for (const auto & q : queues) {
            if (q.first == r.queue) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            queues.emplace_back(r.queue, r.queue_lock);
        }
    }
    for (auto & q : queues) {
        std::lock_guard lock(*q.second);
        vkQueueWaitIdle(q.first);
    }
}

// Submit a single pre-recorded command buffer with an optional fence,
// serialized on the queue's lock. The fence is reset inside the lock; the
// caller waits on it separately, outside the lock, so other frames keep
// submitting while this one is in flight.
inline VkResult submit_with_fence([[maybe_unused]] VkDevice dev, VkQueue queue,
                                  std::mutex * qlock, VkCommandBuffer cb,
                                  VkFence fence) {
    std::lock_guard lock(*qlock);
    if (fence != VK_NULL_HANDLE) {
        if (VkResult r = vkResetFences(dev, 1, &fence); r != VK_SUCCESS) {
            return r;
        }
    }
    VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr
    };
    return vkQueueSubmit(queue, 1, &submit_info, fence);
}

// Timeline-semaphore submit: the command buffer waits (device-side) on
// `waits[i]` reaching `values[i]` at `stages[i]`; if `signal_sem` is given,
// it also signals `signal_value` on completion; the optional `fence` is
// reset inside the lock. All three wait vectors must have equal size. Waits
// are non-destructive, so any number of consumers may wait on one signal.
inline VkResult submit_timeline(
    [[maybe_unused]] VkDevice dev, VkQueue queue, std::mutex * qlock,
    VkCommandBuffer cb, const std::vector<VkSemaphore> & waits,
    const std::vector<uint64_t> & values,
    const std::vector<VkPipelineStageFlags> & stages,
    VkSemaphore signal_sem, uint64_t signal_value, VkFence fence) {
    std::lock_guard lock(*qlock);
    if (fence != VK_NULL_HANDLE) {
        if (VkResult r = vkResetFences(dev, 1, &fence); r != VK_SUCCESS) {
            return r;
        }
    }
    VkTimelineSemaphoreSubmitInfo timeline_info {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = static_cast<uint32_t>(values.size()),
        .pWaitSemaphoreValues = values.empty() ? nullptr : values.data(),
        .signalSemaphoreValueCount = signal_sem != VK_NULL_HANDLE ? 1u : 0u,
        .pSignalSemaphoreValues = signal_sem != VK_NULL_HANDLE ? &signal_value : nullptr
    };
    VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timeline_info,
        .waitSemaphoreCount = static_cast<uint32_t>(waits.size()),
        .pWaitSemaphores = waits.empty() ? nullptr : waits.data(),
        .pWaitDstStageMask = stages.empty() ? nullptr : stages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = signal_sem != VK_NULL_HANDLE ? 1u : 0u,
        .pSignalSemaphores = signal_sem != VK_NULL_HANDLE ? &signal_sem : nullptr
    };
    return vkQueueSubmit(queue, 1, &submit_info, fence);
}

// Env-gated trace/behavior flag. Each filter keeps its own env names
// (VSFEEL_DFTTEST_TRACE, BM3D_TRACE, NLMEANS_TRACE, BILATERAL_NOCPU, ...)
// so flipping one filter's debugging never affects another.
inline bool trace_on(const char * env) {
    return std::getenv(env) != nullptr;
}

// ---------------------------------------------------------------------------
// Filter registration
// ---------------------------------------------------------------------------

void vsfeel_register_bilateral(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_bm3dv2(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_gaussblur(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_dfttest(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_nlmeans(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_eedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_nnedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
