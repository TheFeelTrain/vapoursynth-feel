#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <volk.h>

#include <VapourSynth4.h>

using namespace std::string_literals;

// The Vulkan and VapourSynth structs are built with designated initializers
// that set only the members that matter; C++ zero-initializes the rest, so a
// short initializer list is a deliberate convention here rather than a bug.
// Without this the Vulkan headers alone produce ~190 false positives.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

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

// Env flags. Every filter flag goes through one of these so the naming and the
// `getenv` parsing live in a single place; each filter keeps its own names
// (VSFEEL_DFTTEST_TRACE, VSFEEL_BM3D_TRACE, ...) so flipping one filter's
// debugging never affects another.
inline bool env_flag(const char * env) {
    return std::getenv(env) != nullptr;
}

inline int env_int(const char * env, int default_value) {
    const char * v = std::getenv(env);
    if (!v || !*v) {
        return default_value;
    }
    char * end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    return end != v ? static_cast<int>(parsed) : default_value;
}

inline const char * env_str(const char * env) {
    const char * v = std::getenv(env);
    return (v && *v) ? v : nullptr;
}

const char * vk_result_string(VkResult result);

// Queue sharing: cap how many of the device's compute queues the filter's
// streams are spread over. With one stream per queue each queue drains while
// its worker does post-fence CPU work before the next submit, leaving idle
// bubbles; sharing a queue across streams keeps a next command buffer queued.
// The default cap is a filter-specific starting point; `env_name` (e.g.
// "VSFEEL_GAUSS_QUEUES") overrides it as a durable tuning knob. The result is
// always clamped to [1, min(num_streams, queue_count)].
inline uint32_t resolve_queue_cap(int num_streams, uint32_t queue_count,
                                  const char * env_name, uint32_t default_cap) {
    const uint32_t streams = static_cast<uint32_t>(std::max(num_streams, 1));
    uint32_t cap = std::min({ streams, queue_count, default_cap });
    if (const int q = env_int(env_name, 0); q > 0) {
        cap = std::min({ streams, queue_count, static_cast<uint32_t>(q) });
    }
    return std::max(cap, 1u);
}

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
    uint32_t api_version {};
    uint32_t queue_family {};
    uint32_t queue_count {};
    // VK_EXT_external_memory_host: whether a host pointer can be imported as a
    // Vulkan buffer, and the alignment it must satisfy. The query command is
    // needed to learn which memory types accept a given pointer.
    bool host_import {};
    VkDeviceSize host_pointer_alignment {};
    PFN_vkGetMemoryHostPointerPropertiesEXT get_memory_host_pointer_properties {};
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
    mutable std::mutex pipeline_cache_lock;  // serializes cache access
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

// Vulkan requires external synchronization of host access to a VkPipelineCache,
// and VapourSynth creates filter nodes from several threads, so every
// vkCreateComputePipelines call goes through this raw lock-taking wrapper.
inline VkResult create_compute_pipeline(
    const VK_Device & dev, const VkComputePipelineCreateInfo & info,
    VkPipeline * pipeline) {
    std::lock_guard lock(dev.pipeline_cache_lock);
    return vkCreateComputePipelines(
        dev.device, dev.pipeline_cache, 1, &info, nullptr, pipeline);
}

// Shader module from an embedded SPIR-V blob. Shared by every filter so they
// all report the same failure text.
inline std::variant<VkShaderModule, std::string> create_shader_module(
    const VK_Device & dev, const uint32_t * code, size_t code_size) {

    VkShaderModuleCreateInfo module_info {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code_size,
        .pCode = code
    };

    VkShaderModule module;
    VkResult result = vkCreateShaderModule(dev.device, &module_info, nullptr, &module);
    if (result != VK_SUCCESS) {
        return "vkCreateShaderModule failed: "s + vk_result_string(result);
    }
    return module;
}

// Compute pipeline from a module plus its specialization constants.
// `required_subgroup_size` 0 leaves the driver's default subgroup alone; a
// non-zero value chains VkPipelineShaderStageRequiredSubgroupSizeCreateInfo,
// which needs VK_EXT_subgroup_size_control. `full_subgroups` sets
// REQUIRE_FULL_SUBGROUPS_BIT when the device exposes the feature.
// `entries`/`values`/`values_size` describe the spec-constant block exactly as
// VkSpecializationInfo would; entries == nullptr means no specialization.
// `tag` only names the shader in the VSFEEL_DBG banner.
inline std::variant<VkPipeline, std::string> create_compute_pipeline(
    const VK_Device & dev, VkShaderModule module, VkPipelineLayout layout,
    const VkSpecializationMapEntry * entries, const void * values,
    uint32_t entry_count, size_t values_size, const char * tag,
    uint32_t required_subgroup_size = 0, bool full_subgroups = false) {

    if (entries == nullptr) {
        entry_count = 0;
        values = nullptr;
        values_size = 0;
    }
    if (env_flag("VSFEEL_DBG")) {
        fprintf(stderr, "[vsfeel] pipeline %s subgroup=%u spec=%u\n",
            tag, required_subgroup_size, entry_count);
    }

    VkSpecializationInfo spec_info {
        .mapEntryCount = entry_count,
        .pMapEntries = entries,
        .dataSize = values_size,
        .pData = values
    };

    if (required_subgroup_size != 0 && !dev.subgroup_size_control) {
        return std::string(tag) + " requests subgroup size " +
            std::to_string(required_subgroup_size) +
            " but the device has no VK_EXT_subgroup_size_control";
    }
    // The feature struct must not be chained when no size was requested: its
    // absence is what lets the driver choose.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = required_subgroup_size
    };

    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = required_subgroup_size ? &subgroup_size_info : nullptr,
        .flags = (full_subgroups && dev.feat_compute_full_subgroups)
            ? VkPipelineShaderStageCreateFlags(
                  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
            : VkPipelineShaderStageCreateFlags(0),
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = &spec_info
    };

    VkComputePipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage_info,
        .layout = layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };

    VkPipeline pipeline;
    VkResult result = create_compute_pipeline(dev, pipeline_info, &pipeline);
    if (result != VK_SUCCESS) {
        return "vkCreateComputePipelines failed: "s + vk_result_string(result);
    }
    return pipeline;
}

// Round a size or offset up to a 32-byte boundary. The per-frame buffers pack
// their plane regions this way so each region is 32-byte aligned (what
// non-temporal access and descriptor offsets both want).
inline constexpr VkDeviceSize align32(VkDeviceSize v) {
    return (v + 31) & ~VkDeviceSize(31);
}

// The DFTTest/EEDI3/NNEDI3 shaders are compiled for SPIR-V 1.6, which a Vulkan
// 1.3 device is required to accept; below that, pipeline creation fails with an
// opaque driver error, so report the version instead.
inline std::optional<std::string> require_vulkan_1_3(
    const VK_Device & dev, const char * filter) {
    if (dev.api_version >= VK_API_VERSION_1_3) {
        return std::nullopt;
    }
    return std::string(filter) + " requires Vulkan 1.3 (device reports " +
        std::to_string(VK_API_VERSION_MAJOR(dev.api_version)) + "." +
        std::to_string(VK_API_VERSION_MINOR(dev.api_version)) + ")";
}

// Streaming copy: non-temporal stores bypass the CPU cache so the freshly
// written lines sit clean in DRAM; the GPU can then read them over PCIe
// without snoop/writeback stalls. Non-temporal stores are weakly ordered, so
// the caller must issue _mm_sfence() after the copy and before submitting any
// GPU work that reads `dst` (same for copy_plane_out with nt=true).
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
// write-combined VRAM BAR window); with `nt` the caller must _mm_sfence()
// before submitting the GPU work that reads the destination.
void copy_plane_out(void * dst, ptrdiff_t dst_pitch, const void * src,
                    ptrdiff_t src_pitch, size_t row_bytes, int height, bool nt);

// Row-wise counterpart of copy_stream_read with independent source and
// destination pitches (GPU plane pitch in, frame stride out).
void copy_plane_read(void * dst, ptrdiff_t dst_pitch, const void * src,
                     ptrdiff_t src_pitch, size_t row_bytes, int height);

// Vulkan requires mapped-memory flush/invalidate ranges to be multiples of
// minNonCoherentAtomSize, or to run to the end of the allocation. Callers work
// in exact plane/slice ranges, so round the offset down and the size up here;
// a range that would overrun `mem_size` (or one whose allocation size is
// unknown, `mem_size == 0`) becomes VK_WHOLE_SIZE instead.
inline VkMappedMemoryRange mapped_range(
    const VK_Device & dev, VkDeviceMemory memory, VkDeviceSize offset,
    VkDeviceSize size, VkDeviceSize mem_size = 0) {
    const VkDeviceSize atom =
        dev.limits.nonCoherentAtomSize ? dev.limits.nonCoherentAtomSize : 1;
    const VkDeviceSize start = offset - (offset % atom);
    VkDeviceSize range = VK_WHOLE_SIZE;
    if (mem_size > 0) {
        const VkDeviceSize end = offset + size;
        const VkDeviceSize aligned_end = end + ((atom - (end % atom)) % atom);
        if (aligned_end <= mem_size) {
            range = aligned_end - start;
        }
    }
    return VkMappedMemoryRange {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = memory,
        .offset = start,
        .size = range
    };
}

inline VkResult flush_range(const VK_Device & dev, VkDeviceMemory memory,
                            VkDeviceSize offset, VkDeviceSize size,
                            VkDeviceSize mem_size = 0) {
    if (size == 0) {
        return VK_SUCCESS;
    }
    VkMappedMemoryRange range = mapped_range(dev, memory, offset, size, mem_size);
    return vkFlushMappedMemoryRanges(dev.device, 1, &range);
}

inline VkResult invalidate_range(const VK_Device & dev, VkDeviceMemory memory,
                                 VkDeviceSize offset, VkDeviceSize size,
                                 VkDeviceSize mem_size = 0) {
    if (size == 0) {
        return VK_SUCCESS;
    }
    VkMappedMemoryRange range = mapped_range(dev, memory, offset, size, mem_size);
    return vkInvalidateMappedMemoryRanges(dev.device, 1, &range);
}

struct AllocatedMemory {
    VkDeviceMemory memory;
    uint32_t type_index;
};

std::variant<AllocatedMemory, std::string> allocate_memory(
    const VK_Device & dev, VkBuffer buffer, VkMemoryPropertyFlags required);

// Buffer handle with no memory bound yet. `size` is forced non-zero because
// Vulkan requires a positive size; a zero-byte request becomes 4 bytes so the
// handle is still valid to bind and destroy.
inline std::variant<VkBuffer, std::string> create_buffer(
    VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = std::max<VkDeviceSize>(size, 4),
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr
    };
    VkBuffer buffer;
    if (vkCreateBuffer(dev, &info, nullptr, &buffer) != VK_SUCCESS) {
        return "vkCreateBuffer failed"s;
    }
    return buffer;
}

// Allocate memory for `buffer` with the requested property flags and bind it,
// writing the VkDeviceMemory back through `mem`. allocate_memory() owns the
// fallback ladder; this only surfaces its error.
inline std::optional<std::string> bind_memory(
    const VK_Device & dev, VkBuffer buffer, VkDeviceMemory & mem,
    VkMemoryPropertyFlags required) {

    const auto result = allocate_memory(dev, buffer, required);
    if (std::holds_alternative<std::string>(result)) {
        return std::get<std::string>(result);
    }
    mem = std::get<AllocatedMemory>(result).memory;
    return std::nullopt;
}

// Device-local buffer with storage plus transfer-src/dst usage, the shape the
// per-frame scratch buffers share. `extra_usage` adds the one or two bits a
// specific buffer needs (indirect, ...). `what` only names the buffer in the
// failure text.
inline std::optional<std::string> make_device_buffer(
    const VK_Device & dev, VkBuffer & buf, VkDeviceMemory & mem,
    VkDeviceSize size, const char * what, VkBufferUsageFlags extra_usage = 0) {

    const auto created = create_buffer(dev.device, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage);
    if (std::holds_alternative<std::string>(created)) {
        return "vkCreateBuffer ("s + what + ") failed.";
    }
    buf = std::get<VkBuffer>(created);

    const auto bound = bind_memory(
        dev, buf, mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (bound) {
        // The buffer exists but has no memory: free it so a failed creation
        // loop does not leak handles.
        vkDestroyBuffer(dev.device, buf, nullptr);
        buf = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
        return bound;
    }
    return std::nullopt;
}

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

    // Push-then-fill variant of push() for creation loops: the resource is
    // owned by the pool from the moment it exists, so an early error return
    // leaves it to be torn down by the filter destructor instead of
    // abandoning its buffers, device memory, mapped windows, command pool and
    // fence. The caller must have reserved capacity up front and must not
    // touch the pool concurrently while filling the returned reference.
    T & emplace() {
        std::lock_guard guard(lock);
        return items.emplace_back();
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
