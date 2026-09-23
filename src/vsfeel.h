#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <volk.h>

// The GPU API this plugin targets (VSAPI::getVulkanAPI) only exists from API
// 4.3 on. The build defines it for every TU; the guard keeps a TU that includes
// VapourSynth4.h through some other path from silently dropping to API 4.0.
#ifndef VS_USE_API_43
#define VS_USE_API_43
#endif
#include <VapourSynth4.h>
#include <VSVulkan4.h>

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

// Env flags. Every filter flag goes through one of these so the naming and the
// `getenv` parsing live in a single place; each filter keeps its own names
// (VSFEEL_DFTTEST_TRACE, VSFEEL_BM3D_TRACE, ...) so flipping one filter's
// debugging never affects another.
// A flag is on when it is set to anything other than an empty string or "0",
// so `VAR=0` disables it the way every other switch reads. Presence alone used
// to be the test, which made the documented "=0 to restore the old path"
// controls impossible to use and silently enabled debug/ablation paths.
inline bool env_flag(const char * env) {
    const char * v = std::getenv(env);
    return v && *v && std::strcmp(v, "0") != 0;
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

// ---------------------------------------------------------------------------
// Debug switches (VSFEEL_DEBUG, VSFEEL_TRACE)
// ---------------------------------------------------------------------------
//
// One switch for the whole plugin: every error a filter reports is echoed to
// stderr with the filter that raised it, the output frame, and its position in
// the sequence. The order matters because a lost device makes every later call
// fail too -- the first line is the informative one and the rest are a
// cascade. Create-time errors print `create` in place of a frame number.
//
// **VSFEEL_DEBUG is the one to give a bug reporter**, and it takes a level:
// `=1` prints the one-shot diagnostics (device banner, capability and heap
// dump, creation banners, fallback notices, the full error trace), `=2` adds
// the instrumentation -- the per-filter frame-path traces (several lines per
// frame) and the measurement probes (host stage timings, GPU kernel
// timestamps), which a hang or a stutter needs as much as the trace does.
// VSFEEL_TRACE is the trace alone (1 = first 50 lines, 2 = every line), and the
// per-filter names turn on one filter's own diagnostics alone. Probes only
// observe (a timestamp query, a clock read) but they are not free, so never
// benchmark a level-2 run.

inline int vsfeel_debug_level() {
    static const int level = [] {
        // VSFEEL_DBG is the pre-rename spelling, kept working for one release.
        const char * v = std::getenv("VSFEEL_DEBUG");
        if (!v || !*v) {
            v = std::getenv("VSFEEL_DBG");
        }
        if (!v || !*v || std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0) {
            return 0;
        }
        return std::strcmp(v, "2") == 0 ? 2 : 1;
    }();
    return level;
}

inline bool vsfeel_debug_enabled() {
    return vsfeel_debug_level() > 0;
}

inline int vsfeel_trace_level() {
    static const int level = [] {
        if (vsfeel_debug_enabled()) {
            return 2;
        }
        const char * v = std::getenv("VSFEEL_TRACE");
        if (!v || !*v || std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0) {
            return 0;
        }
        return std::strcmp(v, "2") == 0 ? 2 : 1;
    }();
    return level;
}

inline bool vsfeel_trace_enabled() {
    return vsfeel_trace_level() > 0;
}

// The device banner and the capability lines: also on under VSFEEL_TRACE.
inline bool vsfeel_device_info_enabled() {
    static const bool on = vsfeel_debug_enabled() || vsfeel_trace_enabled();
    return on;
}

// A per-filter one-shot diagnostic (creation banner, fallback notice): the
// filter-specific name turns that one on, VSFEEL_DEBUG does at level 1.
inline bool vsfeel_debug_flag(const char * specific) {
    return vsfeel_debug_enabled() || env_flag(specific);
}

// A per-filter frame-path trace: several lines per frame, so it needs
// VSFEEL_DEBUG=2 or the filter-specific name.
inline bool vsfeel_debug_trace(const char * specific) {
    return vsfeel_debug_level() >= 2 || env_flag(specific);
}

// A per-filter measurement probe (host TIMING, GPU timestamps): it costs a
// little to collect and prints its own periodic summary, so like the frame
// traces it needs VSFEEL_DEBUG=2 or the filter-specific name. A bug report of
// a hang needs these numbers as much as the trace does.
inline bool vsfeel_debug_probe(const char * specific) {
    return vsfeel_debug_level() >= 2 || env_flag(specific);
}

struct GPUDevice;

// Declared here, defined below with the debug helpers: the frame trail is what
// turns one message into a diagnosis.
inline void vsfeel_trace_error(const char * filter, int frame,
                               const std::string & message,
                               const GPUDevice * device);

// Host-side trail of what the failing frame had done when it failed. The frame
// path is synchronous per worker thread, so the marks leading to an error are
// always the ones on the failing thread; thread_local is the whole story. Only
// the first error dumps them -- a lost device fails every later frame too.
struct VK_TraceTrail {
    static constexpr int CAP = 16;
    const char * stage[CAP] {};
    double ms[CAP] {};
    int count {};
    std::chrono::steady_clock::time_point t0 {};
};

inline VK_TraceTrail & vsfeel_trace_trail() {
    thread_local VK_TraceTrail trail;
    return trail;
}

// First call of a frame path. No-ops when tracing is off, like the marks below,
// so they can sit in the frame path unconditionally.
inline void vsfeel_trace_frame_begin() {
    if (vsfeel_trace_level() == 0) {
        return;
    }
    VK_TraceTrail & trail = vsfeel_trace_trail();
    trail.count = 0;
    trail.t0 = std::chrono::steady_clock::now();
}

inline void vsfeel_trace_mark(const char * stage) {
    if (vsfeel_trace_level() == 0) {
        return;
    }
    VK_TraceTrail & trail = vsfeel_trace_trail();
    if (trail.count >= VK_TraceTrail::CAP) {
        return;
    }
    trail.stage[trail.count] = stage;
    trail.ms[trail.count] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - trail.t0).count();
    ++trail.count;
}

const char * vk_result_string(VkResult result);

inline void vsfeel_trace_error(const char * filter, int frame,
                               const std::string & message,
                               const GPUDevice *) {
    const int level = vsfeel_trace_level();
    if (level == 0) {
        return;
    }
    static std::atomic<uint64_t> seen { 0 };
    static const auto start = std::chrono::steady_clock::now();
    const uint64_t seq = seen.fetch_add(1, std::memory_order_relaxed) + 1;
    if (level == 1 && seq == 51) {
        std::fprintf(stderr, "[vsfeel-trace] further errors suppressed "
                             "(VSFEEL_TRACE=2 prints all)\n");
    }
    if (level == 1 && seq > 50) {
        return;
    }

    char where[32];
    if (frame >= 0) {
        std::snprintf(where, sizeof(where), "frame=%d", frame);
    } else {
        std::snprintf(where, sizeof(where), "create");
    }
    // A few error strings are built with embedded newlines; keep one line per
    // error so the log can be pasted whole.
    std::string one_line = message;
    for (char & c : one_line) {
        if (c == '\n' || c == '\r') {
            c = ' ';
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "[vsfeel-trace] %-9s %-9s #%-3llu %-7s t=+%.1fms: %s\n",
        filter, where, static_cast<unsigned long long>(seq),
        seq == 1 ? "(first)" : "", ms, one_line.c_str());

    // How far the failing frame got: the error names the call that failed, the
    // trail says which stage of the frame that call belongs to. First error
    // only -- a lost device repeats the same shape on every later frame.
    if (seq == 1) {
        const VK_TraceTrail & trail = vsfeel_trace_trail();
        if (trail.count > 0) {
            std::fprintf(stderr, "[vsfeel-trace]   trail:");
            for (int i = 0; i < trail.count; ++i) {
                std::fprintf(stderr, "%s %s +%.1fms", i ? " |" : "",
                             trail.stage[i], trail.ms[i]);
            }
            std::fprintf(stderr, "\n");
        }
    }
}

// Round a size or offset up to a 32-byte boundary. The per-frame buffers pack
// their plane regions this way so each region is 32-byte aligned (what
// non-temporal access and descriptor offsets both want).
inline constexpr VkDeviceSize align32(VkDeviceSize v) {
    return (v + 31) & ~VkDeviceSize(31);
}

// ---------------------------------------------------------------------------
// R80 GPU API (VSVulkan4): the device the core owns
// ---------------------------------------------------------------------------
//
// From R80 on there is exactly one Vulkan device in the process and the core
// owns it. A filter no longer creates a device, a queue or a frame: it asks for
// the core's handles and dispatch table, records into an exec pool's command
// buffer, and moves GPU resident frames whose synchronization travels as
// producer pairs. Everything below is the thin shared layer over that API --
// device facts, pooled buffers, pipelines and the two push operations.

// How many storage buffers one dispatch may bind. Matches the push descriptor
// arrays below; a device never reports fewer than this for the filters here.
constexpr uint32_t GPU_MAX_BINDINGS = 32;

struct GPUDevice {
    const VSVULKANAPI * api {};
    const VSVulkanFunctions * vk {};
    VSVulkanCoreHandles handles {};

    VkDevice device {};
    VkQueue compute_queue {};

    VkPhysicalDeviceLimits limits {};
    uint32_t api_version {};
    uint32_t queue_family {};
    // VkQueueFamilyProperties::timestampValidBits for the core's compute queue
    // family. Zero means a vkCmdWriteTimestamp2 there is invalid usage, so the
    // GPU-timing probes must stay off.
    uint32_t timestamp_valid_bits {};

    uint32_t subgroup_size { 32 };
    uint32_t min_subgroup_size { 32 };
    uint32_t max_subgroup_size { 32 };
    // Both are required by the core's device baseline (Vulkan 1.3 features) --
    // there is nothing to check before using them.
    bool subgroup_size_control { true };
    bool subgroup_shuffle { true };
    bool feat_atomic_float32_add { false };

    // Persistent pipeline cache, shared by every instance on this device: the
    // core exposes no cache of its own, and compiling from SPIR-V is seconds
    // per variant on RADV.
    VkPipelineCache pipeline_cache {};
    std::mutex * pipeline_cache_lock {};
    std::string pipeline_cache_path;

    // The core owns the device and destroys it with the core, so the cache has
    // to be flushed while the last reference here is still dropped -- an
    // atexit handler would run after the device is gone. The registry holds a
    // weak reference for exactly that reason.
    ~GPUDevice();

    bool has_subgroup_size(uint32_t size) const {
        if (subgroup_size == size) {
            return true;
        }
        return subgroup_size_control && min_subgroup_size <= size &&
            size <= max_subgroup_size;
    }
};

// The core's device, brought up on first use and shared per VkDevice. Fails with
// the core's message when no usable device exists.
std::variant<std::shared_ptr<GPUDevice>, std::string> get_gpu_device(
    VSCore * core, const VSAPI * vsapi);

// GPU-timing probes: a timestamp write where timestampValidBits is 0 can hang
// the engine, so every probe is gated on this and says so once when skipped.
inline bool vsfeel_probe_timestamps(const GPUDevice & dev, const char * tag) {
    if (dev.timestamp_valid_bits != 0) {
        return true;
    }
    if (vsfeel_debug_enabled()) {
        fprintf(stderr, "[vsfeel] %s: queue family %u reports 0 timestamp bits; "
                        "GPU timings disabled\n", tag, dev.queue_family);
    }
    return false;
}

// A buffer from the core's pool. `handle` owns it; the rest is what a kernel or
// the host needs to use it.
struct GpuBuffer {
    VSGPUBuffer * handle {};
    VkBuffer buffer {};
    VkDeviceAddress address {};
    void * mapped {};
    VkDeviceSize size {};
    VkMemoryPropertyFlags memory_flags {};

    explicit operator bool() const { return handle != nullptr; }
};

// Storage buffer from the pool; empty return means success, otherwise the
// failure text. `extra_usage` adds what a specific buffer needs beyond storage
// (transfer, device address, indirect).
inline std::string gpu_make_buffer(const GPUDevice & g, VSCore * core,
                                   VkDeviceSize bytes, GpuBuffer & out,
                                   VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred = 0,
                                   VkBufferUsageFlags extra_usage = 0,
                                   VkBufferUsageFlags exclude = 0) {
    char err[512] {};
    VSVulkanBufferInfo info {};
    const VkBufferUsageFlags usage =
        ((VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra_usage) & ~exclude);
    out.handle = g.api->createGPUBuffer(core, bytes, usage, required, preferred,
        &info, err, sizeof(err));
    if (!out.handle) {
        return err;
    }
    out.buffer = info.buffer;
    out.address = info.address;
    out.mapped = info.mapped;
    out.size = info.size;
    out.memory_flags = info.memoryFlags;
    return {};
}

// Device local buffer that lives exactly as long as the recording it was made
// for: handed to the context, destroyed when that submission is known complete.
inline std::string gpu_frame_buffer(const GPUDevice & g, VSCore * core,
                                    VSGPUExecContext * ctx, VkDeviceSize bytes,
                                    GpuBuffer & out, VkBufferUsageFlags extra_usage = 0) {
    std::string err = gpu_make_buffer(g, core, bytes, out,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, extra_usage);
    if (!err.empty()) {
        return err;
    }
    g.api->gpuExecUsesBuffer(ctx, out.handle);
    return {};
}

inline void gpu_destroy_buffer(const GPUDevice & g, GpuBuffer & b) {
    if (b.handle) {
        g.api->destroyGPUBuffer(b.handle);
        b = {};
    }
}

// Bind `count` storage buffers to set 0, bindings 0..count-1 through the push
// descriptor set the layout declares. Each dispatch rebinds its own view of the
// planes, which is why nothing here is allocated from a descriptor pool.
inline void gpu_push_buffers(const GPUDevice & g, VkCommandBuffer cmd,
                             VkPipelineLayout layout, const VkBuffer * buffers,
                             uint32_t count) {
    VkDescriptorBufferInfo infos[GPU_MAX_BINDINGS] {};
    VkWriteDescriptorSet writes[GPU_MAX_BINDINGS] {};
    for (uint32_t i = 0; i < count; ++i) {
        infos[i].buffer = buffers[i];
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    g.vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
        count, writes);
}

inline void gpu_push_constants(const GPUDevice & g, VkCommandBuffer cmd,
                               VkPipelineLayout layout, const void * data,
                               uint32_t bytes) {
    VkPushConstantsInfo info {};
    info.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO;
    info.layout = layout;
    info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    info.size = bytes;
    info.pValues = data;
    g.vk->vkCmdPushConstants2(cmd, &info);
}

// Compute-to-compute barrier with both scopes. Every pass that follows one that
// wrote something it reads needs this; only genuinely disjoint passes may skip
// it (record them as one dispatch instead).
inline void gpu_barrier(const GPUDevice & g, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    g.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

// Descriptor set layout for `bindings` storage buffers, in push descriptor
// form: no pool, no allocation, nothing to free but the layout itself.
inline std::variant<VkDescriptorSetLayout, std::string> gpu_push_set_layout(
    const GPUDevice & g, uint32_t bindings) {
    VkDescriptorSetLayoutBinding b[GPU_MAX_BINDINGS] {};
    for (uint32_t i = 0; i < bindings; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
    info.bindingCount = bindings;
    info.pBindings = b;
    VkDescriptorSetLayout layout {};
    if (g.vk->vkCreateDescriptorSetLayout(g.device, &info, nullptr, &layout) != VK_SUCCESS) {
        return "vkCreateDescriptorSetLayout failed"s;
    }
    return layout;
}

inline std::variant<VkPipelineLayout, std::string> gpu_pipeline_layout(
    const GPUDevice & g, VkDescriptorSetLayout set, uint32_t push_bytes) {
    VkPushConstantRange range {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = push_bytes;
    VkPipelineLayoutCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = &set;
    if (push_bytes > 0) {
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
    }
    VkPipelineLayout layout {};
    if (g.vk->vkCreatePipelineLayout(g.device, &info, nullptr, &layout) != VK_SUCCESS) {
        return "vkCreatePipelineLayout failed"s;
    }
    return layout;
}

// Compute pipeline from an embedded SPIR-V blob plus its specialization
// constants. maintenance5 is part of the core's baseline, so the module is
// chained straight into pipeline creation and never exists as an object.
// `entries` == nullptr means no specialization; `required_subgroup_size` 0
// leaves the driver's default width alone. `tag` only names the pipeline in the
// VSFEEL_DEBUG banner.
inline std::variant<VkPipeline, std::string> gpu_create_pipeline(
    const GPUDevice & g, const uint32_t * code, size_t code_size,
    VkPipelineLayout layout, const VkSpecializationMapEntry * entries,
    const void * values, uint32_t entry_count, size_t values_size, const char * tag,
    uint32_t required_subgroup_size = 0, bool full_subgroups = false) {

    if (entries == nullptr) {
        entry_count = 0;
        values = nullptr;
        values_size = 0;
    }
    if (vsfeel_debug_enabled()) {
        fprintf(stderr, "[vsfeel] pipeline %s subgroup=%u spec=%u\n",
            tag, required_subgroup_size, entry_count);
    }
    if (required_subgroup_size != 0 && !g.has_subgroup_size(required_subgroup_size)) {
        return std::string(tag) + " requests subgroup size " +
            std::to_string(required_subgroup_size) +
            ", which this device cannot provide";
    }

    VkShaderModuleCreateInfo module_info {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = code_size;
    module_info.pCode = code;

    VkSpecializationInfo spec_info {};
    spec_info.mapEntryCount = entry_count;
    spec_info.pMapEntries = entries;
    spec_info.dataSize = values_size;
    spec_info.pData = values;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_info {};
    subgroup_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_info.requiredSubgroupSize = required_subgroup_size;

    VkPipelineShaderStageCreateInfo stage {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pNext = &module_info;
    stage.flags = full_subgroups
        ? VkPipelineShaderStageCreateFlags(
              VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
        : VkPipelineShaderStageCreateFlags(0);
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.pName = "main";
    if (entry_count > 0) {
        /* The specialization block is immutable once the stage info is built, so
           it is chained here rather than assigned after. */
        stage.pSpecializationInfo = &spec_info;
    }
    if (required_subgroup_size != 0) {
        subgroup_info.pNext = stage.pNext;
        stage.pNext = &subgroup_info;
    }

    VkComputePipelineCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage = stage;
    info.layout = layout;

    VkPipeline pipeline {};
    VkResult result;
    {
        // vkCreateComputePipelines requires the host to serialize access to the
        // shared cache.
        std::lock_guard lock(*g.pipeline_cache_lock);
        result = g.vk->vkCreateComputePipelines(
            g.device, g.pipeline_cache, 1, &info, nullptr, &pipeline);
    }
    if (result != VK_SUCCESS) {
        return "vkCreateComputePipelines failed: "s + vk_result_string(result);
    }
    return pipeline;
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
