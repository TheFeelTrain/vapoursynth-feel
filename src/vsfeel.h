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
// the per-filter frame-path traces, which print several lines per frame.
// VSFEEL_TRACE is the trace alone (1 = first 50 lines, 2 = every line), and the
// per-filter names turn on one filter alone. Performance probes (TIMING,
// GPUTRACE) stay explicit: they measure, they do not diagnose.

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

struct VK_Device;

// Declared here, defined after VK_Device: a VK_ERROR_DEVICE_LOST also asks the
// driver for its own fault report (VK_EXT_device_fault), the only source that
// says *why* the GPU died rather than that a fence wait returned an error.
inline void vsfeel_trace_error(const char * filter, int frame,
                               const std::string & message,
                               const VK_Device * device);

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
    uint32_t subgroup_size { 0 };    // the driver's default subgroup width
    uint32_t min_subgroup_size { 64 };
    uint32_t max_subgroup_size { 64 };
    bool subgroup_size_control { false };
    // Whether a shader can be guaranteed `size`-lane subgroups: natively, or
    // through size control when the driver's default width differs. Both are
    // sufficient; requiring the extension on a device that already reports the
    // wanted width would refuse a device that can run the kernel.
    bool has_subgroup_size(uint32_t size) const {
        if (subgroup_size == size) {
            return true;
        }
        return subgroup_size_control && min_subgroup_size <= size &&
            size <= max_subgroup_size;
    }
    // VK_SUBGROUP_FEATURE_SHUFFLE_BIT: a shader using subgroupShuffle needs the
    // GroupNonUniformShuffle capability. Only BASIC is required by the spec, so
    // this is queried rather than assumed. The subgroup size control query above
    // is a different question (it can force a *specific* width).
    bool subgroup_shuffle { false };
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
    // VK_EXT_device_fault: after VK_ERROR_DEVICE_LOST the driver can report what
    // the GPU faulted on (address, kind) and, on NVIDIA, a vendor crash dump.
    // Enabled at device creation when advertised; the query is only legal once
    // the device is lost.
    bool feat_device_fault { false };
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

inline const char * device_fault_address_type(VkDeviceFaultAddressTypeEXT type) {
    switch (type) {
        case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT:                        return "none";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:                return "read-invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:               return "write-invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:             return "execute-invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "ip-unknown";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "ip-invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:   return "ip-fault";
        default:                                                          return "unknown";
    }
}

// The driver's own post-mortem of a lost device (VK_EXT_device_fault). This is
// the only source that says *why* the GPU died -- a shader reading or writing an
// address it does not own looks like a bare fence-wait failure without it. The
// vendor binary goes to a file: on NVIDIA it decodes to the faulting
// instruction, which no amount of host-side logging can recover.
inline void vsfeel_trace_device_fault(const VK_Device & dev) {
    // volk only fills the entry point when the driver advertises it; a fault
    // report that cannot crash the process is worth the null check.
    if (dev.device == VK_NULL_HANDLE || !dev.feat_device_fault ||
        vkGetDeviceFaultInfoEXT == nullptr) {
        return;
    }
    // One-shot: the device stays lost, so every later error is a cascade and
    // would repeat the same block.
    static std::atomic<bool> reported { false };
    if (reported.exchange(true, std::memory_order_relaxed)) {
        return;
    }

    VkDeviceFaultCountsEXT counts {
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
        .pNext = nullptr
    };
    VkResult result = vkGetDeviceFaultInfoEXT(dev.device, &counts, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[vsfeel-trace] device fault query failed: %s\n",
                     vk_result_string(result));
        return;
    }

    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
    std::vector<uint8_t> binary(static_cast<size_t>(counts.vendorBinarySize));
    VkDeviceFaultInfoEXT info {
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
        .pNext = nullptr,
        .pAddressInfos = addresses.empty() ? nullptr : addresses.data(),
        .pVendorInfos = vendors.empty() ? nullptr : vendors.data(),
        .pVendorBinaryData = binary.empty() ? nullptr : binary.data()
    };
    result = vkGetDeviceFaultInfoEXT(dev.device, &counts, &info);
    if (result != VK_SUCCESS && !binary.empty()) {
        // Not every driver returns the vendor binary even when it advertises
        // the feature; retry without it rather than losing the description.
        binary.clear();
        VkDeviceFaultCountsEXT no_binary {
            .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
            .pNext = nullptr,
            .addressInfoCount = counts.addressInfoCount,
            .vendorInfoCount = counts.vendorInfoCount,
            .vendorBinarySize = 0
        };
        info.pVendorBinaryData = nullptr;
        result = vkGetDeviceFaultInfoEXT(dev.device, &no_binary, &info);
    }
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[vsfeel-trace] device fault query failed: %s\n",
                     vk_result_string(result));
        return;
    }

    std::fprintf(stderr, "[vsfeel-trace] device fault: %s\n", info.description);
    for (const auto & address : addresses) {
        std::fprintf(stderr, "[vsfeel-trace]   %s at 0x%llx (%llu address bits valid)\n",
            device_fault_address_type(address.addressType),
            static_cast<unsigned long long>(address.reportedAddress),
            static_cast<unsigned long long>(address.addressPrecision));
    }
    for (const auto & vendor : vendors) {
        std::fprintf(stderr, "[vsfeel-trace]   vendor code=0x%llx data=0x%llx: %s\n",
            static_cast<unsigned long long>(vendor.vendorFaultCode),
            static_cast<unsigned long long>(vendor.vendorFaultData),
            vendor.description);
    }
    if (!binary.empty()) {
        std::error_code ec;
        std::filesystem::path path = std::filesystem::temp_directory_path(ec);
        if (ec) {
            path = ".";
        }
        path /= "vsfeel-device-fault-" +
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
            ".bin";
        if (FILE * f = std::fopen(path.string().c_str(), "wb")) {
            std::fwrite(binary.data(), 1, binary.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "[vsfeel-trace]   vendor binary: %zu bytes -> %s\n",
                binary.size(), path.string().c_str());
        } else {
            std::fprintf(stderr, "[vsfeel-trace]   vendor binary: %zu bytes (not saved)\n",
                binary.size());
        }
    }
}

inline void vsfeel_trace_error(const char * filter, int frame,
                               const std::string & message,
                               const VK_Device * device) {
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

    if (device != nullptr && message.find("VK_ERROR_DEVICE_LOST") != std::string::npos) {
        vsfeel_trace_device_fault(*device);
    }
}

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
// `tag` only names the shader in the VSFEEL_DEBUG banner.
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
    if (vsfeel_debug_enabled()) {
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

// The DFTTest/EEDI3/NNEDI3/BM3D shaders are compiled for SPIR-V 1.6, which a
// Vulkan 1.3 device is required to accept; below that, pipeline creation fails
// with an opaque driver error, so report the version instead.
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

// Row-wise copy_stream_out over `height` rows of `row_bytes` bytes. When the
// destination pitch is a multiple of 32 every row shares one 32-byte
// alignment, so the store-alignment prologue is resolved once for the plane
// instead of once per row (the scalar head/tail bytes stay per row — NT
// stores must be aligned). Falls back to per-row copy_stream_out otherwise.
void copy_stream_rows(void * dst, ptrdiff_t dst_pitch, const void * src,
                      ptrdiff_t src_pitch, size_t row_bytes, int height);

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

// Can `bytes` actually be allocated from host-visible device-local memory, i.e.
// is the direct-upload path worth taking? The memory *type* is not the answer:
// a card without Resizable BAR still exposes DEVICE_LOCAL|HOST_VISIBLE, backed
// by the PCIe aperture rather than VRAM (an RX 580 under the Windows driver
// reports a 256 MiB aperture heap next to its 7936 MiB VRAM heap, and staging
// taken from it fails with VK_ERROR_OUT_OF_DEVICE_MEMORY while VRAM sits
// empty). Two gates: the backing heap must be a real slice of VRAM rather than
// an aperture, and the whole request must actually allocate. The probe is
// freed at once; allocate_memory never relaxes DEVICE_LOCAL|HOST_VISIBLE away,
// so a caller taking this path has to be sure it fits up front.
inline bool rebar_available(const VK_Device & dev, VkDeviceSize bytes) {
    constexpr VkMemoryPropertyFlags flags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkDeviceSize vram = 0;
    for (uint32_t i = 0; i < dev.mem_props.memoryHeapCount; ++i) {
        if (dev.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vram = std::max(vram, dev.mem_props.memoryHeaps[i].size);
        }
    }
    bytes = std::max<VkDeviceSize>(bytes, 4);
    for (uint32_t type = 0; type < dev.mem_props.memoryTypeCount; ++type) {
        if ((dev.mem_props.memoryTypes[type].propertyFlags & flags) != flags) {
            continue;
        }
        const VkDeviceSize heap =
            dev.mem_props.memoryHeaps[dev.mem_props.memoryTypes[type].heapIndex].size;
        // A quarter of VRAM is far above any PCIe aperture and far below a real
        // Resizable-BAR window, which is the whole VRAM.
        if (heap < bytes || heap < vram / 4) {
            continue;
        }
        VkMemoryAllocateInfo info {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = bytes,
            .memoryTypeIndex = type
        };
        VkDeviceMemory memory;
        if (vkAllocateMemory(dev.device, &info, nullptr, &memory) != VK_SUCCESS) {
            continue;
        }
        vkFreeMemory(dev.device, memory, nullptr);
        return true;
    }
    return false;
}

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
    if constexpr (requires { r.staging; r.staging_mem; }) {
        if (r.staging_mem) {
            vkFreeMemory(dev, r.staging_mem, nullptr);
        }
        if (r.staging) {
            vkDestroyBuffer(dev, r.staging, nullptr);
        }
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

// Submit command buffers with an optional fence, serialized on the queue's
// lock. The fence is reset inside the lock; the caller waits on it separately,
// outside the lock, so other frames keep submitting while this one is in
// flight. The multi-buffer form shares one lock/fence/signal, so a small
// per-frame re-recorded buffer (e.g. a download copy whose destination changes
// per frame) can ride along with a pre-recorded compute buffer.
inline VkResult submit_with_fence([[maybe_unused]] VkDevice dev, VkQueue queue,
                                  std::mutex * qlock,
                                  const VkCommandBuffer * cbs, uint32_t count,
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
        .commandBufferCount = count,
        .pCommandBuffers = cbs,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr
    };
    return vkQueueSubmit(queue, 1, &submit_info, fence);
}

inline VkResult submit_with_fence(VkDevice dev, VkQueue queue,
                                  std::mutex * qlock, VkCommandBuffer cb,
                                  VkFence fence) {
    return submit_with_fence(dev, queue, qlock, &cb, 1, fence);
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
