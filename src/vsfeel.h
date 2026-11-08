#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// The GPU API this plugin targets (VSAPI::getVulkanAPI) only exists from API
// 4.3 on. The build defines it for every TU; the guard keeps a TU that includes
// VapourSynth4.h through some other path from silently dropping to API 4.0.
#ifndef VS_USE_API_43
#define VS_USE_API_43
#endif
#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

using namespace std::string_literals;

// The Vulkan and VapourSynth structs are built with designated initializers
// that set only the members that matter; C++ zero-initializes the rest, so a
// short initializer list is a deliberate convention here rather than a bug.
// Without this the Vulkan headers alone produce ~190 false positives.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

// Order CPU stores before a submission that lets the GPU read them: the
// host-visible VRAM BAR is write-combining, so a dispatch can otherwise observe
// a half-written table. A full fence because it is once per table, and because
// the portable spelling is what lets this build on arm64 as well as x86-64.
inline void store_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// Mark a deinterlacer's output as progressive and drop the input's field tag.
// `doubled` halves a tagged duration when the filter doubled the frame rate;
// EEDI3H/EEDI3AA keep the frame count and pass false.
inline void vsfeel_mark_progressive(VSFrame * frame, bool doubled,
                                    const VSAPI * vsapi) {
    VSMap * props = vsapi->getFramePropertiesRW(frame);
    vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
    vsapi->mapDeleteKey(props, "_Field");
    if (!doubled) {
        return;
    }
    int err_num = 0;
    int err_den = 0;
    int64_t num = vsapi->mapGetInt(props, "_DurationNum", 0, &err_num);
    int64_t den = vsapi->mapGetInt(props, "_DurationDen", 0, &err_den);
    if (!err_num && !err_den) {
        vsh::muldivRational(&num, &den, 1, 2);
        vsapi->mapSetInt(props, "_DurationNum", num, maReplace);
        vsapi->mapSetInt(props, "_DurationDen", den, maReplace);
    }
}

#define checkVK(expr)                                                          \
    do {                                                                       \
        if (VkResult __result = (expr); __result != VK_SUCCESS) [[unlikely]] { \
            return set_error("'"s + #expr +                                    \
                             "' failed: " + vk_result_string(__result));       \
        }                                                                      \
    } while (0)

// Env flags. Every filter flag goes through one of these so the naming and the
// `getenv` parsing live in a single place; each filter keeps its own names
// (VSFEEL_DFTTEST_TRACE, VSFEEL_BM3D_TRACE, ...) so flipping one filter's
// debugging never affects another.
// A flag is on when it is set to anything other than an empty string, "0" or
// "false", so `VAR=0`/`VAR=false` disable it the way the debug/trace parsers
// read. Presence alone used to be the test, which made the documented "=0 to
// restore the old path" controls impossible to use and silently enabled
// debug/ablation paths.
inline bool env_flag(const char * env) {
    const char * v = std::getenv(env);
    return v && *v && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0;
}

// The inverted default: on when unset, off only for `VAR=0`/`VAR=false`.
inline bool env_flag_default_on(const char * env) {
    const char * v = std::getenv(env);
    return !v || !*v || env_flag(env);
}

inline int env_int(const char * env, int default_value) {
    const char * v = std::getenv(env);
    if (!v || !*v) {
        return default_value;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE || parsed < INT_MIN ||
        parsed > INT_MAX) {
        return default_value;
    }
    return static_cast<int>(parsed);
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
// VSFEEL_TRACE is the trace plus the device/capability banner (1 = first 50
// lines, 2 = every line), and the per-filter names turn on one filter's own
// diagnostics alone. Probes only observe (a timestamp query, a clock read) but
// they are not free, so never benchmark a level-2 run.

inline int vsfeel_debug_level() {
    static const int level = [] {
        const char * v = std::getenv("VSFEEL_DEBUG");
        if (!v || !*v || std::strcmp(v, "0") == 0 ||
            std::strcmp(v, "false") == 0) {
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
        if (!v || !*v || std::strcmp(v, "0") == 0 ||
            std::strcmp(v, "false") == 0) {
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

// A renamed per-filter switch also accepts its previous spelling for one
// release, so a bug report's existing command keeps producing diagnostics.
inline bool vsfeel_debug_flag_alias(const char * specific,
                                    const char * previous) {
    return vsfeel_debug_flag(specific) || env_flag(previous);
}

inline bool vsfeel_debug_probe_alias(const char * specific,
                                     const char * previous) {
    return vsfeel_debug_probe(specific) || env_flag(previous);
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
                                std::chrono::steady_clock::now() - trail.t0)
                                .count();
    ++trail.count;
}

const char * vk_result_string(VkResult result);

inline void vsfeel_trace_error(const char * filter, int frame,
                               const std::string & message, const GPUDevice *) {
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
                          std::chrono::steady_clock::now() - start)
                          .count();
    std::fprintf(stderr,
                 "[vsfeel-trace] %-9s %-9s #%-3llu %-7s t=+%.1fms: %s\n",
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
constexpr VkDeviceSize align32(VkDeviceSize v) {
    return (v + 31) & ~VkDeviceSize(31);
}

// Fetch a filter dependency frame and turn the core's null on error into the
// filter's normal error path. Callers still free frames they fetched earlier.
inline const VSFrame *
vsfeel_get_frame(const VSAPI * vsapi, int n, VSNode * node,
                 VSFrameContext * frameCtx, const char * filter,
                 const char * message, const GPUDevice * device) {
    const VSFrame * frame = vsapi->getFrameFilter(n, node, frameCtx);
    if (!frame) {
        vsfeel_trace_error(filter, n, message, device);
        vsapi->setFilterError((std::string(filter) + ": " + message).c_str(),
                              frameCtx);
    }
    return frame;
}

// Parse a "planes" argument into a bit mask; absent means every plane. The
// core's two plane-selection errors are identical for every filter, so they
// live here instead of being retyped (and reworded) per filter.
inline std::variant<uint32_t, std::string>
vsfeel_parse_planes(const VSMap * in, const VSAPI * vsapi, int num_planes) {
    const int count = vsapi->mapNumElements(in, "planes");
    if (count <= 0) {
        return (uint32_t { 1 } << num_planes) - 1;
    }
    uint32_t mask = 0;
    for (int i = 0; i < count; ++i) {
        int error = 0;
        const int plane = vsapi->mapGetIntSaturated(in, "planes", i, &error);
        if (error || plane < 0 || plane >= num_planes) {
            return "plane index out of range.";
        }
        if (mask & (uint32_t { 1 } << plane)) {
            return "plane specified twice.";
        }
        mask |= uint32_t { 1 } << plane;
    }
    return mask;
}

// Read every plane's sample stride from a scratch frame: a GPU frame carries
// the CPU allocator's pitch, and the kernels address planes through it, so it
// cannot be guessed from an alignment rule. Returns an empty string on success.
// The probe is one chroma block tall (one row for 4:4:4): the pitch depends on
// the width and its alignment, not the height, and a full frame is a whole
// plane of memory at large sizes.
inline std::string vsfeel_probe_strides(const VSAPI * vsapi, VSCore * core,
                                        const VSVideoFormat & fmt, int width,
                                        int elem_bytes, int * strides) {
    // A subsampled chroma plane needs at least one sample row, so 4:2:0 and
    // 4:4:0 probe at two luma rows rather than one.
    const int height = 1 << fmt.subSamplingH;
    VSFrame * probe = vsapi->newVideoFrame(&fmt, width, height, nullptr, core);
    if (probe == nullptr) {
        return "could not allocate a probe frame to read the plane stride.";
    }
    for (int p = 0; p < fmt.numPlanes; ++p) {
        strides[p] = static_cast<int>(vsapi->getStride(probe, p) / elem_bytes);
    }
    vsapi->freeFrame(probe);
    return {};
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

    VkDevice device {};

    VkPhysicalDeviceLimits limits {};
    uint32_t api_version {};
    // VkPhysicalDeviceDriverProperties::driverID: which compiler actually builds
    // the pipelines. A workaround whose cost is a codegen decision has to key
    // off this, not the PCI vendor, because one vendor ships several compilers
    // (AMD's RADV/ACO and its proprietary LLPC share a vendorID).
    uint32_t driver_id {};
    // VkPhysicalDeviceProperties::vendorID/deviceID (PCI IDs), for the tables
    // that key a device fact by part.
    uint32_t vendor_id {};
    uint32_t device_id {};
    uint32_t queue_family {};
    // VkQueueFamilyProperties::timestampValidBits for the core's compute queue
    // family. Zero means a vkCmdWriteTimestamp2 there is invalid usage, so the
    // GPU-timing probes must stay off.
    uint32_t timestamp_valid_bits {};

    uint32_t subgroup_size { 32 };
    uint32_t min_subgroup_size { 32 };
    uint32_t max_subgroup_size { 32 };
    // How many subgroups one compute workgroup may split into when its
    // pipeline requires a subgroup size.
    uint32_t max_compute_workgroup_subgroups { 1 };
    // VkPhysicalDeviceSubgroupSizeControlProperties::requiredSubgroupSizeStages:
    // requesting a size for a stage the mask excludes is invalid usage, and the
    // mask may omit COMPUTE even when size control itself is enabled.
    VkShaderStageFlags subgroup_size_stages {};
    // The core's baseline requires both, but nothing in this plugin may assume
    // it: requesting a subgroup size or full subgroups is invalid usage unless
    // the feature is actually enabled, so they are queried and read.
    bool subgroup_size_control { false };
    bool compute_full_subgroups { false };
    // VkSubgroupFeatureFlags: only COMPUTE/fragment BASIC is mandatory in
    // Vulkan 1.1, so every other intrinsic a kernel uses has to be checked.
    VkSubgroupFeatureFlags subgroup_ops {};
    // VkPhysicalDevicePushDescriptorProperties::maxPushDescriptors: the binding
    // count a push-descriptor set layout may use.
    uint32_t max_push_descriptors { 32 };
    // shaderBufferFloat32AtomicAdd specifically -- atomic exchange support
    // alone does not let a kernel use atomicAdd on a float buffer.
    bool feat_atomic_float32_add { false };
    // shaderFloat16: optional in every core version (see the core's feature
    // table), so a kernel that computes in half rather than only storing it
    // has to ask. Gates NNEDI3's packed half predictor.
    bool feat_shader_float16 { false };

    // The shaders' other extensions are deliberately unchecked, because the
    // core's device baseline guarantees them on the logical device (see
    // VSVulkan4.h: every REQUIRED entry there is mandatory for the Vulkan 1.4
    // device the core creates): GL_EXT_shader_16bit_storage needs
    // storageBuffer16BitAccess (Vulkan11 baseline), GL_EXT_shader_8bit_storage
    // and GL_EXT_shader_explicit_arithmetic_types_int8 need
    // storageBuffer8BitAccess + shaderInt8, GL_KHR_memory_scope_semantics needs
    // vulkanMemoryModel, and GL_EXT_buffer_reference needs bufferDeviceAddress
    // (all Vulkan12 baseline). Only the features the baseline does *not* require
    // -- subgroup size control/full subgroups, the subgroup operation bits, the
    // float-atomic extension, push-descriptor counts -- are queried above.

    // Persistent pipeline cache, shared by every instance on this device: the
    // core exposes no cache of its own, and compiling from SPIR-V is seconds
    // per variant on RADV.
    VkPipelineCache pipeline_cache {};
    std::mutex * pipeline_cache_lock {};
    std::string pipeline_cache_path;
    // Size of the blob the cache was seeded with: a process that compiled
    // nothing new must not rewrite ~150 MB at exit.
    size_t pipeline_cache_loaded_size {};

    // The core owns the device and destroys it with the core, so the cache has
    // to be flushed while the last reference here is still dropped -- an
    // atexit handler would run after the device is gone. The registry holds a
    // weak reference for exactly that reason.
    ~GPUDevice();

    // One device per VkDevice, reached through a shared_ptr: copying the handle
    // bundle would double-destroy it. Declaring these suppresses the implicit
    // default constructor, hence the explicit one.
    GPUDevice() = default;
    GPUDevice(const GPUDevice &) = delete;
    GPUDevice & operator=(const GPUDevice &) = delete;

    bool has_subgroup_ops(VkSubgroupFeatureFlags ops) const {
        return (subgroup_ops & ops) == ops;
    }

    // Whether a pipeline for a workgroup of `workgroup_invocations` invocations
    // may be created with requiredSubgroupSize = `size`. requiredSubgroupSize is
    // a power of two inside [minSubgroupSize, maxSubgroupSize] and invalid usage
    // otherwise, size control has to be enabled for the COMPUTE stage, and the
    // workgroup must not split into more subgroups than
    // maxComputeWorkgroupSubgroups allows.
    bool has_subgroup_size(uint32_t size,
                           uint32_t workgroup_invocations) const {
        if (!subgroup_size_control || size == 0) {
            return false;
        }
        if ((subgroup_size_stages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
            return false;
        }
        if ((size & (size - 1)) != 0) {
            return false;
        }
        if (size < min_subgroup_size || size > max_subgroup_size) {
            return false;
        }
        const uint64_t groups =
            (static_cast<uint64_t>(workgroup_invocations) + size - 1) / size;
        return groups <= max_compute_workgroup_subgroups;
    }

    // Resolves a kernel's fixed subgroup-size requirement: `required` receives
    // the value to pass as requiredSubgroupSize (0 leaves the driver's default,
    // which is already the wanted size), and the result says whether the device
    // can run the kernel at all. A kernel that addresses gl_SubgroupInvocationID
    // as a lane index needs the size itself, not merely something in range.
    bool resolve_subgroup_size(uint32_t size, uint32_t workgroup_invocations,
                               uint32_t & required) const {
        required = 0;
        if (has_subgroup_size(size, workgroup_invocations)) {
            required = size;
            return true;
        }
        return size == subgroup_size;
    }
};

// The core's device, brought up on first use and shared per VkDevice. Fails with
// the core's message when no usable device exists.
std::variant<std::shared_ptr<GPUDevice>, std::string>
get_gpu_device(VSCore * core, const VSAPI * vsapi);

// Core frame planes are bound with VK_WHOLE_SIZE, and the core does not cap
// their allocation; reject a plane whose buffer exceeds this device's range.
inline std::string vsfeel_plane_range_error(VkDeviceSize stride, int height,
                                            int elem_bytes,
                                            const GPUDevice & gpu,
                                            const char * what) {
    const VkDeviceSize bytes =
        stride * VkDeviceSize(height) * VkDeviceSize(elem_bytes);
    if (bytes <= gpu.limits.maxStorageBufferRange) {
        return {};
    }
    return std::string(what) + " needs " + std::to_string(bytes) +
           " bytes, more than this device's maxStorageBufferRange (" +
           std::to_string(gpu.limits.maxStorageBufferRange) +
           "); reduce the frame size.";
}

inline std::string vsfeel_plane_range_error(const VSVulkanPlaneInfo & plane,
                                            const GPUDevice & gpu,
                                            const char * what) {
    if (plane.bufferSize <= gpu.limits.maxStorageBufferRange) {
        return {};
    }
    return std::string(what) + " needs " + std::to_string(plane.bufferSize) +
           " bytes, more than this device's maxStorageBufferRange (" +
           std::to_string(gpu.limits.maxStorageBufferRange) +
           "); reduce the frame size.";
}

// A scratch buffer created for every frame: its size is fixed at creation, so
// reject it there rather than on the first frame, naming the buffer.
inline std::string vsfeel_frame_scratch_error(VkDeviceSize bytes,
                                              const GPUDevice & gpu,
                                              const char * what) {
    if (bytes <= gpu.limits.maxStorageBufferRange) {
        return {};
    }
    return std::string(what) + " needs " + std::to_string(bytes) +
           " bytes, more than this device's maxStorageBufferRange (" +
           std::to_string(gpu.limits.maxStorageBufferRange) +
           " bytes); lower the filter's size parameters.";
}

// GPU-timing probes: a timestamp write where timestampValidBits is 0 can hang
// the engine, so every probe is gated on this and says so once when skipped.
inline bool vsfeel_probe_timestamps(const GPUDevice & dev, const char * tag) {
    if (dev.timestamp_valid_bits != 0) {
        return true;
    }
    if (vsfeel_debug_enabled()) {
        fprintf(stderr,
                "[vsfeel] %s: queue family %u reports 0 timestamp bits; "
                "GPU timings disabled\n",
                tag, dev.queue_family);
    }
    return false;
}

// The core's VRAM eviction limit for this process (VSVulkanCoreInfo::limit), or
// 0 when it reports none. Every tuned allocation target in the plugin is capped
// by this: the optimums behind them were measured on a 24 GiB card, so a device
// (or a setMaxVRAMUse) with a smaller allowance must not have them planned
// anyway. VSFEEL_LIMIT_VRAM_BUDGET lowers it further, so a small device's path
// can be exercised on a large GPU.
inline VkDeviceSize vsfeel_vram_limit(const GPUDevice & dev, VSCore * core) {
    VSVulkanCoreInfo info {};
    char err[256] {};
    const bool known =
        dev.api->getVulkanCoreInfo(core, &info, err, sizeof(err)) == 0 &&
        info.limit > 0;
    VkDeviceSize limit = known ? static_cast<VkDeviceSize>(info.limit) : 0;
    // Parsed by hand rather than through env_int: this is a byte count, and any
    // interesting allowance overflows an int.
    if (const char * v = env_str("VSFEEL_LIMIT_VRAM_BUDGET")) {
        char * end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(v, &end, 10);
        if (end != v && *end == '\0' && errno != ERANGE && parsed > 0) {
            limit = known ? std::min(limit, static_cast<VkDeviceSize>(parsed))
                          : static_cast<VkDeviceSize>(parsed);
        }
    }
    return limit;
}

// A part's last-level cache size in bytes, or 0 when the PCI IDs are not in
// the table. Nothing in Vulkan (up to 1.4) reports a cache size, so sizing a
// cache-resident working set has to key off the part itself: the entries are
// the dies whose last level exceeds the 32 MiB a caller can assume, with AMD's
// Infinity Cache (reported as L3 by amdsmi: 96 MiB on a 7900 XTX, whose L2 is
// 6 MiB) and Nvidia's/Intel's L2. It is a table of published figures, not a
// measurement; extend it by adding rows (and keep it to parts with a vendor
// figure). This is a creation-time lookup, called once per filter instance.
inline VkDeviceSize vsfeel_gpu_llc_bytes(uint32_t vendor_id,
                                         uint32_t device_id) {
    struct Entry {
        uint32_t vendor;
        uint32_t device;
        uint32_t mib;
    };
    static constexpr std::array<Entry, 92> kTable { {
        { 0x1002, 0x73a1, 128 }, // Navi 21
        { 0x1002, 0x73a2, 128 }, // Navi 21
        { 0x1002, 0x73a3, 128 }, // Navi 21
        { 0x1002, 0x73a5, 128 }, // Navi 21
        { 0x1002, 0x73ab, 128 }, // Navi 21
        { 0x1002, 0x73ae, 128 }, // Navi 21
        { 0x1002, 0x73af, 128 }, // Navi 21
        { 0x1002, 0x73bf, 128 }, // Navi 21
        { 0x1002, 0x73df, 96 },  // Navi 22
        { 0x1002, 0x7448, 96 },  // Navi 31
        { 0x1002, 0x7449, 96 },  // Navi 31
        { 0x1002, 0x744a, 96 },  // Navi 31
        { 0x1002, 0x744b, 96 },  // Navi 31
        { 0x1002, 0x744c, 96 },  // Navi 31
        { 0x1002, 0x745e, 96 },  // Navi 31
        { 0x1002, 0x7460, 64 },  // Navi 32
        { 0x1002, 0x7461, 64 },  // Navi 32
        { 0x1002, 0x7470, 64 },  // Navi 32
        { 0x1002, 0x747e, 64 },  // Navi 32
        { 0x1002, 0x7550, 64 },  // Navi 48
        { 0x1002, 0x7551, 64 },  // Navi 48
        { 0x10de, 0x2681, 96 },  // AD102
        { 0x10de, 0x2684, 96 },  // AD102
        { 0x10de, 0x2685, 96 },  // AD102
        { 0x10de, 0x2689, 96 },  // AD102
        { 0x10de, 0x26af, 96 },  // AD102
        { 0x10de, 0x26b1, 96 },  // AD102
        { 0x10de, 0x26b2, 96 },  // AD102
        { 0x10de, 0x26b3, 96 },  // AD102
        { 0x10de, 0x26b5, 96 },  // AD102
        { 0x10de, 0x26b7, 96 },  // AD102
        { 0x10de, 0x26b8, 96 },  // AD102
        { 0x10de, 0x26b9, 96 },  // AD102
        { 0x10de, 0x26ba, 96 },  // AD102
        { 0x10de, 0x26bb, 96 },  // AD102
        { 0x10de, 0x26f5, 96 },  // AD102
        { 0x10de, 0x2702, 64 },  // AD103
        { 0x10de, 0x2703, 64 },  // AD103
        { 0x10de, 0x2704, 64 },  // AD103
        { 0x10de, 0x2705, 64 },  // AD103
        { 0x10de, 0x2709, 64 },  // AD103
        { 0x10de, 0x2717, 64 },  // AD103
        { 0x10de, 0x2730, 64 },  // AD103
        { 0x10de, 0x2757, 64 },  // AD103
        { 0x10de, 0x2770, 64 },  // AD103
        { 0x10de, 0x2782, 48 },  // AD104
        { 0x10de, 0x2783, 48 },  // AD104
        { 0x10de, 0x2785, 48 },  // AD104
        { 0x10de, 0x2786, 48 },  // AD104
        { 0x10de, 0x2788, 48 },  // AD104
        { 0x10de, 0x27a0, 48 },  // AD104
        { 0x10de, 0x27b0, 48 },  // AD104
        { 0x10de, 0x27b1, 48 },  // AD104
        { 0x10de, 0x27b2, 48 },  // AD104
        { 0x10de, 0x27b6, 48 },  // AD104
        { 0x10de, 0x27b7, 48 },  // AD104
        { 0x10de, 0x27b8, 48 },  // AD104
        { 0x10de, 0x27ba, 48 },  // AD104
        { 0x10de, 0x27bb, 48 },  // AD104
        { 0x10de, 0x27e0, 48 },  // AD104
        { 0x10de, 0x27fa, 48 },  // AD104
        { 0x10de, 0x27fb, 48 },  // AD104
        { 0x10de, 0x2b85, 96 },  // GB202
        { 0x10de, 0x2b87, 96 },  // GB202
        { 0x10de, 0x2b8c, 96 },  // GB202
        { 0x10de, 0x2bb1, 96 },  // GB202
        { 0x10de, 0x2bb2, 96 },  // GB202
        { 0x10de, 0x2bb3, 96 },  // GB202
        { 0x10de, 0x2bb4, 96 },  // GB202
        { 0x10de, 0x2bb5, 96 },  // GB202
        { 0x10de, 0x2bb9, 96 },  // GB202
        { 0x10de, 0x2bbc, 96 },  // GB202
        { 0x10de, 0x2c02, 64 },  // GB203
        { 0x10de, 0x2c05, 64 },  // GB203
        { 0x10de, 0x2c09, 64 },  // GB203
        { 0x10de, 0x2c18, 64 },  // GB203
        { 0x10de, 0x2c19, 64 },  // GB203
        { 0x10de, 0x2c31, 64 },  // GB203
        { 0x10de, 0x2c33, 64 },  // GB203
        { 0x10de, 0x2c34, 64 },  // GB203
        { 0x10de, 0x2c38, 64 },  // GB203
        { 0x10de, 0x2c39, 64 },  // GB203
        { 0x10de, 0x2c3a, 64 },  // GB203
        { 0x10de, 0x2c58, 64 },  // GB203
        { 0x10de, 0x2c59, 64 },  // GB203
        { 0x10de, 0x2c77, 64 },  // GB203
        { 0x10de, 0x2c79, 64 },  // GB203
        { 0x10de, 0x2f04, 48 },  // GB205
        { 0x10de, 0x2f06, 48 },  // GB205
        { 0x10de, 0x2f18, 48 },  // GB205
        { 0x10de, 0x2f38, 48 },  // GB205
        { 0x10de, 0x2f58, 48 },  // GB205
    } };
    for (const Entry & e : kTable) {
        if (e.vendor == vendor_id && e.device == device_id) {
            return static_cast<VkDeviceSize>(e.mib) << 20;
        }
    }
    return 0;
}

// A cache-resident budget for a working set the filter wants to keep in the
// last level: the part's known cache size when it is smaller than the tuned
// value, clamped to `floor`. The floor is the filter's measured limit -- below
// it the per-round overhead (stripes, rounds) grows faster than the cache
// saves -- and an unknown part gets it too, so an unknown device gets the
// conservative plan rather than the tuned card's. Callers cap the result with
// their own device bounds (VRAM, buffer range) afterwards.
inline VkDeviceSize vsfeel_llc_budget(const GPUDevice & gpu, VkDeviceSize tuned,
                                      VkDeviceSize floor) {
    const VkDeviceSize f = std::min(floor, tuned);
    const VkDeviceSize llc = vsfeel_gpu_llc_bytes(gpu.vendor_id, gpu.device_id);
    return std::clamp(llc != 0 ? std::min(tuned, llc) : VkDeviceSize(0), f,
                      tuned);
}

// Folds a 1D workgroup count into the device's per-dimension limits: a dispatch
// may not exceed maxComputeWorkGroupCount[i] (Vulkan guarantees only 65535, and
// this box reports exactly that in Y while X is 2^32-1), so X is capped and the
// remainder goes into Y. Kernels that opt in index their work as
// `ID.y * NumWorkGroups.x + ID.x`. Returns false, with `error` naming the
// kernel and both limits, when even the folded grid cannot fit.
inline bool vsfeel_fold_grid(const VkPhysicalDeviceLimits & limits,
                             uint64_t groups, const char * what,
                             uint32_t & gx_out, uint32_t & gy_out,
                             std::string & error) {
    const uint64_t max_x = limits.maxComputeWorkGroupCount[0];
    const uint64_t max_y = limits.maxComputeWorkGroupCount[1];
    const uint64_t gx = std::max<uint64_t>(1, std::min(groups, max_x));
    const uint64_t gy = (groups + gx - 1) / gx;
    if (gy > max_y) {
        error = std::string(what) + " needs " + std::to_string(groups) +
                " workgroups, more than this device can dispatch (" +
                std::to_string(max_x) + "x" + std::to_string(max_y) + ")";
        return false;
    }
    gx_out = static_cast<uint32_t>(gx);
    gy_out = static_cast<uint32_t>(gy);
    return true;
}

// A buffer from the core's pool. `handle` owns it; the rest is what a kernel or
// the host needs to use it.
struct GpuBuffer {
    VSGPUBuffer * handle {};
    VkBuffer buffer {};
    void * mapped {};
    VkMemoryPropertyFlags memory_flags {};
};

// Storage buffer from the pool; empty return means success, otherwise the
// failure text. `extra_usage` adds what a specific buffer needs beyond storage
// (transfer, device address, indirect).
inline std::string gpu_make_buffer(const GPUDevice & g, VSCore * core,
                                   VkDeviceSize bytes, GpuBuffer & out,
                                   VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred = 0,
                                   VkBufferUsageFlags extra_usage = 0) {
    // Every binding here is VK_WHOLE_SIZE, so the descriptor's range is the
    // buffer's size and maxStorageBufferRange (Vulkan's core minimum is 128 MiB)
    // caps it: BM3D's estimate cache is ~190 MiB at 1080p, DFTTest's block
    // buffer ~136 MiB, and a 2x2160p EEDI3 scratch is larger still. Refuse
    // clearly rather than fail inside the driver, or bind a truncated range
    // that the shader's addressing would walk off.
    if (bytes > g.limits.maxStorageBufferRange) {
        return "a " + std::to_string(bytes) +
               "-byte buffer exceeds this device's "
               "maxStorageBufferRange (" +
               std::to_string(g.limits.maxStorageBufferRange) +
               " bytes); lower the filter's size parameters or use one with a larger range"s;
    }
    char err[512] {};
    VSVulkanBufferInfo info {};
    const VkBufferUsageFlags usage =
        (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra_usage);
    out.handle = g.api->createGPUBuffer(core, bytes, usage, required, preferred,
                                        &info, err, sizeof(err));
    if (!out.handle) {
        return err;
    }
    out.buffer = info.buffer;
    out.mapped = info.mapped;
    out.memory_flags = info.memoryFlags;
    return {};
}

// Device local buffer that lives exactly as long as the recording it was made
// for: handed to the context, destroyed when that submission is known complete.
inline std::string gpu_frame_buffer(const GPUDevice & g, VSCore * core,
                                    VSGPUExecContext * ctx, VkDeviceSize bytes,
                                    GpuBuffer & out,
                                    VkBufferUsageFlags extra_usage = 0) {
    std::string err =
        gpu_make_buffer(g, core, bytes, out,
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

// Destroy a set of pipelines once each. A filter that shares pipelines across
// planes or geometries hands the whole set here; duplicates and nulls are
// ignored, so the plan never has to know which ones it already destroyed.
inline void vsfeel_destroy_pipelines(const GPUDevice & g,
                                     const VkPipeline * pipes, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const VkPipeline pipe = pipes[i];
        if (!pipe) {
            continue;
        }
        bool seen = false;
        for (size_t j = 0; j < i && !seen; ++j) {
            seen = pipes[j] == pipe;
        }
        if (!seen) {
            g.vk->vkDestroyPipeline(g.device, pipe, nullptr);
        }
    }
}

inline void vsfeel_destroy_pipelines(const GPUDevice & g,
                                     std::initializer_list<VkPipeline> pipes) {
    vsfeel_destroy_pipelines(g, pipes.begin(), pipes.size());
}

inline void vsfeel_destroy_pipelines(const GPUDevice & g,
                                     const std::vector<VkPipeline> & pipes) {
    vsfeel_destroy_pipelines(g, pipes.data(), pipes.size());
}

// Bind `count` storage buffers to set 0, bindings `first`..first+count-1
// through the push descriptor set the layout declares. Each dispatch rebinds
// its own view of the planes, which is why nothing here is allocated from a
// descriptor pool.
inline void gpu_push_buffers_at(const GPUDevice & g, VkCommandBuffer cmd,
                                VkPipelineLayout layout, uint32_t first,
                                const VkBuffer * buffers, uint32_t count) {
    // The arrays are fixed at GPU_MAX_BINDINGS. No caller exceeds it today;
    // fail loudly in debug builds rather than silently dropping a binding.
    if (first >= GPU_MAX_BINDINGS || count > GPU_MAX_BINDINGS - first) {
        assert(false && "push descriptor binding exceeds GPU_MAX_BINDINGS");
        count = first < GPU_MAX_BINDINGS ? GPU_MAX_BINDINGS - first : 0;
    }
    VkDescriptorBufferInfo infos[GPU_MAX_BINDINGS] {};
    VkWriteDescriptorSet writes[GPU_MAX_BINDINGS] {};
    for (uint32_t i = 0; i < count; ++i) {
        infos[i].buffer = buffers[i];
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = first + i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    g.vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                                 count, writes);
}

inline void gpu_push_buffers(const GPUDevice & g, VkCommandBuffer cmd,
                             VkPipelineLayout layout, const VkBuffer * buffers,
                             uint32_t count) {
    gpu_push_buffers_at(g, cmd, layout, 0, buffers, count);
}

// Bind one descriptor array (`count` consecutive storage buffers) to a single
// binding, the form a shader's `buffer Buf {...} b[count];` declares.
inline void gpu_push_buffer_array(const GPUDevice & g, VkCommandBuffer cmd,
                                  VkPipelineLayout layout, uint32_t binding,
                                  const VkBuffer * buffers, uint32_t count) {
    if (count > GPU_MAX_BINDINGS) {
        assert(false && "descriptor array exceeds GPU_MAX_BINDINGS");
        count = GPU_MAX_BINDINGS;
    }
    VkDescriptorBufferInfo infos[GPU_MAX_BINDINGS] {};
    for (uint32_t i = 0; i < count; ++i) {
        infos[i].buffer = buffers[i];
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet write {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.descriptorCount = count;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = infos;
    g.vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                                 1, &write);
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
// A push-descriptor layout's binding count is bounded by maxPushDescriptors
// (VUID-VkDescriptorSetLayoutCreateInfo-flags-00281), not by the per-stage
// pool limits -- which is what lets the 7..10 buffer bindings above sit on a
// device reporting the minimum of 4 there. maxPushDescriptors is at least 32,
// so a conformant device always fits, but the count is checked rather than
// assumed.
inline std::variant<VkDescriptorSetLayout, std::string>
gpu_push_set_layout(const GPUDevice & g, const uint32_t * counts,
                    uint32_t bindings) {
    if (bindings > GPU_MAX_BINDINGS) {
        return "a pipeline needs more than " +
               std::to_string(GPU_MAX_BINDINGS) +
               " buffer bindings, which this build does not support";
    }
    uint64_t descriptors = 0;
    for (uint32_t i = 0; i < bindings; ++i) {
        descriptors += counts[i];
    }
    if (descriptors > g.max_push_descriptors) {
        return "a pipeline needs " + std::to_string(descriptors) +
               " push descriptors, more than the " +
               std::to_string(
                   std::min(g.max_push_descriptors, GPU_MAX_BINDINGS)) +
               " this build supports";
    }
    VkDescriptorSetLayoutBinding b[GPU_MAX_BINDINGS] {};
    for (uint32_t i = 0; i < bindings; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = counts[i];
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
    info.bindingCount = bindings;
    info.pBindings = b;
    VkDescriptorSetLayout layout {};
    if (g.vk->vkCreateDescriptorSetLayout(g.device, &info, nullptr, &layout) !=
        VK_SUCCESS) {
        return "vkCreateDescriptorSetLayout failed"s;
    }
    return layout;
}

inline std::variant<VkDescriptorSetLayout, std::string>
gpu_push_set_layout(const GPUDevice & g, uint32_t bindings) {
    if (bindings > GPU_MAX_BINDINGS) {
        return "a pipeline needs more than " +
               std::to_string(GPU_MAX_BINDINGS) +
               " buffer bindings, which this build does not support";
    }
    uint32_t counts[GPU_MAX_BINDINGS] {};
    for (uint32_t i = 0; i < bindings; ++i) {
        counts[i] = 1;
    }
    return gpu_push_set_layout(g, counts, bindings);
}

inline std::variant<VkPipelineLayout, std::string>
gpu_pipeline_layout(const GPUDevice & g, VkDescriptorSetLayout set,
                    uint32_t push_bytes) {
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
    if (g.vk->vkCreatePipelineLayout(g.device, &info, nullptr, &layout) !=
        VK_SUCCESS) {
        return "vkCreatePipelineLayout failed"s;
    }
    return layout;
}

// A compute pipeline's workgroup shape and its static shared-memory footprint,
// as written in the shader it was compiled from (per-variant numbers are what
// `tools/shader_limits.py` prints off the built SPIR-V). Vulkan guarantees only
// 128 invocations, 128 per dimension and 16 KiB of workgroup memory, and several
// kernels here ask for 256 invocations or 18 KiB, so every pipeline has to be
// measured against the device before it is created rather than left to fail
// inside vkCreateComputePipelines.
struct GpuWorkgroup {
    uint32_t x { 1 };
    uint32_t y { 1 };
    uint32_t z { 1 };
    uint32_t shared_bytes {};

    uint32_t invocations() const {
        return x * y * z;
    }
};

// Compute pipeline from an embedded SPIR-V blob plus its specialization
// constants. maintenance5 is part of the core's baseline, so the module is
// chained straight into pipeline creation and never exists as an object.
// `entries` == nullptr means no specialization; `required_subgroup_size` 0
// leaves the driver's default width alone, and any non-zero request is checked
// against the device for the `workgroup` the kernel launches with. `tag` only
// names the pipeline in the VSFEEL_DEBUG banner.
inline std::variant<VkPipeline, std::string> gpu_create_pipeline(
    const GPUDevice & g, const uint32_t * code, size_t code_size,
    VkPipelineLayout layout, const VkSpecializationMapEntry * entries,
    const void * values, uint32_t entry_count, size_t values_size,
    const char * tag, uint32_t required_subgroup_size = 0,
    GpuWorkgroup workgroup = {}, bool full_subgroups = false) {

    if (entries == nullptr) {
        entry_count = 0;
        values = nullptr;
        values_size = 0;
    }
    if (vsfeel_debug_enabled()) {
        fprintf(
            stderr,
            "[vsfeel] pipeline %s subgroup=%u local=%ux%ux%u lds=%u spec=%u\n",
            tag, required_subgroup_size, workgroup.x, workgroup.y, workgroup.z,
            workgroup.shared_bytes, entry_count);
    }

    // Workgroup shape and shared memory against what the device reports. The
    // shader's own static LDS is what binds; the driver would reject the
    // pipeline anyway, so this is about saying which kernel and which number.
    const uint32_t invocations = workgroup.invocations();
    const VkPhysicalDeviceLimits & lim = g.limits;
    if (workgroup.x > lim.maxComputeWorkGroupSize[0] ||
        workgroup.y > lim.maxComputeWorkGroupSize[1] ||
        workgroup.z > lim.maxComputeWorkGroupSize[2] ||
        invocations > lim.maxComputeWorkGroupInvocations) {
        return std::string(tag) + " needs a " + std::to_string(workgroup.x) +
               "x" + std::to_string(workgroup.y) + "x" +
               std::to_string(workgroup.z) + " workgroup (" +
               std::to_string(invocations) +
               " invocations); this device allows at most " +
               std::to_string(lim.maxComputeWorkGroupInvocations) +
               " invocations (" +
               std::to_string(lim.maxComputeWorkGroupSize[0]) + "x" +
               std::to_string(lim.maxComputeWorkGroupSize[1]) + "x" +
               std::to_string(lim.maxComputeWorkGroupSize[2]) +
               " per dimension)";
    }
    if (workgroup.shared_bytes > lim.maxComputeSharedMemorySize) {
        return std::string(tag) + " needs " +
               std::to_string(workgroup.shared_bytes) +
               " bytes of workgroup memory; this device allows " +
               std::to_string(lim.maxComputeSharedMemorySize);
    }

    if (required_subgroup_size != 0 &&
        !g.has_subgroup_size(required_subgroup_size, invocations)) {
        return std::string(tag) + " requests subgroup size " +
               std::to_string(required_subgroup_size) + " for a workgroup of " +
               std::to_string(invocations) +
               " invocations, which is not a subgroup size this device supports";
    }
    // REQUIRE_FULL_SUBGROUPS is only valid with the computeFullSubgroups feature.
    if (full_subgroups && !g.compute_full_subgroups) {
        return std::string(tag) +
               " needs full subgroups, which this device does not support";
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
    stage.flags =
        full_subgroups
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
        // Timed so a cold-start report can tell compilation apart from a slow
        // frame: the first run compiles every kernel serially here.
        const auto t0 = std::chrono::steady_clock::now();
        result = g.vk->vkCreateComputePipelines(g.device, g.pipeline_cache, 1,
                                                &info, nullptr, &pipeline);
        if (vsfeel_debug_enabled()) {
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
            fprintf(stderr, "[vsfeel] pipeline %s created in %.1f ms\n", tag,
                    ms);
        }
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
void vsfeel_register_bm3d(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_box(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_gaussblur(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_dfttest(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_nlmeans(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_eedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_nnedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
void vsfeel_register_wnnm(const VSPLUGINAPI * vspapi, VSPlugin * plugin);
