#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <vulkan/vulkan.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

namespace {

constexpr int MAX_RADIUS = 4;

struct Bm3dPlane {
    int width {};
    int height {};
    int stride {};
    VkDeviceSize pe {};             // plane extent in floats (h * stride)
    VkPipeline bm3d_pipeline {};
    VkPipeline agg_pipeline {};
    uint32_t bm3d_grid_x {};
    uint32_t bm3d_grid_y {};
    uint32_t agg_grid_x {};
    uint32_t agg_grid_y {};
};

struct Bm3dStream {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    float * map {};
    VkBuffer dst_buf {};
    VkDeviceMemory dst_mem {};
    float * dst_map {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkCommandBuffer cmd_agg {};   // aggregation phase: recorded after the waits
    VkFence fence {};
    VkSemaphore timeline {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    uint32_t staging_type_index {};
    VkQueryPool ts_query {};
    int stream_id {};        // index of this stream in the pool
    uint64_t seq {1};        // next monotonic timeline value this stream signals
    // cache reservations for the current frame (radius > 0)
    std::array<int, 9> win_slots {};      // res cache slot of each window frame
    std::array<int, 9> win_writers {};    // frame that wrote each window slot
    // identity of the slot's writer, captured under the cache lock when this
    // frame reserved the slot: waiting on (semaphore, value) is exact, while
    // recovering it from the frame number (a modulo-64 table) goes stale as
    // soon as the table entry is reused
    std::array<VkSemaphore, 9> win_writer_sem {};
    std::array<uint64_t, 9> win_writer_value {};
    std::array<bool, 9> win_recompute {}; // true where this frame must recompute the slot
    std::array<bool, 4 * MAX_RADIUS + 1> upload_new {};  // src frames this frame uploads
    std::array<int, 4 * MAX_RADIUS + 1> src_writers {};  // uploader of each window src slot
    std::array<VkSemaphore, 4 * MAX_RADIUS + 1> src_writer_sem {};
    std::array<uint64_t, 4 * MAX_RADIUS + 1> src_writer_value {};
    // unique token identifying this frame's cache reservation. A frame index
    // is not a unique holder identity: the scheduler can process the same
    // frame twice at once, and erasing holders by value would then drop both
    // entries at the first release, freeing a slot another reader still uses.
    uint64_t res_token {};
};


struct BM3DData {
    VSNode * node;
    VSNode * ref_node {};   // optional basic-estimate clip (final/Wiener pass)
    const VSVideoInfo * vi;

    int radius, num_streams;
    int device_id {};
    int tw;                          // 2 * radius + 1
    float sigma;                     // scaled luma sigma
    float sigma_u, sigma_v;
    int block_step, bm_range, ps_num, ps_range;
    bool process;
    bool chroma;
    bool final {};                   // true when a "ref" clip is given
    float extractor;

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule bm3d_module {};
    VkShaderModule agg_module {};
    std::array<Bm3dPlane, 3> planes {};
    int n_planes {};

    // shared device buffers (VRAM); the src is a ring of src_ring slots
    int src_ring {};             // cache slots for the source window (matches the kernel's SRC_RING)
    int res_cap {};              // cache slots for the per-frame estimate stacks
    VkDeviceSize src_size {};    // src_ring * pe elements (per plane, packed)
    VkBuffer src_buf {};
    VkDeviceMemory src_mem {};
    VkBuffer res_buf {};
    VkDeviceMemory res_mem {};
    VkDeviceSize dst_size {};        // pe elements (per stream)

    VkDeviceSize res_size_per_plane {};  // floats per plane in the res buffer
    int nframes {};

    // Per-frame-keyed caches of the res estimate stacks and the source
    // frames. Slots are reserved all-or-nothing for the duration of a frame
    // (shared holds for reads, exclusive for recompute/upload), so no two
    // in-flight frames ever touch the same slot; when the cache cannot hold
    // the working set (e.g. seeking), the acquire blocks like the reference's
    // fused-mode accumulator cache.
    std::vector<int> src_frame {};   // frame index whose data each src slot holds
    std::vector<int> src_writer {};  // frame that reserved each src slot for upload
    // Who last wrote each slot, stored with the reservation itself. The writer
    // is a frame number *and* the (timeline semaphore, value) it signals: the
    // frame number alone is not a stable key (a later frame can reuse it), and
    // a stale lookup makes a reader wait on an unrelated frame's unsubmitted
    // value — a cycle when that frame is itself waiting for the reader.
    std::vector<VkSemaphore> src_writer_sem {};
    std::vector<uint64_t> src_writer_value {};
    std::vector<std::vector<uint64_t>> src_holders {};  // reservation tokens
    std::vector<int> res_frame {};   // frame index whose stack each res slot holds
    std::vector<int> res_writer {};  // frame that computed each res slot's content
    std::vector<VkSemaphore> res_writer_sem {};
    std::vector<uint64_t> res_writer_value {};
    std::vector<std::vector<uint64_t>> res_holders {};  // reservation tokens
    uint64_t next_res_token {1};
    std::mutex cache_lock;
    std::condition_variable cache_cv;
    FramePool<Bm3dStream> pool;

    // Env-gated host-path probe (VSFEEL_BM3D_TIMING=1). The stage split is the
    // prerequisite for any transfer-path change: kernel time alone says nothing
    // about whether the frame is GPU- or host-bound. Reset the clock after every
    // blocking acquire so a wait never leaks into the next stage.
    bool host_timing { false };
    std::atomic<uint64_t> ht_take_ns {}, ht_acquire_ns {}, ht_upload_ns {},
        ht_record_ns {}, ht_srcwait_ns {}, ht_agg_ns {}, ht_fence_ns {},
        ht_down_ns {}, ht_total_ns {}, ht_n {};

    ~BM3DData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[bm3d-timing] frames=%.0f per-frame us: take=%7.1f acquire=%7.1f "
                "upload=%7.1f record=%7.1f srcwait=%7.1f agg=%7.1f fence=%7.1f "
                "download=%7.1f total=%7.1f\n",
                n, ht_take_ns.load() / 1000.0 / n, ht_acquire_ns.load() / 1000.0 / n,
                ht_upload_ns.load() / 1000.0 / n, ht_record_ns.load() / 1000.0 / n,
                ht_srcwait_ns.load() / 1000.0 / n, ht_agg_ns.load() / 1000.0 / n,
                ht_fence_ns.load() / 1000.0 / n, ht_down_ns.load() / 1000.0 / n,
                ht_total_ns.load() / 1000.0 / n);
        }
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        // retire this instance's own submissions (per queue) instead of
        // idling the whole device, which other filters may be sharing
        retire_instance(pool);
        for (auto & s : pool.items) {
            if (s.map) vkUnmapMemory(dev, s.staging_mem);
            if (s.dst_map) vkUnmapMemory(dev, s.dst_mem);
            if (s.ts_query) vkDestroyQueryPool(dev, s.ts_query, nullptr);
            if (s.dst_mem) vkFreeMemory(dev, s.dst_mem, nullptr);
            if (s.dst_buf) vkDestroyBuffer(dev, s.dst_buf, nullptr);
            if (s.cmd_agg) vkFreeCommandBuffers(dev, s.pool, 1, &s.cmd_agg);
            if (s.timeline) vkDestroySemaphore(dev, s.timeline, nullptr);
            destroy_common(dev, s);
        }
        if (res_mem) vkFreeMemory(dev, res_mem, nullptr);
        if (res_buf) vkDestroyBuffer(dev, res_buf, nullptr);
        if (src_mem) vkFreeMemory(dev, src_mem, nullptr);
        if (src_buf) vkDestroyBuffer(dev, src_buf, nullptr);
        if (desc_pool) vkDestroyDescriptorPool(dev, desc_pool, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        for (auto & p : planes) {
            if (p.bm3d_pipeline && p.bm3d_pipeline != p.agg_pipeline) {
                vkDestroyPipeline(dev, p.bm3d_pipeline, nullptr);
            }
            if (p.agg_pipeline) {
                vkDestroyPipeline(dev, p.agg_pipeline, nullptr);
            }
        }
        if (agg_module) vkDestroyShaderModule(dev, agg_module, nullptr);
        if (bm3d_module) vkDestroyShaderModule(dev, bm3d_module, nullptr);
        release_device(device);
    }
};

struct VkPhysicalDeviceFeaturesCompat {
    VkBool32 robustBufferAccess;
    VkBool32 fullDrawIndexUint32;
    VkBool32 imageCubeArray;
    VkBool32 independentBlend;
    VkBool32 geometryShader;
    VkBool32 tessellationShader;
    VkBool32 sampleRateShading;
    VkBool32 dualSrcBlend;
    VkBool32 logicOp;
    VkBool32 multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance;
    VkBool32 depthClamp;
    VkBool32 depthBiasClamp;
    VkBool32 fillModeNonSolid;
    VkBool32 depthBounds;
    VkBool32 wideLines;
    VkBool32 largePoints;
    VkBool32 alphaToOne;
    VkBool32 multiViewport;
    VkBool32 samplerAnisotropy;
    VkBool32 textureCompressionETC2;
    VkBool32 textureCompressionASTC_LDR;
    VkBool32 textureCompressionBC;
    VkBool32 occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery;
    VkBool32 vertexPipelineStoresAndAtomics;
    VkBool32 fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize;
    VkBool32 shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats;
    VkBool32 shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat;
    VkBool32 shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing;
    VkBool32 shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing;
    VkBool32 shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance;
    VkBool32 shaderCullDistance;
    VkBool32 shaderFloat64;
    VkBool32 shaderInt64;
    VkBool32 shaderInt16;
    VkBool32 shaderResourceResidency;
    VkBool32 shaderResourceMinLod;
    VkBool32 sparseBinding;
    VkBool32 sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D;
    VkBool32 sparseResidencyImage3D;
    VkBool32 sparseResidency2Samples;
    VkBool32 sparseResidency4Samples;
    VkBool32 sparseResidency8Samples;
    VkBool32 sparseResidency16Samples;
    VkBool32 sparseResidencyAliased;
    VkBool32 variableMultisampleRate;
    VkBool32 inheritedQueries;
};

// The device shared with the bilateral plugin may not enable fp64; we enable
// it here via the device features if the driver supports it.
std::optional<std::string> enable_float64(const std::shared_ptr<VK_Device> & dev) {
    if (!dev->feat_float64) {
        return "shaderFloat64 is not supported by this device";
    }
    return std::nullopt;
}

} // namespace

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

static std::variant<VkPipeline, std::string> create_bm3d_pipeline(
    const VK_Device & dev, const Bm3dPlane & plane,
    const BM3DData & d, VkShaderModule module, VkPipelineLayout layout) {

    const float sigma_y = d.sigma;
    struct Spec {
        int32_t width, height, stride;
        float sigma_y;
        int32_t block_step, bm_range, radius, ps_num, ps_range;
        float extractor;
        int32_t nosearch, noestimate, src_ring, final;
    } spec {
        plane.width, plane.height, plane.stride, sigma_y,
        d.block_step, d.bm_range, d.radius, d.ps_num, d.ps_range, d.extractor,
        std::getenv("BM3D_NOSEARCH") ? 1 : 0,
        std::getenv("BM3D_NOESTIMATE") ? 1 : 0,
        d.src_ring,
        d.final ? 1 : 0
    };
    const std::array<VkSpecializationMapEntry, 14> entries {{
        { 0,  0, sizeof(int32_t) },
        { 1,  4, sizeof(int32_t) },
        { 2,  8, sizeof(int32_t) },
        { 3, 12, sizeof(float) },
        { 4, 16, sizeof(int32_t) },
        { 5, 20, sizeof(int32_t) },
        { 6, 24, sizeof(int32_t) },
        { 7, 28, sizeof(int32_t) },
        { 8, 32, sizeof(int32_t) },
        { 9, 36, sizeof(float) },
        { 10, 40, sizeof(int32_t) },
        { 11, 44, sizeof(int32_t) },
        { 12, 48, sizeof(int32_t) },
        { 13, 52, sizeof(int32_t) },
    }};
    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(entries.size()),
        .pMapEntries = entries.data(),
        .dataSize = sizeof(spec),
        .pData = &spec
    };
    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
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
    VkResult result = vkCreateComputePipelines(
        dev.device, dev.pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return "vkCreateComputePipelines failed: "s + vk_result_string(result);
    }
    return pipeline;
}

static std::variant<VkPipeline, std::string> create_agg_pipeline(
    const VK_Device & dev, const Bm3dPlane & plane,
    const BM3DData & d, VkShaderModule module, VkPipelineLayout layout) {

    struct Spec {
        int32_t width, height, stride, tw;
    } spec { plane.width, plane.height, plane.stride, d.tw };
    const std::array<VkSpecializationMapEntry, 4> entries {{
        { 0,  0, sizeof(int32_t) },
        { 1,  4, sizeof(int32_t) },
        { 2,  8, sizeof(int32_t) },
        { 3, 12, sizeof(int32_t) },
    }};
    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(entries.size()),
        .pMapEntries = entries.data(),
        .dataSize = sizeof(spec),
        .pData = &spec
    };
    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
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
    VkResult result = vkCreateComputePipelines(
        dev.device, dev.pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return "vkCreateComputePipelines failed: "s + vk_result_string(result);
    }
    return pipeline;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static int agg_z(int i, int n, int nframes, int radius) {
    return std::min(std::max(2 * radius - i, n - nframes + 1 + radius), n + radius);
}

// Reserve the cache slots this frame needs, all-or-nothing: the res slots of
// the temporal window (recomputing the missing stacks) and the src slots of
// the source window (re-uploading the missing frames). Slots held by other
// in-flight frames block until they complete, so no two frames ever touch the
// same slot concurrently; the blocking only happens when the working set
// exceeds the cache (seeks, very out-of-order arrivals) and never holds the
// lock or any stream while waiting.
static void acquire_cache(BM3DData * d, Bm3dStream & stream, int n, uint64_t seq) {
    if (d->radius == 0) {
        // per-frame slots: keyed by the stream so concurrent out-of-order
        // frames never share a slot (a stream processes one frame at a time)
        stream.win_slots.fill(-1);
        stream.win_writers.fill(-1);
        stream.win_writer_sem.fill(VK_NULL_HANDLE);
        stream.win_writer_value.fill(0);
        stream.win_recompute.fill(false);
        stream.upload_new.fill(false);
        stream.win_slots[0] = stream.stream_id;
        stream.win_writers[0] = n;
        stream.win_recompute[0] = true;
        stream.upload_new[0] = true;
        return;
    }
    const int r = d->radius;
    const int nf = d->nframes;
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    std::unique_lock lock(d->cache_lock);
    for (;;) {
        bool ok = true;
        stream.win_slots.fill(-1);
        stream.win_writers.fill(-1);
        stream.win_writer_sem.fill(VK_NULL_HANDLE);
        stream.win_writer_value.fill(0);
        stream.win_recompute.fill(false);
        stream.upload_new.fill(false);
        stream.src_writers.fill(-1);
        stream.src_writer_sem.fill(VK_NULL_HANDLE);
        stream.src_writer_value.fill(0);
        // Phase 1: check-only, with no side effects. The failed passes must
        // not leave half-applied reservations behind, or a retry would treat
        // the abandoned slots as cached and never recompute them.
        for (int i = 0; i < d->tw; ++i) {
            const int m = std::clamp(n - r + i, 0, nf - 1);
            const int slot = m % d->res_cap;
            stream.win_slots[i] = slot;
            stream.win_writers[i] = d->res_writer[slot];
            stream.win_writer_sem[i] = d->res_writer_sem[slot];
            stream.win_writer_value[i] = d->res_writer_value[slot];
            if (d->res_frame[slot] != m && !d->res_holders[slot].empty()) {
                ok = false;   // slot in use by an in-flight frame
                break;
            }
        }
        if (ok) {
            for (int f = lo; f <= hi; ++f) {
                const int slot = f % d->src_ring;
                stream.src_writers[f - lo] = d->src_writer[slot];
                stream.src_writer_sem[f - lo] = d->src_writer_sem[slot];
                stream.src_writer_value[f - lo] = d->src_writer_value[slot];
                if (d->src_frame[slot] != f && !d->src_holders[slot].empty()) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            d->cache_cv.wait(lock);
            continue;
        }
        // Phase 2: apply the reservations (the lock is held, so the phase-1
        // checks are still valid).
        stream.res_token = d->next_res_token++;
        for (int i = 0; i < d->tw; ++i) {
            const int m = std::clamp(n - r + i, 0, nf - 1);
            const int slot = stream.win_slots[i];
            if (d->res_frame[slot] != m) {
                d->res_frame[slot] = m;
                d->res_writer[slot] = n;
                d->res_writer_sem[slot] = stream.timeline;
                d->res_writer_value[slot] = seq;
                stream.win_recompute[i] = true;
                // this frame overwrites the slot itself, so the previous
                // writer's contents (and its dependency) no longer matter
                stream.win_writers[i] = -1;
                stream.win_writer_sem[i] = VK_NULL_HANDLE;
                stream.win_writer_value[i] = 0;
            }
            d->res_holders[slot].push_back(stream.res_token);
        }
        for (int f = lo; f <= hi; ++f) {
            const int slot = f % d->src_ring;
            if (d->src_frame[slot] != f) {
                d->src_frame[slot] = f;
                d->src_writer[slot] = n;
                d->src_writer_sem[slot] = stream.timeline;
                d->src_writer_value[slot] = seq;
                stream.upload_new[f - lo] = true;
            }
            d->src_holders[slot].push_back(stream.res_token);
        }
        return;
    }
}

// Release the cache reservations after this frame's aggregation has been
// submitted: the queue order guarantees the later recomputes run after it.
static void release_cache(BM3DData * d, Bm3dStream & stream, int n) {
    if (d->radius == 0) {
        return;
    }
    std::lock_guard lock(d->cache_lock);
    for (int i = 0; i < d->tw; ++i) {
        auto & h = d->res_holders[stream.win_slots[i]];
        h.erase(std::remove(h.begin(), h.end(), stream.res_token), h.end());
    }
    const int r = d->radius;
    const int nf = d->nframes;
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    for (int f = lo; f <= hi; ++f) {
        auto & h = d->src_holders[f % d->src_ring];
        h.erase(std::remove(h.begin(), h.end(), stream.res_token), h.end());
    }
    d->cache_cv.notify_all();
}

// Record the estimation phase of the command buffer (staging copies, the
// zero-fills and the search/estimate dispatches). It has no dependency on the
// other in-flight frames: the src ring is sized for the union of all their
// windows, and each frame only writes its own res slot. It is submitted before
// the cross-frame wait so the GPU is busy with this heavy work while the host
// blocks on the previous frames' timelines.
static int record_bm3d_kernels(BM3DData * d, Bm3dStream & stream, int n,
                     const std::array<bool, 4 * MAX_RADIUS + 1> & uploaded) {
    VkDevice dev = d->device->device;
    VkCommandBuffer cmd = stream.cmd;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    const int nf = d->nframes;
    const int r = d->radius;
    const bool gputrace = std::getenv("BM3D_GPUTRACE") != nullptr;

    // copy the frames uploaded by the host (the union of all windows that this
    // record's dispatches may need, clamped to [n-2r, n+2r]) into the src ring
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    const int clips = d->final ? 2 : 1;
    for (int f = lo; f <= hi; ++f) {
        if (!uploaded[f - lo]) {
            continue;
        }
        const int src_slot = (r == 0) ? stream.stream_id : (f % d->src_ring);
        const VkDeviceSize slot_staging = static_cast<VkDeviceSize>(f - lo) * clips * d->planes[0].pe;
        const VkDeviceSize slot_device = static_cast<VkDeviceSize>(src_slot) * clips * d->planes[0].pe;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            const VkDeviceSize pe = p.pe;
            const VkDeviceSize plane_off = static_cast<VkDeviceSize>(plane) * d->src_size;
            // source clip: second half of the slot in final mode
            {
                VkBufferCopy region {
                    .srcOffset = (slot_staging + static_cast<VkDeviceSize>(clips - 1) * pe + plane_off) * 4,
                    .dstOffset = (slot_device + static_cast<VkDeviceSize>(clips - 1) * pe + plane_off) * 4,
                    .size = pe * 4
                };
                vkCmdCopyBuffer(cmd, stream.staging, d->src_buf, 1, &region);
            }
            // ref clip (final mode only): first half of the slot
            if (d->final) {
                VkBufferCopy region {
                    .srcOffset = (slot_staging + plane_off) * 4,
                    .dstOffset = (slot_device + plane_off) * 4,
                    .size = pe * 4
                };
                vkCmdCopyBuffer(cmd, stream.staging, d->src_buf, 1, &region);
            }
        }
    }

    // the estimation dispatches read the freshly copied source/ref frames, so
    // make the transfer writes visible to the compute stage before launching
    // them (and order the res zero-fill ahead of the atomic accumulation)
    {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
    }

    // compute the missing result slots for the aggregation window
    // (the frame n+r at the steady state; the boundary frames too)
    if (gputrace) {
        // a query must be reset before first use and before each reuse; doing
        // it inside the command buffer keeps the reset ordered with the stamps
        // (and with the aggregation command buffer submitted after this one)
        vkCmdResetQueryPool(cmd, stream.ts_query, 0, 4);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, stream.ts_query, 0);
    }
    int n_dispatches = 0;
    for (int i = 0; i < d->tw; ++i) {
        if (!stream.win_recompute[i]) {
            continue;
        }
        const int m_i = std::clamp(n - r + i, 0, nf - 1);
        const int slot = stream.win_slots[i];
        n_dispatches++;
        if (std::getenv("BM3D_DUMP")) fprintf(stderr, "[d] n=%d computes slot %d for frame %d\n", n, slot, m_i);
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            const VkDeviceSize pe = p.pe;

            const VkDeviceSize res_off = (static_cast<VkDeviceSize>(slot) * d->tw * 2 * pe +
                static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
            vkCmdFillBuffer(cmd, d->res_buf, res_off * 4, d->tw * 2 * pe * 4, 0);

            // the zero-fill must be visible to the atomic accumulation that
            // follows it in the next dispatch
            {
                VkMemoryBarrier mem_barrier {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                };
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
            }

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.bm3d_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->pipeline_layout, 0, 1, &stream.desc_set, 0, nullptr);
            {
                const int32_t pushes[4] {
                    static_cast<int32_t>(res_off),
                    m_i,
                    nf,
                    static_cast<int32_t>((r == 0) ? stream.stream_id : 0)
                };
                vkCmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(pushes), pushes);
            }
            vkCmdDispatch(cmd, p.bm3d_grid_x, p.bm3d_grid_y, 1);
        }
    }

    if (gputrace) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, stream.ts_query, 1);

    // The estimation kernels' atomic accumulation must be visible to the
    // aggregation reads, which are dispatched from a separate command buffer
    // (submitted later on the same queue): make the writes available to the
    // queue-wide scope so the aggregation sees complete slot contents.
    {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
    }

    vkEndCommandBuffer(cmd);
    return n_dispatches;
}

// Record the aggregation phase of the command buffer. It reads the res slots
// accumulated by the in-flight frames, so the host must have waited for their
// timelines before submitting it.
static void record_bm3d_agg(BM3DData * d, Bm3dStream & stream, int n) {
    VkCommandBuffer cmd = stream.cmd_agg;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    const int nf = d->nframes;
    const int r = d->radius;
    const bool gputrace = std::getenv("BM3D_GPUTRACE") != nullptr;

    for (int plane = 0; plane < d->n_planes; ++plane) {
        const auto & p = d->planes[plane];
        const VkDeviceSize pe = p.pe;

        // aggregation: tw stacked slices (clamped frame indices, aggZ blocks)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.agg_pipeline);
        // descriptor bindings do not carry across command buffers: cmd_agg is
        // recorded separately from the estimation phase, so without this bind
        // the dispatch runs on undefined descriptor state (black output, and
        // device loss under concurrent submissions)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d->pipeline_layout, 0, 1, &stream.desc_set, 0, nullptr);
        {
            int32_t bases[9] {};
            if (r == 0) {
                // non-temporal: aggregate the single center slice
                const int32_t base = static_cast<int32_t>(
                    static_cast<VkDeviceSize>(stream.win_slots[0]) * 2 * pe +
                    static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
                for (int i = 0; i < d->tw; ++i) bases[i] = base;
            } else {
                for (int i = 0; i < d->tw; ++i) {
                    const int z = agg_z(i, n, nf, r);
                    bases[i] = static_cast<int32_t>(
                        static_cast<VkDeviceSize>(stream.win_slots[i]) * d->tw * 2 * pe +
                        static_cast<VkDeviceSize>(plane) * d->res_size_per_plane +
                        static_cast<VkDeviceSize>(z) * 2 * pe);
                }
            }
            vkCmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(bases), bases);
        }
        (void)0;
        {
            // the estimation kernel's atomic accumulation (and the fill that
            // zeroes the slots) must be visible to the aggregation reads; the
            // aggregation kernel reads with atomic loads, but the RADV driver
            // still needs an explicit barrier for the cross-dispatch visibility
            VkMemoryBarrier mem_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
        }
        if (gputrace) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, stream.ts_query, 2);
        vkCmdDispatch(cmd, p.agg_grid_x, p.agg_grid_y, 1);
        if (gputrace) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, stream.ts_query, 3);
    }

    vkEndCommandBuffer(cmd);
}

static const VSFrame *VS_CC BM3DGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    BM3DData * d = static_cast<BM3DData *>(instanceData);

    if (activationReason == arInitial) {
        const int r = d->radius;
        for (int i = -2 * r; i <= 2 * r; ++i) {
            const int idx = std::clamp(n + i, 0, d->nframes - 1);
            vsapi->requestFrameFilter(idx, d->node, frameCtx);
            if (d->ref_node) {
                vsapi->requestFrameFilter(idx, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        VSFrame * dst;
        if (d->chroma) {
            // process the luma plane in place, copy the chroma planes and the
            // frame props from the source frame
            const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);
            const int pl[] = { 0, 1, 2 };
            const VSFrame * fr[] = { nullptr, src, src };
            dst = vsapi->newVideoFrame2(
                &d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);
            vsapi->freeFrame(src);
        } else {
            // the grayscale path has no planes to copy, but the center source
            // frame is still the property donor: without it the output loses
            // _DurationNum/_DurationDen/_ColorRange and application metadata
            // that the YUV path preserves
            const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);
            dst = vsapi->newVideoFrame(
                &d->vi->format, d->vi->width, d->vi->height, src, core);
            vsapi->freeFrame(src);
        }

        auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        auto stream = d->pool.take();
        auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d acquired\n", n);
        const int my_stream = stream.stream_id;
        const uint64_t my_seq = stream.seq++;
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d stream=%d\n", n, my_stream);

        // reserve this frame's cache slots (blocks only when the working set
        // exceeds the cache, e.g. on seeks; never holds a stream while waiting)
        acquire_cache(d, stream, n, my_seq);
        auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        const auto set_error = [&](const std::string & error_message) {
            // unblock any frames already waiting on this frame's timeline
            VkSemaphoreSignalInfo signal_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                .pNext = nullptr,
                .semaphore = stream.timeline,
                .value = my_seq
            };
            vkSignalSemaphore(d->device->device, &signal_info);
            release_cache(d, stream, n);
            d->pool.give_back(std::move(stream));
            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        VkDevice dev = d->device->device;

        // The estimation phase is submitted first so the GPU stays busy with
        // the heavy kernels while this host thread waits for the writers of
        // the aggregation slots before submitting the tiny aggregation.

        // upload only the frames whose cache slots the acquire reserved. The
        // needed range is the union of every window that this record's
        // dispatches may read: [clamp(n-2r), clamp(n+2r)].
        const int r = d->radius;
        const int lo = std::clamp(n - 2 * r, 0, d->nframes - 1);
        const int hi = std::clamp(n + 2 * r, 0, d->nframes - 1);
        std::array<bool, 4 * MAX_RADIUS + 1> uploaded {};
        bool any_uploaded = false;
        const int clips = d->final ? 2 : 1;
        for (int f = lo; f <= hi; ++f) {
            if (!stream.upload_new[f - lo]) {
                continue;
            }
            const VkDeviceSize slot_base = static_cast<VkDeviceSize>(f - lo) * clips * d->planes[0].pe;
            // source clip (the one actually denoised): the second half of the
            // slot in final mode, so the block matching / Wiener reference (the
            // first half) can be uploaded alongside it
            const VSFrame * src = vsapi->getFrameFilter(f, d->node, frameCtx);
            for (int plane = 0; plane < d->n_planes; ++plane) {
                const auto & p = d->planes[plane];
                auto srcp = vsapi->getReadPtr(src, plane);
                float * dstp = stream.map + slot_base +
                    static_cast<VkDeviceSize>(clips - 1) * p.pe +
                    static_cast<VkDeviceSize>(plane) * d->src_size;
                // the frame's stride is not necessarily the GPU plane pitch
                // (width rounded up to 4 floats), so copy the visible rows one
                // at a time unless both are tight
                copy_plane_out(dstp, static_cast<ptrdiff_t>(p.stride) * sizeof(float),
                    srcp, vsapi->getStride(src, plane),
                    static_cast<size_t>(p.width) * sizeof(float), p.height, true);
            }
            vsapi->freeFrame(src);
            if (d->final) {
                const VSFrame * rsrc = vsapi->getFrameFilter(f, d->ref_node, frameCtx);
                for (int plane = 0; plane < d->n_planes; ++plane) {
                    const auto & p = d->planes[plane];
                    auto srcp = vsapi->getReadPtr(rsrc, plane);
                    float * dstp = stream.map + slot_base +
                        static_cast<VkDeviceSize>(plane) * d->src_size;
                    copy_plane_out(dstp, static_cast<ptrdiff_t>(p.stride) * sizeof(float),
                        srcp, vsapi->getStride(rsrc, plane),
                        static_cast<size_t>(p.width) * sizeof(float), p.height, true);
                }
                vsapi->freeFrame(rsrc);
            }
            uploaded[f - lo] = true;
            any_uploaded = true;
        }

        // make the host-written staging visible to the device copies (the
        // cached mapping may hold dirty lines that the GPU would miss)
        if (any_uploaded) {
            VkMappedMemoryRange flush_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = stream.staging_mem,
                .offset = 0,
                .size = VK_WHOLE_SIZE
            };
            vkFlushMappedMemoryRanges(dev, 1, &flush_range);
        }
        auto t3 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        const int ndisp = record_bm3d_kernels(d, stream, n, uploaded);
        auto t4 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d kernels recorded\n", n);

        // Cross-frame dependencies are expressed on the writers' per-stream
        // timelines, signalled device-side by each writer's kernel submit. A
        // host-side wait alone does not establish device memory visibility, and
        // a single end-of-frame signal would deadlock on out-of-order arrivals
        // (the aggregation of an early frame waits on the kernels of a later
        // frame whose kernels wait on that early frame's source upload). The
        // estimation kernels read the source window (so they wait for the
        // uploaders), and the aggregation reads the estimate stacks (so it
        // waits for the stack writers plus its own kernels).
        const int src_lo = std::clamp(n - 2 * r, 0, d->nframes - 1);
        const int src_hi = std::clamp(n + 2 * r, 0, d->nframes - 1);

        std::vector<VkSemaphore> src_waits, res_waits;
        std::vector<uint64_t> src_values, res_values;
        // one entry per semaphore, at the maximum value requested: the same
        // stream can be the writer of several slots of this window, and a
        // submission must not carry the same timeline semaphore twice
        const auto add_wait = [](std::vector<VkSemaphore> & sems,
                                 std::vector<uint64_t> & values,
                                 VkSemaphore sem, uint64_t value) {
            for (size_t i = 0; i < sems.size(); ++i) {
                if (sems[i] == sem) {
                    values[i] = std::max(values[i], value);
                    return;
                }
            }
            sems.push_back(sem);
            values.push_back(value);
        };
        if (d->radius > 0) {
            for (int f = src_lo; f <= src_hi; ++f) {
                if (stream.upload_new[f - src_lo]) {
                    continue;   // own upload: copied ahead of our dispatches
                }
                const int w = stream.src_writers[f - src_lo];
                const VkSemaphore sem = stream.src_writer_sem[f - src_lo];
                if (w < 0 || sem == VK_NULL_HANDLE) {
                    continue;
                }
                if (sem == stream.timeline) {
                    continue;   // same stream: already queue-ordered
                }
                if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d waits on src uploader w=%d val=%llu\n", n, w, static_cast<unsigned long long>(stream.src_writer_value[f - src_lo]));
                add_wait(src_waits, src_values, sem, stream.src_writer_value[f - src_lo]);
            }
            for (int i = 0; i < d->tw; ++i) {
                const int w = stream.win_writers[i];
                const VkSemaphore sem = stream.win_writer_sem[i];
                if (w < 0 || sem == VK_NULL_HANDLE) {
                    continue;   // own work or a slot this frame recomputes
                }
                if (sem == stream.timeline) {
                    continue;   // same stream: already queue-ordered
                }
                if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d waits on writer w=%d val=%llu\n", n, w, static_cast<unsigned long long>(stream.win_writer_value[i]));
                add_wait(res_waits, res_values, sem, stream.win_writer_value[i]);
            }
        }

        std::vector<VkPipelineStageFlags> res_stages(res_waits.size(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        std::vector<VkPipelineStageFlags> src_stages(src_waits.size(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // The source window is read by this frame's estimation kernels. The
        // uploaders' copies are ordered ahead of their own kernels on the same
        // queue, so waiting for the uploaders' timelines here (host-side) is
        // deadlock-free: a GPU-side wait on a timeline signalled by a later
        // submission would block the whole queue. The host wait alone does not
        // establish the device-side producer->consumer memory dependency for
        // transfers that ran on another queue, so the same timelines are also
        // fed to the estimation submit as device waits (they are signalled by
        // now, so those waits are immediate).
        if (d->radius > 0 && !src_waits.empty()) {
            VkSemaphoreWaitInfo wait_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = static_cast<uint32_t>(src_waits.size()),
                .pSemaphores = src_waits.data(),
                .pValues = src_values.data()
            };
            checkVK(vkWaitSemaphores(dev, &wait_info, UINT64_MAX));
        }

        /* no fence here: the fence is signalled by the aggregation submit on
               the same queue, and a fence must not be attached to a second
               submission while a first one still holds it */
            checkVK(submit_timeline(dev, stream.queue, stream.queue_lock, stream.cmd,
                src_waits, src_values, src_stages, stream.timeline, my_seq, VK_NULL_HANDLE));
        auto t5 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        record_bm3d_agg(d, stream, n);
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d agg recorded\n", n);

        {
            // wait on our own kernels (the timeline is signalled by the kernel
            // submit) and on the frames that computed the aggregation slots.
            // The res waits stay device-side: a host-side wait here would let
            // two frames deadlock waiting on each other's aggregation before
            // either submits.
            std::vector<VkSemaphore> agg_waits { stream.timeline };
            std::vector<uint64_t> agg_values { my_seq };
            std::vector<VkPipelineStageFlags> agg_stages { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
            agg_waits.insert(agg_waits.end(), res_waits.begin(), res_waits.end());
            agg_values.insert(agg_values.end(), res_values.begin(), res_values.end());
            agg_stages.insert(agg_stages.end(), res_stages.begin(), res_stages.end());

            checkVK(submit_timeline(dev, stream.queue, stream.queue_lock,
                stream.cmd_agg, agg_waits, agg_values, agg_stages,
                VK_NULL_HANDLE, 0, stream.fence));
        }
        auto t6 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        checkVK(vkWaitForFences(dev, 1, &stream.fence, VK_TRUE, UINT64_MAX));
        auto t7 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d fenced\n", n);

        // hand the cache slots back only after the aggregation has completed:
        // the slots are shared with the other streams, and on a device with
        // more than one compute queue a later recompute (fill) could otherwise
        // run on another queue before this aggregation finished reading them.
        release_cache(d, stream, n);

        if (stream.ts_query && std::getenv("BM3D_GPUTRACE")) {
            static std::atomic<uint64_t> ts_k {}, ts_a {};
            static std::atomic<uint32_t> ts_nf {};
            uint64_t ts[4] {};
            if (vkGetQueryPoolResults(dev, stream.ts_query, 0, 4,
                    sizeof(ts), ts, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                const float period = d->device->limits.timestampPeriod;
                // fetch_add returns the previous count: the first completed
                // frame is sample 1, not 0 (dividing by 0 printed garbage)
                uint32_t nq = ts_nf.fetch_add(1) + 1;
                if (ts[0] && ts[1]) ts_k += (ts[1] - ts[0]) * period;
                if (ts[2] && ts[3]) ts_a += (ts[3] - ts[2]) * period;
                if (nq % 50 == 0) {
                    fprintf(stderr, "[bm3dgpu] n=%u kernel=%.3f agg=%.3f (ms) disp=%d\n",
                        nq, ts_k.load() / double(nq) / 1e3, ts_a.load() / double(nq) / 1e3, ndisp);
                }
            }
        }

        // make the device-written dst visible to the host (the cached mapping
        // may still hold stale lines even on coherent types)
        {
            VkMappedMemoryRange invalidate_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = stream.dst_mem,
                .offset = 0,
                .size = VK_WHOLE_SIZE
            };
            vkInvalidateMappedMemoryRanges(dev, 1, &invalidate_range);
        }

        // download the result into the frame
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            auto dstp = vsapi->getWritePtr(dst, plane);
            const float * h_bufferp = stream.dst_map +
                static_cast<VkDeviceSize>(plane) * p.pe;
            copy_plane_read(dstp, vsapi->getStride(dst, plane), h_bufferp,
                static_cast<ptrdiff_t>(p.stride) * sizeof(float),
                static_cast<size_t>(p.width) * sizeof(float), p.height);
        }

        d->pool.give_back(std::move(stream));

        if (d->host_timing) {
            auto t8 = std::chrono::steady_clock::now();
            const auto us = [](auto a, auto b) {
                return static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
            };
            d->ht_take_ns += us(t0, t1);
            d->ht_acquire_ns += us(t1, t2);
            d->ht_upload_ns += us(t2, t3);
            d->ht_record_ns += us(t3, t4);
            d->ht_srcwait_ns += us(t4, t5);
            d->ht_agg_ns += us(t5, t6);
            d->ht_fence_ns += us(t6, t7);
            // download runs after the fence: t7 -> t8 covers it (and the
            // give_back above)
            d->ht_down_ns += us(t7, t8);
            d->ht_total_ns += us(t0, t8);
            d->ht_n.fetch_add(1, std::memory_order_relaxed);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC BM3DFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    BM3DData * d = static_cast<BM3DData *>(instanceData);
    vsapi->freeNode(d->node);
    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    delete d;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BM3DData>() };

    // Opt-in host-path probe: the default path records no clocks.
    d->host_timing = getenv("VSFEEL_BM3D_TIMING") != nullptr;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    // optional "ref": a basic-estimate clip used for the final (Wiener) pass;
    // the block matching runs on it and its patches provide the Wiener
    // reference, while the noisy clip is the clip actually denoised
    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
    if (d->ref_node) {
        const VSVideoInfo * rvi = vsapi->getVideoInfo(d->ref_node);
        if (rvi->format.colorFamily != d->vi->format.colorFamily ||
            rvi->format.sampleType != d->vi->format.sampleType ||
            rvi->format.bitsPerSample != d->vi->format.bitsPerSample ||
            rvi->format.subSamplingW != d->vi->format.subSamplingW ||
            rvi->format.subSamplingH != d->vi->format.subSamplingH)
        {
            return set_error("\"ref\" must be of the same format as \"clip\"");
        }
        if (rvi->width != d->vi->width || rvi->height != d->vi->height) {
            return set_error("\"ref\" must be of the same dimensions as \"clip\"");
        }
        if (rvi->numFrames != d->vi->numFrames) {
            return set_error("\"ref\" must be of the same number of frames as \"clip\"");
        }
        d->final = true;
    }

    if (d->vi->width <= 0 || d->vi->height <= 0 ||
        d->vi->format.sampleType != stFloat || d->vi->format.bitsPerSample != 32) {
        return set_error("only constant format 32 bit float input supported");
    }

    // The block-matching kernel always loads unconditional 8x8 patches and
    // clamps its block origin to (dimension - 8), so anything smaller would
    // index negative buffer offsets. Reject it before allocating GPU
    // resources rather than relying on robust buffer access (which is not
    // enabled).
    if (d->vi->width < 8 || d->vi->height < 8) {
        return set_error("clip dimensions must be at least 8x8");
    }

    std::array<float, 3> sigma;
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] = static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &error));
        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be non-negative");
        }
    }
    for (int i = 0; i < std::ssize(sigma); ++i) {
        d->process = sigma[i] >= FLT_EPSILON;
    }

    // match the reference sigma scaling exactly (different factor for the
    // final Wiener pass)
    const float sigma_factor = d->final
        ? std::bit_cast<float>(0x3e40c0c1u)
        : std::bit_cast<float>(0x3f021bb6u);
    for (auto & sv : sigma) {
        sv *= sigma_factor;
    }
    d->sigma = sigma[0];
    d->sigma_u = sigma[1];
    d->sigma_v = sigma[2];

    std::array<int, 3> block_step;
    for (int i = 0; i < std::ssize(block_step); ++i) {
        block_step[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_step", i, &error));
        if (error) {
            block_step[i] = (i == 0) ? 8 : block_step[i - 1];
        } else if (block_step[i] <= 0 || block_step[i] > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }
    }
    d->block_step = block_step[0];

    std::array<int, 3> bm_range;
    for (int i = 0; i < std::ssize(bm_range); ++i) {
        bm_range[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "bm_range", i, &error));
        if (error) {
            bm_range[i] = (i == 0) ? 9 : bm_range[i - 1];
        } else if (bm_range[i] <= 0) {
            return set_error("\"bm_range\" must be positive");
        }
    }
    d->bm_range = bm_range[0];

    int64_t radius_raw = vsapi->mapGetInt(in, "radius", 0, &error);
    d->radius = vsh::int64ToIntS(radius_raw);
    if (error) {
        d->radius = 0;
    }
    if (d->radius < 0 || d->radius > 4) {
        return set_error("\"radius\" must be in range [0, 4]");
    }
    d->tw = 2 * d->radius + 1;

    std::array<int, 3> ps_num;
    for (int i = 0; i < std::ssize(ps_num); ++i) {
        ps_num[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_num", i, &error));
        if (error) {
            ps_num[i] = (i == 0) ? 2 : ps_num[i - 1];
        } else if (ps_num[i] <= 0 || ps_num[i] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }
    }
    d->ps_num = ps_num[0];

    std::array<int, 3> ps_range;
    for (int i = 0; i < std::ssize(ps_range); ++i) {
        ps_range[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_range", i, &error));
        if (error) {
            ps_range[i] = (i == 0) ? 4 : ps_range[i - 1];
        } else if (ps_range[i] <= 0) {
            return set_error("\"ps_range\" must be positive");
        }
    }
    d->ps_range = ps_range[0];

    d->num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 4;
    }
    if (d->num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("\"device_id\" must be non-negative");
    }
    d->device_id = device_id;

    // at radius 0 every frame only touches its own slot and never depends on
    // the previous frames' estimates, so give each in-flight frame its own
    // src/res slot to keep the pipeline full; otherwise the ring of 1 would
    // serialize the frames behind the timeline waits. At radius > 0 the caches
    // are keyed by frame modulo the capacity and reserved all-or-nothing per
    // frame, so they only need to cover the working set of the concurrent
    // frames (like the reference's fused-mode accumulator cache); anything
    // beyond that (e.g. seeking) blocks in the acquire instead of corrupting.
    d->src_ring = (d->radius == 0) ? d->num_streams : 4 * d->radius + d->num_streams;
    d->res_cap = (d->radius == 0) ? d->num_streams : d->tw + d->num_streams + 2 * d->radius;

    const int extractor_exp = vsh::int64ToIntS(vsapi->mapGetInt(in, "extractor_exp", 0, &error));
    d->extractor = (extractor_exp != 0)
        ? std::ldexp(1.0f, extractor_exp) : 0.0f;

    d->nframes = d->vi->numFrames;

    {
        const auto result = get_device(d->device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
    }

    // enable fp64 for the exactly-rounded aggregation division; the BM3D
    // kernels also accumulate into float SSBOs with atomicAdd
    // (GL_EXT_shader_atomic_float), so both features are required
    if (auto err = enable_float64(d->device)) {
        return set_error(*err);
    }
    if (!d->device->feat_atomic_float32_add) {
        return set_error("shaderBufferFloat32AtomicAdd is not supported by this device");
    }

    VkDevice dev = d->device->device;

    {
        VkDescriptorSetLayoutBinding bindings[3] {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 3,
            .pBindings = bindings
        };
        checkVK(vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &d->set_layout));
    }
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 9 * sizeof(int32_t)
        };
        VkPipelineLayoutCreateInfo pipeline_layout_info {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &d->set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range
        };
        checkVK(vkCreatePipelineLayout(dev, &pipeline_layout_info, nullptr, &d->pipeline_layout));
    }
    {
        VkDescriptorPoolSize pool_size {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * static_cast<uint32_t>(d->num_streams)
        };
        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(d->num_streams),
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size
        };
        checkVK(vkCreateDescriptorPool(dev, &pool_info, nullptr, &d->desc_pool));
    }

    // plane configs (luma plane 0; YUV chroma is passed through unprocessed)
    d->n_planes = 0;
    if (d->vi->format.colorFamily == cfGray) {
        auto & p = d->planes[0];
        p.width = d->vi->width;
        p.height = d->vi->height;
        p.stride = (d->vi->width + 3) & ~3;
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        d->n_planes = 1;
    } else if (d->vi->format.colorFamily == cfYUV) {
        auto & p = d->planes[0];
        p.width = d->vi->width;
        p.height = d->vi->height;
        p.stride = (d->vi->width + 3) & ~3;
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        d->n_planes = 1;
        d->chroma = true;
    } else {
        return set_error("BM3D: only Gray and YUV input are currently supported");
    }

    // shared buffers
    {
        VkDeviceSize src_size = 0;
        VkDeviceSize res_size = 0;
        VkDeviceSize dst_size = 0;
        (void)res_size;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            // in final mode each ring slot holds [ref][source], so the ring
            // doubles in size
            const int clips = d->final ? 2 : 1;
            src_size += static_cast<VkDeviceSize>(d->src_ring) * clips * p.pe;
            res_size += static_cast<VkDeviceSize>(d->res_cap) * d->tw * 2 * p.pe;
            dst_size += p.pe;
        }
        d->src_size = src_size;
        d->res_size_per_plane = static_cast<VkDeviceSize>(d->res_cap) * d->tw * 2 * d->planes[0].pe;
        d->dst_size = dst_size;
        (void)res_size;

        VkBufferCreateInfo src_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = src_size * 4,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };
        checkVK(vkCreateBuffer(dev, &src_info, nullptr, &d->src_buf));
        {
            const auto result = allocate_memory(*d->device, d->src_buf,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->src_mem = std::get<AllocatedMemory>(result).memory;
        }

        VkBufferCreateInfo res_info = src_info;
        res_info.size = res_size * 4;
        res_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        checkVK(vkCreateBuffer(dev, &res_info, nullptr, &d->res_buf));
        {
            const auto result = allocate_memory(*d->device, d->res_buf,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->res_mem = std::get<AllocatedMemory>(result).memory;
        }
    }

    // shader modules
    {
        VkShaderModuleCreateInfo module_info {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = bm3d_spv_size,
            .pCode = bm3d_spv
        };
        checkVK(vkCreateShaderModule(dev, &module_info, nullptr, &d->bm3d_module));
        VkShaderModuleCreateInfo module_info2 {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = bm3d_agg_spv_size,
            .pCode = bm3d_agg_spv
        };
        checkVK(vkCreateShaderModule(dev, &module_info2, nullptr, &d->agg_module));
    }

    // pipelines
    for (int plane = 0; plane < d->n_planes; ++plane) {
        auto & p = d->planes[plane];
        {
            const auto result = create_bm3d_pipeline(*d->device, p, *d, d->bm3d_module, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            p.bm3d_pipeline = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_agg_pipeline(*d->device, p, *d, d->agg_module, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            p.agg_pipeline = std::get<VkPipeline>(result);
        }
        p.bm3d_grid_x = static_cast<uint32_t>((p.width + 4 * d->block_step - 1) / (4 * d->block_step));
        p.bm3d_grid_y = static_cast<uint32_t>((p.height + d->block_step - 1) / d->block_step);
        p.agg_grid_x = static_cast<uint32_t>((p.width + 31) / 32);
        p.agg_grid_y = static_cast<uint32_t>((p.height + 7) / 8);
    }

    // streams
    d->pool.semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->pool.reserve(d->num_streams);

    // Queue sharing is swept independently of the stream count (see
    // resolve_queue_cap): override with VSFEEL_BM3D_QUEUES=N.
    uint32_t num_queues = resolve_queue_cap(d->num_streams,
        d->device->queue_count, "VSFEEL_BM3D_QUEUES", UINT32_MAX);

    for (int i = 0; i < d->num_streams; ++i) {
        Bm3dStream stream;

        VkDeviceSize staging_size = d->src_size * 4;
        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &stream.staging));
        }
        {
            const auto result = allocate_memory(*d->device, stream.staging,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            stream.staging_mem = std::get<AllocatedMemory>(result).memory;
            stream.staging_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        checkVK(vkMapMemory(dev, stream.staging_mem, 0, staging_size, 0,
            reinterpret_cast<void **>(&stream.map)));

        {
            // per-stream destination buffer: the aggregation writes the frame
            // here and the host downloads it after the fence, so sharing it
            // across streams would let a later aggregation clobber an earlier
            // frame's result before its host-side download.
            const VkDeviceSize dst_bytes = d->dst_size * 4;
            VkBufferCreateInfo dst_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = dst_bytes,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &dst_info, nullptr, &stream.dst_buf));
            {
                const auto result = allocate_memory(*d->device, stream.dst_buf,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                stream.dst_mem = std::get<AllocatedMemory>(result).memory;
            }
            checkVK(vkMapMemory(dev, stream.dst_mem, 0, dst_bytes, 0,
                reinterpret_cast<void **>(&stream.dst_map)));
        }


        {
            VkCommandPoolCreateInfo pool_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                /* the per-frame recording re-begins the same command buffers;
                   without this flag the implicit reset in vkBeginCommandBuffer
                   is invalid usage (intermittent stale submissions: black
                   output, device loss under load) */
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = d->device->queue_family
            };
            checkVK(vkCreateCommandPool(dev, &pool_info, nullptr, &stream.pool));
        }
        {
            VkCommandBufferAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = stream.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 2
            };
            VkCommandBuffer cmds[2] {};
            checkVK(vkAllocateCommandBuffers(dev, &alloc_info, cmds));
            stream.cmd = cmds[0];
            stream.cmd_agg = cmds[1];
        }
        {
            VkFenceCreateInfo fence_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            checkVK(vkCreateFence(dev, &fence_info, nullptr, &stream.fence));
        }
        {
            VkSemaphoreTypeCreateInfo type_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = nullptr,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = 0
            };
            VkSemaphoreCreateInfo sem_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &type_info,
                .flags = 0
            };
            checkVK(vkCreateSemaphore(dev, &sem_info, nullptr, &stream.timeline));
        }
        if (std::getenv("BM3D_GPUTRACE")) {
            VkQueryPoolCreateInfo qp_info {
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = 4
            };
            checkVK(vkCreateQueryPool(dev, &qp_info, nullptr, &stream.ts_query));
        }
        {
            VkDescriptorSetAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = d->desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &d->set_layout
            };
            checkVK(vkAllocateDescriptorSets(dev, &alloc_info, &stream.desc_set));

            VkDescriptorBufferInfo res_info { d->res_buf, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo src_info { d->src_buf, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo dst_info { stream.dst_buf, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet writes[3] {
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, stream.desc_set, 0, 0, 1,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res_info, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, stream.desc_set, 1, 0, 1,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &src_info, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, stream.desc_set, 2, 0, 1,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dst_info, nullptr },
            };
            vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);
        }

        stream.queue = d->device->queues[i % num_queues].queue;
        stream.queue_lock = d->device->queues[i % num_queues].lock.get();
        stream.stream_id = i;

        d->pool.push(std::move(stream));
    }

    d->src_frame.assign(d->src_ring, -1);
    d->src_writer.assign(d->src_ring, -1);
    d->src_writer_sem.assign(d->src_ring, VK_NULL_HANDLE);
    d->src_writer_value.assign(d->src_ring, 0);
    d->src_holders.resize(d->src_ring);
    d->res_frame.assign(d->res_cap, -1);
    d->res_writer.assign(d->res_cap, -1);
    d->res_writer_sem.assign(d->res_cap, VK_NULL_HANDLE);
    d->res_writer_value.assign(d->res_cap, 0);
    d->res_holders.resize(d->res_cap);

    BM3DData * data = d.release();

    VSFilterDependency deps[2] = {
        { data->node, rpStrictSpatial },
        { data->ref_node, rpStrictSpatial }
    };

    vsapi->createVideoFilter(
        out, "BM3D", data->vi,
        BM3DGetFrame, BM3DFree,
        fmParallel, deps, data->ref_node ? 2 : 1, data, core);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bm3dv2(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "BM3Dv2",
        "clip:vnode;"
        "ref:vnode:opt;"
        "sigma:float[]:opt;"
        "block_step:int[]:opt;"
        "bm_range:int[]:opt;"
        "radius:int:opt;"
        "ps_num:int[]:opt;"
        "ps_range:int[]:opt;"
        "num_streams:int:opt;"
        "extractor_exp:int:opt;"
        "device_id:int:opt;",
        "clip:vnode;",
        BM3DCreate, nullptr, plugin
    );
}
