#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkSemaphore timeline {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    uint32_t staging_type_index {};
    VkQueryPool ts_query {};
    int stream_id {};        // index into BM3DData::timelines
};


struct BM3DData {
    VSNode * node;
    const VSVideoInfo * vi;

    int radius, num_streams;
    int tw;                          // 2 * radius + 1
    float sigma;                     // scaled luma sigma
    float sigma_u, sigma_v;
    int block_step, bm_range, ps_num, ps_range;
    bool process;
    bool chroma;
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
    int src_ring {};             // 4 * radius + 1 (the window union, matches the kernel)
    int res_ring {};             // tw: one slot per frame of the temporal window
    VkDeviceSize src_size {};    // src_ring * pe elements (per plane, packed)
    VkBuffer src_buf {};
    VkDeviceMemory src_mem {};
    VkBuffer res_buf {};
    VkDeviceMemory res_mem {};
    VkBuffer dst_buf {};
    VkDeviceMemory dst_mem {};
    VkDeviceSize dst_size {};        // pe * 4 bytes

    VkDeviceSize res_size_per_plane {};  // floats per plane in the res buffer
    float * dst_map {};
    int nframes {};

    ticket_semaphore semaphore;
    std::vector<VkSemaphore> timelines;
    std::array<int, 4 * MAX_RADIUS + 1> src_content {};  // frame index currently in each src slot
    std::vector<int> res_content {};  // frame index whose estimates occupy each res slot
    std::array<int, 64> frame_stream {};   // stream index that processed frame n (mod 64)
    std::vector<Bm3dStream> streams;
    std::mutex streams_lock;

    ~BM3DData() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);
        for (auto & s : streams) {
            if (s.map) vkUnmapMemory(dev, s.staging_mem);
            if (s.staging_mem) vkFreeMemory(dev, s.staging_mem, nullptr);
            if (s.staging) vkDestroyBuffer(dev, s.staging, nullptr);
            if (s.cmd) vkFreeCommandBuffers(dev, s.pool, 1, &s.cmd);
            if (s.pool) vkDestroyCommandPool(dev, s.pool, nullptr);
            if (s.fence) vkDestroyFence(dev, s.fence, nullptr);
            if (s.timeline) vkDestroySemaphore(dev, s.timeline, nullptr);
        }
        if (dst_map) vkUnmapMemory(dev, dst_mem);
        if (dst_mem) vkFreeMemory(dev, dst_mem, nullptr);
        if (dst_buf) vkDestroyBuffer(dev, dst_buf, nullptr);
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
    VkPhysicalDeviceFeatures2 features2 {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = nullptr,
        .features = {}
    };
    vkGetPhysicalDeviceFeatures2(dev->physical_device, &features2);
    if (!features2.features.shaderFloat64) {
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
        int32_t nosearch, noestimate, src_ring;
    } spec {
        plane.width, plane.height, plane.stride, sigma_y,
        d.block_step, d.bm_range, d.radius, d.ps_num, d.ps_range, d.extractor,
        std::getenv("BM3D_NOSEARCH") ? 1 : 0,
        std::getenv("BM3D_NOESTIMATE") ? 1 : 0,
        d.src_ring
    };
    const std::array<VkSpecializationMapEntry, 13> entries {{
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
        dev.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
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
        dev.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
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

static int record_bm3d_cmd(BM3DData * d, Bm3dStream & stream, int n,
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
    for (int f = lo; f <= hi; ++f) {
        if (!uploaded[f - lo]) {
            continue;
        }
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            const VkDeviceSize pe = p.pe;
            VkBufferCopy region {
                .srcOffset = (static_cast<VkDeviceSize>(f - lo) * pe +
                    static_cast<VkDeviceSize>(plane) * d->src_size) * 4,
                .dstOffset = (static_cast<VkDeviceSize>(f % d->src_ring) * pe +
                    static_cast<VkDeviceSize>(plane) * d->src_size) * 4,
                .size = pe * 4
            };
            vkCmdCopyBuffer(cmd, stream.staging, d->src_buf, 1, &region);
        }
    }

    // compute the missing result slots for the aggregation window
    // (the frame n+r at the steady state; the boundary frames too)
    if (gputrace) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, stream.ts_query, 0);
    int n_dispatches = 0;
    for (int i = 0; i < d->tw; ++i) {
        const int m_i = std::clamp(n - r + i, 0, nf - 1);
        const int slot = m_i % d->res_ring;
        if (d->res_content[slot] == m_i) {
            continue;
        }
        d->res_content[slot] = m_i;
        n_dispatches++;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            const VkDeviceSize pe = p.pe;

            const VkDeviceSize res_off = (static_cast<VkDeviceSize>(slot) * d->tw * 2 * pe +
                static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
            vkCmdFillBuffer(cmd, d->res_buf, res_off * 4, d->tw * 2 * pe * 4, 0);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.bm3d_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->pipeline_layout, 0, 1, &stream.desc_set, 0, nullptr);
            {
                const int32_t pushes[3] {
                    static_cast<int32_t>(res_off),
                    m_i,
                    nf
                };
                vkCmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(pushes), pushes);
            }
            vkCmdDispatch(cmd, p.bm3d_grid_x, p.bm3d_grid_y, 1);
        }
    }

    if (gputrace) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, stream.ts_query, 1);

    for (int plane = 0; plane < d->n_planes; ++plane) {
        const auto & p = d->planes[plane];
        const VkDeviceSize pe = p.pe;

        // aggregation: tw stacked slices (clamped frame indices, aggZ blocks)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.agg_pipeline);
        {
            int32_t bases[9] {};
            if (r == 0) {
                // non-temporal: aggregate the single center slice
                const int m_i = std::clamp(n, 0, nf - 1);
                const int32_t base = static_cast<int32_t>(
                    static_cast<VkDeviceSize>(m_i % d->res_ring) * 2 * pe +
                    static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
                for (int i = 0; i < d->tw; ++i) bases[i] = base;
            } else {
                for (int i = 0; i < d->tw; ++i) {
                    const int m_i = std::clamp(n - r + i, 0, nf - 1);
                    const int z = agg_z(i, n, nf, r);
                    bases[i] = static_cast<int32_t>(
                        static_cast<VkDeviceSize>(m_i % d->res_ring) * d->tw * 2 * pe +
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
    return n_dispatches;
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
        }
    } else if (activationReason == arAllFramesReady) {
        VSFrame * dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, nullptr, core);

        d->semaphore.acquire();
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d acquired\n", n);
        d->streams_lock.lock();
        auto stream = std::move(d->streams.back());
        d->streams.pop_back();
        const int my_stream = stream.stream_id;
        d->frame_stream[n % 64] = my_stream;
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d stream=%d\n", n, my_stream);
        d->streams_lock.unlock();

        const auto set_error = [&](const std::string & error_message) {
            // unblock any frames already waiting on this frame's timeline
            VkSemaphoreSignalInfo signal_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                .pNext = nullptr,
                .semaphore = stream.timeline,
                .value = static_cast<uint64_t>(n + 1)
            };
            vkSignalSemaphore(d->device->device, &signal_info);
            d->streams_lock.lock();
            d->streams.push_back(std::move(stream));
            d->streams_lock.unlock();
            d->semaphore.release();
            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        VkDevice dev = d->device->device;

        // wait for the results of the previous frames that this frame's
        // aggregation depends on (the res slots of m = n-2..n+1)
        {
            std::vector<VkSemaphore> waits;
            std::vector<uint64_t> values;
            for (int f = std::max(n - 4, 0); f <= n - 1; ++f) {
                const int sf = d->frame_stream[f % 64];
                if (sf < 0) {
                    continue;   // frame never processed: its slots are computed here
                }
                if (sf == my_stream) {
                    continue;   // same stream: already fence-ordered
                }
                if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d waits on f=%d sf=%d val=%d\n", n, f, sf, f + 1);
                waits.push_back(d->timelines[sf]);
                values.push_back(static_cast<uint64_t>(f + 1));
            }
            if (!waits.empty() && !std::getenv("BM3D_NOWAIT")) {
                if (std::getenv("BM3D_TRACE")) {
                    for (size_t w = 0; w < waits.size(); ++w) {
                        uint64_t cur = 0;
                        vkGetSemaphoreCounterValue(dev, waits[w], &cur);
                        fprintf(stderr, "[t]   sem[%zu] cur=%llu want=%llu\n", w,
                            static_cast<unsigned long long>(cur),
                            static_cast<unsigned long long>(values[w]));
                    }
                }
                VkSemaphoreWaitInfo wait_info {
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .semaphoreCount = static_cast<uint32_t>(waits.size()),
                    .pSemaphores = waits.data(),
                    .pValues = values.data()
                };
                checkVK(vkWaitSemaphores(dev, &wait_info, UINT64_MAX));
                if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d waits done (%zu)\n", n, waits.size());
            }
        }

        // upload only the frames whose ring slots do not yet hold them. The
        // needed range is the union of every window that this record's
        // dispatches may read: [clamp(n-2r), clamp(n+2r)].
        // The data ordering is already guaranteed by the stream-timeline waits
        // above: the uploader of frame f (frame f-4 at the latest) is always
        // included in the waited frames n-4..n-1.
        const int r = d->radius;
        const int lo = std::clamp(n - 2 * r, 0, d->nframes - 1);
        const int hi = std::clamp(n + 2 * r, 0, d->nframes - 1);
        std::array<bool, 4 * MAX_RADIUS + 1> uploaded {};
        bool any_uploaded = false;
        for (int f = lo; f <= hi; ++f) {
            const int slot = f % d->src_ring;
            if (d->src_content[slot] == f) {
                continue;
            }
            d->src_content[slot] = f;
            const VSFrame * src = vsapi->getFrameFilter(f, d->node, frameCtx);
            for (int plane = 0; plane < d->n_planes; ++plane) {
                const auto & p = d->planes[plane];
                auto srcp = vsapi->getReadPtr(src, plane);
                float * dstp = stream.map +
                    static_cast<VkDeviceSize>(f - lo) * p.pe +
                    static_cast<VkDeviceSize>(plane) * d->src_size;
                const auto bytes = static_cast<size_t>(p.width) * sizeof(float) * p.height;
                copy_stream_out(dstp, srcp, bytes);
            }
            vsapi->freeFrame(src);
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

        const int ndisp = record_bm3d_cmd(d, stream, n, uploaded);
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d recorded\n", n);

        {
            std::lock_guard lock(*stream.queue_lock);

            checkVK(vkResetFences(dev, 1, &stream.fence));

            VkSubmitInfo submit_info {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &stream.cmd,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr
            };
            checkVK(vkQueueSubmit(stream.queue, 1, &submit_info, stream.fence));
        }

        checkVK(vkWaitForFences(dev, 1, &stream.fence, VK_TRUE, UINT64_MAX));
        if (std::getenv("BM3D_TRACE")) fprintf(stderr, "[t] n=%d fenced\n", n);

        if (stream.ts_query && std::getenv("BM3D_GPUTRACE")) {
            static std::atomic<uint64_t> ts_k {}, ts_a {};
            static std::atomic<uint32_t> ts_nf {};
            uint64_t ts[4] {};
            if (vkGetQueryPoolResults(dev, stream.ts_query, 0, 4,
                    sizeof(ts), ts, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                const float period = d->device->limits.timestampPeriod;
                uint32_t nq = ts_nf.fetch_add(1);
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
                .memory = d->dst_mem,
                .offset = 0,
                .size = VK_WHOLE_SIZE
            };
            vkInvalidateMappedMemoryRanges(dev, 1, &invalidate_range);
        }

        // download the result into the frame
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            auto dstp = vsapi->getWritePtr(dst, plane);
            const float * h_bufferp = d->dst_map +
                static_cast<VkDeviceSize>(plane) * p.pe;
            const auto bytes = static_cast<size_t>(p.width) * sizeof(float) * p.height;
            copy_stream_read(dstp, h_bufferp, bytes);
        }

        // signal the timeline so the dependent frames can aggregate
        {
            VkSemaphoreSignalInfo signal_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                .pNext = nullptr,
                .semaphore = stream.timeline,
                .value = static_cast<uint64_t>(n + 1)
            };
            vkSignalSemaphore(dev, &signal_info);
            if (std::getenv("BM3D_TRACE")) {
                uint64_t cur = 0;
                vkGetSemaphoreCounterValue(dev, stream.timeline, &cur);
                fprintf(stderr, "[t] n=%d signaled val=%llu cur=%llu\n", n,
                    static_cast<unsigned long long>(n + 1),
                    static_cast<unsigned long long>(cur));
            }
        }

        d->streams_lock.lock();
        d->streams.push_back(std::move(stream));
        d->streams_lock.unlock();
        d->semaphore.release();

        return dst;
    }

    return nullptr;
}

static void VS_CC BM3DFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    BM3DData * d = static_cast<BM3DData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BM3DData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    if (d->vi->width <= 0 || d->vi->height <= 0 ||
        d->vi->format.sampleType != stFloat || d->vi->format.bitsPerSample != 32) {
        return set_error("only constant format 32 bit float input supported");
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

    // match the reference sigma scaling exactly
    const float sigma_factor = std::bit_cast<float>(0x3f021bb6u);
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
    d->src_ring = 4 * d->radius + 1;
    d->res_ring = d->tw;

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

    const int extractor_exp = vsh::int64ToIntS(vsapi->mapGetInt(in, "extractor_exp", 0, &error));
    d->extractor = (extractor_exp != 0)
        ? std::ldexp(1.0f, extractor_exp) : 0.0f;

    d->nframes = d->vi->numFrames;

    {
        const auto result = get_device(0);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
    }

    // enable fp64 for the exactly-rounded aggregation division
    if (auto err = enable_float64(d->device)) {
        return set_error(*err);
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

    // plane configs (luma plane 0; chroma uses YUV444 only, one entry per plane)
    d->n_planes = 0;
    if (d->vi->format.colorFamily == cfGray) {
        auto & p = d->planes[0];
        p.width = d->vi->width;
        p.height = d->vi->height;
        p.stride = (d->vi->width + 3) & ~3;
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        d->n_planes = 1;
    } else {
        return set_error("BM3D: only Gray input is currently supported");
    }

    // shared buffers
    {
        VkDeviceSize src_size = 0;
        VkDeviceSize res_size = 0;
        VkDeviceSize dst_size = 0;
        (void)res_size;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            src_size += static_cast<VkDeviceSize>(d->src_ring) * p.pe;
            res_size += static_cast<VkDeviceSize>(d->res_ring) * d->tw * 2 * p.pe;
            dst_size += p.pe;
        }
        d->src_size = src_size;
        d->res_size_per_plane = static_cast<VkDeviceSize>(d->res_ring) * d->tw * 2 * d->planes[0].pe;
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

        VkBufferCreateInfo dst_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = dst_size * 4,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };
        checkVK(vkCreateBuffer(dev, &dst_info, nullptr, &d->dst_buf));
        {
            const auto result = allocate_memory(*d->device, d->dst_buf,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->dst_mem = std::get<AllocatedMemory>(result).memory;
        }
        checkVK(vkMapMemory(dev, d->dst_mem, 0, dst_size * 4, 0,
            reinterpret_cast<void **>(&d->dst_map)));
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
    d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->streams.reserve(d->num_streams);
    d->timelines.resize(d->num_streams);

    uint32_t num_queues = std::min(d->num_streams, static_cast<int>(d->device->queue_count));

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
            VkCommandPoolCreateInfo pool_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
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
                .commandBufferCount = 1
            };
            checkVK(vkAllocateCommandBuffers(dev, &alloc_info, &stream.cmd));
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
            d->timelines[i] = stream.timeline;
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
            VkDescriptorBufferInfo dst_info { d->dst_buf, 0, VK_WHOLE_SIZE };
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

        d->streams.push_back(std::move(stream));
    }

    d->res_content.assign(d->res_ring, -1);
    std::fill(d->src_content.begin(), d->src_content.end(), -1);
    std::fill(d->frame_stream.begin(), d->frame_stream.end(), -1);

    BM3DData * data = d.release();

    VSFilterDependency deps[1] = {{data->node, rpStrictSpatial}};

    vsapi->createVideoFilter(
        out, "BM3D", data->vi,
        BM3DGetFrame, BM3DFree,
        fmParallel, deps, 1, data, core);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bm3dv2(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "BM3Dv2",
        "clip:vnode;"
        "sigma:float[]:opt;"
        "block_step:int[]:opt;"
        "bm_range:int[]:opt;"
        "radius:int:opt;"
        "ps_num:int[]:opt;"
        "ps_range:int[]:opt;"
        "num_streams:int:opt;"
        "extractor_exp:int:opt;",
        "clip:vnode;",
        BM3DCreate, nullptr, plugin
    );
}
