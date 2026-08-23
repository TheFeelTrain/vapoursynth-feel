#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <strings.h>

#include <vulkan/vulkan.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

namespace {

// Workgroup tile geometry of the weight kernel (must match nlmeans.comp).
constexpr int BLK_X = 16;
constexpr int BLK_Y = 8;
constexpr int VRT_RESULT = 3;

// FLT_EPS bit pattern: u5 is seeded with it via vkCmdFillBuffer so the final
// denominator stays > 0 without an explicit guard (matches the reference).
constexpr uint32_t FLT_EPS_BITS = 0x34000000u;

struct SpecData {
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t pstride;
    int32_t pad;
    int32_t ph;
    int32_t s;
    int32_t d;
    int32_t ref;
    int32_t channels;
    int32_t wmode;
    float h;
    float wref;
    float h2_inv_norm;
};

static constexpr std::array<VkSpecializationMapEntry, 14> spec_entries {{
    { 0,  0, sizeof(int32_t) },
    { 1,  4, sizeof(int32_t) },
    { 2,  8, sizeof(int32_t) },
    { 3, 12, sizeof(int32_t) },
    { 4, 16, sizeof(int32_t) },
    { 5, 20, sizeof(int32_t) },
    { 6, 24, sizeof(int32_t) },
    { 7, 28, sizeof(int32_t) },
    { 8, 32, sizeof(int32_t) },
    { 9, 36, sizeof(int32_t) },
    { 10, 40, sizeof(int32_t) },
    { 11, 44, sizeof(float) },
    { 12, 48, sizeof(float) },
    { 13, 52, sizeof(float) },
}};

// One sweep-table variant per reachable temporal boundary count m=min(d, n).
struct Variant {
    uint32_t w_base {};
    uint32_t q_base {};
    uint32_t q_cnt {};
    std::vector<uint32_t> w_boff;
};

struct NLStream {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    uint32_t staging_type_index {};
    uint8_t * staging_map {};
    VkBuffer u1 {};
    VkDeviceMemory u1_mem {};
    VkBuffer u1r {};
    VkDeviceMemory u1r_mem {};
    VkBuffer u1z {};
    VkDeviceMemory u1z_mem {};
    uint32_t u1z_type_index {};
    uint8_t * u1z_map {};
    VkBuffer u2 {};
    VkDeviceMemory u2_mem {};
    VkBuffer u4a {};
    VkDeviceMemory u4a_mem {};
    VkBuffer u5 {};
    VkDeviceMemory u5_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
};

struct NLMeansData {
    VSNode * node {};
    VSNode * ref_node {};
    const VSVideoInfo * vi;

    int bits {}, elem_bytes {};
    bool has_ref {};

    int ref_mode {};       // 0 luma, 1 chroma, 2 yuv, 3 rgb
    int channels {};       // processed channel count (1/2/3)
    int plane0 {};         // first processed VS plane

    int d {}, a {}, s {}, wmode {};
    float h_param {}, wref_param {};

    int width {}, height {};   // processed lattice dims (chroma-subsampled for UV)
    int stride {};             // internal element stride (= width)
    int pad {}, pstride {}, ph {}, layers {};
    int64_t npix {};
    int qb {};
    int num_streams {};
    VkDeviceSize u1_bytes {};  // bytes of one padded window (all channels/layers)

    // create()-time q-sweep tables (stride-8 rows), shared by all streams
    std::vector<int> wq_host;
    std::vector<int> aq_host;
    std::vector<Variant> variants;

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule weight_module {};
    VkShaderModule acc_module {};
    VkShaderModule fin_module {};
    VkPipeline weight_pipeline {};
    VkPipeline acc_pipeline {};
    VkPipeline fin_pipeline {};
    VkBuffer tables_buf {};
    VkDeviceMemory tables_mem {};
    VkDeviceSize aq_offset {};

    ticket_semaphore semaphore;
    std::vector<NLStream> streams;
    std::mutex streams_lock;

    ~NLMeansData() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);

        for (auto & st : streams) {
            if (st.staging_map) vkUnmapMemory(dev, st.staging_mem);
            if (st.staging_mem) vkFreeMemory(dev, st.staging_mem, nullptr);
            if (st.staging) vkDestroyBuffer(dev, st.staging, nullptr);
            if (st.u1_mem) vkFreeMemory(dev, st.u1_mem, nullptr);
            if (st.u1) vkDestroyBuffer(dev, st.u1, nullptr);
            if (st.u1r_mem) vkFreeMemory(dev, st.u1r_mem, nullptr);
            if (st.u1r) vkDestroyBuffer(dev, st.u1r, nullptr);
            if (st.u1z_map) vkUnmapMemory(dev, st.u1z_mem);
            if (st.u1z_mem) vkFreeMemory(dev, st.u1z_mem, nullptr);
            if (st.u1z) vkDestroyBuffer(dev, st.u1z, nullptr);
            if (st.u2_mem) vkFreeMemory(dev, st.u2_mem, nullptr);
            if (st.u2) vkDestroyBuffer(dev, st.u2, nullptr);
            if (st.u4a_mem) vkFreeMemory(dev, st.u4a_mem, nullptr);
            if (st.u4a) vkDestroyBuffer(dev, st.u4a, nullptr);
            if (st.u5_mem) vkFreeMemory(dev, st.u5_mem, nullptr);
            if (st.u5) vkDestroyBuffer(dev, st.u5, nullptr);
            if (st.cmd) vkFreeCommandBuffers(dev, st.pool, 1, &st.cmd);
            if (st.pool) vkDestroyCommandPool(dev, st.pool, nullptr);
            if (st.fence) vkDestroyFence(dev, st.fence, nullptr);
        }
        if (tables_mem) vkFreeMemory(dev, tables_mem, nullptr);
        if (tables_buf) vkDestroyBuffer(dev, tables_buf, nullptr);
        if (fin_pipeline) vkDestroyPipeline(dev, fin_pipeline, nullptr);
        if (acc_pipeline) vkDestroyPipeline(dev, acc_pipeline, nullptr);
        if (weight_pipeline) vkDestroyPipeline(dev, weight_pipeline, nullptr);
        if (fin_module) vkDestroyShaderModule(dev, fin_module, nullptr);
        if (acc_module) vkDestroyShaderModule(dev, acc_module, nullptr);
        if (weight_module) vkDestroyShaderModule(dev, weight_module, nullptr);
        if (desc_pool) vkDestroyDescriptorPool(dev, desc_pool, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        release_device(device);
    }
};

std::variant<VkBuffer, std::string> create_buffer(VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage) {
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

std::optional<std::string> bind_memory(
    const NLMeansData & d, VkBuffer buffer, VkDeviceMemory & mem,
    VkMemoryPropertyFlags required) {

    const auto result = allocate_memory(*d.device, buffer, required);
    if (std::holds_alternative<std::string>(result)) {
        return std::get<std::string>(result);
    }
    mem = std::get<AllocatedMemory>(result).memory;
    return std::nullopt;
}

std::variant<VkPipeline, std::string> create_pipeline(
    const NLMeansData & d, const SpecData & spec,
    VkShaderModule module, VkPipelineLayout layout) {

    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(spec_entries.size()),
        .pMapEntries = spec_entries.data(),
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
        d.device->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return "vkCreateComputePipelines failed"s;
    }
    return pipeline;
}

void record_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src_stage,
                    VkPipelineStageFlags dst_stage,
                    VkAccessFlags src_access, VkAccessFlags dst_access) {
    VkMemoryBarrier barrier {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void record_dispatch(NLMeansData * d, NLStream & st, VkPipeline pipeline,
                     const int32_t (&push)[2],
                     uint32_t gx, uint32_t gy, uint32_t gz) {
    vkCmdBindPipeline(st.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(st.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        d->pipeline_layout, 0, 1, &st.desc_set, 0, nullptr);
    vkCmdPushConstants(st.cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), push);
    vkCmdDispatch(st.cmd, gx, gy, gz);
}

static const VSFrame *VS_CC NLMeansGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    NLMeansData * d = static_cast<NLMeansData *>(instanceData);

    if (activationReason == arInitial) {
        const int m = std::min(d->d, n);
        for (int k = -m; k <= m; ++k) {
            const int idx = std::clamp(n + k, 0, d->vi->numFrames - 1);
            vsapi->requestFrameFilter(idx, d->node, frameCtx);
            if (d->has_ref) {
                vsapi->requestFrameFilter(idx, d->ref_node, frameCtx);
            }
        }
        return nullptr;
    } else if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const int nf = d->vi->numFrames;
    const int m = std::min(d->d, n);
    const int count = 2 * m + 1;
    const int k_start = -m;
    const int C = d->channels;

    std::vector<const VSFrame *> frames(count);
    std::vector<const VSFrame *> rframes(d->has_ref ? count : 0);
    for (int i = 0; i < count; ++i) {
        const int idx = std::clamp(n + k_start + i, 0, nf - 1);
        frames[i] = vsapi->getFrameFilter(idx, d->node, frameCtx);
        if (d->has_ref) {
            rframes[i] = vsapi->getFrameFilter(idx, d->ref_node, frameCtx);
        }
    }

    // unprocessed planes pass through from the center frame
    const int num_planes = d->vi->format.numPlanes;
    std::array<const VSFrame *, 3> fr {};
    std::array<int, 3> pl {};
    for (int p = 0; p < num_planes; ++p) {
        pl[p] = p;
        if (p < d->plane0 || p >= d->plane0 + C) {
            fr[p] = frames[m];
        }
    }
    VSFrame * dst = vsapi->newVideoFrame2(
        &d->vi->format, d->vi->width, d->vi->height, fr.data(), pl.data(),
        frames[m], core);

    d->semaphore.acquire();
    d->streams_lock.lock();
    auto stream = std::move(d->streams.back());
    d->streams.pop_back();
    d->streams_lock.unlock();

    auto cleanup_frames = [&] {
        for (int i = 0; i < count; ++i) {
            vsapi->freeFrame(frames[i]);
            if (d->has_ref) {
                vsapi->freeFrame(rframes[i]);
            }
        }
    };

    auto set_error = [&](const std::string & error_message) {
        d->streams_lock.lock();
        d->streams.push_back(std::move(stream));
        d->streams_lock.unlock();
        d->semaphore.release();
        vsapi->setFilterError(("NLMeans: " + error_message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        cleanup_frames();
        return static_cast<const VSFrame *>(nullptr);
    };

    VkDevice dev = d->device->device;

    // ------------------------------------------------------------------
    // upload: compose the zero-padded temporal window in the staging
    // mirror of u1 (+ u1r); margins were zeroed once at init and stay zero.
    // ------------------------------------------------------------------
    {
        const size_t lay_el = static_cast<size_t>(d->pstride) * d->ph;
        const size_t pad_off = static_cast<size_t>(d->pad) * d->pstride + d->pad;
        for (int i = 0; i < count; ++i) {
            const size_t t_layer = static_cast<size_t>(d->d) - m + i;
            for (int c = 0; c < C; ++c) {
                const int plane = d->plane0 + c;
                const uint8_t * srcp =
                    static_cast<const uint8_t *>(vsapi->getReadPtr(frames[i], plane));
                const size_t s_pitch = vsapi->getStride(frames[i], plane);
                const size_t row_bytes =
                    static_cast<size_t>(d->width) * d->elem_bytes;
                uint8_t * dstp = stream.staging_map +
                    ((static_cast<size_t>(c) * d->layers + t_layer) * lay_el +
                     pad_off) * d->elem_bytes;
                for (int y = 0; y < d->height; ++y) {
                    copy_stream_out(dstp + static_cast<size_t>(y) * d->pstride * d->elem_bytes,
                        srcp + static_cast<size_t>(y) * s_pitch, row_bytes);
                }
                if (d->has_ref) {
                    const uint8_t * rsrcp =
                        static_cast<const uint8_t *>(vsapi->getReadPtr(rframes[i], plane));
                    const size_t r_pitch = vsapi->getStride(rframes[i], plane);
                    uint8_t * rgp = stream.staging_map + d->u1_bytes +
                        ((static_cast<size_t>(c) * d->layers + t_layer) * lay_el +
                         pad_off) * d->elem_bytes;
                    for (int y = 0; y < d->height; ++y) {
                        copy_stream_out(rgp + static_cast<size_t>(y) * d->pstride * d->elem_bytes,
                            rsrcp + static_cast<size_t>(y) * r_pitch, row_bytes);
                    }
                }
            }
        }
    }

    const bool staging_coherent =
        !!(d->device->mem_props.memoryTypes[
            stream.staging_type_index].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!staging_coherent) {
        VkMappedMemoryRange range {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = nullptr,
            .memory = stream.staging_mem,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };
        vkFlushMappedMemoryRanges(dev, 1, &range);
    }

    // ------------------------------------------------------------------
    // command buffer: staging -> padded windows, scratch init, batched
    // weight/accumulation sweep, finish
    // ------------------------------------------------------------------
    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    if (vkBeginCommandBuffer(stream.cmd, &begin_info) != VK_SUCCESS) {
        return set_error("vkBeginCommandBuffer failed");
    }

    const size_t lay_bytes = static_cast<size_t>(d->pstride) * d->ph * d->elem_bytes;
    const size_t win_bytes = static_cast<size_t>(count) * lay_bytes;
    for (int c = 0; c < C; ++c) {
        const size_t off = static_cast<size_t>(c) * d->layers * lay_bytes +
            static_cast<size_t>(d->d - m) * lay_bytes;
        VkBufferCopy region {
            .srcOffset = off,
            .dstOffset = off,
            .size = win_bytes
        };
        vkCmdCopyBuffer(stream.cmd, stream.staging, stream.u1, 1, &region);
        if (d->has_ref) {
            VkBufferCopy region_ref {
                .srcOffset = d->u1_bytes + off,
                .dstOffset = off,
                .size = win_bytes
            };
            vkCmdCopyBuffer(stream.cmd, stream.staging, stream.u1r, 1, &region_ref);
        }
    }

    // sub-dword stores of the finish kernel OR into u1z dwords
    if (d->elem_bytes == 2) {
        vkCmdFillBuffer(stream.cmd, stream.u1z, 0, VK_WHOLE_SIZE, 0);
    }
    vkCmdFillBuffer(stream.cmd, stream.u2, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(stream.cmd, stream.u5, 0, VK_WHOLE_SIZE, FLT_EPS_BITS);

    record_barrier(stream.cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    const uint32_t gx = static_cast<uint32_t>((d->width + BLK_X - 1) / BLK_X);
    const uint32_t gy = static_cast<uint32_t>((d->height + BLK_Y - 1) / BLK_Y);
    const uint32_t gy_w = static_cast<uint32_t>(
        (d->height + VRT_RESULT * BLK_Y - 1) / (VRT_RESULT * BLK_Y));

    const Variant & v = d->variants[m];
    uint32_t q0 = 0;
    size_t bi = 0;
    while (q0 < v.q_cnt) {
        const uint32_t nb = std::min<uint32_t>(d->qb, v.q_cnt - q0);
        const uint32_t p0 = v.w_boff[bi];
        const uint32_t p1 = v.w_boff[bi + 1];

        const int32_t w_push[2] {
            static_cast<int32_t>(v.w_base + p0), 0
        };
        record_dispatch(d, stream, d->weight_pipeline, w_push, gx, gy_w, p1 - p0);
        record_barrier(stream.cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        const int32_t a_push[2] {
            static_cast<int32_t>(v.q_base + q0), static_cast<int32_t>(nb)
        };
        record_dispatch(d, stream, d->acc_pipeline, a_push, gx, gy, 1);
        record_barrier(stream.cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        q0 += nb;
        ++bi;
    }

    {
        const int32_t f_push[2] { 0, 0 };
        record_dispatch(d, stream, d->fin_pipeline, f_push, gx, gy, 1);
    }

    if (vkEndCommandBuffer(stream.cmd) != VK_SUCCESS) {
        return set_error("vkEndCommandBuffer failed");
    }

    {
        std::lock_guard lock(*stream.queue_lock);
        if (vkResetFences(dev, 1, &stream.fence) != VK_SUCCESS) {
            return set_error("vkResetFences failed");
        }
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
        if (vkQueueSubmit(stream.queue, 1, &submit_info, stream.fence) != VK_SUCCESS) {
            return set_error("vkQueueSubmit failed");
        }
    }

    if (vkWaitForFences(dev, 1, &stream.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return set_error("vkWaitForFences failed");
    }

    const bool dst_coherent =
        !!(d->device->mem_props.memoryTypes[
            stream.u1z_type_index].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!dst_coherent) {
        VkMappedMemoryRange range {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = nullptr,
            .memory = stream.u1z_mem,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };
        vkInvalidateMappedMemoryRanges(dev, 1, &range);
    }

    // download the packed channel-major result into the destination planes
    // (plain row copies: copy_stream_read's non-temporal loads require
    // 32-byte-aligned sources, which per-row offsets cannot guarantee)
    for (int c = 0; c < C; ++c) {
        uint8_t * dstp = static_cast<uint8_t *>(vsapi->getWritePtr(dst, d->plane0 + c));
        const size_t d_pitch = vsapi->getStride(dst, d->plane0 + c);
        const uint8_t * srcp = stream.u1z_map +
            static_cast<size_t>(c) * d->npix * d->elem_bytes;
        const size_t row_bytes = static_cast<size_t>(d->width) * d->elem_bytes;
        for (int y = 0; y < d->height; ++y) {
            memcpy(dstp + static_cast<size_t>(y) * d_pitch,
                srcp + static_cast<size_t>(y) * row_bytes, row_bytes);
        }
    }

    cleanup_frames();

    d->streams_lock.lock();
    d->streams.push_back(std::move(stream));
    d->streams_lock.unlock();
    d->semaphore.release();

    return dst;
}

static void VS_CC NLMeansFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    NLMeansData * d = static_cast<NLMeansData *>(instanceData);
    vsapi->freeNode(d->node);
    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    delete d;
}

static void VS_CC NLMeansCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<NLMeansData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("NLMeans: " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    d->ref_node = vsapi->mapGetNode(in, "rclip", 0, &error);
    d->has_ref = d->ref_node != nullptr;
    if (d->has_ref) {
        const VSVideoInfo * rvi = vsapi->getVideoInfo(d->ref_node);
        if (!vsh::isSameVideoInfo(rvi, d->vi) ||
            rvi->numFrames != d->vi->numFrames ||
            rvi->width != d->vi->width || rvi->height != d->vi->height) {
            return set_error("'rclip' must match the source clip's format, "
                             "dimensions and frame count.");
        }
    }

    const auto & fmt = d->vi->format;
    if (!vsh::isConstantVideoFormat(d->vi) ||
        (fmt.sampleType == stInteger && fmt.bitsPerSample != 16) ||
        (fmt.sampleType == stFloat && fmt.bitsPerSample != 32)) {
        return set_error("input bitdepth must be 16 (integer) or 32 (float).");
    }
    d->bits = fmt.bitsPerSample;
    d->elem_bytes = fmt.bytesPerSample;

    if (d->vi->width <= 0 || d->vi->height <= 0) {
        return set_error("clip must have constant dimensions.");
    }
    if (d->vi->width > 8192 || d->vi->height > 8192) {
        return set_error("8192x8192 is the highest supported resolution.");
    }

    enum { REF_LUMA = 0, REF_CHROMA = 1, REF_YUV = 2, REF_RGB = 3 };

    int dd = vsh::int64ToIntS(vsapi->mapGetInt(in, "d", 0, &error));
    if (error) {
        dd = 1;
    }
    if (dd < 0 || dd > 16) {
        return set_error("d must be 0..16.");
    }

    int aa = vsh::int64ToIntS(vsapi->mapGetInt(in, "a", 0, &error));
    if (error) {
        aa = 2;
    }
    if (aa < 1 || aa > 64) {
        return set_error("a must be 1..64.");
    }

    int ss = vsh::int64ToIntS(vsapi->mapGetInt(in, "s", 0, &error));
    if (error) {
        ss = 4;
    }
    if (ss < 0 || ss > 8) {
        return set_error("s must be 0..8.");
    }

    d->h_param = static_cast<float>(vsapi->mapGetFloat(in, "h", 0, &error));
    if (error) {
        d->h_param = 1.2f;
    }
    if (!(d->h_param > 0.0f)) {
        return set_error("h must be > 0.");
    }

    d->wmode = vsh::int64ToIntS(vsapi->mapGetInt(in, "wmode", 0, &error));
    if (error) {
        d->wmode = 0;
    }
    if (d->wmode < 0 || d->wmode > 3) {
        return set_error("wmode must be 0..3.");
    }

    d->wref_param = static_cast<float>(vsapi->mapGetFloat(in, "wref", 0, &error));
    if (error) {
        d->wref_param = 1.0f;
    }
    if (d->wref_param < 0.0f) {
        return set_error("wref must be >= 0.");
    }

    d->num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 1;
    }
    if (d->num_streams < 1 || d->num_streams > 32) {
        return set_error("num_streams must be 1..32.");
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    const char * chstr = vsapi->mapGetData(in, "channels", 0, &error);
    if (error || !chstr) {
        chstr = "auto";
    }
    auto eq = [](const char * x, const char * y) {
        return strcasecmp(x, y) == 0;
    };

    switch (fmt.colorFamily) {
        case cfGray:
            if (!(eq(chstr, "Y") || eq(chstr, "auto"))) {
                return set_error("'channels' must be 'Y' with Gray.");
            }
            d->ref_mode = REF_LUMA;
            d->channels = 1;
            d->plane0 = 0;
            break;
        case cfYUV:
            if (eq(chstr, "YUV")) {
                if (fmt.subSamplingW != 0 || fmt.subSamplingH != 0) {
                    return set_error("'channels'='YUV' requires 4:4:4.");
                }
                d->ref_mode = REF_YUV;
                d->channels = 3;
                d->plane0 = 0;
            } else if (eq(chstr, "Y") || eq(chstr, "auto")) {
                d->ref_mode = REF_LUMA;
                d->channels = 1;
                d->plane0 = 0;
            } else if (eq(chstr, "UV")) {
                d->ref_mode = REF_CHROMA;
                d->channels = 2;
                d->plane0 = 1;
            } else {
                return set_error("'channels' must be 'YUV', 'Y' or 'UV' with YUV.");
            }
            break;
        case cfRGB:
            if (!(eq(chstr, "RGB") || eq(chstr, "auto"))) {
                return set_error("'channels' must be 'RGB' with RGB.");
            }
            d->ref_mode = REF_RGB;
            d->channels = 3;
            d->plane0 = 0;
            break;
        default:
            return set_error("unsupported color family.");
    }

    const int ssw = fmt.subSamplingW;
    const int ssh = fmt.subSamplingH;
    d->width = (d->ref_mode == REF_CHROMA) ? d->vi->width >> ssw : d->vi->width;
    d->height = (d->ref_mode == REF_CHROMA) ? d->vi->height >> ssh : d->vi->height;

    if (2 * aa + 1 > d->width || 2 * aa + 1 > d->height) {
        return set_error("research window (2*a+1) larger than the frame.");
    }

    d->d = dd;
    d->a = aa;
    d->s = ss;
    d->layers = 2 * dd + 1;
    d->pad = aa;
    d->pstride = (d->width + 2 * aa + 7) & ~7;
    d->ph = d->height + 2 * aa;
    d->stride = d->width;
    d->npix = static_cast<int64_t>(d->stride) * d->height;
    d->qb = d->npix <= static_cast<int64_t>(1920) * 1152 ? 8 : 4;
    const int slots = (dd == 0) ? d->qb : 2 * d->qb;

    // int32 addressing bound of the device-side layouts (the reference falls
    // back to 64-bit indices here; we reject instead)
    {
        const int64_t lay = static_cast<int64_t>(d->pstride) * d->ph;
        const int64_t idx_max = std::max({
            lay * d->layers * static_cast<int64_t>(d->channels),
            d->npix * slots,
            d->npix * d->channels });
        if (idx_max >= (INT64_C(1) << 31)) {
            return set_error("resolution/temporal radius combination exceeds "
                             "the addressable range.");
        }
    }

    // q-batched sweep tables, built exactly like the reference: for each
    // reachable boundary count m the half-space of displacements (exploiting
    // weight(p,p+q)==weight(p,p-q)) in k/j/i order, grouped into batches of qb
    {
        const int64_t spt_side = 2LL * aa + 1;
        const int64_t spt_area = spt_side * spt_side;
        d->variants.resize(dd + 1);
        for (int mm = 0; mm <= dd; ++mm) {
            Variant & v = d->variants[mm];
            v.w_base = static_cast<uint32_t>(d->wq_host.size() / 8);
            v.q_base = static_cast<uint32_t>(d->aq_host.size() / 8);
            uint32_t q_idx = 0;
            for (int kk = -mm; kk <= 0; ++kk) {
                for (int j = -aa; j <= aa; ++j) {
                    for (int i = -aa; i <= aa; ++i) {
                        if (static_cast<int64_t>(kk) * spt_area +
                                static_cast<int64_t>(j) * spt_side + i < 0) {
                            const uint32_t b_local = q_idx % static_cast<uint32_t>(d->qb);
                            if (b_local == 0) {
                                v.w_boff.push_back(static_cast<uint32_t>(
                                    d->wq_host.size() / 8 - v.w_base));
                            }
                            const int slot_c = (dd == 0)
                                ? static_cast<int>(b_local)
                                : 2 * static_cast<int>(b_local);
                            const int slot_m = (kk != 0) ? slot_c + 1 : slot_c;
                            const int wrow_c[8] { dd, i, j, kk, slot_c, 0, 0, 0 };
                            d->wq_host.insert(d->wq_host.end(), std::begin(wrow_c),
                                std::end(wrow_c));
                            if (kk != 0) {
                                const int wrow_m[8] { dd - kk, i, j, kk, slot_m, 0, 0, 0 };
                                d->wq_host.insert(d->wq_host.end(), std::begin(wrow_m),
                                    std::end(wrow_m));
                            }
                            const int arow[8] { i, j, kk, slot_c, slot_m, 0, 0, 0 };
                            d->aq_host.insert(d->aq_host.end(), std::begin(arow),
                                std::end(arow));
                            ++q_idx;
                        }
                    }
                }
            }
            v.w_boff.push_back(static_cast<uint32_t>(
                d->wq_host.size() / 8 - v.w_base));
            v.q_cnt = q_idx;
        }
    }

    {
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
    }

    VkDevice dev = d->device->device;

    // H2_INV_NORM = 255^2 / (3*h*h*(2*s+1)^2), evaluated left-to-right like
    // the reference macro expansion
    const float nlm_norm = 255.0f * 255.0f;
    const float s_size = static_cast<float>((2 * ss + 1) * (2 * ss + 1));
    float denom = 3.0f * d->h_param;
    denom *= d->h_param;
    denom *= s_size;
    const float h2_inv_norm = nlm_norm / denom;

    SpecData spec {
        .width = d->width,
        .height = d->height,
        .stride = d->stride,
        .pstride = d->pstride,
        .pad = d->pad,
        .ph = d->ph,
        .s = d->s,
        .d = d->d,
        .ref = d->ref_mode,
        .channels = d->channels,
        .wmode = d->wmode,
        .h = d->h_param,
        .wref = d->wref_param,
        .h2_inv_norm = h2_inv_norm
    };

    // descriptor set layout: 8 storage buffer bindings
    {
        VkDescriptorSetLayoutBinding bindings[8];
        for (uint32_t b = 0; b < 8; ++b) {
            bindings[b] = VkDescriptorSetLayoutBinding {
                .binding = b,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr
            };
        }
        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 8,
            .pBindings = bindings
        };
        if (vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &d->set_layout) != VK_SUCCESS) {
            return set_error("vkCreateDescriptorSetLayout failed");
        }
    }
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 2 * sizeof(int32_t)
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
        if (vkCreatePipelineLayout(dev, &pipeline_layout_info, nullptr, &d->pipeline_layout) != VK_SUCCESS) {
            return set_error("vkCreatePipelineLayout failed");
        }
    }
    {
        VkDescriptorPoolSize pool_size {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 * static_cast<uint32_t>(d->num_streams)
        };
        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(d->num_streams),
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size
        };
        if (vkCreateDescriptorPool(dev, &pool_info, nullptr, &d->desc_pool) != VK_SUCCESS) {
            return set_error("vkCreateDescriptorPool failed");
        }
    }

    // shader modules + pipelines for this instance's io format
    {
        const uint32_t * weight_code;
        const uint32_t * acc_code;
        const uint32_t * fin_code;
        size_t weight_size, acc_size, fin_size;
        if (d->bits == 16) {
            weight_code = nlmeans_16_weight_spv;
            weight_size = nlmeans_16_weight_spv_size;
            acc_code = nlmeans_16_acc_spv;
            acc_size = nlmeans_16_acc_spv_size;
            fin_code = nlmeans_16_finish_spv;
            fin_size = nlmeans_16_finish_spv_size;
        } else {
            weight_code = nlmeans_32_weight_spv;
            weight_size = nlmeans_32_weight_spv_size;
            acc_code = nlmeans_32_acc_spv;
            acc_size = nlmeans_32_acc_spv_size;
            fin_code = nlmeans_32_finish_spv;
            fin_size = nlmeans_32_finish_spv_size;
        }

        auto make_module = [&](const uint32_t * code, size_t size,
                               VkShaderModule & mod) -> std::optional<std::string> {
            VkShaderModuleCreateInfo module_info {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = size,
                .pCode = code
            };
            if (vkCreateShaderModule(dev, &module_info, nullptr, &mod) != VK_SUCCESS) {
                return "vkCreateShaderModule failed"s;
            }
            return std::nullopt;
        };

        if (auto err = make_module(weight_code, weight_size, d->weight_module)) {
            return set_error(*err);
        }
        if (auto err = make_module(acc_code, acc_size, d->acc_module)) {
            return set_error(*err);
        }
        if (auto err = make_module(fin_code, fin_size, d->fin_module)) {
            return set_error(*err);
        }

        const std::pair<VkShaderModule, VkPipeline *> pipes[] {
            { d->weight_module, &d->weight_pipeline },
            { d->acc_module, &d->acc_pipeline },
            { d->fin_module, &d->fin_pipeline }
        };
        for (auto [module, pipeline] : pipes) {
            const auto result = create_pipeline(*d, spec, module, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            *pipeline = std::get<VkPipeline>(result);
        }
    }

    // sweep tables live in a small host-visible buffer shared by all streams
    {
        const VkDeviceSize wq_bytes = static_cast<VkDeviceSize>(d->wq_host.size()) * sizeof(int32_t);
        const VkDeviceSize aq_bytes = static_cast<VkDeviceSize>(d->aq_host.size()) * sizeof(int32_t);
        VkDeviceSize align = std::max<VkDeviceSize>(
            256, d->device->limits.minStorageBufferOffsetAlignment);
        d->aq_offset = (wq_bytes + align - 1) / align * align;
        const VkDeviceSize tables_bytes = d->aq_offset + aq_bytes;

        {
            const auto result = create_buffer(dev, tables_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->tables_buf = std::get<VkBuffer>(result);
        }
        if (auto err = bind_memory(*d, d->tables_buf, d->tables_mem,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
            return set_error(*err);
        }
        void * tables_map;
        if (vkMapMemory(dev, d->tables_mem, 0, tables_bytes, 0, &tables_map) != VK_SUCCESS) {
            return set_error("vkMapMemory failed");
        }
        memcpy(tables_map, d->wq_host.data(), wq_bytes);
        memcpy(static_cast<uint8_t *>(tables_map) + d->aq_offset,
            d->aq_host.data(), aq_bytes);
        vkUnmapMemory(dev, d->tables_mem);
    }

    // per-stream resources
    d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->streams.reserve(d->num_streams);

    uint32_t num_queues = std::min(
        d->num_streams, static_cast<int>(d->device->queue_count));

    const VkDeviceSize lay_bytes =
        static_cast<VkDeviceSize>(d->pstride) * d->ph * d->elem_bytes;
    const VkDeviceSize u1_bytes = lay_bytes * d->layers * d->channels;
    const int clips = d->has_ref ? 2 : 1;
    d->u1_bytes = u1_bytes;

    const VkDeviceSize npix_v = static_cast<VkDeviceSize>(d->npix);
    const VkDeviceSize u1z_bytes =
        (npix_v * d->channels * d->elem_bytes + 3) / 4 * 4;
    const VkDeviceSize u2_bytes = npix_v * (d->channels + 1) * sizeof(float);
    const VkDeviceSize u4a_bytes = npix_v * slots * sizeof(float);
    const VkDeviceSize u5_bytes = npix_v * sizeof(float);

    for (int i = 0; i < d->num_streams; ++i) {
        NLStream st;

        // staging mirror of the padded source (+guide) window
        {
            const auto result = create_buffer(dev, u1_bytes * clips,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.staging = std::get<VkBuffer>(result);
        }
        {
            const auto result = allocate_memory(*d->device, st.staging,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.staging_mem = std::get<AllocatedMemory>(result).memory;
            st.staging_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        if (vkMapMemory(dev, st.staging_mem, 0, u1_bytes * clips, 0,
                reinterpret_cast<void **>(&st.staging_map)) != VK_SUCCESS) {
            return set_error("vkMapMemory failed");
        }
        // the pad margin of every layer keeps its one-time-init zero
        memset(st.staging_map, 0, static_cast<size_t>(u1_bytes * clips));

        {
            const auto result = create_buffer(dev, u1_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u1 = std::get<VkBuffer>(result);
        }
        if (auto err = bind_memory(*d, st.u1, st.u1_mem,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return set_error(*err);
        }

        if (d->has_ref) {
            const auto result = create_buffer(dev, u1_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u1r = std::get<VkBuffer>(result);
            if (auto err = bind_memory(*d, st.u1r, st.u1r_mem,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                return set_error(*err);
            }
        }

        // result buffer, written by the finish kernel, downloaded by the host
        {
            const auto result = create_buffer(dev, u1z_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u1z = std::get<VkBuffer>(result);
        }
        {
            const auto result = allocate_memory(*d->device, st.u1z,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u1z_mem = std::get<AllocatedMemory>(result).memory;
            st.u1z_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        if (vkMapMemory(dev, st.u1z_mem, 0, u1z_bytes, 0,
                reinterpret_cast<void **>(&st.u1z_map)) != VK_SUCCESS) {
            return set_error("vkMapMemory failed");
        }

        {
            const auto result = create_buffer(dev, u2_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u2 = std::get<VkBuffer>(result);
        }
        if (auto err = bind_memory(*d, st.u2, st.u2_mem,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return set_error(*err);
        }

        {
            const auto result = create_buffer(dev, u4a_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u4a = std::get<VkBuffer>(result);
        }
        if (auto err = bind_memory(*d, st.u4a, st.u4a_mem,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return set_error(*err);
        }

        {
            const auto result = create_buffer(dev, u5_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            st.u5 = std::get<VkBuffer>(result);
        }
        if (auto err = bind_memory(*d, st.u5, st.u5_mem,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return set_error(*err);
        }

        {
            VkCommandPoolCreateInfo pool_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                /* per-frame recording re-begins the same command buffer */
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = d->device->queue_family
            };
            if (vkCreateCommandPool(dev, &pool_info, nullptr, &st.pool) != VK_SUCCESS) {
                return set_error("vkCreateCommandPool failed");
            }
        }
        {
            VkCommandBufferAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = st.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
            if (vkAllocateCommandBuffers(dev, &alloc_info, &st.cmd) != VK_SUCCESS) {
                return set_error("vkAllocateCommandBuffers failed");
            }
        }
        {
            VkFenceCreateInfo fence_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            if (vkCreateFence(dev, &fence_info, nullptr, &st.fence) != VK_SUCCESS) {
                return set_error("vkCreateFence failed");
            }
        }
        {
            VkDescriptorSetAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = d->desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &d->set_layout
            };
            if (vkAllocateDescriptorSets(dev, &alloc_info, &st.desc_set) != VK_SUCCESS) {
                return set_error("vkAllocateDescriptorSets failed");
            }

            VkDescriptorBufferInfo infos[8] {
                { st.u1, 0, VK_WHOLE_SIZE },
                { d->has_ref ? st.u1r : st.u1, 0, VK_WHOLE_SIZE },
                { st.u1z, 0, VK_WHOLE_SIZE },
                { st.u2, 0, VK_WHOLE_SIZE },
                { st.u4a, 0, VK_WHOLE_SIZE },
                { st.u5, 0, VK_WHOLE_SIZE },
                { d->tables_buf, 0,
                  static_cast<VkDeviceSize>(d->wq_host.size()) * sizeof(int32_t) },
                { d->tables_buf, d->aq_offset, VK_WHOLE_SIZE }
            };

            VkWriteDescriptorSet writes[8];
            for (uint32_t b = 0; b < 8; ++b) {
                writes[b] = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = st.desc_set,
                    .dstBinding = b,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &infos[b],
                    .pTexelBufferView = nullptr
                };
            }
            vkUpdateDescriptorSets(dev, 8, writes, 0, nullptr);
        }

        st.queue = d->device->queues[i % num_queues].queue;
        st.queue_lock = d->device->queues[i % num_queues].lock.get();

        d->streams.push_back(std::move(st));
    }

    NLMeansData * data = d.release();

    VSFilterDependency deps[2] = {
        { data->node, rpStrictSpatial },
        { data->ref_node, rpStrictSpatial }
    };

    vsapi->createVideoFilter(
        out, "NLMeans", data->vi,
        NLMeansGetFrame, NLMeansFree,
        fmParallel, deps, data->has_ref ? 2 : 1, data, core);
}

} // namespace

void vsfeel_register_nlmeans(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "NLMeans",
        "clip:vnode;"
        "d:int:opt;"
        "a:int:opt;"
        "s:int:opt;"
        "h:float:opt;"
        "wmode:int:opt;"
        "wref:float:opt;"
        "channels:data:opt;"
        "rclip:vnode:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode;",
        NLMeansCreate, nullptr, plugin
    );
}
