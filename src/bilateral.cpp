#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <numbers>
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

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------

struct PlaneConfig {
    int width {};                    // pixels
    int height {};                   // pixels
    int stride {};                   // pitch in elements (round_up(width, alignment))
    int pitch_bytes {};              // row pitch in bytes
    VkDeviceSize upload_offset {};   // bytes, offset within staging
    VkDeviceSize upload_size {};     // bytes, incl. ref plane
    VkDeviceSize download_offset {}; // bytes, offset within download area
    VkDeviceSize download_size {};   // bytes
    VkDeviceSize src_elem {};        // element offset into the VRAM source buffer
    VkDeviceSize dst_elem {};        // element offset into the VRAM destination buffer
    VkDeviceSize dst_stage_elem {};  // element offset of the download region in staging
    VkPipeline pipeline {};
    uint32_t grid_x {};
    uint32_t grid_y {};
};

struct BilateralResource {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    VkBuffer src_buf {};        // device-local input planes (VRAM)
    VkDeviceMemory src_mem {};
    void * src_map {};          // mapped VRAM window when the upload is host-direct
    uint32_t src_type_index {};
    VkBuffer dst_buf {};        // device-local output planes (VRAM)
    VkDeviceMemory dst_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    float * map {};
    uint32_t staging_type_index {};
};

struct BilateralData {
    VSNode * node;
    VSNode * ref_node;
    const VSVideoInfo * vi;

    int device_id, num_streams;
    int bits, elem_bytes;
    bool process[3] { true, true, true };

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule shared_module {};
    VkShaderModule plain_module {};
    VkDeviceSize upload_total {};
    VkDeviceSize download_total {};
    bool host_direct_upload {};  // src VRAM is host-mapped (ReBAR): no H2D copy
    bool kd_download {};         // kernels write the GTT download staging directly
    std::array<PlaneConfig, 3> planes {};
    FramePool<BilateralResource> pool;

    ~BilateralData() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);

        for (auto & resource : pool.items) {
            if (resource.map) {
                vkUnmapMemory(dev, resource.staging_mem);
            }
            if (resource.src_map) {
                vkUnmapMemory(dev, resource.src_mem);
            }
            if (resource.dst_mem) {
                vkFreeMemory(dev, resource.dst_mem, nullptr);
            }
            if (resource.dst_buf) {
                vkDestroyBuffer(dev, resource.dst_buf, nullptr);
            }
            if (resource.src_mem) {
                vkFreeMemory(dev, resource.src_mem, nullptr);
            }
            if (resource.src_buf) {
                vkDestroyBuffer(dev, resource.src_buf, nullptr);
            }
            destroy_common(dev, resource);
        }

        VkPipeline destroyed_pipelines[3] {};
        int num_destroyed = 0;
        for (auto & plane : planes) {
            if (!plane.pipeline) {
                continue;
            }
            bool already_destroyed = false;
            for (int i = 0; i < num_destroyed; ++i) {
                if (destroyed_pipelines[i] == plane.pipeline) {
                    already_destroyed = true;
                    break;
                }
            }
            if (already_destroyed) {
                continue;
            }
            destroyed_pipelines[num_destroyed++] = plane.pipeline;
            vkDestroyPipeline(dev, plane.pipeline, nullptr);
        }
        if (desc_pool) {
            vkDestroyDescriptorPool(dev, desc_pool, nullptr);
        }
        if (pipeline_layout) {
            vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
        if (shared_module) {
            vkDestroyShaderModule(dev, shared_module, nullptr);
        }
        if (plain_module) {
            vkDestroyShaderModule(dev, plain_module, nullptr);
        }

        release_device(device);
    }
};

// ---------------------------------------------------------------------------
// Shader specialization constants
// ---------------------------------------------------------------------------

struct SpecData {
    int32_t width;
    int32_t height;
    int32_t stride;
    float sigma_spatial_scaled;
    float sigma_color_scaled;
    int32_t radius;
    int32_t has_ref;
    int32_t tile_x;
    int32_t tile_y;
    int32_t shared_floats;
    int32_t block_x;
    int32_t block_y;
};

static constexpr std::array<VkSpecializationMapEntry, 12> shared_entries {{
    { 0,  0, sizeof(int32_t) },
    { 1,  4, sizeof(int32_t) },
    { 2,  8, sizeof(int32_t) },
    { 3, 12, sizeof(float) },
    { 4, 16, sizeof(float) },
    { 5, 20, sizeof(int32_t) },
    { 6, 24, sizeof(int32_t) },
    { 7, 28, sizeof(int32_t) },
    { 8, 32, sizeof(int32_t) },
    { 9, 36, sizeof(int32_t) },
    { 10, 40, sizeof(int32_t) },
    { 11, 44, sizeof(int32_t) },
}};

static constexpr std::array<VkSpecializationMapEntry, 9> plain_entries {{
    { 0,  0, sizeof(int32_t) },
    { 1,  4, sizeof(int32_t) },
    { 2,  8, sizeof(int32_t) },
    { 3, 12, sizeof(float) },
    { 4, 16, sizeof(float) },
    { 5, 20, sizeof(int32_t) },
    { 6, 24, sizeof(int32_t) },
    { 10, 40, sizeof(int32_t) },
    { 11, 44, sizeof(int32_t) },
}};

static std::variant<VkShaderModule, std::string> create_shader_module(
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

// use_shared selects between the two kernels
static std::variant<VkPipeline, std::string> create_pipeline(
    const VK_Device & dev, bool use_shared, const SpecData & spec,
    VkShaderModule module, VkPipelineLayout layout) {

    const VkSpecializationMapEntry * entries;
    size_t entry_count;
    if (use_shared) {
        entries = shared_entries.data();
        entry_count = std::size(shared_entries);
    } else {
        entries = plain_entries.data();
        entry_count = std::size(plain_entries);
    }

    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(entry_count),
        .pMapEntries = entries,
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

// Records the dispatch sequence for all planes into a single pre-recorded
// command buffer. The frame bytes are moved by two big DMA copies at the
// head and tail (staging upload area -> VRAM src, VRAM dst -> staging
// download area); the kernels then read and write device-local VRAM only
// (or host-mapped VRAM / GTT staging directly on the fast paths, with the
// corresponding copy skipped). All plane regions are disjoint, so the
// dispatches of different planes can overlap on the GPU.
static std::optional<std::string> record_command_buffer(
    const BilateralData & d, BilateralResource & resource) {

    VkDevice dev = d.device->device;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };

    if (vkBeginCommandBuffer(resource.cmd, &begin_info) != VK_SUCCESS) {
        return "vkBeginCommandBuffer failed";
    }

        // upload: staging -> VRAM src (one copy, all planes are contiguous).
        // Skipped when the src VRAM is host-mapped: the CPU memcpy of the
        // frame already landed the bytes in VRAM before submission.
        if (!d.host_direct_upload && d.upload_total > 0) {
            const VkBufferCopy upload_region { 0, 0, d.upload_total };
            vkCmdCopyBuffer(resource.cmd, resource.staging, resource.src_buf,
                1, &upload_region);
            VkMemoryBarrier copy_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &copy_barrier, 0, nullptr, 0, nullptr);
        }

        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
            continue;
        }
        const auto & cfg = d.planes[plane];

        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pipeline);
        vkCmdBindDescriptorSets(
            resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
        {
            // push constants are element offsets into the bound buffers;
            // kd_download: the output SSBO is the GTT staging (the kernels'
            // plain coalesced stores land straight in host memory); the dst
            // push constant then addresses the staging download region.
            int32_t push_constants[2] {
                static_cast<int32_t>(cfg.src_elem),
                static_cast<int32_t>(d.kd_download ? cfg.dst_stage_elem : cfg.dst_elem)
            };
            vkCmdPushConstants(
                resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(push_constants), push_constants);
        }
        vkCmdDispatch(resource.cmd, cfg.grid_x, cfg.grid_y, 1);
        if (std::getenv("BILATERAL_NODISPATCH")) {
            vkCmdDispatch(resource.cmd, 1, 1, 1);
        }
    }

        // download: VRAM dst -> staging (one copy, all planes contiguous).
        // Skipped for kd_download — the kernels already wrote the staging
        // download region directly (plain stores, no atomics).
        if (!d.kd_download && d.download_total > 0) {
            VkMemoryBarrier kernel_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &kernel_barrier, 0, nullptr, 0, nullptr);
            const VkBufferCopy download_region { 0, d.upload_total, d.download_total };
            vkCmdCopyBuffer(resource.cmd, resource.dst_buf, resource.staging,
                1, &download_region);
        }

    if (vkEndCommandBuffer(resource.cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer failed";
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static const VSFrame *VS_CC BilateralGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->ref_node) {
            vsapi->requestFrameFilter(n, d->ref_node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame * ref = nullptr;
        if (d->ref_node) {
            ref = vsapi->getFrameFilter(n, d->ref_node, frameCtx);
        }

        const int pl[] = { 0, 1, 2 };
        const VSFrame * fr[] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        VSFrame * dst = vsapi->newVideoFrame2(
            &d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        {
            // Host phase probe (VSFEEL_BILAT_TRACE): accumulated microsecond
            // stage timings, reported as averages every 200 frames. Zero
            // overhead when unset (no clock reads, no atomic traffic).
            static const bool trace = trace_on("VSFEEL_BILAT_TRACE");
            static std::atomic<uint64_t> t_up {}, t_wait {}, t_dl {}, t_bit {}, t_sub {}, t_acq {};
            static std::atomic<uint32_t> t_nf {};
            static std::atomic<int> t_inf {}, t_peak {};
            auto now_us = [] { return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count(); };
            auto t0 = trace ? now_us() : 0;
            auto mark = [&](std::atomic<uint64_t> & acc) {
                if (trace) {
                    acc += now_us() - t0; t0 = now_us();
                }
            };
            auto dump = [&] {
                if (!trace) {
                    return;
                }
                uint32_t nf = t_nf.load();
                if (nf > 0 && nf % 200 == 0) {
                    fprintf(stderr, "[perf] frames=%u inf=%d acq=%.2f up=%.2f sub=%.2f wait=%.2f bit=%.2f (ms avg)\n",
                        nf, t_peak.load(),
                        t_acq.load()/double(nf)/1e3,
                        t_up.load()/double(nf)/1e3, t_sub.load()/double(nf)/1e3,
                        t_wait.load()/double(nf)/1e3,
                        t_bit.load()/double(nf)/1e3);
                }
            };

            auto t_acq0 = trace ? now_us() : 0;
            auto resource = d->pool.take();
            if (trace) {
                t_acq += now_us() - t_acq0;
            }
            // reset the stage clock after the acquire: the take() wait is
            // accounted in t_acq and must not leak into t_up
            if (trace) {
                t0 = now_us();
            }
            int peak = 0;
            if (trace) {
                int inf = t_inf.fetch_add(1) + 1;
                peak = t_peak.load();
                while (inf > peak && !t_peak.compare_exchange_weak(peak, inf)) {}
            }

        auto set_error = [&](const std::string & error_message) {
            d->pool.give_back(std::move(resource));
            vsapi->setFilterError(("BilateralVK: " + error_message).c_str(), frameCtx);
            if (d->ref_node) {
                vsapi->freeFrame(ref);
            }
            vsapi->freeFrame(src);
            return nullptr;
        };

        VkDevice dev = d->device->device;
        float * map = resource.map;

        const bool coherent =
            !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        const bool nocpu = std::getenv("BILATERAL_NOCPU") != nullptr;
        const bool nodl = std::getenv("BILATERAL_NODL") != nullptr;

        // the upload target is either the host-mapped VRAM src window
        // (host-direct path: the bytes land in VRAM with no GPU copy) or the
        // staging upload area (the command buffer's H2D copy moves them)
        uint8_t * const upload_base = d->host_direct_upload
            ? static_cast<uint8_t *>(resource.src_map)
            : static_cast<uint8_t *>(static_cast<void *>(map));

        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }

            int height = vsapi->getFrameHeight(src, plane);
            int s_pitch = vsapi->getStride(src, plane);
            const auto & cfg = d->planes[plane];

            auto srcp = vsapi->getReadPtr(src, plane);
            auto dstp = upload_base + cfg.upload_offset;

            // raw byte copy of the plane (staging/VRAM layout matches the
            // frame pitch); plain memcpy into the mapped VRAM window (the
            // window is write-combined: streaming stores measured slower),
            // streaming stores into GTT staging (keeps the lines clean in
            // DRAM so the GPU does not pay snoop/writeback stalls)
            if (!nocpu) {
                const auto bytes = static_cast<size_t>(cfg.width * d->elem_bytes) * height;
                if (d->host_direct_upload) {
                    memcpy(dstp, srcp, bytes);
                } else {
                    copy_stream_out(dstp, srcp, bytes);
                }

                // reference plane goes directly below the source plane
                if (d->ref_node) {
                    auto refp = vsapi->getReadPtr(ref, plane);
                    copy_stream_out(dstp + static_cast<int64_t>(height) * cfg.pitch_bytes, refp, bytes);
                }
            }
        }

        mark(t_up);

        if (!coherent && !nocpu) {
            std::vector<VkMappedMemoryRange> ranges;
            ranges.reserve(d->vi->format.numPlanes * 2);
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }
                const auto & cfg = d->planes[plane];
                ranges.push_back(VkMappedMemoryRange {
                    .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                    .pNext = nullptr,
                    .memory = resource.staging_mem,
                    .offset = cfg.upload_offset,
                    .size = cfg.upload_size,
                });
            }
            checkVK(vkFlushMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
        }

        checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
            resource.cmd, resource.fence));
        mark(t_sub);

        checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));
        mark(t_wait);

        if (!coherent && !nocpu) {
            std::vector<VkMappedMemoryRange> ranges;
            ranges.reserve(d->vi->format.numPlanes);
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }
                const auto & cfg = d->planes[plane];
                ranges.push_back(VkMappedMemoryRange {
                    .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                    .pNext = nullptr,
                    .memory = resource.staging_mem,
                    .offset = d->upload_total + cfg.download_offset,
                    .size = cfg.download_size,
                });
            }
            checkVK(vkInvalidateMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
        }

        if (!nocpu) {
        if (!nodl) {
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }

                int width = vsapi->getFrameWidth(src, plane);
                int height = vsapi->getFrameHeight(src, plane);
                int s_pitch = vsapi->getStride(src, plane);
                const auto & cfg = d->planes[plane];

                auto dstp = vsapi->getWritePtr(dst, plane);
                const uint8_t * h_bufferp =
                    static_cast<const uint8_t *>(static_cast<const void *>(map)) + d->upload_total + cfg.download_offset;

                // raw byte copy of the plane (staging layout matches the frame pitch)
                copy_stream_read(dstp, h_bufferp,
                    static_cast<size_t>(cfg.width * d->elem_bytes) * height);
            }
        }
        }

        mark(t_bit);
        mark(t_dl);
        if (trace) {
            t_inf.fetch_sub(1);
            t_nf.fetch_add(1);
            dump();
        }

        d->pool.give_back(std::move(resource));
        }

        if (d->ref_node) {
            vsapi->freeFrame(ref);
        }
        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BilateralFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC BilateralCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BilateralData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
    bool has_ref = d->ref_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("BilateralVK: " + error_message).c_str());
        if (has_ref) {
            vsapi->freeNode(d->ref_node);
        }
        vsapi->freeNode(d->node);
    };

    if (auto [bps, sample] = std::pair{
            d->vi->format.bitsPerSample,
            d->vi->format.sampleType
        };
        !vsh::isConstantVideoFormat(d->vi) ||
        (sample == stInteger && bps != 16) ||
        (sample == stFloat && bps != 32)
    ) {

        return set_error("input bitdepth must be 16 (integer) or 32 (float).");
    }

    d->bits = d->vi->format.bitsPerSample;
    d->elem_bytes = d->bits / 8;

    const auto ref_vi = vsapi->getVideoInfo(d->ref_node);
    if (d->ref_node && (!vsh::isSameVideoInfo(d->vi, ref_vi) || d->vi->numFrames != ref_vi->numFrames)) {
        return set_error("\"ref\" must be of the same format as \"clip\"");
    }

    std::array<float, 3> sigma_spatial;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial[i] = static_cast<float>(
            vsapi->mapGetFloat(in, "sigma_spatial", i, &error));

        if (error) {
            if (i == 0) {
                sigma_spatial[i] = 3.0f;
            } else if (i == 1) {
                auto subH = d->vi->format.subSamplingH;
                auto subW = d->vi->format.subSamplingW;
                sigma_spatial[i] = static_cast<float>(
                    sigma_spatial[0] / std::sqrt((1 << subH) * (1 << subW)));
            } else {
                sigma_spatial[i] = sigma_spatial[i - 1];
            }
        } else if (sigma_spatial[i] < 0.f) {
            return set_error("\"sigma_spatial\" must be non-negative");
        }

        if (sigma_spatial[i] < FLT_EPSILON) {
            d->process[i] = false;
        }
    }

    std::array<float, 3> sigma_spatial_scaled;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial_scaled[i] = (-0.5f / (sigma_spatial[i] * sigma_spatial[i])) * std::numbers::log2e_v<float>;
    }

    std::array<float, 3> sigma_color;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        sigma_color[i] = static_cast<float>(
            vsapi->mapGetFloat(in, "sigma_color", i, &error));

        if (error) {
            if (i == 0) {
                sigma_color[i] = 0.02f;
            } else {
                sigma_color[i] = sigma_color[i - 1];
            }
        } else if (sigma_color[i] < 0.f) {
            return set_error("\"sigma_color\" must be non-negative");
        }
    }

    std::array<float, 3> sigma_color_scaled;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            sigma_color_scaled[i] = (-0.5f / (sigma_color[i] * sigma_color[i])) * std::numbers::log2e_v<float>;
        }
    }

    std::array<int, 3> radius;
    for (int i = 0; i < std::ssize(radius); ++i) {
        radius[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", i, &error));

        if (error) {
            radius[i] = std::max(1, static_cast<int>(std::roundf(sigma_spatial[i] * 3.f)));
        } else if (radius[i] <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    d->num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 4;
    }
    if (d->num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

    bool use_shared_memory = !!vsapi->mapGetInt(in, "use_shared_memory", 0, &error);
    if (error) {
        use_shared_memory = true;
    }

    int block_x = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_x", 0, &error));
    const bool block_x_default = !!error;
    int block_y = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_y", 0, &error));
    const bool block_y_default = !!error;
    if (block_x_default || block_y_default) {
        // Auto-tuned workgroup shape from the benchmark matrix (RX 7900 XTX,
        // ns=4 real-clip GRAY16 medians): tall blocks hide the exp-pipeline
        // latency of large windows (R=24: 16x16 = 343 fps vs 16x8 = 228),
        // while wide blocks are marginally better for small windows
        // (R=9: 32x8 = 1742 vs 16x16 = 1705; R=3: 2449 vs 2412).
        const int max_radius = std::max({ radius[0], radius[1], radius[2] });
        if (block_x_default) {
            block_x = 32;
        }
        if (block_y_default) {
            block_y = (max_radius > 12) ? 16 : 8;
        }
    }

    {
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
        d->device_id = device_id;
    }

    {
        const VkPhysicalDeviceLimits & limits = d->device->limits;

        // shrink the default block size if the device cannot host it
        if (block_x > limits.maxComputeWorkGroupSize[0] ||
            block_y > limits.maxComputeWorkGroupSize[1] ||
            static_cast<uint32_t>(block_x) * block_y > limits.maxComputeWorkGroupInvocations) {
            block_x = std::min<int>(16, limits.maxComputeWorkGroupSize[0]);
            block_y = std::min<int>(16, limits.maxComputeWorkGroupSize[1]);
        }

        if (block_x <= 0 || block_y <= 0 ||
            static_cast<uint32_t>(block_x) > limits.maxComputeWorkGroupSize[0] ||
            static_cast<uint32_t>(block_y) > limits.maxComputeWorkGroupSize[1] ||
            static_cast<uint32_t>(block_x) * block_y > limits.maxComputeWorkGroupInvocations) {
            return set_error("invalid \"block_x\"/\"block_y\" for this device");
        }
    }

    // Pipeline layouts and descriptor pool
    {
        VkDescriptorSetLayoutBinding bindings[2] {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };

        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 2,
            .pBindings = bindings
        };

        checkVK(vkCreateDescriptorSetLayout(
            d->device->device, &layout_info, nullptr, &d->set_layout));
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

        checkVK(vkCreatePipelineLayout(
            d->device->device, &pipeline_layout_info, nullptr, &d->pipeline_layout));
    }
    {
        VkDescriptorPoolSize pool_size {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * static_cast<uint32_t>(d->num_streams)
        };

        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(d->num_streams),
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size
        };

        checkVK(vkCreateDescriptorPool(
            d->device->device, &pool_info, nullptr, &d->desc_pool));
    }

    int width = d->vi->width;
    int height = d->vi->height;
    int ssw = d->vi->format.subSamplingW;
    int ssh = d->vi->format.subSamplingH;

    // Per-plane pipeline configuration, with deduplication for identical planes
    struct PipelineKey {
        int width;
        int height;
        int stride;
        float ss;
        float sc;
        int radius;

        bool operator==(const PipelineKey & other) const noexcept {
            return width == other.width && height == other.height && stride == other.stride &&
                   ss == other.ss && sc == other.sc && radius == other.radius;
        }
    };
    std::array<PipelineKey, 3> pipeline_keys {};
    std::array<bool, 3> pipeline_valid {};
    std::array<bool, 3> plane_shared {};

    std::array<PlaneConfig, 3> & planes = d->planes;
    bool need_plain = false;

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }

        int plane_width { plane == 0 ? width : width >> ssw };
        int plane_height { plane == 0 ? height : height >> ssh };
        int pitch_bytes = (plane_width * d->elem_bytes + 15) & ~15;
        int stride = pitch_bytes / d->elem_bytes;

        pipeline_keys[plane] = {
            plane_width, plane_height, stride,
            sigma_spatial_scaled[plane], sigma_color_scaled[plane], radius[plane]
        };
        pipeline_valid[plane] = true;
    }

    VkDevice dev = d->device->device;

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!pipeline_valid[plane]) {
            continue;
        }

        auto & cfg = planes[plane];
        const auto & key = pipeline_keys[plane];

        cfg.width = key.width;
        cfg.height = key.height;
        cfg.stride = key.stride;
        cfg.pitch_bytes = cfg.stride * d->elem_bytes;

        const uint32_t max_grid_x = d->device->limits.maxComputeWorkGroupCount[0];
        const uint32_t max_grid_y = d->device->limits.maxComputeWorkGroupCount[1];

        cfg.grid_x = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.width - 1) / block_x + 1, static_cast<int64_t>(max_grid_x)));
        cfg.grid_y = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.height - 1) / block_y + 1, static_cast<int64_t>(max_grid_y)));

        const int tile_x = 2 * key.radius + block_x;
        const int tile_y = 2 * key.radius + block_y;
        // the shared kernel keeps the source tile(s) plus the output tile
        const size_t shared_bytes =
            static_cast<size_t>(2 + has_ref) * tile_x * tile_y * sizeof(float);

        bool use_shared = use_shared_memory &&
            shared_bytes <= std::min<size_t>(48 * 1024, d->device->limits.maxComputeSharedMemorySize);
        plane_shared[plane] = use_shared;
    }

    // Shader modules (create lazily — only those actually used)
    {
        const uint32_t * shared_code = nullptr;
        size_t shared_size = 0;
        const uint32_t * plain_code = nullptr;
        size_t plain_size = 0;
        switch (d->bits) {
            case 16:
                shared_code = bilateral_shared_16_spv; shared_size = bilateral_shared_16_spv_size;
                plain_code = bilateral_plain_16_spv; plain_size = bilateral_plain_16_spv_size;
                break;
            case 32:
                shared_code = bilateral_shared_32_spv; shared_size = bilateral_shared_32_spv_size;
                plain_code = bilateral_plain_32_spv; plain_size = bilateral_plain_32_spv_size;
                break;
            default:
                return set_error("unsupported bit depth");
        }

        bool need_shared = false;
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (pipeline_valid[plane]) {
                need_shared |= plane_shared[plane];
                need_plain |= !plane_shared[plane];
            }
        }
        if (need_shared) {
            auto result = create_shader_module(*d->device, shared_code, shared_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->shared_module = std::get<VkShaderModule>(result);
        }
        if (need_plain) {
            auto result = create_shader_module(*d->device, plain_code, plain_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->plain_module = std::get<VkShaderModule>(result);
        }
    }

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!pipeline_valid[plane]) {
            continue;
        }

        auto & cfg = planes[plane];
        const auto & key = pipeline_keys[plane];

        // reuse an existing pipeline for identical plane configurations
        bool found = false;
        for (int other = 0; other < plane; ++other) {
            if (pipeline_valid[other] && pipeline_keys[other] == key) {
                cfg.pipeline = planes[other].pipeline;
                found = true;
                break;
            }
        }
        if (found) {
            continue;
        }

        const int tile_x = 2 * key.radius + block_x;
        const int tile_y = 2 * key.radius + block_y;
        // the shared kernel keeps the source tile(s) plus the output tile
        const size_t shared_bytes =
            static_cast<size_t>(2 + has_ref) * tile_x * tile_y * sizeof(float);

        SpecData spec {
            .width = key.width,
            .height = key.height,
            .stride = key.stride,
            .sigma_spatial_scaled = key.ss,
            .sigma_color_scaled = key.sc,
            .radius = key.radius,
            .has_ref = has_ref,
            .tile_x = tile_x,
            .tile_y = tile_y,
            .shared_floats = static_cast<int32_t>(shared_bytes / sizeof(float)),
            .block_x = block_x,
            .block_y = block_y
        };

        const auto result = create_pipeline(
            *d->device, plane_shared[plane], spec,
            plane_shared[plane] ? d->shared_module : d->plain_module,
            d->pipeline_layout);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        cfg.pipeline = std::get<VkPipeline>(result);
    }

    // Buffer region offsets (raw bytes), each region 32-byte aligned so the
    // streaming copies can use aligned loads/stores. The staging layout and
    // the VRAM src/dst layouts are identical, so one DMA copy per direction
    // moves every plane at once.
    auto align32 = [](VkDeviceSize v) { return (v + 31) & ~VkDeviceSize(31); };

    VkDeviceSize upload_total = 0;
    VkDeviceSize download_total = 0;

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!pipeline_valid[plane]) {
            continue;
        }

        auto & cfg = planes[plane];
        VkDeviceSize plane_bytes =
            static_cast<VkDeviceSize>(cfg.height) * cfg.pitch_bytes;

        cfg.upload_offset = align32(upload_total);
        cfg.upload_size = (1 + has_ref) * plane_bytes;
        upload_total = align32(cfg.upload_offset + cfg.upload_size);

        cfg.download_offset = align32(download_total);
        cfg.download_size = plane_bytes;
        download_total = align32(cfg.download_offset + cfg.download_size);

        // the VRAM buffers mirror the staging layout byte-for-byte
        cfg.src_elem = cfg.upload_offset / d->elem_bytes;
        cfg.dst_elem = cfg.download_offset / d->elem_bytes;
    }

    d->upload_total = upload_total;
    d->download_total = download_total;

    // kd download addresses the staging download area, which begins at the
    // FINAL upload_total — only known now, after every plane was laid out
    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!pipeline_valid[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        cfg.dst_stage_elem = (upload_total + cfg.download_offset) / d->elem_bytes;
    }

    const VkDeviceSize min_size = 4;
    // the host-visible staging only carries raw plane bytes; the kernels'
    // src/dst live in device-local VRAM (see below)
    const VkDeviceSize staging_size = std::max(upload_total + download_total, 2 * min_size);
    const VkDeviceSize src_size = std::max(upload_total, min_size);
    const VkDeviceSize dst_size = std::max(download_total, min_size);

    // Resources
    // host-direct upload: the CPU memcpy writes the host-mapped VRAM src
    // window directly (no GPU-side H2D copy); opt out with VSFEEL_BILAT_HD=0
    // (plain VRAM src + staging upload + in-CB copy)
    d->host_direct_upload = !getenv("VSFEEL_BILAT_HD") || atoi(getenv("VSFEEL_BILAT_HD")) != 0;
    // kernel-direct download: the bilateral kernels' plain coalesced stores
    // write the GTT staging download region over PCIe directly, removing the
    // GPU-side D2H copy; opt out with VSFEEL_BILAT_KD=0 for the VRAM+copy path
    d->kd_download = !getenv("VSFEEL_BILAT_KD") || atoi(getenv("VSFEEL_BILAT_KD")) != 0;
    d->pool.semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->pool.reserve(d->num_streams);

    // Two queues feed the GPU with no idle bubbles: with one stream per
    // queue (num_queues == num_streams) each queue drains while its worker
    // does the post-fence CPU work (download memcpy + VS bookkeeping +
    // next-frame upload) before the next submit; sharing a queue across
    // streams keeps a next CB queued (ns=4: 2 queues = 1986 fps vs 4 queues
    // = 1709 fps). Beyond 2 the gains stop (lock/CP overhead). Override
    // with VSFEEL_BILAT_QUEUES=N for tuning.
    uint32_t num_queues = std::min({
        d->num_streams, static_cast<int>(d->device->queue_count), 2 });
    if (const char * qn = std::getenv("VSFEEL_BILAT_QUEUES")) {
        const int q = atoi(qn);
        if (q > 0) {
            num_queues = std::min<uint32_t>(
                static_cast<uint32_t>(d->num_streams),
                std::min<uint32_t>(static_cast<uint32_t>(q),
                    d->device->queue_count));
        }
    }

    for (int i = 0; i < d->num_streams; ++i) {
        BilateralResource resource;

        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
                // STORAGE_BUFFER: with kd_download the bilateral kernels write
                // the download region as an SSBO; TRANSFER_*: the H2D/D2H
                // copies of the non-host-direct paths
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.staging));
        }

        {
            const auto result = allocate_memory(
                *d->device, resource.staging,
                std::getenv("BILATERAL_NOCPU")
                    ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.staging_mem = std::get<AllocatedMemory>(result).memory;
            resource.staging_type_index = std::get<AllocatedMemory>(result).type_index;
        }

        // device-local input planes: the kernels read them. With ReBAR the
        // buffer is host-mapped so the CPU memcpy writes VRAM directly and
        // the command buffer needs no H2D copy; if no host-visible device-
        // local memory exists, fall back to a plain VRAM buffer filled by
        // the H2D copy.
        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = src_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.src_buf));

            auto result = allocate_memory(
                *d->device, resource.src_buf,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (std::holds_alternative<std::string>(result)) {
                result = allocate_memory(
                    *d->device, resource.src_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
            } else if (d->host_direct_upload) {
                // only take the host-mapped path when the allocation really
                // is device-local (allocate_memory may relax the requirement)
                const uint32_t ti = std::get<AllocatedMemory>(result).type_index;
                const auto flags = d->device->mem_props.memoryTypes[ti].propertyFlags;
                d->host_direct_upload =
                    (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                    (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                    (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            resource.src_mem = std::get<AllocatedMemory>(result).memory;
            resource.src_type_index = std::get<AllocatedMemory>(result).type_index;
            if (d->host_direct_upload) {
                checkVK(vkMapMemory(dev, resource.src_mem, 0, src_size, 0, &resource.src_map));
            }
        }

        // device-local output planes: the kernels write them, the D2H DMA
        // copy reads them. Not needed when the kernels write the staging
        // download region directly (kd_download).
        if (!d->kd_download) {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = dst_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.dst_buf));

            const auto result = allocate_memory(
                *d->device, resource.dst_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.dst_mem = std::get<AllocatedMemory>(result).memory;
        }

        {
            VkCommandPoolCreateInfo pool_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = d->device->queue_family
            };
            checkVK(vkCreateCommandPool(dev, &pool_info, nullptr, &resource.pool));
        }

        {
            VkCommandBufferAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = resource.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
            checkVK(vkAllocateCommandBuffers(dev, &alloc_info, &resource.cmd));
        }

        {
            VkFenceCreateInfo fence_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            checkVK(vkCreateFence(dev, &fence_info, nullptr, &resource.fence));
        }

        {
            VkDescriptorSetAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = d->desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &d->set_layout
            };
            checkVK(vkAllocateDescriptorSets(dev, &alloc_info, &resource.desc_set));
        }

        {
            VkDescriptorBufferInfo src_info {
                .buffer = resource.src_buf,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo dst_info {
                .buffer = d->kd_download ? resource.staging : resource.dst_buf,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };

            VkWriteDescriptorSet writes[2] {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &src_info,
                    .pTexelBufferView = nullptr
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &dst_info,
                    .pTexelBufferView = nullptr
                },
            };

            vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);
        }

        if (!std::getenv("BILATERAL_NOCPU")) {
            checkVK(vkMapMemory(dev, resource.staging_mem, 0, staging_size, 0, reinterpret_cast<void **>(&resource.map)));
        } else {
            resource.map = nullptr;
        }

        resource.queue = d->device->queues[i % num_queues].queue;
        resource.queue_lock = d->device->queues[i % num_queues].lock.get();

        if (const auto err = record_command_buffer(*d, resource)) {
            return set_error(*err);
        }

        d->pool.push(std::move(resource));
    }

    VSFilterDependency deps[2] = {{d->node, rpStrictSpatial}};
    int num_deps = 1;
    if (has_ref) {
        deps[1].source = d->ref_node;
        deps[1].requestPattern = rpStrictSpatial;
        num_deps = 2;
    }

    BilateralData *data = d.release();

    vsapi->createVideoFilter(
        out, "Bilateral", data->vi,
        BilateralGetFrame, BilateralFree,
        fmParallel, deps, num_deps, data, core);
}


// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bilateral(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "Bilateral",
        "clip:vnode;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "use_shared_memory:int:opt;"
        "block_x:int:opt;"
        "block_y:int:opt;"
        "ref:vnode:opt;",
        "clip:vnode;",
        BilateralCreate, nullptr, plugin
    );
}
