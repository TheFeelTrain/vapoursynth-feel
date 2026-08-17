#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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

// Work-group / code-path constants matching the reference implementations
// (vszipcl gaussglur.zig and vszipcu gaussblur.zig).
constexpr int BLK_X = 16;
constexpr int BLK_Y = 8;
constexpr int VRT = 3;         // output rows per thread, small path
constexpr int LARGE_R = 8;     // outputs per thread, large path
constexpr int LARGE_THRESHOLD = 32;  // radius <= 32 => fused small path

struct PlaneConfig {
    int width {};                    // pixels
    int height {};                   // pixels
    int stride {};                   // pitch in elements (round_up(width, 16b/elem))
    int pitch_bytes {};              // row pitch in bytes
    int ksize {};                    // kernel taps
    int radius {};                   // ksize / 2
    bool small {};                   // fused small path vs two-pass large path
    VkPipeline pipeline {};          // small path (gauss_blur entry)
    VkPipeline v_pipeline {};        // large path vertical pass
    VkPipeline h_pipeline {};        // large path horizontal pass
    uint32_t grid_x {};
    uint32_t grid_y {};              // small path dispatch
    uint32_t v_grid_x {};
    uint32_t v_grid_y {};            // large vertical dispatch
    uint32_t h_grid_x {};
    uint32_t h_grid_y {};            // large horizontal dispatch
    VkDeviceSize upload_offset {};   // bytes, offset within staging
    VkDeviceSize upload_size {};     // bytes
    VkDeviceSize download_offset {}; // bytes, offset within download area
    VkDeviceSize download_size {};   // bytes
    VkDeviceSize tmp_offset {};      // bytes, offset within tmp area (large path)
    VkDeviceSize tmp_size {};        // bytes
    uint32_t wt_base {};             // float element offset into the weights buffer
};

struct VK_Resource {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    float * map {};
    uint32_t staging_type_index {};
};

struct GaussData {
    VSNode * node;
    const VSVideoInfo * vi;

    int device_id, num_streams;
    int bits, elem_bytes;
    bool half {};
    bool process[3] { true, true, true };

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule module {};      // fused small path (gauss entry)
    VkShaderModule v_module {};    // large path vertical pass
    VkShaderModule h_module {};    // large path horizontal pass

    // shared, constant weights buffer (all configs' kernels concatenated)
    VkBuffer wt_buf {};
    VkDeviceMemory wt_mem {};
    float * wt_map {};
    uint32_t wt_type_index {};
    VkDeviceSize wt_bytes {};

    VkDeviceSize upload_total {};
    VkDeviceSize download_total {};
    VkDeviceSize tmp_total {};
    bool need_fill {};
    std::array<PlaneConfig, 3> planes {};
    ticket_semaphore semaphore;
    std::vector<VK_Resource> resources;
    std::mutex resources_lock;

    ~GaussData() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);

        for (auto & resource : resources) {
            if (resource.map) {
                vkUnmapMemory(dev, resource.staging_mem);
            }
            if (resource.staging_mem) {
                vkFreeMemory(dev, resource.staging_mem, nullptr);
            }
            if (resource.staging) {
                vkDestroyBuffer(dev, resource.staging, nullptr);
            }
            if (resource.cmd) {
                vkFreeCommandBuffers(dev, resource.pool, 1, &resource.cmd);
            }
            if (resource.pool) {
                vkDestroyCommandPool(dev, resource.pool, nullptr);
            }
            if (resource.fence) {
                vkDestroyFence(dev, resource.fence, nullptr);
            }
        }

        if (wt_map) {
            vkUnmapMemory(dev, wt_mem);
        }
        if (wt_mem) {
            vkFreeMemory(dev, wt_mem, nullptr);
        }
        if (wt_buf) {
            vkDestroyBuffer(dev, wt_buf, nullptr);
        }

        VkPipeline destroyed_pipelines[9] {};
        int num_destroyed = 0;
        for (auto & plane : planes) {
            const VkPipeline pipelines[3] { plane.pipeline, plane.v_pipeline, plane.h_pipeline };
            for (int p = 0; p < 3; ++p) {
                if (!pipelines[p]) {
                    continue;
                }
                bool already_destroyed = false;
                for (int i = 0; i < num_destroyed; ++i) {
                    if (destroyed_pipelines[i] == pipelines[p]) {
                        already_destroyed = true;
                        break;
                    }
                }
                if (already_destroyed) {
                    continue;
                }
                destroyed_pipelines[num_destroyed++] = pipelines[p];
                vkDestroyPipeline(dev, pipelines[p], nullptr);
            }
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
        if (module) {
            vkDestroyShaderModule(dev, module, nullptr);
        }
        if (v_module) {
            vkDestroyShaderModule(dev, v_module, nullptr);
        }
        if (h_module) {
            vkDestroyShaderModule(dev, h_module, nullptr);
        }

        release_device(device);
    }
};

// ---------------------------------------------------------------------------
// Gaussian kernel (matches the reference getGaussKernel bit-for-bit)
// ---------------------------------------------------------------------------

// The reference builds the half-kernel in f64 but coerces `sigma` to f32 for
// the two scale factors (`1/(sqrt(2*pi)*sigma)` and `2*sigma*sigma`), so the
// float32 rounding of those factors is reproduced here.
static std::vector<float> get_gauss_kernel(float sigma) {
    int taps = static_cast<int>(std::ceil(sigma * 6.0f + 1.0f));
    if (taps % 2 == 0) {
        taps += 1;
    }

    const int half_taps = taps / 2;
    const float factor = 1.0f / (static_cast<float>(std::sqrt(2.0 * M_PI)) * sigma);

    std::vector<double> kernel;
    kernel.reserve(half_taps);
    for (int x = 0; x < half_taps; ++x) {
        const double xd = static_cast<double>(x);
        const float denom = (2.0f * sigma) * sigma;
        const double value = static_cast<double>(factor) *
            std::exp(-(xd * xd) / static_cast<double>(denom));
        kernel.push_back(value);
    }

    const double first_value = kernel[0];
    for (size_t i = 1; i < kernel.size(); ++i) {
        kernel[i] *= 1.0 / first_value;
    }
    kernel[0] = 1.0;

    std::vector<double> full_kernel;
    full_kernel.reserve(taps);
    for (int i = static_cast<int>(kernel.size()) - 1; i >= 0; --i) {
        full_kernel.push_back(kernel[i]);
    }
    full_kernel.insert(full_kernel.end(), kernel.begin() + 1, kernel.end());

    double sum = 0.0;
    for (double v : full_kernel) {
        sum += v;
    }

    std::vector<float> out;
    out.reserve(full_kernel.size());
    for (double f : full_kernel) {
        out.push_back(static_cast<float>(f / sum));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

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

static std::variant<VkPipeline, std::string> create_pipeline(
    const VK_Device & dev, const PlaneConfig & cfg,
    VkShaderModule module, VkPipelineLayout layout) {

    struct Spec {
        int32_t width, height, stride, ksize, radius;
    } spec { cfg.width, cfg.height, cfg.stride, cfg.ksize, cfg.radius };

    const std::array<VkSpecializationMapEntry, 5> entries {{
        { 0,  0, sizeof(int32_t) },
        { 1,  4, sizeof(int32_t) },
        { 2,  8, sizeof(int32_t) },
        { 3, 12, sizeof(int32_t) },
        { 4, 16, sizeof(int32_t) },
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

// Records the dispatch sequence for all planes into a single pre-recorded
// command buffer (all plane regions are disjoint, so the dispatches of
// different planes can overlap on the GPU). The kernel reads its input and
// writes its output straight from/to the staging buffer; the large path also
// uses a float tmp region within the same buffer.
static std::optional<std::string> record_command_buffer(
    const GaussData & d, VK_Resource & resource) {

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

        // the 8/16-bit io kernels use masked atomicOr for sub-dword stores, so
        // their download regions must be cleared before every submission; the
        // 32-bit kernel packs complete dwords and needs no clearing
        if (d.need_fill) {
            vkCmdFillBuffer(resource.cmd, resource.staging,
                d.upload_total, d.download_total, 0);
            {
                VkMemoryBarrier mem_barrier {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                };
                vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
            }
        }

        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
            if (!d.process[plane]) {
                continue;
            }
            const auto & cfg = d.planes[plane];

            // push constants are RAW BYTE offsets (the kernel converts to dword
            // indices itself); wt_base is a float element offset
            const int32_t push_constants[4] {
                static_cast<int32_t>(cfg.upload_offset),
                static_cast<int32_t>(d.upload_total + cfg.download_offset),
                static_cast<int32_t>(d.upload_total + d.download_total + cfg.tmp_offset),
                static_cast<int32_t>(cfg.wt_base)
            };

            if (cfg.small) {
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
                vkCmdPushConstants(
                    resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(push_constants), push_constants);
                vkCmdDispatch(resource.cmd, cfg.grid_x, cfg.grid_y, 1);
            } else {
                // vertical pass: src -> float tmp
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.v_pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
                vkCmdPushConstants(
                    resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(push_constants), push_constants);
                vkCmdDispatch(resource.cmd, cfg.v_grid_x, cfg.v_grid_y, 1);

                // the horizontal pass reads the vertical pass' writes
                {
                    VkMemoryBarrier mem_barrier {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                    };
                    vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
                }

                // horizontal pass: float tmp -> dst
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.h_pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
                vkCmdPushConstants(
                    resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(push_constants), push_constants);
                vkCmdDispatch(resource.cmd, cfg.h_grid_x, cfg.h_grid_y, 1);
            }
        }

    if (vkEndCommandBuffer(resource.cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer failed";
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static const VSFrame *VS_CC GaussGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    GaussData * d = static_cast<GaussData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);

        const int pl[] = { 0, 1, 2 };
        const VSFrame * fr[] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        VSFrame * dst = vsapi->newVideoFrame2(
            &d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        d->semaphore.acquire();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        auto set_error = [&](const std::string & error_message) {
            d->resources_lock.lock();
            d->resources.push_back(std::move(resource));
            d->resources_lock.unlock();
            d->semaphore.release();
            vsapi->setFilterError(("GaussBlur: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(src);
            return nullptr;
        };

        VkDevice dev = d->device->device;
        float * map = resource.map;

        const bool coherent =
            !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }

            int height = vsapi->getFrameHeight(src, plane);
            const auto & cfg = d->planes[plane];

            auto srcp = vsapi->getReadPtr(src, plane);
            auto dstp = static_cast<uint8_t *>(static_cast<void *>(map)) + cfg.upload_offset;

            // raw byte copy of the plane (staging layout matches the frame pitch);
            // non-temporal stores keep the lines clean in DRAM so the GPU does
            // not pay snoop/writeback stalls when it reads them
            const auto bytes = static_cast<size_t>(cfg.width * d->elem_bytes) * height;
            copy_stream_out(dstp, srcp, bytes);
        }

        if (!coherent) {
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
                    .offset = cfg.upload_offset,
                    .size = cfg.upload_size,
                });
            }
            checkVK(vkFlushMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
        }

        {
            std::lock_guard lock(*resource.queue_lock);

            checkVK(vkResetFences(dev, 1, &resource.fence));

            VkSubmitInfo submit_info {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &resource.cmd,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr
            };

            checkVK(vkQueueSubmit(resource.queue, 1, &submit_info, resource.fence));
        }

        checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));

        if (!coherent) {
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

        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }

            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);
            const auto & cfg = d->planes[plane];

            auto dstp = vsapi->getWritePtr(dst, plane);
            const uint8_t * h_bufferp =
                static_cast<const uint8_t *>(static_cast<const void *>(map)) + d->upload_total + cfg.download_offset;

            // raw byte copy of the plane (staging layout matches the frame pitch)
            copy_stream_read(dstp, h_bufferp,
                static_cast<size_t>(cfg.width * d->elem_bytes) * height);
        }

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC GaussFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    GaussData * d = static_cast<GaussData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC GaussCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<GaussData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("GaussBlur: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    const auto & fmt = d->vi->format;
    const int bits = fmt.bitsPerSample;
    const bool depth_ok = (fmt.sampleType == stFloat && (bits == 32 || bits == 16)) ||
                          (fmt.sampleType == stInteger && (bits == 8 || bits == 16));
    if (!depth_ok || d->vi->width <= 0 || d->vi->height <= 0 ||
        (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV && fmt.colorFamily != cfRGB)) {
        return set_error("input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    }

    d->bits = bits;
    d->half = fmt.sampleType == stFloat && bits == 16;
    d->elem_bytes = bits / 8;

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    int num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams < 1 || num_streams > 32) {
        return set_error("num_streams must be 1..32.");
    }
    d->num_streams = num_streams;

    // sigma defaults: plane 0 = 0.5; chroma = sigma[0]/sqrt((1<<subW)*(1<<subH))
    // (computed in double, then narrowed); plane 2 = plane 1
    const int subW = fmt.subSamplingW;
    const int subH = fmt.subSamplingH;
    std::array<float, 3> sigma;
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] = static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &error));
        if (error) {
            if (i == 0) {
                sigma[i] = 0.5f;
            } else if (i == 1) {
                const double sub_factor = std::sqrt(static_cast<double>((1 << subH) * (1 << subW)));
                sigma[i] = static_cast<float>(static_cast<double>(sigma[0]) / sub_factor);
            } else {
                sigma[i] = sigma[i - 1];
            }
        } else if (!std::isfinite(sigma[i]) || sigma[i] < 0.0f) {
            return set_error("sigma must be a finite value >= 0.");
        }
    }

    bool any_process = false;
    for (int i = 0; i < std::ssize(sigma); ++i) {
        d->process[i] = i < fmt.numPlanes && sigma[i] >= FLT_EPSILON;
        any_process |= d->process[i];
    }
    if (!any_process) {
        return set_error("all planes have sigma < FLT_EPSILON (nothing to process).");
    }

    {
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
        d->device_id = device_id;
    }

    // Per-plane configuration, with deduplication for identical planes
    struct ConfigKey {
        int w, h, stride;
        float sigma;

        bool operator==(const ConfigKey & other) const noexcept {
            return w == other.w && h == other.h && stride == other.stride &&
                   sigma == other.sigma;
        }
    };

    std::vector<ConfigKey> keys;
    std::vector<std::vector<float>> weights;
    std::array<ConfigKey, 3> plane_keys {};
    std::array<bool, 3> plane_valid {};
    std::array<int, 3> plane_cfg {};

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }

        const int plane_width = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        const int plane_height = (plane == 0) ? d->vi->height : d->vi->height >> subH;
        const int pitch_bytes = (plane_width * d->elem_bytes + 15) & ~15;
        const int stride = pitch_bytes / d->elem_bytes;

        const ConfigKey key { plane_width, plane_height, stride, sigma[plane] };

        int ci = 0;
        for (; ci < std::ssize(keys); ++ci) {
            if (keys[ci] == key) {
                break;
            }
        }
        if (ci == std::ssize(keys)) {
            if (key.sigma > static_cast<float>(std::min(key.w, key.h))) {
                return set_error("sigma too large for plane (radius >= dimension).");
            }
            auto kernel = get_gauss_kernel(key.sigma);
            const int ksize = static_cast<int>(kernel.size());
            const int radius = ksize / 2;
            if (radius > key.w - 1 || radius > key.h - 1) {
                return set_error("sigma too large for plane (radius >= dimension).");
            }
            keys.push_back(key);
            weights.push_back(std::move(kernel));
        }
        plane_keys[plane] = key;
        plane_valid[plane] = true;
        plane_cfg[plane] = ci;
    }

    const int n_cfg = static_cast<int>(keys.size());

    VkDevice dev = d->device->device;

    // Determine the code path per config: the fused small path only when its
    // shared-memory tile fits the device.
    std::array<bool, 3> cfg_small {};
    for (int ci = 0; ci < n_cfg; ++ci) {
        const auto & key = keys[ci];
        const int ksize = static_cast<int>(weights[ci].size());
        const int radius = ksize / 2;
        const size_t tile_bytes =
            static_cast<size_t>(VRT) * BLK_Y * (BLK_X + 2 * radius) * sizeof(float);
        cfg_small[ci] = radius <= LARGE_THRESHOLD &&
            tile_bytes <= std::min<size_t>(48 * 1024, d->device->limits.maxComputeSharedMemorySize);
    }

    // Pipeline layout, descriptor set layout and descriptor pool
    {
        VkDescriptorSetLayoutBinding bindings[4] {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };

        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 4,
            .pBindings = bindings
        };

        checkVK(vkCreateDescriptorSetLayout(
            d->device->device, &layout_info, nullptr, &d->set_layout));
    }
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 4 * sizeof(int32_t)
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * static_cast<uint32_t>(d->num_streams)
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

    // Weights buffer: all configs' kernels concatenated
    {
        VkDeviceSize wt_bytes = 0;
        for (const auto & w : weights) {
            wt_bytes += static_cast<VkDeviceSize>(w.size()) * sizeof(float);
        }
        d->wt_bytes = std::max<VkDeviceSize>(wt_bytes, 4);

        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = d->wt_bytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };
        checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &d->wt_buf));
        {
            const auto result = allocate_memory(*d->device, d->wt_buf,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->wt_mem = std::get<AllocatedMemory>(result).memory;
            d->wt_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        checkVK(vkMapMemory(dev, d->wt_mem, 0, d->wt_bytes, 0, reinterpret_cast<void **>(&d->wt_map)));

        uint32_t wt_base = 0;
        for (int ci = 0; ci < n_cfg; ++ci) {
            std::memcpy(d->wt_map + wt_base, weights[ci].data(),
                weights[ci].size() * sizeof(float));
            wt_base += static_cast<uint32_t>(weights[ci].size());
        }

        const bool wt_coherent =
            !!(d->device->mem_props.memoryTypes[d->wt_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!wt_coherent) {
            VkMappedMemoryRange flush_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = d->wt_mem,
                .offset = 0,
                .size = VK_WHOLE_SIZE
            };
            checkVK(vkFlushMappedMemoryRanges(dev, 1, &flush_range));
        }
    }

    // Shader modules (create lazily — only those actually used)
    {
        const uint32_t * gauss_code = nullptr;
        size_t gauss_size = 0;
        const uint32_t * vert_code = nullptr;
        size_t vert_size = 0;
        const uint32_t * horiz_code = nullptr;
        size_t horiz_size = 0;
        switch (d->bits) {
            case 8:
                gauss_code = gaussblur_8_gauss_spv;  gauss_size = gaussblur_8_gauss_spv_size;
                vert_code = gaussblur_8_vert_spv;    vert_size = gaussblur_8_vert_spv_size;
                horiz_code = gaussblur_8_horiz_spv;  horiz_size = gaussblur_8_horiz_spv_size;
                break;
            case 16:
                if (d->half) {
                    gauss_code = gaussblur_16h_gauss_spv;  gauss_size = gaussblur_16h_gauss_spv_size;
                    vert_code = gaussblur_16h_vert_spv;    vert_size = gaussblur_16h_vert_spv_size;
                    horiz_code = gaussblur_16h_horiz_spv;  horiz_size = gaussblur_16h_horiz_spv_size;
                } else {
                    gauss_code = gaussblur_16_gauss_spv;  gauss_size = gaussblur_16_gauss_spv_size;
                    vert_code = gaussblur_16_vert_spv;    vert_size = gaussblur_16_vert_spv_size;
                    horiz_code = gaussblur_16_horiz_spv;  horiz_size = gaussblur_16_horiz_spv_size;
                }
                break;
            case 32:
                gauss_code = gaussblur_32_gauss_spv;  gauss_size = gaussblur_32_gauss_spv_size;
                vert_code = gaussblur_32_vert_spv;    vert_size = gaussblur_32_vert_spv_size;
                horiz_code = gaussblur_32_horiz_spv;  horiz_size = gaussblur_32_horiz_spv_size;
                break;
            default:
                return set_error("unsupported bit depth");
        }

        bool need_gauss = false;
        bool need_large = false;
        for (int ci = 0; ci < n_cfg; ++ci) {
            need_gauss |= cfg_small[ci];
            need_large |= !cfg_small[ci];
        }

        if (need_gauss) {
            const auto result = create_shader_module(*d->device, gauss_code, gauss_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->module = std::get<VkShaderModule>(result);
        }
        if (need_large) {
            {
                const auto result = create_shader_module(*d->device, vert_code, vert_size);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                d->v_module = std::get<VkShaderModule>(result);
            }
            {
                const auto result = create_shader_module(*d->device, horiz_code, horiz_size);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                d->h_module = std::get<VkShaderModule>(result);
            }
        }
    }

    // Per-plane pipelines, with deduplication for identical plane configs
    const uint32_t max_grid_x = d->device->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d->device->limits.maxComputeWorkGroupCount[1];

    auto & planes = d->planes;

    // Deduplicated pipelines: reuse an existing pipeline for identical configs
    std::array<VkPipeline, 3> cfg_pipeline {};
    std::array<VkPipeline, 3> cfg_v_pipeline {};
    std::array<VkPipeline, 3> cfg_h_pipeline {};
    std::array<bool, 3> cfg_created {};

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!plane_valid[plane]) {
            continue;
        }
        const int ci = plane_cfg[plane];
        if (cfg_created[ci]) {
            continue;
        }

        const auto & key = keys[ci];
        const int ksize = static_cast<int>(weights[ci].size());
        const int radius = ksize / 2;

        PlaneConfig cfg;
        cfg.width = key.w;
        cfg.height = key.h;
        cfg.stride = key.stride;
        cfg.ksize = ksize;
        cfg.radius = radius;
        cfg.small = cfg_small[ci];

        if (cfg.small) {
            const auto result = create_pipeline(*d->device, cfg, d->module, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            cfg_pipeline[ci] = std::get<VkPipeline>(result);
        } else {
            {
                const auto result = create_pipeline(*d->device, cfg, d->v_module, d->pipeline_layout);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                cfg_v_pipeline[ci] = std::get<VkPipeline>(result);
            }
            {
                const auto result = create_pipeline(*d->device, cfg, d->h_module, d->pipeline_layout);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                cfg_h_pipeline[ci] = std::get<VkPipeline>(result);
            }
        }
        cfg_created[ci] = true;
    }

    // assign the pipeline handles and grid sizes to the planes
    std::array<bool, 3> wt_assigned {};
    uint32_t wt_base = 0;
    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!plane_valid[plane]) {
            continue;
        }
        const int ci = plane_cfg[plane];
        auto & cfg = planes[plane];
        const auto & key = plane_keys[plane];
        const int ksize = static_cast<int>(weights[ci].size());
        const int radius = ksize / 2;

        cfg.width = key.w;
        cfg.height = key.h;
        cfg.stride = key.stride;
        cfg.pitch_bytes = cfg.stride * d->elem_bytes;
        cfg.ksize = ksize;
        cfg.radius = radius;
        cfg.small = cfg_small[ci];
        cfg.pipeline = cfg_pipeline[ci];
        cfg.v_pipeline = cfg_v_pipeline[ci];
        cfg.h_pipeline = cfg_h_pipeline[ci];

        cfg.grid_x = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.width + BLK_X - 1) / BLK_X, static_cast<int64_t>(max_grid_x)));
        cfg.grid_y = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.height + VRT * BLK_Y - 1) / (VRT * BLK_Y), static_cast<int64_t>(max_grid_y)));
        cfg.v_grid_x = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.width + BLK_X - 1) / BLK_X, static_cast<int64_t>(max_grid_x)));
        cfg.v_grid_y = static_cast<uint32_t>(std::min<int64_t>(
            ((cfg.height + LARGE_R - 1) / LARGE_R + BLK_Y - 1) / BLK_Y, static_cast<int64_t>(max_grid_y)));
        cfg.h_grid_x = static_cast<uint32_t>(std::min<int64_t>(
            ((cfg.width + LARGE_R - 1) / LARGE_R + BLK_X - 1) / BLK_X, static_cast<int64_t>(max_grid_x)));
        cfg.h_grid_y = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.height + BLK_Y - 1) / BLK_Y, static_cast<int64_t>(max_grid_y)));

        if (wt_assigned[ci]) {
            // reuse the weights offset of the first plane sharing this config
            for (int other = 0; other < plane; ++other) {
                if (plane_valid[other] && plane_cfg[other] == ci) {
                    cfg.wt_base = planes[other].wt_base;
                    break;
                }
            }
        } else {
            wt_assigned[ci] = true;
            cfg.wt_base = wt_base;
            wt_base += static_cast<uint32_t>(weights[ci].size());
        }
    }

    // Buffer region offsets (raw bytes), each region 32-byte aligned so the
    // streaming copies can use aligned loads/stores
    auto align32 = [](VkDeviceSize v) { return (v + 31) & ~VkDeviceSize(31); };

    VkDeviceSize upload_total = 0;
    VkDeviceSize download_total = 0;
    VkDeviceSize tmp_total = 0;

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!plane_valid[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        const VkDeviceSize plane_bytes =
            static_cast<VkDeviceSize>(cfg.height) * cfg.pitch_bytes;
        const VkDeviceSize tmp_bytes =
            plane_bytes * (4 / d->elem_bytes);

        cfg.upload_offset = align32(upload_total);
        cfg.upload_size = plane_bytes;
        upload_total = align32(cfg.upload_offset + cfg.upload_size);

        cfg.download_offset = align32(download_total);
        cfg.download_size = plane_bytes;
        download_total = align32(cfg.download_offset + cfg.download_size);

        cfg.tmp_offset = align32(tmp_total);
        cfg.tmp_size = tmp_bytes;
        tmp_total = align32(cfg.tmp_offset + cfg.tmp_size);
    }

    d->upload_total = upload_total;
    d->download_total = download_total;
    d->tmp_total = tmp_total;
    d->need_fill = d->bits != 32;

    const VkDeviceSize min_size = 4;
    const VkDeviceSize staging_size = std::max(upload_total + download_total + tmp_total, 2 * min_size);

    // Resources
    d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->resources.reserve(d->num_streams);

    uint32_t num_queues = std::min(
        d->num_streams, static_cast<int>(d->device->queue_count));

    for (int i = 0; i < d->num_streams; ++i) {
        VK_Resource resource;

        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.staging));
        }

        {
            const auto result = allocate_memory(
                *d->device, resource.staging,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.staging_mem = std::get<AllocatedMemory>(result).memory;
            resource.staging_type_index = std::get<AllocatedMemory>(result).type_index;
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
            VkDescriptorBufferInfo wt_info {
                .buffer = d->wt_buf,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo src_info {
                .buffer = resource.staging,
                .offset = 0,
                .range = staging_size
            };
            VkDescriptorBufferInfo dst_info {
                .buffer = resource.staging,
                .offset = 0,
                .range = staging_size
            };
            VkDescriptorBufferInfo tmp_info {
                .buffer = resource.staging,
                .offset = 0,
                .range = staging_size
            };

            VkWriteDescriptorSet writes[4] {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &wt_info,
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
                    .pBufferInfo = &src_info,
                    .pTexelBufferView = nullptr
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &dst_info,
                    .pTexelBufferView = nullptr
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &tmp_info,
                    .pTexelBufferView = nullptr
                },
            };

            vkUpdateDescriptorSets(dev, 4, writes, 0, nullptr);
        }

        checkVK(vkMapMemory(dev, resource.staging_mem, 0, staging_size, 0, reinterpret_cast<void **>(&resource.map)));

        resource.queue = d->device->queues[i % num_queues].queue;
        resource.queue_lock = d->device->queues[i % num_queues].lock.get();

        if (const auto err = record_command_buffer(*d, resource)) {
            return set_error(*err);
        }

        d->resources.push_back(std::move(resource));
    }

    VSFilterDependency deps[1] = {{d->node, rpStrictSpatial}};

    GaussData *data = d.release();

    vsapi->createVideoFilter(
        out, "GaussBlur", data->vi,
        GaussGetFrame, GaussFree,
        fmParallel, deps, 1, data, core);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_gaussblur(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "GaussBlur",
        "clip:vnode;"
        "sigma:float[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode;",
        GaussCreate, nullptr, plugin
    );
}