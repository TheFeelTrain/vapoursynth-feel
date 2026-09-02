#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
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
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// EEDI3 — full-pel edge-directed line interpolation.
//
// Family-A semantics: reproduces HolyWu's eedi3m (the CPU ground truth) and
// agrees with eedi3vk2 (same family, GPU). This file follows the vsfeel host
// idiom (gaussblur/bilateral): a per-frame resource with a staging buffer for
// uploads/downloads and device-local buffers for the interleaved kernels.
//
// Supported formats: u16 integer and f32 float only (per the vsfeel scope).
// mclip is a SINGLE Gray plane driving every processed plane (vszip CPU
// semantic; non-Gray8 masks are converted to Gray8 internally). hp is not
// implemented (parameter absent).
//
// Kernel pipeline per plane:
//   ENTRY_ROW    one workgroup per interp row; DP + backtrack + interpolate.
//                Writes interp rows (rows = dstH/2, tight) to b1, pbt to b2,
//                dmap/cint to b3/b6 (vcheck only).
//   ENTRY_VCHECK single workgroup per plane; serial interp-row walk; writes
//                the vchecked interp rows back to b1.
// After the GPU, the host blits the interp rows from the download region into
// the dst frame at field+2r and copies the kept rows (parity 1-field) from
// src, exactly like eedi3vk2.
// ---------------------------------------------------------------------------

constexpr int MARGIN_H = 12;
constexpr int MARGIN_V = 4;
constexpr int MAX_PLANES = 3;

// Host-side pad upload is always float (see eedi3.comp header): u16 native
// values are exact integers < 2^24 so float cost/interp math is bit-exact.
enum Binding : uint32_t {
    B_PAD = 0,
    B_DST = 1,
    B_PBT = 2,
    B_DMAP = 3,
    B_BMASK = 4,
    B_SCLIP = 5,
    B_CINT = 6,
    BIND_COUNT = 7
};

struct PlaneConfig {
    int width {};                     // output plane width (pixels)
    int height {};                    // output plane height (pixels)
    int rows {};                      // number of interp rows == height / 2
    int pad_stride {};                // float elements per padded row
    int pad_height {};                // padded rows == height + 2*MARGIN_V
    int tpitch {};                    // 2*mdis + 1

    VkPipeline row_pipeline {};
    VkPipeline vcheck_pipeline {};    // only when vcheck > 0

    // staging (host-visible) regions, byte offsets
    VkDeviceSize pad_offset {};       // float pad plane (upload)
    VkDeviceSize pad_bytes {};
    VkDeviceSize bmask_offset {};     // dilated mask rows (upload; mclip only)
    VkDeviceSize bmask_bytes {};
    VkDeviceSize sclip_offset {};     // sclip interp rows (upload; sclip only)
    VkDeviceSize sclip_bytes {};
    VkDeviceSize dl_offset {};        // interp row download (device -> host)
    VkDeviceSize dl_bytes {};

    // device-local buffer regions, byte offsets (offsets into d->dev_buf)
    VkDeviceSize dst_offset {};       // interp rows (io type)
    VkDeviceSize dst_bytes {};
    VkDeviceSize pbt_offset {};       // int8 rows*width*tpitch
    VkDeviceSize pbt_bytes {};
    VkDeviceSize dmap_offset {};      // int8 rows*width (vcheck only)
    VkDeviceSize dmap_bytes {};
    VkDeviceSize cint_offset {};      // io rows*width (vcheck && !sclip)
    VkDeviceSize cint_bytes {};
};

struct Eedi3Resource {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    VkBuffer dev_buf {};          // device-local kernels' buffers
    VkDeviceMemory dev_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    float * map {};
    uint32_t staging_type_index {};
    uint32_t dev_type_index {};
};

struct Eedi3Data {
    VSNode * node {};
    VSNode * sclip_node {};
    VSNode * mclip_node {};
    const VSVideoInfo * vi {};

    int device_id {}, num_streams {};
    int bits {}, elem_bytes {};
    bool process[MAX_PLANES] { true, true, true };

    int field {}, nrad { 2 }, mdis { 20 }, vcheck { 2 };
    bool dh {};
    float alpha { 0.2f }, beta { 0.25f }, gamma { 20.0f };
    float vthresh2 { 4.0f };
    float rw {}, rcp_vth0 {}, rcp_vth1 {}, rcp_vth2 {};
    int peak {};

    // one pipeline set per distinct plane width (like eedi3vk2); pipelines
    // for the same width are shared by all planes of that width
    struct WidthKey {
        int width, rows, tpitch, pad_stride, pad_height;
        int has_mclip, has_sclip, vcheck;
        bool operator==(const WidthKey & o) const {
            return width == o.width && rows == o.rows && tpitch == o.tpitch &&
                   pad_stride == o.pad_stride && pad_height == o.pad_height &&
                   has_mclip == o.has_mclip && has_sclip == o.has_sclip &&
                   vcheck == o.vcheck;
        }
    };

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule row_module {};
    VkShaderModule vcheck_module {};
    VkDeviceSize upload_total {}, download_total {}, dev_total {};
    std::array<PlaneConfig, MAX_PLANES> planes {};
    FramePool<Eedi3Resource> pool;

    // module/pipeline cache keyed by width (values owned by the pipelines map)
    std::vector<std::pair<WidthKey, std::array<VkPipeline, 2>>> width_pipes {};

    ~Eedi3Data() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);

        for (auto & resource : pool.items) {
            if (resource.map) {
                vkUnmapMemory(dev, resource.staging_mem);
            }
            if (resource.dev_mem) {
                vkFreeMemory(dev, resource.dev_mem, nullptr);
            }
            if (resource.dev_buf) {
                vkDestroyBuffer(dev, resource.dev_buf, nullptr);
            }
            destroy_common(dev, resource);
        }

        VkPipeline destroyed[2 * 3] {};
        int nd = 0;
        for (auto & [key, pair] : width_pipes) {
            for (int i = 0; i < 2; ++i) {
                if (!pair[i]) {
                    continue;
                }
                bool seen = false;
                for (int j = 0; j < nd; ++j) {
                    if (destroyed[j] == pair[i]) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    destroyed[nd++] = pair[i];
                    vkDestroyPipeline(dev, pair[i], nullptr);
                }
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
        if (row_module) {
            vkDestroyShaderModule(dev, row_module, nullptr);
        }
        if (vcheck_module) {
            vkDestroyShaderModule(dev, vcheck_module, nullptr);
        }

        release_device(device);
    }
};

// ---------------------------------------------------------------------------
// CPU-side pad / mask staging (mirrors eedi3m's copyPad exactly)
// ---------------------------------------------------------------------------

// Writes one full padded row (including the mirrored margins) of the given
// source row into tmp, then copies it to pad at `y` (in pad-row units).
// srcRow may be negative / past the end (vertical mirror handles it above).
template <typename Tsrc>
static void copy_pad_row(const Tsrc * src_line, const int src_stride,
                         float * tmp, const int pad_w, const int src_row,
                         const int y, float * pad, const int pad_stride,
                         const int src_width, const int src_height) {
    // resolve mirrored source row
    int real = src_row;
    if (real < 0) {
        real = -1 - real;
    }
    if (real >= src_height) {
        real = 2 * src_height - 1 - real;
    }
    real = std::max(real, 0);
    if (real >= src_height) {
        real = src_height - 1;
    }

    const Tsrc * line = src_line + static_cast<int64_t>(src_stride) * real;

    for (int x = 0; x < src_width; ++x) {
        tmp[MARGIN_H + x] = static_cast<float>(line[x]);
    }
    for (int x = 0; x < MARGIN_H; ++x) {
        tmp[x] = tmp[MARGIN_H * 2 - x];
    }
    for (int x = pad_w - MARGIN_H, c = 2; x < pad_w; ++x, c += 2) {
        tmp[x] = tmp[x - c];
    }

    std::memcpy(pad + static_cast<int64_t>(pad_stride) * y, tmp, pad_w * sizeof(float));
}

// Builds the host-side padded plane exactly like eedi3m's copyPad: kept rows
// (parity `off` in the source, which is 1-field for dh=0 and every row with
// 2-row output pitch for dh=1) plus mirrored vertical margins. `pad` is the
// float upload destination (native values for u16 / raw f32 for float).
template <typename Tsrc>
static void copy_pad_plane(const Tsrc * src, const ptrdiff_t src_stride_elems,
                           const int src_width, const int src_height,
                           float * pad, const int pad_stride, const int pad_w,
                           const int pad_h, const bool dh, const int off,
                           float * tmp) {
    // Interior kept rows.
    if (!dh) {
        // pad row MARGIN_V + off + 2k  <- src row off + 2k, k = 0..H/2-1
        for (int k = 0; k < src_height / 2; ++k) {
            const int y = MARGIN_V + off + 2 * k;
            copy_pad_row(src, src_stride_elems, tmp, pad_w,
                         off + 2 * k, y, pad, pad_stride, src_width, src_height);
        }
    } else {
        // dh: pad row MARGIN_V + off + 2k <- src row k for k in 0..srcH-1
        for (int k = 0; k < src_height; ++k) {
            const int y = MARGIN_V + off + 2 * k;
            copy_pad_row(src, src_stride_elems, tmp, pad_w,
                         k, y, pad, pad_stride, src_width, src_height);
        }
    }

    // Vertical margins (copy whole mirrored rows).
    for (int y = off; y < MARGIN_V; y += 2) {
        std::memcpy(pad + static_cast<int64_t>(pad_stride) * y,
                    pad + static_cast<int64_t>(pad_stride) * (MARGIN_V * 2 - y),
                    pad_w * sizeof(float));
    }
    for (int y = pad_h - MARGIN_V + off, c = 2 + 2 * off; y < pad_h; y += 2, c += 4) {
        std::memcpy(pad + static_cast<int64_t>(pad_stride) * y,
                    pad + static_cast<int64_t>(pad_stride) * (y - c),
                    pad_w * sizeof(float));
    }
}

// Mirrors eedi3m's bmask dilation scan for one row: marks columns within
// mdis of a set mask pixel ("last" propagation).
static void build_bmask_row(const uint8_t * maskp, uint8_t * out,
                            const int width, const int mdis) {
    const int minmdis = std::min(width, mdis);
    int last = -666999;

    for (int x = 0; x < minmdis; ++x) {
        if (maskp[x] != 0) {
            last = x + mdis;
        }
    }
    for (int x = 0; x < width - minmdis; ++x) {
        if (maskp[x + mdis] != 0) {
            last = x + mdis * 2;
        }
        out[x] = (x <= last) ? 1 : 0;
    }
    for (int x = std::max(width - minmdis, 0); x < width; ++x) {
        out[x] = (x <= last) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Shader module + pipeline helpers (vsfeel idiom)
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

struct RowSpecData {
    int32_t width;
    int32_t nrad;
    int32_t mdis;
    int32_t has_mclip;
    int32_t has_sclip;
    int32_t vcheck;
    int32_t lsz_row;
    int32_t lsz_vcheck;
};

static constexpr std::array<VkSpecializationMapEntry, 8> row_entries {{
    { 0,  0, sizeof(int32_t) },
    { 1,  4, sizeof(int32_t) },
    { 2,  8, sizeof(int32_t) },
    { 3, 12, sizeof(int32_t) },
    { 4, 16, sizeof(int32_t) },
    { 5, 20, sizeof(int32_t) },
    { 6, 24, sizeof(int32_t) },
    { 7, 28, sizeof(int32_t) },
}};

static std::variant<VkPipeline, std::string> create_pipeline(
    const VK_Device & dev, const RowSpecData & spec, VkShaderModule module,
    VkPipelineLayout layout) {

    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(row_entries.size()),
        .pMapEntries = row_entries.data(),
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

// Push constant layout must match the shader's PC struct (int block then
// float block; see eedi3.comp).
struct PushConstants {
    int32_t pad_base;
    int32_t dst_base;
    int32_t pbt_base;
    int32_t dmap_base;
    int32_t bmask_base;
    int32_t sclip_base;
    int32_t cint_base;
    int32_t pad_stride;
    int32_t pad_height;
    int32_t field;
    int32_t rows;
    float alpha;
    float beta;
    float gamma;
    float rw;
    float vth0r;
    float vth1r;
    float vth2r;
    float vth2;
};
static_assert(sizeof(PushConstants) == 11 * 4 + 8 * 4, "push constants size");

// base offsets in ELEMENTS for each binding of a plane's regions (element
// type per binding; the descriptors range the whole buffer so the shader
// indexes base + offset).
struct PlaneBases {
    int32_t pad;       // float elements into staging (pad region)
    int32_t dst;       // io elements into dev_buf (interp rows)
    int32_t pbt;       // int8 elements into dev_buf
    int32_t dmap;      // int8 elements into dev_buf
    int32_t bmask;     // uint8 elements into staging
    int32_t sclip;     // io elements into staging
    int32_t cint;      // io elements into dev_buf
};

static std::optional<std::string> record_command_buffer(
    const Eedi3Data & d, Eedi3Resource & resource, const int field) {

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

    const int32_t elem = d.elem_bytes;

    // The row kernel reads the pad / bmask / sclip straight from the staging
    // (host) buffer and writes dst / pbt / dmap / cint to dev_buf; a D2H copy
    // then brings the interp rows back into the staging download area. The
    // command buffer is recorded per frame because the interp-row parity
    // `field` varies (field > 1 doubles frames and _FieldBased sources).
    (void)field;
    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
            continue;
        }
        const auto & cfg = d.planes[plane];
        const int has_mclip = d.mclip_node ? 1 : 0;
        const int has_sclip = (d.vcheck > 0 && d.sclip_node) ? 1 : 0;

        PushConstants pc {
            .pad_base = static_cast<int32_t>(cfg.pad_offset / sizeof(float)),
            .dst_base = static_cast<int32_t>(cfg.dst_offset / elem),
            .pbt_base = static_cast<int32_t>(cfg.pbt_offset),
            .dmap_base = static_cast<int32_t>(cfg.dmap_offset),
            .bmask_base = static_cast<int32_t>(cfg.bmask_offset),
            .sclip_base = static_cast<int32_t>(cfg.sclip_offset / elem),
            .cint_base = static_cast<int32_t>(cfg.cint_offset / elem),
            .pad_stride = cfg.pad_stride,
            .pad_height = cfg.pad_height,
            .field = field,
            .rows = cfg.rows,
            .alpha = d.alpha,
            .beta = d.beta,
            .gamma = d.gamma,
            .rw = d.rw,
            .vth0r = d.rcp_vth0,
            .vth1r = d.rcp_vth1,
            .vth2r = d.rcp_vth2,
            .vth2 = d.vthresh2,
        };

        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.row_pipeline);
        vkCmdBindDescriptorSets(
            resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
        vkCmdPushConstants(
            resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);
        vkCmdDispatch(resource.cmd, 1, static_cast<uint32_t>(cfg.rows), 1);

        if (d.vcheck > 0) {
            // the vcheck kernel reads the row kernel's writes
            VkMemoryBarrier mem_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.vcheck_pipeline);
            vkCmdBindDescriptorSets(
                resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdPushConstants(
                resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(pc), &pc);
            vkCmdDispatch(resource.cmd, 1, 1, 1);
        }
    }

    // download: dev_buf interp rows -> staging download region
    if (d.download_total > 0) {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
            if (!d.process[plane]) {
                continue;
            }
            const auto & cfg = d.planes[plane];
            const VkBufferCopy region {
                .srcOffset = cfg.dst_offset,
                .dstOffset = d.upload_total + cfg.dl_offset,
                .size = cfg.dl_bytes
            };
            vkCmdCopyBuffer(resource.cmd, resource.dev_buf, resource.staging, 1, &region);
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

static const VSFrame *VS_CC Eedi3GetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    const int sn = (d->field > 1) ? n / 2 : n;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
        if (d->vcheck > 0 && d->sclip_node) {
            vsapi->requestFrameFilter(n, d->sclip_node, frameCtx);
        }
        if (d->mclip_node) {
            vsapi->requestFrameFilter(sn, d->mclip_node, frameCtx);
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const VSFrame * src = vsapi->getFrameFilter(sn, d->node, frameCtx);

    // Plane passthrough. newVideoFrame2 requires a non-null planeSrc array but
    // tolerates null entries (those planes are freshly allocated). Processed
    // planes are always freshly allocated. Unprocessed planes may share src's
    // plane data when !dh (dims match); under dh the output is 2x tall so no
    // plane can be shared (eedi3m/eedi3vk2 also leave unprocessed planes
    // undefined there).
    const int pl[] = { 0, 1, 2 };
    const VSFrame * fr[] = {
        (!d->dh && !d->process[0]) ? src : nullptr,
        (!d->dh && !d->process[1]) ? src : nullptr,
        (!d->dh && !d->process[2]) ? src : nullptr
    };
    // dh doubles the output height (width is never changed by EEDI3)
    const int out_height = d->dh ? d->vi->height * 2 : d->vi->height;
    VSFrame * dst = vsapi->newVideoFrame2(
        &d->vi->format, d->vi->width, out_height, fr, pl, src, core);

    auto resource = d->pool.take();

    auto set_error = [&](const std::string & error_message) {
        d->pool.give_back(std::move(resource));
        vsapi->setFilterError(("EEDI3VK: " + error_message).c_str(), frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    };

    VkDevice dev = d->device->device;
    float * map = resource.map;

    const VSFrame * scp = nullptr;
    if (d->vcheck > 0 && d->sclip_node) {
        scp = vsapi->getFrameFilter(n, d->sclip_node, frameCtx);
    }
    const VSFrame * mcp = nullptr;
    if (d->mclip_node) {
        mcp = vsapi->getFrameFilter(sn, d->mclip_node, frameCtx);
    }

    int field = d->field & 1;
    int err;
    const int fieldBased = vsapi->mapGetIntSaturated(
        vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err);
    if (fieldBased == VSC_FIELD_BOTTOM) {
        field = 0;
    } else if (fieldBased == VSC_FIELD_TOP) {
        field = 1;
    }
    if (d->field > 1) {
        field = (n & 1) ^ field;
    }
    const int off = 1 - field;

    // Re-record the command buffer with this frame's interp-row parity (the
    // previous submit on this resource was waited on before give_back, so the
    // pool reset is safe).
    checkVK(vkResetCommandPool(dev, resource.pool, 0));
    if (const auto err = record_command_buffer(*d, resource, field)) {
        return set_error(*err);
    }

    const bool coherent =
        !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // CPU: stage pad (float), dilated bmask rows, and sclip interp rows.
    uint8_t * const staging = static_cast<uint8_t *>(static_cast<void *>(map));
    const int max_pad_row = [&] {
        int m = 0;
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (d->process[plane]) {
                m = std::max(m, d->planes[plane].pad_stride * static_cast<int>(sizeof(float)));
            }
        }
        return m;
    }();
    std::vector<float> tmp_row(max_pad_row / sizeof(float));

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        const auto & cfg = d->planes[plane];

        const auto srcp = vsapi->getReadPtr(src, plane);
        const ptrdiff_t src_stride = vsapi->getStride(src, plane);
        const int src_width = vsapi->getFrameWidth(src, plane);
        const int src_height = vsapi->getFrameHeight(src, plane);

        float * pad = reinterpret_cast<float *>(staging + cfg.pad_offset);
        if (d->bits == 16) {
            copy_pad_plane<uint16_t>(
                reinterpret_cast<const uint16_t *>(srcp), src_stride / 2,
                src_width, src_height, pad, cfg.pad_stride,
                cfg.width + MARGIN_H * 2, cfg.pad_height, d->dh, off,
                tmp_row.data());
        } else {
            copy_pad_plane<float>(
                reinterpret_cast<const float *>(srcp), src_stride / 4,
                src_width, src_height, pad, cfg.pad_stride,
                cfg.width + MARGIN_H * 2, cfg.pad_height, d->dh, off,
                tmp_row.data());
        }

        if (d->mclip_node && mcp) {
            // single Gray mask drives every processed plane: mask row for
            // interp row r is (dh ? r : field + 2r) of the mclip frame
            const uint8_t * maskp = vsapi->getReadPtr(mcp, 0);
            const ptrdiff_t mask_stride = vsapi->getStride(mcp, 0);
            uint8_t * bm = staging + cfg.bmask_offset;
            for (int r = 0; r < cfg.rows; ++r) {
                const int mrow = d->dh ? r : field + 2 * r;
                build_bmask_row(maskp + mask_stride * mrow,
                                bm + static_cast<int64_t>(r) * cfg.width,
                                cfg.width, d->mdis);
            }
        }

        if (d->vcheck > 0 && d->sclip_node && scp) {
            const auto scpp = vsapi->getReadPtr(scp, plane);
            const ptrdiff_t scp_stride = vsapi->getStride(scp, plane);
            const size_t row_bytes = static_cast<size_t>(cfg.width) * d->elem_bytes;
            uint8_t * sc = staging + cfg.sclip_offset;
            for (int r = 0; r < cfg.rows; ++r) {
                std::memcpy(sc + static_cast<size_t>(r) * row_bytes,
                            scpp + scp_stride * (field + 2 * r), row_bytes);
            }
        }
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
                .offset = 0,
                .size = d->upload_total,
            });
        }
        checkVK(vkFlushMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
    }

    checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
        resource.cmd, resource.fence));

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
                .offset = d->upload_total + cfg.dl_offset,
                .size = cfg.dl_bytes,
            });
        }
        checkVK(vkInvalidateMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
    }

    // Copy results into dst: interp rows from the download region, kept rows
    // straight from src (like eedi3vk2).
    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        const auto & cfg = d->planes[plane];
        const size_t row_bytes = static_cast<size_t>(cfg.width) * d->elem_bytes;
        const uint8_t * dl = static_cast<const uint8_t *>(static_cast<const void *>(map)) +
                             d->upload_total + cfg.dl_offset;

        auto dstp = vsapi->getWritePtr(dst, plane);
        const ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
        const auto srcp = vsapi->getReadPtr(src, plane);
        const ptrdiff_t src_stride = vsapi->getStride(src, plane);

        // interp rows (the row kernel wrote rows r at dst rows field+2r)
        for (int r = 0; r < cfg.rows; ++r) {
            std::memcpy(dstp + dst_stride * (field + 2 * r),
                        dl + static_cast<size_t>(r) * row_bytes, row_bytes);
        }

        // kept rows: parity off in dst. dh=0: same coords as src; dh=1: src
        // row k -> dst row 2k + off (dst height == 2*src height).
        const int dst_height = vsapi->getFrameHeight(dst, plane);
        const int kept_rows = dst_height / 2;
        if (!d->dh) {
            vsh::bitblt(dstp + dst_stride * off, dst_stride * 2,
                        srcp + src_stride * off, src_stride * 2,
                        row_bytes, kept_rows);
        } else {
            vsh::bitblt(dstp + dst_stride * off, dst_stride * 2,
                        srcp, src_stride, row_bytes, kept_rows);
        }
    }

    d->pool.give_back(std::move(resource));

    vsapi->freeFrame(src);
    vsapi->freeFrame(scp);
    vsapi->freeFrame(mcp);

    VSMap * props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);

    if (d->field > 1) {
        int errNum, errDen;
        int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
        int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, 1, 2);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC Eedi3Free(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sclip_node);
    vsapi->freeNode(d->mclip_node);
    delete d;
}

static void VS_CC Eedi3Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<Eedi3Data>() };
    int err = 0;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    d->sclip_node = vsapi->mapGetNode(in, "sclip", 0, &err);
    bool has_sclip = d->sclip_node != nullptr;
    err = 0;
    d->mclip_node = vsapi->mapGetNode(in, "mclip", 0, &err);
    bool has_mclip = d->mclip_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("EEDI3VK: " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (has_sclip) {
            vsapi->freeNode(d->sclip_node);
        }
        if (has_mclip) {
            vsapi->freeNode(d->mclip_node);
        }
    };

    if (auto [bps, sample] = std::pair{
            d->vi->format.bitsPerSample, d->vi->format.sampleType };
        !vsh::isConstantVideoFormat(d->vi) ||
        (sample == stInteger && bps != 16) ||
        (sample == stFloat && bps != 32)
    ) {
        return set_error("input bitdepth must be 16 (integer) or 32 (float).");
    }

    d->bits = d->vi->format.bitsPerSample;
    d->elem_bytes = d->bits / 8;

    d->field = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "field", 0, nullptr));

    d->dh = !!vsapi->mapGetInt(in, "dh", 0, &err);
    err = 0;

    const int m = vsapi->mapNumElements(in, "planes");
    for (int i = 0; i < MAX_PLANES; ++i) {
        d->process[i] = (m <= 0);
    }
    for (int i = 0; i < m; ++i) {
        const int n = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "planes", i, nullptr));
        if (n < 0 || n >= d->vi->format.numPlanes) {
            return set_error("plane index out of range");
        }
        if (d->process[n]) {
            return set_error("plane specified twice");
        }
        d->process[n] = true;
    }

    auto get_float = [&](const char * key, float def) {
        err = 0;
        const float v = static_cast<float>(vsapi->mapGetFloatSaturated(in, key, 0, &err));
        return err ? def : v;
    };
    auto get_int = [&](const char * key, int def) {
        err = 0;
        const int v = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, key, 0, &err));
        return err ? def : v;
    };

    d->alpha = get_float("alpha", 0.2f);
    d->beta = get_float("beta", 0.25f);
    d->gamma = get_float("gamma", 20.0f);
    d->nrad = get_int("nrad", 2);
    d->mdis = get_int("mdis", 20);
    d->vcheck = get_int("vcheck", 2);
    float vthresh0 = get_float("vthresh0", 32.0f);
    float vthresh1 = get_float("vthresh1", 64.0f);
    d->vthresh2 = get_float("vthresh2", 4.0f);

    // opt accepted for eedi3m parity but ignored (GPU is always AVX2-class)
    (void)get_int("opt", 0);

    if (d->field < 0 || d->field > 3) {
        return set_error("field must be 0, 1, 2, or 3");
    }
    if (!d->dh) {
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (d->process[plane] &&
                ((d->vi->height >> (plane > 0 ? d->vi->format.subSamplingH : 0)) & 1)) {
                return set_error("plane's height must be mod 2 when dh=False");
            }
        }
    }
    if (d->dh && d->field > 1) {
        return set_error("field must be 0 or 1 when dh=True");
    }
    if (d->alpha < 0.0f || d->alpha > 1.0f) {
        return set_error("alpha must be between 0.0 and 1.0 (inclusive)");
    }
    if (d->beta < 0.0f || d->beta > 1.0f) {
        return set_error("beta must be between 0.0 and 1.0 (inclusive)");
    }
    if (d->alpha + d->beta > 1.0f) {
        return set_error("alpha+beta must be between 0.0 and 1.0 (inclusive)");
    }
    if (d->gamma < 0.0f) {
        return set_error("gamma must be greater than or equal to 0.0");
    }
    if (d->nrad < 0 || d->nrad > 3) {
        return set_error("nrad must be between 0 and 3 (inclusive)");
    }
    if (d->mdis < 1 || d->mdis > 40) {
        return set_error("mdis must be between 1 and 40 (inclusive)");
    }
    if (d->vcheck < 0 || d->vcheck > 3) {
        return set_error("vcheck must be 0, 1, 2, or 3");
    }
    if (d->vcheck > 0 && (vthresh0 <= 0.0f || vthresh1 <= 0.0f || d->vthresh2 <= 0.0f)) {
        return set_error("vthresh0, vthresh1 and vthresh2 must be greater than 0.0");
    }

    // mclip must be a single Gray plane (vszip CPU semantic); non-Gray8 masks
    // are converted to Gray8 internally (SetFrameProps _Range=1 + resize.Point,
    // like vszip CPU).
    if (d->mclip_node) {
        const auto mvi = vsapi->getVideoInfo(d->mclip_node);
        if (mvi->format.colorFamily != cfGray) {
            return set_error("mclip must be Gray");
        }
        if (mvi->width != d->vi->width || mvi->height != d->vi->height) {
            return set_error("mclip's dimensions don't match");
        }
        if (mvi->numFrames != d->vi->numFrames) {
            return set_error("mclip's number of frames doesn't match");
        }

        if (mvi->format.bitsPerSample != 8 || mvi->format.sampleType != stInteger) {
            VSMap * args = vsapi->createMap();
            vsapi->mapConsumeNode(args, "clip", d->mclip_node, maReplace);
            d->mclip_node = nullptr;  // ownership moved into args

            // std.SetFrameProps(_Range=1)
            vsapi->mapSetInt(args, "_Range", 1, maReplace);
            VSMap * ret = vsapi->invoke(
                vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "SetFrameProps", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                vsapi->freeNode(d->node);
                if (d->sclip_node) {
                    vsapi->freeNode(d->sclip_node);
                }
                return;
            }
            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->freeMap(ret);

            // resize.Point -> Gray8
            vsapi->mapSetInt(args, "format", vsapi->queryVideoFormatID(
                cfGray, stInteger, 8, 0, 0, core), maReplace);
            ret = vsapi->invoke(
                vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args);
            vsapi->freeMap(args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(ret);
                vsapi->freeNode(d->node);
                if (d->sclip_node) {
                    vsapi->freeNode(d->sclip_node);
                }
                return;
            }
            d->mclip_node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(ret);
        }
    }

    // sclip only validated when vcheck > 0 (eedi3m semantics)
    if (d->vcheck > 0 && d->sclip_node) {
        const auto svi = vsapi->getVideoInfo(d->sclip_node);
        if (!vsh::isSameVideoInfo(svi, d->vi)) {
            return set_error("sclip's format and dimensions don't match");
        }
        if (svi->numFrames != d->vi->numFrames) {
            return set_error("sclip's number of frames doesn't match");
        }
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &err));
    if (err) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    int num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &err));
    if (err) {
        num_streams = 4;
    }
    if (num_streams < 1 || num_streams > 32) {
        return set_error("num_streams must be 1..32.");
    }
    d->num_streams = num_streams;
    d->device_id = device_id;

    {
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
    }

    // eedi3m scaling (see EEDI3.cpp create), with cost3 always on (GPU family
    // semantics — eedi3vk2/vszip* have no cost3 switch and always do alpha/3
    // and sum s0+s1+s2):
    //   remainingWeight = 1 - alpha - beta   (raw, before alpha/3)
    //   alpha /= 3   (cost3)
    //   int:  beta *= 2^(bits-8), gamma *= 2^(bits-8),
    //         vthresh0 *= 2^(bits-8), vthresh1 *= 2^(bits-8)
    //   float: beta /= 255, gamma /= 255, vthresh0 /= 255, vthresh1 /= 255
    //   vthresh2 is never scaled; alpha is never /255.
    d->rw = 1.0f - d->alpha - d->beta;
    d->alpha /= 3.0f;
    if (d->vi->format.sampleType == stInteger) {
        d->peak = (1 << d->vi->format.bitsPerSample) - 1;
        const int scale = 1 << (d->vi->format.bitsPerSample - 8);
        d->beta *= static_cast<float>(scale);
        d->gamma *= static_cast<float>(scale);
        vthresh0 *= static_cast<float>(scale);
        vthresh1 *= static_cast<float>(scale);
    } else {
        d->peak = 1;
        d->beta /= 255.0f;
        d->gamma /= 255.0f;
        vthresh0 /= 255.0f;
        vthresh1 /= 255.0f;
    }
    d->rcp_vth0 = 1.0f / vthresh0;
    d->rcp_vth1 = 1.0f / vthresh1;
    d->rcp_vth2 = 1.0f / d->vthresh2;

    // Workgroup sizes (spec constants 6/7 in the shaders): the row kernel's DP
    // needs exactly one lane per direction u in [-mdis..mdis], so its local
    // size is TPITCH (a non-power-of-two like 81 is fine — no subgroup ops).
    // The vcheck kernel is a single WG striding over columns (up to WIDTH).
    const int tpitch = 2 * d->mdis + 1;
    // The row kernel's DP owns one direction per lane (TPITCH lanes) but its
    // backtrack/interpolate tiles walk BT_TILE columns in parallel, so the WG
    // must span both: max(TPITCH, 32). Lanes beyond TPITCH sit out of the DP
    // (their direction u exceeds mdis and the u_in guard skips them).
    constexpr int BT_TILE = 32;   // must match the shader
    const auto & lim = d->device->limits;
    const int max_invoc = static_cast<int>(lim.maxComputeWorkGroupInvocations);
    const int max_x = static_cast<int>(lim.maxComputeWorkGroupSize[0]);
    const int lsz_row = std::min(std::max(tpitch, BT_TILE), std::min(max_invoc, max_x));
    int lsz_vcheck = std::min({ 1024, max_invoc, max_x });
    if (lsz_row < std::max(tpitch, BT_TILE)) {
        return set_error("mdis too large for this device's workgroup limit");
    }

    VkDevice dev = d->device->device;

    // Pipeline layout / descriptors: 7 storage buffers.
    {
        VkDescriptorSetLayoutBinding bindings[BIND_COUNT];
        for (uint32_t i = 0; i < BIND_COUNT; ++i) {
            bindings[i] = VkDescriptorSetLayoutBinding {
                i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr
            };
        }
        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = BIND_COUNT,
            .pBindings = bindings
        };
        checkVK(vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &d->set_layout));
    }
    {
        VkPushConstantRange pcr {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        VkPipelineLayoutCreateInfo plci {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &d->set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcr
        };
        checkVK(vkCreatePipelineLayout(dev, &plci, nullptr, &d->pipeline_layout));
    }
    {
        VkDescriptorPoolSize pool_size {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, BIND_COUNT * static_cast<uint32_t>(d->num_streams)
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

    const int elem_bytes = d->elem_bytes;
    const int padBps = 4;   // pad is float for both formats

    // Shader modules for this bit depth.
    {
        const uint32_t * row_code = nullptr;
        size_t row_size = 0;
        const uint32_t * vc_code = nullptr;
        size_t vc_size = 0;
        switch (d->bits) {
            case 16:
                row_code = eedi3_16_row_spv; row_size = eedi3_16_row_spv_size;
                vc_code = eedi3_16_vcheck_spv; vc_size = eedi3_16_vcheck_spv_size;
                break;
            case 32:
                row_code = eedi3_32_row_spv; row_size = eedi3_32_row_spv_size;
                vc_code = eedi3_32_vcheck_spv; vc_size = eedi3_32_vcheck_spv_size;
                break;
            default:
                return set_error("unsupported bit depth");
        }
        auto r1 = create_shader_module(*d->device, row_code, row_size);
        if (std::holds_alternative<std::string>(r1)) {
            return set_error(std::get<std::string>(r1));
        }
        d->row_module = std::get<VkShaderModule>(r1);
        if (d->vcheck > 0) {
            auto r2 = create_shader_module(*d->device, vc_code, vc_size);
            if (std::holds_alternative<std::string>(r2)) {
                return set_error(std::get<std::string>(r2));
            }
            d->vcheck_module = std::get<VkShaderModule>(r2);
        }
    }

    const int numPlanes = d->vi->format.numPlanes;
    const int subW = d->vi->format.subSamplingW;
    const int subH = d->vi->format.subSamplingH;

    // Plane geometry (output dims; dh doubles the height BEFORE this: d->vi
    // already doubled? No - the filter doubles vi at create AFTER validation.
    // Do it now:
    VSVideoInfo * out_vi = const_cast<VSVideoInfo *>(d->vi);
    // We must NOT modify the const node videoInfo in place (it is shared with
    // the upstream node). Instead build an output vi copy.
    VSVideoInfo out_video = *d->vi;
    if (d->field > 1) {
        if (d->vi->numFrames > INT32_MAX / 2) {
            return set_error("resulting clip is too long");
        }
        out_video.numFrames *= 2;
        vsh::muldivRational(&out_video.fpsNum, &out_video.fpsDen, 2, 1);
    }
    if (d->dh) {
        out_video.height *= 2;
    }

    auto align32 = [](VkDeviceSize v) { return (v + 31) & ~VkDeviceSize(31); };

    VkDeviceSize upload_total = 0;
    VkDeviceSize download_total = 0;
    VkDeviceSize dev_total = 0;

    auto & planes = d->planes;
    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        const int pw = (plane == 0) ? out_video.width : out_video.width >> subW;
        const int ph = (plane == 0) ? out_video.height : out_video.height >> subH;

        cfg.width = pw;
        cfg.height = ph;
        cfg.rows = ph / 2;
        cfg.tpitch = tpitch;
        cfg.pad_stride = (pw + MARGIN_H * 2 + 15) & ~15;   // float elements
        cfg.pad_height = ph + MARGIN_V * 2;

        // staging regions
        cfg.pad_bytes = static_cast<VkDeviceSize>(cfg.pad_stride) * cfg.pad_height * sizeof(float);
        cfg.pad_offset = align32(upload_total);
        upload_total = align32(cfg.pad_offset + cfg.pad_bytes);

        if (d->mclip_node) {
            cfg.bmask_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows;  // uint8
            cfg.bmask_offset = align32(upload_total);
            upload_total = align32(cfg.bmask_offset + cfg.bmask_bytes);
        }
        if (d->vcheck > 0 && d->sclip_node) {
            cfg.sclip_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * elem_bytes;
            cfg.sclip_offset = align32(upload_total);
            upload_total = align32(cfg.sclip_offset + cfg.sclip_bytes);
        }

        // download region (host staging): interp rows only (tight)
        cfg.dl_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * elem_bytes;
        cfg.dl_offset = align32(download_total);
        download_total = align32(cfg.dl_offset + cfg.dl_bytes);

        // device-local regions
        cfg.dst_bytes = cfg.dl_bytes;
        cfg.dst_offset = align32(dev_total);
        dev_total = align32(cfg.dst_offset + cfg.dst_bytes);

        cfg.pbt_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * tpitch;  // int8
        cfg.pbt_offset = align32(dev_total);
        dev_total = align32(cfg.pbt_offset + cfg.pbt_bytes);

        if (d->vcheck > 0) {
            cfg.dmap_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows;      // int8
            cfg.dmap_offset = align32(dev_total);
            dev_total = align32(cfg.dmap_offset + cfg.dmap_bytes);

            if (!(d->vcheck > 0 && d->sclip_node)) {
                cfg.cint_bytes = cfg.dl_bytes;  // io
                cfg.cint_offset = align32(dev_total);
                dev_total = align32(cfg.cint_offset + cfg.cint_bytes);
            }
        }
    }
    d->upload_total = upload_total;
    d->download_total = download_total;
    d->dev_total = dev_total;

    const VkDeviceSize staging_size = std::max(upload_total + download_total, VkDeviceSize(4));
    const VkDeviceSize dev_size = std::max(dev_total, VkDeviceSize(4));

    // Per-width pipelines (deduplicated).
    RowSpecData base_spec {
        .width = 0,
        .nrad = d->nrad,
        .mdis = d->mdis,
        .has_mclip = d->mclip_node ? 1 : 0,
        .has_sclip = (d->vcheck > 0 && d->sclip_node) ? 1 : 0,
        .vcheck = d->vcheck,
        .lsz_row = lsz_row,
        .lsz_vcheck = lsz_vcheck,
    };

    // helper to fetch-or-create the (row, vcheck) pipelines for a width key;
    // uses Eedi3Data::WidthKey (all filter-level params like vcheck/mclip are
    // identical across planes, so the geometry fields dominate)
    using WidthKey = Eedi3Data::WidthKey;
    auto get_pipelines = [&](const WidthKey & key, VkPipeline & row_pipe,
                             VkPipeline & vc_pipe) -> std::optional<std::string> {
        for (auto & [k, pair] : d->width_pipes) {
            if (k == key) {
                row_pipe = pair[0];
                vc_pipe = pair[1];
                return std::nullopt;
            }
        }
        RowSpecData spec = base_spec;
        spec.width = key.width;

        auto r1 = create_pipeline(*d->device, spec, d->row_module, d->pipeline_layout);
        if (std::holds_alternative<std::string>(r1)) {
            return std::get<std::string>(r1);
        }
        VkPipeline rowp = std::get<VkPipeline>(r1);
        VkPipeline vcp = VK_NULL_HANDLE;
        if (d->vcheck > 0) {
            auto r2 = create_pipeline(*d->device, spec, d->vcheck_module, d->pipeline_layout);
            if (std::holds_alternative<std::string>(r2)) {
                vkDestroyPipeline(dev, rowp, nullptr);
                return std::get<std::string>(r2);
            }
            vcp = std::get<VkPipeline>(r2);
        }
        d->width_pipes.emplace_back(key, std::array<VkPipeline, 2>{ rowp, vcp });
        row_pipe = rowp;
        vc_pipe = vcp;
        return std::nullopt;
    };

    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        WidthKey key { cfg.width, cfg.rows, cfg.tpitch, cfg.pad_stride, cfg.pad_height };
        if (auto err = get_pipelines(key, cfg.row_pipeline, cfg.vcheck_pipeline)) {
            return set_error(*err);
        }
    }

    // Resources (one per stream)
    d->pool.semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->pool.reserve(d->num_streams);

    uint32_t num_queues = std::min(
        d->num_streams, static_cast<int>(d->device->queue_count));

    for (int i = 0; i < d->num_streams; ++i) {
        Eedi3Resource resource;

        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
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
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.staging_mem = std::get<AllocatedMemory>(result).memory;
            resource.staging_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = dev_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.dev_buf));
            const auto result = allocate_memory(
                *d->device, resource.dev_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.dev_mem = std::get<AllocatedMemory>(result).memory;
            resource.dev_type_index = std::get<AllocatedMemory>(result).type_index;
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
            VkDescriptorBufferInfo pad_info {
                .buffer = resource.staging, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo dst_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo pbt_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo dmap_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo bmask_info {
                .buffer = resource.staging, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo sclip_info {
                .buffer = resource.staging, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo cint_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            const VkDescriptorBufferInfo * infos[BIND_COUNT] {
                &pad_info, &dst_info, &pbt_info, &dmap_info,
                &bmask_info, &sclip_info, &cint_info
            };
            VkWriteDescriptorSet writes[BIND_COUNT];
            for (uint32_t b = 0; b < BIND_COUNT; ++b) {
                writes[b] = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = b,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = infos[b],
                    .pTexelBufferView = nullptr
                };
            }
            vkUpdateDescriptorSets(dev, BIND_COUNT, writes, 0, nullptr);
        }

        checkVK(vkMapMemory(dev, resource.staging_mem, 0, staging_size, 0,
                            reinterpret_cast<void **>(&resource.map)));

        resource.queue = d->device->queues[i % num_queues].queue;
        resource.queue_lock = d->device->queues[i % num_queues].lock.get();

        // The command buffer is recorded per frame (interp-row parity varies),
        // so the resource is pushed empty; record_command_buffer runs in
        // GetFrame before each submit.
        d->pool.push(std::move(resource));
    }

    // Dependencies: node (strict spatial, or general when field > 1), sclip
    // (strict, only when vcheck > 0), mclip (strict/general).
    std::vector<VSFilterDependency> deps;
    deps.push_back({ d->node, d->field > 1 ? rpGeneral : rpStrictSpatial });
    if (d->vcheck > 0 && d->sclip_node) {
        deps.push_back({ d->sclip_node, rpStrictSpatial });
    }
    if (d->mclip_node) {
        deps.push_back({ d->mclip_node, d->field > 1 ? rpGeneral : rpStrictSpatial });
    }

    // Store the output dims into d->vi? No: createVideoFilter reads the local
    // `out_video`. The data struct keeps a pointer to the ORIGINAL vi (input
    // dims) — the frame processing reads src planes (original dims) and writes
    // the dst frame (output dims). Keep both.
    Eedi3Data * data = d.release();

    vsapi->createVideoFilter(
        out, "EEDI3", &out_video,
        Eedi3GetFrame, Eedi3Free,
        fmParallel, deps.data(), static_cast<int>(deps.size()), data, core);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_eedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "EEDI3",
        "clip:vnode;"
        "field:int;"
        "dh:int:opt;"
        "planes:int[]:opt;"
        "alpha:float:opt;"
        "beta:float:opt;"
        "gamma:float:opt;"
        "nrad:int:opt;"
        "mdis:int:opt;"
        "vcheck:int:opt;"
        "vthresh0:float:opt;"
        "vthresh1:float:opt;"
        "vthresh2:float:opt;"
        "sclip:vnode:opt;"
        "mclip:vnode:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode;",
        Eedi3Create, nullptr, plugin
    );
}
