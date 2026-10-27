#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------
//
// Box runs on the R80 GPU API: `vnode:gpu` in, `ffGPUOutput` out, so the core
// owns every transfer and the filter writes the output frame's planes in
// place. Two dispatches per plane (vertical into a scratch, horizontal out of
// it) with the scratch in the frame's own sample type, because that rounding
// is what makes the result match resize2.

constexpr int BLK_X = 16;
constexpr int BLK_Y = 8;

// (first source tap, tap count) for one output position. See the shader header
// and notes/BOX.md for why this is a table and not a formula.
struct BoxTaps {
    int32_t first {};
    int32_t count {};
};

struct BoxPlaneConfig {
    int src_w {}, src_h {}, dst_w {}, dst_h {};
    int src_stride {}, dst_stride {};
    VkPipeline v_pipeline {}, h_pipeline {};
    uint32_t v_grid_x {}, v_grid_y {}, h_grid_x {}, h_grid_y {};
    uint32_t mid_elem {}; // element offset of this plane's scratch
    bool skip_vert {}; // dst_h == src_h: the horizontal pass reads the source
};

struct BoxData {
    VSNode * node {};
    const VSVideoInfo * vi {}; // input geometry
    VSVideoInfo out_vi {};     // output geometry (same format)

    int bits {}, elem_bytes {};
    bool half {}; // stFloat + 16 bits: a 2-byte element whose sample is a float
    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    std::array<BoxPlaneConfig, 3> planes {};

    VkDeviceSize mid_total {}; // bytes of per-frame two-pass scratch
    GpuBuffer taps {};         // x table then y table
    uint32_t y_tap_base {};    // element offset of the y table
    VSGPUExecPool * pool {};

    ~BoxData() {
        if (!gpu) {
            return;
        }
        // The pool drains every submission it made before it returns, so the
        // pipelines, layouts and buffers below are safe to destroy afterwards.
        if (pool) {
            gpu->api->freeGPUExecPool(pool);
            pool = nullptr;
        }
        VkDevice dev = gpu->device;
        VkPipeline destroyed[6] {};
        int num_destroyed = 0;
        for (auto & plane : planes) {
            const VkPipeline pipelines[2] { plane.v_pipeline,
                                            plane.h_pipeline };
            for (VkPipeline p : pipelines) {
                if (!p) {
                    continue;
                }
                bool seen = false;
                for (int i = 0; i < num_destroyed; ++i) {
                    seen |= destroyed[i] == p;
                }
                if (seen) {
                    continue;
                }
                destroyed[num_destroyed++] = p;
                gpu->vk->vkDestroyPipeline(dev, p, nullptr);
            }
        }
        if (pipeline_layout) {
            gpu->vk->vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            gpu->vk->vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
        gpu_destroy_buffer(*gpu, taps);
    }
};

// ---------------------------------------------------------------------------
// Tap table
// ---------------------------------------------------------------------------

// resize2 is zimg's resizer with the box kernel. A box kernel is not an area
// average: zimg samples the kernel at half-integer source positions around
// `pos = (i + 0.5) / scale`, so at a fractional ratio the tap count alternates
// and an output pixel can average one, two or three source pixels. The set of
// samples that land inside the kernel is decided by zimg's own coordinate
// arithmetic; reproducing that loop, expression for expression, is what makes
// the boundary cases agree. A closed-form "pixel centres in the output cell"
// formula disagrees with it whenever the cell edge lands exactly on a sample
// centre (613 -> 306 and 613 -> 612 are the measured examples).
//
// Included taps all have weight 1 and are contiguous, so the table can stay the
// shader's `(first, count)` pair. `ok` goes false if that ever stops being true
// (for example a mirrored tap landing on an already-counted sample), which is a
// creation error rather than a silently wrong average.
static double round_halfup(double x) noexcept {
    // zimg's rounding, including the epsilon that preserves
    // round(x - 1) == round(x) - 1 on the pixel grid.
    return x < 0.0 ? std::floor(x + 0.5) : std::floor(x + 0.49999999999999994);
}

static std::vector<BoxTaps> box_taps(int src, int dst, bool & ok) {
    std::vector<BoxTaps> out(static_cast<size_t>(dst));
    const double scale = static_cast<double>(dst) / static_cast<double>(src);
    const double step = std::min(scale, 1.0);
    const double support = 1.0 / step;
    const unsigned filter_size =
        std::max(static_cast<unsigned>(std::ceil(support)) * 2U, 1U);
    const double last_pos = std::nextafter(
        static_cast<double>(src), -std::numeric_limits<double>::infinity());

    for (int i = 0; i < dst; ++i) {
        const double pos = (static_cast<double>(i) + 0.5) / scale;
        const double begin =
            round_halfup(pos - static_cast<double>(filter_size) / 2.0) + 0.5;
        int first = 0;
        int count = 0;
        for (unsigned j = 0; j < filter_size; ++j) {
            const double xpos = begin + static_cast<double>(j);
            const double w = (xpos - pos) * step;
            if (w < -0.5 || w >= 0.5) {
                continue; // outside the box kernel
            }
            // zimg mirrors a position that leaves the image and clamps the
            // result; included taps are in bounds for a downscale, but mirror
            // them the same way so the table cannot drift from the reference.
            double real = xpos;
            if (real < 0.0) {
                real = -real;
            } else if (real >= static_cast<double>(src)) {
                real = 2.0 * static_cast<double>(src) - real;
            }
            real = std::clamp(real, 0.0, last_pos);
            const int idx = static_cast<int>(std::floor(real));
            if (count == 0) {
                first = idx;
                count = 1;
            } else if (idx == first + count) {
                ++count;
            } else {
                ok =
                    false; // a gap or a duplicate: equal weights no longer hold
            }
        }
        if (count == 0) {
            ok = false;
            first = std::clamp(static_cast<int>(std::floor(pos)), 0, src - 1);
            count = 1;
        }
        out[static_cast<size_t>(i)] = { first, count };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

struct BoxSpecData {
    int32_t src_w;
    int32_t src_h;
    int32_t dst_w;
    int32_t dst_h;
    int32_t src_stride;
    int32_t dst_stride;
};

static constexpr std::array<VkSpecializationMapEntry, 6> spec_entries { {
    { 0, 0, sizeof(int32_t) },
    { 1, 4, sizeof(int32_t) },
    { 2, 8, sizeof(int32_t) },
    { 3, 12, sizeof(int32_t) },
    { 4, 16, sizeof(int32_t) },
    { 5, 20, sizeof(int32_t) },
} };

static std::variant<VkPipeline, std::string>
create_pipeline(const GPUDevice & gpu, VkPipelineLayout layout,
                const uint32_t * code, size_t code_size,
                const BoxSpecData & spec) {

    const GpuWorkgroup workgroup { .x = BLK_X, .y = BLK_Y, .shared_bytes = 0 };
    return gpu_create_pipeline(gpu, code, code_size, layout,
                               spec_entries.data(), &spec,
                               static_cast<uint32_t>(spec_entries.size()),
                               sizeof(spec), "box", 0, workgroup);
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static const VSFrame * box_gpu_frame(BoxData * d, int n,
                                     VSFrameContext * frameCtx, VSCore * core,
                                     const VSAPI * vsapi) {

    const int numPlanes = d->out_vi.format.numPlanes;
    const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);

    VSFrame * dst = d->gpu->api->newGPUVideoFrame(
        &d->out_vi.format, d->out_vi.width, d->out_vi.height, src, core);
    if (!dst) {
        vsfeel_trace_error("Box", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("Box: failed to allocate the output frame",
                              frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    }

    vsfeel_trace_frame_begin();
    vsfeel_trace_mark("acquire");

    char errbuf[512] {};
    VSGPUExecContext * ctx =
        d->gpu->api->gpuExecAcquire(d->pool, errbuf, sizeof(errbuf));
    auto fail = [&](const std::string & message) -> const VSFrame * {
        if (ctx) {
            d->gpu->api->gpuExecAbandon(ctx);
            ctx = nullptr;
        }
        vsfeel_trace_error("Box", n, message, d->gpu.get());
        vsapi->setFilterError(("Box: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        vsapi->freeFrame(src);
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }

    // One transient scratch per frame, retired by the submission.
    GpuBuffer mid {};
    if (d->mid_total > 0) {
        if (auto e = gpu_frame_buffer(*d->gpu, core, ctx, d->mid_total, mid);
            !e.empty()) {
            return fail("scratch buffer: " + e);
        }
    }

    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);

    // The plane dispatches touch disjoint buffers and disjoint scratch regions,
    // so only the two passes of one plane are ordered.
    for (int p = 0; p < numPlanes; ++p) {
        const auto & cfg = d->planes[p];

        VSVulkanPlaneInfo sp {};
        if (d->gpu->api->getGPUPlane(src, p, &sp)) {
            return fail("source plane " + std::to_string(p) +
                        " is not GPU resident");
        }
        VSVulkanPlaneInfo dp {};
        if (d->gpu->api->getGPUPlane(dst, p, &dp)) {
            return fail("output plane " + std::to_string(p) +
                        " is not GPU resident");
        }

        // A plane whose height is unchanged skips the vertical pass, so it
        // reads the source where the others read the scratch.
        const VkBuffer mid_buffer = cfg.skip_vert ? sp.buffer : mid.buffer;
        const VkBuffer buffers[4] { sp.buffer, dp.buffer, mid_buffer,
                                    d->taps.buffer };
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 4);

        if (!cfg.skip_vert) {
            const int32_t push[4] { 0, 0, static_cast<int32_t>(cfg.mid_elem),
                                    static_cast<int32_t>(d->y_tap_base) };
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push,
                               sizeof(push));
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.v_pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.v_grid_x, cfg.v_grid_y, 1);
            gpu_barrier(*d->gpu, cmd); // the horizontal pass reads these writes
        }

        const int32_t push[4] { 0, 0, static_cast<int32_t>(cfg.mid_elem), 0 };
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push,
                           sizeof(push));
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      cfg.h_pipeline);
        d->gpu->vk->vkCmdDispatch(cmd, cfg.h_grid_x, cfg.h_grid_y, 1);
    }

    d->gpu->api->gpuExecReadsFrame(ctx, src);
    for (int p = 0; p < numPlanes; ++p) {
        d->gpu->api->gpuExecWritesPlane(ctx, dst, p);
    }

    vsfeel_trace_mark("submit");
    uint64_t signaled = 0;
    const int submit_error =
        d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf, sizeof(errbuf));
    ctx = nullptr; // consumed either way
    if (submit_error) {
        return fail("submit failed: "s + errbuf);
    }

    vsapi->freeFrame(src);
    return dst;
}

static const VSFrame * VS_CC BoxGetFrame(int n, int activationReason,
                                         void * instanceData,
                                         [[maybe_unused]] void ** frameData,
                                         VSFrameContext * frameCtx,
                                         VSCore * core, const VSAPI * vsapi) {

    BoxData * d = static_cast<BoxData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    return box_gpu_frame(d, n, frameCtx, core, vsapi);
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BoxFree(void * instanceData, [[maybe_unused]] VSCore * core,
                          [[maybe_unused]] const VSAPI * vsapi) {

    BoxData * d = static_cast<BoxData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC BoxCreate(const VSMap * in, VSMap * out,
                            [[maybe_unused]] void * userData, VSCore * core,
                            const VSAPI * vsapi) {

    auto d { std::make_unique<BoxData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("Box", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("Box: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    const auto & fmt = d->vi->format;
    const int bits = fmt.bitsPerSample;
    const bool depth_ok =
        (fmt.sampleType == stFloat && (bits == 16 || bits == 32)) ||
        (fmt.sampleType == stInteger && bits == 16);
    if (!vsh::isConstantVideoFormat(d->vi) || !depth_ok || d->vi->width <= 0 ||
        d->vi->height <= 0 ||
        (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV &&
         fmt.colorFamily != cfRGB)) {
        return set_error(
            "input bitdepth must be u16 (16-bit integer), f16 or f32 (16/32-bit float), Gray/YUV/RGB.");
    }

    d->bits = bits;
    d->elem_bytes = bits / 8;
    d->half = fmt.sampleType == stFloat && bits == 16;

    const int dst_w = vsapi->mapGetIntSaturated(in, "width", 0, &error);
    const int dst_w_set = !error;
    const int dst_h = vsapi->mapGetIntSaturated(in, "height", 0, &error);
    const int dst_h_set = !error;

    const int out_w = dst_w_set ? dst_w : d->vi->width;
    const int out_h = dst_h_set ? dst_h : d->vi->height;
    if (out_w <= 0 || out_h <= 0) {
        return set_error("width and height must be positive.");
    }
    if (out_w > d->vi->width || out_h > d->vi->height) {
        return set_error(
            "Box only downscales: width/height must not exceed the input.");
    }
    // A subsampled format's frame dimensions must be multiples of its
    // subsampling factors. The output-stride probe frame below is allocated at
    // this size, and VapourSynth treats an illegal size as a fatal error
    // (printing and terminating the process), so the check has to happen here
    // as a normal vs.Error.
    const int subW = fmt.subSamplingW;
    const int subH = fmt.subSamplingH;
    if (subW > 0 && out_w % (1 << subW) != 0) {
        return set_error("width must be a multiple of " +
                         std::to_string(1 << subW) +
                         " for this subsampled format");
    }
    if (subH > 0 && out_h % (1 << subH) != 0) {
        return set_error("height must be a multiple of " +
                         std::to_string(1 << subH) +
                         " for this subsampled format");
    }

    d->out_vi = *d->vi;
    d->out_vi.width = out_w;
    d->out_vi.height = out_h;

    {
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

    // The kernel addresses both planes through signed 32-bit element offsets
    // and needs the core's row pitch, which is the CPU frame's (a GPU frame's
    // stride is the CPU frame's). Read both off scratch frames rather than
    // guessing from an alignment rule.
    int src_stride[3] {}, dst_stride[3] {};
    {
        VSFrame * probe = vsapi->newVideoFrame(&fmt, d->vi->width,
                                               d->vi->height, nullptr, core);
        if (probe == nullptr) {
            return set_error(
                "could not allocate a probe frame to read the input stride");
        }
        for (int p = 0; p < fmt.numPlanes; ++p) {
            src_stride[p] =
                static_cast<int>(vsapi->getStride(probe, p) / d->elem_bytes);
        }
        vsapi->freeFrame(probe);
    }
    {
        VSFrame * probe =
            vsapi->newVideoFrame(&fmt, out_w, out_h, nullptr, core);
        if (probe == nullptr) {
            return set_error(
                "could not allocate a probe frame to read the output stride");
        }
        for (int p = 0; p < fmt.numPlanes; ++p) {
            dst_stride[p] =
                static_cast<int>(vsapi->getStride(probe, p) / d->elem_bytes);
        }
        vsapi->freeFrame(probe);
    }

    // The tap tables depend only on the ratio, which every plane shares, so
    // the plane-0 tables serve the subsampled planes as prefixes.
    bool taps_ok = true;
    const auto x_taps = box_taps(d->vi->width, out_w, taps_ok);
    const auto y_taps = box_taps(d->vi->height, out_h, taps_ok);
    if (!taps_ok) {
        return set_error(
            "could not express the box tap window as one equal-weight run");
    }
    d->y_tap_base = static_cast<uint32_t>(x_taps.size());

    // Push descriptor layout: one whole-buffer binding per shader buffer, so
    // each dispatch rebinds its own view of the planes and nothing is
    // allocated from a descriptor pool.
    {
        const auto result = gpu_push_set_layout(*d->gpu, 4);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result =
            gpu_pipeline_layout(*d->gpu, d->set_layout, 4 * sizeof(int32_t));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // Tap buffer: the two tables concatenated, host visible so creation fills
    // it with one memcpy and the kernels read it every dispatch.
    {
        const VkDeviceSize bytes =
            static_cast<VkDeviceSize>(x_taps.size() + y_taps.size()) *
            sizeof(BoxTaps);
        auto e = gpu_make_buffer(*d->gpu, core,
                                 std::max<VkDeviceSize>(bytes, 8), d->taps,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!e.empty()) {
            return set_error("tap buffer: " + e);
        }
        if (d->taps.mapped == nullptr) {
            return set_error("tap buffer is not host visible");
        }
        auto * map = static_cast<BoxTaps *>(d->taps.mapped);
        std::memcpy(map, x_taps.data(), x_taps.size() * sizeof(BoxTaps));
        std::memcpy(map + x_taps.size(), y_taps.data(),
                    y_taps.size() * sizeof(BoxTaps));
        // The buffer may have landed in the host-visible VRAM BAR, whose
        // mapping is write-combining: the first submission must not read a
        // tail the CPU store buffer has not drained yet.
        store_fence();
    }

    const uint32_t * vert_code = nullptr;
    size_t vert_size = 0;
    const uint32_t * horiz_code = nullptr;
    size_t horiz_size = 0;
    if (d->half) {
        vert_code = box_f16_vert_spv;
        vert_size = box_f16_vert_spv_size;
        horiz_code = box_f16_horiz_spv;
        horiz_size = box_f16_horiz_spv_size;
    } else if (d->bits == 16) {
        vert_code = box_u16_vert_spv;
        vert_size = box_u16_vert_spv_size;
        horiz_code = box_u16_horiz_spv;
        horiz_size = box_u16_horiz_spv_size;
    } else {
        vert_code = box_f32_vert_spv;
        vert_size = box_f32_vert_spv_size;
        horiz_code = box_f32_horiz_spv;
        horiz_size = box_f32_horiz_spv_size;
    }

    const uint32_t max_grid_x = d->gpu->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d->gpu->limits.maxComputeWorkGroupCount[1];

    auto & planes = d->planes;
    VkDeviceSize mid_total = 0;

    for (int plane = 0; plane < fmt.numPlanes; ++plane) {
        BoxPlaneConfig & cfg = planes[plane];

        const int plane_src_w =
            (plane == 0) ? d->vi->width : (d->vi->width >> subW);
        const int plane_src_h =
            (plane == 0) ? d->vi->height : (d->vi->height >> subH);
        const int plane_dst_w = (plane == 0) ? out_w : (out_w >> subW);
        const int plane_dst_h = (plane == 0) ? out_h : (out_h >> subH);

        cfg.src_w = plane_src_w;
        cfg.src_h = plane_src_h;
        cfg.dst_w = plane_dst_w;
        cfg.dst_h = plane_dst_h;
        cfg.src_stride = src_stride[plane];
        cfg.dst_stride = dst_stride[plane];
        cfg.skip_vert = (plane_dst_h == plane_src_h);

        // The kernel addresses the plane through signed 32-bit element
        // offsets; reject a plane whose last element would not fit rather than
        // letting it wrap and read outside the buffer.
        const int64_t last =
            static_cast<int64_t>(plane_src_h - 1) * cfg.src_stride +
            plane_src_w - 1;
        const int64_t last_dst =
            static_cast<int64_t>(plane_dst_h - 1) * cfg.dst_stride +
            plane_dst_w - 1;
        if (last > INT32_MAX || last_dst > INT32_MAX) {
            return set_error(
                "plane " + std::to_string(plane) +
                " is too large for the kernel's 32-bit addressing");
        }

        // The scratch is SRC_W wide (so the horizontal pass reads contiguous
        // taps) and DST_H tall, at the source plane's pitch.
        if (!cfg.skip_vert) {
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(plane_dst_h) *
                                       cfg.src_stride * d->elem_bytes;
            const VkDeviceSize off = align32(mid_total);
            const int64_t last_mid =
                static_cast<int64_t>(off / d->elem_bytes) +
                static_cast<int64_t>(plane_dst_h) * cfg.src_stride;
            if (last_mid > INT32_MAX) {
                return set_error(
                    "plane " + std::to_string(plane) +
                    " scratch overflows the kernel's 32-bit addressing");
            }
            cfg.mid_elem = static_cast<uint32_t>(off / d->elem_bytes);
            mid_total = align32(off + bytes);
        }

        // 64-bit: a width/height near INT32_MAX would otherwise overflow the
        // additions before the quotient widens, and a wrapped grid could pass
        // the `over` test and be dispatched.
        const int64_t v_x = (plane_src_w + BLK_X - 1) / BLK_X;
        const int64_t v_y = (plane_dst_h + BLK_Y - 1) / BLK_Y;
        const int64_t h_x = (plane_dst_w + BLK_X - 1) / BLK_X;
        const int64_t h_y = (plane_dst_h + BLK_Y - 1) / BLK_Y;
        // A 2D grid cannot fold an over-wide plane into Y: refuse it instead
        // of dispatching a clamped grid that leaves pixels unwritten.
        const auto over = [&](int64_t gx, int64_t gy,
                              const char * what) -> std::optional<std::string> {
            if (gx <= max_grid_x && gy <= max_grid_y) {
                return std::nullopt;
            }
            return "plane " + std::to_string(plane) + "'s " + what +
                   " needs a " + std::to_string(gx) + "x" + std::to_string(gy) +
                   " workgroup grid, more than this device can dispatch (" +
                   std::to_string(max_grid_x) + "x" +
                   std::to_string(max_grid_y) + ")";
        };
        if (auto e = over(h_x, h_y, "horizontal pass")) {
            return set_error(*e);
        }
        if (!cfg.skip_vert) {
            if (auto e = over(v_x, v_y, "vertical pass")) {
                return set_error(*e);
            }
        }
        cfg.v_grid_x = static_cast<uint32_t>(v_x);
        cfg.v_grid_y = static_cast<uint32_t>(v_y);
        cfg.h_grid_x = static_cast<uint32_t>(h_x);
        cfg.h_grid_y = static_cast<uint32_t>(h_y);

        const BoxSpecData spec { .src_w = cfg.src_w,
                                 .src_h = cfg.src_h,
                                 .dst_w = cfg.dst_w,
                                 .dst_h = cfg.dst_h,
                                 .src_stride = cfg.src_stride,
                                 .dst_stride = cfg.dst_stride };

        if (!cfg.skip_vert) {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                                                vert_code, vert_size, spec);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            cfg.v_pipeline = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                                                horiz_code, horiz_size, spec);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            cfg.h_pipeline = std::get<VkPipeline>(result);
        }
    }
    d->mid_total = mid_total;

    {
        char err[512] {};
        d->pool =
            d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    BoxData * data = d.release();

    // A spatial filter, so the strict-spatial request pattern is the honest
    // declaration.
    VSFilterDependency deps[1] = { { data->node, rpStrictSpatial } };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "Box", &data->out_vi, BoxGetFrame, BoxFree, fmParallel, ffGPUOutput,
        deps, 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "Box: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_box(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction("Box",
                             "clip:vnode:gpu;"
                             "width:int:opt;"
                             "height:int:opt;",
                             "clip:vnode:gpu;", BoxCreate, nullptr, plugin);
}
