#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <string>
#include <variant>
#include <vector>

#include <volk.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------
//
// Bilateral is a pure per-pixel gather over the frame's own planes. It takes
// `vnode:gpu` and returns `vnode:gpu` with ffGPUOutput: the kernels read the
// core's GPU frame planes and write the output frame's planes in place, and the
// core owns every transfer (std.GPUUpload/GPUDownload). The filter owns no copy
// of the pixels — one pipeline per plane and one exec pool is all of it.

struct BilateralPlaneConfig {
    int width {};          // visible pixels
    int height {};
    int stride {};         // the core's plane pitch in elements
    VkPipeline pipeline {};
    uint32_t grid_x {};
    uint32_t grid_y {};
};

struct BilateralData {
    VSNode * node {};
    VSNode * ref_node {};   // optional guide clip
    const VSVideoInfo * vi {};

    int bits {}, elem_bytes {};
    bool process[3] { true, true, true };

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    std::array<BilateralPlaneConfig, 3> planes {};
    VSGPUExecPool * pool {};

    // VSFEEL_BILAT_TIMING=1: per-frame host stage split. Under the API the host
    // side is only acquire/record/submit, but the split still says whether the
    // frame is host- or GPU-bound, which no kernel timing can.
    bool host_timing { false };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_record_ns {}, ht_submit_ns {},
        ht_total_ns {}, ht_n {};

    ~BilateralData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[bilat-timing] frames=%.0f per-frame us: acquire=%7.1f "
                "record=%7.1f submit=%7.1f total=%7.1f\n",
                n, ht_acquire_ns.load() / 1000.0 / n,
                ht_record_ns.load() / 1000.0 / n,
                ht_submit_ns.load() / 1000.0 / n, ht_total_ns.load() / 1000.0 / n);
        }
        if (!gpu) {
            return;
        }
        // The pool drains every submission it made before it returns, so the
        // pipelines and layouts below are safe to destroy afterwards.
        if (pool) {
            gpu->api->freeGPUExecPool(pool);
            pool = nullptr;
        }
        VkDevice dev = gpu->device;
        VkPipeline destroyed[3] {};
        int num_destroyed = 0;
        for (auto & plane : planes) {
            if (!plane.pipeline) {
                continue;
            }
            bool seen = false;
            for (int i = 0; i < num_destroyed; ++i) {
                seen |= destroyed[i] == plane.pipeline;
            }
            if (seen) {
                continue;
            }
            destroyed[num_destroyed++] = plane.pipeline;
            gpu->vk->vkDestroyPipeline(dev, plane.pipeline, nullptr);
        }
        if (pipeline_layout) {
            gpu->vk->vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            gpu->vk->vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
    }
};

// ---------------------------------------------------------------------------
// Shader specialization constants
// ---------------------------------------------------------------------------

struct BilateralSpecData {
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

static std::variant<VkPipeline, std::string> create_pipeline(
    const GPUDevice & dev, bool use_shared, const BilateralSpecData & spec,
    const uint32_t * code, size_t code_size, VkPipelineLayout layout) {

    const VkSpecializationMapEntry * entries;
    uint32_t entry_count;
    if (use_shared) {
        entries = shared_entries.data();
        entry_count = static_cast<uint32_t>(std::size(shared_entries));
    } else {
        entries = plain_entries.data();
        entry_count = static_cast<uint32_t>(std::size(plain_entries));
    }
    return gpu_create_pipeline(dev, code, code_size, layout, entries, &spec,
        entry_count, sizeof(spec), "bilateral");
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static void bilateral_process_mask(const BilateralData & d, int numPlanes,
                                   bool & any_process, bool & all_process) {
    any_process = false;
    all_process = true;
    for (int p = 0; p < numPlanes; ++p) {
        any_process |= d.process[p];
        all_process &= d.process[p];
    }
}

// GPU input: the kernels read the source (and guide) planes and write the output
// frame's planes in place. The core owns every transfer.
static const VSFrame * bilateral_gpu_frame(
    BilateralData * d, int n, VSFrameContext * frameCtx, VSCore * core,
    const VSAPI * vsapi) {

    const int numPlanes = d->vi->format.numPlanes;
    const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSFrame * ref = d->ref_node
        ? vsapi->getFrameFilter(n, d->ref_node, frameCtx) : nullptr;

    bool any_process = false, all_process = true;
    bilateral_process_mask(*d, numPlanes, any_process, all_process);

    // Unprocessed planes ride along from the source frame, keeping their own
    // producer pairs; everything processed is written by this submission.
    // newVideoFrame2 infers residency from the plane sources, so a frame with no
    // source plane at all has to come from newGPUVideoFrame.
    const int pl[] = { 0, 1, 2 };
    const VSFrame * fr[] = {
        d->process[0] ? nullptr : src,
        d->process[1] ? nullptr : src,
        d->process[2] ? nullptr : src
    };
    VSFrame * dst = all_process
        ? d->gpu->api->newGPUVideoFrame(&d->vi->format, d->vi->width,
              d->vi->height, src, core)
        : vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height,
              fr, pl, src, core);
    if (!dst) {
        vsfeel_trace_error("BilateralVK", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("BilateralVK: failed to allocate the output frame", frameCtx);
        if (ref) {
            vsapi->freeFrame(ref);
        }
        vsapi->freeFrame(src);
        return nullptr;
    }

    // Nothing to run: every plane shares from the source, so the frame is
    // already complete and an empty submission would only cost a round trip.
    if (!any_process) {
        if (ref) {
            vsapi->freeFrame(ref);
        }
        vsapi->freeFrame(src);
        return dst;
    }

    auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};
    vsfeel_trace_frame_begin();
    vsfeel_trace_mark("acquire");

    char errbuf[512] {};
    VSGPUExecContext * ctx = d->gpu->api->gpuExecAcquire(d->pool, errbuf, sizeof(errbuf));
    auto fail = [&](const std::string & message) -> const VSFrame * {
        if (ctx) {
            d->gpu->api->gpuExecAbandon(ctx);
            ctx = nullptr;
        }
        vsfeel_trace_error("BilateralVK", n, message, d->gpu.get());
        vsapi->setFilterError(("BilateralVK: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        if (ref) {
            vsapi->freeFrame(ref);
        }
        vsapi->freeFrame(src);
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    // The kernels read the source (and the guide) and write the output planes
    // in place; each plane is a disjoint dispatch, so no barrier is needed.
    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
    for (int p = 0; p < numPlanes; ++p) {
        if (!d->process[p]) {
            continue;
        }
        VSVulkanPlaneInfo src_plane {};
        if (d->gpu->api->getGPUPlane(src, p, &src_plane)) {
            return fail("source plane " + std::to_string(p) + " is not GPU resident");
        }
        VSVulkanPlaneInfo dst_plane {};
        if (d->gpu->api->getGPUPlane(dst, p, &dst_plane)) {
            return fail("output plane " + std::to_string(p) + " is not GPU resident");
        }
        VkBuffer ref_buffer = src_plane.buffer;
        if (d->ref_node) {
            VSVulkanPlaneInfo ref_plane {};
            if (d->gpu->api->getGPUPlane(ref, p, &ref_plane)) {
                return fail("guide plane " + std::to_string(p) + " is not GPU resident");
            }
            ref_buffer = ref_plane.buffer;
        }

        const auto & cfg = d->planes[p];
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pipeline);
        // binding 2 is the guide; without one the source stands in for it and
        // the shader's HAS_REF==0 path never reads it
        const VkBuffer buffers[3] { src_plane.buffer, dst_plane.buffer, ref_buffer };
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 3);
        d->gpu->vk->vkCmdDispatch(cmd, cfg.grid_x, cfg.grid_y, 1);
    }
    auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    d->gpu->api->gpuExecReadsFrame(ctx, src);
    if (d->ref_node) {
        d->gpu->api->gpuExecReadsFrame(ctx, ref);
    }
    for (int p = 0; p < numPlanes; ++p) {
        if (d->process[p]) {
            d->gpu->api->gpuExecWritesPlane(ctx, dst, p);
        }
    }

    vsfeel_trace_mark("submit");
    uint64_t signaled = 0;
    const int submit_error = d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf, sizeof(errbuf));
    ctx = nullptr;  // consumed either way
    if (submit_error) {
        return fail("submit failed: "s + errbuf);
    }
    auto t3 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    if (d->host_timing) {
        const auto ns = [](auto a, auto b) {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
        };
        d->ht_acquire_ns += ns(t0, t1);
        d->ht_record_ns += ns(t1, t2);
        d->ht_submit_ns += ns(t2, t3);
        d->ht_total_ns += ns(t0, t3);
        d->ht_n.fetch_add(1, std::memory_order_relaxed);
    }

    if (ref) {
        vsapi->freeFrame(ref);
    }
    vsapi->freeFrame(src);
    return dst;
}

static const VSFrame *VS_CC BilateralGetFrame(
    int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->ref_node) {
            vsapi->requestFrameFilter(n, d->ref_node, frameCtx);
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    return bilateral_gpu_frame(d, n, frameCtx, core, vsapi);
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BilateralFree(
    void *instanceData, [[maybe_unused]] VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC BilateralCreate(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BilateralData>() };

    // Opt-in host-path probe: the default path records no clocks.
    d->host_timing = vsfeel_debug_probe("VSFEEL_BILAT_TIMING");

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
    const bool has_ref = d->ref_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("BilateralVK", -1, error_message, d->gpu.get());
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

    if (has_ref) {
        const auto ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (!vsh::isSameVideoInfo(d->vi, ref_vi) ||
            d->vi->numFrames != ref_vi->numFrames) {
            return set_error("\"ref\" must be of the same format and dimensions as \"clip\"");
        }
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
        } else if (!std::isfinite(sigma_spatial[i]) || sigma_spatial[i] < 0.f) {
            return set_error("\"sigma_spatial\" must be finite and non-negative");
        }

        if (sigma_spatial[i] < FLT_EPSILON) {
            d->process[i] = false;
        }
    }

    std::array<float, 3> sigma_spatial_scaled;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial_scaled[i] = (-0.5f / (sigma_spatial[i] * sigma_spatial[i])) *
            std::numbers::log2e_v<float>;
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
        } else if (!std::isfinite(sigma_color[i]) || sigma_color[i] < 0.f) {
            return set_error("\"sigma_color\" must be finite and non-negative");
        }
    }

    std::array<float, 3> sigma_color_scaled;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            sigma_color_scaled[i] = (-0.5f / (sigma_color[i] * sigma_color[i])) *
                std::numbers::log2e_v<float>;
        }
    }

    std::array<int, 3> radius;
    for (int i = 0; i < std::ssize(radius); ++i) {
        radius[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", i, &error));

        if (error) {
            // clamp before the cast: a huge finite sigma would otherwise make
            // the float-to-int conversion undefined (the reference clamps too)
            radius[i] = std::max(1, static_cast<int>(
                std::min(std::roundf(sigma_spatial[i] * 3.f), 1000000.f)));
        } else if (radius[i] <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    // device_id and num_streams are registered but never read: the core owns
    // the one device and sizes in-flight depth itself (exec pool ring).

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
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

    {
        const VkPhysicalDeviceLimits & limits = d->gpu->limits;

        // shrink the default block size if the device cannot host it
        if (static_cast<uint32_t>(block_x) > limits.maxComputeWorkGroupSize[0] ||
            static_cast<uint32_t>(block_y) > limits.maxComputeWorkGroupSize[1] ||
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

    // Push descriptors: each dispatch rebinds its own view of the planes, so
    // nothing is allocated from a pool and nothing survives the command buffer.
    {
        const auto result = gpu_push_set_layout(*d->gpu, 3);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result = gpu_pipeline_layout(*d->gpu, d->set_layout, 0);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // The plane geometry has to be the one the core's GPU frames carry, because
    // kernel addressing is STRIDE elements per row. A GPU frame's stride is the
    // CPU frame's stride (the core stores planes exactly as the CPU allocator
    // would), so it is read off a scratch frame here rather than guessed from an
    // alignment rule.
    int plane_stride[3] {};
    {
        VSFrame * probe = vsapi->newVideoFrame(&d->vi->format, d->vi->width,
                                               d->vi->height, nullptr, core);
        if (probe == nullptr) {
            return set_error("could not allocate a probe frame to read the plane stride");
        }
        for (int p = 0; p < d->vi->format.numPlanes; ++p) {
            plane_stride[p] = static_cast<int>(vsapi->getStride(probe, p) / d->elem_bytes);
        }
        vsapi->freeFrame(probe);
    }

    int width = d->vi->width;
    int height = d->vi->height;
    int ssw = d->vi->format.subSamplingW;
    int ssh = d->vi->format.subSamplingH;

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

    std::array<BilateralPlaneConfig, 3> & planes = d->planes;

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }

        const int plane_width { plane == 0 ? width : width >> ssw };
        const int plane_height { plane == 0 ? height : height >> ssh };
        const int stride = plane_stride[plane];

        // The kernel addresses a plane through signed 32-bit element offsets;
        // reject a plane whose last element would not fit rather than letting it
        // wrap and write outside the buffer.
        const int64_t last = static_cast<int64_t>(plane_height - 1) * stride + plane_width - 1;
        if (last > INT32_MAX) {
            return set_error("plane " + std::to_string(plane) + " is too large: " +
                std::to_string(plane_width) + "x" + std::to_string(plane_height) +
                " at stride " + std::to_string(stride) +
                " overflows the kernel's 32-bit addressing");
        }

        pipeline_keys[plane] = {
            plane_width, plane_height, stride,
            sigma_spatial_scaled[plane], sigma_color_scaled[plane], radius[plane]
        };
        pipeline_valid[plane] = true;
    }

    const uint32_t max_grid_x = d->gpu->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d->gpu->limits.maxComputeWorkGroupCount[1];

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!pipeline_valid[plane]) {
            continue;
        }

        auto & cfg = planes[plane];
        const auto & key = pipeline_keys[plane];

        cfg.width = key.width;
        cfg.height = key.height;
        cfg.stride = key.stride;

        cfg.grid_x = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.width - 1) / block_x + 1, static_cast<int64_t>(max_grid_x)));
        cfg.grid_y = static_cast<uint32_t>(std::min<int64_t>(
            (cfg.height - 1) / block_y + 1, static_cast<int64_t>(max_grid_y)));

        const int tile_x = 2 * key.radius + block_x;
        const int tile_y = 2 * key.radius + block_y;
        // the shared kernel keeps the source tile plus the guide tile (if any);
        // the output goes straight to dst[] — there is no output tile
        const size_t shared_bytes =
            static_cast<size_t>(1 + has_ref) * tile_x * tile_y * sizeof(float);

        // gate on the device's real LDS limit: the old hardcoded 48 KiB cap sent
        // wide radii that still fit to the ~4x slower plain kernel
        plane_shared[plane] = use_shared_memory &&
            shared_bytes <= d->gpu->limits.maxComputeSharedMemorySize;
    }

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
        const size_t shared_bytes =
            static_cast<size_t>(1 + has_ref) * tile_x * tile_y * sizeof(float);

        BilateralSpecData spec {
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
            *d->gpu, plane_shared[plane], spec,
            plane_shared[plane] ? shared_code : plain_code,
            plane_shared[plane] ? shared_size : plain_size,
            d->pipeline_layout);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        cfg.pipeline = std::get<VkPipeline>(result);
    }

    {
        char err[512] {};
        d->pool = d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    BilateralData *data = d.release();

    // A spatial filter, so the strict-spatial request pattern is the honest
    // declaration for both inputs.
    VSFilterDependency deps[2] = {
        { data->node, rpStrictSpatial },
        { data->ref_node, rpStrictSpatial }
    };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "Bilateral", data->vi,
        BilateralGetFrame, BilateralFree,
        fmParallel, ffGPUOutput, deps, data->ref_node ? 2 : 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "BilateralVK: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bilateral(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "Bilateral",
        "clip:vnode:gpu;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "use_shared_memory:int:opt;"
        "block_x:int:opt;"
        "block_y:int:opt;"
        "ref:vnode:gpu:opt;",
        "clip:vnode:gpu;",
        BilateralCreate, nullptr, plugin
    );
}
