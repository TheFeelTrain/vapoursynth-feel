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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <immintrin.h>

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
// GaussBlur runs on the R80 GPU API: `vnode:gpu` in, `ffGPUOutput` out, so the
// core owns every transfer (std.GPUUpload/GPUDownload) and the filter reads the
// core's frame planes and writes the output frame's planes in place. What it
// owns is its pipelines, the constant kernel-weights buffer and one exec pool
// -- no staging buffers, no per-stream resources, no fences, no queue cap.

// Work-group / code-path constants matching the reference implementations
// (vszipcl gaussglur.zig and vszipcu gaussblur.zig).
constexpr int BLK_X = 16;
constexpr int BLK_Y = 8;
constexpr int VRT = 3;              // output rows per thread, small path
constexpr int LARGE_R = 8;          // outputs per thread, large path
constexpr int LARGE_THRESHOLD = 32; // radius <= 32 => fused small path

struct GaussPlaneConfig {
    int width {};         // visible pixels
    int height {};
    int stride {};        // the core's plane pitch in elements
    int ksize {};         // kernel taps
    int radius {};        // ksize / 2
    bool small {};        // fused small path vs two-pass large path
    VkPipeline pipeline {};   // small path (gauss entry)
    VkPipeline v_pipeline {}; // large path vertical pass
    VkPipeline h_pipeline {}; // large path horizontal pass
    uint32_t grid_x {}, grid_y {};
    uint32_t v_grid_x {}, v_grid_y {};
    uint32_t h_grid_x {}, h_grid_y {};
    VkDeviceSize tmp_elem {}; // float offset of this plane's two-pass scratch
    uint32_t wt_base {};      // float offset into the weights buffer
};

struct GaussData {
    VSNode * node {};
    const VSVideoInfo * vi {};

    int bits {}, elem_bytes {};
    bool process[3] { true, true, true };

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    std::array<GaussPlaneConfig, 3> planes {};

    // All processed planes' kernels concatenated, host visible so creation
    // fills it with one memcpy; the kernels read it every dispatch.
    GpuBuffer wt {};
    // Float scratch of the two-pass path (large planes only): one transient
    // buffer per frame, handed to the recording context.
    VkDeviceSize tmp_total {};
    VSGPUExecPool * pool {};

    // VSFEEL_GAUSS_TIMING=1: per-frame host stage split. Under the API the host
    // side is only acquire/record/submit, but the split still says whether the
    // frame is host- or GPU-bound, which no kernel timing can.
    bool host_timing { false };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_record_ns {}, ht_submit_ns {},
        ht_total_ns {}, ht_n {};

    ~GaussData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[gauss-timing] frames=%.0f per-frame us: acquire=%7.1f "
                "record=%7.1f submit=%7.1f total=%7.1f\n",
                n, ht_acquire_ns.load() / 1000.0 / n,
                ht_record_ns.load() / 1000.0 / n,
                ht_submit_ns.load() / 1000.0 / n, ht_total_ns.load() / 1000.0 / n);
        }
        if (!gpu) {
            return;
        }
        // The pool drains every submission it made before it returns, so the
        // pipelines, layouts and weights below are safe to destroy afterwards.
        if (pool) {
            gpu->api->freeGPUExecPool(pool);
            pool = nullptr;
        }
        VkDevice dev = gpu->device;
        VkPipeline destroyed[9] {};
        int num_destroyed = 0;
        for (auto & plane : planes) {
            const VkPipeline pipelines[3] {
                plane.pipeline, plane.v_pipeline, plane.h_pipeline
            };
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
        gpu_destroy_buffer(*gpu, wt);
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

struct GaussSpecData {
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t ksize;
    int32_t radius;
};

static constexpr std::array<VkSpecializationMapEntry, 5> spec_entries {{
    { 0, 0, sizeof(int32_t) },
    { 1, 4, sizeof(int32_t) },
    { 2, 8, sizeof(int32_t) },
    { 3, 12, sizeof(int32_t) },
    { 4, 16, sizeof(int32_t) },
}};

static std::variant<VkPipeline, std::string> create_pipeline(
    const GPUDevice & gpu, VkPipelineLayout layout,
    const uint32_t * code, size_t code_size, const GaussSpecData & spec) {

    return gpu_create_pipeline(gpu, code, code_size, layout, spec_entries.data(),
        &spec, static_cast<uint32_t>(spec_entries.size()), sizeof(spec),
        "gaussblur");
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// GPU input: the kernels read the source planes and write the output frame's
// planes in place. The core owns every transfer.
static const VSFrame * gauss_gpu_frame(
    GaussData * d, int n, VSFrameContext * frameCtx, VSCore * core,
    const VSAPI * vsapi) {

    const int numPlanes = d->vi->format.numPlanes;
    const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);

    // Creation guarantees at least one processed plane; unprocessed ones ride
    // along from the source frame, keeping their own producer pairs.
    // newVideoFrame2 infers residency from the plane sources, so a frame with no
    // source plane at all has to come from newGPUVideoFrame.
    bool all_process = true;
    for (int p = 0; p < numPlanes; ++p) {
        all_process &= d->process[p];
    }
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
        vsfeel_trace_error("GaussBlur", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("GaussBlur: failed to allocate the output frame",
                              frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
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
        vsfeel_trace_error("GaussBlur", n, message, d->gpu.get());
        vsapi->setFilterError(("GaussBlur: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        vsapi->freeFrame(src);
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    // Two-pass scratch: one transient float buffer per frame, retired by the
    // submission (the pool's size buckets make per-frame allocation cheap).
    // The fused small path never reads it.
    GpuBuffer tmp {};
    if (d->tmp_total > 0) {
        if (auto e = gpu_frame_buffer(*d->gpu, core, ctx, d->tmp_total, tmp);
            !e.empty()) {
            return fail("temp buffer: " + e);
        }
    }

    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);

    // The plane dispatches touch disjoint buffers (and disjoint scratch
    // regions), so no barrier is needed between them; only the two passes of
    // one plane's large path are ordered.
    for (int p = 0; p < numPlanes; ++p) {
        if (!d->process[p]) {
            continue;
        }
        const auto & cfg = d->planes[p];

        VSVulkanPlaneInfo sp {};
        if (d->gpu->api->getGPUPlane(src, p, &sp)) {
            return fail("source plane " + std::to_string(p) + " is not GPU resident");
        }
        VSVulkanPlaneInfo dp {};
        if (d->gpu->api->getGPUPlane(dst, p, &dp)) {
            return fail("output plane " + std::to_string(p) + " is not GPU resident");
        }

        // Binding 4 is the dword view of the destination (packed pair stores
        // of the horizontal pass). Without a two-pass plane there is no
        // scratch to bind, so the source stands in where no shader reads it.
        const VkBuffer buffers[5] {
            d->wt.buffer, sp.buffer, dp.buffer,
            d->tmp_total > 0 ? tmp.buffer : sp.buffer, dp.buffer
        };
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 5);

        // Element offsets into the bound buffers: src and dst are whole plane
        // buffers now, so both bases are 0; only the scratch region and the
        // weights offset carry values.
        const int32_t push[4] {
            0, 0,
            static_cast<int32_t>(cfg.tmp_elem),
            static_cast<int32_t>(cfg.wt_base)
        };
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push, sizeof(push));

        if (cfg.small) {
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.grid_x, cfg.grid_y, 1);
        } else {
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.v_pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.v_grid_x, cfg.v_grid_y, 1);

            // the horizontal pass reads the vertical pass' writes
            gpu_barrier(*d->gpu, cmd);

            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.h_pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.h_grid_x, cfg.h_grid_y, 1);
        }
    }
    auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    d->gpu->api->gpuExecReadsFrame(ctx, src);
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

    vsapi->freeFrame(src);
    return dst;
}

static const VSFrame *VS_CC GaussGetFrame(
    int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    GaussData * d = static_cast<GaussData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    return gauss_gpu_frame(d, n, frameCtx, core, vsapi);
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC GaussFree(
    void *instanceData, [[maybe_unused]] VSCore *core, const VSAPI *vsapi) {

    GaussData * d = static_cast<GaussData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC GaussCreate(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<GaussData>() };

    d->host_timing = vsfeel_debug_probe("VSFEEL_GAUSS_TIMING");

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("GaussBlur", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("GaussBlur: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    const auto & fmt = d->vi->format;
    const int bits = fmt.bitsPerSample;
    const bool depth_ok = (fmt.sampleType == stFloat && bits == 32) ||
                          (fmt.sampleType == stInteger && bits == 16);
    if (!vsh::isConstantVideoFormat(d->vi) || !depth_ok || d->vi->width <= 0 ||
        d->vi->height <= 0 ||
        (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV && fmt.colorFamily != cfRGB)) {
        return set_error("input bitdepth must be 16 (integer) or 32 (float), Gray/YUV/RGB.");
    }

    d->bits = bits;
    d->elem_bytes = bits / 8;

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    // num_streams is a registered no-op: in-flight depth is the core's
    // (exec pool ring) call, so the argument is accepted and never read.

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
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

    // The plane geometry has to be the one the core's GPU frames carry, because
    // kernel addressing is STRIDE elements per row. A GPU frame's stride is the
    // CPU frame's stride, so it is read off a scratch frame here rather than
    // guessed from an alignment rule.
    int plane_stride[3] {};
    {
        VSFrame * probe = vsapi->newVideoFrame(&fmt, d->vi->width, d->vi->height,
                                               nullptr, core);
        if (probe == nullptr) {
            return set_error("could not allocate a probe frame to read the plane stride");
        }
        for (int p = 0; p < fmt.numPlanes; ++p) {
            plane_stride[p] = static_cast<int>(vsapi->getStride(probe, p) / d->elem_bytes);
        }
        vsapi->freeFrame(probe);
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

    for (int plane = 0; plane < fmt.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }

        const int plane_width = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        const int plane_height = (plane == 0) ? d->vi->height : d->vi->height >> subH;
        const int stride = plane_stride[plane];

        // The kernel addresses the plane through signed 32-bit element
        // offsets; reject a plane whose last element would not fit rather than
        // letting it wrap and read outside the buffer.
        const int64_t last = static_cast<int64_t>(plane_height - 1) * stride + plane_width - 1;
        if (last > INT32_MAX) {
            return set_error("plane " + std::to_string(plane) + " is too large: " +
                std::to_string(plane_width) + "x" + std::to_string(plane_height) +
                " at stride " + std::to_string(stride) +
                " overflows the kernel's 32-bit addressing");
        }

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

    // Code path per config: the fused small path only when its shared-memory
    // tile fits the device.
    std::array<bool, 3> cfg_small {};
    for (int ci = 0; ci < n_cfg; ++ci) {
        const int ksize = static_cast<int>(weights[ci].size());
        const int radius = ksize / 2;
        const size_t tile_bytes =
            static_cast<size_t>(VRT) * BLK_Y * (BLK_X + 2 * radius) * sizeof(float);
        cfg_small[ci] = radius <= LARGE_THRESHOLD &&
            tile_bytes <= std::min<size_t>(48 * 1024, d->gpu->limits.maxComputeSharedMemorySize);
    }

    // Push descriptor layout: one whole-buffer binding per shader buffer, so
    // each dispatch rebinds its own view of the planes and nothing is
    // allocated from a descriptor pool.
    {
        const auto result = gpu_push_set_layout(*d->gpu, 5);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result = gpu_pipeline_layout(*d->gpu, d->set_layout,
            4 * sizeof(int32_t));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // Weights buffer: all configs' kernels concatenated.
    {
        VkDeviceSize wt_bytes = 0;
        for (const auto & w : weights) {
            wt_bytes += static_cast<VkDeviceSize>(w.size()) * sizeof(float);
        }
        wt_bytes = std::max<VkDeviceSize>(wt_bytes, 4);

        auto e = gpu_make_buffer(*d->gpu, core, wt_bytes, d->wt,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!e.empty()) {
            return set_error("weights buffer: " + e);
        }
        if (d->wt.mapped == nullptr) {
            return set_error("weights buffer is not host visible");
        }
        auto * map = static_cast<float *>(d->wt.mapped);
        uint32_t wt_base = 0;
        for (int ci = 0; ci < n_cfg; ++ci) {
            std::memcpy(map + wt_base, weights[ci].data(),
                weights[ci].size() * sizeof(float));
            wt_base += static_cast<uint32_t>(weights[ci].size());
        }
        // The buffer may have landed in the host-visible VRAM BAR, whose
        // mapping is write-combining: the first submission must not read a
        // tail the CPU store buffer has not drained yet.
        _mm_sfence();
    }

    // Shader blobs, selected by bit depth (the entry point by code path).
    const uint32_t * gauss_code = nullptr;
    size_t gauss_size = 0;
    const uint32_t * vert_code = nullptr;
    size_t vert_size = 0;
    const uint32_t * horiz_code = nullptr;
    size_t horiz_size = 0;
    switch (d->bits) {
        case 16:
            gauss_code = gaussblur_16_gauss_spv;  gauss_size = gaussblur_16_gauss_spv_size;
            vert_code = gaussblur_16_vert_spv;    vert_size = gaussblur_16_vert_spv_size;
            horiz_code = gaussblur_16_horiz_spv;  horiz_size = gaussblur_16_horiz_spv_size;
            break;
        default:
            gauss_code = gaussblur_32_gauss_spv;  gauss_size = gaussblur_32_gauss_spv_size;
            vert_code = gaussblur_32_vert_spv;    vert_size = gaussblur_32_vert_spv_size;
            horiz_code = gaussblur_32_horiz_spv;  horiz_size = gaussblur_32_horiz_spv_size;
            break;
    }

    const uint32_t max_grid_x = d->gpu->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d->gpu->limits.maxComputeWorkGroupCount[1];

    // Per-plane pipelines, grids and scratch layout. A plane identical to an
    // earlier one reuses its pipelines and offsets outright.
    auto & planes = d->planes;
    VkDeviceSize tmp_total = 0;
    uint32_t wt_running = 0;

    for (int plane = 0; plane < fmt.numPlanes; ++plane) {
        if (!plane_valid[plane]) {
            continue;
        }
        const int ci = plane_cfg[plane];
        GaussPlaneConfig & cfg = planes[plane];

        // Two-pass scratch region, assigned per plane even when the pipelines
        // are reused: identical planes' dispatches are unordered within one
        // command buffer, so sharing one region would race (WAR between the
        // reused plane's horizontal pass and this one's vertical pass).
        VkDeviceSize plane_tmp_elem = 0;
        if (!cfg_small[ci]) {
            const auto & pkey = plane_keys[plane];
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(pkey.h) * pkey.stride * sizeof(float);
            const VkDeviceSize off = align32(tmp_total);
            const int64_t last_tmp =
                static_cast<int64_t>(off / 4) +
                static_cast<int64_t>(pkey.h) * pkey.stride;
            if (last_tmp > INT32_MAX) {
                return set_error("plane " + std::to_string(plane) +
                    " two-pass scratch overflows the kernel's 32-bit addressing");
            }
            plane_tmp_elem = off / 4;
            tmp_total = align32(off + bytes);
        }

        bool found = false;
        for (int other = 0; other < plane; ++other) {
            if (plane_valid[other] && plane_cfg[other] == ci) {
                cfg = planes[other];
                cfg.tmp_elem = plane_tmp_elem;
                found = true;
                break;
            }
        }
        if (found) {
            continue;
        }

        const auto & key = plane_keys[plane];
        const int ksize = static_cast<int>(weights[ci].size());
        const int radius = ksize / 2;

        cfg.width = key.w;
        cfg.height = key.h;
        cfg.stride = key.stride;
        cfg.ksize = ksize;
        cfg.radius = radius;
        cfg.small = cfg_small[ci];
        cfg.tmp_elem = plane_tmp_elem;

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

        cfg.wt_base = wt_running;
        wt_running += static_cast<uint32_t>(weights[ci].size());

        const GaussSpecData spec {
            .width = cfg.width,
            .height = cfg.height,
            .stride = cfg.stride,
            .ksize = cfg.ksize,
            .radius = cfg.radius
        };

        if (cfg.small) {
            const auto result = create_pipeline(
                *d->gpu, d->pipeline_layout, gauss_code, gauss_size, spec);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            cfg.pipeline = std::get<VkPipeline>(result);
        } else {
            {
                const auto result = create_pipeline(
                    *d->gpu, d->pipeline_layout, vert_code, vert_size, spec);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                cfg.v_pipeline = std::get<VkPipeline>(result);
            }
            {
                const auto result = create_pipeline(
                    *d->gpu, d->pipeline_layout, horiz_code, horiz_size, spec);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                cfg.h_pipeline = std::get<VkPipeline>(result);
            }
        }
    }
    d->tmp_total = tmp_total;

    {
        char err[512] {};
        d->pool = d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    GaussData * data = d.release();

    // A spatial filter, so the strict-spatial request pattern is the honest
    // declaration.
    VSFilterDependency deps[1] = {{ data->node, rpStrictSpatial }};

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "GaussBlur", data->vi,
        GaussGetFrame, GaussFree,
        fmParallel, ffGPUOutput, deps, 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "GaussBlur: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_gaussblur(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "GaussBlur",
        "clip:vnode:gpu;"
        "sigma:float[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode:gpu;",
        GaussCreate, nullptr, plugin
    );
}
