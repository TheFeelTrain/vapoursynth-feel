#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <volk.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// DFTTest — Vulkan port of the vszipcu dfttest (dfttest.zig + dfttest.cu)
// and the vs-dfttest2 hiprtc backend (dft_kernels.hpp + kernel.hpp).
//
// Pipeline per processed plane:
//   pad:     reflect-pad each of the (2*radius+1) GPU source frame planes into
//            a device-local buffer (one dispatch per temporal slice)
//   fused:   per 16x16 block, im2col + window, 3D DFT, frequency filter,
//            inverse DFT, writes the center temporal slice to a float buffer
//   col2im:  overlap-adds the windowed blocks straight into the output plane
// ---------------------------------------------------------------------------

constexpr int BS = 16;              // spatial block size (sbsize, fixed)

// ---------------------------------------------------------------------------
// Host-side window / sigma table math (ported from vszipcu dfttest.zig)
// ---------------------------------------------------------------------------

static double besselI0(double p_in) {
    const double p = p_in / 2.0;
    double n = 1.0;
    double t = 1.0;
    double d = 1.0;
    int k = 1;
    while (true) {
        n *= p;
        d *= static_cast<double>(k);
        const double v = n / d;
        t += v * v;
        k += 1;
        if (k >= 15 || v <= 1e-8) {
            break;
        }
    }
    return t;
}

static double getWindowValue(double location, int size, int mode, double beta) {
    const double size_f = static_cast<double>(size);
    const double temp = std::numbers::pi * location / size_f;
    switch (mode) {
        case 0: return 0.5 * (1.0 - std::cos(2.0 * temp));
        case 1: return 0.53836 - 0.46164 * std::cos(2.0 * temp);
        case 2: return 0.42 - 0.5 * std::cos(2.0 * temp) + 0.08 * std::cos(4.0 * temp);
        case 3: return 0.35875 - 0.48829 * std::cos(2.0 * temp) + 0.14128 * std::cos(4.0 * temp)
                - 0.01168 * std::cos(6.0 * temp);
        case 4: {
            const double v = 2.0 * location / size_f - 1.0;
            return besselI0(std::numbers::pi * beta * std::sqrt(1.0 - v * v)) /
                   besselI0(std::numbers::pi * beta);
        }
        case 5: return 0.27105140069342415
                - 0.433297939234486060 * std::cos(2.0 * temp)
                + 0.218122999543110620 * std::cos(4.0 * temp)
                - 0.065925446388030898 * std::cos(6.0 * temp)
                + 0.010811742098372268 * std::cos(8.0 * temp)
                - 7.7658482522509342e-4 * std::cos(10.0 * temp)
                + 1.3887217350903198e-5 * std::cos(12.0 * temp);
        case 6: return 0.2810639 - 0.5208972 * std::cos(2.0 * temp) + 0.1980399 * std::cos(4.0 * temp);
        case 7: return 1.0;
        case 8: return 1.0 - 2.0 * std::abs(location - size_f / 2.0) / size_f;
        case 9: return 0.62 - 0.48 * (location / size_f - 0.5) - 0.38 * std::cos(2.0 * temp);
        case 10: return 0.355768 - 0.487396 * std::cos(2.0 * temp) + 0.144232 * std::cos(4.0 * temp)
                 - 0.012604 * std::cos(6.0 * temp);
        case 11: return 0.3635819 - 0.4891775 * std::cos(2.0 * temp) + 0.1365995 * std::cos(4.0 * temp)
                 - 0.0106411 * std::cos(6.0 * temp);
        default: return 0.0;
    }
}

// normalizeWindow with a fixed 16-entry buffer (window must be 16 wide)
static void normalizeWindow(double * window, int size, int step) {
    double nw[16] {};
    for (int q = 0; q < size; ++q) {
        for (int h = q; h >= 0; h -= step) {
            nw[q] += window[h] * window[h];
        }
        for (int h = q + step; h < size; h += step) {
            nw[q] += window[h] * window[h];
        }
    }
    for (int q = 0; q < size; ++q) {
        window[q] = window[q] / std::sqrt(nw[q]);
    }
}

static std::vector<double> getWindow(
    int radius, int block_step, int swin, double sbeta, int twin, double tbeta) {

    const int tw = 2 * radius + 1;

    double temporal[7] {};
    for (int i = 0; i < tw; ++i) {
        temporal[i] = getWindowValue(static_cast<double>(i) + 0.5, tw, twin, tbeta);
    }

    double spatial[16] {};
    for (int i = 0; i < 16; ++i) {
        spatial[i] = getWindowValue(static_cast<double>(i) + 0.5, BS, swin, sbeta);
    }
    normalizeWindow(spatial, BS, block_step);

    std::vector<double> window(static_cast<size_t>(tw) * 256);
    const double div = std::sqrt(static_cast<double>(tw)) * static_cast<double>(BS);
    size_t idx = 0;
    for (int t = 0; t < tw; ++t) {
        for (int s1 = 0; s1 < 16; ++s1) {
            for (int s2 = 0; s2 < 16; ++s2) {
                window[idx] = (temporal[t] * spatial[s1] * spatial[s2]) / div;
                idx += 1;
            }
        }
    }
    return window;
}

// Shewchuk exact sum (the reference uses this for the wscale, plain `+=`
// differs by a ulp).
static double fsum(const double * values, size_t n) {
    double partials[64] {};
    size_t n_partials = 0;
    for (size_t item = 0; item < n; ++item) {
        double x = values[item];
        size_t i = 0;
        for (size_t j = 0; j < n_partials; ++j) {
            double y = partials[j];
            if (std::abs(x) < std::abs(y)) {
                std::swap(x, y);
            }
            const double hi = x + y;
            const double lo = y - (hi - x);
            if (lo != 0.0) {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials[i] = x;
        n_partials = i + 1;
    }
    if (n_partials == 0) {
        return 0.0;
    }
    double hi = partials[n_partials - 1];
    double lo = 0.0;
    size_t k = n_partials - 1;
    while (k > 0) {
        const double x = hi;
        const double y = partials[k - 1];
        k -= 1;
        hi = x + y;
        const double yr = hi - x;
        lo = y - yr;
        if (lo != 0.0) {
            break;
        }
    }
    if (k > 0 && ((lo < 0.0 && partials[k - 1] < 0.0) || (lo > 0.0 && partials[k - 1] > 0.0))) {
        const double y2 = lo * 2.0;
        const double x2 = hi + y2;
        if (y2 == x2 - hi) {
            hi = x2;
        }
    }
    return hi;
}

struct Complex {
    double re, im;
};

static void dftReal(Complex * dst, size_t dst_stride, const double * src, size_t src_stride, int n) {
    const int out_num = n / 2 + 1;
    for (int i = 0; i < out_num; ++i) {
        Complex sum {};
        for (int j = 0; j < n; ++j) {
            const double imag = static_cast<double>(-2 * i * j) * std::numbers::pi / static_cast<double>(n);
            const double s = src[static_cast<size_t>(j) * src_stride];
            sum.re += s * std::cos(imag);
            sum.im += s * std::sin(imag);
        }
        dst[static_cast<size_t>(i) * dst_stride] = sum;
    }
}

static void dftCplx(Complex * dst, const Complex * src, int n, size_t stride) {
    Complex out[16] {};
    for (int i = 0; i < n; ++i) {
        Complex sum {};
        for (int j = 0; j < n; ++j) {
            const double imag = static_cast<double>(-2 * i * j) * std::numbers::pi / static_cast<double>(n);
            const double wre = std::cos(imag);
            const double wim = std::sin(imag);
            const Complex & s = src[static_cast<size_t>(j) * stride];
            sum.re += s.re * wre - s.im * wim;
            sum.im += s.re * wim + s.im * wre;
        }
        out[i] = sum;
    }
    for (int i = 0; i < n; ++i) {
        dst[static_cast<size_t>(i) * stride] = out[i];
    }
}

// 3D real DFT of the (tw x 16 x 16) window; returns tw*16*9 complex pairs.
static std::vector<double> rdftTables(int radius, const std::vector<double> & input) {
    const size_t tw = static_cast<size_t>(2 * radius + 1);
    const size_t cols = 9;
    const size_t csize = tw * 16 * cols;

    std::vector<Complex> output(csize);
    std::vector<Complex> output2(csize);

    if (radius == 0) {
        for (size_t i = 0; i < 16; ++i) {
            dftReal(output.data() + i * cols, 1, input.data() + i * 16, 1, BS);
        }
        for (size_t i = 0; i < cols; ++i) {
            dftCplx(output2.data() + i, output.data() + i, BS, cols);
        }
        std::vector<double> ret(csize * 2);
        for (size_t i = 0; i < csize; ++i) {
            ret[i * 2] = output2[i].re;
            ret[i * 2 + 1] = output2[i].im;
        }
        return ret;
    }

    for (size_t i = 0; i < tw * 16; ++i) {
        dftReal(output.data() + i * cols, 1, input.data() + i * 16, 1, BS);
    }
    for (size_t i = 0; i < tw; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dftCplx(output2.data() + i * 16 * cols + j, output.data() + i * 16 * cols + j, BS, cols);
        }
    }
    for (size_t i = 0; i < 16 * cols; ++i) {
        dftCplx(output.data() + i, output2.data() + i, static_cast<int>(tw), 16 * cols);
    }
    std::vector<double> ret(csize * 2);
    for (size_t i = 0; i < csize; ++i) {
        ret[i * 2] = output[i].re;
        ret[i * 2 + 1] = output[i].im;
    }
    return ret;
}

enum class Norm { identity, sqrt, cbrt };

static double applyNorm(Norm n, double x) {
    switch (n) {
        case Norm::identity: return x;
        case Norm::sqrt: return std::sqrt(x);
        case Norm::cbrt: return std::pow(x, 1.0 / 3.0);
    }
    return x;
}

struct SigmaFunc {
    std::vector<double> locs;
    std::vector<double> sigmas;
    double constant = 0.0;

    static SigmaFunc initConst(Norm n, double sigma) {
        SigmaFunc f;
        f.constant = applyNorm(n, sigma);
        return f;
    }

    static SigmaFunc initPacks(const double * data, size_t count, Norm n) {
        SigmaFunc f;
        const size_t cnt = count / 2;
        f.locs.resize(cnt);
        f.sigmas.resize(cnt);
        for (size_t i = 0; i < cnt; ++i) {
            f.locs[i] = data[i * 2];
            f.sigmas[i] = data[i * 2 + 1];
        }
        // insertion sort by loc (matches the reference)
        for (size_t i = 1; i < cnt; ++i) {
            const double kl = f.locs[i];
            const double ks = f.sigmas[i];
            size_t j = i;
            while (j > 0 && f.locs[j - 1] > kl) {
                f.locs[j] = f.locs[j - 1];
                f.sigmas[j] = f.sigmas[j - 1];
                j -= 1;
            }
            f.locs[j] = kl;
            f.sigmas[j] = ks;
        }
        for (double & s : f.sigmas) {
            s = applyNorm(n, s);
        }
        return f;
    }

    std::optional<double> eval(double x) const {
        if (locs.empty()) {
            return constant;
        }
        for (size_t i = 0; i + 1 < locs.size(); ++i) {
            if (x <= locs[i + 1]) {
                const double w = (x - locs[i]) / (locs[i + 1] - locs[i]);
                return (1.0 - w) * sigmas[i] + w * sigmas[i + 1];
            }
        }
        return std::nullopt;
    }
};

static double getLocation(int position, int length) {
    if (length == 1) {
        return 0.0;
    }
    const int half = length / 2;
    if (position > half) {
        return static_cast<double>(length - position) / static_cast<double>(half);
    }
    return static_cast<double>(position) / static_cast<double>(half);
}

static std::optional<double> getSigma(int position, int length, const SigmaFunc & func) {
    if (length == 1) {
        return 1.0;
    }
    return func.eval(getLocation(position, length));
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

static int calcPadSize(int size, int block_step) {
    const int rem = size % BS;
    return size + (rem != 0 ? BS - rem : 0) + std::max(BS - block_step, block_step) * 2;
}

static int calcPadNum(int size, int block_step) {
    return (calcPadSize(size, block_step) - BS) / block_step + 1;
}

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------
//
// DFTTest runs on the R80 GPU API: the input clip is GPU resident, the filter
// reads the core's frame planes and writes a GPU output frame, and the core
// owns every transfer (std.GPUUpload/GPUDownload). Per output frame a single
// recording pads each of the tw temporal source planes into a device-local
// buffer, runs the fused im2col+3D-DFT+filter+inverse kernel, then overlap-adds
// (col2im) straight into the output frame plane. There is no host copy, no
// frame cache and no filter-owned semaphore: one exec pool, one submission.

struct DftPlaneConfig {
    int width {};                   // frame plane pixels
    int height {};
    int pw {};                      // padded dims
    int ph {};
    int num_blocks {};              // block grid
    VkDeviceSize padded_bytes {};   // tw * pw * ph * bytes
    VkDeviceSize spatial_bytes {};  // num_blocks * 256 floats
};

// Every offset pushed to the shader is int32; the per-plane buffers below are
// separate allocations, so the bases are 0 and only the plane-internal offsets
// (spatial float index, pad slice) carry real values.
struct DftPushConstants {
    int32_t padded_base;
    int32_t spatial_base;
    int32_t dst_base;
    int32_t src_base;
    int32_t pad_t0;
    int32_t wt_base;
    int32_t wf_base;
    int32_t sigma_base;
    int32_t radius;
    int32_t block_step;
    int32_t width;
    int32_t height;
    int32_t src_stride;
    int32_t dst_stride;
    float sigma;
    float sigma2;
    float pmin;
    float pmax;
    float beta;
};

// GPU-timing probe (VSFEEL_DFFTEST_GPUTRACE=<frame>, default 100 under
// VSFEEL_DEBUG=2): one warm frame stamps the head, pad, fused and col2im
// boundaries of the first processed plane into a query pool, whose results the
// same command buffer copies into a mapped buffer. The host waits the
// submission out once to read them.
struct DftGpuProbe {
    VkQueryPool query {};
    GpuBuffer buf;
    uint64_t * map {};
    std::atomic<int> armed { 0 };
};

struct DftData {
    VSNode * node {};
    const VSVideoInfo * vi {};

    int bits {}, elem_bytes {};
    bool process[3] { false, false, false };

    int radius {}, block_step {}, filter_type {}, tw {};
    bool zmean {};
    bool sigma_is_scalar { true };
    float sigma_scalar {};          // scaled by wscale when ftype < 2
    float sigma2 {};
    float pmin {};
    float pmax {};
    float beta {};                  // f0beta (unscaled)

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkPipeline pad_pipeline {};
    VkPipeline col2im_pipeline {};
    VkPipeline fused_pipeline[4] {};

    // shared constant buffer: window, then window_freq, then the sigma array
    GpuBuffer wt;
    int32_t wf_base {};             // float offset of window_freq (-1 if !zmean)
    int32_t sigma_base {};          // float offset of sigma array (-1 if scalar)

    std::array<DftPlaneConfig, 3> planes {};
    VSGPUExecPool * pool {};

    // VSFEEL_DFFTEST_GPUTRACE=<frame>: one-shot GPU kernel timings on a warm
    // frame. Only created when the compute queue family can timestamp at all
    // (writing one where timestampValidBits is 0 can hang the engine).
    bool gpu_trace { false };
    int gpu_trace_frame { 100 };
    DftGpuProbe probe;

    // VSFEEL_DFTTEST_TIMING=1: per-frame host stage split. The host side is
    // only acquire/record/submit now, but the split still says whether a frame
    // is host- or GPU-bound, which no kernel timing can.
    bool host_timing { false };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_record_ns {}, ht_submit_ns {},
        ht_total_ns {}, ht_n {};

    ~DftData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[dfttest-timing] frames=%.0f per-frame us: acquire=%7.1f "
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
        if (pad_pipeline) {
            gpu->vk->vkDestroyPipeline(dev, pad_pipeline, nullptr);
        }
        if (col2im_pipeline) {
            gpu->vk->vkDestroyPipeline(dev, col2im_pipeline, nullptr);
        }
        for (auto & p : fused_pipeline) {
            if (p) {
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
        if (probe.query) {
            gpu->vk->vkDestroyQueryPool(dev, probe.query, nullptr);
        }
        gpu_destroy_buffer(*gpu, probe.buf);
    }
};

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

// The fused kernel's shared memory is one tile per 16 invocations, exchanged
// across subgroupBarrier() -- a barrier whose scope is one subgroup -- so every
// aligned 16-lane tile has to live inside a single subgroup. That holds when the
// subgroup size in use is a multiple of 16, and only then; 32 is what the target
// GPU is tuned for. The pad/col2im kernels do not use subgroups but are given
// the same required size, so the count the device limits, subgroups per
// workgroup, is read off the largest of the three launches.
constexpr GpuWorkgroup kFusedWorkgroup { .x = 8 * 16,   // SUB_BLOCKS * 16
    .shared_bytes = (8 * 288 + 8) * 4 };                // trans[SUB_BLOCKS*TR_SUB] + s_gf
constexpr GpuWorkgroup kPadWorkgroup { .x = 32, .y = 8 };

// The fused kernel's variant is baked in as specialization constants so the
// dead filter branches (and their divisions) vanish, matching the reference's
// compile-time `#if FILTER_TYPE` selection. The env overrides exist so a probe
// run can force a subgroup size (or deliberately request an invalid one and
// watch the driver reject it).
static std::variant<VkPipeline, std::string> create_pipeline(
    const GPUDevice & gpu, VkPipelineLayout layout, const uint32_t * code,
    size_t code_size, GpuWorkgroup workgroup, int32_t filter_type = -1,
    int32_t zmean = -1) {

    // The probe knobs win over the default selection below on purpose: they
    // exist so a run can force a subgroup size, or deliberately request one the
    // device does not offer and watch it be rejected.
    const int forced_sgsize = env_int("VSFEEL_DFFTEST_SGSIZE", 0);
    const bool forced_invalid = env_flag("VSFEEL_DFFTEST_SGSIZE_INVALID");

    // Ask for 32 when the device can be asked at all; otherwise keep the
    // driver's default only if it is itself a multiple of 16. Taking any
    // default, as this did before, ran the fused kernel on a device whose
    // subgroups are 8 lanes wide with every 16-lane tile split across two
    // subgroups: a silent race the subgroup barrier cannot order.
    uint32_t subgroup_size = 0;
    if (forced_invalid) {
        subgroup_size = 17;   // invalid on purpose, to test driver validation
    } else if (forced_sgsize > 0) {
        subgroup_size = static_cast<uint32_t>(forced_sgsize);
    } else if (gpu.has_subgroup_size(32, kPadWorkgroup.invocations())) {
        subgroup_size = 32;
    } else if (gpu.subgroup_size % 16 != 0) {
        return std::string("dfttest's fused kernel needs a subgroup size that is "
            "a multiple of 16 lanes (16-lane tiles share data across a subgroup "
            "barrier); this device's default is ") +
            std::to_string(gpu.subgroup_size) + ", and it cannot be asked for another";
    }

    // Build the spec-constant map from an explicit (id, value) list so the
    // offset always matches the slot the value is stored in.
    VkSpecializationMapEntry spec_entries[2] {};
    int32_t spec_values[2] {};
    uint32_t n_spec = 0;
    if (filter_type >= 0) {
        spec_entries[n_spec] = { .constantID = 1,
            .offset = static_cast<uint32_t>(n_spec * sizeof(int32_t)),
            .size = sizeof(int32_t) };
        spec_values[n_spec] = filter_type;
        ++n_spec;
    }
    if (zmean >= 0) {
        spec_entries[n_spec] = { .constantID = 2,
            .offset = static_cast<uint32_t>(n_spec * sizeof(int32_t)),
            .size = sizeof(int32_t) };
        spec_values[n_spec] = zmean;
        ++n_spec;
    }

    return gpu_create_pipeline(gpu, code, code_size, layout,
        n_spec ? spec_entries : nullptr, n_spec ? spec_values : nullptr,
        n_spec, n_spec * sizeof(int32_t), "dfttest", subgroup_size, workgroup);}

static bool dfttest_trace() {
    static const bool v = vsfeel_debug_trace("VSFEEL_DFFTEST_TRACE");
    return v;
}


static DftPushConstants base_pc(const DftData & d) {
    DftPushConstants pc {};
    pc.wt_base = 0;
    pc.wf_base = d.wf_base;
    pc.sigma_base = d.sigma_base;
    pc.radius = d.radius;
    pc.block_step = d.block_step;
    pc.sigma = d.sigma_scalar;
    pc.sigma2 = d.sigma2;
    pc.pmin = d.pmin;
    pc.pmax = d.pmax;
    pc.beta = d.beta;
    return pc;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// GPU input, GPU output: the kernels read the core's source frame planes
// directly (pad) and write the output frame plane (col2im). The core owns
// every transfer.
static const VSFrame * dft_gpu_frame(
    DftData * d, int n, VSFrameContext * frameCtx, VSCore * core,
    const VSAPI * vsapi) {

    const int numPlanes = d->vi->format.numPlanes;
    const int tw = d->tw;

    std::array<const VSFrame *, 7> src {};
    for (int t = 0; t < tw; ++t) {
        const int idx = std::clamp(n - d->radius + t, 0, d->vi->numFrames - 1);
        src[t] = vsapi->getFrameFilter(idx, d->node, frameCtx);
    }
    const VSFrame * center = src[d->radius];

    bool any_process = false, all_process = true;
    for (int p = 0; p < numPlanes; ++p) {
        any_process |= d->process[p];
        all_process &= d->process[p];
    }

    // Unprocessed planes ride along from the center frame, keeping their own
    // producer pairs; everything processed is written by this submission.
    // newVideoFrame2 infers residency from the plane sources, so a frame with no
    // source plane at all has to come from newGPUVideoFrame.
    const int pl[] = { 0, 1, 2 };
    const VSFrame * fr[] = {
        d->process[0] ? nullptr : center,
        d->process[1] ? nullptr : center,
        d->process[2] ? nullptr : center
    };
    VSFrame * dst = all_process
        ? d->gpu->api->newGPUVideoFrame(&d->vi->format, d->vi->width,
              d->vi->height, center, core)
        : vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height,
              fr, pl, center, core);
    if (!dst) {
        vsfeel_trace_error("DFTTest", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("DFTTest: failed to allocate the output frame", frameCtx);
        for (int t = 0; t < tw; ++t) {
            vsapi->freeFrame(src[t]);
        }
        return nullptr;
    }

    // Nothing to run: every plane shares from the center frame, so the frame is
    // already complete and an empty submission would only cost a round trip.
    if (!any_process) {
        for (int t = 0; t < tw; ++t) {
            vsapi->freeFrame(src[t]);
        }
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
        vsfeel_trace_error("DFTTest", n, message, d->gpu.get());
        vsapi->setFilterError(("DFTTest: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        for (int t = 0; t < tw; ++t) {
            vsapi->freeFrame(src[t]);
        }
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
    const VkPhysicalDeviceLimits & lim = d->gpu->limits;
    const bool gputrace = d->gpu_trace && n == d->gpu_trace_frame &&
        d->probe.armed.exchange(1) == 0;
    int probe_plane = -1;
    if (gputrace) {
        for (int p = 0; p < numPlanes && probe_plane < 0; ++p) {
            if (d->process[p]) {
                probe_plane = p;
            }
        }
        d->gpu->vk->vkCmdResetQueryPool(cmd, d->probe.query, 0, 4);
        d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            d->probe.query, 0);
    }

    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        const auto & cfg = d->planes[plane];

        // Per-frame scratch: the padded source window and the float block
        // buffer. Both are handed to the context, which recycles them once the
        // submission completes (the pool's size buckets make this cheap).
        GpuBuffer padded {}, spatial {};
        if (auto e = gpu_frame_buffer(*d->gpu, core, ctx, cfg.padded_bytes, padded);
            !e.empty()) {
            return fail("padded buffer: " + e);
        }
        if (auto e = gpu_frame_buffer(*d->gpu, core, ctx, cfg.spatial_bytes, spatial);
            !e.empty()) {
            return fail("spatial buffer: " + e);
        }

        VSVulkanPlaneInfo dst_plane {};
        if (d->gpu->api->getGPUPlane(dst, plane, &dst_plane)) {
            return fail("output plane " + std::to_string(plane) + " is not GPU resident");
        }
        const int dst_stride = static_cast<int>(
            vsapi->getStride(dst, plane) / d->elem_bytes);

        // pad and col2im both walk the padded plane, so they share a grid
        const uint32_t plane_gx = std::max(std::min<uint32_t>(
            (static_cast<uint32_t>(cfg.pw) + 31u) / 32u, lim.maxComputeWorkGroupCount[0]), 1u);
        const uint32_t plane_gy = std::max(std::min<uint32_t>(
            (static_cast<uint32_t>(cfg.ph) + 7u) / 8u, lim.maxComputeWorkGroupCount[1]), 1u);
        const uint32_t blocks = static_cast<uint32_t>(cfg.num_blocks);
        const uint32_t fused_gx = std::max(std::min<uint32_t>(
            (blocks + 7u) / 8u, lim.maxComputeWorkGroupCount[0]), 1u);

        // pad: one dispatch per temporal slice, reading its own source frame
        // plane (so the temporal offset lives in the buffer bound at binding 3,
        // not in an address the kernel walks).
        vsfeel_trace_mark("pad");
        for (int t = 0; t < tw; ++t) {
            VSVulkanPlaneInfo sp {};
            if (d->gpu->api->getGPUPlane(src[t], plane, &sp)) {
                return fail("source plane " + std::to_string(plane) +
                            " is not GPU resident");
            }
            const VkBuffer buffers[5] {
                d->wt.buffer, padded.buffer, spatial.buffer,
                sp.buffer, dst_plane.buffer
            };
            DftPushConstants pc = base_pc(*d);
            pc.pad_t0 = t;
            pc.width = cfg.width;
            pc.height = cfg.height;
            pc.src_stride = static_cast<int32_t>(
                vsapi->getStride(src[t], plane) / d->elem_bytes);
            pc.dst_stride = dst_stride;
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->pad_pipeline);
            gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 5);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &pc, sizeof(pc));
            d->gpu->vk->vkCmdDispatch(cmd, plane_gx, plane_gy, 1);
        }
        gpu_barrier(*d->gpu, cmd);
        if (gputrace && plane == probe_plane) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->probe.query, 1);
        }

        // fused: im2col + spatial/temporal DFT + filter + inverse, writing the
        // center temporal slice of each block into the spatial buffer.
        vsfeel_trace_mark("fused");
        {
            VSVulkanPlaneInfo sp {};
            if (d->gpu->api->getGPUPlane(center, plane, &sp)) {
                return fail("center plane " + std::to_string(plane) +
                            " is not GPU resident");
            }
            const VkBuffer buffers[5] {
                d->wt.buffer, padded.buffer, spatial.buffer,
                sp.buffer, dst_plane.buffer
            };
            DftPushConstants pc = base_pc(*d);
            pc.width = cfg.width;
            pc.height = cfg.height;
            pc.src_stride = static_cast<int32_t>(
                vsapi->getStride(center, plane) / d->elem_bytes);
            pc.dst_stride = dst_stride;
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->fused_pipeline[d->radius]);
            gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 5);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &pc, sizeof(pc));
            d->gpu->vk->vkCmdDispatch(cmd, fused_gx, 1, 1);
        }
        gpu_barrier(*d->gpu, cmd);
        if (gputrace && plane == probe_plane) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->probe.query, 2);
        }

        // col2im: overlap-add the windowed blocks straight into the output plane
        vsfeel_trace_mark("col2im");
        {
            const VkBuffer buffers[5] {
                d->wt.buffer, padded.buffer, spatial.buffer,
                padded.buffer, dst_plane.buffer
            };
            DftPushConstants pc = base_pc(*d);
            pc.width = cfg.width;
            pc.height = cfg.height;
            pc.src_stride = static_cast<int32_t>(
                vsapi->getStride(center, plane) / d->elem_bytes);
            pc.dst_stride = dst_stride;
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->col2im_pipeline);
            gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 5);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &pc, sizeof(pc));
            d->gpu->vk->vkCmdDispatch(cmd, plane_gx, plane_gy, 1);
        }
        if (gputrace && plane == probe_plane) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->probe.query, 3);
        }
    }
    if (gputrace) {
        d->gpu->vk->vkCmdCopyQueryPoolResults(cmd, d->probe.query, 0, 4,
            d->probe.buf.buffer, 0, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }
    auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    // The pool turns the source planes' producer pairs into device-side waits
    // and publishes the output's producers; frames stay alive until the
    // submission completes. Clamped temporal boundaries map several slices to
    // one frame, so identical frames are declared once.
    for (int t = 0; t < tw; ++t) {
        bool dup = false;
        for (int u = 0; u < t; ++u) {
            dup |= src[u] == src[t];
        }
        if (!dup) {
            d->gpu->api->gpuExecReadsFrame(ctx, src[t]);
        }
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

    if (dfttest_trace()) {
        fprintf(stderr, "[dfttest-trace] n=%d tw=%d submitted\n", n, tw);
    }

    if (gputrace) {
        // One-shot probe: wait this submission out so the query results are
        // final, then read the mapped copy the command buffer made.
        char perr[256] {};
        if (d->gpu->api->gpuExecWaitValue(d->pool, signaled, perr, sizeof(perr)) == gdDrained) {
            const double period = lim.timestampPeriod;
            const auto us = [period](uint64_t a, uint64_t b) {
                return static_cast<double>(b - a) * period / 1000.0;
            };
            const uint64_t * ts = d->probe.map;
            fprintf(stderr,
                "[dfttest-gpu] n=%d pad=%.1fus fused=%.1fus col2im=%.1fus total=%.1fus\n",
                n, us(ts[0], ts[1]), us(ts[1], ts[2]), us(ts[2], ts[3]),
                us(ts[0], ts[3]));
        } else {
            fprintf(stderr, "[dfttest-gpu] probe wait failed: %s\n", perr);
        }
    }

    for (int t = 0; t < tw; ++t) {
        vsapi->freeFrame(src[t]);
    }

    return dst;
}

static const VSFrame *VS_CC DftGetFrame(
    int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    DftData * d = static_cast<DftData *>(instanceData);

    if (activationReason == arInitial) {
        const int start = std::max(n - d->radius, 0);
        const int end = std::min(n + d->radius, d->vi->numFrames - 1);
        for (int i = start; i <= end; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    return dft_gpu_frame(d, n, frameCtx, core, vsapi);
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC DftFree(
    void *instanceData, [[maybe_unused]] VSCore *core, const VSAPI *vsapi) {

    DftData * d = static_cast<DftData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC DftCreate(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<DftData>() };

    // Opt-in host-path timing; the default path records no clocks.
    d->host_timing = vsfeel_debug_probe("VSFEEL_DFTTEST_TIMING");

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("DFTTest", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("DFTTest: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    const auto & fmt = d->vi->format;
    const int bits = fmt.bitsPerSample;
    const bool depth_ok = (fmt.sampleType == stFloat && bits == 32) ||
                          (fmt.sampleType == stInteger && bits == 16);
    if (!depth_ok || d->vi->width <= 0 || d->vi->height <= 0 ||
        (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV && fmt.colorFamily != cfRGB)) {
        return set_error("input must be 16 bit integer or 32 bit float, Gray/YUV/RGB, constant format.");
    }
    d->bits = bits;
    d->elem_bytes = bits / 8;

    int ftype = vsh::int64ToIntS(vsapi->mapGetInt(in, "ftype", 0, &error));
    if (error) {
        ftype = 0;
    }
    if (ftype < 0 || ftype > 4) {
        return set_error("ftype must be 0, 1, 2, 3, or 4.");
    }

    double sigma = vsapi->mapGetFloat(in, "sigma", 0, &error);
    if (error) {
        sigma = 8.0;
    }
    double sigma2 = vsapi->mapGetFloat(in, "sigma2", 0, &error);
    if (error) {
        sigma2 = 8.0;
    }
    double pmin = vsapi->mapGetFloat(in, "pmin", 0, &error);
    if (error) {
        pmin = 0.0;
    }
    double pmax = vsapi->mapGetFloat(in, "pmax", 0, &error);
    if (error) {
        pmax = 500.0;
    }
    if (!std::isfinite(sigma) || !std::isfinite(sigma2) || !std::isfinite(pmin) || !std::isfinite(pmax)) {
        return set_error("sigma/sigma2/pmin/pmax must be finite.");
    }

    int sbsize = vsh::int64ToIntS(vsapi->mapGetInt(in, "sbsize", 0, &error));
    if (error) {
        sbsize = 16;
    }
    if (sbsize != 16) {
        return set_error("sbsize must be 16 (hipRTC-backend port).");
    }
    int sosize = vsh::int64ToIntS(vsapi->mapGetInt(in, "sosize", 0, &error));
    if (error) {
        sosize = 12;
    }
    if (sosize < 0 || sosize > 15) {
        return set_error("sosize must be 0..15.");
    }
    if (sosize > 8 && sbsize % (sbsize - sosize) != 0) {
        return set_error("spatial overlap > 50% requires that sbsize-sosize is a divisor of sbsize.");
    }
    int tbsize = vsh::int64ToIntS(vsapi->mapGetInt(in, "tbsize", 0, &error));
    if (error) {
        tbsize = 3;
    }
    if (tbsize < 1 || tbsize > 7) {
        return set_error("tbsize must be odd, 1..7 (temporal radius 0..3).");
    }
    if (tbsize % 2 == 0) {
        return set_error("tbsize must be odd (dfttest2 silently aliases even values to tbsize-1).");
    }
    int swin = vsh::int64ToIntS(vsapi->mapGetInt(in, "swin", 0, &error));
    if (error) {
        swin = 0;
    }
    int twin = vsh::int64ToIntS(vsapi->mapGetInt(in, "twin", 0, &error));
    if (error) {
        twin = 7;
    }
    if (swin < 0 || swin > 11 || twin < 0 || twin > 11) {
        return set_error("swin/twin must be 0..11.");
    }
    double sbeta = vsapi->mapGetFloat(in, "sbeta", 0, &error);
    if (error) {
        sbeta = 2.5;
    }
    double tbeta = vsapi->mapGetFloat(in, "tbeta", 0, &error);
    if (error) {
        tbeta = 2.5;
    }
    if (!std::isfinite(sbeta) || !std::isfinite(tbeta)) {
        return set_error("sbeta/tbeta must be finite.");
    }
    int zmean = vsh::int64ToIntS(vsapi->mapGetInt(in, "zmean", 0, &error));
    if (error) {
        zmean = 1;
    }
    double f0beta = vsapi->mapGetFloat(in, "f0beta", 0, &error);
    if (error) {
        f0beta = 1.0;
    }
    if (!std::isfinite(f0beta)) {
        return set_error("f0beta must be finite.");
    }
    int ssystem = vsh::int64ToIntS(vsapi->mapGetInt(in, "ssystem", 0, &error));
    if (error) {
        ssystem = 0;
    }
    if (ssystem < 0 || ssystem > 1) {
        return set_error("ssystem must be 0 or 1.");
    }

    const double * slocation = nullptr;
    const double * ssx = nullptr;
    const double * ssy = nullptr;
    const double * sst = nullptr;
    int n_slocation = 0, n_ssx = 0, n_ssy = 0, n_sst = 0;
    if (vsapi->mapNumElements(in, "slocation") > 0) {
        slocation = vsapi->mapGetFloatArray(in, "slocation", &error);
        if (error) {
            return set_error("slocation must be an array of floats.");
        }
        n_slocation = vsapi->mapNumElements(in, "slocation");
    }
    if (vsapi->mapNumElements(in, "ssx") > 0) {
        ssx = vsapi->mapGetFloatArray(in, "ssx", &error);
        if (error) {
            return set_error("ssx must be an array of floats.");
        }
        n_ssx = vsapi->mapNumElements(in, "ssx");
    }
    if (vsapi->mapNumElements(in, "ssy") > 0) {
        ssy = vsapi->mapGetFloatArray(in, "ssy", &error);
        if (error) {
            return set_error("ssy must be an array of floats.");
        }
        n_ssy = vsapi->mapNumElements(in, "ssy");
    }
    if (vsapi->mapNumElements(in, "sst") > 0) {
        sst = vsapi->mapGetFloatArray(in, "sst", &error);
        if (error) {
            return set_error("sst must be an array of floats.");
        }
        n_sst = vsapi->mapNumElements(in, "sst");
    }
    const int array_counts[4] { n_slocation, n_ssx, n_ssy, n_sst };
    for (int cnt : array_counts) {
        if (cnt != 0 && (cnt % 2 != 0 || cnt < 2)) {
            return set_error("number of elements in slocation/ssx/ssy/sst must be a non-zero multiple of 2.");
        }
    }

    // device_id and num_streams are registered but never read: the core owns
    // the one device and sizes in-flight depth itself (exec pool ring).

    d->radius = (tbsize - 1) / 2;
    d->block_step = sbsize - sosize;
    d->tw = 2 * d->radius + 1;
    d->zmean = zmean != 0;
    d->beta = static_cast<float>(f0beta);

    const int num_planes = fmt.numPlanes;
    if (vsapi->mapNumElements(in, "planes") > 0) {
        for (int i = 0; i < vsapi->mapNumElements(in, "planes"); ++i) {
            const int idx = vsh::int64ToIntS(vsapi->mapGetInt(in, "planes", i, &error));
            if (idx < 0 || idx >= num_planes) {
                return set_error("plane index out of range.");
            }
            if (d->process[idx]) {
                return set_error("plane specified twice.");
            }
            d->process[idx] = true;
        }
    } else {
        for (int i = 0; i < num_planes; ++i) {
            d->process[i] = true;
        }
    }

    // FILTER_TYPE mapping (ftype 0 with f0beta variants)
    int filter_type = ftype;
    if (ftype == 0) {
        if (std::abs(f0beta - 1.0) < 0.00005) {
            filter_type = 0;
        } else if (std::abs(f0beta - 0.5) < 0.0005) {
            filter_type = 6;
        } else {
            filter_type = 5;
        }
    }
    d->filter_type = filter_type;

    const int subW = fmt.subSamplingW;
    const int subH = fmt.subSamplingH;

    // Per-plane geometry
    bool any_plane = false;
    for (int plane = 0; plane < num_planes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        any_plane = true;
        auto & cfg = d->planes[plane];
        cfg.width = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        cfg.height = (plane == 0) ? d->vi->height : d->vi->height >> subH;
        cfg.pw = calcPadSize(cfg.width, d->block_step);
        cfg.ph = calcPadSize(cfg.height, d->block_step);
        cfg.num_blocks = calcPadNum(cfg.width, d->block_step) *
            calcPadNum(cfg.height, d->block_step);

        // single-fold reflect_pad requires pad <= dim-1
        const int ox = (cfg.pw - cfg.width) / 2;
        const int oy = (cfg.ph - cfg.height) / 2;
        if (ox > cfg.width - 1 || (cfg.pw - cfg.width - ox) > cfg.width - 1 ||
            oy > cfg.height - 1 || (cfg.ph - cfg.height - oy) > cfg.height - 1) {
            return set_error("a processed plane is too small for the padded block layout.");
        }

        const VkDeviceSize pad_elems = static_cast<VkDeviceSize>(cfg.pw) * cfg.ph;
        const VkDeviceSize nblk = cfg.num_blocks;
        cfg.padded_bytes = static_cast<VkDeviceSize>(d->tw) * pad_elems * d->elem_bytes;
        cfg.spatial_bytes = nblk * 256 * sizeof(float);

        // Every region below is addressed by an int32 push constant, so bound
        // each per-plane region in the units the shader actually uses.
        if (d->tw * pad_elems >= (1ll << 31) ||
            cfg.padded_bytes >= (1ll << 31) ||
            nblk * 256 >= (1ll << 31)) {
            return set_error("frame too large (a plane region exceeds the 2^31 addressing limit).");
        }
    }
    if (!any_plane) {
        return set_error("no planes to process.");
    }

    const auto window = getWindow(d->radius, d->block_step, swin, sbeta, twin, tbeta);

    // wscale = Shewchuk sum of the squared window
    std::vector<double> sq(window.size());
    for (size_t i = 0; i < sq.size(); ++i) {
        sq[i] = window[i] * window[i];
    }
    const double wscale = fsum(sq.data(), sq.size());

    // sigma array (per-bin) unless every sigma source is scalar
    d->sigma_is_scalar = (slocation == nullptr && ssx == nullptr && ssy == nullptr && sst == nullptr);
    std::vector<double> sigma_array;
    if (!d->sigma_is_scalar) {
        const Norm norm = (slocation != nullptr && ssystem == 1) ? Norm::identity
            : (tbsize == 1) ? Norm::sqrt : Norm::cbrt;

        // slocation is the one shared 3-D table: all three axes use the same
        // function, so the per-axis sources are just aliases of it.
        SigmaFunc fx, fy, ft;
        if (slocation != nullptr) {
            fx = SigmaFunc::initPacks(slocation, n_slocation, norm);
            fy = fx;
            ft = fx;
        } else {
            fx = (ssx != nullptr) ? SigmaFunc::initPacks(ssx, n_ssx, norm)
                                  : SigmaFunc::initConst(norm, sigma);
            fy = (ssy != nullptr) ? SigmaFunc::initPacks(ssy, n_ssy, norm)
                                  : SigmaFunc::initConst(norm, sigma);
            ft = (sst != nullptr) ? SigmaFunc::initPacks(sst, n_sst, norm)
                                  : SigmaFunc::initConst(norm, sigma);
        }

        sigma_array.resize(static_cast<size_t>(d->tw) * 16 * 9);
        size_t idx = 0;
        bool fail = false;
        if (ssystem == 0) {
            for (int t = 0; t < d->tw && !fail; ++t) {
                const auto st = getSigma(t, d->tw, ft);
                if (!st) {
                    fail = true;
                    break;
                }
                for (int y = 0; y < BS && !fail; ++y) {
                    const auto sy = getSigma(y, BS, fy);
                    if (!sy) {
                        fail = true;
                        break;
                    }
                    for (int x = 0; x < BS / 2 + 1; ++x) {
                        const auto sx = getSigma(x, BS, fx);
                        if (!sx) {
                            fail = true;
                            break;
                        }
                        sigma_array[idx] = *st * *sy * *sx;
                        idx += 1;
                    }
                }
            }
        } else {
            const double ndim = (d->radius > 0) ? 3.0 : 2.0;
            for (int t = 0; t < d->tw && !fail; ++t) {
                const double lt = getLocation(t, d->tw);
                for (int y = 0; y < BS && !fail; ++y) {
                    const double ly = getLocation(y, BS);
                    for (int x = 0; x < BS / 2 + 1; ++x) {
                        const double lx = getLocation(x, BS);
                        const double location = std::sqrt((lt * lt + ly * ly + lx * lx) / ndim);
                        const auto v = ft.eval(location);
                        if (!v) {
                            fail = true;
                            break;
                        }
                        sigma_array[idx] = *v;
                        idx += 1;
                    }
                }
            }
        }
        if (fail) {
            return set_error("slocation/ssx/ssy/sst must cover the full [0, 1] frequency range.");
        }
    }

    // scale sigma/sigma2/pmin/pmax by the window scale factor (ftype < 2)
    if (ftype < 2) {
        if (d->sigma_is_scalar) {
            sigma *= wscale;
        } else {
            for (double & s : sigma_array) {
                s *= wscale;
            }
        }
        sigma2 *= wscale;
    }
    pmin *= wscale;
    pmax *= wscale;
    d->sigma_scalar = static_cast<float>(sigma);
    d->sigma2 = static_cast<float>(sigma2);
    d->pmin = static_cast<float>(pmin);
    d->pmax = static_cast<float>(pmax);

    // window_freq (only when zmean)
    std::vector<double> window_freq;
    if (zmean) {
        std::vector<double> scaled(window.size());
        for (size_t i = 0; i < scaled.size(); ++i) {
            scaled[i] = window[i] * 255.0;
        }
        window_freq = rdftTables(d->radius, scaled);
    }

    {
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

    // ------------------------------------------------------------------
    // Push-descriptor layout and pipeline layout
    // ------------------------------------------------------------------
    // One descriptor set per dispatch through the push descriptor set: each
    // recording rebinds its own view of the planes (binding 3 is a different
    // source frame per pad dispatch), so nothing is allocated from a pool.
    {
        const auto result = gpu_push_set_layout(*d->gpu, 5);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result = gpu_pipeline_layout(*d->gpu, d->set_layout,
            sizeof(DftPushConstants));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // ------------------------------------------------------------------
    // Constant buffer: window + window_freq + sigma array
    // ------------------------------------------------------------------
    {
        const size_t n_window = static_cast<size_t>(d->tw) * 256;
        const size_t n_freq = zmean ? static_cast<size_t>(d->tw) * 16 * 9 * 2 : 0;
        const size_t n_sigma = sigma_array.empty() ? 0 : static_cast<size_t>(d->tw) * 16 * 9;

        VkDeviceSize wt_bytes = static_cast<VkDeviceSize>(
            (n_window + n_freq + n_sigma) * sizeof(float));
        wt_bytes = std::max<VkDeviceSize>(wt_bytes, 16);
        d->wf_base = zmean ? static_cast<int32_t>(n_window) : -1;
        d->sigma_base = !sigma_array.empty() ? static_cast<int32_t>(n_window + n_freq) : -1;

        // Host visible and coherent, so a plain memcpy lands and no flush is
        // needed; the table is tiny and read through L2 every frame.
        auto e = gpu_make_buffer(*d->gpu, core, wt_bytes, d->wt,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!e.empty()) {
            return set_error("window buffer: " + e);
        }
        if (d->wt.mapped == nullptr) {
            return set_error("window buffer is not host visible");
        }
        auto * map = static_cast<float *>(d->wt.mapped);
        for (size_t i = 0; i < n_window; ++i) {
            map[i] = static_cast<float>(window[i]);
        }
        if (zmean) {
            for (size_t i = 0; i < n_freq; ++i) {
                map[n_window + i] = static_cast<float>(window_freq[i]);
            }
        }
        if (!sigma_array.empty()) {
            for (size_t i = 0; i < n_sigma; ++i) {
                map[n_window + n_freq + i] = static_cast<float>(sigma_array[i]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Pipelines
    // ------------------------------------------------------------------
    {
        const uint32_t * pad_code = nullptr;
        size_t pad_size = 0;
        const uint32_t * col2im_code = nullptr;
        size_t col2im_size = 0;
        const uint32_t * fused_code[4] {};
        size_t fused_size[4] {};
        switch (d->bits) {
            case 16:
                pad_code = dfttest_16_pad_spv; pad_size = dfttest_16_pad_spv_size;
                col2im_code = dfttest_16_col2im_spv; col2im_size = dfttest_16_col2im_spv_size;
                fused_code[0] = dfttest_16_fused_r0_spv; fused_size[0] = dfttest_16_fused_r0_spv_size;
                fused_code[1] = dfttest_16_fused_r1_spv; fused_size[1] = dfttest_16_fused_r1_spv_size;
                fused_code[2] = dfttest_16_fused_r2_spv; fused_size[2] = dfttest_16_fused_r2_spv_size;
                fused_code[3] = dfttest_16_fused_r3_spv; fused_size[3] = dfttest_16_fused_r3_spv_size;
                break;
            case 32:
                pad_code = dfttest_32_pad_spv; pad_size = dfttest_32_pad_spv_size;
                col2im_code = dfttest_32_col2im_spv; col2im_size = dfttest_32_col2im_spv_size;
                fused_code[0] = dfttest_32_fused_r0_spv; fused_size[0] = dfttest_32_fused_r0_spv_size;
                fused_code[1] = dfttest_32_fused_r1_spv; fused_size[1] = dfttest_32_fused_r1_spv_size;
                fused_code[2] = dfttest_32_fused_r2_spv; fused_size[2] = dfttest_32_fused_r2_spv_size;
                fused_code[3] = dfttest_32_fused_r3_spv; fused_size[3] = dfttest_32_fused_r3_spv_size;
                break;
            default:
                return set_error("unsupported bit depth");
        }

        {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                pad_code, pad_size, kPadWorkgroup);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_pipeline = std::get<VkPipeline>(result);
        }
        for (int r = 0; r < 4; ++r) {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                fused_code[r], fused_size[r], kFusedWorkgroup,
                d->filter_type, d->zmean ? 1 : 0);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->fused_pipeline[r] = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                col2im_code, col2im_size, kPadWorkgroup);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->col2im_pipeline = std::get<VkPipeline>(result);
        }
    }

    // ------------------------------------------------------------------
    // The output plane stride has to be known before the frame path runs: the
    // core's GPU frames keep the CPU stride, so it is read off a scratch CPU
    // frame here and re-read per frame (getStride applies to both).
    // ------------------------------------------------------------------
    {
        VSFrame * probe = vsapi->newVideoFrame(&fmt, d->vi->width, d->vi->height,
                                               nullptr, core);
        if (probe == nullptr) {
            return set_error("could not allocate a probe frame to read the plane stride");
        }
        for (int plane = 0; plane < num_planes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & cfg = d->planes[plane];
            const int stride = static_cast<int>(vsapi->getStride(probe, plane) / d->elem_bytes);
            const int64_t last = static_cast<int64_t>(cfg.height - 1) * stride +
                cfg.width - 1;
            if (last > INT32_MAX) {
                vsapi->freeFrame(probe);
                return set_error("plane " + std::to_string(plane) + " is too large: " +
                    std::to_string(cfg.width) + "x" + std::to_string(cfg.height) +
                    " at stride " + std::to_string(stride) +
                    " overflows the kernel's 32-bit addressing");
            }
        }
        vsapi->freeFrame(probe);
    }

    {
        char err[512] {};
        d->pool = d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    // GPU-timing probe: only when the queue family can timestamp at all, since
    // vkCmdWriteTimestamp2 there is invalid usage and a driver taking one can
    // hang the engine.
    d->gpu_trace_frame = env_int("VSFEEL_DFFTEST_GPUTRACE", 100);
    d->gpu_trace = vsfeel_debug_probe("VSFEEL_DFFTEST_GPUTRACE") &&
        vsfeel_probe_timestamps(*d->gpu, "DFTTest");
    if (d->gpu_trace) {
        VkQueryPoolCreateInfo qp_info {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 4
        };
        if (d->gpu->vk->vkCreateQueryPool(d->gpu->device, &qp_info, nullptr,
                &d->probe.query) != VK_SUCCESS) {
            d->probe.query = VK_NULL_HANDLE;
            d->gpu_trace = false;
        }
    }
    if (d->gpu_trace) {
        auto e = gpu_make_buffer(*d->gpu, core, 4 * sizeof(uint64_t), d->probe.buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!e.empty() || d->probe.buf.mapped == nullptr) {
            d->gpu_trace = false;
        } else {
            d->probe.map = static_cast<uint64_t *>(d->probe.buf.mapped);
        }
    }

    if (vsfeel_debug_flag("VSFEEL_DFTTEST_VRAM")) {
        VkDeviceSize per_frame = 0;
        for (int plane = 0; plane < num_planes; ++plane) {
            if (d->process[plane]) {
                per_frame += d->planes[plane].padded_bytes + d->planes[plane].spatial_bytes;
            }
        }
        fprintf(stderr, "[dfttest] %.1f MiB per in-flight frame "
                        "(%d planes, tw=%d, %u bytes); tw is re-padded every frame\n",
            per_frame / (1024.0 * 1024.0), num_planes, d->tw,
            static_cast<unsigned>(per_frame));
    }

    DftData *data = d.release();

    VSFilterDependency deps[1] = {
        { data->node, data->radius > 0 ? rpGeneral : rpStrictSpatial }
    };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "DFTTest", data->vi, DftGetFrame, DftFree,
        fmParallel, ffGPUOutput, deps, 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "DFTTest: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_dfttest(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    // Under the R80 GPU API both the input and the output are GPU resident: the
    // core inserts the upload for a CPU clip and a GPUDownload for a CPU
    // consumer, so the filter itself never moves a frame.
    vspapi->registerFunction(
        "DFTTest",
        "clip:vnode:gpu;"
        "ftype:int:opt;"
        "sigma:float:opt;"
        "sigma2:float:opt;"
        "pmin:float:opt;"
        "pmax:float:opt;"
        "sbsize:int:opt;"
        "sosize:int:opt;"
        "tbsize:int:opt;"
        "swin:int:opt;"
        "twin:int:opt;"
        "sbeta:float:opt;"
        "tbeta:float:opt;"
        "zmean:int:opt;"
        "f0beta:float:opt;"
        "ssystem:int:opt;"
        "slocation:float[]:opt;"
        "ssx:float[]:opt;"
        "ssy:float[]:opt;"
        "sst:float[]:opt;"
        "planes:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode:gpu;",
        DftCreate, nullptr, plugin
    );
}
