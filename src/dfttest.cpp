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

#include <vulkan/vulkan.h>

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
//   host: reflection-pad each of the (2*radius+1) source planes into staging
//   fused:   per 16x16 block, im2col + window, 3D DFT, frequency filter,
//            inverse DFT, writes the center temporal slice to a float buffer
//   col2im:  overlap-adds the windowed blocks into the final plane
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

struct PlaneConfig {
    int width {};                   // frame plane pixels
    int height {};
    int pw {};                      // padded dims
    int ph {};
    int num_blocks {};              // block grid
    VkDeviceSize upload_offset {};  // staging byte offset of the tight upload region
    VkDeviceSize upload_bytes {};   // tw * height * width * bytes (tight rows)
    VkDeviceSize padded_offset {};  // padded buffer byte offset
    VkDeviceSize padded_bytes {};   // tw * pw * ph * bytes
    VkDeviceSize download_offset {};// staging byte offset of the output region
    VkDeviceSize download_bytes {}; // h * width * bytes (tight rows)
    VkDeviceSize spatial_offset {}; // float element offset into the spatial buffer
    VkDeviceSize slot_offset {};    // slot buffer byte offset of this plane's slot region
    VkDeviceSize slot_plane_bytes {}; // pw * ph * bytes (one padded plane)
};

// per-resource stable metadata (the pool moves VK_Resource objects, so the
// frame generation counter lives in a stable allocation)
struct ResMeta {
    std::atomic<long long> frame_gen { 0 };
};

// per-slot frame-cache state (index: plane * slot_count + slot). A slot
// owns exactly one source frame (gen) at a time: the first frame that needs
// it pads it (claim) and every later frame whose window contains it just
// D2D-copies it (commit). A slot may only be reclaimed for a different
// source once all committed readers' resources have been reused (their
// frames are then fully done, copies included), which is checked against
// frame_gen — no host fence waits, no deadlocks. `sem` holds the current
// generation's binary semaphores: fresh per generation (created at claim,
// destroyed at reclaim), so a stale signal can never leak across
// generations.
struct SlotState {
    long long gen { -1 };        // source frame the slot owns (-1 = empty)
    std::vector<std::pair<int, long long>> committed; // readers: {res_id, frame_gen}
    std::vector<VkSemaphore> sem; // tw semaphores, one per reader offset
};

struct VK_Resource {
        int id {};
        VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    VkBuffer padded_buf {};         // device-local padded source planes
    VkDeviceMemory padded_mem {};
    VkBuffer spatial_buf {};        // device-local float block buffer
    VkDeviceMemory spatial_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};      // pre-recorded fused + col2im
    VkCommandBuffer cmd2 {};     // per-frame D2D copies (slot -> padded planes)
    // one pad command buffer per (plane, temporal slice): a pad op records
    // and submits its own at claim time; a per-op buffer guarantees a buffer
    // is never re-recorded while its previous submission is still executing
    std::vector<VkCommandBuffer> cmd_pad {};
    VkFence fence {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    float * map {};
    uint32_t staging_type_index {};
};

struct PushConstants {
    int32_t padded_base;    // byte offset into the padded buffer
    int32_t spatial_base;   // float element offset into the spatial buffer
    int32_t dst_base;       // byte offset of the download region in staging
    int32_t src_base;       // byte offset of the upload region in staging
    int32_t pad_t0;         // temporal plane the pad kernel processes (frame cache)
    int32_t wt_base;        // float element offset of window[] in wt
    int32_t wf_base;        // float element offset of window_freq[] in wt (-1 if !zmean)
    int32_t sigma_base;     // float element offset of sigma[] in wt (-1 if scalar)
    int32_t radius;
    int32_t block_step;
    int32_t width;
    int32_t height;
    int32_t src_stride;     // upload plane row stride in elements
    int32_t dst_stride;     // frame plane row stride in elements
    int32_t filter_type;
    float sigma;
    float sigma2;
    float pmin;
    float pmax;
    float beta;
};

struct DftData {
    VSNode * node;
    const VSVideoInfo * vi;

    int num_streams;
    int bits, bytes;

    int radius, block_step;
    int filter_type;
    bool zmean {};
    bool sigma_is_scalar { true };
    float sigma_scalar {};          // scaled by wscale when ftype < 2
    float sigma2 {};
    float pmin {};
    float pmax {};
    float beta {};                  // f0beta (unscaled)
    int tw {};

    bool process[3] { false, false, false };

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule pad_slot_module {};
    VkShaderModule pad_direct_module {};
    VkShaderModule col2im_module {};
    VkShaderModule fused_module[4] {};
    VkPipeline pad_slot_pipeline {};
    VkPipeline pad_direct_pipeline {};
    VkPipeline col2im_pipeline {};
    VkPipeline fused_pipeline[4] {};

    // shared constant buffer: window, then window_freq, then the sigma array
    VkBuffer wt_buf {};
    VkDeviceMemory wt_mem {};
    float * wt_map {};
    uint32_t wt_type_index {};
    VkDeviceSize wt_bytes {};
    int32_t wf_base {};             // float offset of window_freq (-1 if !zmean)
    int32_t sigma_base {};          // float offset of sigma array (-1 if scalar)

    VkDeviceSize upload_total {};   // tight upload planes region (staging)
    VkDeviceSize download_total {}; // output region (staging)
    VkDeviceSize padded_total {};   // padded planes region (device-local)
    VkDeviceSize spatial_total {};  // float elements across planes
    VkDeviceSize slot_total {};     // frame-cache slot region (device-local)

    // padded-source frame cache: each source frame is reflect-pad'd once into a
    // slot and reused across the tw-frame window via D2D copies (vszipcl-style).
    // Works for any num_streams: the slot state machine (gen / committed readers
    // under slot_lock) decides who pads and who reads; cross-queue visibility is
    // provided by per-slot semaphores (the pad submit signals, the copy submit
    // waits). The pad is submitted at claim time (under slot_lock) so a backward
    // semaphore dependency on the queue is impossible, and a slot is only
    // reclaimed once every committed reader's resource has been reused (frame
    // done), so no host fence waits are needed.
    VkBuffer slot_buf {};
    VkDeviceMemory slot_mem {};
    int slot_count {};              // K slots per plane
    std::mutex slot_lock {};
    std::vector<SlotState> slots {};
    std::vector<std::unique_ptr<ResMeta>> res_meta {};
    std::array<PlaneConfig, 3> planes {};
    ticket_semaphore semaphore;
    std::vector<VK_Resource> resources;
    std::mutex resources_lock;

    // ---- debug timing accumulators ----
    std::atomic<uint64_t> t_acquire_ns {0}, t_upload_ns {0}, t_submit_ns {0},
        t_wait_ns {0}, t_download_ns {0}, t_total_ns {0};
    std::atomic<uint64_t> nframes {0};

    ~DftData() {
        uint64_t n = nframes.load();
        if (n) {
            fprintf(stderr,
                "[dfttest-timing] frames=%llu avg_total=%.3fms acquire=%.3fms upload=%.3fms submit=%.3fms wait=%.3fms download=%.3fms\n",
                (unsigned long long)n,
                t_total_ns.load() / 1e6 / n, t_acquire_ns.load() / 1e6 / n,
                t_upload_ns.load() / 1e6 / n, t_submit_ns.load() / 1e6 / n,
                t_wait_ns.load() / 1e6 / n, t_download_ns.load() / 1e6 / n);
        }
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
            if (resource.padded_mem) {
                vkFreeMemory(dev, resource.padded_mem, nullptr);
            }
            if (resource.padded_buf) {
                vkDestroyBuffer(dev, resource.padded_buf, nullptr);
            }
            if (resource.spatial_mem) {
                vkFreeMemory(dev, resource.spatial_mem, nullptr);
            }
            if (resource.spatial_buf) {
                vkDestroyBuffer(dev, resource.spatial_buf, nullptr);
            }
            if (resource.cmd) {
                VkCommandBuffer cbs[2] { resource.cmd, resource.cmd2 };
                vkFreeCommandBuffers(dev, resource.pool, 2, cbs);
            }
            if (!resource.cmd_pad.empty()) {
                vkFreeCommandBuffers(dev, resource.pool,
                    static_cast<uint32_t>(resource.cmd_pad.size()),
                    resource.cmd_pad.data());
            }
            if (resource.pool) {
                vkDestroyCommandPool(dev, resource.pool, nullptr);
            }
            if (resource.fence) {
                vkDestroyFence(dev, resource.fence, nullptr);
            }
        }

        for (auto & st : slots) {
            for (auto & sem : st.sem) {
                vkDestroySemaphore(dev, sem, nullptr);
            }
        }
        if (slot_mem) {
            vkFreeMemory(dev, slot_mem, nullptr);
        }
        if (slot_buf) {
            vkDestroyBuffer(dev, slot_buf, nullptr);
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

        if (pad_slot_pipeline) {
            vkDestroyPipeline(dev, pad_slot_pipeline, nullptr);
        }
        if (pad_direct_pipeline) {
            vkDestroyPipeline(dev, pad_direct_pipeline, nullptr);
        }
        for (auto & p : fused_pipeline) {
            if (p) {
                vkDestroyPipeline(dev, p, nullptr);
            }
        }
        if (col2im_pipeline) {
            vkDestroyPipeline(dev, col2im_pipeline, nullptr);
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
        if (pad_slot_module) {
            vkDestroyShaderModule(dev, pad_slot_module, nullptr);
        }
        if (pad_direct_module) {
            vkDestroyShaderModule(dev, pad_direct_module, nullptr);
        }
        for (auto & m : fused_module) {
            if (m) {
                vkDestroyShaderModule(dev, m, nullptr);
            }
        }
        if (col2im_module) {
            vkDestroyShaderModule(dev, col2im_module, nullptr);
        }

        release_device(device);
    }
};

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
    const VK_Device & dev, VkShaderModule module, VkPipelineLayout layout,
    uint32_t required_subgroup_size = 0, int32_t filter_type = -1,
    int32_t zmean = -1) {

    if (const char * dbg = getenv("VSFEEL_DFTTEST_DBG")) {
        fprintf(stderr, "[dfttest] create_pipeline required_subgroup_size=%u filter_type=%d\n",
            required_subgroup_size, filter_type);
    }

    uint32_t subgroup_size = required_subgroup_size;
    if (const char * sw = getenv("VSFEEL_DFTTEST_SGSIZE")) {
        subgroup_size = atoi(sw);
    }
    if (const char * sw = getenv("VSFEEL_DFTTEST_SGSIZE_INVALID")) {
        subgroup_size = 17;   // invalid on purpose, to test driver validation
    }
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = subgroup_size
    };

    VkSpecializationMapEntry spec_entries[2] {
        { .constantID = 1, .offset = 0, .size = sizeof(int32_t) },
        { .constantID = 2, .offset = sizeof(int32_t), .size = sizeof(int32_t) }
    };
    int32_t spec_values[2] = { filter_type, zmean };
    uint32_t n_spec = 0;
    if (filter_type >= 0) {
        n_spec++;
    }
    if (zmean >= 0) {
        spec_entries[n_spec] = { .constantID = 2, .offset = n_spec * sizeof(int32_t), .size = sizeof(int32_t) };
        n_spec++;
    }
    VkSpecializationInfo spec_info {
        .mapEntryCount = n_spec,
        .pMapEntries = spec_entries,
        .dataSize = n_spec * sizeof(int32_t),
        .pData = (n_spec > 0) ? spec_values : nullptr
    };

    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = subgroup_size ? &subgroup_size_info : nullptr,
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

// Records the per-plane fused + col2im dispatch sequence.
static bool trivial_kernels() {
    static const bool v = getenv("VSFEEL_DFTTEST_TRIVIAL") != nullptr;
    return v;
}

static bool dfttest_trace() {
    static const bool v = getenv("VSFEEL_DFTTEST_TRACE") != nullptr;
    return v;
}

// Frame-cache slot operation for one (plane, temporal slice) of an output
// frame: the source frame f is either new (padded now, into its slot or
// directly into the padded buffer if every slot is busy) or already cached
// (D2D-copied from its slot).
struct SlotOp {
    int plane {};
    int t {};
    int slot { -1 };           // owning slot, or -1 (direct pad, no copy)
    VkDeviceSize slot_base {};
    bool is_pad {};
    int which {};  // reader offset: n - pad_frame (selects the slot semaphore)
};

static PushConstants base_pc(const DftData & d) {
    return PushConstants {
        .padded_base = 0,
        .spatial_base = 0,
        .dst_base = 0,
        .src_base = 0,
        .pad_t0 = 0,
        .wt_base = 0,
        .wf_base = d.wf_base,
        .sigma_base = d.sigma_base,
        .radius = d.radius,
        .block_step = d.block_step,
        .width = 0,
        .height = 0,
        .src_stride = 0,
        .dst_stride = 0,
        .filter_type = d.filter_type,
        .sigma = d.sigma_scalar,
        .sigma2 = d.sigma2,
        .pmin = d.pmin,
        .pmax = d.pmax,
        .beta = d.beta
    };
}

// Appends one pad dispatch (an open command buffer) for a pad op: into the
// shared slot (pad_slot_pipeline) or, when the op has no slot, straight into
// the resource's padded buffer (pad_direct_pipeline).
static void record_one_pad_dispatch(
    VkCommandBuffer cmd, const DftData & d, const VK_Resource & resource,
    const SlotOp & op) {

    const auto & cfg = d.planes[op.plane];
    PushConstants pc = base_pc(d);
    // the pad kernels apply the temporal offset themselves (pc.pad_t0 *
    // up_slice); do not add it to src_base as well
    pc.src_base = static_cast<int32_t>(cfg.upload_offset);
    pc.pad_t0 = op.t;
    pc.width = cfg.width;
    pc.height = cfg.height;
    pc.src_stride = cfg.width;
    pc.dst_stride = cfg.width;

    if (op.slot >= 0) {
        pc.padded_base = static_cast<int32_t>(op.slot_base);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pad_slot_pipeline);
    } else {
        pc.padded_base = static_cast<int32_t>(cfg.padded_offset);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pad_direct_pipeline);
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(PushConstants), &pc);
    const uint32_t max_grid_x = d.device->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d.device->limits.maxComputeWorkGroupCount[1];
    const uint32_t gx = std::min<uint32_t>(
        (static_cast<uint32_t>(cfg.pw) + 31) / 32, max_grid_x);
    const uint32_t gy = std::min<uint32_t>(
        (static_cast<uint32_t>(cfg.ph) + 7) / 8, max_grid_y);
    vkCmdDispatch(cmd, std::max(gx, 1u), std::max(gy, 1u), 1);
}

// Records the slot-pad command buffer (resource.cmd_pad[0]): one pad
// dispatch per pad op. Used by the GPU bench (debug); in the normal flow
// each pad op records and submits its own command buffer at claim time.
static std::optional<std::string> record_pad_cb(
    const DftData & d, VK_Resource & resource, const std::vector<SlotOp> & ops) {

    VkCommandBuffer cmd = resource.cmd_pad[0];

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        return "vkBeginCommandBuffer (pad) failed";
    }

    if (!trivial_kernels()) {
        for (const SlotOp & op : ops) {
            if (!op.is_pad) {
                continue;
            }
            record_one_pad_dispatch(cmd, d, resource, op);
        }
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer (pad) failed";
    }
    return std::nullopt;
}

// Records and submits one pad op's command buffer immediately (called at
// claim time, still holding slot_lock; takes queue_lock). Signalling the
// slot's semaphores here — before any reader of this generation can commit —
// makes a backward semaphore dependency on the queue impossible: every
// reader's copy submit is queued after this pad submit.
static bool submit_pad_op(
    const DftData & d, VK_Resource & resource, const SlotOp & op) {

    const VkDevice dev = d.device->device;
    VkCommandBuffer cmd = resource.cmd_pad[op.plane * d.tw + op.t];
    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        return false;
    }
    if (!trivial_kernels()) {
        record_one_pad_dispatch(cmd, d, resource, op);
    }
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        return false;
    }

    std::lock_guard qlock(*resource.queue_lock);
    VkSubmitInfo si {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };
    if (op.slot >= 0) {
        const auto & st = d.slots[op.plane * d.slot_count + op.slot];
        si.signalSemaphoreCount = static_cast<uint32_t>(st.sem.size());
        si.pSignalSemaphores = st.sem.data();
    }
    return vkQueueSubmit(resource.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
}

// Records the per-frame D2D copy command buffer (resource.cmd2): every slot
// of the temporal window -> this resource's padded planes. The copies read
// the slots (written by pads that may have run on another queue), so the
// submit carrying this buffer waits on the slot semaphores.
static std::optional<std::string> record_copy_cb(
    const DftData & d, VK_Resource & resource, const std::vector<SlotOp> & ops) {

    VkCommandBuffer cmd = resource.cmd2;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        return "vkBeginCommandBuffer (copy) failed";
    }

    if (!trivial_kernels()) {
        for (const SlotOp & op : ops) {
            if (op.slot < 0) {
                continue;  // direct pad: the plane is already in padded_buf
            }
            const auto & cfg = d.planes[op.plane];
            VkBufferCopy copy {
                .srcOffset = op.slot_base,
                .dstOffset = cfg.padded_offset +
                    static_cast<VkDeviceSize>(op.t) * cfg.slot_plane_bytes,
                .size = cfg.slot_plane_bytes
            };
            vkCmdCopyBuffer(cmd, d.slot_buf, resource.padded_buf, 1, &copy);
        }

        // the copies must be visible to the fused kernel (next command buffer)
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer (copy) failed";
    }
    return std::nullopt;
}

// Records the pre-recorded fused + col2im command buffer (resource.cmd).
static std::optional<std::string> record_fused_col2im_cb(
    const DftData & d, VK_Resource & resource,
    bool with_fused, bool with_col2im) {

    const VkDevice dev = d.device->device;
    VkCommandBuffer cmd = resource.cmd;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        return "vkBeginCommandBuffer (fused) failed";
    }

    const PushConstants base = base_pc(d);
    const uint32_t max_grid_x = d.device->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d.device->limits.maxComputeWorkGroupCount[1];

    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
            continue;
        }
        if (trivial_kernels()) {
            continue;
        }
        const auto & cfg = d.planes[plane];

        PushConstants pc = base;
        pc.padded_base = static_cast<int32_t>(cfg.padded_offset);
        pc.spatial_base = static_cast<int32_t>(cfg.spatial_offset);
        pc.dst_base = static_cast<int32_t>(d.upload_total + cfg.download_offset);
        pc.src_base = static_cast<int32_t>(cfg.upload_offset);
        pc.width = cfg.width;
        pc.height = cfg.height;
        pc.src_stride = cfg.width;
        pc.dst_stride = cfg.width;

        if (with_fused) {
            // fused kernel (SUB_BLOCKS=8 blocks per 128-thread workgroup)
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.fused_pipeline[d.radius]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdPushConstants(cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(PushConstants), &pc);
            const uint32_t blocks = static_cast<uint32_t>(cfg.num_blocks);
            const uint32_t sub_blocks = blocks / 8 + (blocks % 8 != 0 ? 1 : 0);
            const uint32_t grid_x = std::min<uint32_t>(sub_blocks, max_grid_x);
            vkCmdDispatch(cmd, std::max(grid_x, 1u), 1, 1);
        }

        if (with_col2im) {
            // the col2im kernel reads the fused kernel's writes
            VkMemoryBarrier mem_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

            // col2im kernel
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.col2im_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdPushConstants(cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(PushConstants), &pc);
            const uint32_t grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.pw) + 31) / 32, max_grid_x);
            const uint32_t grid_y = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.ph) + 7) / 8, max_grid_y);
            vkCmdDispatch(cmd, std::max(grid_x, 1u), std::max(grid_y, 1u), 1);
        }
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer (fused) failed";
    }
    return std::nullopt;
}

static const VSFrame *VS_CC DftGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    DftData * d = static_cast<DftData *>(instanceData);

    if (activationReason == arInitial) {
        const int start = std::max(n - d->radius, 0);
        const int end = std::min(n + d->radius, d->vi->numFrames - 1);
        for (int i = start; i <= end; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const int tw = d->tw;

        std::array<const VSFrame *, 7> src {};
        for (int t = 0; t < tw; ++t) {
            const int idx = std::clamp(n - d->radius + t, 0, d->vi->numFrames - 1);
            src[t] = vsapi->getFrameFilter(idx, d->node, frameCtx);
        }
        const VSFrame * center = src[d->radius];

        const int pl[] = { 0, 1, 2 };
        const VSFrame * fr[] = {
            d->process[0] ? nullptr : center,
            d->process[1] ? nullptr : center,
            d->process[2] ? nullptr : center
        };

        VSFrame * dst = vsapi->newVideoFrame2(
            &d->vi->format, d->vi->width, d->vi->height, fr, pl, center, core);

        auto t0 = std::chrono::steady_clock::now();

        // Note on the frame cache and out-of-order processing: the slot
        // state (gen/committed) under slot_lock decides who pads a slot (the
        // first frame that touches it, in wall-clock order) and who reads it.
        // The pad is submitted at claim time, still under slot_lock, so every
        // reader's copy submit (which needs slot_lock to commit first) is
        // queued after the pad submit that signals the slot's semaphores — a
        // backward semaphore dependency on the queue, which stalls it, is
        // structurally impossible. A slot is only reclaimed for a new source
        // once every committed reader's resource has been reused (frame_gen
        // advanced past their commit generation), which implies their frames
        // — copies included — are fully done, so the overwrite never races a
        // reader's copy. No host fence waits anywhere: no deadlocks.
        d->semaphore.acquire();
        auto t1 = std::chrono::steady_clock::now();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        auto set_error = [&](const std::string & error_message) {
            d->resources_lock.lock();
            d->resources.push_back(std::move(resource));
            d->resources_lock.unlock();
            d->semaphore.release();
            vsapi->setFilterError(("DFTTest: " + error_message).c_str(), frameCtx);
            for (int t = 0; t < tw; ++t) {
                vsapi->freeFrame(src[t]);
            }
            vsapi->freeFrame(dst);
            return nullptr;
        };

        VkDevice dev = d->device->device;
        float * map = resource.map;

        // Frame generation for this resource: a slot may only be reclaimed
        // for a new source once every committed reader's resource has been
        // reused (frame_gen advanced past their commit generation), which
        // implies those frames — copies included — are fully done.
        const long long my_gen =
            d->res_meta[resource.id]->frame_gen.fetch_add(1, std::memory_order_relaxed) + 1;

        checkVK(vkResetFences(dev, 1, &resource.fence));

        if (dfttest_trace()) {
            fprintf(stderr, "[dfttest-trace] n=%d res=%d gen=%lld start\n",
                n, resource.id, (long long)my_gen);
        }

        auto t2 = std::chrono::steady_clock::now();

        const bool coherent =
            !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Padded-source frame cache: each source frame of the temporal window
        // is either already reflect-pad'd into a slot (D2D copy) or new
        // (upload + pad into a slot now; if every slot is busy, pad the slice
        // straight into this resource's padded buffer instead). The slot state
        // machine (under slot_lock) decides who pads (first frame in
        // wall-clock order) and who reads. See the note above for why the pad
        // is submitted at claim time and how slot reclamation stays safe.
        std::vector<SlotOp> ops;
        ops.reserve(d->vi->format.numPlanes * tw);
        std::vector<VkSemaphore> waits;
        waits.reserve(ops.capacity());
        const bool force_pad = getenv("VSFEEL_DFTTEST_FORCEPAD") != nullptr;
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & cfg = d->planes[plane];
            const size_t row_bytes = static_cast<size_t>(cfg.width) * d->bytes;
            for (int t = 0; t < tw; ++t) {
                const int idx = std::clamp(n - d->radius + t, 0, d->vi->numFrames - 1);
                const int which = n - std::max(0, idx - d->radius);
                SlotOp op;
                op.plane = plane;
                op.t = t;
                op.which = which;

                // find the slot that owns this source (any slot may hold it:
                // reclamation can move a source off its natural slot)
                auto find_owner = [&]() {
                    for (int s = 0; s < d->slot_count; ++s) {
                        if (d->slots[plane * d->slot_count + s].gen == idx) {
                            return s;
                        }
                    }
                    return -1;
                };

                // register this frame as a reader of the slot's current
                // generation and wait on its semaphore for this reader offset
                // (deduped: at temporal boundaries two slices of one frame map
                // to the same source, hence the same slot and which)
                auto commit_reader = [&](int s) {
                    SlotState & st = d->slots[plane * d->slot_count + s];
                    st.committed.push_back({ resource.id, my_gen });
                    op.slot = s;
                    op.slot_base = cfg.slot_offset +
                        static_cast<VkDeviceSize>(s) * cfg.slot_plane_bytes;
                    const VkSemaphore wsem = st.sem[which];
                    bool dup = false;
                    for (auto w : waits) {
                        if (w == wsem) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup) {
                        waits.push_back(wsem);
                    }
                };

                bool done = false;
                {
                    std::lock_guard lk(d->slot_lock);
                    if (!force_pad) {
                        const int owner = find_owner();
                        if (owner >= 0) {
                            commit_reader(owner);
                            done = true;
                        }
                    }
                }
                if (!done) {
                    // padder candidate: upload this source plane to this
                    // resource's staging, flush it, then re-claim (another
                    // frame may have padded this source in the meantime, in
                    // which case this upload was wasted)
                    const uint8_t * srcp = vsapi->getReadPtr(src[t], plane);
                    const int src_stride = vsapi->getStride(src[t], plane);
                    uint8_t * dstp = reinterpret_cast<uint8_t *>(map) + cfg.upload_offset +
                        static_cast<size_t>(t) * cfg.upload_bytes / tw;
                    if (src_stride == static_cast<int>(row_bytes)) {
                        copy_stream_out(dstp, srcp, static_cast<size_t>(cfg.height) * row_bytes);
                    } else {
                        for (int y = 0; y < cfg.height; ++y) {
                            copy_stream_out(dstp + static_cast<size_t>(y) * row_bytes,
                                srcp + static_cast<size_t>(y) * src_stride, row_bytes);
                        }
                    }
                    if (!coherent) {
                        VkMappedMemoryRange flush_range {
                            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                            .pNext = nullptr,
                            .memory = resource.staging_mem,
                            .offset = cfg.upload_offset +
                                static_cast<size_t>(t) * cfg.upload_bytes / tw,
                            .size = cfg.upload_bytes / tw,
                        };
                        checkVK(vkFlushMappedMemoryRanges(dev, 1, &flush_range));
                    }

                    std::lock_guard lk(d->slot_lock);
                    int owner = -1;
                    if (!force_pad) {
                        owner = find_owner();
                    }
                    if (owner >= 0) {
                        commit_reader(owner);
                        done = true;
                    } else {
                        // padder: the slot may be taken iff it is empty or its
                        // old generation's readers are all done (their
                        // resources reused); natural slot first, then any free
                        auto slot_free = [&](const SlotState & st) {
                            if (st.gen == -1) {
                                return true;
                            }
                            for (const auto & cr : st.committed) {
                                if (d->res_meta[cr.first]->frame_gen.load(
                                        std::memory_order_relaxed) <= cr.second) {
                                    return false;
                                }
                            }
                            return true;
                        };
                        const int natural = static_cast<int>(idx % d->slot_count);
                        int slot = -1;
                        for (int i = 0; i < d->slot_count; ++i) {
                            const int s = (natural + i) % d->slot_count;
                            if (slot_free(d->slots[plane * d->slot_count + s])) {
                                slot = s;
                                break;
                            }
                        }
                        if (slot >= 0) {
                            SlotState & st = d->slots[plane * d->slot_count + slot];
                            if (st.gen != -1) {
                                if (dfttest_trace()) {
                                    fprintf(stderr,
                                        "[dfttest-trace]   n=%d reclaim p%d slot=%d oldgen=%lld newgen=%d\n",
                                        n, plane, slot, (long long)st.gen, idx);
                                }
                                for (auto & sem : st.sem) {
                                    vkDestroySemaphore(dev, sem, nullptr);
                                }
                                st.sem.clear();
                            }
                            st.gen = idx;
                            st.committed = { { resource.id, my_gen } };
                            st.sem.resize(d->tw);
                            for (auto & sem : st.sem) {
                                VkSemaphoreCreateInfo sem_info {
                                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                    .pNext = nullptr,
                                    .flags = 0
                                };
                                checkVK(vkCreateSemaphore(dev, &sem_info, nullptr, &sem));
                            }
                            op.slot = slot;
                            op.slot_base = cfg.slot_offset +
                                static_cast<VkDeviceSize>(slot) * cfg.slot_plane_bytes;
                            op.is_pad = true;
                            if (!submit_pad_op(*d, resource, op)) {
                                return set_error("vkQueueSubmit (pad) failed");
                            }
                        } else {
                            // every slot is busy: pad this slice straight into
                            // the padded buffer (no slot, no copy, no sems)
                            op.is_pad = true;
                            if (!submit_pad_op(*d, resource, op)) {
                                return set_error("vkQueueSubmit (pad direct) failed");
                            }
                        }
                    }
                }

                if (dfttest_trace()) {
                    fprintf(stderr, "[dfttest-trace]   n=%d op p%d t%d idx=%d slot=%d which=%d %s\n",
                        n, plane, t, idx, op.slot, which, op.is_pad ? "PAD" : "read");
                }
                ops.push_back(op);
            }
        }

        auto t3 = std::chrono::steady_clock::now();

        if (const char * gb = getenv("VSFEEL_DFTTEST_GPU_BENCH"); gb && n == 0) {
            VkDevice dev0 = d->device->device;
            const int iters = atoi(gb);
            // the ops loop already submitted this frame's pads; wait for the
            // queue to drain before re-recording their command buffers
            vkDeviceWaitIdle(dev0);
            auto bench_stage = [&](int stage, const char * name) {
                if (record_pad_cb(*d, resource, ops)) {
                    return;
                }
                if (record_copy_cb(*d, resource, ops)) {
                    return;
                }
                if (record_fused_col2im_cb(*d, resource, stage >= 1, stage >= 2)) {
                    return;
                }
                auto gb_start = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) {
                    std::lock_guard lock(*resource.queue_lock);
                    vkResetFences(dev0, 1, &resource.fence);
                    VkCommandBuffer pad_cb = resource.cmd_pad[0];
                    VkSubmitInfo pad_si {
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .pNext = nullptr,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &pad_cb
                    };
                    vkQueueSubmit(resource.queue, 1, &pad_si, VK_NULL_HANDLE);
                    VkCommandBuffer copy_cb = resource.cmd2;
                    VkSubmitInfo si {
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .pNext = nullptr,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &copy_cb
                    };
                    vkQueueSubmit(resource.queue, 1, &si, resource.fence);
                    vkWaitForFences(dev0, 1, &resource.fence, VK_TRUE, UINT64_MAX);
                    if (stage >= 1) {
                        VkCommandBuffer cb = resource.cmd;
                        VkSubmitInfo si2 {
                            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .pNext = nullptr,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &cb
                        };
                        vkQueueSubmit(resource.queue, 1, &si2, resource.fence);
                        vkWaitForFences(dev0, 1, &resource.fence, VK_TRUE, UINT64_MAX);
                    }
                }
                auto gb_end = std::chrono::steady_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(gb_end - gb_start).count() / 1e6 / iters;
                fprintf(stderr, "[dfttest-gpubench] %-20s %.4f ms\n", name, ms);
            };
            bench_stage(0, "pad+copy");
            bench_stage(1, "pad+copy+fused");
            bench_stage(2, "pad+copy+fused+col2im");
        }

        if (const auto err = record_copy_cb(*d, resource, ops)) {
            set_error(*err);
            return nullptr;
        }
        if (const auto err = record_fused_col2im_cb(*d, resource, true, true)) {
            set_error(*err);
            return nullptr;
        }

        if (dfttest_trace()) {
            fprintf(stderr, "[dfttest-trace]   n=%d res=%d submit: waits=%zu\n",
                n, resource.id, waits.size());
        }

        {
            std::lock_guard lock(*resource.queue_lock);
            const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkCommandBuffer copy_cb = resource.cmd2;
            VkSubmitInfo si {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = static_cast<uint32_t>(waits.size()),
                .pWaitSemaphores = waits.empty() ? nullptr : waits.data(),
                .pWaitDstStageMask = waits.empty() ? nullptr : &wait_stage,
                .commandBufferCount = 1,
                .pCommandBuffers = &copy_cb
            };

            checkVK(vkQueueSubmit(resource.queue, 1, &si, VK_NULL_HANDLE));
            if (dfttest_trace()) {
                fprintf(stderr, "[dfttest-trace]   n=%d res=%d copy submitted\n", n, resource.id);
            }
        }

        {
            std::lock_guard lock(*resource.queue_lock);
            VkCommandBuffer cb = resource.cmd;
            VkSubmitInfo si {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &cb
            };
            checkVK(vkQueueSubmit(resource.queue, 1, &si, resource.fence));
            if (dfttest_trace()) {
                fprintf(stderr, "[dfttest-trace]   n=%d res=%d fused submitted\n", n, resource.id);
            }
        }

        auto t4 = std::chrono::steady_clock::now();
        if (dfttest_trace()) {
            fprintf(stderr, "[dfttest-trace]   n=%d res=%d wait fence\n", n, resource.id);
        }
        checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));
        auto t5 = std::chrono::steady_clock::now();

        if (const char * dp = getenv("VSFEEL_DFTTEST_DUMP_PAD"); dp && n == atoi(dp)) {
            VkDeviceSize dump_size = 0;
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (d->process[plane]) {
                    dump_size = d->planes[plane].padded_bytes;
                }
            }
            {
                VkCommandBufferAllocateInfo ai {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .commandPool = resource.pool,
                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    .commandBufferCount = 1
                };
                VkCommandBuffer dcmd;
                checkVK(vkAllocateCommandBuffers(dev, &ai, &dcmd));
                VkCommandBufferBeginInfo bi { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(dcmd, &bi);
                VkBufferCopy bc { .srcOffset = 0, .dstOffset = 0, .size = dump_size };
                vkCmdCopyBuffer(dcmd, resource.padded_buf, resource.staging, 1, &bc);
                vkEndCommandBuffer(dcmd);
                vkResetFences(dev, 1, &resource.fence);
                VkSubmitInfo si { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1, .pCommandBuffers = &dcmd };
                vkQueueSubmit(resource.queue, 1, &si, resource.fence);
                vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX);
                const char * path = getenv("VSFEEL_DFTTEST_DUMP_PATH");
                FILE * f = fopen(path ? path : "/tmp/opencode/pad_dump.bin", "wb");
                fwrite(map, 1, dump_size, f);
                fclose(f);
                vkFreeCommandBuffers(dev, resource.pool, 1, &dcmd);
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
                    .offset = d->upload_total + cfg.download_offset,
                    .size = cfg.download_bytes,
                });
            }
            checkVK(vkInvalidateMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
        }

        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & cfg = d->planes[plane];
            const int dst_stride = vsapi->getStride(dst, plane);
            const int row_bytes = cfg.width * d->bytes;
            const uint8_t * h_bufferp = reinterpret_cast<const uint8_t *>(map) +
                d->upload_total + cfg.download_offset;
            uint8_t * dstp = vsapi->getWritePtr(dst, plane);
            for (int y = 0; y < cfg.height; ++y) {
                copy_stream_read(dstp + static_cast<size_t>(y) * dst_stride,
                    h_bufferp + static_cast<size_t>(y) * row_bytes, row_bytes);
            }
        }

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        auto t6 = std::chrono::steady_clock::now();
        d->t_acquire_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        d->t_upload_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        d->t_submit_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
        d->t_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count();
        d->t_download_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count();
        d->t_total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t0).count();
        d->nframes.fetch_add(1, std::memory_order::relaxed);

        for (int t = 0; t < tw; ++t) {
            vsapi->freeFrame(src[t]);
        }

        return dst;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC DftFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    DftData * d = static_cast<DftData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC DftCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<DftData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
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
    d->bytes = bits / 8;

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
        n_slocation = vsapi->mapNumElements(in, "slocation");
    }
    if (vsapi->mapNumElements(in, "ssx") > 0) {
        ssx = vsapi->mapGetFloatArray(in, "ssx", &error);
        n_ssx = vsapi->mapNumElements(in, "ssx");
    }
    if (vsapi->mapNumElements(in, "ssy") > 0) {
        ssy = vsapi->mapGetFloatArray(in, "ssy", &error);
        n_ssy = vsapi->mapNumElements(in, "ssy");
    }
    if (vsapi->mapNumElements(in, "sst") > 0) {
        sst = vsapi->mapGetFloatArray(in, "sst", &error);
        n_sst = vsapi->mapNumElements(in, "sst");
    }
    const int array_counts[4] { n_slocation, n_ssx, n_ssy, n_sst };
    for (int cnt : array_counts) {
        if (cnt != 0 && (cnt % 2 != 0 || cnt < 2)) {
            return set_error("number of elements in slocation/ssx/ssy/sst must be a non-zero multiple of 2.");
        }
    }

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
    VkDeviceSize upload_sum = 0;
    VkDeviceSize download_sum = 0;
    VkDeviceSize padded_sum = 0;
    VkDeviceSize spatial_sum = 0;
    for (int plane = 0; plane < num_planes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
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

        cfg.upload_offset = upload_sum;
        cfg.upload_bytes = static_cast<VkDeviceSize>(d->tw) * cfg.height * cfg.width * d->bytes;
        upload_sum += cfg.upload_bytes;

        cfg.padded_offset = padded_sum;
        cfg.padded_bytes = static_cast<VkDeviceSize>(d->tw) * pad_elems * d->bytes;
        padded_sum += cfg.padded_bytes;

        cfg.download_offset = download_sum;
        cfg.download_bytes = static_cast<VkDeviceSize>(cfg.height) * cfg.width * d->bytes;
        download_sum += cfg.download_bytes;

        cfg.spatial_offset = spatial_sum;
        spatial_sum += nblk * 256;           // floats (center slice only)

        cfg.slot_plane_bytes = pad_elems * d->bytes;

        if (d->tw * pad_elems >= (1ll << 31) || nblk * 256 >= (1ll << 31)) {
            return set_error("frame too large (padded plane exceeds 2^31 elements).");
        }
    }
    if (upload_sum == 0) {
        return set_error("no planes to process.");
    }
    d->upload_total = (upload_sum + 31) & ~VkDeviceSize(31);
    d->download_total = (download_sum + 31) & ~VkDeviceSize(31);
    d->padded_total = (padded_sum + 31) & ~VkDeviceSize(31);
    d->spatial_total = (spatial_sum + 7) & ~VkDeviceSize(7);
    if (padded_sum >= (1ull << 32) || spatial_sum >= (1ull << 32)) {
        return set_error("frame too large (device buffers exceed 4 GiB).");
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

        SigmaFunc fx, fy, ft;
        bool shared = false;
        if (slocation != nullptr) {
            fx = SigmaFunc::initPacks(slocation, n_slocation, norm);
            fy = fx;
            ft = fx;
            shared = true;
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
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
    }

    VkDevice dev = d->device->device;

    int effective_streams = std::max(d->num_streams, 8);
    if (const char * es = getenv("VSFEEL_DFTTEST_STREAMS")) {
        effective_streams = atoi(es);
        if (effective_streams < 1) effective_streams = 1;
    }

    // frame-cache slots: a slot holds one padded source plane; K must exceed the
    // number of distinct source frames in flight (S-1 behind + 1 ahead + reuse)
    d->slot_count = effective_streams + 3;
    {
        VkDeviceSize slot_sum = 0;
        for (int plane = 0; plane < num_planes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            auto & cfg = d->planes[plane];
            cfg.slot_offset = slot_sum;
            slot_sum += static_cast<VkDeviceSize>(d->slot_count) * cfg.slot_plane_bytes;
        }
        d->slot_total = (slot_sum + 31) & ~VkDeviceSize(31);
    }

    // ------------------------------------------------------------------
    // Pipeline layout, descriptor set layout and descriptor pool
    // ------------------------------------------------------------------
    {
        VkDescriptorSetLayoutBinding bindings[5] {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };

        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 5,
            .pBindings = bindings
        };

        checkVK(vkCreateDescriptorSetLayout(
            d->device->device, &layout_info, nullptr, &d->set_layout));
    }
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * static_cast<uint32_t>(effective_streams)
        };

        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(effective_streams),
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size
        };

        checkVK(vkCreateDescriptorPool(
            d->device->device, &pool_info, nullptr, &d->desc_pool));
    }

    // ------------------------------------------------------------------
    // Constant buffer: window + window_freq + sigma array
    // ------------------------------------------------------------------
    {
        const size_t n_window = static_cast<size_t>(d->tw) * 256;
        const size_t n_freq = zmean ? static_cast<size_t>(d->tw) * 16 * 9 * 2 : 0;
        const size_t n_sigma = sigma_array.empty() ? 0 : static_cast<size_t>(d->tw) * 16 * 9;

        d->wt_bytes = static_cast<VkDeviceSize>((n_window + n_freq + n_sigma) * sizeof(float));
        d->wt_bytes = std::max<VkDeviceSize>(d->wt_bytes, 16);
        d->wf_base = zmean ? static_cast<int32_t>(n_window) : -1;
        d->sigma_base = !sigma_array.empty() ? static_cast<int32_t>(n_window + n_freq) : -1;

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

        for (size_t i = 0; i < n_window; ++i) {
            d->wt_map[i] = static_cast<float>(window[i]);
        }
        if (zmean) {
            for (size_t i = 0; i < n_freq; ++i) {
                d->wt_map[n_window + i] = static_cast<float>(window_freq[i]);
            }
        }
        if (!sigma_array.empty()) {
            for (size_t i = 0; i < n_sigma; ++i) {
                d->wt_map[n_window + n_freq + i] = static_cast<float>(sigma_array[i]);
            }
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

    // ------------------------------------------------------------------
    // Shader modules and pipelines
    // ------------------------------------------------------------------
    {
        const uint32_t * pad_slot_code = nullptr;
        size_t pad_slot_size = 0;
        const uint32_t * pad_direct_code = nullptr;
        size_t pad_direct_size = 0;
        const uint32_t * col2im_code = nullptr;
        size_t col2im_size = 0;
        const uint32_t * fused_code[4] {};
        size_t fused_size[4] {};
        switch (d->bits) {
            case 16:
                pad_slot_code = dfttest_16_pad_slot_spv; pad_slot_size = dfttest_16_pad_slot_spv_size;
                pad_direct_code = dfttest_16_pad_direct_spv; pad_direct_size = dfttest_16_pad_direct_spv_size;
                col2im_code = dfttest_16_col2im_spv; col2im_size = dfttest_16_col2im_spv_size;
                fused_code[0] = dfttest_16_fused_r0_spv; fused_size[0] = dfttest_16_fused_r0_spv_size;
                fused_code[1] = dfttest_16_fused_r1_spv; fused_size[1] = dfttest_16_fused_r1_spv_size;
                fused_code[2] = dfttest_16_fused_r2_spv; fused_size[2] = dfttest_16_fused_r2_spv_size;
                fused_code[3] = dfttest_16_fused_r3_spv; fused_size[3] = dfttest_16_fused_r3_spv_size;
                break;
            case 32:
                pad_slot_code = dfttest_32_pad_slot_spv; pad_slot_size = dfttest_32_pad_slot_spv_size;
                pad_direct_code = dfttest_32_pad_direct_spv; pad_direct_size = dfttest_32_pad_direct_spv_size;
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
            const auto result = create_shader_module(*d->device, pad_slot_code, pad_slot_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_slot_module = std::get<VkShaderModule>(result);
        }
        {
            const auto result = create_shader_module(*d->device, pad_direct_code, pad_direct_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_direct_module = std::get<VkShaderModule>(result);
        }
        for (int r = 0; r < 4; ++r) {
            const auto result = create_shader_module(*d->device, fused_code[r], fused_size[r]);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->fused_module[r] = std::get<VkShaderModule>(result);
        }
        {
            const auto result = create_shader_module(*d->device, col2im_code, col2im_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->col2im_module = std::get<VkShaderModule>(result);
        }
        {
            const auto result = create_pipeline(*d->device, d->pad_slot_module, d->pipeline_layout,
                d->device->subgroup_size_control ? 32 : 0);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_slot_pipeline = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->device, d->pad_direct_module, d->pipeline_layout,
                d->device->subgroup_size_control ? 32 : 0);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_direct_pipeline = std::get<VkPipeline>(result);
        }
        for (int r = 0; r < 4; ++r) {
            const auto result = create_pipeline(*d->device, d->fused_module[r], d->pipeline_layout,
                d->device->subgroup_size_control ? 32 : 0, d->filter_type,
                d->zmean ? 1 : 0);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->fused_pipeline[r] = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->device, d->col2im_module, d->pipeline_layout,
                d->device->subgroup_size_control ? 32 : 0);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->col2im_pipeline = std::get<VkPipeline>(result);
        }
    }

    // ------------------------------------------------------------------
    // Buffers and resources
    // ------------------------------------------------------------------
    const VkDeviceSize min_size = 4;
    const VkDeviceSize staging_size = std::max(d->upload_total + d->download_total, 2 * min_size);
    const VkDeviceSize padded_size = std::max(d->padded_total, min_size);
    const VkDeviceSize spatial_size = std::max<VkDeviceSize>(
        d->spatial_total * sizeof(float), min_size);

    d->semaphore.current.store(effective_streams - 1, std::memory_order::relaxed);
    d->resources.reserve(effective_streams);

    // ---- frame-cache slot buffer (semaphores are created per generation) ----
    {
        const VkDeviceSize slot_size = std::max(d->slot_total, VkDeviceSize(4));
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = slot_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };
        checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &d->slot_buf));
        const auto result = allocate_memory(
            *d->device, d->slot_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->slot_mem = std::get<AllocatedMemory>(result).memory;

        d->slots.resize(num_planes * d->slot_count);
    }

    const uint32_t num_queues = std::min(
        d->num_streams, static_cast<int>(d->device->queue_count));

    for (int i = 0; i < effective_streams; ++i) {
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
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = padded_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.padded_buf));
        }
        {
            const auto result = allocate_memory(
                *d->device, resource.padded_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.padded_mem = std::get<AllocatedMemory>(result).memory;
        }

        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = spatial_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.spatial_buf));
        }
        {
            const auto result = allocate_memory(
                *d->device, resource.spatial_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.spatial_mem = std::get<AllocatedMemory>(result).memory;
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
            const uint32_t n_pad = static_cast<uint32_t>(d->tw) * num_planes;
            VkCommandBufferAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = resource.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 2 + n_pad
            };
            std::vector<VkCommandBuffer> cbs(2 + n_pad);
            checkVK(vkAllocateCommandBuffers(dev, &alloc_info, cbs.data()));
            resource.cmd = cbs[0];
            resource.cmd2 = cbs[1];
            resource.cmd_pad.assign(cbs.begin() + 2, cbs.end());
        }
        {
            VkFenceCreateInfo fence_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            checkVK(vkCreateFence(dev, &fence_info, nullptr, &resource.fence));
        }
        resource.id = i;
        d->res_meta.push_back(std::make_unique<ResMeta>());
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
            VkDescriptorBufferInfo staging_info {
                .buffer = resource.staging,
                .offset = 0,
                .range = staging_size
            };
            VkDescriptorBufferInfo spatial_info {
                .buffer = resource.spatial_buf,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo padded_info {
                .buffer = resource.padded_buf,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo slot_info {
                .buffer = d->slot_buf,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };

            VkWriteDescriptorSet writes[5] {
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
                    .pBufferInfo = &staging_info,
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
                    .pBufferInfo = &spatial_info,
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
                    .pBufferInfo = &padded_info,
                    .pTexelBufferView = nullptr
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = 4,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &slot_info,
                    .pTexelBufferView = nullptr
                },
            };

            vkUpdateDescriptorSets(dev, 5, writes, 0, nullptr);
        }

        checkVK(vkMapMemory(dev, resource.staging_mem, 0, staging_size, 0, reinterpret_cast<void **>(&resource.map)));

        resource.queue = d->device->queues[i % num_queues].queue;
        resource.queue_lock = d->device->queues[i % num_queues].lock.get();

        if (const auto err = record_fused_col2im_cb(*d, resource, true, true)) {
            return set_error(*err);
        }

        d->resources.push_back(std::move(resource));
    }

    VSFilterDependency deps[1] = {
        { d->node, d->radius > 0 ? rpGeneral : rpStrictSpatial }
    };

    DftData *data = d.release();

    vsapi->createVideoFilter(
        out, "DFTTest", data->vi,
        DftGetFrame, DftFree,
        fmParallel, deps, 1, data, core);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_dfttest(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "DFTTest",
        "clip:vnode;"
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
        "clip:vnode;",
        DftCreate, nullptr, plugin
    );
}
