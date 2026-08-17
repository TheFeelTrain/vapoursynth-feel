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
constexpr int MAX_RADIUS = 3;       // tbsize up to 7

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
    int hn {};                      // block counts
    int vn {};
    int num_blocks {};
    VkDeviceSize upload_offset {};  // staging byte offset of the tight upload region
    VkDeviceSize upload_bytes {};   // tw * height * width * bytes (tight rows)
    VkDeviceSize padded_offset {};  // padded buffer byte offset
    VkDeviceSize padded_bytes {};   // tw * pw * ph * bytes
    VkDeviceSize download_offset {};// staging byte offset of the output region
    VkDeviceSize download_bytes {}; // h * width * bytes (tight rows)
    VkDeviceSize spatial_offset {}; // float element offset into the spatial buffer
};

struct VK_Resource {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    VkBuffer padded_buf {};         // device-local padded source planes
    VkDeviceMemory padded_mem {};
    VkBuffer spatial_buf {};        // device-local float block buffer
    VkDeviceMemory spatial_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
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

    int device_id, num_streams;
    int bits, bytes;
    bool sample_type_float {};

    int radius, block_step, tbsize;
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
    VkShaderModule pad_module {};
    VkShaderModule col2im_module {};
    VkShaderModule fused_module[4] {};
    VkPipeline pad_pipeline {};
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
    bool need_fill {};
    std::array<PlaneConfig, 3> planes {};
    ticket_semaphore semaphore;
    std::vector<VK_Resource> resources;
    std::mutex resources_lock;

    ~DftData() {
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

        if (pad_pipeline) {
            vkDestroyPipeline(dev, pad_pipeline, nullptr);
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
        if (pad_module) {
            vkDestroyShaderModule(dev, pad_module, nullptr);
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
    const VK_Device & dev, VkShaderModule module, VkPipelineLayout layout) {

    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = nullptr
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
static std::optional<std::string> record_command_buffer(
    const DftData & d, VK_Resource & resource) {

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
    // their download regions (staging) and padded regions (device-local) must
    // be cleared before every submission
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
        vkCmdFillBuffer(resource.cmd, resource.padded_buf, 0, d.padded_total, 0);
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

    const PushConstants base {
        .padded_base = 0,
        .spatial_base = 0,
        .dst_base = 0,
        .src_base = 0,
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

    const uint32_t max_grid_x = d.device->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d.device->limits.maxComputeWorkGroupCount[1];

    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
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

        // pad kernel: reflection-pad the tight upload rows into the device-local
        // padded buffer
        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pad_pipeline);
        vkCmdBindDescriptorSets(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
        vkCmdPushConstants(resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);
        {
            const uint32_t grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.pw) + 31) / 32, max_grid_x);
            const uint32_t grid_y = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.ph) + 7) / 8, max_grid_y);
            vkCmdDispatch(resource.cmd, std::max(grid_x, 1u), std::max(grid_y, 1u), 1);
        }

        // the fused kernel reads the pad kernel's writes
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

        // fused kernel (SUB_BLOCKS=4 blocks per 64-thread workgroup)
        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.fused_pipeline[d.radius]);
        vkCmdBindDescriptorSets(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
        vkCmdPushConstants(resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);
        {
            const uint32_t blocks = static_cast<uint32_t>(cfg.num_blocks);
            const uint32_t sub_blocks = blocks / 4 + (blocks % 4 != 0 ? 1 : 0);
            const uint32_t grid_x = std::min<uint32_t>(sub_blocks, max_grid_x);
            vkCmdDispatch(resource.cmd, std::max(grid_x, 1u), 1, 1);
        }

        // the col2im kernel reads the fused kernel's writes
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

        // col2im kernel
        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.col2im_pipeline);
        vkCmdBindDescriptorSets(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
        vkCmdPushConstants(resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);
        {
            const uint32_t grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.pw) + 31) / 32, max_grid_x);
            const uint32_t grid_y = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.ph) + 7) / 8, max_grid_y);
            vkCmdDispatch(resource.cmd, std::max(grid_x, 1u), std::max(grid_y, 1u), 1);
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
            vsapi->setFilterError(("DFTTest: " + error_message).c_str(), frameCtx);
            for (int t = 0; t < tw; ++t) {
                vsapi->freeFrame(src[t]);
            }
            vsapi->freeFrame(dst);
            return nullptr;
        };

        VkDevice dev = d->device->device;
        float * map = resource.map;

        const bool coherent =
            !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // copy the tight (unpadded) rows of every source plane of the temporal
        // window into the staging upload region; the pad kernel mirrors them
        // into the device-local padded buffer
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & cfg = d->planes[plane];
            const size_t row_bytes = static_cast<size_t>(cfg.width) * d->bytes;
            for (int t = 0; t < tw; ++t) {
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
                    .offset = cfg.upload_offset,
                    .size = cfg.upload_bytes,
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
    d->sample_type_float = fmt.sampleType == stFloat;

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
    d->tbsize = tbsize;
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
        cfg.hn = calcPadNum(cfg.width, d->block_step);
        cfg.vn = calcPadNum(cfg.height, d->block_step);
        cfg.num_blocks = cfg.hn * cfg.vn;

        // single-fold reflect_pad requires pad <= dim-1
        const int ox = (cfg.pw - cfg.width) / 2;
        const int oy = (cfg.ph - cfg.height) / 2;
        if (ox > cfg.width - 1 || (cfg.pw - cfg.width - ox) > cfg.width - 1 ||
            oy > cfg.height - 1 || (cfg.ph - cfg.height - oy) > cfg.height - 1) {
            return set_error("a processed plane is too small for the padded block layout.");
        }

        const VkDeviceSize pad_elems = static_cast<VkDeviceSize>(cfg.pw) * cfg.ph;
        const VkDeviceSize nblk = static_cast<VkDeviceSize>(cfg.hn) * cfg.vn;

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
        spatial_sum += nblk * d->tw * 256;   // floats

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
        d->device_id = device_id;
    }

    VkDevice dev = d->device->device;

    // ------------------------------------------------------------------
    // Pipeline layout, descriptor set layout and descriptor pool
    // ------------------------------------------------------------------
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * static_cast<uint32_t>(std::max(d->num_streams, 3))
        };

        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(std::max(d->num_streams, 3)),
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
        const uint32_t * pad_code = nullptr;
        size_t pad_size = 0;
        const uint32_t * col2im_code = nullptr;
        size_t col2im_size = 0;
        const uint32_t * fused_code[4] {};
        size_t fused_size[4] {};
        switch (d->bits) {
            case 16:
                pad_code = dfttest_16_pad_spv;       pad_size = dfttest_16_pad_spv_size;
                col2im_code = dfttest_16_col2im_spv; col2im_size = dfttest_16_col2im_spv_size;
                fused_code[0] = dfttest_16_fused_r0_spv; fused_size[0] = dfttest_16_fused_r0_spv_size;
                fused_code[1] = dfttest_16_fused_r1_spv; fused_size[1] = dfttest_16_fused_r1_spv_size;
                fused_code[2] = dfttest_16_fused_r2_spv; fused_size[2] = dfttest_16_fused_r2_spv_size;
                fused_code[3] = dfttest_16_fused_r3_spv; fused_size[3] = dfttest_16_fused_r3_spv_size;
                break;
            case 32:
                pad_code = dfttest_32_pad_spv;       pad_size = dfttest_32_pad_spv_size;
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
            const auto result = create_shader_module(*d->device, pad_code, pad_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_module = std::get<VkShaderModule>(result);
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
            const auto result = create_pipeline(*d->device, d->pad_module, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_pipeline = std::get<VkPipeline>(result);
        }
        for (int r = 0; r < 4; ++r) {
            const auto result = create_pipeline(*d->device, d->fused_module[r], d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->fused_pipeline[r] = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->device, d->col2im_module, d->pipeline_layout);
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

    d->need_fill = d->bits != 32;
    const int effective_streams = std::max(d->num_streams, 3);
    d->semaphore.current.store(effective_streams - 1, std::memory_order::relaxed);
    d->resources.reserve(effective_streams);

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
