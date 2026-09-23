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
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// _mm_sfence for ordering the weight stores before the first dispatch.
#include <immintrin.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <volk.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// NNEDI3 — intra-field interpolator on the core's Vulkan device (the R80 GPU
// API: vnode:gpu in, ffGPUOutput out, one exec pool).
//
// vsfeel's own implementation of the predictor/prescreener math both GPU
// references compute (nnedi3vk and vszipcu agree bit-exactly, see
// notes/NNEDI3.md). The filter moves no pixels itself: the kernels read the
// source frame plane in place at its own row pitch and write kept rows and
// interpolated rows straight into the output frame's planes, so the whole
// frame is one recorded command buffer whose synchronization travels as the
// producer pairs the exec pool publishes.
//
// Per plane (src/nnedi3.comp):
//   ENTRY_KEEP      kept rows (and the dh zero-fill of skipped planes)
//   ENTRY_PRESCREEN cubic taps + prescreener decision, compacts rejected
//                   pixel indices and maintains the indirect grid size
//   ENTRY_PREDICT   the network over the listed pixels, launched indirectly
// ---------------------------------------------------------------------------

// Window / network tables indexed by the filter arguments.
constexpr int NNEDI3_XDIM[7] { 8, 16, 32, 48, 8, 16, 32 };
constexpr int NNEDI3_YDIM[7] { 6, 6, 6, 6, 4, 4, 4 };
constexpr int NNEDI3_NNS[5] { 16, 32, 64, 128, 256 };

// Weight blob linked into the binary (see CMakeLists.txt): objcopy on every
// toolchain that has it, an RCDATA resource on Windows, where none does.
#if !defined(_WIN32)
extern "C" {
extern const uint8_t _binary_nnedi3_weights_bin_start[];
extern const uint8_t _binary_nnedi3_weights_bin_end[];
}
#endif

static std::span<const uint8_t> weights_blob() {
#if defined(_WIN32)
    static const std::span<const uint8_t> blob = []() -> std::span<const uint8_t> {
        HMODULE module = nullptr;
        // Our own module, not the host executable's: FindResourceW(nullptr, …)
        // searches the process image, which is VapourSynth, not this plugin.
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&weights_blob), &module) == 0) {
            return {};
        }
        // MAKEINTRESOURCEW(10) is RT_RCDATA in its wide form: RT_RCDATA itself
        // follows the UNICODE macro, which must not decide whether this builds.
        const HRSRC res = FindResourceW(module, L"NNEDI3_WEIGHTS",
                                        MAKEINTRESOURCEW(10));
        if (res == nullptr) {
            return {};
        }
        const HGLOBAL handle = LoadResource(module, res);
        if (handle == nullptr) {
            return {};
        }
        const auto * bytes = static_cast<const uint8_t *>(LockResource(handle));
        return { bytes, static_cast<size_t>(SizeofResource(module, res)) };
    }();
    return blob;
#else
    return { _binary_nnedi3_weights_bin_start, _binary_nnedi3_weights_bin_end };
#endif
}

// ---------------------------------------------------------------------------
// Weight blob parsing
//
// Layout of the f32 blob (see notes/NNEDI3.md): old prescreener, three new
// prescreeners (layer-0 stored transposed), then etype x nns x nsize model
// pairs (qual 1 + qual 2). Only the selected model is retained.
// ---------------------------------------------------------------------------

struct PsOldWeights {
    float k0[4][48] {};
    float b0[4] {};
    float k1[4][4] {};
    float b1[4] {};
    float k2[4][8] {};
    float b2[4] {};
};

struct PsNewWeights {
    float k0[4][64] {};
    float b0[4] {};
    float k1[4][4] {};
    float b1[4] {};
};

struct ModelWeights {
    int xdim {}, ydim {}, nns {};
    std::vector<float> sm1, el1, sm_b1, el_b1;
    std::vector<float> sm2, el2, sm_b2, el_b2;
};

struct WeightReader {
    const float * data {};
    size_t count {};
    size_t pos {};

    bool read(float * dst, size_t n) {
        if (pos + n > count) {
            return false;
        }
        std::memcpy(dst, data + pos, n * sizeof(float));
        pos += n;
        return true;
    }

    bool skip(size_t n) {
        if (pos + n > count) {
            return false;
        }
        pos += n;
        return true;
    }
};

static double vec_mean(const float * v, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += v[i];
    }
    return acc / static_cast<double>(n);
}

// Prescreener prep: subtract each layer-0 neuron's own mean and scale by
// 1/pixel_half, so the shader dots raw pixel values directly.
template <size_t W>
static void prescreener_prep(float (&k0)[4][W], double pixel_half) {
    for (int n = 0; n < 4; ++n) {
        const double m = vec_mean(k0[n], W);
        for (size_t k = 0; k < W; ++k) {
            k0[n][k] = static_cast<float>((k0[n][k] - m) / pixel_half);
        }
    }
}

// Model prep: project the per-neuron means and the shared mean filter out of
// the weights (one pass per qual), so the shader normalizes with v*mstd2 and
// re-adds the window mean only in the final blend.
static void model_prep_pass(std::vector<float> & sm, std::vector<float> & el,
                            std::vector<float> & sm_b, size_t fs, size_t nns) {
    std::vector<double> sm_means(nns), el_means(nns), mean_filter(fs, 0.0);
    for (size_t p = 0; p < nns; ++p) {
        sm_means[p] = vec_mean(sm.data() + p * fs, fs);
        el_means[p] = vec_mean(el.data() + p * fs, fs);
        for (size_t k = 0; k < fs; ++k) {
            mean_filter[k] += sm[p * fs + k] - sm_means[p];
        }
    }
    for (size_t k = 0; k < fs; ++k) {
        mean_filter[k] /= static_cast<double>(nns);
    }
    const double bias_mean = vec_mean(sm_b.data(), nns);
    for (size_t p = 0; p < nns; ++p) {
        for (size_t k = 0; k < fs; ++k) {
            sm[p * fs + k] -= static_cast<float>(sm_means[p] + mean_filter[k]);
            el[p * fs + k] -= static_cast<float>(el_means[p]);
        }
        sm_b[p] -= static_cast<float>(bias_mean);
    }
}

static std::optional<std::string> parse_weights(int nsize, int nns_sel, int etype,
                                                int pscrn, double pixel_half,
                                                PsOldWeights & ps_old,
                                                PsNewWeights & ps_new,
                                                ModelWeights & model) {
    const std::span<const uint8_t> blob = weights_blob();
    if (blob.empty()) {
        return "weight blob missing from the plugin binary";
    }
    const auto * data = reinterpret_cast<const float *>(blob.data());
    const size_t count = blob.size() / sizeof(float);
    WeightReader r { data, count, 0 };

    for (int n = 0; n < 4; ++n) {
        if (!r.read(ps_old.k0[n], 48)) return "weight blob truncated (ps_old l0)";
    }
    if (!r.read(ps_old.b0, 4)) return "weight blob truncated (ps_old b0)";
    for (int n = 0; n < 4; ++n) {
        if (!r.read(ps_old.k1[n], 4)) return "weight blob truncated (ps_old l1)";
    }
    if (!r.read(ps_old.b1, 4)) return "weight blob truncated (ps_old b1)";
    for (int n = 0; n < 4; ++n) {
        if (!r.read(ps_old.k2[n], 8)) return "weight blob truncated (ps_old l2)";
    }
    if (!r.read(ps_old.b2, 4)) return "weight blob truncated (ps_old b2)";

    PsNewWeights all_new[3] {};
    for (int i = 0; i < 3; ++i) {
        float l0s[4 * 64] {};
        float l1s[4 * 4] {};
        if (!r.read(l0s, 4 * 64)) return "weight blob truncated (ps_new l0)";
        if (!r.read(all_new[i].b0, 4)) return "weight blob truncated (ps_new b0)";
        if (!r.read(l1s, 4 * 4)) return "weight blob truncated (ps_new l1)";
        if (!r.read(all_new[i].b1, 4)) return "weight blob truncated (ps_new b1)";
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 64; ++k) {
                all_new[i].k0[n][k] = l0s[(k / 8) * 32 + n * 8 + k % 8];
            }
            for (int k = 0; k < 4; ++k) {
                all_new[i].k1[n][k] = l1s[k * 4 + n];
            }
        }
    }

    bool found = false;
    for (int m = 0; m < 2; ++m) {
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 7; ++j) {
                const size_t nns = NNEDI3_NNS[i];
                const size_t fs = static_cast<size_t>(NNEDI3_XDIM[j]) * NNEDI3_YDIM[j];
                if (m == etype && i == nns_sel && j == nsize) {
                    model.xdim = NNEDI3_XDIM[j];
                    model.ydim = NNEDI3_YDIM[j];
                    model.nns = static_cast<int>(nns);
                    model.sm1.resize(nns * fs);
                    model.el1.resize(nns * fs);
                    model.sm_b1.resize(nns);
                    model.el_b1.resize(nns);
                    model.sm2.resize(nns * fs);
                    model.el2.resize(nns * fs);
                    model.sm_b2.resize(nns);
                    model.el_b2.resize(nns);
                    if (!r.read(model.sm1.data(), nns * fs) ||
                        !r.read(model.el1.data(), nns * fs) ||
                        !r.read(model.sm_b1.data(), nns) ||
                        !r.read(model.el_b1.data(), nns) ||
                        !r.read(model.sm2.data(), nns * fs) ||
                        !r.read(model.el2.data(), nns * fs) ||
                        !r.read(model.sm_b2.data(), nns) ||
                        !r.read(model.el_b2.data(), nns)) {
                        return "weight blob truncated (model)";
                    }
                    found = true;
                } else {
                    if (!r.skip(4 * nns * fs + 4 * nns)) {
                        return "weight blob truncated (skip)";
                    }
                }
            }
        }
    }
    if (!found) {
        return "model not found in weight blob";
    }
    if (r.pos != r.count) {
        return "weight blob size mismatch";
    }

    if (pscrn == 1) {
        prescreener_prep(ps_old.k0, pixel_half);
    } else if (pscrn >= 2) {
        ps_new = all_new[pscrn - 2];
        prescreener_prep(ps_new.k0, pixel_half);
    }
    model_prep_pass(model.sm1, model.el1, model.sm_b1,
        model.sm1.size() / static_cast<size_t>(model.nns), model.nns);
    model_prep_pass(model.sm2, model.el2, model.sm_b2,
        model.sm2.size() / static_cast<size_t>(model.nns), model.nns);
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------

struct Nnedi3Plane {
    int width {};
    int rows {};                      // field rows == interpolated output rows
    uint32_t pre_grid_x {};           // prescreen dispatch (128-thread groups)
    uint32_t pred_grid_direct_x {};   // direct predict grid (pscrn == 0)
    uint32_t keep_grid_x {}, keep_grid_y {};
    VkPipeline pre_pipeline {};       // null when pscrn == 0 or the plane is skipped under dh
    VkPipeline pred_pipeline {};
    VkPipeline keep_pipeline {};
    // Regions of the per-frame scratch buffer (the rejected-pixel list and
    // the indirect dispatch struct), one pair per plane.
    VkDeviceSize list_offset {};
    VkDeviceSize ind_offset {};
    int32_t list_elem {};             // uint element offsets into the scratch
    int32_t ind_elem {};
};

// VSFEEL_NNEDI3_TSTAMP=<frame>: one warm frame stamps head / prescreen done /
// predict done into a query pool, whose results the same command buffer copies
// into a mapped buffer. The host waits that submission out once to read them.
struct Nnedi3Probe {
    VkQueryPool query {};
    GpuBuffer buf;
    uint64_t * map {};
    std::atomic<int> armed { 0 };
};

struct Nnedi3Data {
    VSNode * node {};
    const VSVideoInfo * vi {};
    VSVideoInfo vi_out {};

    int field {};
    bool dh {};
    int qual {}, pscrn {};
    bool use_list {};               // prescreen compacts a list (pscrn > 0)
    int peak {}, elem_bytes {};
    int xdim {}, ydim {}, nns {};
    bool process[3] { true, true, true };

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    // Prescreener / predictor weights: host visible so creation fills them
    // with three memcpys; the kernels stream them every dispatch.
    GpuBuffer ps {};
    GpuBuffer pdw {};
    GpuBuffer pdb {};

    VkDeviceSize scratch_bytes { 4 };
    std::array<Nnedi3Plane, 3> planes {};
    VSGPUExecPool * pool {};

    // VSFEEL_NNEDI3_TIMING=1: per-frame host stage split. Under the API the
    // host side is only acquire/record/submit, but the split still says
    // whether the frame is host- or GPU-bound.
    bool host_timing { false };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_record_ns {}, ht_submit_ns {},
        ht_total_ns {}, ht_n {};

    bool gpu_trace { false };
    int gpu_trace_frame { 100 };
    Nnedi3Probe probe;

    ~Nnedi3Data() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[nnedi3-timing] frames=%.0f per-frame us: acquire=%7.1f "
                "record=%7.1f submit=%7.1f total=%7.1f\n",
                n, ht_acquire_ns.load() / 1000.0 / n,
                ht_record_ns.load() / 1000.0 / n,
                ht_submit_ns.load() / 1000.0 / n, ht_total_ns.load() / 1000.0 / n);
        }
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
        VkPipeline seen[6] {};
        int n_seen = 0;
        for (auto & plane : planes) {
            const VkPipeline pipes[3] {
                plane.pre_pipeline, plane.pred_pipeline, plane.keep_pipeline
            };
            for (VkPipeline p : pipes) {
                if (!p) {
                    continue;
                }
                bool dup = false;
                for (int i = 0; i < n_seen; ++i) {
                    dup |= seen[i] == p;
                }
                if (!dup && n_seen < 6) {
                    seen[n_seen++] = p;
                    gpu->vk->vkDestroyPipeline(dev, p, nullptr);
                }
            }
        }
        if (pipeline_layout) {
            gpu->vk->vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            gpu->vk->vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
        gpu_destroy_buffer(*gpu, ps);
        gpu_destroy_buffer(*gpu, pdw);
        gpu_destroy_buffer(*gpu, pdb);
        if (probe.query) {
            gpu->vk->vkDestroyQueryPool(dev, probe.query, nullptr);
        }
        gpu_destroy_buffer(*gpu, probe.buf);
    }
};

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

struct Nnedi3Spec {
    int32_t width, rows, peak, pscrn, xdim, ydim, nns, qual, use_list, dh, zero;
};

static constexpr std::array<VkSpecializationMapEntry, 11> spec_entries = [] {
    std::array<VkSpecializationMapEntry, 11> e {};
    for (uint32_t i = 0; i < 11; ++i) {
        e[i] = { i, i * static_cast<uint32_t>(sizeof(int32_t)), sizeof(int32_t) };
    }
    return e;
}();

// The cooperative kernels (prescreen, predict) keep subgroup-uniform control
// flow and run subgroup intrinsics, so they ask for full 32-lane subgroups;
// the kept-row writer is a plain copy and takes the driver's default width.
static std::variant<VkPipeline, std::string> create_pipeline(
    const GPUDevice & gpu, const Nnedi3Spec & spec, const uint32_t * code,
    size_t code_size, VkPipelineLayout layout,
    uint32_t required_subgroup_size = 0, bool full_subgroups = false) {

    return gpu_create_pipeline(gpu, code, code_size, layout, spec_entries.data(),
        &spec, static_cast<uint32_t>(spec_entries.size()), sizeof(spec), "nnedi3",
        required_subgroup_size, full_subgroups);
}

// The indirect struct arrives through vkCmdFillBuffer, and prescreen's
// atomics feed the indirect launch: both need explicit edges, back-to-back
// commands order nothing.
static void barrier_transfer_to_compute(const GPUDevice & g, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    g.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

static void barrier_prescreen_to_predict(const GPUDevice & g, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    g.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// GPU input, GPU output: the kernels read the source plane and write the
// output frame's planes in place; the core owns every transfer.
static const VSFrame *VS_CC Nnedi3GetFrame(
    int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    Nnedi3Data * d = static_cast<Nnedi3Data *>(instanceData);

    const int sn = d->field > 1 ? n / 2 : n;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const int numPlanes = d->vi->format.numPlanes;
    const VSFrame * src = vsapi->getFrameFilter(sn, d->node, frameCtx);

    // Output frame: every plane is fresh when all are processed or dh doubles
    // the height (unprocessed planes then need the keep kernel's expand too);
    // otherwise the unprocessed planes share the source's, producer pairs and
    // all. newVideoFrame2 infers residency from the shared planes.
    bool all_process = true;
    for (int p = 0; p < numPlanes; ++p) {
        all_process &= d->process[p];
    }
    const int pl[] = { 0, 1, 2 };
    const VSFrame * fr[] = {
        (!d->dh && !d->process[0]) ? src : nullptr,
        (!d->dh && !d->process[1]) ? src : nullptr,
        (!d->dh && !d->process[2]) ? src : nullptr
    };
    VSFrame * dst = (all_process || d->dh)
        ? d->gpu->api->newGPUVideoFrame(&d->vi_out.format, d->vi_out.width,
              d->vi_out.height, src, core)
        : vsapi->newVideoFrame2(&d->vi_out.format, d->vi_out.width, d->vi_out.height,
              fr, pl, src, core);
    if (!dst) {
        vsfeel_trace_error("NNEDI3", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("NNEDI3: failed to allocate the output frame", frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    }

    // Source field parity, mirroring the references: parity == 1 keeps the
    // bottom field. Double-rate flips parity on odd outputs; _Field /
    // _FieldBased decide it under dh / field > 1.
    const int default_parity = (d->field == 0 || d->field == 2) ? 1 : 0;
    int parity;
    {
        int err;
        const VSMap * props = vsapi->getFramePropertiesRO(src);
        if (d->dh) {
            parity = static_cast<int>(vsapi->mapGetIntSaturated(props, "_Field", 0, &err));
            if (err) {
                parity = default_parity;
            }
        } else if (d->field > 1) {
            const int field_based = static_cast<int>(
                vsapi->mapGetIntSaturated(props, "_FieldBased", 0, &err));
            if (field_based == VSC_FIELD_BOTTOM) {
                parity = 1;
            } else if (field_based == VSC_FIELD_TOP) {
                parity = 0;
            } else {
                parity = default_parity;
            }
            if (n & 1) {
                parity = !parity;
            }
        } else {
            parity = d->field == 0 ? 1 : 0;
        }
        parity = !!parity;
    }

    vsfeel_trace_frame_begin();
    vsfeel_trace_mark("acquire");
    char errbuf[512] {};
    auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};
    VSGPUExecContext * ctx = d->gpu->api->gpuExecAcquire(d->pool, errbuf, sizeof(errbuf));
    auto fail = [&](const std::string & message) -> const VSFrame * {
        if (ctx) {
            d->gpu->api->gpuExecAbandon(ctx);
            ctx = nullptr;
        }
        vsfeel_trace_error("NNEDI3", n, message, d->gpu.get());
        vsapi->setFilterError(("NNEDI3: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        vsapi->freeFrame(src);
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    // The rejected-pixel list and the indirect struct: one transient buffer
    // per frame, retired by the submission (the pool's size buckets make the
    // per-frame allocation cheap). pscrn=0 needs neither.
    GpuBuffer scratch {};
    if (d->use_list) {
        if (const auto e = gpu_frame_buffer(*d->gpu, core, ctx, d->scratch_bytes,
                scratch,
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            !e.empty()) {
            return fail("scratch buffer: " + e);
        }
    }

    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);

    const bool gputrace = d->gpu_trace &&
        n == d->gpu_trace_frame && d->probe.armed.exchange(1) == 0;
    int probe_plane = -1;
    for (int p = 0; p < numPlanes && probe_plane < 0; ++p) {
        if (d->process[p]) {
            probe_plane = p;
        }
    }
    if (gputrace) {
        d->gpu->vk->vkCmdResetQueryPool(cmd, d->probe.query, 0, 3);
        d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            d->probe.query, 0);
    }

    // Plane dispatches touch disjoint buffers (and disjoint scratch regions),
    // so no barrier separates them; only the two stages of one list-mode plane
    // are ordered.
    for (int p = 0; p < numPlanes; ++p) {
        const bool keep = d->process[p] || d->dh;
        if (!keep) {
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
        const int src_stride = static_cast<int>(vsapi->getStride(src, p) / d->elem_bytes);
        const int dst_stride = static_cast<int>(vsapi->getStride(dst, p) / d->elem_bytes);

        // All seven bindings re-pushed per dispatch; the scratch slots stand
        // in where this entry statically reads nothing (see src/nnedi3.comp).
        const VkBuffer buffers[7] {
            sp.buffer, dp.buffer, d->ps.buffer, d->pdw.buffer, d->pdb.buffer,
            d->use_list ? scratch.buffer : sp.buffer,
            d->use_list ? scratch.buffer : sp.buffer
        };
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 7);

        // Kept rows (and the dh zero-fill of a skipped plane) straight into
        // the output plane; they overlap nothing the network writes.
        {
            const int32_t push[5] { 0, src_stride, dst_stride, parity, 0 };
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push, sizeof(push));
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.keep_pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.keep_grid_x, cfg.keep_grid_y, 1);
        }
        if (!d->process[p]) {
            continue;
        }

        const int32_t push[5] { cfg.list_elem, src_stride, dst_stride, parity,
                                cfg.ind_elem };
        if (d->use_list) {
            // Reset this plane's indirect struct to {groupsX, groupsY,
            // groupsZ, count} = {0, 1, 1, 0}: prescreen bumps the count and
            // the groupsX with atomics every frame.
            const VkDeviceSize base = cfg.ind_offset;
            d->gpu->vk->vkCmdFillBuffer(cmd, scratch.buffer, base + 0, 4, 0);
            d->gpu->vk->vkCmdFillBuffer(cmd, scratch.buffer, base + 4, 8, 1);
            d->gpu->vk->vkCmdFillBuffer(cmd, scratch.buffer, base + 12, 4, 0);
            barrier_transfer_to_compute(*d->gpu, cmd);

            // Prescreen: cubic stores into the output plane plus rejected-pixel
            // compaction into the list.
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push, sizeof(push));
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.pre_pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.pre_grid_x, 1, 1);
        }
        if (gputrace && p == probe_plane) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                d->probe.query, 1);
        }
        if (d->use_list) {
            // The predictor launches off the count prescreen accumulated.
            // (An over-launch probe measured 2026-09-05: a full-grid direct
            // launch was SLOWER -- exiting subgroups still pay the window
            // gather -- so the exact indirect grid stays.)
            barrier_prescreen_to_predict(*d->gpu, cmd);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push, sizeof(push));
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.pred_pipeline);
            d->gpu->vk->vkCmdDispatchIndirect(cmd, scratch.buffer, cfg.ind_offset);
        } else {
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, push, sizeof(push));
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          cfg.pred_pipeline);
            d->gpu->vk->vkCmdDispatch(cmd, cfg.pred_grid_direct_x, 1, 1);
        }
        if (gputrace && p == probe_plane) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->probe.query, 2);
        }
    }

    if (gputrace) {
        d->gpu->vk->vkCmdCopyQueryPoolResults(cmd, d->probe.query, 0, 3,
            d->probe.buf.buffer, 0, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }
    auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    // The pool turns the source plane's producer pair into a device-side wait
    // and publishes the output planes' producers; frames stay alive until the
    // submission completes.
    d->gpu->api->gpuExecReadsFrame(ctx, src);
    for (int p = 0; p < numPlanes; ++p) {
        if (d->process[p] || d->dh) {
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

    if (gputrace) {
        // One-shot probe: wait this submission out so the query results are
        // final, then read the mapped copy the command buffer made.
        char perr[256] {};
        if (d->gpu->api->gpuExecWaitValue(d->pool, signaled, perr, sizeof(perr)) ==
            gdDrained) {
            const double period = d->gpu->limits.timestampPeriod;
            const auto us = [period](uint64_t a, uint64_t b) {
                return static_cast<double>(b - a) * period / 1000.0;
            };
            const uint64_t * ts = d->probe.map;
            fprintf(stderr, "[nnedi3-gpu] n=%d pre=%.1fus pred=%.1fus total=%.1fus\n",
                n, us(ts[0], ts[1]), us(ts[1], ts[2]), us(ts[0], ts[2]));
        } else {
            fprintf(stderr, "[nnedi3-gpu] probe wait failed: %s\n", perr);
        }
    }

    vsapi->freeFrame(src);

    VSMap * props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
    vsapi->mapDeleteKey(props, "_Field");
    if (d->field > 1) {
        int err_num, err_den;
        int64_t dur_num = vsapi->mapGetInt(props, "_DurationNum", 0, &err_num);
        int64_t dur_den = vsapi->mapGetInt(props, "_DurationDen", 0, &err_den);
        if (!err_num && !err_den) {
            vsh::muldivRational(&dur_num, &dur_den, 1, 2);
            vsapi->mapSetInt(props, "_DurationNum", dur_num, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", dur_den, maReplace);
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC Nnedi3Free(
    void *instanceData, [[maybe_unused]] VSCore *core, const VSAPI *vsapi) {

    Nnedi3Data * d = static_cast<Nnedi3Data *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC Nnedi3Create(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<Nnedi3Data>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    d->vi_out = *d->vi;

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("NNEDI3", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("NNEDI3: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    const auto & fmt = d->vi->format;
    const int bits = fmt.bitsPerSample;
    const bool depth_ok = (fmt.sampleType == stInteger && bits == 16) ||
                          (fmt.sampleType == stFloat && bits == 32);
    if (!depth_ok || d->vi->width <= 0 || d->vi->height <= 0 ||
        (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV && fmt.colorFamily != cfRGB)) {
        return set_error("only 16-bit integer and 32-bit float Gray/YUV/RGB input supported.");
    }

    d->peak = fmt.sampleType == stInteger ? (1 << bits) - 1 : 0;
    d->elem_bytes = bits / 8;

    d->field = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "field", 0, nullptr));
    if (d->field < 0 || d->field > 3) {
        return set_error("field must be 0, 1, 2, or 3.");
    }
    d->dh = !!vsapi->mapGetInt(in, "dh", 0, &error);
    if (d->dh && d->field > 1) {
        return set_error("field must be 0 or 1 when dh is true.");
    }

    for (int i = 0; i < 3; ++i) {
        d->process[i] = true;
    }
    const int num_plane_args = vsapi->mapNumElements(in, "planes");
    if (num_plane_args > 0) {
        for (int i = 0; i < 3; ++i) {
            d->process[i] = false;
        }
        for (int i = 0; i < num_plane_args; ++i) {
            const int p = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "planes", i, nullptr));
            if (p < 0 || p >= fmt.numPlanes) {
                return set_error("plane index out of range.");
            }
            if (d->process[p]) {
                return set_error("plane specified twice.");
            }
            d->process[p] = true;
        }
    }
    bool any_process = false;
    for (int p = 0; p < fmt.numPlanes; ++p) {
        any_process |= d->process[p];
    }
    if (!any_process) {
        return set_error("no planes to process.");
    }

    const int nsize = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "nsize", 0, &error));
    const int nsize_v = error ? 6 : nsize;
    const int nns_sel = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "nns", 0, &error));
    const int nns_v = error ? 1 : nns_sel;
    d->qual = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "qual", 0, &error));
    if (error) {
        d->qual = 1;
    }
    const int etype = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "etype", 0, &error));
    const int etype_v = error ? 0 : etype;
    d->pscrn = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "pscrn", 0, &error));
    if (error) {
        d->pscrn = 2;
    }
    if (nsize_v < 0 || nsize_v > 6) {
        return set_error("nsize must be between 0 and 6 (inclusive).");
    }
    if (nns_v < 0 || nns_v > 4) {
        return set_error("nns must be between 0 and 4 (inclusive).");
    }
    if (d->qual < 1 || d->qual > 2) {
        return set_error("qual must be 1 or 2.");
    }
    if (etype_v < 0 || etype_v > 1) {
        return set_error("etype must be 0 or 1.");
    }
    if (d->pscrn < 0 || d->pscrn > 4) {
        return set_error("pscrn must be between 0 and 4 (inclusive).");
    }
    d->use_list = d->pscrn > 0;

    if (!d->dh) {
        for (int plane = 0; plane < fmt.numPlanes; ++plane) {
            const int ph = d->vi->height >> (plane > 0 ? fmt.subSamplingH : 0);
            if (d->process[plane] && (ph & 1) != 0) {
                return set_error("plane height must be mod 2 when dh is false.");
            }
        }
    }

    // device_id and num_streams are registered but never read: the core owns
    // the one device and sizes in-flight depth itself (exec pool ring).

    if (d->field > 1) {
        if (d->vi_out.numFrames > INT32_MAX / 2) {
            return set_error("resulting clip is too long.");
        }
        // numFrames == -1 is the unknown-length sentinel, not a length.
        if (d->vi_out.numFrames > 0) {
            d->vi_out.numFrames *= 2;
        }
        vsh::muldivRational(&d->vi_out.fpsNum, &d->vi_out.fpsDen, 2, 1);
    }
    if (d->dh) {
        d->vi_out.height *= 2;
    }

    d->host_timing = vsfeel_debug_probe("VSFEEL_NNEDI3_TIMING");

    {
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

    // Weights: parse the embedded blob, prep, and pack the GPU layouts.
    PsOldWeights ps_old {};
    PsNewWeights ps_new {};
    ModelWeights model {};
    {
        const double pixel_half = fmt.sampleType == stFloat
            ? 0.5 : static_cast<double>(d->peak) / 2.0;
        if (const auto err = parse_weights(
                nsize_v, nns_v, etype_v, d->pscrn, pixel_half, ps_old, ps_new, model)) {
            return set_error(*err);
        }
    }
    d->xdim = model.xdim;
    d->ydim = model.ydim;
    d->nns = model.nns;
    const size_t fs = static_cast<size_t>(d->xdim) * d->ydim;
    const size_t nns = static_cast<size_t>(d->nns);
    const size_t num_q = static_cast<size_t>(d->qual);

    // prescreener vec4 blob: transposed layer-0 + inline deeper layers
    std::vector<float> ps_blob;
    if (d->pscrn == 1) {
        ps_blob.resize(63 * 4, 0.0f);
        for (int k = 0; k < 48; ++k) {
            for (int n = 0; n < 4; ++n) {
                ps_blob[k * 4 + n] = ps_old.k0[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[48 * 4 + n] = ps_old.b0[n];
        }
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 4; ++k) {
                ps_blob[(49 + n) * 4 + k] = ps_old.k1[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[53 * 4 + n] = ps_old.b1[n];
        }
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 4; ++k) {
                ps_blob[(54 + n * 2) * 4 + k] = ps_old.k2[n][k];
                ps_blob[(55 + n * 2) * 4 + k] = ps_old.k2[n][4 + k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[62 * 4 + n] = ps_old.b2[n];
        }
    } else if (d->pscrn >= 2) {
        ps_blob.resize(70 * 4, 0.0f);
        for (int k = 0; k < 64; ++k) {
            for (int n = 0; n < 4; ++n) {
                ps_blob[k * 4 + n] = ps_new.k0[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[64 * 4 + n] = ps_new.b0[n];
        }
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 4; ++k) {
                ps_blob[(65 + n) * 4 + k] = ps_new.k1[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[69 * 4 + n] = ps_new.b1[n];
        }
    } else {
        ps_blob.resize(4, 0.0f);
    }

    // predictor weights: (softmax, elliott) pairs in [q][k][p] order
    std::vector<float> pdw_blob(num_q * fs * nns * 2);
    for (size_t q = 0; q < num_q; ++q) {
        const std::vector<float> & sm = q == 0 ? model.sm1 : model.sm2;
        const std::vector<float> & el = q == 0 ? model.el1 : model.el2;
        for (size_t k = 0; k < fs; ++k) {
            for (size_t p = 0; p < nns; ++p) {
                pdw_blob[((q * fs + k) * nns + p) * 2 + 0] = sm[p * fs + k];
                pdw_blob[((q * fs + k) * nns + p) * 2 + 1] = el[p * fs + k];
            }
        }
    }
    // predictor biases: (smB, elB) pairs in [q][p] order
    std::vector<float> pdb_blob(num_q * nns * 2);
    for (size_t q = 0; q < num_q; ++q) {
        const std::vector<float> & sm_b = q == 0 ? model.sm_b1 : model.sm_b2;
        const std::vector<float> & el_b = q == 0 ? model.el_b1 : model.el_b2;
        for (size_t p = 0; p < nns; ++p) {
            pdb_blob[(q * nns + p) * 2 + 0] = sm_b[p];
            pdb_blob[(q * nns + p) * 2 + 1] = el_b[p];
        }
    }

    // Push descriptor layout: one whole-buffer binding per shader buffer, so
    // each dispatch rebinds its own view of the planes and nothing is
    // allocated from a descriptor pool.
    {
        const auto result = gpu_push_set_layout(*d->gpu, 7);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result = gpu_pipeline_layout(*d->gpu, d->set_layout,
            5 * sizeof(int32_t));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // Weight buffers: host visible (creation fills them with three memcpys),
    // device-local preferred so on a ReBAR window they land in VRAM -- the
    // predictor streams the whole matrix per subgroup and thrashes L2.
    auto make_weights = [&](GpuBuffer & buf, const std::vector<float> & values,
                            const char * what) -> std::string {
        std::string e = gpu_make_buffer(*d->gpu, core,
            std::max<VkDeviceSize>(values.size() * sizeof(float), 4), buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!e.empty()) {
            return std::string(what) + " buffer: " + e;
        }
        if (buf.mapped == nullptr) {
            return std::string(what) + " buffer is not host visible";
        }
        std::memcpy(buf.mapped, values.data(), values.size() * sizeof(float));
        // The mapping may be the host-visible VRAM BAR (write-combining): the
        // first dispatch must not read a tail the CPU store buffer missed.
        _mm_sfence();
        return {};
    };
    if (std::string e = make_weights(d->ps, ps_blob, "prescreener weights"); !e.empty()) {
        return set_error(e);
    }
    if (std::string e = make_weights(d->pdw, pdw_blob, "predictor weights"); !e.empty()) {
        return set_error(e);
    }
    if (std::string e = make_weights(d->pdb, pdb_blob, "predictor biases"); !e.empty()) {
        return set_error(e);
    }

    // Shader blobs for this io depth (the entry point by role; dh / the zero
    // fill are specialization constants).
    const uint32_t * pre_code = nullptr;
    size_t pre_size = 0;
    const uint32_t * pred_code = nullptr;
    size_t pred_size = 0;
    const uint32_t * pred_n4_code = nullptr;
    size_t pred_n4_size = 0;
    const uint32_t * pred_n4s_code = nullptr;
    size_t pred_n4s_size = 0;
    const uint32_t * keep_code = nullptr;
    size_t keep_size = 0;
    if (d->elem_bytes == 2) {
        pre_code = nnedi3_16_prescreen_spv; pre_size = nnedi3_16_prescreen_spv_size;
        pred_code = nnedi3_16_predict_spv; pred_size = nnedi3_16_predict_spv_size;
        pred_n4_code = nnedi3_16_predict_n4_spv; pred_n4_size = nnedi3_16_predict_n4_spv_size;
        pred_n4s_code = nnedi3_16_predict_n4s_spv; pred_n4s_size = nnedi3_16_predict_n4s_spv_size;
        keep_code = nnedi3_16_keep_spv; keep_size = nnedi3_16_keep_spv_size;
    } else {
        pre_code = nnedi3_32_prescreen_spv; pre_size = nnedi3_32_prescreen_spv_size;
        pred_code = nnedi3_32_predict_spv; pred_size = nnedi3_32_predict_spv_size;
        pred_n4_code = nnedi3_32_predict_n4_spv; pred_n4_size = nnedi3_32_predict_n4_spv_size;
        pred_n4s_code = nnedi3_32_predict_n4s_spv; pred_n4s_size = nnedi3_32_predict_n4s_spv_size;
        keep_code = nnedi3_32_keep_spv; keep_size = nnedi3_32_keep_spv_size;
    }

    // The cooperative predictor needs one 32-lane subgroup per 4 pixels.
    if (!d->gpu->has_subgroup_size(32)) {
        return set_error("device cannot run 32-lane subgroups "
                         "(required by the predictor kernel).");
    }

    // Per-plane geometry, grids and pipelines, deduplicated across identical
    // planes. The predict module is chosen per key: narrow networks (PPL<=2
    // and FS<=128) use the PXP=8 module, wide networks the PXP=4 one (the
    // shader's own PXP rule), FS<=64 the small-tile PXP=4 module.
    const uint32_t max_grid_x = d->gpu->limits.maxComputeWorkGroupCount[0];
    const auto use_pxp8 = [](int net_nns, int net_fs) {
        return ((net_nns + 31) / 32 <= 2) && (net_fs <= 128);
    };

    struct Key { int w, rows, pscrn, xdim, ydim, nns, qual, zero; };
    std::array<Key, 3> keys {};
    std::array<VkPipeline, 3> pre_pipes {};
    std::array<VkPipeline, 3> pred_pipes {};
    std::array<VkPipeline, 3> keep_pipes {};
    int n_keys = 0;
    VkDeviceSize list_run = 0;

    for (int plane = 0; plane < fmt.numPlanes; ++plane) {
        // Geometry follows the INPUT clip (d->vi), not the output: rows is
        // the field height (input rows in dh mode, half otherwise). Under dh
        // a skipped plane still needs the kept-row writer (expand + zero).
        const bool zero = d->dh && !d->process[plane];
        if (!d->process[plane] && !d->dh) {
            continue;
        }
        auto & cfg = d->planes[plane];
        const int in_w = plane == 0 ? d->vi->width : d->vi->width >> fmt.subSamplingW;
        const int in_h = plane == 0 ? d->vi->height : d->vi->height >> fmt.subSamplingH;
        cfg.width = in_w;
        cfg.rows = d->dh ? in_h : in_h / 2;
        // 2* in the bound: the dh zero-fill indexes the whole doubled output.
        if (static_cast<int64_t>(2) * cfg.width * cfg.rows >= (int64_t(1) << 31) ||
            cfg.rows < 1 || cfg.width < 1) {
            return set_error("plane geometry out of range.");
        }

        if (d->process[plane]) {
            // prescreen: one thread per pixel group (P=1 old, P=4 new), one
            // group of 128 threads. The shader groups pixels per ROW as
            // ceil(width/P), so the grid must cover rows*ceil(width/P)
            // threads -- never ceil(width*rows/P), which leaves the tail of
            // every non-divisible row unwritten.
            const uint32_t pps = d->pscrn == 1 ? 1 : 4;
            const uint32_t groups_per_row =
                (static_cast<uint32_t>(cfg.width) + pps - 1) / pps;
            cfg.pre_grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.rows) * groups_per_row + 127) / 128,
                max_grid_x);
            // cooperative direct grid (pscrn==0): 4 subgroups x PXP pixels
            // per 128-thread workgroup.
            const int host_pxp = use_pxp8(d->nns, d->xdim * d->ydim) ? 8 : 4;
            const uint32_t ppg = static_cast<uint32_t>(4 * host_pxp);
            cfg.pred_grid_direct_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.width) * static_cast<uint32_t>(cfg.rows) + ppg - 1) / ppg,
                max_grid_x);

            // Scratch regions: the list holds one uint per field pixel, then
            // this plane's 16-byte indirect struct.
            cfg.list_offset = align32(list_run);
            list_run = align32(cfg.list_offset +
                static_cast<VkDeviceSize>(cfg.width) * cfg.rows * sizeof(uint32_t));
            cfg.ind_offset = align32(list_run);
            list_run = align32(cfg.ind_offset + 16);
            cfg.list_elem = static_cast<int32_t>(cfg.list_offset / 4);
            cfg.ind_elem = static_cast<int32_t>(cfg.ind_offset / 4);
        }

        // The kept-row writer's grid is linearized over x and y, so a 4K
        // double-rate plane stays under the 65535-group x limit.
        {
            const int64_t total = static_cast<int64_t>(zero ? 2 : 1) *
                cfg.rows * cfg.width;
            const int64_t gx = std::clamp<int64_t>((total + 255) / 256, 1, max_grid_x);
            cfg.keep_grid_x = static_cast<uint32_t>(gx);
            cfg.keep_grid_y = static_cast<uint32_t>(
                std::clamp<int64_t>((total + 256 * gx - 1) / (256 * gx), 1, max_grid_x));
        }

        int ki = 0;
        for (; ki < n_keys; ++ki) {
            if (keys[ki].w == cfg.width && keys[ki].rows == cfg.rows &&
                keys[ki].pscrn == d->pscrn && keys[ki].xdim == d->xdim &&
                keys[ki].ydim == d->ydim && keys[ki].nns == d->nns &&
                keys[ki].qual == d->qual && keys[ki].zero == zero) {
                break;
            }
        }
        if (ki == n_keys) {
            const Nnedi3Spec spec {
                cfg.width, cfg.rows, d->peak, d->pscrn, d->xdim, d->ydim, d->nns,
                d->qual, d->use_list ? 1 : 0, d->dh ? 1 : 0, zero ? 1 : 0
            };
            if (d->process[plane]) {
                if (d->use_list) {
                    const auto result = create_pipeline(*d->gpu, spec, pre_code,
                        pre_size, d->pipeline_layout, 32, /*full_subgroups=*/true);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    pre_pipes[n_keys] = std::get<VkPipeline>(result);
                }
                {
                    const int net_fs = d->xdim * d->ydim;
                    const uint32_t * mod = pred_code;
                    size_t mod_size = pred_size;
                    if (!use_pxp8(d->nns, net_fs)) {
                        mod = net_fs <= 64 ? pred_n4s_code : pred_n4_code;
                        mod_size = net_fs <= 64 ? pred_n4s_size : pred_n4_size;
                    }
                    const auto result = create_pipeline(*d->gpu, spec, mod, mod_size,
                        d->pipeline_layout, 32, /*full_subgroups=*/true);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    pred_pipes[n_keys] = std::get<VkPipeline>(result);
                }
            }
            {
                const auto result = create_pipeline(*d->gpu, spec, keep_code,
                    keep_size, d->pipeline_layout);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                keep_pipes[n_keys] = std::get<VkPipeline>(result);
            }
            keys[n_keys] = { cfg.width, cfg.rows, d->pscrn, d->xdim, d->ydim,
                             d->nns, d->qual, zero };
            ++n_keys;
        }
        cfg.pre_pipeline = pre_pipes[ki];
        cfg.pred_pipeline = pred_pipes[ki];
        cfg.keep_pipeline = keep_pipes[ki];
    }
    d->scratch_bytes = d->use_list ? std::max<VkDeviceSize>(list_run, 16) : 4;

    // GPU-timing probe: only when the queue family can timestamp at all, since
    // a timestamp write where timestampValidBits is 0 can hang the engine.
    d->gpu_trace_frame = env_int("VSFEEL_NNEDI3_TSTAMP", 100);
    d->gpu_trace = vsfeel_debug_probe("VSFEEL_NNEDI3_TSTAMP") &&
        vsfeel_probe_timestamps(*d->gpu, "NNEDI3");
    if (d->gpu_trace) {
        VkQueryPoolCreateInfo qp_info {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 3
        };
        if (d->gpu->vk->vkCreateQueryPool(d->gpu->device, &qp_info, nullptr,
                &d->probe.query) != VK_SUCCESS) {
            d->probe.query = VK_NULL_HANDLE;
            d->gpu_trace = false;
        }
    }
    if (d->gpu_trace) {
        auto e = gpu_make_buffer(*d->gpu, core, 3 * sizeof(uint64_t), d->probe.buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!e.empty() || d->probe.buf.mapped == nullptr) {
            d->gpu_trace = false;
        } else {
            d->probe.map = static_cast<uint64_t *>(d->probe.buf.mapped);
        }
    }

    {
        char err[512] {};
        d->pool = d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    VSFilterDependency deps[1] = {{ d->node, d->field > 1 ? rpGeneral : rpStrictSpatial }};

    Nnedi3Data * data = d.release();

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "NNEDI3", &data->vi_out,
        Nnedi3GetFrame, Nnedi3Free,
        fmParallel, ffGPUOutput, deps, 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "NNEDI3: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_nnedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    // Under the R80 GPU API every input and the output are GPU resident: the
    // core inserts the upload for a CPU clip and a GPUDownload for a CPU
    // consumer, so the filter itself never moves a frame.
    vspapi->registerFunction(
        "NNEDI3",
        "clip:vnode:gpu;"
        "field:int;"
        "dh:int:opt;"
        "planes:int[]:opt;"
        "nsize:int:opt;"
        "nns:int:opt;"
        "qual:int:opt;"
        "etype:int:opt;"
        "pscrn:int:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode:gpu;",
        Nnedi3Create, nullptr, plugin
    );
}
