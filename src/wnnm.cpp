#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <bit>
#include <chrono>
#include <cfloat>
#include <climits>
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

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

namespace {

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------
//
// WNNM needs f32 f32 input, like the reference (its sigma is in 0..1 units and
// the whole pipeline is float). The R80 API gives the kernel the source planes
// as device addresses: every temporal layer is a separate VkBuffer, and the
// temporal mode reaches 4*radius+1 of them, more than the 32 push descriptors
// a dispatch may bind, so the layers travel as a host-written address table.
//
// One workgroup owns one reference block of the step lattice (src/wnnm.comp).
// The per-frame scratch is the (num, den) accumulator, the group patch matrix,
// and (temporal) the per-group search-location list.

constexpr int kMaxRadius = 15;
constexpr int kMaxList = 256; // group_size cap
// ps_num sizes the fused matcher's list arrays (KMAXL in src/wnnm.comp), so it
// stops where they do; the reference allows 256.
constexpr int kMaxPsNum = 64;
constexpr int kWorkgroup = 128;
// Threads in the split-matching group kernel: one reference block needs a
// warp for its Gram rows and its round-robin Jacobi, so the other 96 idle.
constexpr int kGroupThreads = 32;
// Floats of per-group scratch in the split estimate: the Gram matrix (64), the
// per-pixel mean for residual (64) and the aggregation weight (1). Must match
// MAT_MEAN/MAT_WEIGHT in src/wnnm.comp.
constexpr int kSplitStride = 136;

static int round_up_pow2(int v) {
    int p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

struct WnnmSpec {
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t bs;
    int32_t step;
    int32_t gs;
    int32_t bm_range;
    int32_t radius;
    int32_t ps_num;
    int32_t ps_range;
    int32_t gx;
    int32_t ng;
    int32_t residual;
    int32_t adaptive;
    int32_t locs_cap;
    int32_t addr_base;
    float sigma;
    int32_t acc_base;
    int32_t pmat_base;
    int32_t trace;
    int32_t tr_base;
    int32_t pmat_stride;
    int32_t locs_base;
    int32_t dist_base;
    int32_t match_base;
    int32_t ring;
    int32_t cache;
};

static constexpr std::array<VkSpecializationMapEntry, 27> spec_entries { {
    { 0, offsetof(WnnmSpec, width), sizeof(int32_t) },
    { 1, offsetof(WnnmSpec, height), sizeof(int32_t) },
    { 2, offsetof(WnnmSpec, stride), sizeof(int32_t) },
    { 3, offsetof(WnnmSpec, bs), sizeof(int32_t) },
    { 4, offsetof(WnnmSpec, step), sizeof(int32_t) },
    { 5, offsetof(WnnmSpec, gs), sizeof(int32_t) },
    { 6, offsetof(WnnmSpec, bm_range), sizeof(int32_t) },
    { 7, offsetof(WnnmSpec, radius), sizeof(int32_t) },
    { 8, offsetof(WnnmSpec, ps_num), sizeof(int32_t) },
    { 9, offsetof(WnnmSpec, ps_range), sizeof(int32_t) },
    { 10, offsetof(WnnmSpec, gx), sizeof(int32_t) },
    { 11, offsetof(WnnmSpec, ng), sizeof(int32_t) },
    { 12, offsetof(WnnmSpec, residual), sizeof(int32_t) },
    { 13, offsetof(WnnmSpec, adaptive), sizeof(int32_t) },
    { 14, offsetof(WnnmSpec, locs_cap), sizeof(int32_t) },
    { 15, offsetof(WnnmSpec, addr_base), sizeof(int32_t) },
    { 16, offsetof(WnnmSpec, sigma), sizeof(float) },
    { 17, offsetof(WnnmSpec, acc_base), sizeof(int32_t) },
    { 18, offsetof(WnnmSpec, pmat_base), sizeof(int32_t) },
    { 19, offsetof(WnnmSpec, trace), sizeof(int32_t) },
    { 20, offsetof(WnnmSpec, tr_base), sizeof(int32_t) },
    { 21, offsetof(WnnmSpec, pmat_stride), sizeof(int32_t) },
    { 22, offsetof(WnnmSpec, locs_base), sizeof(int32_t) },
    { 23, offsetof(WnnmSpec, dist_base), sizeof(int32_t) },
    { 24, offsetof(WnnmSpec, match_base), sizeof(int32_t) },
    { 25, offsetof(WnnmSpec, ring), sizeof(int32_t) },
    { 26, offsetof(WnnmSpec, cache), sizeof(int32_t) },
} };

struct WnnmPlane {
    bool process {};
    int width {};
    int height {};
    int stride {};
    int64_t npix {};
    int gx {};
    int gy {};
    int ng {};
    float sigma {}; // split by 255, as the reference's sigma / 255.0f

    VkPipeline group_pipe {};
    VkPipeline gram_pipe {};
    VkPipeline solve_pipe {};
    VkPipeline apply_pipe {};
    VkPipeline finish_pipe {};
    VkPipeline dist_pipe {};
    VkPipeline select_pipe {};

    int64_t acc_base {};
    int64_t pmat_base {};
    int64_t locs_base {};
    int64_t trace_base {};
    int64_t dist_base {};
    int64_t match_base {};
    int64_t acc_elems {};
    int64_t pmat_elems {};
    int64_t locs_elems {};
    int64_t dist_elems {};
    int64_t match_elems {};

    uint32_t grid_x {};
    uint32_t grid_y {};
    uint32_t finish_x {};
};

// VSFEEL_WNNM_GPUTRACE=<frame>: one-shot per-kernel GPU timings on that frame.
// Only created when the compute queue family can timestamp at all (writing one
// where timestampValidBits is 0 can hang the engine).
struct WnnmGpuProbe {
    VkQueryPool query {};
    GpuBuffer buf;
    uint64_t * map {};
    std::atomic<int> armed { 0 };
};

// The probe slots: one per dispatch, per plane, plus a start and an end.
constexpr int kProbeSlots = 1 + 3 * 6 + 1;

struct GPUWNNM {
    VSNode * node {};
    VSNode * ref_node {};
    const VSVideoInfo * vi {};

    int bs { 8 }, step { 8 }, gs { 8 }, bm_range { 7 }, radius { 0 };
    int ps_num { 2 }, ps_range { 4 }, q {};
    int planes {};
    std::array<WnnmPlane, 3> plane;

    int patch_elems {}; // bs*bs
    int64_t acc_bytes {}, pmat_bytes {}, locs_bytes {};
    int64_t dist_bytes {}, match_bytes {};
    int folds { 1 };
    // Per-centre match cache: at radius > 0 one centre's search feeds 2r+1
    // output frames, so its selection and record are computed once and held in
    // a ring of RING slots keyed by centre frame. Off (one set of per-frame
    // slots per fold) when the ring would not fit the scratch budget.
    int ring { 1 };
    GpuBuffer ring_buf;
    int64_t ring_bytes {};
    std::mutex cache_lock;
    std::vector<int> ring_tag; // centre frame each slot holds, or INT_MIN
    int dist_stride_max {};

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VSGPUExecPool * pool {};

    std::atomic<uint64_t> ht_acquire_ns {}, ht_record_ns {}, ht_submit_ns {},
        ht_n {};
    // VSFEEL_WNNM_DUMP=<frame>: dump {n, kept, weight, centre} per group.
    int dump_frame { -1 };
    int64_t trace_elems {};
    int gpu_trace_frame { 100 };
    WnnmGpuProbe probe;

    // Every flag is packed here, after the pointers and atomics, so the struct
    // does not pay a padding run before each one.
    bool residual {}, adaptive {};
    // Block matching split into distance + selection kernels (radius 0 and a
    // group that fits the selection kernel's register top-K).
    bool split_match {};
    // Per-centre match cache enabled (the ring fit the scratch budget).
    bool cache {};
    bool skip_group {}; // ablation: record the match phase without the estimate
    bool skip_match {}; // ablation: the estimate without the match phase
    bool skip_gram {};  // ablation: the estimate without its Gram dispatch
    bool skip_solve {}; // ablation: the estimate without its eigen dispatch
    bool skip_apply {}; // ablation: the estimate without its scatter
    bool host_timing {};
    bool gpu_trace {}; // VSFEEL_WNNM_GPUTRACE=<frame>

    ~GPUWNNM() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                    "[wnnm-timing] frames=%.0f per-frame us: acquire=%7.1f "
                    "record=%7.1f submit=%7.1f\n",
                    n, ht_acquire_ns.load() / 1000.0 / n,
                    ht_record_ns.load() / 1000.0 / n,
                    ht_submit_ns.load() / 1000.0 / n);
        }
        if (!gpu) {
            return;
        }
        if (pool) {
            gpu->api->freeGPUExecPool(pool);
            pool = nullptr;
        }
        VkDevice dev = gpu->device;
        gpu_destroy_buffer(*gpu, ring_buf);
        if (probe.query) {
            gpu->vk->vkDestroyQueryPool(dev, probe.query, nullptr);
        }
        for (auto & p : plane) {
            if (p.group_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.group_pipe, nullptr);
            }
            if (p.gram_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.gram_pipe, nullptr);
            }
            if (p.solve_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.solve_pipe, nullptr);
            }
            if (p.apply_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.apply_pipe, nullptr);
            }
            if (p.finish_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.finish_pipe, nullptr);
            }
            if (p.dist_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.dist_pipe, nullptr);
            }
            if (p.select_pipe) {
                gpu->vk->vkDestroyPipeline(dev, p.select_pipe, nullptr);
            }
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
// Pipeline creation
// ---------------------------------------------------------------------------

struct WnnmVariant {
    const uint32_t * group;
    size_t group_size;
    const uint32_t * finish;
    size_t finish_size;
    const uint32_t * dist;
    size_t dist_size;
    const uint32_t * select;
    size_t select_size;
};

static WnnmVariant pick_variant(int qmax, bool hw_atomics, bool match_only) {
    WnnmVariant v { nullptr,         0,
                    wnnm_finish_spv, sizeof(wnnm_finish_spv),
                    wnnm_dist_spv,   sizeof(wnnm_dist_spv),
                    wnnm_select_spv, sizeof(wnnm_select_spv) };
    if (match_only) {
        v.group = wnnm_group_q8_mo_spv;
        v.group_size = sizeof(wnnm_group_q8_mo_spv);
    } else if (qmax == 8 && hw_atomics) {
        v.group = wnnm_group_q8_spv;
        v.group_size = sizeof(wnnm_group_q8_spv);
    } else if (qmax == 8) {
        v.group = wnnm_group_q8_cas_spv;
        v.group_size = sizeof(wnnm_group_q8_cas_spv);
    } else if (qmax == 16 && hw_atomics) {
        v.group = wnnm_group_q16_spv;
        v.group_size = sizeof(wnnm_group_q16_spv);
    } else if (qmax == 16) {
        v.group = wnnm_group_q16_cas_spv;
        v.group_size = sizeof(wnnm_group_q16_cas_spv);
    } else if (hw_atomics) {
        v.group = wnnm_group_q32_spv;
        v.group_size = sizeof(wnnm_group_q32_spv);
    } else {
        v.group = wnnm_group_q32_cas_spv;
        v.group_size = sizeof(wnnm_group_q32_cas_spv);
    }
    return v;
}

// Static LDS of the group kernel, exactly as src/wnnm.comp declares it: eight
// 256-entry arrays (candidate tile, final list, frame list, centre list, seed
// order), the eigen workspace, the round-robin rotation cache and seven
// scalars. tools/shader_limits.py --check
// compares this against the built module.
static uint32_t group_shared_bytes(int qmax) {
    const uint32_t lists = 8u * 256u * 4u;
    const uint32_t q2 = uint32_t(qmax) * uint32_t(qmax);
    // + the packed centre block (REFBLK_MAX) and the split matcher's window.
    return lists + 2u * q2 * 4u + q2 * 4u + uint32_t(qmax) * 8u + 24u +
           256u * 4u;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static VkDeviceAddress plane_address(const GPUDevice & gpu, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return gpu.vk->vkGetBufferDeviceAddress(gpu.device, &info);
}

// A whole-queue barrier: the cache's ring is written by earlier frames, so this
// frame's dispatches start behind everything already submitted on the compute
// queue (the same cross-submission handoff the BM3D ring relies on).
static void record_queue_barrier(const GPUDevice & gpu, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask =
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.dstAccessMask =
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    gpu.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

// Zero the accumulator regions and make the writes visible to the dispatches.
static void record_acc_clear(const GPUDevice & gpu, VkCommandBuffer cmd,
                             VkBuffer acc, VkDeviceSize bytes) {
    if (bytes == 0) {
        return;
    }
    gpu.vk->vkCmdFillBuffer(cmd, acc, 0, bytes, 0);
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
    gpu.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

static const VSFrame * wnnm_frame(GPUWNNM * d, int n, VSFrameContext * frameCtx,
                                  VSCore * core, const VSAPI * vsapi) {

    const int layers = d->radius == 0 ? 1 : 4 * d->radius + 1;
    const int first = d->radius == 0 ? n : n - 2 * d->radius;
    const int nf = d->vi->numFrames;

    std::vector<const VSFrame *> frames(layers);
    std::vector<const VSFrame *> rframes(layers);
    for (int l = 0; l < layers; ++l) {
        const int f = std::clamp(first + l, 0, nf - 1);
        frames[l] = vsapi->getFrameFilter(f, d->node, frameCtx);
        rframes[l] = d->ref_node
                         ? vsapi->getFrameFilter(f, d->ref_node, frameCtx)
                         : frames[l];
    }
    const VSFrame * center = frames[layers / 2];

    auto cleanup = [&] {
        for (int l = 0; l < layers; ++l) {
            vsapi->freeFrame(frames[l]);
            if (d->ref_node) {
                vsapi->freeFrame(rframes[l]);
            }
        }
    };

    const int pl[3] = { 0, 1, 2 };
    const VSFrame * fr[3] = { d->plane[0].process ? nullptr : center,
                              d->plane[1].process ? nullptr : center,
                              d->plane[2].process ? nullptr : center };
    bool all = true;
    for (int p = 0; p < d->planes; ++p) {
        all = all && d->plane[p].process;
    }
    VSFrame * dst =
        all ? d->gpu->api->newGPUVideoFrame(&d->vi->format, d->vi->width,
                                            d->vi->height, center, core)
            : vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height,
                                    fr, pl, center, core);
    if (!dst) {
        vsfeel_trace_error("WNNM", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("WNNM: failed to allocate the output frame",
                              frameCtx);
        cleanup();
        return nullptr;
    }

    const auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point {};
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
        vsfeel_trace_error("WNNM", n, message, d->gpu.get());
        vsapi->setFilterError(("WNNM: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        cleanup();
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    const auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point {};

    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
    const bool gputrace = d->gpu_trace && n == d->gpu_trace_frame &&
                          d->probe.armed.exchange(1) == 0;
    int probe_slot = 0;
    auto probe_ts = [&] {
        if (gputrace) {
            d->gpu->vk->vkCmdWriteTimestamp2(
                cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, d->probe.query,
                probe_slot++);
        }
    };
    if (gputrace) {
        d->gpu->vk->vkCmdResetQueryPool(cmd, d->probe.query, 0, kProbeSlots);
        d->gpu->vk->vkCmdWriteTimestamp2(cmd,
                                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                         d->probe.query, probe_slot++);
    }

    GpuBuffer acc_buf {}, pmat_buf {}, locs_buf {}, addr_buf {}, tr_buf {},
        dist_buf {}, match_buf {};
    // The cache's ring is written by earlier frames' submissions and read here,
    // so every frame on that path starts behind everything already queued (the
    // BM3D ring uses the same handoff). Holding the lock from here to the
    // submit is what makes record order the submission order the ring needs.
    std::unique_lock<std::mutex> cache_guard(d->cache_lock, std::defer_lock);
    if (d->cache) {
        cache_guard.lock();
    }
    // Folds this frame has to search: a cached centre is one an earlier frame
    // already computed, and a seek throws the whole ring away.
    std::vector<int> todo;
    if (d->cache) {
        for (int i = 0; i < d->folds; ++i) {
            const int cm = std::clamp(n - d->radius + i, 0, nf - 1);
            if (d->ring_tag[cm & (d->ring - 1)] == cm) {
                continue;
            }
            bool dup = false;
            for (int t : todo) {
                dup = dup || std::clamp(n - d->radius + t, 0, nf - 1) == cm;
            }
            if (!dup) {
                todo.push_back(i);
            }
        }
        if (vsfeel_debug_flag("VSFEEL_WNNM")) {
            fprintf(stderr, "[wnnm] n=%d cached %d of %d folds, todo:", n,
                    d->folds - static_cast<int>(todo.size()), d->folds);
            for (int t : todo) {
                fprintf(stderr, " %d", t);
            }
            fprintf(stderr, "\n");
        }
    }
    // The dump pipelines carry TRACE=1 for every frame, so the trace buffer has
    // to be bound on every frame (dumping to a dummy would scribble on the
    // accumulator); only the selected frame's contents are printed.
    const bool tracing = d->dump_frame >= 0 && d->trace_elems > 0;
    const bool dumping = tracing && d->dump_frame == n;
    if (d->acc_bytes) {
        if (auto e = gpu_frame_buffer(*d->gpu, core, ctx, d->acc_bytes, acc_buf,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            !e.empty()) {
            return fail("accumulator: " + e);
        }
    }
    if (d->pmat_bytes) {
        if (auto e =
                gpu_frame_buffer(*d->gpu, core, ctx, d->pmat_bytes, pmat_buf);
            !e.empty()) {
            return fail("patch scratch: " + e);
        }
    }
    if (d->locs_bytes) {
        if (auto e =
                gpu_frame_buffer(*d->gpu, core, ctx, d->locs_bytes, locs_buf);
            !e.empty()) {
            return fail("location scratch: " + e);
        }
    }
    if (d->dist_bytes) {
        if (auto e =
                gpu_frame_buffer(*d->gpu, core, ctx, d->dist_bytes, dist_buf);
            !e.empty()) {
            return fail("distance scratch: " + e);
        }
    }
    if (d->match_bytes) {
        if (auto e =
                gpu_frame_buffer(*d->gpu, core, ctx, d->match_bytes, match_buf);
            !e.empty()) {
            return fail("match scratch: " + e);
        }
    }
    // Address table: [plane][ref/src][layer], 64-bit device addresses.
    if (auto e = gpu_make_buffer(
            *d->gpu, core,
            VkDeviceSize(3) * 2 * layers * sizeof(VkDeviceAddress), addr_buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        !e.empty()) {
        return fail("address table: " + e);
    }
    d->gpu->api->gpuExecUsesBuffer(ctx, addr_buf.handle);
    if (tracing) {
        if (auto e = gpu_make_buffer(
                *d->gpu, core, VkDeviceSize(d->trace_elems) * sizeof(uint32_t),
                tr_buf,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            !e.empty()) {
            return fail("trace buffer: " + e);
        }
        d->gpu->api->gpuExecUsesBuffer(ctx, tr_buf.handle);
    }
    auto * table = static_cast<VkDeviceAddress *>(addr_buf.mapped);
    if (table == nullptr) {
        return fail("address table is not host visible");
    }
    for (int p = 0; p < 3; ++p) {
        if (!d->plane[p].process) {
            // Never read by a kernel; keep the table fully written anyway.
            for (int l = 0; l < layers; ++l) {
                table[p * 2 * layers + l] = 0;
                table[p * 2 * layers + layers + l] = 0;
            }
            continue;
        }
        for (int l = 0; l < layers; ++l) {
            VSVulkanPlaneInfo sp {};
            if (d->gpu->api->getGPUPlane(rframes[l], p, &sp)) {
                return fail("reference plane " + std::to_string(p) +
                            " is not GPU resident");
            }
            VSVulkanPlaneInfo pp {};
            if (d->gpu->api->getGPUPlane(frames[l], p, &pp)) {
                return fail("source plane " + std::to_string(p) +
                            " is not GPU resident");
            }
            table[p * 2 * layers + l] = plane_address(*d->gpu, sp.buffer);
            table[p * 2 * layers + layers + l] =
                plane_address(*d->gpu, pp.buffer);
        }
    }
    // The table may land in the host-visible VRAM BAR, whose mapping is
    // write-combined: the first submission must not read a tail that has not
    // drained yet.
    store_fence();

    VkBuffer dst_plane[3] {};
    for (int p = 0; p < 3; ++p) {
        if (!d->plane[p].process) {
            continue;
        }
        VSVulkanPlaneInfo dp {};
        if (d->gpu->api->getGPUPlane(dst, p, &dp)) {
            return fail("output plane " + std::to_string(p) +
                        " is not GPU resident");
        }
        dst_plane[p] = dp.buffer;
    }

    VkBuffer dummy = acc_buf.buffer ? acc_buf.buffer : addr_buf.buffer;
    // With the cache on, the match records live in the persistent ring.
    const VkBuffer match_view =
        d->cache ? d->ring_buf.buffer
                 : (match_buf.buffer ? match_buf.buffer : dummy);
    if (d->cache) {
        record_queue_barrier(*d->gpu, cmd);
    }
    record_acc_clear(*d->gpu, cmd, acc_buf.buffer, d->acc_bytes);

    vsfeel_trace_mark("dispatch");
    // Phase-major rather than plane-major: every per-plane region (accumulator,
    // patch scratch, ring slots, address table) is disjoint, so the planes'
    // dispatches inside one phase are independent and the GPU can run them
    // together. That is what the eigen step needs: it is one thread per group,
    // so one plane's dispatch leaves most of the CU idle, and the other plane's
    // work fills it without a barrier in between. A barrier is only needed
    // between phases, where the next phase reads what the previous wrote.
    const auto bind_match = [&] {
        const VkBuffer buffers[8] { addr_buf.buffer,
                                    dummy,
                                    dummy,
                                    dummy,
                                    dummy,
                                    dummy,
                                    dist_buf.buffer ? dist_buf.buffer : dummy,
                                    match_view };
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 8);
    };
    const auto bind_estimate = [&](VkBuffer dst_buf) {
        const VkBuffer buffers[8] { addr_buf.buffer,
                                    acc_buf.buffer,
                                    locs_buf.buffer ? locs_buf.buffer : dummy,
                                    pmat_buf.buffer ? pmat_buf.buffer : dummy,
                                    dst_buf,
                                    tr_buf.buffer ? tr_buf.buffer : dummy,
                                    dist_buf.buffer ? dist_buf.buffer : dummy,
                                    match_view };
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 8);
    };
    const bool spatial_match =
        d->split_match && d->radius == 0 && !d->skip_match;

    // Distance kernel (spatial split path): one dispatch per plane.
    if (spatial_match) {
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pln = d->plane[p];
            if (!pln.process) {
                continue;
            }
            bind_match();
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pln.dist_pipe);
            d->gpu->vk->vkCmdDispatch(cmd, pln.grid_x, pln.grid_y, 1);
        }
        gpu_barrier(*d->gpu, cmd);
        // Selection kernel: one thread per group, reads the distances above.
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pln = d->plane[p];
            if (!pln.process) {
                continue;
            }
            bind_match();
            const uint32_t sel_groups =
                (static_cast<uint32_t>(pln.ng) + kWorkgroup - 1) / kWorkgroup;
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pln.select_pipe);
            d->gpu->vk->vkCmdDispatch(cmd, sel_groups, 1, 1);
        }
        gpu_barrier(*d->gpu, cmd);
    }
    // Matcher: the temporal split's one dispatch per missing centre, or the
    // fused kernel that also estimates and scatters.
    if (!d->skip_group) {
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pln = d->plane[p];
            if (!pln.process) {
                continue;
            }
            bind_estimate(dst_plane[p]);
            if (d->split_match) {
                if (d->radius > 0 && !d->skip_match) {
                    // The predictive search is inherently serial and publishes
                    // one selection per centre; with the cache on only the
                    // folds whose centre is not in the ring run.
                    const int32_t pc_all[3] { n, d->vi->numFrames, 0 };
                    for (size_t t = 0; t < todo.size(); ++t) {
                        const int32_t one[3] { n, d->vi->numFrames, todo[t] };
                        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout,
                                           one, sizeof(one));
                        d->gpu->vk->vkCmdBindPipeline(
                            cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pln.group_pipe);
                        d->gpu->vk->vkCmdDispatch(cmd, pln.grid_x, pln.grid_y,
                                                  1);
                        gpu_barrier(*d->gpu, cmd);
                    }
                    if (!d->cache) {
                        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout,
                                           pc_all, sizeof(pc_all));
                        d->gpu->vk->vkCmdBindPipeline(
                            cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pln.group_pipe);
                        d->gpu->vk->vkCmdDispatch(cmd, pln.grid_x, pln.grid_y,
                                                  1);
                        gpu_barrier(*d->gpu, cmd);
                    }
                }
            } else {
                d->gpu->vk->vkCmdBindPipeline(
                    cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pln.group_pipe);
                const int32_t fold[2] { n, d->vi->numFrames };
                gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, fold,
                                   sizeof(fold));
                d->gpu->vk->vkCmdDispatch(cmd, pln.grid_x, pln.grid_y, 1);
            }
        }
        gpu_barrier(*d->gpu, cmd);
    }
    probe_ts();

    // Gram matrix, eigen step and weighted scatter: one kernel each, over every
    // (lattice point, fold) slot of the plane. Each phase is dispatched for all
    // planes before the barrier, so the two planes' kernels overlap.
    const bool estimate = d->split_match && !d->skip_group;
    if (estimate) {
        // The three estimate kernels key their slots by fold, so they need the
        // output frame and the clip length; the matcher above left its own
        // values behind.
        const int32_t pc[3] { n, d->vi->numFrames, 0 };
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, pc, sizeof(pc));
    }
    if (estimate && !d->skip_gram) {
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pln = d->plane[p];
            if (!pln.process) {
                continue;
            }
            bind_estimate(dst_plane[p]);
            const uint32_t lanes =
                (static_cast<uint32_t>(pln.ng) * uint32_t(d->folds) + 15) / 16;
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pln.gram_pipe);
            d->gpu->vk->vkCmdDispatch(cmd, lanes, 1, 1);
        }
        gpu_barrier(*d->gpu, cmd);
    }
    probe_ts();
    if (estimate && !d->skip_solve) {
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pln = d->plane[p];
            if (!pln.process) {
                continue;
            }
            bind_estimate(dst_plane[p]);
            const uint32_t slots =
                static_cast<uint32_t>(pln.ng) * uint32_t(d->folds);
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pln.solve_pipe);
            d->gpu->vk->vkCmdDispatch(cmd, (slots + 255) / 256, 1, 1);
        }
        gpu_barrier(*d->gpu, cmd);
    }
    probe_ts();
    if (estimate && !d->skip_apply) {
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pln = d->plane[p];
            if (!pln.process) {
                continue;
            }
            bind_estimate(dst_plane[p]);
            const uint32_t lanes =
                (static_cast<uint32_t>(pln.ng) * uint32_t(d->folds) + 15) / 16;
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pln.apply_pipe);
            d->gpu->vk->vkCmdDispatch(cmd, lanes, 1, 1);
        }
        gpu_barrier(*d->gpu, cmd);
    }
    probe_ts();
    // Normalize: the scatter's atomics have to be visible, so this phase starts
    // behind the barrier above.
    for (int p = 0; p < 3; ++p) {
        WnnmPlane & pln = d->plane[p];
        if (!pln.process) {
            continue;
        }
        bind_estimate(dst_plane[p]);
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pln.finish_pipe);
        d->gpu->vk->vkCmdDispatch(cmd, pln.finish_x,
                                  static_cast<uint32_t>(pln.height), 1);
    }
    gpu_barrier(*d->gpu, cmd);
    probe_ts();
    if (gputrace) {
        gpu_barrier(*d->gpu, cmd);
        d->gpu->vk->vkCmdCopyQueryPoolResults(
            cmd, d->probe.query, 0, kProbeSlots, d->probe.buf.buffer, 0,
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    }
    const auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point {};

    for (int l = 0; l < layers; ++l) {
        bool dup = false;
        for (int u = 0; u < l; ++u) {
            dup |= frames[u] == frames[l];
        }
        if (!dup) {
            d->gpu->api->gpuExecReadsFrame(ctx, frames[l]);
        }
        if (d->ref_node) {
            dup = false;
            for (int u = 0; u < l; ++u) {
                dup |= rframes[u] == rframes[l];
            }
            if (!dup) {
                d->gpu->api->gpuExecReadsFrame(ctx, rframes[l]);
            }
        }
    }
    for (int p = 0; p < 3; ++p) {
        if (d->plane[p].process) {
            d->gpu->api->gpuExecWritesPlane(ctx, dst, p);
        }
    }

    vsfeel_trace_mark("submit");
    uint64_t signaled = 0;
    const int submit_error =
        d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf, sizeof(errbuf));
    ctx = nullptr;
    if (submit_error) {
        return fail("submit failed: "s + errbuf);
    }
    // The centres this frame searched are now submitted, so a later frame that
    // consumes them may record against the ring.
    if (d->cache) {
        for (int f : todo) {
            const int cm = std::clamp(n - d->radius + f, 0, nf - 1);
            d->ring_tag[cm & (d->ring - 1)] = cm;
        }
    }
    const auto t3 = d->host_timing ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point {};

    if (gputrace) {
        char perr[256] {};
        if (d->gpu->api->gpuExecWaitValue(d->pool, signaled, perr,
                                          sizeof(perr)) == gdDrained) {
            const double period = d->gpu->limits.timestampPeriod;
            const uint64_t * ts = d->probe.map;
            const auto us = [period](uint64_t a, uint64_t b) {
                return static_cast<double>(b - a) * period / 1000.0;
            };
            fprintf(stderr,
                    "[wnnm-gpu] n=%d total=%.1fus period=%.3f raw=%llu %llu\n",
                    n, us(ts[0], ts[probe_slot - 1]), period,
                    (unsigned long long)ts[0],
                    (unsigned long long)ts[probe_slot - 1]);
            // One stamp pair per phase, so each line covers every plane the
            // phase dispatched.
            const char * names[5] { "match", "gram", "solve", "apply",
                                    "finish" };
            for (int k = 1; k < probe_slot; ++k) {
                fprintf(stderr, "[wnnm-gpu]   %s=%.1fus\n",
                        k <= 5 ? names[k - 1] : "phase", us(ts[k - 1], ts[k]));
            }
        } else {
            fprintf(stderr, "[wnnm-gpu] probe wait failed: %s\n", perr);
        }
    }

    if (dumping) {
        char perr[256] {};
        if (d->gpu->api->gpuExecWaitValue(d->pool, signaled, perr,
                                          sizeof(perr)) == gdDrained) {
            const uint32_t * tr = static_cast<const uint32_t *>(tr_buf.mapped);
            for (int p = 0; p < 3; ++p) {
                const WnnmPlane & dpl = d->plane[p];
                if (!dpl.process) {
                    continue;
                }
                for (int gg = 0; gg < dpl.ng; ++gg) {
                    for (int td = 0; td <= 2 * d->radius; ++td) {
                        const uint32_t * e =
                            tr + dpl.trace_base +
                            (int64_t(gg) * (2 * d->radius + 1) + td) * 28;
                        fprintf(stderr,
                                "[wnnm-dump] p=%d gx=%d gy=%d td=%d n=%u "
                                "kept=%u w=%.9g nl=%u\n",
                                p, gg % dpl.gx, gg / dpl.gx, td, e[0], e[1],
                                std::bit_cast<float>(e[2]), e[3]);
                        for (int j = 0; j < 8 && j < int(e[0]); ++j) {
                            fprintf(
                                stderr,
                                "[wnnm-dump]   j=%d err=%.9g xy=0x%x t=%u\n", j,
                                std::bit_cast<float>(e[4 + j * 3]),
                                e[5 + j * 3], e[6 + j * 3]);
                        }
                    }
                }
            }
        } else {
            fprintf(stderr, "[wnnm-dump] wait failed: %s\n", perr);
        }
    }

    if (d->host_timing) {
        const auto ns = [](auto a, auto b) {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                    .count());
        };
        d->ht_acquire_ns += ns(t0, t1);
        d->ht_record_ns += ns(t1, t2);
        d->ht_submit_ns += ns(t2, t3);
        d->ht_n.fetch_add(1, std::memory_order_relaxed);
    }

    cleanup();
    return dst;
}

static const VSFrame * VS_CC WNNMGetFrame(int n, int activationReason,
                                          void * instanceData,
                                          [[maybe_unused]] void ** frameData,
                                          VSFrameContext * frameCtx,
                                          VSCore * core, const VSAPI * vsapi) {

    GPUWNNM * d = static_cast<GPUWNNM *>(instanceData);

    if (activationReason == arInitial) {
        const int layers = d->radius == 0 ? 1 : 4 * d->radius + 1;
        const int first = d->radius == 0 ? n : n - 2 * d->radius;
        for (int l = 0; l < layers; ++l) {
            const int f = std::clamp(first + l, 0, d->vi->numFrames - 1);
            vsapi->requestFrameFilter(f, d->node, frameCtx);
            if (d->ref_node) {
                vsapi->requestFrameFilter(f, d->ref_node, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }
    return wnnm_frame(d, n, frameCtx, core, vsapi);
}

static void VS_CC WNNMFree(void * instanceData, [[maybe_unused]] VSCore * core,
                           [[maybe_unused]] const VSAPI * vsapi) {
    delete static_cast<GPUWNNM *>(instanceData);
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC WNNMCreate(const VSMap * in, VSMap * out,
                             [[maybe_unused]] void * userData, VSCore * core,
                             const VSAPI * vsapi) {

    auto d { std::make_unique<GPUWNNM>() };

    d->host_timing = vsfeel_debug_probe("VSFEEL_WNNM_TIMING");
    d->skip_group = env_flag("VSFEEL_WNNM_SKIPGROUP");
    d->skip_match = env_flag("VSFEEL_WNNM_SKIPMATCH");
    d->skip_gram = env_flag("VSFEEL_WNNM_NOGRAM");
    d->skip_solve = env_flag("VSFEEL_WNNM_NOSOLVE");
    d->skip_apply = env_flag("VSFEEL_WNNM_NOAPPLY");
    d->dump_frame = env_int("VSFEEL_WNNM_DUMP", -1);

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & message) {
        vsfeel_trace_error("WNNM", -1, message, d->gpu.get());
        vsapi->mapSetError(out, ("WNNM: " + message).c_str());
        vsapi->freeNode(d->node);
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    if (!vsh::isConstantVideoFormat(d->vi) ||
        d->vi->format.sampleType != stFloat ||
        d->vi->format.bitsPerSample != 32) {
        return set_error("only constant format 32 bit float input supported");
    }
    if (d->vi->width <= 0 || d->vi->height <= 0) {
        return set_error("clip must have constant dimensions");
    }
    if (d->vi->numFrames <= 0) {
        return set_error("clip frame count must be known and positive");
    }
    if (d->vi->width > 65535 || d->vi->height > 65535) {
        return set_error("clip dimensions must not exceed 65535x65535");
    }

    int bs = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_size", 0, &error));
    if (error) {
        bs = 8;
    }
    if (bs < 1 || bs > 64) {
        return set_error("\"block_size\" must be in [1, 64]");
    }
    int step = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_step", 0, &error));
    if (error) {
        step = bs;
    }
    if (step < 1 || step > bs) {
        return set_error(
            "\"block_step\" must be positive and no larger than \"block_size\"");
    }
    int gs = vsh::int64ToIntS(vsapi->mapGetInt(in, "group_size", 0, &error));
    if (error) {
        gs = 8;
    }
    if (gs < 1 || gs > kMaxList) {
        return set_error("\"group_size\" must be in [1, 256]");
    }
    const int m = bs * bs;
    if (std::min(m, gs) > 32) {
        // The LDS eigen workspace is sized by -DQMAX (8/16/32); the reference
        // caps this product at 64.
        return set_error("min(block_size^2, group_size) must be <= 32 on this "
                         "GPU implementation");
    }
    d->bs = bs;
    d->step = step;
    d->gs = gs;
    d->patch_elems = m;
    d->q = std::min(m, gs);

    int bm = vsh::int64ToIntS(vsapi->mapGetInt(in, "bm_range", 0, &error));
    if (error) {
        bm = 7;
    }
    if (bm < 0) {
        return set_error("\"bm_range\" must be non-negative");
    }
    d->bm_range = bm;

    int radius = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", 0, &error));
    if (error) {
        radius = 0;
    }
    if (radius < 0 || radius > kMaxRadius) {
        return set_error("\"radius\" must be in [0, 15]");
    }
    d->radius = radius;

    int ps_num = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_num", 0, &error));
    if (error) {
        ps_num = 2;
    }
    if (ps_num < 1 || ps_num > kMaxPsNum) {
        return set_error("\"ps_num\" must be in [1, 64]");
    }
    d->ps_num = ps_num;

    int ps_range =
        vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_range", 0, &error));
    if (error) {
        ps_range = 4;
    }
    if (ps_range < 0) {
        return set_error("\"ps_range\" must be non-negative");
    }
    d->ps_range = ps_range;

    d->residual = vsapi->mapGetInt(in, "residual", 0, &error) != 0;
    d->adaptive = vsapi->mapGetInt(in, "adaptive_aggregation", 0, &error) != 0;
    if (error) {
        d->adaptive = true;
    }

    // Per-plane sigma, exactly as the reference: a missing entry repeats the
    // previous one and the first defaults to 3; a plane below FLT_EPSILON is
    // not processed and rides along from the centre frame.
    std::array<float, 3> sigma {};
    for (int i = 0; i < 3; ++i) {
        sigma[i] =
            static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &error));
        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (!std::isfinite(sigma[i]) || sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be finite and non-negative");
        }
    }

    d->ref_node = vsapi->mapGetNode(in, "rclip", 0, &error);
    if (d->ref_node) {
        const VSVideoInfo * rvi = vsapi->getVideoInfo(d->ref_node);
        if (!vsh::isSameVideoInfo(rvi, d->vi) ||
            rvi->numFrames != d->vi->numFrames) {
            return set_error("\"rclip\" must be of the same format, dimensions "
                             "and number of frames as \"clip\"");
        }
    }

    const int num_planes = d->vi->format.numPlanes;
    d->planes = num_planes;
    bool any = false;
    for (int p = 0; p < 3; ++p) {
        d->plane[p].process = p < num_planes && sigma[p] >= FLT_EPSILON;
        any = any || d->plane[p].process;
        d->plane[p].sigma = sigma[p] / 255.0f;
    }
    if (!any) {
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
            d->ref_node = nullptr;
        }
        vsapi->mapConsumeNode(out, "clip", d->node, maReplace);
        return;
    }

    const auto dev_result = get_gpu_device(core, vsapi);
    if (std::holds_alternative<std::string>(dev_result)) {
        return set_error(std::get<std::string>(dev_result));
    }
    d->gpu = std::get<std::shared_ptr<GPUDevice>>(dev_result);

    const int qmax = d->q <= 8 ? 8 : (d->q <= 16 ? 16 : 32);
    // The split estimate runs any radius; only a group too wide for its
    // register-resistant kernels keeps the fused one.
    d->split_match = d->gs <= 8 && !env_flag("VSFEEL_WNNM_NOSPLIT");
    // One estimate slot per (lattice point, temporal fold).
    d->folds = 2 * d->radius + 1;

    // Geometry and per-plane scratch. The kernel's group lattice is the
    // reference's `min(_x, temp_c)` walk. The layout runs again when the
    // centre cache's ring has to shrink to its budget.
    // The ring is persistent, not per-frame scratch, and spans the request
    // depth rather than 2r+1 slots, so it gets a budget of its own: 160 MiB
    // fits the depth-sized ring at 1080p luma and 4K chroma (the old 2r+1 ring
    // held under 20 MiB there).
    constexpr int64_t kCacheBudget = INT64_C(160) << 20;
    d->cache =
        d->split_match && d->radius > 0 && !env_flag("VSFEEL_WNNM_NOCACHE");
    // A centre's record is only published when its own frame is submitted, so
    // the ring has to span the frames the core keeps in flight: a smaller ring
    // evicts a tag before the neighbouring frame that would hit it reads it,
    // and every frame re-searches all folds (0.95 of 3 cached instead of 2 at
    // the default request depth of 32, ~3x the matcher's time).
    int depth = 2 * d->folds;
    {
        VSCoreInfo info {};
        vsapi->getCoreInfo(core, &info);
        depth += std::max(0, info.numThreads);
    }
    d->ring = d->cache ? std::min(256, round_up_pow2(depth)) : d->folds;
    auto layout = [&] {
        int64_t acc_off = 0, pmat_off = 0, locs_off = 0, dist_off = 0,
                match_off = 0;
        for (int p = 0; p < 3; ++p) {
            WnnmPlane & pl = d->plane[p];
            if (!pl.process) {
                continue;
            }
            const int ssw = p == 0 ? 0 : d->vi->format.subSamplingW;
            const int ssh = p == 0 ? 0 : d->vi->format.subSamplingH;
            pl.width = p == 0 ? d->vi->width : d->vi->width >> ssw;
            pl.height = p == 0 ? d->vi->height : d->vi->height >> ssh;
            if (d->bs > pl.width || d->bs > pl.height) {
                return std::string(
                    "\"block_size\" must not exceed the dimensions "
                    "of any processed plane");
            }
            VSFrame * probe = vsapi->newVideoFrame(
                &d->vi->format, d->vi->width, d->vi->height, nullptr, core);
            if (probe == nullptr) {
                return std::string("could not allocate a probe frame");
            }
            pl.stride =
                static_cast<int>(vsapi->getStride(probe, p) / sizeof(float));
            vsapi->freeFrame(probe);
            pl.npix = int64_t(pl.stride) * pl.height;

            const int temp_c = pl.width - d->bs;
            const int temp_r = pl.height - d->bs;
            {
                const int stride = std::min(2 * d->bm_range + 1, temp_r + 1) *
                                   std::min(2 * d->bm_range + 1, temp_c + 1);
                d->dist_stride_max = std::max(d->dist_stride_max, stride);
            }
            pl.gx = temp_c / d->step + 1 + (temp_c % d->step != 0 ? 1 : 0);
            pl.gy = temp_r / d->step + 1 + (temp_r % d->step != 0 ? 1 : 0);
            pl.ng = pl.gx * pl.gy;
            pl.finish_x = uint32_t((pl.width + kWorkgroup - 1) / kWorkgroup);

            pl.acc_elems = int64_t(2) * pl.width * pl.height;
            pl.pmat_elems = d->split_match
                                ? int64_t(pl.ng) * d->folds * kSplitStride
                                : int64_t(pl.ng) * m *
                                      (int64_t(d->gs) + (d->residual ? 1 : 0));
            if (d->radius > 0) {
                const int area = std::min(2 * d->bm_range + 1, temp_r + 1) *
                                 std::min(2 * d->bm_range + 1, temp_c + 1);
                const int centers = std::min(d->ps_num, area);
                const int rect = std::min(2 * d->ps_range + 1, temp_c + 1) *
                                 std::min(2 * d->ps_range + 1, temp_r + 1);
                pl.locs_elems =
                    int64_t(pl.ng) *
                    std::min<int64_t>(int64_t(centers) * rect,
                                      int64_t(temp_c + 1) * (temp_r + 1));
            } else {
                pl.locs_elems = int64_t(pl.ng);
            }
            // Bases are element offsets (the kernels index float/uint arrays), kept
            // 32-byte aligned so every plane region starts on a cache-line boundary.
            pl.dist_elems = int64_t(pl.ng) * d->dist_stride_max;
            pl.match_elems = int64_t(pl.ng) * (d->split_match ? d->ring : 1) *
                             (1 + 2 * d->gs);
            pl.acc_base = acc_off;
            pl.pmat_base = pmat_off;
            pl.locs_base = locs_off;
            pl.dist_base = dist_off;
            pl.match_base = match_off;
            acc_off += (pl.acc_elems + 7) & ~int64_t(7);
            pmat_off += (pl.pmat_elems + 7) & ~int64_t(7);
            locs_off += (pl.locs_elems + 7) & ~int64_t(7);
            dist_off += (pl.dist_elems + 7) & ~int64_t(7);
            match_off += (pl.match_elems + 7) & ~int64_t(7);
        }
        d->acc_bytes = acc_off * sizeof(float);
        d->pmat_bytes = pmat_off * sizeof(float);
        d->locs_bytes = locs_off * sizeof(uint32_t);
        if (d->split_match) {
            d->dist_bytes = dist_off * sizeof(float);
            // A cached ring is persistent, not per-frame scratch.
            d->ring_bytes =
                d->cache ? match_off * int64_t(sizeof(uint32_t)) : 0;
            d->match_bytes =
                d->cache ? 0 : match_off * int64_t(sizeof(uint32_t));
        }
        // The kernels address these regions with signed 32-bit indices, so
        // every region (and its base) has to stay below 2^31 elements.
        constexpr int64_t kIndexMax = INT64_C(1) << 31;
        if (acc_off >= kIndexMax || pmat_off >= kIndexMax ||
            locs_off >= kIndexMax) {
            return std::string("the scratch layout exceeds the 2^31-element "
                               "index range; lower block_size/group_size or "
                               "radius");
        }
        return std::string();
    };
    if (auto e = layout(); !e.empty()) {
        return set_error(e);
    }
    if (d->cache && d->ring_bytes > kCacheBudget) {
        // Records over budget: halve the ring while it still spans more than
        // one centre's folds. A smaller ring keeps the hits it can, which beats
        // dropping to per-frame slots; below the fold count there is no reuse
        // left to keep and the cache goes away entirely.
        const int floor_ring = round_up_pow2(2 * d->folds);
        while (d->ring > floor_ring && d->ring_bytes > kCacheBudget) {
            d->ring /= 2;
            if (auto e = layout(); !e.empty()) {
                return set_error(e);
            }
        }
        if (d->ring_bytes > kCacheBudget) {
            d->cache = false;
            d->ring = d->folds;
            if (auto e = layout(); !e.empty()) {
                return set_error(e);
            }
        }
    }
    if (d->dump_frame >= 0) {
        int64_t tr = 0;
        for (auto & p : d->plane) {
            if (p.process) {
                p.trace_base = tr;
                tr += int64_t(p.ng) * (2 * d->radius + 1) * 28;
            }
        }
        d->trace_elems = tr;
        constexpr int64_t kIndexMax = INT64_C(1) << 31;
        if (d->trace_elems >= kIndexMax) {
            return set_error("the trace layout exceeds the 2^31-element index "
                             "range; lower block_size/group_size or radius");
        }
    }

    // Per-frame scratch budget: the core's allowance has to hold the in-flight
    // frames' scratch as well as their pixels.
    const int64_t per_frame = d->acc_bytes + d->pmat_bytes + d->locs_bytes +
                              d->dist_bytes + d->match_bytes;
    {
        const VkDeviceSize limit = vsfeel_vram_limit(*d->gpu, core);
        if (limit > 0 && per_frame > static_cast<int64_t>(limit / 8)) {
            return set_error(
                "per-frame scratch (" + std::to_string(per_frame) +
                " bytes) exceeds one eighth of the VRAM allowance; "
                "lower block_size/group_size or radius");
        }
        if (per_frame >
            static_cast<int64_t>(d->gpu->limits.maxStorageBufferRange)) {
            return set_error("per-frame scratch exceeds this device's "
                             "maxStorageBufferRange");
        }
    }

    // Grid folds.
    for (int p = 0; p < 3; ++p) {
        WnnmPlane & pl = d->plane[p];
        if (!pl.process) {
            continue;
        }
        std::string err;
        if (!vsfeel_fold_grid(d->gpu->limits, pl.ng, "WNNM group grid",
                              pl.grid_x, pl.grid_y, err)) {
            return set_error(err);
        }
    }

    // Pipelines: one group + one finish per processed plane (the geometry is
    // baked into specialization constants).
    {
        const auto result = gpu_push_set_layout(*d->gpu, 8);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result =
            gpu_pipeline_layout(*d->gpu, d->set_layout, 3 * sizeof(int32_t));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    const bool hw_atomics = d->gpu->feat_atomic_float32_add;
    if (!hw_atomics && vsfeel_debug_flag("VSFEEL_WNNM")) {
        fprintf(stderr, "[wnnm] no buffer float atomics: using the CAS "
                        "accumulator\n");
    }
    const WnnmVariant variant =
        pick_variant(qmax, hw_atomics, d->split_match && d->radius > 0);
    for (int p = 0; p < 3; ++p) {
        WnnmPlane & pl = d->plane[p];
        if (!pl.process) {
            continue;
        }
        WnnmSpec spec {};
        spec.width = pl.width;
        spec.height = pl.height;
        spec.stride = pl.stride;
        spec.bs = d->bs;
        spec.step = d->step;
        spec.gs = d->gs;
        spec.bm_range = d->bm_range;
        spec.radius = d->radius;
        spec.ps_num = d->ps_num;
        spec.ps_range = d->ps_range;
        spec.gx = pl.gx;
        spec.ng = pl.ng;
        spec.residual = d->residual ? 1 : 0;
        spec.adaptive = d->adaptive ? 1 : 0;
        spec.locs_cap =
            d->radius ? static_cast<int32_t>(pl.locs_elems / pl.ng) : 1;
        spec.addr_base = p * 2 * (d->radius == 0 ? 1 : 4 * d->radius + 1);
        spec.sigma = pl.sigma;
        spec.acc_base = static_cast<int32_t>(pl.acc_base);
        spec.pmat_base = static_cast<int32_t>(pl.pmat_base);
        spec.trace = d->dump_frame >= 0 ? 1 : 0;
        spec.tr_base = static_cast<int32_t>(pl.trace_base);
        spec.pmat_stride =
            d->split_match
                ? int32_t(kSplitStride)
                : static_cast<int32_t>(
                      int64_t(m) * (int64_t(d->gs) + (d->residual ? 1 : 0)));
        spec.locs_base = static_cast<int32_t>(pl.locs_base);
        spec.dist_base = static_cast<int32_t>(pl.dist_base);
        spec.match_base = static_cast<int32_t>(pl.match_base);
        spec.ring = d->ring;
        spec.cache = d->cache ? 1 : 0;

        if (vsfeel_debug_flag("VSFEEL_WNNM")) {
            fprintf(stderr,
                    "[wnnm] plane %d: %dx%d stride=%d ng=%d grid=%ux%u "
                    "acc_base=%lld pmat_base=%lld addr_base=%d locs=%lld "
                    "pmat_stride=%d\n",
                    p, spec.width, spec.height, spec.stride, spec.ng, pl.grid_x,
                    pl.grid_y, static_cast<long long>(spec.acc_base),
                    static_cast<long long>(spec.pmat_base), spec.addr_base,
                    static_cast<long long>(pl.locs_elems), spec.pmat_stride);
        }
        auto add = [&](const std::string & tag, const uint32_t * code,
                       size_t bytes, const GpuWorkgroup & wg) -> VkPipeline {
            auto r = gpu_create_pipeline(
                *d->gpu, code, bytes, d->pipeline_layout, spec_entries.data(),
                &spec, static_cast<uint32_t>(spec_entries.size()), sizeof(spec),
                tag.c_str(), 0, wg);
            if (std::holds_alternative<std::string>(r)) {
                set_error(std::get<std::string>(r));
                return VK_NULL_HANDLE;
            }
            return std::get<VkPipeline>(r);
        };
        const std::string tag = "wnnm group p" + std::to_string(p);
        if (d->split_match) {
            // The split estimate replaces the fused group kernel: the Gram and
            // the scatter are eight-lane per group, the eigen step one thread
            // per group (and needs no workgroup memory at all).
            const GpuWorkgroup lanes { .x = 128, .shared_bytes = 0 };
            const GpuWorkgroup one { .x = 256, .shared_bytes = 0 };
            pl.gram_pipe =
                add(tag + " gram", wnnm_gram_spv, sizeof(wnnm_gram_spv), lanes);
            pl.solve_pipe = add(tag + " solve", wnnm_solve_spv,
                                sizeof(wnnm_solve_spv), one);
            pl.apply_pipe = hw_atomics ? add(tag + " apply", wnnm_apply_spv,
                                             sizeof(wnnm_apply_spv), lanes)
                                       : add(tag + " apply", wnnm_apply_cas_spv,
                                             sizeof(wnnm_apply_cas_spv), lanes);
            if (d->radius > 0) {
                // The temporal matcher still runs the fused kernel, up to the
                // point where it publishes its selection.
                const GpuWorkgroup gwg { .x = 32u,
                                         .shared_bytes =
                                             group_shared_bytes(qmax) };
                pl.group_pipe =
                    add(tag, variant.group, variant.group_size, gwg);
            }
            if (!pl.gram_pipe || !pl.solve_pipe || !pl.apply_pipe ||
                (d->radius > 0 && !pl.group_pipe)) {
                return;
            }
        } else {
            // The group kernel's LDS footprint is a function of qmax and the
            // eigen width; gpu_create_pipeline checks it against the device.
            const GpuWorkgroup wg { .x = uint32_t(kWorkgroup),
                                    .shared_bytes = group_shared_bytes(qmax) };
            pl.group_pipe = add(tag, variant.group, variant.group_size, wg);
            if (!pl.group_pipe) {
                return;
            }
        }
        {
            const GpuWorkgroup wg { .x = kWorkgroup };
            auto r = gpu_create_pipeline(
                *d->gpu, variant.finish, variant.finish_size,
                d->pipeline_layout, spec_entries.data(), &spec,
                static_cast<uint32_t>(spec_entries.size()), sizeof(spec),
                ("wnnm finish p" + std::to_string(p)).c_str(), 0, wg);
            if (std::holds_alternative<std::string>(r)) {
                return set_error(std::get<std::string>(r));
            }
            pl.finish_pipe = std::get<VkPipeline>(r);
        }
        if (d->split_match && d->radius == 0 && !d->skip_match) {
            // The distance kernel stages its search window: 1024 floats for
            // any window up to 32x32 (see WIN_MAX in src/wnnm.comp).
            const GpuWorkgroup wg { .x = kWorkgroup, .shared_bytes = 4096 };
            auto r = gpu_create_pipeline(
                *d->gpu, variant.dist, variant.dist_size, d->pipeline_layout,
                spec_entries.data(), &spec,
                static_cast<uint32_t>(spec_entries.size()), sizeof(spec),
                ("wnnm dist p" + std::to_string(p)).c_str(), 0, wg);
            if (std::holds_alternative<std::string>(r)) {
                return set_error(std::get<std::string>(r));
            }
            pl.dist_pipe = std::get<VkPipeline>(r);
            auto r2 = gpu_create_pipeline(
                *d->gpu, variant.select, variant.select_size,
                d->pipeline_layout, spec_entries.data(), &spec,
                static_cast<uint32_t>(spec_entries.size()), sizeof(spec),
                ("wnnm select p" + std::to_string(p)).c_str(), 0, wg);
            if (std::holds_alternative<std::string>(r2)) {
                return set_error(std::get<std::string>(r2));
            }
            pl.select_pipe = std::get<VkPipeline>(r2);
        }
    }

    {
        char err[512] {};
        d->pool =
            d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    if (d->cache) {
        if (auto e = gpu_make_buffer(*d->gpu, core, VkDeviceSize(d->ring_bytes),
                                     d->ring_buf,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            !e.empty()) {
            // The ring is an optimization: no room for it means per-frame
            // slots, which is also what the layout has to be rebuilt for.
            d->cache = false;
            d->ring = d->folds;
            if (auto e2 = layout(); !e2.empty()) {
                return set_error(e2);
            }
        } else {
            d->ring_tag.assign(static_cast<size_t>(d->ring), INT_MIN);
        }
    }

    d->gpu_trace_frame = env_int("VSFEEL_WNNM_GPUTRACE", 100);
    d->gpu_trace = vsfeel_debug_probe("VSFEEL_WNNM_GPUTRACE") &&
                   vsfeel_probe_timestamps(*d->gpu, "WNNM");
    if (d->gpu_trace) {
        VkQueryPoolCreateInfo qp_info {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kProbeSlots
        };
        if (d->gpu->vk->vkCreateQueryPool(d->gpu->device, &qp_info, nullptr,
                                          &d->probe.query) != VK_SUCCESS) {
            d->probe.query = VK_NULL_HANDLE;
            d->gpu_trace = false;
        }
    }
    if (d->gpu_trace) {
        auto e = gpu_make_buffer(*d->gpu, core,
                                 VkDeviceSize(kProbeSlots) * sizeof(uint64_t),
                                 d->probe.buf,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 0, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!e.empty() || d->probe.buf.mapped == nullptr) {
            d->gpu_trace = false;
        } else {
            d->probe.map = static_cast<uint64_t *>(d->probe.buf.mapped);
        }
    }

    if (vsfeel_debug_flag("VSFEEL_WNNM")) {
        fprintf(stderr,
                "[wnnm] bs=%d step=%d gs=%d bm=%d radius=%d ps=%d/%d qmax=%d; "
                "per frame acc=%.1f MiB pmat=%.1f MiB locs=%.1f MiB; "
                "cache=%d ring=%d ring=%.1f MiB\n",
                d->bs, d->step, d->gs, d->bm_range, d->radius, d->ps_num,
                d->ps_range, qmax, double(d->acc_bytes) / 1048576.0,
                double(d->pmat_bytes) / 1048576.0,
                double(d->locs_bytes) / 1048576.0, d->cache ? 1 : 0, d->ring,
                double(d->ring_bytes) / 1048576.0);
    }

    GPUWNNM * data = d.release();
    const VSRequestPattern policy =
        data->radius > 0 ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2] = { { data->node, policy },
                                   { data->ref_node, policy } };
    VSNode * result = vsapi->createVideoFilterEx2(
        "WNNM", data->vi, WNNMGetFrame, WNNMFree, fmParallel, ffGPUOutput, deps,
        data->ref_node ? 2 : 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "WNNM: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

} // namespace

void vsfeel_register_wnnm(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction("WNNM",
                             "clip:vnode:gpu;"
                             "rclip:vnode:gpu:opt;"
                             "sigma:float[]:opt;"
                             "block_size:int:opt;"
                             "block_step:int:opt;"
                             "group_size:int:opt;"
                             "bm_range:int:opt;"
                             "radius:int:opt;"
                             "ps_num:int:opt;"
                             "ps_range:int:opt;"
                             "residual:int:opt;"
                             "adaptive_aggregation:int:opt;"
                             "device_id:int:opt;"
                             "num_streams:int:opt;",
                             "clip:vnode:gpu;", WNNMCreate, nullptr, plugin);
}
