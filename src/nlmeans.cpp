#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <strings.h>

#include <immintrin.h>

#include <volk.h>

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
// NLMeans reads the core's GPU resident source (and guide) frame planes
// directly and writes the output frame's planes in place: `vnode:gpu` in,
// `vnode:gpu` out with ffGPUOutput. There is no tile pool, no host upload, no
// download and no custom queue/fence: the core caches the temporal window's
// frames in VRAM and the exec pool carries synchronization as producer pairs.
//
// The sweep still needs to index an arbitrary temporal layer, and each layer is
// a separate VkBuffer. A small host-visible table (binding 0) is filled per
// frame with the device address of every (clip, channel, layer) plane, and a
// compose pass copies those planes once into one zero-padded device-local
// window (binding 1) so the sweep kernels use 32-bit offsets into a single
// buffer instead of 64-bit buffer-reference addressing on every load.

struct NLMeansSpecData {
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t pstride;
    int32_t ph;
    int32_t pad;
    int32_t s;
    int32_t d;
    int32_t ref;
    int32_t channels;
    int32_t wmode;
    float wref;
    float h2_inv_norm;
    int32_t guide_off;
};

static constexpr std::array<VkSpecializationMapEntry, 14> spec_entries {{
    { 0,  0, sizeof(int32_t) },
    { 1,  4, sizeof(int32_t) },
    { 2,  8, sizeof(int32_t) },
    { 3, 12, sizeof(int32_t) },
    { 4, 16, sizeof(int32_t) },
    { 5, 20, sizeof(int32_t) },
    { 6, 24, sizeof(int32_t) },
    { 7, 28, sizeof(int32_t) },
    { 8, 32, sizeof(int32_t) },
    { 9, 36, sizeof(int32_t) },
    { 10, 40, sizeof(int32_t) },
    { 11, 44, sizeof(float) },
    { 12, 48, sizeof(float) },
    { 13, 52, sizeof(int32_t) },
}};

// Workgroup tile geometry of the weight kernel (must match nlmeans.comp).
constexpr int BLK_X = 32;
constexpr int BLK_Y = 8;
constexpr int VRT_RESULT = 3;

// GPU-trace timestamp pool size: slots 0..3 fixed, then one pair per
// weight+acc batch.
constexpr uint32_t NLMEANS_TS_MAX = 130;
constexpr uint32_t NLMEANS_TS_RESERVED = 4;

// One sweep-table variant per reachable temporal boundary count m=min(d, n).
struct Variant {
    uint32_t w_base {};
    uint32_t q_base {};
    uint32_t q_cnt {};
    std::vector<uint32_t> w_boff;
};

struct NLMeansData {
    VSNode * node {};
    VSNode * ref_node {};   // optional guide clip
    const VSVideoInfo * vi {};

    int bits {}, elem_bytes {};
    bool has_ref {};

    int ref_mode {};        // 0 luma, 1 chroma, 2 yuv, 3 rgb
    int channels {};        // processed channel count (1/2/3)
    int plane0 {};          // first processed VS plane
    bool process[3] { true, true, true };

    int d {}, a {}, s {}, wmode {};
    float h_param {}, wref_param {};

    int width {}, height {};   // processed lattice dims (chroma-subsampled for UV)
    int stride {};             // plane stride in elements
    int64_t npix {};
    int pad {};                // padded-window margin (= a)
    int pstride {};            // padded tile pitch
    int ph {};                 // padded tile height
    int64_t tile_elems {};
    int clips {};              // 1, or 2 with rclip
    int guide_off {};          // window tile offset of the guide half
    int qb {};
    uint32_t pack {1};         // sweep rounds: entries per weight+acc round
    int slots {};              // u4a ring slots = ring_base * pack

    // create()-time q-sweep tables (stride-8 rows), shared by all recordings
    std::vector<int> wq_host;
    std::vector<int> aq_host;
    std::vector<Variant> variants;

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkPipeline compose_pipeline {};
    VkPipeline weight_pipeline {};
    VkPipeline acc_pipeline {};
    GpuBuffer tables_wq {};
    GpuBuffer tables_aq {};

    // VSFEEL_NLMEANS_TIMING=1: per-frame host stage split (acquire/record/
    // submit). The host side is only those three under the API, but the split
    // still says whether the frame is host- or GPU-bound.
    bool host_timing { false };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_record_ns {}, ht_submit_ns {},
        ht_total_ns {}, ht_n {};

    // VSFEEL_NLMEANS_GPUTRACE=1: one-shot per-batch GPU timestamps.
    bool gputrace { false };
    uint32_t gputrace_frame { 100 };
    VkQueryPool ts_query {};
    GpuBuffer ts_buf {};
    uint64_t * ts_map {};
    std::atomic<uint32_t> ts_armed {};

    ~NLMeansData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[nlmeans-timing] frames=%.0f per-frame us: acquire=%7.1f "
                "record=%7.1f submit=%7.1f total=%7.1f\n",
                n, ht_acquire_ns.load() / 1000.0 / n,
                ht_record_ns.load() / 1000.0 / n,
                ht_submit_ns.load() / 1000.0 / n, ht_total_ns.load() / 1000.0 / n);
        }
        if (!gpu) {
            return;
        }
        // The pool drains every submission it made before it returns, so the
        // pipelines, layouts and tables below are safe to destroy afterwards.
        if (pool_owned) {
            gpu->api->freeGPUExecPool(pool_owned);
            pool_owned = nullptr;
        }
        VkDevice dev = gpu->device;
        if (ts_query) {
            gpu->vk->vkDestroyQueryPool(dev, ts_query, nullptr);
        }
        gpu_destroy_buffer(*gpu, ts_buf);
        gpu_destroy_buffer(*gpu, tables_wq);
        gpu_destroy_buffer(*gpu, tables_aq);
        if (acc_pipeline) {
            gpu->vk->vkDestroyPipeline(dev, acc_pipeline, nullptr);
        }
        if (weight_pipeline) {
            gpu->vk->vkDestroyPipeline(dev, weight_pipeline, nullptr);
        }
        if (compose_pipeline) {
            gpu->vk->vkDestroyPipeline(dev, compose_pipeline, nullptr);
        }
        if (pipeline_layout) {
            gpu->vk->vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            gpu->vk->vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
    }

    // Set once the pool exists; declared here so the destructor can drain it.
    VSGPUExecPool * pool_owned {};
};

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

static std::variant<VkPipeline, std::string> create_pipeline(
    const GPUDevice & gpu, VkPipelineLayout layout,
    const uint32_t * code, size_t code_size, const NLMeansSpecData & spec) {

    // explicit wave32, same knob the legacy build used (measured neutral here)
    const uint32_t subgroup_size = gpu.has_subgroup_size(32) ? 32 : 0;
    return gpu_create_pipeline(gpu, code, code_size, layout, spec_entries.data(),
        &spec, static_cast<uint32_t>(spec_entries.size()), sizeof(spec),
        "nlmeans", subgroup_size);
}

// Device address of a frame plane buffer, for the per-frame address table.
static VkDeviceAddress plane_address(const GPUDevice & gpu, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return gpu.vk->vkGetBufferDeviceAddress(gpu.device, &info);
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static const VSFrame * nlmeans_gpu_frame(
    NLMeansData * d, int n, VSFrameContext * frameCtx, VSCore * core,
    const VSAPI * vsapi) {

    const int C = d->channels;
    const int L = 2 * d->d + 1;
    const int plane0 = d->plane0;
    const int numPlanes = d->vi->format.numPlanes;
    const int nf = d->vi->numFrames;

    // frames[l] is the frame at clamp(n - d + l, 0, nf-1). The sweep variant
    // only ever reads layers [d-m, d+m], so the clamped ends are never touched
    // by a kernel; carrying them keeps the table full and the indexing simple.
    std::vector<const VSFrame *> frames(L);
    std::vector<const VSFrame *> rframes(d->has_ref ? L : 0);
    for (int l = 0; l < L; ++l) {
        const int idx = std::clamp(n - d->d + l, 0, nf - 1);
        frames[l] = vsapi->getFrameFilter(idx, d->node, frameCtx);
        if (d->has_ref) {
            rframes[l] = vsapi->getFrameFilter(idx, d->ref_node, frameCtx);
        }
    }
    const VSFrame * center = frames[d->d];

    bool any_process = false, all_process = true;
    for (int p = 0; p < numPlanes; ++p) {
        any_process |= d->process[p];
        all_process &= d->process[p];
    }

    auto cleanup_frames = [&] {
        for (int l = 0; l < L; ++l) {
            vsapi->freeFrame(frames[l]);
            if (d->has_ref) {
                vsapi->freeFrame(rframes[l]);
            }
        }
    };

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
        vsfeel_trace_error("NLMeans", n, "failed to allocate the output frame",
                           d->gpu.get());
        vsapi->setFilterError("NLMeans: failed to allocate the output frame", frameCtx);
        cleanup_frames();
        return nullptr;
    }

    // Nothing to run: every plane shares from the center frame, so the frame is
    // already complete and an empty submission would only cost a round trip.
    if (!any_process) {
        cleanup_frames();
        return dst;
    }

    auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};
    vsfeel_trace_frame_begin();
    vsfeel_trace_mark("acquire");

    char errbuf[512] {};
    VSGPUExecContext * ctx = d->gpu->api->gpuExecAcquire(d->pool_owned, errbuf, sizeof(errbuf));
    auto fail = [&](const std::string & message) -> const VSFrame * {
        if (ctx) {
            d->gpu->api->gpuExecAbandon(ctx);
            ctx = nullptr;
        }
        vsfeel_trace_error("NLMeans", n, message, d->gpu.get());
        vsapi->setFilterError(("NLMeans: " + message).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        cleanup_frames();
        return nullptr;
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    vsfeel_trace_mark("record");
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);

    const bool gputrace = d->gputrace && n == static_cast<int>(d->gputrace_frame) &&
        d->ts_armed.exchange(1) == 0;
    uint32_t ts_used = 0;
    if (gputrace) {
        d->gpu->vk->vkCmdResetQueryPool(cmd, d->ts_query, 0, NLMEANS_TS_MAX);
        d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            d->ts_query, 0);
        ts_used = NLMEANS_TS_RESERVED;
    }

    // Per-frame scratch, handed to the context and recycled when the
    // submission retires. The weight ring dominates and is sized for the
    // 64 MiB ring budget below.
    GpuBuffer u4a {}, u2 {}, u5 {};
    if (auto e = gpu_frame_buffer(*d->gpu, core, ctx,
            static_cast<VkDeviceSize>(d->npix) * d->slots * sizeof(uint16_t), u4a);
        !e.empty()) {
        return fail("weight ring: " + e);
    }
    if (auto e = gpu_frame_buffer(*d->gpu, core, ctx,
            static_cast<VkDeviceSize>(d->npix) * (C + 1) * sizeof(float), u2);
        !e.empty()) {
        return fail("accumulator buffer: " + e);
    }
    if (auto e = gpu_frame_buffer(*d->gpu, core, ctx,
            static_cast<VkDeviceSize>(d->npix) * sizeof(float), u5);
        !e.empty()) {
        return fail("max-weight buffer: " + e);
    }

    // The zero-padded temporal window the sweep reads: clips*C*L tiles, one per
    // (clip, channel, temporal layer). Rebuilt every frame by the compose pass.
    GpuBuffer window {};
    if (auto e = gpu_frame_buffer(*d->gpu, core, ctx,
            static_cast<VkDeviceSize>(d->clips) * C * L * d->tile_elems *
                d->elem_bytes, window);
        !e.empty()) {
        return fail("temporal window: " + e);
    }

    // Address table: [0, C*L) source planes, [C*L, 2*C*L) guide planes.
    GpuBuffer addr {};
    if (auto e = gpu_make_buffer(*d->gpu, core,
            static_cast<VkDeviceSize>(2) * C * L * sizeof(VkDeviceAddress), addr,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        !e.empty()) {
        return fail("address table: " + e);
    }
    d->gpu->api->gpuExecUsesBuffer(ctx, addr.handle);
    auto * table = static_cast<VkDeviceAddress *>(addr.mapped);
    if (table == nullptr) {
        return fail("address table is not host visible");
    }
    for (int c = 0; c < C; ++c) {
        for (int l = 0; l < L; ++l) {
            VSVulkanPlaneInfo sp {};
            if (d->gpu->api->getGPUPlane(frames[l], plane0 + c, &sp)) {
                return fail("source plane " + std::to_string(plane0 + c) +
                            " is not GPU resident");
            }
            table[c * L + l] = plane_address(*d->gpu, sp.buffer);
            if (d->has_ref) {
                VSVulkanPlaneInfo gp {};
                if (d->gpu->api->getGPUPlane(rframes[l], plane0 + c, &gp)) {
                    return fail("guide plane " + std::to_string(plane0 + c) +
                                " is not GPU resident");
                }
                table[C * L + c * L + l] = plane_address(*d->gpu, gp.buffer);
            } else {
                table[C * L + c * L + l] = table[c * L + l];
            }
        }
    }
    // The table may have landed in the host-visible VRAM BAR, whose mapping is
    // write-combined: without this the GPU can read a stale pointer, or a
    // half-written one, before the CPU store buffer drains.
    _mm_sfence();

    VkBuffer dst_buf[3] {};
    for (int c = 0; c < C; ++c) {
        VSVulkanPlaneInfo dp {};
        if (d->gpu->api->getGPUPlane(dst, plane0 + c, &dp)) {
            return fail("output plane " + std::to_string(plane0 + c) +
                        " is not GPU resident");
        }
        dst_buf[c] = dp.buffer;
    }
    for (int c = C; c < 3; ++c) {
        dst_buf[c] = dst_buf[0];
    }

    const VkBuffer buffers[10] {
        addr.buffer, window.buffer,
        dst_buf[0], dst_buf[1], dst_buf[2],
        d->tables_wq.buffer, d->tables_aq.buffer,
        u4a.buffer, u2.buffer, u5.buffer
    };
    gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, buffers, 10);

    // Compose: copy every (clip, channel, layer) plane into its padded tile.
    // The sweep then reads one buffer with 32-bit offsets.
    vsfeel_trace_mark("compose");
    {
        const uint32_t cgx = static_cast<uint32_t>(
            (d->pstride + BLK_X - 1) / BLK_X);
        const uint32_t cgy = static_cast<uint32_t>(
            (d->ph + BLK_Y - 1) / BLK_Y);
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d->compose_pipeline);
        d->gpu->vk->vkCmdDispatch(cmd, cgx, cgy,
            static_cast<uint32_t>(d->clips) * C * L);
        gpu_barrier(*d->gpu, cmd);
    }

    const uint32_t gx = static_cast<uint32_t>((d->width + BLK_X - 1) / BLK_X);
    const uint32_t gy_w = static_cast<uint32_t>(
        (d->height + VRT_RESULT * BLK_Y - 1) / (VRT_RESULT * BLK_Y));
    const uint32_t gy = static_cast<uint32_t>((d->height + BLK_Y - 1) / BLK_Y);

    const Variant & v = d->variants[std::min(d->d, n)];

    // Interleaved weight/accumulation rounds: a round's weights overwrite the
    // shared u4a slot ring, so its accumulation MUST complete (barrier) before
    // the next round's weights launch. Rounds pack many sweep entries each.
    uint32_t q0 = 0;
    size_t bi = 0;
    while (q0 < v.q_cnt) {
        const uint32_t nb = std::min<uint32_t>(
            static_cast<uint32_t>(d->qb) * d->pack, v.q_cnt - q0);
        const uint32_t p0 = v.w_boff[bi];
        const uint32_t p1 = v.w_boff[bi + 1];

        if (gputrace && ts_used < NLMEANS_TS_MAX) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                d->ts_query, ts_used++);
        }
        {
            const int32_t w_push[4] {
                static_cast<int32_t>(v.w_base + p0), 0, 0, 0
            };
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->weight_pipeline);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, w_push, sizeof(w_push));
            d->gpu->vk->vkCmdDispatch(cmd, gx, gy_w, p1 - p0);
        }
        gpu_barrier(*d->gpu, cmd);
        if (gputrace && ts_used < NLMEANS_TS_MAX) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->ts_query, ts_used++);
        }

        {
            const int32_t a_push[4] {
                static_cast<int32_t>(v.q_base + q0), static_cast<int32_t>(nb),
                bi == 0 ? 1 : 0,
                q0 + nb >= v.q_cnt ? 1 : 0
            };
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d->acc_pipeline);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, a_push, sizeof(a_push));
            d->gpu->vk->vkCmdDispatch(cmd, gx, gy, 1);
        }
        gpu_barrier(*d->gpu, cmd);
        if (gputrace && ts_used < NLMEANS_TS_MAX) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->ts_query, ts_used++);
        }

        q0 += nb;
        ++bi;
    }
    if (gputrace) {
        d->gpu->vk->vkCmdCopyQueryPoolResults(cmd, d->ts_query, 0, ts_used,
            d->ts_buf.buffer, 0, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }
    auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point {};

    // The pool turns the source planes' producer pairs into device-side waits
    // and publishes the output's producers; frames stay alive until the
    // submission completes. Clamped layers map several entries to one frame, so
    // identical frames are declared once.
    for (int l = 0; l < L; ++l) {
        bool dup = false;
        for (int u = 0; u < l; ++u) {
            dup |= frames[u] == frames[l];
        }
        if (!dup) {
            d->gpu->api->gpuExecReadsFrame(ctx, frames[l]);
        }
        if (d->has_ref) {
            dup = false;
            for (int u = 0; u < l; ++u) {
                dup |= rframes[u] == rframes[l];
            }
            if (!dup) {
                d->gpu->api->gpuExecReadsFrame(ctx, rframes[l]);
            }
        }
    }
    for (int c = 0; c < C; ++c) {
        d->gpu->api->gpuExecWritesPlane(ctx, dst, plane0 + c);
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
        char perr[256] {};
        if (d->gpu->api->gpuExecWaitValue(d->pool_owned, signaled, perr, sizeof(perr)) ==
            gdDrained) {
            const double period = d->gpu->limits.timestampPeriod;
            const auto us = [period](uint64_t a, uint64_t b) {
                return static_cast<double>(b - a) * period / 1000.0;
            };
            const uint64_t * ts = d->ts_map;
            double sum_w = 0.0, sum_a = 0.0, first_w = 0.0, first_a = 0.0;
            uint32_t nw = 0;
            uint64_t prev = ts[0];
            for (uint32_t s = NLMEANS_TS_RESERVED; s + 1 < ts_used; s += 2) {
                const double w = us(prev, ts[s]);
                const double a = us(ts[s], ts[s + 1]);
                if (s == NLMEANS_TS_RESERVED) { first_w = w; first_a = a; }
                else { sum_w += w; sum_a += a; ++nw; }
                prev = ts[s + 1];
            }
            fprintf(stderr,
                "[nlmeans-gpu] n=%d batches=%u total=%.1fus first w=%.1f a=%.1f "
                "rest_avg w=%.1f a=%.1f\n",
                n, (ts_used - NLMEANS_TS_RESERVED) / 2, us(ts[0], prev),
                first_w, first_a, nw ? sum_w / nw : 0.0, nw ? sum_a / nw : 0.0);
        } else {
            fprintf(stderr, "[nlmeans-gpu] probe wait failed: %s\n", perr);
        }
    }

    cleanup_frames();
    return dst;
}

static const VSFrame *VS_CC NLMeansGetFrame(
    int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    NLMeansData * d = static_cast<NLMeansData *>(instanceData);

    if (activationReason == arInitial) {
        const int L = 2 * d->d + 1;
        for (int l = 0; l < L; ++l) {
            const int idx = std::clamp(n - d->d + l, 0, d->vi->numFrames - 1);
            vsapi->requestFrameFilter(idx, d->node, frameCtx);
            if (d->has_ref) {
                vsapi->requestFrameFilter(idx, d->ref_node, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    return nlmeans_gpu_frame(d, n, frameCtx, core, vsapi);
}

static void VS_CC NLMeansFree(
    void *instanceData, [[maybe_unused]] VSCore *core, const VSAPI *vsapi) {

    NLMeansData * d = static_cast<NLMeansData *>(instanceData);
    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    vsapi->freeNode(d->node);
    delete d;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC NLMeansCreate(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<NLMeansData>() };

    d->host_timing = vsfeel_debug_probe("VSFEEL_NLMEANS_TIMING");

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("NLMeans", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("NLMeans: " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    d->ref_node = vsapi->mapGetNode(in, "rclip", 0, &error);
    d->has_ref = d->ref_node != nullptr;
    if (d->has_ref) {
        const VSVideoInfo * rvi = vsapi->getVideoInfo(d->ref_node);
        if (!vsh::isSameVideoInfo(rvi, d->vi) ||
            rvi->numFrames != d->vi->numFrames ||
            rvi->width != d->vi->width || rvi->height != d->vi->height) {
            return set_error("'rclip' must match the source clip's format, "
                             "dimensions and frame count.");
        }
    }

    const auto & fmt = d->vi->format;
    if (!vsh::isConstantVideoFormat(d->vi) ||
        (fmt.sampleType == stInteger && fmt.bitsPerSample != 16) ||
        (fmt.sampleType == stFloat && fmt.bitsPerSample != 32)) {
        return set_error("input bitdepth must be 16 (integer) or 32 (float).");
    }
    d->bits = fmt.bitsPerSample;
    d->elem_bytes = fmt.bytesPerSample;

    if (d->vi->width <= 0 || d->vi->height <= 0) {
        return set_error("clip must have constant dimensions.");
    }
    if (d->vi->width > 8192 || d->vi->height > 8192) {
        return set_error("8192x8192 is the highest supported resolution.");
    }

    enum { REF_LUMA = 0, REF_CHROMA = 1, REF_YUV = 2, REF_RGB = 3 };

    int dd = vsh::int64ToIntS(vsapi->mapGetInt(in, "d", 0, &error));
    if (error) {
        dd = 1;
    }
    if (dd < 0 || dd > 16) {
        return set_error("d must be 0..16.");
    }

    int aa = vsh::int64ToIntS(vsapi->mapGetInt(in, "a", 0, &error));
    if (error) {
        aa = 2;
    }
    if (aa < 1 || aa > 64) {
        return set_error("a must be 1..64.");
    }

    int ss = vsh::int64ToIntS(vsapi->mapGetInt(in, "s", 0, &error));
    if (error) {
        ss = 4;
    }
    if (ss < 0 || ss > 8) {
        return set_error("s must be 0..8.");
    }

    d->h_param = static_cast<float>(vsapi->mapGetFloat(in, "h", 0, &error));
    if (error) {
        d->h_param = 1.2f;
    }
    if (!(d->h_param > 0.0f)) {
        return set_error("h must be > 0.");
    }

    d->wmode = vsh::int64ToIntS(vsapi->mapGetInt(in, "wmode", 0, &error));
    if (error) {
        d->wmode = 0;
    }
    if (d->wmode < 0 || d->wmode > 3) {
        return set_error("wmode must be 0..3.");
    }

    d->wref_param = static_cast<float>(vsapi->mapGetFloat(in, "wref", 0, &error));
    if (error) {
        d->wref_param = 1.0f;
    }
    if (!std::isfinite(d->wref_param) || d->wref_param < 0.0f) {
        return set_error("wref must be >= 0.");
    }

    // num_streams is a registered no-op: in-flight depth is the core's
    // (exec pool ring) call, so the argument is accepted and never read.

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    const char * chstr = vsapi->mapGetData(in, "channels", 0, &error);
    if (error || !chstr) {
        chstr = "auto";
    }
    auto eq = [](const char * x, const char * y) {
        return strcasecmp(x, y) == 0;
    };

    switch (fmt.colorFamily) {
        case cfGray:
            if (!(eq(chstr, "Y") || eq(chstr, "auto"))) {
                return set_error("'channels' must be 'Y' with Gray.");
            }
            d->ref_mode = REF_LUMA;
            d->channels = 1;
            d->plane0 = 0;
            break;
        case cfYUV:
            if (eq(chstr, "YUV")) {
                if (fmt.subSamplingW != 0 || fmt.subSamplingH != 0) {
                    return set_error("'channels'='YUV' requires 4:4:4.");
                }
                d->ref_mode = REF_YUV;
                d->channels = 3;
                d->plane0 = 0;
            } else if (eq(chstr, "Y") || eq(chstr, "auto")) {
                d->ref_mode = REF_LUMA;
                d->channels = 1;
                d->plane0 = 0;
            } else if (eq(chstr, "UV")) {
                d->ref_mode = REF_CHROMA;
                d->channels = 2;
                d->plane0 = 1;
            } else {
                return set_error("'channels' must be 'YUV', 'Y' or 'UV' with YUV.");
            }
            break;
        case cfRGB:
            if (!(eq(chstr, "RGB") || eq(chstr, "auto"))) {
                return set_error("'channels' must be 'RGB' with RGB.");
            }
            d->ref_mode = REF_RGB;
            d->channels = 3;
            d->plane0 = 0;
            break;
        default:
            return set_error("unsupported color family.");
    }

    const int ssw = fmt.subSamplingW;
    const int ssh = fmt.subSamplingH;
    d->width = (d->ref_mode == REF_CHROMA) ? d->vi->width >> ssw : d->vi->width;
    d->height = (d->ref_mode == REF_CHROMA) ? d->vi->height >> ssh : d->vi->height;

    if (2 * aa + 1 > d->width || 2 * aa + 1 > d->height) {
        return set_error("research window (2*a+1) larger than the frame.");
    }

    d->d = dd;
    d->a = aa;
    d->s = ss;

    for (int p = 0; p < 3; ++p) {
        d->process[p] = p >= d->plane0 && p < d->plane0 + d->channels;
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
    // CPU frame's stride, so it is read off a scratch frame here.
    {
        VSFrame * probe = vsapi->newVideoFrame(&fmt, d->vi->width, d->vi->height,
                                               nullptr, core);
        if (probe == nullptr) {
            return set_error("could not allocate a probe frame to read the plane stride");
        }
        d->stride = static_cast<int>(vsapi->getStride(probe, d->plane0) / d->elem_bytes);
        vsapi->freeFrame(probe);
    }
    d->npix = static_cast<int64_t>(d->stride) * d->height;
    d->qb = d->npix <= static_cast<int64_t>(1920) * 1152 ? 8 : 4;

    // Zero-padded temporal window geometry. PAD = a makes every OOB candidate
    // read land in the zero margin, which is the reference's zero border.
    d->pad = d->a;
    d->pstride = (d->width + 2 * d->pad + 7) & ~7;
    d->ph = d->height + 2 * d->pad;
    d->tile_elems = static_cast<int64_t>(d->pstride) * d->ph;
    d->clips = d->has_ref ? 2 : 1;
    d->guide_off = d->has_ref ? d->channels * (2 * dd + 1) : 0;

    // Pack as many sweep entries into one weight+accumulation round as the u4a
    // plane-ring budget allows: the kernels' arithmetic is cheap next to the
    // fixed per-dispatch/barrier drain. Keep the ring cache-friendly (~64 MiB):
    // measured optimum on the target GPU, larger rings stream weights through
    // DRAM between the weight and accumulation launches.
    {
        const int64_t ring_base_slots = (dd == 0) ? d->qb : 2 * static_cast<int64_t>(d->qb);
        const int64_t bytes_per_pack = ring_base_slots * d->npix * sizeof(uint16_t);
        constexpr int64_t U4A_RING_BUDGET = 64LL << 20;
        int64_t pack = U4A_RING_BUDGET / std::max<int64_t>(bytes_per_pack, 1);
        pack = std::clamp<int64_t>(pack, 1, 16384);
        const int pack_env = env_int("VSFEEL_NLMEANS_PACK", env_int("NLMEANS_PACK", 0));
        if (pack_env > 0) {
            pack = std::clamp<int64_t>(pack_env, 1, 16384);
        }
        d->pack = static_cast<uint32_t>(pack);
    }
    d->slots = ((dd == 0) ? d->qb : 2 * d->qb) * static_cast<int>(d->pack);

    // int32 addressing bound of the device-side layouts (the reference falls
    // back to 64-bit indices here; we reject instead)
    {
        const int64_t window_elems = d->clips * static_cast<int64_t>(d->channels) *
            (2 * dd + 1) * d->tile_elems;
        const int64_t idx_max = std::max({
            d->npix * static_cast<int64_t>(d->slots),
            d->npix * static_cast<int64_t>(d->channels),
            window_elems });
        if (idx_max >= (INT64_C(1) << 31)) {
            return set_error("resolution/temporal radius combination exceeds "
                             "the addressable range.");
        }
    }

    // Run-merged sweep tables, equivalent to the reference q-batched sweep:
    // for each reachable boundary count m the half-space of displacements
    // (exploiting weight(p,p+q)==weight(p,p-q)) in k/j/i order, with
    // consecutive-i displacements sharing (qy,qz) merged into RUN GROUPS of at
    // most qb entries (so the u4a slot ring still covers one whole group).
    {
        const int64_t spt_side = 2LL * aa + 1;
        const int64_t spt_area = spt_side * spt_side;
        d->variants.resize(dd + 1);
        const uint32_t batch = static_cast<uint32_t>(d->qb) * d->pack;
        for (int mm = 0; mm <= dd; ++mm) {
            Variant & v = d->variants[mm];
            v.w_base = static_cast<uint32_t>(d->wq_host.size() / 8);
            v.q_base = static_cast<uint32_t>(d->aq_host.size() / 8);
            uint32_t q_idx = 0;
            for (int kk = -mm; kk <= 0; ++kk) {
                for (int j = -aa; j <= aa; ++j) {
                    for (int i = -aa; i <= aa; ++i) {
                        if (static_cast<int64_t>(kk) * spt_area +
                                static_cast<int64_t>(j) * spt_side + i < 0) {
                            const uint32_t b_local = q_idx % batch;
                            if (b_local == 0) {
                                v.w_boff.push_back(static_cast<uint32_t>(
                                    d->wq_host.size() / 8 - v.w_base));
                            }
                            const int slot_c = (dd == 0)
                                ? static_cast<int>(b_local)
                                : 2 * static_cast<int>(b_local);
                            const int slot_m = (kk != 0) ? slot_c + 1 : slot_c;
                            const int wrow_c[8] { dd, i, j, kk, slot_c, 0, 0, 0 };
                            d->wq_host.insert(d->wq_host.end(), std::begin(wrow_c),
                                std::end(wrow_c));
                            if (kk != 0) {
                                const int wrow_m[8] { dd - kk, i, j, kk, slot_m, 0, 0, 0 };
                                d->wq_host.insert(d->wq_host.end(), std::begin(wrow_m),
                                    std::end(wrow_m));
                            }
                            const int arow[8] { i, j, kk, slot_c, slot_m, 0, 0, 0 };
                            d->aq_host.insert(d->aq_host.end(), std::begin(arow),
                                std::end(arow));
                            ++q_idx;
                        }
                    }
                }
            }
            v.w_boff.push_back(static_cast<uint32_t>(
                d->wq_host.size() / 8 - v.w_base));
            v.q_cnt = q_idx;
        }
    }

    d->gputrace = vsfeel_debug_probe("VSFEEL_NLMEANS_GPUTRACE") &&
        vsfeel_probe_timestamps(*d->gpu, "NLMeans");
    if (d->gputrace) {
        d->gputrace_frame = static_cast<uint32_t>(
            env_int("VSFEEL_NLMEANS_GPUTRACE", 100));
        VkQueryPoolCreateInfo qp_info {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = NLMEANS_TS_MAX
        };
        if (d->gpu->vk->vkCreateQueryPool(d->gpu->device, &qp_info, nullptr,
                &d->ts_query) != VK_SUCCESS) {
            d->ts_query = VK_NULL_HANDLE;
            d->gputrace = false;
        }
    }
    if (d->gputrace) {
        auto e = gpu_make_buffer(*d->gpu, core,
            NLMEANS_TS_MAX * sizeof(uint64_t), d->ts_buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!e.empty() || d->ts_buf.mapped == nullptr) {
            d->gputrace = false;
        } else {
            d->ts_map = static_cast<uint64_t *>(d->ts_buf.mapped);
        }
    }

    // H2_INV_NORM = 255^2 / (3*h*h*(2*s+1)^2), evaluated left-to-right like
    // the reference macro expansion
    const float nlm_norm = 255.0f * 255.0f;
    const float s_size = static_cast<float>((2 * ss + 1) * (2 * ss + 1));
    float denom = 3.0f * d->h_param;
    denom *= d->h_param;
    denom *= s_size;
    const NLMeansSpecData spec {
        .width = d->width,
        .height = d->height,
        .stride = d->stride,
        .pstride = d->pstride,
        .ph = d->ph,
        .pad = d->pad,
        .s = d->s,
        .d = d->d,
        .ref = d->ref_mode,
        .channels = d->channels,
        .wmode = d->wmode,
        .wref = d->wref_param,
        .h2_inv_norm = nlm_norm / denom,
        .guide_off = d->guide_off
    };

    {
        const auto result = gpu_push_set_layout(*d->gpu, 10);
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

    // Sweep tables in two small buffers, host visible and coherent so a plain
    // memcpy lands; the device-local preference puts them in VRAM when the
    // device has a ReBAR window, and in system RAM otherwise.
    {
        const VkDeviceSize wq_bytes =
            static_cast<VkDeviceSize>(d->wq_host.size()) * sizeof(int32_t);
        auto e = gpu_make_buffer(*d->gpu, core, wq_bytes, d->tables_wq,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!e.empty()) {
            return set_error("wq table: " + e);
        }
        if (d->tables_wq.mapped == nullptr) {
            return set_error("wq table is not host visible");
        }
        memcpy(d->tables_wq.mapped, d->wq_host.data(), wq_bytes);

        const VkDeviceSize aq_bytes =
            static_cast<VkDeviceSize>(d->aq_host.size()) * sizeof(int32_t);
        e = gpu_make_buffer(*d->gpu, core, aq_bytes, d->tables_aq,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!e.empty()) {
            return set_error("aq table: " + e);
        }
        if (d->tables_aq.mapped == nullptr) {
            return set_error("aq table is not host visible");
        }
        memcpy(d->tables_aq.mapped, d->aq_host.data(), aq_bytes);
        // Same write-combining caveat as the per-frame address table: these
        // tables may live in the host-visible VRAM BAR, and the first
        // submission must not read a tail that has not drained yet.
        _mm_sfence();
    }

    {
        const uint32_t * compose_code = nullptr;
        const uint32_t * weight_code = nullptr;
        const uint32_t * acc_code = nullptr;
        size_t compose_size = 0, weight_size = 0, acc_size = 0;
        if (d->bits == 16) {
            compose_code = nlmeans_16_compose_spv;
            compose_size = nlmeans_16_compose_spv_size;
            weight_code = nlmeans_16_weight_spv;
            weight_size = nlmeans_16_weight_spv_size;
            acc_code = nlmeans_16_acc_spv;
            acc_size = nlmeans_16_acc_spv_size;
        } else {
            compose_code = nlmeans_32_compose_spv;
            compose_size = nlmeans_32_compose_spv_size;
            weight_code = nlmeans_32_weight_spv;
            weight_size = nlmeans_32_weight_spv_size;
            acc_code = nlmeans_32_acc_spv;
            acc_size = nlmeans_32_acc_spv_size;
        }
        {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                compose_code, compose_size, spec);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->compose_pipeline = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                weight_code, weight_size, spec);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->weight_pipeline = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_pipeline(*d->gpu, d->pipeline_layout,
                acc_code, acc_size, spec);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->acc_pipeline = std::get<VkPipeline>(result);
        }
    }

    {
        char err[512] {};
        d->pool_owned = d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (d->pool_owned == nullptr) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    if (vsfeel_debug_flag("VSFEEL_NLMEANS_VRAM")) {
        const double mib = 1024.0 * 1024.0;
        const double ring_mib = static_cast<double>(d->npix) * d->slots *
            sizeof(uint16_t) / mib;
        const double window_mib = static_cast<double>(d->clips) * d->channels *
            (2 * d->d + 1) * d->tile_elems * d->elem_bytes / mib;
        const double per_frame = ring_mib + window_mib +
            static_cast<double>(d->npix) * (d->channels + 2) *
                sizeof(float) / mib;
        fprintf(stderr, "[nlmeans] %.1f MiB per in-flight frame (ring %.1f MiB, "
                        "%d slots, pack %u; window %.1f MiB, %dx%d tiles)\n",
            per_frame, ring_mib, d->slots, d->pack, window_mib, d->pstride, d->ph);
    }

    NLMeansData * data = d.release();

    // A temporal filter requests frames outside n, which the strict-spatial
    // policy does not permit; only d = 0 is purely spatial.
    const VSRequestPattern policy =
        data->d > 0 ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2] = {
        { data->node, policy },
        { data->ref_node, policy }
    };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "NLMeans", data->vi, NLMeansGetFrame, NLMeansFree,
        fmParallel, ffGPUOutput, deps, data->has_ref ? 2 : 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "NLMeans: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

} // namespace

void vsfeel_register_nlmeans(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "NLMeans",
        "clip:vnode:gpu;"
        "d:int:opt;"
        "a:int:opt;"
        "s:int:opt;"
        "h:float:opt;"
        "wmode:int:opt;"
        "wref:float:opt;"
        "channels:data:opt;"
        "rclip:vnode:gpu:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode:gpu;",
        NLMeansCreate, nullptr, plugin
    );
}
