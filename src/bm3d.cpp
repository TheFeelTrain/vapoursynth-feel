#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <memory>
#include <mutex>
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

namespace {

constexpr int MAX_RADIUS = 4;
// The shader does the search-window arithmetic ((2*range+1)^2, x±range) in
// int32; beyond this, absurd-but-accepted values overflow it.
constexpr int kMaxSearchRange = 8192;
// In-flight frames the private caches are sized for. The core's exec pool owns
// the real pipelining depth; this is the working set the estimate/source rings
// cover, kept at the old two-stream depth for VRAM.
constexpr int kInflightFrames = 2;

struct Bm3dPlane {
    int width {};
    int height {};
    int stride {};
    VkDeviceSize pe {};             // plane extent in floats (h * stride)
    VkPipeline bm3d_pipeline {};
    VkPipeline agg_pipeline {};
    uint32_t bm3d_grid_x {};
    uint32_t bm3d_grid_y {};
    uint32_t agg_grid_x {};
    uint32_t agg_grid_y {};
};

// Per-frame bookkeeping. The exec pool owns the command buffers, the timeline
// and the in-flight gate, so nothing here outlives the getFrame call: a frame's
// window reservations are described by this struct, and what a reader needs
// afterwards lives in the reservation tables (see BM3DData).
struct Bm3dFrame {
    // unique token identifying this frame's cache reservation. A frame index
    // is not a unique holder identity: the scheduler can process the same
    // frame twice at once, and erasing holders by value would then drop both
    // entries at the first release, freeing a slot another reader still uses.
    uint64_t token {};
    int slot0 {};                         // radius 0: this frame's private slot
    std::array<int, 2 * MAX_RADIUS + 1> win_slots {};     // res slot per window position
    std::array<bool, 2 * MAX_RADIUS + 1> win_recompute {};// this frame computes it
    std::array<int, 2 * MAX_RADIUS + 1> win_writer {};    // frame that wrote it (trace)
    // Window positions this frame must compute, deduplicated (a clamped window
    // maps several positions onto one slot and one centre frame).
    std::array<int, 2 * MAX_RADIUS + 1> est_pos {};
    int n_pos {};
    int src_lo {};
    int n_src {};
    std::array<bool, 4 * MAX_RADIUS + 1> upload_new {};   // src frames this frame copies
    std::array<int, 4 * MAX_RADIUS + 1> src_slot {};      // ring slot of each window frame
    std::array<int, 4 * MAX_RADIUS + 1> src_writer {};    // copier of it (trace)
};

// GPU-timing probe (VSFEEL_BM3D_GPUTRACE): one timestamp query pool, shared by
// every frame because the instrumented path holds probe.lock from the
// estimation recording through the host readback. Reading needs the submission
// complete, so an instrumented run is serialized by construction -- it is a
// development tool, never a benchmark configuration.
struct Bm3dProbe {
    VkQueryPool query {};
    std::mutex lock;
    bool enabled {};
};


struct BM3DData {
    VSNode * node;
    VSNode * ref_node {};   // optional basic-estimate clip (final/Wiener pass)
    const VSVideoInfo * vi;

    int radius;
    int tw;                          // 2 * radius + 1
    float sigma;                     // scaled luma sigma
    float sigma_u, sigma_v;
    int block_step, bm_range, ps_num, ps_range;
    bool process;
    bool chroma;
    bool final {};                   // true when a "ref" clip is given
    float extractor;
    bool cas_atomics {};             // aggregate with the CAS kernel (no float32 add atomics)

    std::shared_ptr<GPUDevice> gpu;
    // Only the destructor needs it, and the destructor has no VSAPI argument.
    const VSAPI * vsapi {};
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    std::array<Bm3dPlane, 3> planes {};
    int n_planes {};

    // The core's exec pool: one per instance, on the compute queue. It owns the
    // command buffers, the timeline and the backpressure; the filter records and
    // submits, and hands it the frames and scratch a submission must keep alive.
    VSGPUExecPool * exec {};

    // shared device buffers (VRAM); the src is a ring of src_ring slots
    int src_ring {};             // cache slots for the source window (matches the kernel's SRC_RING)
    int res_cap {};              // cache slots for the per-frame estimate stacks
    VkDeviceSize src_size {};    // src_ring * pe elements (per plane, packed)
    GpuBuffer src;
    GpuBuffer res;

    VkDeviceSize res_size_per_plane {};  // floats per plane in the res buffer
    int nframes {};

    // Per-frame-keyed caches of the res estimate stacks and the source
    // frames. Slots are reserved all-or-nothing for the duration of a frame
    // (shared holds for reads, exclusive for recompute/upload), so no two
    // in-flight frames ever touch the same slot; when the cache cannot hold
    // the working set (e.g. seeking), the acquire blocks like the reference's
    // fused-mode accumulator cache.
    //
    // Cross-frame ordering is a ready flag per slot, not a timeline value: the
    // pool allocates signal values at submit time, so a writer cannot name the
    // value it will signal when it reserves. A reader instead waits until the
    // writer's estimation has been submitted; queue order on the one compute
    // queue then puts the writer's command buffer first, and the reader's
    // leading pipeline barrier carries both the execution and the memory
    // dependency across the two submissions.
    std::vector<int> src_frame {};   // frame index whose data each src slot holds
    std::vector<int> src_writer {};  // frame that reserved each src slot for copying
    std::vector<uint8_t> src_ready {};  // its estimation submission is enqueued
    std::vector<std::vector<uint64_t>> src_holders {};  // reservation tokens
    std::vector<int> res_frame {};   // frame index whose stack each res slot holds
    std::vector<int> res_writer {};  // frame that computed each res slot's content
    std::vector<uint8_t> res_ready {};
    std::vector<std::vector<uint64_t>> res_holders {};  // reservation tokens
    uint64_t next_res_token {1};
    // Radius 0 has no cross-frame sharing, so each in-flight frame takes one of
    // these slots outright for its whole life instead of going through the
    // window cache. Size is the old in-flight depth.
    std::vector<int> r0_free {};
    std::mutex cache_lock;
    std::condition_variable cache_cv;
    Bm3dProbe probe {};

    // Env-gated host-path probe (VSFEEL_BM3D_TIMING=1). The stage split is the
    // prerequisite for any transfer-path change: kernel time alone says nothing
    // about whether the frame is GPU- or host-bound. Reset the clock after every
    // blocking acquire so a wait never leaks into the next stage.
    bool host_timing { false };
    // Cached at creation: the timestamp query pool only exists when the env
    // var was set then, so recording must not be driven by a frame-time getenv.
    bool gpu_trace { false };
    // Trace/dump flags are read once: the frame path used to call getenv
    // sixteen times per frame on the default path.
    bool trace { false };
    bool dump { false };
    // Submit one command buffer per recomputed window position instead of one
    // holding them all (VSFEEL_BM3D_SPLIT=0 restores the single submission).
    bool split_est { true };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_source_ns {}, ht_est_ns {},
        ht_agg_ns {}, ht_release_ns {}, ht_total_ns {}, ht_n {};

    ~BM3DData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[bm3d-timing] frames=%.0f per-frame us: acquire=%7.1f source=%7.1f "
                "est=%7.1f agg=%7.1f release=%7.1f total=%7.1f\n",
                n, ht_acquire_ns.load() / 1000.0 / n, ht_source_ns.load() / 1000.0 / n,
                ht_est_ns.load() / 1000.0 / n, ht_agg_ns.load() / 1000.0 / n,
                ht_release_ns.load() / 1000.0 / n, ht_total_ns.load() / 1000.0 / n);
        }
        if (!gpu) {
            return;
        }
        VkDevice dev = gpu->device;
        // freeGPUExecPool drains every submission this instance made and runs
        // every retention it held -- source frames included -- before it
        // returns, so nothing below can still be in use.
        if (exec) {
            gpu->api->freeGPUExecPool(exec);
            exec = nullptr;
        }
        if (probe.query) {
            gpu->vk->vkDestroyQueryPool(dev, probe.query, nullptr);
        }
        gpu_destroy_buffer(*gpu, res);
        gpu_destroy_buffer(*gpu, src);
        if (pipeline_layout) gpu->vk->vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        if (set_layout) gpu->vk->vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        for (auto & p : planes) {
            if (p.bm3d_pipeline && p.bm3d_pipeline != p.agg_pipeline) {
                gpu->vk->vkDestroyPipeline(dev, p.bm3d_pipeline, nullptr);
            }
            if (p.agg_pipeline) {
                gpu->vk->vkDestroyPipeline(dev, p.agg_pipeline, nullptr);
            }
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

static std::variant<VkPipeline, std::string> create_bm3d_pipeline(
    const GPUDevice & gpu, const Bm3dPlane & plane,
    const BM3DData & d, const uint32_t * code, size_t code_size,
    VkPipelineLayout layout) {

    const float sigma_y = d.sigma;
    struct Spec {
        int32_t width, height, stride;
        float sigma_y;
        int32_t block_step, bm_range, radius, ps_num, ps_range;
        float extractor;
        int32_t nosearch, noestimate, src_ring, final;
    } spec {
        plane.width, plane.height, plane.stride, sigma_y,
        d.block_step, d.bm_range, d.radius, d.ps_num, d.ps_range, d.extractor,
        env_flag("VSFEEL_BM3D_NOSEARCH") ? 1 : 0,
        env_flag("VSFEEL_BM3D_NOESTIMATE") ? 1 : 0,
        d.src_ring,
        d.final ? 1 : 0
    };
    const std::array<VkSpecializationMapEntry, 14> entries {{
        { 0,  0, sizeof(int32_t) },
        { 1,  4, sizeof(int32_t) },
        { 2,  8, sizeof(int32_t) },
        { 3, 12, sizeof(float) },
        { 4, 16, sizeof(int32_t) },
        { 5, 20, sizeof(int32_t) },
        { 6, 24, sizeof(int32_t) },
        { 7, 28, sizeof(int32_t) },
        { 8, 32, sizeof(int32_t) },
        { 9, 36, sizeof(float) },
        { 10, 40, sizeof(int32_t) },
        { 11, 44, sizeof(int32_t) },
        { 12, 48, sizeof(int32_t) },
        { 13, 52, sizeof(int32_t) },
    }};
    // The kernel's 8-lane shuffles keep each aligned 8-lane group inside one
    // subgroup, and 32 is the measured-best width on the target GPU
    // (VSFEEL_BM3D_SUBGROUP=64 forces the wave64 path). The request is checked
    // against the device: subgroup size control is in the core's baseline, but
    // the size itself is not, so a device that cannot provide it is told so
    // here instead of at dispatch.
    const int forced_subgroup = env_int("VSFEEL_BM3D_SUBGROUP", 0);
    const uint32_t subgroup_size = forced_subgroup > 0
        ? static_cast<uint32_t>(forced_subgroup)
        : 32u;
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(), &spec,
        static_cast<uint32_t>(entries.size()), sizeof(spec), "bm3d",
        subgroup_size, /*workgroup_invocations=*/32);
}

static std::variant<VkPipeline, std::string> create_agg_pipeline(
    const GPUDevice & gpu, const Bm3dPlane & plane,
    const BM3DData & d, const uint32_t * code, size_t code_size,
    VkPipelineLayout layout) {

    struct Spec {
        int32_t width, height, stride, tw;
    } spec { plane.width, plane.height, plane.stride, d.tw };
    const std::array<VkSpecializationMapEntry, 4> entries {{
        { 0,  0, sizeof(int32_t) },
        { 1,  4, sizeof(int32_t) },
        { 2,  8, sizeof(int32_t) },
        { 3, 12, sizeof(int32_t) },
    }};
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(), &spec,
        static_cast<uint32_t>(entries.size()), sizeof(spec), "bm3d_agg");
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static int agg_z(int i, int n, int nframes, int radius) {
    return std::min(std::max(2 * radius - i, n - nframes + 1 + radius), n + radius);
}

// Radius 0 has no cross-frame sharing, so a frame takes one slot outright for
// its whole life instead of going through the window cache.
static int take_r0_slot(BM3DData * d) {
    std::unique_lock lock(d->cache_lock);
    d->cache_cv.wait(lock, [&] { return !d->r0_free.empty(); });
    const int slot = d->r0_free.back();
    d->r0_free.pop_back();
    return slot;
}

static void give_r0_slot(BM3DData * d, int slot) {
    std::lock_guard lock(d->cache_lock);
    d->r0_free.push_back(slot);
    d->cache_cv.notify_all();
}

// Reserve the cache slots this frame needs, all-or-nothing: the res slots of
// the temporal window (recomputing the missing stacks) and the src slots of
// the source window (re-uploading the missing frames). Slots held by other
// in-flight frames block until they complete, so no two frames ever touch the
// same slot concurrently; the blocking only happens when the working set
// exceeds the cache (seeks, very out-of-order arrivals) and never holds a
// recording context while waiting.
static void acquire_cache(BM3DData * d, Bm3dFrame & fr, int n) {
    fr.win_slots.fill(-1);
    fr.win_writer.fill(-1);
    fr.win_recompute.fill(false);
    fr.upload_new.fill(false);
    fr.src_slot.fill(-1);
    fr.src_writer.fill(-1);

    if (d->radius == 0) {
        // per-frame slots: private to one in-flight frame, so concurrent
        // out-of-order frames never share a slot
        fr.slot0 = take_r0_slot(d);
        fr.win_slots[0] = fr.slot0;
        fr.win_writer[0] = n;
        fr.win_recompute[0] = true;
        fr.upload_new[0] = true;
        fr.src_slot[0] = fr.slot0;
        fr.src_lo = n;
        fr.n_src = 1;
        return;
    }
    const int r = d->radius;
    const int nf = d->nframes;
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    fr.src_lo = lo;
    fr.n_src = hi - lo + 1;
    std::unique_lock lock(d->cache_lock);
    for (;;) {
        bool ok = true;
        fr.win_slots.fill(-1);
        fr.win_writer.fill(-1);
        fr.win_recompute.fill(false);
        fr.upload_new.fill(false);
        fr.src_slot.fill(-1);
        fr.src_writer.fill(-1);
        // Phase 1: check-only, with no side effects. The failed passes must
        // not leave half-applied reservations behind, or a retry would treat
        // the abandoned slots as cached and never recompute them.
        for (int i = 0; i < d->tw; ++i) {
            const int m = std::clamp(n - r + i, 0, nf - 1);
            const int slot = m % d->res_cap;
            fr.win_slots[i] = slot;
            fr.win_writer[i] = d->res_writer[slot];
            // A clamped window maps several positions onto one slot, so this
            // must be decided from the state *before* phase 2 mutates it: the
            // later positions would otherwise look cached and keep the stale
            // writer of the slot's previous contents as a dependency.
            fr.win_recompute[i] = (d->res_frame[slot] != m);
            if (fr.win_recompute[i] && !d->res_holders[slot].empty()) {
                ok = false;   // slot in use by an in-flight frame
                break;
            }
        }
        if (ok) {
            for (int k = 0; k < fr.n_src; ++k) {
                const int slot = (lo + k) % d->src_ring;
                fr.src_slot[k] = slot;
                fr.src_writer[k] = d->src_writer[slot];
                if (d->src_frame[slot] != lo + k && !d->src_holders[slot].empty()) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            d->cache_cv.wait(lock);
            continue;
        }
        // Phase 2: apply the reservations (the lock is held, so the phase-1
        // checks are still valid).
        fr.token = d->next_res_token++;
        for (int i = 0; i < d->tw; ++i) {
            const int m = std::clamp(n - r + i, 0, nf - 1);
            const int slot = fr.win_slots[i];
            if (fr.win_recompute[i]) {
                if (d->res_frame[slot] != m) {   // first position mapping here
                    d->res_frame[slot] = m;
                    d->res_writer[slot] = n;
                    d->res_ready[slot] = 0;
                }
                // every position that maps here drops the previous writer's
                // dependency: this frame overwrites the slot's contents
                fr.win_writer[i] = -1;
            }
            d->res_holders[slot].push_back(fr.token);
        }
        for (int k = 0; k < fr.n_src; ++k) {
            const int slot = fr.src_slot[k];
            if (d->src_frame[slot] != lo + k) {
                d->src_frame[slot] = lo + k;
                d->src_writer[slot] = n;
                d->src_ready[slot] = 0;
                fr.upload_new[k] = true;
            }
            d->src_holders[slot].push_back(fr.token);
        }
        return;
    }
}

// The slots this frame wrote are submitted, so a reader waiting on them may
// record and submit: submission order on the one compute queue plus the
// reader's leading barrier is the whole cross-frame handoff. Called on the
// error path too, where the frame will never submit -- a reader must proceed
// (and fail on its own) rather than block forever on a signal that is never
// coming; that is the same trade the legacy host-signalled timeline made.
static void publish_est_submitted(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        return;   // radius 0 slots are private to one frame
    }
    std::lock_guard lock(d->cache_lock);
    for (int i = 0; i < d->tw; ++i) {
        if (fr.win_recompute[i]) {
            d->res_ready[fr.win_slots[i]] = 1;
        }
    }
    for (int k = 0; k < fr.n_src; ++k) {
        if (fr.upload_new[k]) {
            d->src_ready[fr.src_slot[k]] = 1;
        }
    }
    d->cache_cv.notify_all();
}

// Wait host side until every source slot this frame reads has been submitted by
// its copier. The frame holds those slots, so the writer cannot be re-reserved
// underneath it.
static void wait_src_submitted(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        return;
    }
    std::unique_lock lock(d->cache_lock);
    d->cache_cv.wait(lock, [&] {
        for (int k = 0; k < fr.n_src; ++k) {
            if (!fr.upload_new[k] && !d->src_ready[fr.src_slot[k]]) {
                return false;
            }
        }
        return true;
    });
}

// Same for the estimate stacks the aggregation reads.
static void wait_res_submitted(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        return;
    }
    std::unique_lock lock(d->cache_lock);
    d->cache_cv.wait(lock, [&] {
        for (int i = 0; i < d->tw; ++i) {
            if (!fr.win_recompute[i] && !d->res_ready[fr.win_slots[i]]) {
                return false;
            }
        }
        return true;
    });
}

// Release the cache reservations after this frame's aggregation has been
// submitted: queue order already makes a later recompute run after that
// aggregation, and the later frame's leading barrier completes the ordering.
static void release_cache(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        give_r0_slot(d, fr.slot0);
        return;
    }
    std::lock_guard lock(d->cache_lock);
    for (int i = 0; i < d->tw; ++i) {
        auto & h = d->res_holders[fr.win_slots[i]];
        h.erase(std::remove(h.begin(), h.end(), fr.token), h.end());
    }
    for (int k = 0; k < fr.n_src; ++k) {
        auto & h = d->src_holders[fr.src_slot[k]];
        h.erase(std::remove(h.begin(), h.end(), fr.token), h.end());
    }
    d->cache_cv.notify_all();
}

// Window positions whose estimate this frame must compute. A clamped window
// collapses several positions onto one slot and one centre frame, i.e.
// identical dispatches whose fills wipe each other, so only one is kept.
static void collect_est_positions(BM3DData * d, Bm3dFrame & fr, int n) {
    fr.n_pos = 0;
    if (d->radius == 0) {
        fr.est_pos[fr.n_pos++] = 0;
        return;
    }
    for (int i = 0; i < d->tw; ++i) {
        if (!fr.win_recompute[i]) {
            continue;
        }
        const int m_i = std::clamp(n - d->radius + i, 0, d->nframes - 1);
        const int slot = fr.win_slots[i];
        bool duplicate_later = false;
        for (int j = i + 1; j < d->tw && !duplicate_later; ++j) {
            duplicate_later = fr.win_recompute[j] && fr.win_slots[j] == slot &&
                std::clamp(n - d->radius + j, 0, d->nframes - 1) == m_i;
        }
        if (!duplicate_later) {
            fr.est_pos[fr.n_pos++] = i;
        }
    }
}

// The three buffers both kernels address, in the order the shader declares
// them: estimate stacks, source ring, destination plane.
static void bm3d_bind(const GPUDevice & gpu, VkCommandBuffer cmd,
                      VkPipelineLayout layout, VkBuffer src, VkBuffer res,
                      VkBuffer dst) {
    const VkBuffer bufs[3] { res, src, dst };
    gpu_push_buffers(gpu, cmd, layout, bufs, 3);
}

// A whole-command barrier: every write made visible to every later read. Used
// where the producer and consumer are different dispatches or different
// submissions, which is what the cache handoff needs.
static void bm3d_full_barrier(const GPUDevice & gpu, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    gpu.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

// Zero-fill one result slot and dispatch the estimation kernel for it. Separate
// so each recomputed position can be recorded into its own command buffer: a
// frame's estimation is the search run over every window position it is missing,
// and on a slow card the total can run past the driver's watchdog window.
static void record_est_position(BM3DData * d, const Bm3dFrame & fr, VkCommandBuffer cmd,
                                int n, int i, VkBuffer dst_plane) {
    const int r = d->radius;
    const int nf = d->nframes;
    const int slot = fr.win_slots[i];
    const int m_i = std::clamp(n - r + i, 0, nf - 1);
    if (d->dump) fprintf(stderr, "[d] n=%d computes slot %d for frame %d\n", n, slot, m_i);
    for (int plane = 0; plane < d->n_planes; ++plane) {
        const auto & p = d->planes[plane];
        const VkDeviceSize pe = p.pe;

        const VkDeviceSize res_off = (static_cast<VkDeviceSize>(slot) * d->tw * 2 * pe +
            static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
        d->gpu->vk->vkCmdFillBuffer(cmd, d->res.buffer, res_off * 4, d->tw * 2 * pe * 4, 0);

        // the zero-fill must be visible to the atomic accumulation that
        // follows it in the next dispatch
        bm3d_full_barrier(*d->gpu, cmd);

        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.bm3d_pipeline);
        bm3d_bind(*d->gpu, cmd, d->pipeline_layout, d->src.buffer, d->res.buffer, dst_plane);
        {
            const int32_t pushes[4] {
                static_cast<int32_t>(res_off),
                m_i,
                nf,
                static_cast<int32_t>((r == 0) ? fr.slot0 : 0)
            };
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, pushes, sizeof(pushes));
        }
        d->gpu->vk->vkCmdDispatch(cmd, p.bm3d_grid_x, p.bm3d_grid_y, 1);
    }
}

// The input planes one source frame contributes to the ring copy: the frame
// being denoised, plus the basic-estimate clip in final mode.
struct Bm3dWindowCopy {
    VkBuffer source[3] {};
    VkBuffer ref[3] {};
};

// Copy the source window's planes (the union of all windows that this record's
// dispatches may need, clamped to [n-2r, n+2r]) into the src ring.
static void record_src_copies(BM3DData * d, const Bm3dFrame & fr, VkCommandBuffer cmd,
                              const std::vector<Bm3dWindowCopy> & window) {
    const int clips = d->final ? 2 : 1;
    for (int k = 0; k < fr.n_src; ++k) {
        if (!fr.upload_new[k]) {
            continue;
        }
        const int src_slot = fr.src_slot[k];
        const VkDeviceSize slot_device =
            static_cast<VkDeviceSize>(src_slot) * clips * d->planes[0].pe;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            const VkDeviceSize pe = p.pe;
            const VkDeviceSize plane_off = static_cast<VkDeviceSize>(plane) * d->src_size;
            const Bm3dWindowCopy & w = window[k];
            // source clip: second half of the slot in final mode
            {
                VkBufferCopy2 region {};
                region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
                region.srcOffset = 0;
                region.dstOffset = (slot_device + static_cast<VkDeviceSize>(clips - 1) * pe + plane_off) * 4;
                region.size = pe * 4;
                VkCopyBufferInfo2 copy {};
                copy.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
                copy.srcBuffer = w.source[plane];
                copy.dstBuffer = d->src.buffer;
                copy.regionCount = 1;
                copy.pRegions = &region;
                d->gpu->vk->vkCmdCopyBuffer2(cmd, &copy);
            }
            // ref clip (final mode only): first half of the slot
            if (d->final) {
                VkBufferCopy2 region {};
                region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
                region.srcOffset = 0;
                region.dstOffset = (slot_device + plane_off) * 4;
                region.size = pe * 4;
                VkCopyBufferInfo2 copy {};
                copy.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
                copy.srcBuffer = w.ref[plane];
                copy.dstBuffer = d->src.buffer;
                copy.regionCount = 1;
                copy.pRegions = &region;
                d->gpu->vk->vkCmdCopyBuffer2(cmd, &copy);
            }
        }
    }
}

// Record one estimation submission into the context's command buffer. The
// first chunk carries the ring copies and a leading whole-queue barrier; the
// last carries the trailing barrier that makes the atomic accumulation visible
// to the aggregation. One position per submission (VSFEEL_BM3D_SPLIT, the
// default) keeps a single submission from running past a slow card's watchdog.
static void record_est_chunk(BM3DData * d, VSGPUExecContext * ctx, const Bm3dFrame & fr,
                             int n, int c, int chunks,
                             const std::vector<Bm3dWindowCopy> & window,
                             VkBuffer dst_plane, bool gputrace) {
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);

    if (c == 0) {
        // Order this frame's writes (the ring copies and every slot zero-fill)
        // behind everything already submitted on the queue. The cross-frame
        // handoff is exactly this: a reader waited host side until the writer's
        // estimation was submitted, so the writer's commands are earlier in
        // submission order and this barrier covers both the execution and the
        // memory dependency. It also orders the overwrite of a recycled slot
        // after the previous reader's aggregation.
        bm3d_full_barrier(*d->gpu, cmd);
        record_src_copies(d, fr, cmd, window);
        // the estimation dispatches read the freshly copied source/ref frames,
        // so make the transfer writes visible to the compute stage before
        // launching them (and order the res zero-fill ahead of the atomic
        // accumulation)
        bm3d_full_barrier(*d->gpu, cmd);
        if (gputrace) {
            // a query must be reset before first use and before each reuse;
            // doing it inside the command buffer keeps the reset ordered with
            // the stamps (and with the aggregation command buffer submitted
            // after this one)
            d->gpu->vk->vkCmdResetQueryPool(cmd, d->probe.query, 0, 4);
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                d->probe.query, 0);
        }
    }

    if (c < fr.n_pos) {
        if (d->split_est) {
            record_est_position(d, fr, cmd, n, fr.est_pos[c], dst_plane);
        } else {
            for (int k = 0; k < fr.n_pos; ++k) {
                record_est_position(d, fr, cmd, n, fr.est_pos[k], dst_plane);
            }
        }
    }

    if (c == chunks - 1) {
        if (gputrace) {
            d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                d->probe.query, 1);
        }
        // The estimation kernels' atomic accumulation must be visible to the
        // aggregation reads, which are dispatched from a separate submission:
        // make the writes available to the queue-wide scope so the aggregation
        // sees complete slot contents.
        bm3d_full_barrier(*d->gpu, cmd);
    }
}

// Record the aggregation submission into the context's command buffer. It
// reads the res slots the in-flight frames wrote, so the host must have waited
// for their submissions first; its leading barrier makes those writes visible.
// The result goes straight into the output plane, which is why nothing is
// downloaded afterwards.
static void record_bm3d_agg(BM3DData * d, const Bm3dFrame & fr, VkCommandBuffer cmd,
                            int n, VkBuffer dst_plane, bool gputrace) {
    const int nf = d->nframes;
    const int r = d->radius;

    for (int plane = 0; plane < d->n_planes; ++plane) {
        const auto & p = d->planes[plane];
        const VkDeviceSize pe = p.pe;

        // aggregation: tw stacked slices (clamped frame indices, aggZ blocks)
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.agg_pipeline);
        // descriptor bindings do not carry across command buffers: the
        // aggregation is recorded separately from the estimation phase, so
        // without this bind the dispatch runs on undefined descriptor state
        // (black output, and device loss under concurrent submissions)
        bm3d_bind(*d->gpu, cmd, d->pipeline_layout, d->src.buffer, d->res.buffer, dst_plane);
        {
            int32_t bases[9] {};
            if (r == 0) {
                // non-temporal: aggregate the single center slice
                const int32_t base = static_cast<int32_t>(
                    static_cast<VkDeviceSize>(fr.win_slots[0]) * 2 * pe +
                    static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
                for (int i = 0; i < d->tw; ++i) bases[i] = base;
            } else {
                for (int i = 0; i < d->tw; ++i) {
                    const int z = agg_z(i, n, nf, r);
                    bases[i] = static_cast<int32_t>(
                        static_cast<VkDeviceSize>(fr.win_slots[i]) * d->tw * 2 * pe +
                        static_cast<VkDeviceSize>(plane) * d->res_size_per_plane +
                        static_cast<VkDeviceSize>(z) * 2 * pe);
                }
            }
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, bases, sizeof(bases));
        }
        // the estimation kernel's atomic accumulation (and the fill that
        // zeroes the slots) must be visible to the aggregation reads; the
        // aggregation kernel reads with atomic loads, but the RADV driver
        // still needs an explicit barrier for the cross-dispatch visibility
        bm3d_full_barrier(*d->gpu, cmd);
        if (gputrace) d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, d->probe.query, 2);
        d->gpu->vk->vkCmdDispatch(cmd, p.agg_grid_x, p.agg_grid_y, 1);
        if (gputrace) d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, d->probe.query, 3);
    }
}

static const VSFrame *VS_CC BM3DGetFrame(
    int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    BM3DData * d = static_cast<BM3DData *>(instanceData);

    if (activationReason == arInitial) {
        const int r = d->radius;
        for (int i = -2 * r; i <= 2 * r; ++i) {
            const int idx = std::clamp(n + i, 0, d->nframes - 1);
            vsapi->requestFrameFilter(idx, d->node, frameCtx);
            if (d->ref_node) {
                vsapi->requestFrameFilter(idx, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        // A sigma below FLT_EPSILON passes its plane through, as the reference's
        // PROC_MASK does. The mask is instance-constant, so a skipped plane has
        // no cache reservation and no cross-frame synchronization to honour.
        const bool skip = !d->process;

        // The centre source frame is the property donor and, for a skipped or
        // unprocessed plane, the plane's owner: newVideoFrame2 propagates GPU
        // residency -- and each shared plane's producer pair -- from it.
        const VSFrame * center = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrame * dst = nullptr;
        if (skip) {
            if (d->chroma) {
                const int pl[] = { 0, 1, 2 };
                const VSFrame * fr[] = { center, center, center };
                dst = vsapi->newVideoFrame2(
                    &d->vi->format, d->vi->width, d->vi->height, fr, pl, center, core);
            } else {
                const int pl[] = { 0 };
                const VSFrame * fr[] = { center };
                dst = vsapi->newVideoFrame2(
                    &d->vi->format, d->vi->width, d->vi->height, fr, pl, center, core);
            }
        } else if (d->chroma) {
            // the luma plane is computed here; chroma and the frame props are
            // shared straight from the source frame
            const int pl[] = { 0, 1, 2 };
            const VSFrame * fr[] = { nullptr, center, center };
            dst = vsapi->newVideoFrame2(
                &d->vi->format, d->vi->width, d->vi->height, fr, pl, center, core);
        } else {
            dst = d->gpu->api->newGPUVideoFrame(
                &d->vi->format, d->vi->width, d->vi->height, center, core);
        }
        vsapi->freeFrame(center);
        if (!dst) {
            vsapi->setFilterError("BM3D: failed to allocate the output frame", frameCtx);
            return nullptr;
        }
        if (skip) {
            return dst;
        }

        vsfeel_trace_frame_begin();
        auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        vsfeel_trace_mark("acquire");

        // The GPU-timing probe serializes instrumented frames from before the
        // cache reservation: the one query pool is shared, and locking only
        // around the recording would let a frame hold the lock while waiting on
        // another frame's submission -- which could itself be blocked on the
        // lock. Taken here, every writer a frame waits on has already finished.
        bool gputrace = false;
        std::unique_lock<std::mutex> probe_lock(d->probe.lock, std::defer_lock);
        if (d->probe.enabled) {
            probe_lock.lock();
            gputrace = true;
        }

        // Reserve this frame's cache slots (blocks only when the working set
        // exceeds the cache, e.g. on seeks; holds no recording context while
        // waiting).
        Bm3dFrame fr;
        acquire_cache(d, fr, n);
        auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        std::vector<Bm3dWindowCopy> window(static_cast<size_t>(fr.n_src));
        std::vector<const VSFrame *> sources;

        const auto set_error = [&](const std::string & error_message) -> const VSFrame * {
            vsfeel_trace_error("BM3D", n, error_message, d->gpu.get());
            // A reader may be waiting for this frame's estimation; mark the
            // slots submitted so it goes ahead instead of blocking forever on
            // a signal that is never coming. This frame's own output is an
            // error either way.
            publish_est_submitted(d, fr);
            release_cache(d, fr);
            for (const VSFrame * f : sources) {
                vsapi->freeFrame(f);
            }
            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        // Copy only the frames whose cache slots the acquire reserved. The
        // needed range is the union of every window that this record's
        // dispatches may read: [clamp(n-2r), clamp(n+2r)].
        for (int k = 0; k < fr.n_src; ++k) {
            if (!fr.upload_new[k]) {
                continue;
            }
            const int f = fr.src_lo + k;
            const VSFrame * src = vsapi->getFrameFilter(f, d->node, frameCtx);
            for (int plane = 0; plane < d->n_planes; ++plane) {
                VSVulkanPlaneInfo plane_info {};
                if (d->gpu->api->getGPUPlane(src, plane, &plane_info)) {
                    vsapi->freeFrame(src);
                    return set_error("clip " + std::to_string(f) + " plane " +
                        std::to_string(plane) + " is not GPU resident");
                }
                window[k].source[plane] = plane_info.buffer;
            }
            sources.push_back(src);
            if (d->final) {
                const VSFrame * rsrc = vsapi->getFrameFilter(f, d->ref_node, frameCtx);
                for (int plane = 0; plane < d->n_planes; ++plane) {
                    VSVulkanPlaneInfo plane_info {};
                    if (d->gpu->api->getGPUPlane(rsrc, plane, &plane_info)) {
                        vsapi->freeFrame(rsrc);
                        return set_error("ref clip " + std::to_string(f) + " plane " +
                            std::to_string(plane) + " is not GPU resident");
                    }
                    window[k].ref[plane] = plane_info.buffer;
                }
                sources.push_back(rsrc);
            }
        }
        auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        // The destination plane the aggregation writes: the output frame's own
        // storage, so nothing is downloaded.
        VSVulkanPlaneInfo dst_plane {};
        if (d->gpu->api->getGPUPlane(dst, 0, &dst_plane)) {
            return set_error("the output frame is not GPU resident");
        }

        collect_est_positions(d, fr, n);
        const int chunks = (d->split_est && fr.n_pos > 0) ? fr.n_pos : 1;

        if (d->trace) {
            for (int k = 0; k < fr.n_src; ++k) {
                if (!fr.upload_new[k]) {
                    fprintf(stderr, "[t] n=%d reads src slot %d (copier w=%d)\n",
                        n, fr.src_slot[k], fr.src_writer[k]);
                }
            }
            for (int i = 0; i < d->tw; ++i) {
                if (!fr.win_recompute[i]) {
                    fprintf(stderr, "[t] n=%d reads res slot %d (writer w=%d)\n",
                        n, fr.win_slots[i], fr.win_writer[i]);
                }
            }
        }

        // The source window this frame reads may have been copied by other
        // frames; wait host side until those copies have been submitted. This
        // frame holds the slots, so no writer can be re-reserved underneath it.
        wait_src_submitted(d, fr);

        // The estimation is submitted first, so the GPU stays busy with the
        // heavy kernels while the host waits for the writers of the
        // aggregation slots.
        vsfeel_trace_mark("sub est");
        for (int c = 0; c < chunks; ++c) {
            char errbuf[512] {};
            VSGPUExecContext * ctx = d->gpu->api->gpuExecAcquire(d->exec, errbuf, sizeof(errbuf));
            if (!ctx) {
                return set_error("could not acquire a recording context: "s + errbuf);
            }
            if (c == 0) {
                // The producer pairs of the frames this submission copies from
                // become device-side waits, and the pool keeps the frames alive
                // until the submission completes.
                for (const VSFrame * f : sources) {
                    d->gpu->api->gpuExecReadsFrame(ctx, f);
                }
            }
            record_est_chunk(d, ctx, fr, n, c, chunks, window, dst_plane.buffer, gputrace);
            if (d->gpu->api->gpuExecSubmit(ctx, nullptr, errbuf, sizeof(errbuf))) {
                return set_error("estimation submit failed: "s + errbuf);
            }
        }
        for (const VSFrame * f : sources) {
            vsapi->freeFrame(f);
        }
        sources.clear();
        auto t3 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        // Everything this frame wrote is in the queue now, so a reader may go
        // ahead: queue order plus its leading barrier is the whole handoff.
        publish_est_submitted(d, fr);

        // The aggregation reads the estimate stacks, so wait host side until
        // every stack writer has submitted (see publish_est_submitted). The
        // waits follow cache-acquisition order, which is acyclic.
        wait_res_submitted(d, fr);

        char aerr[512] {};
        VSGPUExecContext * agg_ctx = d->gpu->api->gpuExecAcquire(d->exec, aerr, sizeof(aerr));
        if (!agg_ctx) {
            return set_error("could not acquire a recording context: "s + aerr);
        }
        // The pool publishes the output plane's producer pair at submit.
        d->gpu->api->gpuExecWritesPlane(agg_ctx, dst, 0);
        vsfeel_trace_mark("sub agg");
        record_bm3d_agg(d, fr, d->gpu->api->gpuExecCommandBuffer(agg_ctx), n,
                        dst_plane.buffer, gputrace);
        uint64_t agg_value = 0;
        if (d->gpu->api->gpuExecSubmit(agg_ctx, &agg_value, aerr, sizeof(aerr))) {
            return set_error("aggregation submit failed: "s + aerr);
        }
        auto t4 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (d->trace) fprintf(stderr, "[t] n=%d submitted (%d est chunks)\n", n, chunks);

        if (gputrace) {
            // The stamps were written by this frame's submissions; waiting the
            // aggregation out leaves the query results final.
            char perr[256] {};
            if (d->gpu->api->gpuExecWaitValue(d->exec, agg_value, perr, sizeof(perr)) == gdDrained) {
                uint64_t ts[4] {};
                if (d->gpu->vk->vkGetQueryPoolResults(d->gpu->device, d->probe.query, 0, 4,
                        sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                    const double period = d->gpu->limits.timestampPeriod;
                    static std::atomic<uint64_t> ts_k {}, ts_a {};
                    static std::atomic<uint32_t> ts_nf {};
                    const uint32_t nq = ts_nf.fetch_add(1) + 1;
                    const auto ns = [period](uint64_t a, uint64_t b) {
                        return static_cast<uint64_t>(
                            std::llround(static_cast<double>(b - a) * period));
                    };
                    if (ts[0] && ts[1]) ts_k += ns(ts[0], ts[1]);
                    if (ts[2] && ts[3]) ts_a += ns(ts[2], ts[3]);
                    if (nq % 50 == 0) {
                        fprintf(stderr, "[bm3dgpu] n=%u kernel=%.3f agg=%.3f (ms) disp=%d\n",
                            nq, ts_k.load() / double(nq) / 1e6, ts_a.load() / double(nq) / 1e6,
                            fr.n_pos);
                    }
                }
            }
        }

        // Reserved slots go back as soon as the aggregation is queued: a later
        // frame's recompute is ordered after it by the queue, and its leading
        // barrier completes the ordering.
        release_cache(d, fr);
        auto t5 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        if (d->host_timing) {
            const auto us = [](auto a, auto b) {
                return static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
            };
            d->ht_acquire_ns += us(t0, t1);
            d->ht_source_ns += us(t1, t2);
            d->ht_est_ns += us(t2, t3);
            d->ht_agg_ns += us(t3, t4);
            d->ht_release_ns += us(t4, t5);
            d->ht_total_ns += us(t0, t5);
            d->ht_n.fetch_add(1, std::memory_order_relaxed);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC BM3DFree(void *instanceData, [[maybe_unused]] VSCore *core, const VSAPI *vsapi) {
    BM3DData * d = static_cast<BM3DData *>(instanceData);
    vsapi->freeNode(d->node);
    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    delete d;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BM3DData>() };

    // Opt-in host-path probe: the default path records no clocks.
    d->host_timing = vsfeel_debug_probe("VSFEEL_BM3D_TIMING");
    // The timestamp pool is created only when this is set, so every later
    // recording/readback must use the cached flag, not a frame-time getenv.
    d->gpu_trace = vsfeel_debug_probe("VSFEEL_BM3D_GPUTRACE");
    // Cached too: the frame path must not pay a getenv for a flag that is fixed
    // per instance.
    d->trace = vsfeel_debug_trace("VSFEEL_BM3D_TRACE");
    d->dump = env_flag("VSFEEL_BM3D_DUMP");
    d->split_est = env_int("VSFEEL_BM3D_SPLIT", 1) != 0;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("BM3D", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    // optional "ref": a basic-estimate clip used for the final (Wiener) pass;
    // the block matching runs on it and its patches provide the Wiener
    // reference, while the noisy clip is the clip actually denoised
    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
    if (d->ref_node) {
        const VSVideoInfo * rvi = vsapi->getVideoInfo(d->ref_node);
        if (rvi->format.colorFamily != d->vi->format.colorFamily ||
            rvi->format.sampleType != d->vi->format.sampleType ||
            rvi->format.bitsPerSample != d->vi->format.bitsPerSample ||
            rvi->format.subSamplingW != d->vi->format.subSamplingW ||
            rvi->format.subSamplingH != d->vi->format.subSamplingH)
        {
            return set_error("\"ref\" must be of the same format as \"clip\"");
        }
        if (rvi->width != d->vi->width || rvi->height != d->vi->height) {
            return set_error("\"ref\" must be of the same dimensions as \"clip\"");
        }
        if (rvi->numFrames != d->vi->numFrames) {
            return set_error("\"ref\" must be of the same number of frames as \"clip\"");
        }
        d->final = true;
    }

    if (d->vi->width <= 0 || d->vi->height <= 0 ||
        d->vi->format.sampleType != stFloat || d->vi->format.bitsPerSample != 32) {
        return set_error("only constant format 32 bit float input supported");
    }

    // The block-matching kernel always loads unconditional 8x8 patches and
    // clamps its block origin to (dimension - 8), so anything smaller would
    // index negative buffer offsets. Reject it before allocating GPU
    // resources rather than relying on robust buffer access (which is not
    // enabled).
    if (d->vi->width < 8 || d->vi->height < 8) {
        return set_error("clip dimensions must be at least 8x8");
    }

    // The scan lists pack a candidate's (x, y) into one 32-bit word (16 + 15
    // bits) so the top-8 insert shifts three arrays instead of four.
    if (d->vi->width > 0xFFFF || d->vi->height > 0x7FFF) {
        return set_error("clip dimensions must not exceed 65535x32767");
    }

    // The window clamps and the cache keys assume [0, numFrames-1]; an empty
    // or unknown length would index the slot tables before their base.
    if (d->vi->numFrames <= 0) {
        return set_error("clip frame count must be known and positive");
    }

    std::array<float, 3> sigma;
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] = static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &error));
        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (!std::isfinite(sigma[i]) || sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be finite and non-negative");
        }
    }
    // A plane whose sigma is below FLT_EPSILON is passed through, exactly as
    // the reference's PROC_MASK does. vsfeel denoises luma only, so the mask
    // decides whether luma runs at all: a skipped plane is a source copy,
    // which also removes the 0/0 the Wiener coefficient produced at sigma=0.
    d->process = sigma[0] >= FLT_EPSILON;

    // match the reference sigma scaling exactly (different factor for the
    // final Wiener pass)
    const float sigma_factor = d->final
        ? std::bit_cast<float>(0x3e40c0c1u)
        : std::bit_cast<float>(0x3f021bb6u);
    for (auto & sv : sigma) {
        sv *= sigma_factor;
    }
    d->sigma = sigma[0];
    d->sigma_u = sigma[1];
    d->sigma_v = sigma[2];

    std::array<int, 3> block_step;
    for (int i = 0; i < std::ssize(block_step); ++i) {
        block_step[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_step", i, &error));
        if (error) {
            block_step[i] = (i == 0) ? 8 : block_step[i - 1];
        } else if (block_step[i] <= 0 || block_step[i] > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }
    }
    d->block_step = block_step[0];

    std::array<int, 3> bm_range;
    for (int i = 0; i < std::ssize(bm_range); ++i) {
        bm_range[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "bm_range", i, &error));
        if (error) {
            bm_range[i] = (i == 0) ? 9 : bm_range[i - 1];
        } else if (bm_range[i] <= 0 || bm_range[i] > kMaxSearchRange) {
            return set_error("\"bm_range\" must be in range [1, 8192]");
        }
    }
    d->bm_range = bm_range[0];

    int64_t radius_raw = vsapi->mapGetInt(in, "radius", 0, &error);
    d->radius = vsh::int64ToIntS(radius_raw);
    if (error) {
        d->radius = 0;
    }
    if (d->radius < 0 || d->radius > 4) {
        return set_error("\"radius\" must be in range [0, 4]");
    }
    d->tw = 2 * d->radius + 1;

    std::array<int, 3> ps_num;
    for (int i = 0; i < std::ssize(ps_num); ++i) {
        ps_num[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_num", i, &error));
        if (error) {
            ps_num[i] = (i == 0) ? 2 : ps_num[i - 1];
        } else if (ps_num[i] <= 0 || ps_num[i] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }
    }
    d->ps_num = ps_num[0];

    std::array<int, 3> ps_range;
    for (int i = 0; i < std::ssize(ps_range); ++i) {
        ps_range[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_range", i, &error));
        if (error) {
            ps_range[i] = (i == 0) ? 4 : ps_range[i - 1];
        } else if (ps_range[i] <= 0 || ps_range[i] > kMaxSearchRange) {
            return set_error("\"ps_range\" must be in range [1, 8192]");
        }
    }
    d->ps_range = ps_range[0];

    // device_id and num_streams are registered but never read: the core owns
    // the one device and sizes in-flight depth itself (exec pool ring).

    // at radius 0 every frame only touches its own slot and never depends on
    // the previous frames' estimates, so give each in-flight frame its own
    // src/res slot to keep the pipeline full; otherwise the ring of 1 would
    // serialize the frames behind the timeline waits. At radius > 0 the caches
    // are keyed by frame modulo the capacity and reserved all-or-nothing per
    // frame, so they only need to cover the working set of the concurrent
    // frames (like the reference's fused-mode accumulator cache); anything
    // beyond that (e.g. seeking) blocks in the acquire instead of corrupting.
    d->src_ring = (d->radius == 0) ? kInflightFrames : 4 * d->radius + kInflightFrames;
    // One in-flight frame needs the stacks of centre frames [n-r, n+r], so
    // kInflightFrames concurrent frames span kInflightFrames + 2r slots. That
    // working set is the default: the estimate cache is the largest allocation,
    // and on an 8 GiB card the slack below is the difference between running
    // radius 4 and failing to allocate. VSFEEL_BM3D_CACHE=1 adds a whole extra
    // window, so an out-of-order (seek) request finds a warm slot instead of
    // waiting in the acquire, for tw/(ns+2r+tw) more VRAM.
    const int res_working_set = kInflightFrames + 2 * d->radius;
    const bool cache_slack = env_int("VSFEEL_BM3D_CACHE", 0) != 0;
    d->res_cap = (d->radius == 0) ? kInflightFrames
        : (cache_slack ? res_working_set + d->tw : res_working_set);

    const int extractor_exp = vsh::int64ToIntS(vsapi->mapGetInt(in, "extractor_exp", 0, &error));
    // outside the normal float exponent range the extractor add/subtract pair
    // turns the aggregation into NaN
    if (extractor_exp < -126 || extractor_exp > 127) {
        return set_error("\"extractor_exp\" must be in range [-126, 127]");
    }
    d->extractor = (extractor_exp != 0)
        ? std::ldexp(1.0f, extractor_exp) : 0.0f;

    d->nframes = d->vi->numFrames;

    {
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }
    d->vsapi = vsapi;

    // The GPU-timing probe is invalid usage on a queue family whose
    // timestampValidBits is 0, where a timestamp write can hang the engine
    // (a machine-wide freeze, not just a lost device): keep it off there.
    d->gpu_trace = d->gpu_trace && vsfeel_probe_timestamps(*d->gpu, "BM3D");

    // The BM3D kernels accumulate into float SSBOs with atomicAdd, which needs
    // shaderBufferFloat32AtomicAdd: VK_EXT_shader_atomic_float reports the
    // load/store/exchange atomics separately, so that bit alone must not select
    // this path. No pre-RDNA3 AMD driver reports the add at all (RADV: GFX11+;
    // the Windows driver does not expose it on Polaris either); on anything
    // older the accumulation falls back to the CAS loop the OpenCL reference
    // itself uses (atom_add_f), so the filter runs everywhere instead of
    // failing at creation.
    d->cas_atomics = env_flag("VSFEEL_BM3D_CAS") || !d->gpu->feat_atomic_float32_add;
    if (vsfeel_device_info_enabled()) {
        fprintf(stderr, "[bm3d] aggregation: %s\n",
            d->cas_atomics ? "CAS loop (no buffer float32 add atomics available)"
                           : "hardware buffer float atomics");
    }
    // The 8x8 group transposes and the group-8 reduction are subgroup shuffles,
    // and the kernel's per-lane layout puts each 8-lane group inside one
    // subgroup, so a device without the shuffle intrinsics cannot run it. BASIC
    // is the only operation Vulkan mandates, so this is a real check.
    if (!d->gpu->has_subgroup_ops(VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) {
        return set_error("subgroup shuffle is not supported by this device "
                         "(VK_SUBGROUP_FEATURE_SHUFFLE_BIT is required)");
    }

    VkDevice dev = d->gpu->device;

    {
        // Push descriptors: the bindings change every frame (the destination
        // plane is the frame's own storage), so nothing is allocated from a
        // pool and nothing survives the command buffer.
        const auto result = gpu_push_set_layout(*d->gpu, 3);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result = gpu_pipeline_layout(*d->gpu, d->set_layout, 9 * sizeof(int32_t));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // The plane layout has to be the one the core's GPU frames carry, because
    // a source frame's plane is copied into the ring with a single flat
    // buffer-to-buffer region. A GPU frame's stride is the CPU frame's stride
    // (the core stores planes exactly as the CPU allocator would), so it is
    // read off a scratch frame here rather than guessed from an alignment rule.
    int frame_stride_elems = 0;
    {
        VSFrame * probe = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, nullptr, core);
        if (probe == nullptr) {
            return set_error("could not allocate a probe frame to read the plane stride");
        }
        frame_stride_elems = static_cast<int>(vsapi->getStride(probe, 0) / sizeof(float));
        vsapi->freeFrame(probe);
        if (frame_stride_elems < d->vi->width) {
            return set_error("the core reported an unexpected plane stride");
        }
    }

    // plane configs (luma plane 0; YUV chroma is passed through unprocessed)
    d->n_planes = 0;
    if (d->vi->format.colorFamily == cfGray) {
        auto & p = d->planes[0];
        p.width = d->vi->width;
        p.height = d->vi->height;
        p.stride = frame_stride_elems;
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        d->n_planes = 1;
    } else if (d->vi->format.colorFamily == cfYUV) {
        auto & p = d->planes[0];
        p.width = d->vi->width;
        p.height = d->vi->height;
        p.stride = frame_stride_elems;
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        d->n_planes = 1;
        d->chroma = true;
    } else {
        return set_error("BM3D: only Gray and YUV input are currently supported");
    }

    // The kernel addresses the estimate stacks through signed 32-bit offsets,
    // so a stack at or above 2^31 floats wraps and writes outside the slot.
    {
        const VkDeviceSize res_floats = static_cast<VkDeviceSize>(d->res_cap) *
            d->tw * 2 * d->planes[0].pe;
        if (res_floats > static_cast<VkDeviceSize>(INT32_MAX)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "frame is too large: the estimate cache needs %llu floats per "
                "plane (radius %d), which overflows the 32-bit kernel "
                "addressing; reduce radius",
                static_cast<unsigned long long>(res_floats), d->radius);
            return set_error(msg);
        }
    }

    // shared buffers
    {
        VkDeviceSize src_size = 0;
        VkDeviceSize res_size = 0;
        (void)res_size;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            // in final mode each ring slot holds [ref][source], so the ring
            // doubles in size
            const int clips = d->final ? 2 : 1;
            src_size += static_cast<VkDeviceSize>(d->src_ring) * clips * p.pe;
            res_size += static_cast<VkDeviceSize>(d->res_cap) * d->tw * 2 * p.pe;
        }
        d->src_size = src_size;
        d->res_size_per_plane = static_cast<VkDeviceSize>(d->res_cap) * d->tw * 2 * d->planes[0].pe;

        // Both buffers come from the core's pool, so they count against the
        // VRAM budget the frame cache and the thread pool's admission control
        // also see. The source ring only ever receives copies; the estimate
        // cache is filled and read by kernels.
        {
            std::string err = gpu_make_buffer(*d->gpu, core, src_size * 4, d->src,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (!err.empty()) {
                return set_error(err);
            }
        }
        {
            std::string err = gpu_make_buffer(*d->gpu, core, res_size * 4, d->res,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (!err.empty()) {
                // The estimate cache is by far the largest allocation, so an
                // out-of-memory here is the usual "radius too high for this
                // card"; name the size and what shrinks it.
                char msg[320];
                snprintf(msg, sizeof(msg),
                    "%s; the estimate cache needs %.0f MiB (radius %d): "
                    "lower radius",
                    err.c_str(),
                    static_cast<double>(res_size) * 4.0 / (1024.0 * 1024.0),
                    d->radius);
                return set_error(msg);
            }
        }
    }

    if (d->gpu_trace) {
        VkQueryPoolCreateInfo qp_info {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 4
        };
        checkVK(d->gpu->vk->vkCreateQueryPool(dev, &qp_info, nullptr, &d->probe.query));
        d->probe.enabled = true;
    }

    // pipelines
    for (int plane = 0; plane < d->n_planes; ++plane) {
        auto & p = d->planes[plane];
        {
            const uint32_t * code = d->cas_atomics ? bm3d_cas_spv : bm3d_spv;
            const size_t code_size = d->cas_atomics ? bm3d_cas_spv_size : bm3d_spv_size;
            const auto result = create_bm3d_pipeline(*d->gpu, p, *d, code, code_size, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            p.bm3d_pipeline = std::get<VkPipeline>(result);
        }
        {
            const auto result = create_agg_pipeline(*d->gpu, p, *d, bm3d_agg_spv,
                bm3d_agg_spv_size, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            p.agg_pipeline = std::get<VkPipeline>(result);
        }
        p.bm3d_grid_x = static_cast<uint32_t>((p.width + 4 * d->block_step - 1) / (4 * d->block_step));
        p.bm3d_grid_y = static_cast<uint32_t>((p.height + d->block_step - 1) / d->block_step);
        p.agg_grid_x = static_cast<uint32_t>((p.stride + 127) / 128);
        p.agg_grid_y = static_cast<uint32_t>((p.height + 7) / 8);
    }

    // One exec pool per instance, on the core's compute queue: it owns the
    // command buffers, the timeline and the backpressure, and the frame path
    // records and submits through it.
    {
        char err[512] {};
        d->exec = d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (!d->exec) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    // Per-instance VRAM budget. Both shared buffers come from the core's pool,
    // so this is the whole footprint now: there is no per-stream staging and no
    // download buffer. Gated by an env flag so a normal creation prints nothing.
    if (vsfeel_debug_flag("VSFEEL_BM3D_VRAM")) {
        const double mib = 1024.0 * 1024.0;
        const double total = static_cast<double>(d->src_size + d->res_size_per_plane) * 4.0;
        const double res_only = static_cast<double>(d->res_size_per_plane) * 4.0;
        fprintf(stderr,
            "[bm3d] vram: src=%.1f MiB res=%.1f MiB (%.0f%% of total) -> total=%.1f MiB "
            "(radius=%d inflight=%d res_cap=%d src_ring=%d stride=%d)\n",
            static_cast<double>(d->src_size) * 4.0 / mib, res_only / mib,
            100.0 * res_only / total, total / mib,
            d->radius, kInflightFrames, d->res_cap, d->src_ring, d->planes[0].stride);
    }

    d->src_frame.assign(d->src_ring, -1);
    d->src_writer.assign(d->src_ring, -1);
    d->src_ready.assign(d->src_ring, 0);
    d->src_holders.resize(d->src_ring);
    d->res_frame.assign(d->res_cap, -1);
    d->res_writer.assign(d->res_cap, -1);
    d->res_ready.assign(d->res_cap, 0);
    d->res_holders.resize(d->res_cap);
    d->r0_free.reserve(kInflightFrames);
    for (int i = 0; i < kInflightFrames; ++i) {
        d->r0_free.push_back(i);
    }

    BM3DData * data = d.release();

    // A temporal filter requests frames outside n, which the strict-spatial
    // policy does not permit; only radius 0 is purely spatial.
    const VSRequestPattern policy =
        data->radius > 0 ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2] = {
        { data->node, policy },
        { data->ref_node, policy }
    };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "BM3D", data->vi,
        BM3DGetFrame, BM3DFree,
        fmParallel, ffGPUOutput, deps, data->ref_node ? 2 : 1, data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "BM3D: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bm3dv2(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "BM3Dv2",
        "clip:vnode:gpu;"
        "ref:vnode:gpu:opt;"
        "sigma:float[]:opt;"
        "block_step:int[]:opt;"
        "bm_range:int[]:opt;"
        "radius:int:opt;"
        "ps_num:int[]:opt;"
        "ps_range:int[]:opt;"
        "num_streams:int:opt;"
        "extractor_exp:int:opt;"
        "device_id:int:opt;",
        "clip:vnode:gpu;",
        BM3DCreate, nullptr, plugin
    );
}
