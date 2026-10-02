#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
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

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

namespace {

// The reference implementations' cap: vszipcl accepts 16, bm3dvk 15.
constexpr int MAX_RADIUS = 16;
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
    VkDeviceSize pe {}; // plane extent in floats (h * stride)
    // Scaled sigma, or exactly 0 when the plane is not processed (the
    // reference zeroes it so its kernel's epsilon test agrees with the
    // unscaled decision the host made).
    float sigma {};
    bool process {};
    int block_step {};
    int bm_range {};
    int ps_num {};
    int ps_range {};
    VkPipeline agg_pipeline {};
    uint32_t agg_grid_x {};
    uint32_t agg_grid_y {};
};

// One estimation entry: a source ring and an estimate stack that cover the
// planes it filters. The reference's default mode builds one entry per
// processed plane (each plane block-matches on its own content, at its own
// geometry); chroma mode builds one entry over the clip's three 4:4:4 planes,
// whose groups come from luma. Every plane of an entry shares one geometry.
struct Bm3dGroup {
    int n_planes {};
    std::array<int, 3> planes {}; // absolute plane indices, in packing order
    VkDeviceSize pe {};           // per-plane extent in floats
    VkPipeline bm3d_pipeline {};
    uint32_t bm3d_grid_x {};
    uint32_t bm3d_grid_y {};

    int src_ring {};          // cache slots for the source window
    int res_cap {};           // cache slots for the per-frame estimate stacks
    int tag_base {};          // first witness slot of this entry's tags region
    VkDeviceSize src_size {}; // src_ring * clips * n_planes * pe
    VkDeviceSize res_size {}; // res_cap * tw * 2 * n_planes * pe
    GpuBuffer src;
    GpuBuffer res;

    // Slots are reserved all-or-nothing for the duration of a frame (shared
    // holds for reads, exclusive for recompute/upload), so no two in-flight
    // frames ever touch the same slot; when the cache cannot hold the working
    // set (e.g. seeking), the acquire blocks like the reference's fused-mode
    // accumulator cache. One set per entry, keyed by frame modulo the capacity.
    std::vector<int> src_frame {};
    std::vector<int> src_writer {};
    std::vector<uint8_t> src_ready {};
    std::vector<std::vector<uint64_t>> src_holders {};
    std::vector<int> res_frame {};
    std::vector<int> res_writer {};
    std::vector<uint8_t> res_ready {};
    std::vector<std::vector<uint64_t>> res_holders {};
    // Radius 0 has no cross-frame sharing, so each in-flight frame takes one of
    // these slots outright for its whole life instead of going through the
    // window cache. Size is the old in-flight depth.
    std::vector<int> r0_free {};
};

// Per-frame bookkeeping, per entry. The exec pool owns the command buffers,
// the timeline and the in-flight gate, so nothing here outlives the getFrame
// call: a frame's window reservations are described by this struct, and what a
// reader needs afterwards lives in the reservation tables (see BM3DData).
struct Bm3dGroupFrame {
    // this frame computes it
    std::array<bool, 2 * MAX_RADIUS + 1> win_recompute {};
    // res slot per window position
    std::array<int, 2 * MAX_RADIUS + 1> win_slots {};
    // frame that wrote it (trace)
    std::array<int, 2 * MAX_RADIUS + 1> win_writer {};
    // Window positions this frame must compute, deduplicated (a clamped window
    // maps several positions onto one slot and one centre frame).
    std::array<int, 2 * MAX_RADIUS + 1> est_pos {};
    int n_pos {};
    int src_lo {};
    int n_src {};
    // src frames this frame copies
    std::array<bool, 4 * MAX_RADIUS + 1> upload_new {};
    // ring slot of each window frame
    std::array<int, 4 * MAX_RADIUS + 1> src_slot {};
    // copier of it (trace)
    std::array<int, 4 * MAX_RADIUS + 1> src_writer {};
};

struct Bm3dFrame {
    // unique token identifying this frame's cache reservation. A frame index
    // is not a unique holder identity: the scheduler can process the same
    // frame twice at once, and erasing holders by value would then drop both
    // entries at the first release, freeing a slot another reader still uses.
    uint64_t token {};
    std::array<int, 3>
        slot0 {}; // radius 0: this frame's private slot, per entry
    std::array<Bm3dGroupFrame, 3> g {};
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
    VSNode * node {};
    VSNode * ref_node {}; // optional basic-estimate clip (final/Wiener pass)
    const VSVideoInfo * vi {};

    int radius {};
    int tw {};     // 2 * radius + 1
    bool final {}; // true when a "ref" clip is given
    float extractor {};
    bool
        cas_atomics {}; // aggregate with the CAS kernel (no float32 add atomics)
    // chroma=True: one joint entry over the clip's three 4:4:4 planes, whose
    // groups come from luma (the references' "chroma" mode).
    bool joint {};

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    int num_planes {}; // planes of the clip's format
    std::array<Bm3dPlane, 3> planes {};
    // One entry per processed plane, or a single one covering all three under
    // chroma=True. Every buffer a kernel reads lives in its entry.
    std::array<Bm3dGroup, 3> groups {};
    int n_groups {};

    // The core's exec pool: one per instance, on the compute queue. It owns the
    // command buffers, the timeline and the backpressure; the filter records and
    // submits, and hands it the frames and scratch a submission must keep alive.
    VSGPUExecPool * exec {};

    // Per-(entry, slot, slice) frame witness: each entry's estimate slots are
    // keyed by frame index, so two entries that hold different frames in the
    // same slot index must not share witnesses.
    GpuBuffer tags {};
    GpuBuffer skipped {}; // host-visible: dispatches that skipped a slice
    volatile uint32_t * skipped_mapped {};
    GpuBuffer refusal {}; // host-visible: first refusal fingerprint
    volatile uint32_t * refusal_mapped {};
    VkDeviceSize tags_size {}; // uints
    int nframes {};

    // Cross-frame ordering is a ready flag per slot, not a timeline value: the
    // pool allocates signal values at submit time, so a writer cannot name the
    // value it will signal when it reserves. A reader instead waits until the
    // writer's estimation has been submitted; queue order on the one compute
    // queue then puts the writer's command buffer first, and the reader's
    // leading pipeline barrier carries both the execution and the memory
    // dependency across the two submissions. (The per-slot tables live in each
    // Bm3dGroup.)
    //
    // Set when a frame of this instance fails before its chunk-0 submission: its
    // ring keys were cleared, so a reader already waiting on one must be
    // released and then fail rather than read a slot that was never copied.
    bool failed {};
    // Diagnostic (VSFEEL_BM3D_RINGWAIT): the ring copier's chunk-0 submission
    // value per frame, so a reader can wait the copy out instead of trusting
    // the barrier. The witness covers the estimate slots but cannot see a ring
    // copy that lands late (the estimation writes the witness itself).
    bool ring_wait { false };
    std::vector<uint64_t> chunk0_value {};
    uint64_t next_res_token { 1 };
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
    // VSFEEL_BM3D_FAULT=<n> fails frame n's estimation on a fresh instance,
    // before its chunk-0 submission, so tests can exercise the error path's
    // cache handoff (-1 disables).
    int fault_frame { -1 };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_source_ns {}, ht_est_ns {},
        ht_agg_ns {}, ht_release_ns {}, ht_total_ns {}, ht_n {};

    ~BM3DData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(
                stderr,
                "[bm3d-timing] frames=%.0f per-frame us: acquire=%7.1f source=%7.1f "
                "est=%7.1f agg=%7.1f release=%7.1f total=%7.1f\n",
                n, ht_acquire_ns.load() / 1000.0 / n,
                ht_source_ns.load() / 1000.0 / n, ht_est_ns.load() / 1000.0 / n,
                ht_agg_ns.load() / 1000.0 / n,
                ht_release_ns.load() / 1000.0 / n,
                ht_total_ns.load() / 1000.0 / n);
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
        for (auto & g : groups) {
            gpu_destroy_buffer(*gpu, g.res);
            gpu_destroy_buffer(*gpu, g.src);
            if (g.bm3d_pipeline) {
                gpu->vk->vkDestroyPipeline(dev, g.bm3d_pipeline, nullptr);
            }
        }
        for (auto & p : planes) {
            if (p.agg_pipeline) {
                gpu->vk->vkDestroyPipeline(dev, p.agg_pipeline, nullptr);
            }
        }
        gpu_destroy_buffer(*gpu, tags);
        if (skipped_mapped) {
            const uint32_t n = *skipped_mapped;
            if (n) {
                fprintf(stderr,
                        "[bm3d] %u dispatches skipped an unwitnessed slice\n",
                        n);
            }
        }
        if (refusal_mapped && refusal_mapped[0]) {
            const uint32_t frame = refusal_mapped[1];
            const uint32_t expected = refusal_mapped[2];
            const uint32_t found = refusal_mapped[3];
            fprintf(stderr,
                    "[bm3d] first refusal: frame %u wanted witness %u, found %u"
                    " -- %s\n",
                    frame, expected, found,
                    found == 0
                        ? "slice never written (fill without estimation)"
                        : "slice holds another frame's stacks (recycled slot)");
        }
        gpu_destroy_buffer(*gpu, skipped);
        gpu_destroy_buffer(*gpu, refusal);
        if (pipeline_layout) {
            gpu->vk->vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            gpu->vk->vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

static std::variant<VkPipeline, std::string>
create_bm3d_pipeline(const GPUDevice & gpu, const BM3DData & d,
                     const Bm3dGroup & group, const uint32_t * code,
                     size_t code_size, VkPipelineLayout layout) {

    // One entry's planes share their geometry, and the search parameters are
    // the first plane's (the reference's joint entry does the same); only the
    // per-plane sigmas differ.
    const auto & first = d.planes[group.planes[0]];
    const auto group_sigma = [&](int i) {
        return i < group.n_planes ? d.planes[group.planes[i]].sigma : 0.0f;
    };
    // The block below is the shader's spec-constant ids in order: 0..13 are the
    // original single-plane block, 14..16 the joint-entry additions.
    struct Spec {
        int32_t width, height, stride;
        float sigma_y;
        int32_t block_step, bm_range, radius, ps_num, ps_range;
        float extractor;
        int32_t nosearch, noestimate, src_ring, final;
        float sigma_u, sigma_v;
        int32_t nplanes;
    } spec { first.width,
             first.height,
             first.stride,
             group_sigma(0),
             first.block_step,
             first.bm_range,
             d.radius,
             first.ps_num,
             first.ps_range,
             d.extractor,
             env_flag("VSFEEL_BM3D_NOSEARCH") ? 1 : 0,
             env_flag("VSFEEL_BM3D_NOESTIMATE") ? 1 : 0,
             group.src_ring,
             d.final ? 1 : 0,
             group_sigma(1),
             group_sigma(2),
             group.n_planes };
    const std::array<VkSpecializationMapEntry, 17> entries { {
        { 0, 0, sizeof(int32_t) },
        { 1, 4, sizeof(int32_t) },
        { 2, 8, sizeof(int32_t) },
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
        { 14, 56, sizeof(float) },
        { 15, 60, sizeof(float) },
        { 16, 64, sizeof(int32_t) },
    } };
    // The kernel's 8-lane shuffles keep each aligned 8-lane group inside one
    // subgroup, so any width that is a multiple of 8 runs it; 32 is the measured
    // best on the target GPU (VSFEEL_BM3D_SUBGROUP=64 forces the wave64 path,
    // and the request is validated when the pipeline is created). A device that
    // cannot be asked for 32 keeps its own width when that already suits the
    // groups, and is otherwise asked for 16; only a device whose width both is
    // not a multiple of 8 and cannot be changed is refused.
    const int forced_subgroup = env_int("VSFEEL_BM3D_SUBGROUP", 0);
    uint32_t subgroup_size = 0;
    if (forced_subgroup > 0) {
        subgroup_size = static_cast<uint32_t>(forced_subgroup);
    } else if (gpu.has_subgroup_size(32, 32)) {
        subgroup_size = 32;
    } else if (gpu.subgroup_size % 8 == 0) {
        subgroup_size = 0;
    } else if (gpu.has_subgroup_size(16, 32)) {
        subgroup_size = 16;
    } else {
        return "BM3D needs a subgroup size that is a multiple of 8 lanes "
               "(its 8-lane groups share data with subgroup shuffles), and this "
               "device's is " +
               std::to_string(gpu.subgroup_size) + " and cannot be changed"s;
    }
    // LDS: l_e/l_x/l_y/l_s are [4][64] each, three int arrays and one float.
    const GpuWorkgroup workgroup { .x = 32, .shared_bytes = 4 * 64 * 4 * 4 };
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(),
                               &spec, static_cast<uint32_t>(entries.size()),
                               sizeof(spec), "bm3d", subgroup_size, workgroup);
}

static std::variant<VkPipeline, std::string>
create_agg_pipeline(const GPUDevice & gpu, const Bm3dPlane & plane,
                    const Bm3dGroup & group, const BM3DData & d,
                    const uint32_t * code, size_t code_size,
                    VkPipelineLayout layout) {

    struct Spec {
        int32_t height, stride, tw, radius, res_cap, nplanes, tag_base;
    } spec { plane.height,  plane.stride,   d.tw,          d.radius,
             group.res_cap, group.n_planes, group.tag_base };
    const std::array<VkSpecializationMapEntry, 7> entries { {
        { 0, 0, sizeof(int32_t) },
        { 1, 4, sizeof(int32_t) },
        { 2, 8, sizeof(int32_t) },
        { 3, 12, sizeof(int32_t) },
        { 4, 16, sizeof(int32_t) },
        { 5, 20, sizeof(int32_t) },
        { 6, 24, sizeof(int32_t) },
    } };
    // The aggregation kernel is a plain 32x8 grid-stride kernel with no LDS.
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(),
                               &spec, static_cast<uint32_t>(entries.size()),
                               sizeof(spec), "bm3d_agg", 0,
                               GpuWorkgroup { .x = 32, .y = 8 });
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// Radius 0 has no cross-frame sharing, so a frame takes one slot per entry
// outright for its whole life instead of going through the window cache; the
// acquisition itself lives in acquire_cache, which takes every entry's slot in
// one step.
static void give_r0_slot(BM3DData * d, Bm3dGroup & g, int slot) {
    std::lock_guard lock(d->cache_lock);
    g.r0_free.push_back(slot);
    d->cache_cv.notify_all();
}

// Reserve the cache slots this frame needs, all-or-nothing: the res slots of
// the temporal window (recomputing the missing stacks) and the src slots of the
// source window (re-uploading the missing frames). Slots held by other
// in-flight frames block until they complete, so no two frames ever touch the
// same slot concurrently; the blocking only happens when the working set
// exceeds the cache (seeks, very out-of-order arrivals) and never holds a
// recording context while waiting.
//
// The reservation is atomic across every entry, not per entry: a frame that
// held one entry's slots while waiting for another's could deadlock against a
// frame doing the mirror image, because a reader takes a shared hold on a slot
// whose writer has not published yet (a cache hit does not check holders). One
// atomic step keeps the wait graph on the acquisition order, which is acyclic.
static void acquire_cache(BM3DData * d, Bm3dFrame & fr, int n) {
    const int r = d->radius;
    const int nf = d->nframes;
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & gf = fr.g[gi];
        gf.win_slots.fill(-1);
        gf.win_writer.fill(-1);
        gf.win_recompute.fill(false);
        gf.upload_new.fill(false);
        gf.src_slot.fill(-1);
        gf.src_writer.fill(-1);
        gf.n_src = 1;
        gf.src_lo = n;
    }

    std::unique_lock lock(d->cache_lock);
    if (r == 0) {
        // per-frame slots: private to one in-flight frame, so concurrent
        // out-of-order frames never share a slot. Every entry's slot is taken
        // in the same step, so no frame waits for another while holding one.
        d->cache_cv.wait(lock, [&] {
            for (int gi = 0; gi < d->n_groups; ++gi) {
                if (d->groups[gi].r0_free.empty()) {
                    return false;
                }
            }
            return true;
        });
        for (int gi = 0; gi < d->n_groups; ++gi) {
            auto & g = d->groups[gi];
            auto & gf = fr.g[gi];
            fr.slot0[gi] = g.r0_free.back();
            g.r0_free.pop_back();
            gf.win_slots[0] = fr.slot0[gi];
            gf.win_writer[0] = n;
            gf.win_recompute[0] = true;
            gf.upload_new[0] = true;
            gf.src_slot[0] = fr.slot0[gi];
        }
        return;
    }
    // Every entry covers the same source window, clamped to the clip.
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    for (int gi = 0; gi < d->n_groups; ++gi) {
        fr.g[gi].src_lo = lo;
        fr.g[gi].n_src = hi - lo + 1;
    }
    for (;;) {
        bool ok = true;
        for (int gi = 0; gi < d->n_groups && ok; ++gi) {
            auto & g = d->groups[gi];
            auto & gf = fr.g[gi];
            gf.win_slots.fill(-1);
            gf.win_writer.fill(-1);
            gf.win_recompute.fill(false);
            gf.upload_new.fill(false);
            gf.src_slot.fill(-1);
            gf.src_writer.fill(-1);
            // Phase 1: check-only, with no side effects. The failed passes must
            // not leave half-applied reservations behind, or a retry would treat
            // the abandoned slots as cached and never recompute them.
            for (int i = 0; i < d->tw; ++i) {
                const int m = std::clamp(n - r + i, 0, nf - 1);
                const int slot = m % g.res_cap;
                gf.win_slots[i] = slot;
                gf.win_writer[i] = g.res_writer[slot];
                // A clamped window maps several positions onto one slot, so this
                // must be decided from the state *before* phase 2 mutates it: the
                // later positions would otherwise look cached and keep the stale
                // writer of the slot's previous contents as a dependency.
                gf.win_recompute[i] = (g.res_frame[slot] != m);
                if (gf.win_recompute[i] && !g.res_holders[slot].empty()) {
                    ok = false; // slot in use by an in-flight frame
                    break;
                }
            }
            if (!ok) {
                break;
            }
            for (int k = 0; k < gf.n_src; ++k) {
                const int slot = (lo + k) % g.src_ring;
                gf.src_slot[k] = slot;
                gf.src_writer[k] = g.src_writer[slot];
                if (g.src_frame[slot] != lo + k &&
                    !g.src_holders[slot].empty()) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            d->cache_cv.wait(lock);
            continue;
        }
        // Phase 2: apply every entry's reservations (the lock is held, so the
        // phase-1 checks are still valid).
        if (fr.token == 0) {
            fr.token = d->next_res_token++;
        }
        for (int gi = 0; gi < d->n_groups; ++gi) {
            auto & g = d->groups[gi];
            auto & gf = fr.g[gi];
            for (int i = 0; i < d->tw; ++i) {
                const int m = std::clamp(n - r + i, 0, nf - 1);
                const int slot = gf.win_slots[i];
                if (gf.win_recompute[i]) {
                    if (g.res_frame[slot] != m) { // first position mapping here
                        g.res_frame[slot] = m;
                        g.res_writer[slot] = n;
                        g.res_ready[slot] = 0;
                    }
                    // every position that maps here drops the previous writer's
                    // dependency: this frame overwrites the slot's contents
                    gf.win_writer[i] = -1;
                }
                g.res_holders[slot].push_back(fr.token);
            }
            for (int k = 0; k < gf.n_src; ++k) {
                const int slot = gf.src_slot[k];
                if (g.src_frame[slot] != lo + k) {
                    g.src_frame[slot] = lo + k;
                    g.src_writer[slot] = n;
                    g.src_ready[slot] = 0;
                    gf.upload_new[k] = true;
                }
                g.src_holders[slot].push_back(fr.token);
            }
        }
        return;
    }
}

// The slots this frame wrote are submitted, so a reader waiting on them may
// record and submit: submission order on the one compute queue plus the
// reader's leading barrier is the whole cross-frame handoff.
static void publish_est_submitted(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        return; // radius 0 slots are private to one frame
    }
    std::lock_guard lock(d->cache_lock);
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        const auto & gf = fr.g[gi];
        for (int i = 0; i < d->tw; ++i) {
            if (gf.win_recompute[i]) {
                g.res_ready[gf.win_slots[i]] = 1;
            }
        }
        for (int k = 0; k < gf.n_src; ++k) {
            if (gf.upload_new[k]) {
                g.src_ready[gf.src_slot[k]] = 1;
            }
        }
    }
    d->cache_cv.notify_all();
}

// A frame that will never submit must not advertise slots it never wrote. The
// ring copies live only in chunk 0, so when that submission did not land the
// source keys are cleared: a later frame then copies those frames itself
// instead of block-matching against whatever the slot held. The failure flag
// releases a reader already blocked on a cleared ready flag (it would otherwise
// hang) and makes it fail instead of reading the slot. The res keys stay as
// published: the per-slice witness in `tags` makes an unwritten stack refuse
// itself and fall back to the source pixel, which the cleared ring keys make
// honest again.
static void fail_pending_frame(BM3DData * d, const Bm3dFrame & fr,
                               bool ring_copied) {
    if (d->radius == 0) {
        return; // radius 0 slots are private to one frame
    }
    std::lock_guard lock(d->cache_lock);
    d->failed = true;
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        const auto & gf = fr.g[gi];
        for (int i = 0; i < d->tw; ++i) {
            if (gf.win_recompute[i]) {
                g.res_ready[gf.win_slots[i]] = 1;
            }
        }
        for (int k = 0; k < gf.n_src; ++k) {
            if (!gf.upload_new[k]) {
                continue;
            }
            g.src_frame[gf.src_slot[k]] = ring_copied ? gf.src_lo + k : -1;
            g.src_ready[gf.src_slot[k]] = ring_copied ? 1 : 0;
        }
    }
    d->cache_cv.notify_all();
}

// True once any frame of this instance has failed: the cache's keys are no
// longer trustworthy, so a reader must fail fast instead of reading a slot.
static bool frame_failed(BM3DData * d) {
    std::lock_guard lock(d->cache_lock);
    return d->failed;
}

// Wait host side until every source slot this frame reads has been submitted by
// its copier. The frame holds those slots, so the writer cannot be re-reserved
// underneath it. With VSFEEL_BM3D_RINGWAIT the copier's copy is additionally
// waited out on the device, which lifts the whole handoff off the barrier's
// scopes (diagnostic: the witness cannot see a ring copy that lands late).
static void wait_src_submitted(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        return;
    }
    std::vector<uint64_t> waits;
    {
        std::unique_lock lock(d->cache_lock);
        d->cache_cv.wait(lock, [&] {
            if (d->failed) {
                return true; // the copier will never submit; let the caller fail
            }
            for (int gi = 0; gi < d->n_groups; ++gi) {
                const auto & g = d->groups[gi];
                const auto & gf = fr.g[gi];
                for (int k = 0; k < gf.n_src; ++k) {
                    if (!gf.upload_new[k] && !g.src_ready[gf.src_slot[k]]) {
                        return false;
                    }
                }
            }
            return true;
        });
        if (!d->ring_wait) {
            return;
        }
        for (int gi = 0; gi < d->n_groups; ++gi) {
            const auto & g = d->groups[gi];
            const auto & gf = fr.g[gi];
            for (int k = 0; k < gf.n_src; ++k) {
                if (gf.upload_new[k]) {
                    continue;
                }
                const int writer = g.src_writer[gf.src_slot[k]];
                if (writer >= 0 &&
                    writer < static_cast<int>(d->chunk0_value.size())) {
                    const uint64_t value = d->chunk0_value[writer];
                    if (value != 0) {
                        waits.push_back(value);
                    }
                }
            }
        }
    }
    for (const uint64_t value : waits) {
        char err[256] {};
        d->gpu->api->gpuExecWaitValue(d->exec, value, err, sizeof(err));
    }
}

// Same for the estimate stacks the aggregation reads.
static void wait_res_submitted(BM3DData * d, const Bm3dFrame & fr) {
    if (d->radius == 0) {
        return;
    }
    std::unique_lock lock(d->cache_lock);
    d->cache_cv.wait(lock, [&] {
        if (d->failed) {
            return true; // the writer will never submit; let the caller fail
        }
        for (int gi = 0; gi < d->n_groups; ++gi) {
            const auto & g = d->groups[gi];
            const auto & gf = fr.g[gi];
            for (int i = 0; i < d->tw; ++i) {
                if (!gf.win_recompute[i] && !g.res_ready[gf.win_slots[i]]) {
                    return false;
                }
            }
        }
        return true;
    });
}

// Release the cache reservations after this frame's aggregation has been
// submitted: queue order already makes a later recompute run after that
// aggregation, and the later frame's leading barrier completes the ordering.
static void release_cache(BM3DData * d, const Bm3dFrame & fr) {
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        const auto & gf = fr.g[gi];
        if (d->radius == 0) {
            give_r0_slot(d, g, fr.slot0[gi]);
            continue;
        }
        std::lock_guard lock(d->cache_lock);
        for (int i = 0; i < d->tw; ++i) {
            auto & h = g.res_holders[gf.win_slots[i]];
            h.erase(std::remove(h.begin(), h.end(), fr.token), h.end());
        }
        for (int k = 0; k < gf.n_src; ++k) {
            auto & h = g.src_holders[gf.src_slot[k]];
            h.erase(std::remove(h.begin(), h.end(), fr.token), h.end());
        }
        d->cache_cv.notify_all();
    }
}

// Window positions whose estimate this frame must compute. A clamped window
// collapses several positions onto one slot and one centre frame, i.e.
// identical dispatches whose fills wipe each other, so only one is kept.
static void collect_est_positions(BM3DData * d, Bm3dGroupFrame & gf, int n) {
    gf.n_pos = 0;
    if (d->radius == 0) {
        gf.est_pos[gf.n_pos++] = 0;
        return;
    }
    for (int i = 0; i < d->tw; ++i) {
        if (!gf.win_recompute[i]) {
            continue;
        }
        const int m_i = std::clamp(n - d->radius + i, 0, d->nframes - 1);
        const int slot = gf.win_slots[i];
        bool duplicate_later = false;
        for (int j = i + 1; j < d->tw && !duplicate_later; ++j) {
            duplicate_later =
                gf.win_recompute[j] && gf.win_slots[j] == slot &&
                std::clamp(n - d->radius + j, 0, d->nframes - 1) == m_i;
        }
        if (!duplicate_later) {
            gf.est_pos[gf.n_pos++] = i;
        }
    }
}

// The three buffers both kernels address, in the order the shader declares
// them: estimate stacks, source ring, destination plane.
static void bm3d_bind(const GPUDevice & gpu, VkCommandBuffer cmd,
                      VkPipelineLayout layout, VkBuffer src, VkBuffer res,
                      VkBuffer dst, VkBuffer tags, VkBuffer skipped,
                      VkBuffer refusal) {
    const VkBuffer bufs[6] { res, src, dst, tags, skipped, refusal };
    gpu_push_buffers(gpu, cmd, layout, bufs, 6);
}

// A whole-command barrier across submissions: the slots are written again here
// (zero-fill, ring copies, atomics) while an earlier frame may still be reading
// or writing them, and only write-after-read needs no memory dependency of its own.
static void bm3d_full_barrier(const GPUDevice & gpu, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask =
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.dstAccessMask =
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    gpu.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

// Zero-fill one entry's result slot and dispatch its estimation kernel. Separate
// so each recomputed position can be recorded into its own command buffer: a
// frame's estimation is the search run over every window position it is missing,
// and on a slow card the total can run past the driver's watchdog window.
static void record_est_position(BM3DData * d, const Bm3dFrame & fr,
                                VkCommandBuffer cmd, int n, int gi, int c,
                                VkBuffer dst_plane) {
    const int r = d->radius;
    const int nf = d->nframes;
    const auto & g = d->groups[gi];
    const auto & gf = fr.g[gi];
    const int i = gf.est_pos[c];
    const int slot = gf.win_slots[i];
    const int m_i = std::clamp(n - r + i, 0, nf - 1);
    if (d->dump) {
        fprintf(stderr, "[d] n=%d entry %d computes slot %d for frame %d\n", n,
                gi, slot, m_i);
    }

    // The whole slot is zeroed in one fill: every plane of the entry packs
    // into it, and one plane's estimation run owns all of them.
    const VkDeviceSize slot_elems =
        static_cast<VkDeviceSize>(g.n_planes) * d->tw * 2 * g.pe;
    const VkDeviceSize res_off = static_cast<VkDeviceSize>(slot) * slot_elems;
    d->gpu->vk->vkCmdFillBuffer(cmd, g.res.buffer, res_off * 4, slot_elems * 4,
                                0);
    // Clear only this exclusively reserved slot's witnesses in the same
    // ordered submission; a filter-wide lazy clear races first frames. The
    // entry's own region starts at its tag base.
    d->gpu->vk->vkCmdFillBuffer(cmd, d->tags.buffer,
                                static_cast<VkDeviceSize>(g.tag_base + slot) *
                                    d->tw * 4,
                                static_cast<VkDeviceSize>(d->tw) * 4, 0);

    // Both the zero-fill and the tag clear precede the estimation dispatch.
    bm3d_full_barrier(*d->gpu, cmd);

    d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  g.bm3d_pipeline);
    // The estimation kernel reads only the estimate stacks and the source ring;
    // the destination binding carries the output plane and is unused here.
    bm3d_bind(*d->gpu, cmd, d->pipeline_layout, g.src.buffer, g.res.buffer,
              dst_plane, d->tags.buffer, d->skipped.buffer, d->refusal.buffer);
    {
        const int32_t pushes[5] {
            static_cast<int32_t>(res_off), m_i, nf,
            static_cast<int32_t>((r == 0) ? fr.slot0[gi] : 0),
            static_cast<int32_t>(static_cast<VkDeviceSize>(g.tag_base + slot) *
                                 d->tw)
        };
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, pushes,
                           sizeof(pushes));
    }
    d->gpu->vk->vkCmdDispatch(cmd, g.bm3d_grid_x, g.bm3d_grid_y, 1);
}

// The input planes one source frame contributes to the ring copy: the frame
// being denoised, plus the basic-estimate clip in final mode.
struct Bm3dWindowCopy {
    VkBuffer source[3] {};
    VkBuffer ref[3] {};
};

// Copy the source window's planes (the union of all windows that this record's
// dispatches may need, clamped to [n-2r, n+2r]) into each entry's src ring.
// Every entry packs [clip][plane] per slot, so a plane is one `pe` hop away and
// the slot stride spans the entry's planes.
static void record_src_copies(BM3DData * d, const Bm3dFrame & fr,
                              VkCommandBuffer cmd,
                              const std::vector<Bm3dWindowCopy> & window) {
    const int clips = d->final ? 2 : 1;
    const auto copy_plane = [&](VkBuffer src, VkBuffer dst, VkDeviceSize offset,
                                VkDeviceSize bytes) {
        VkBufferCopy2 region {};
        region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
        region.srcOffset = 0;
        region.dstOffset = offset * 4;
        region.size = bytes * 4;
        VkCopyBufferInfo2 copy {};
        copy.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
        copy.srcBuffer = src;
        copy.dstBuffer = dst;
        copy.regionCount = 1;
        copy.pRegions = &region;
        d->gpu->vk->vkCmdCopyBuffer2(cmd, &copy);
    };
    for (int gi = 0; gi < d->n_groups; ++gi) {
        const auto & g = d->groups[gi];
        const auto & gf = fr.g[gi];
        for (int k = 0; k < gf.n_src; ++k) {
            if (!gf.upload_new[k]) {
                continue;
            }
            const VkDeviceSize slot_base =
                static_cast<VkDeviceSize>(gf.src_slot[k]) * clips * g.n_planes *
                g.pe;
            const Bm3dWindowCopy & w = window[k];
            for (int pi = 0; pi < g.n_planes; ++pi) {
                const int plane = g.planes[pi];
                // A joint entry's skipped plane is never read: only plane 0
                // feeds block matching and the estimate loop skips the rest.
                if (!d->planes[plane].process && pi != 0) {
                    continue;
                }
                // The clip that is denoised rides in the slot's last clip
                // section; in final mode the Wiener reference clip precedes it.
                copy_plane(w.source[plane], g.src.buffer,
                           slot_base +
                               static_cast<VkDeviceSize>(clips - 1) *
                                   g.n_planes * g.pe +
                               static_cast<VkDeviceSize>(pi) * g.pe,
                           g.pe);
                if (d->final) {
                    copy_plane(w.ref[plane], g.src.buffer,
                               slot_base + static_cast<VkDeviceSize>(pi) * g.pe,
                               g.pe);
                }
            }
        }
    }
}

// Record one estimation submission into the context's command buffer. The
// first chunk carries the ring copies and a leading whole-queue barrier; the
// last carries the trailing barrier that makes the atomic accumulation visible
// to the aggregation. One position per submission (VSFEEL_BM3D_SPLIT, the
// default) keeps a single submission from running past a slow card's watchdog.
static void record_est_chunk(BM3DData * d, VSGPUExecContext * ctx,
                             const Bm3dFrame & fr, int n, int c, int chunks,
                             const std::vector<Bm3dWindowCopy> & window,
                             VkBuffer dst_plane, bool gputrace) {
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);

    // Every chunk starts behind everything already submitted on the queue, not
    // just the first one: chunk 0 carries the ring copies and every later
    // chunk's dispatch reads them, so without this they may execute against
    // whatever the slot held before. A barrier's first scope is every earlier
    // command in submission order on the queue -- the same handoff the
    // cross-frame readers use -- and it also orders this frame's writes (the
    // copies and every slot zero-fill) behind everything already queued, and
    // the overwrite of a recycled slot behind the previous reader's
    // aggregation. Dropping it after chunk 0 re-enables the fill/aggregation
    // race, so it is not an optional barrier.
    bm3d_full_barrier(*d->gpu, cmd);

    if (c == 0) {
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
            d->gpu->vk->vkCmdWriteTimestamp2(
                cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, d->probe.query, 0);
        }
    }

    // A chunk carries position c of every entry that still has one; with the
    // default split that is one position of every entry per submission, so a
    // three-plane frame submits the same number of command buffers as a
    // single-plane one.
    for (int gi = 0; gi < d->n_groups; ++gi) {
        const auto & gf = fr.g[gi];
        if (c >= gf.n_pos) {
            continue;
        }
        if (d->split_est) {
            record_est_position(d, fr, cmd, n, gi, c, dst_plane);
        } else {
            for (int k = 0; k < gf.n_pos; ++k) {
                record_est_position(d, fr, cmd, n, gi, k, dst_plane);
            }
        }
    }

    if (c == chunks - 1) {
        if (gputrace) {
            d->gpu->vk->vkCmdWriteTimestamp2(
                cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, d->probe.query, 1);
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
// downloaded afterwards. One dispatch per processed plane, each with its own
// geometry and its entry's estimate stacks.
static void record_bm3d_agg(BM3DData * d, const Bm3dFrame & fr,
                            VkCommandBuffer cmd, int n,
                            const std::array<VkBuffer, 3> & dst_planes,
                            bool gputrace) {
    const int nf = d->nframes;
    const int r = d->radius;
    bool first_dispatch = true;
    int remaining = 0;
    for (int gi = 0; gi < d->n_groups; ++gi) {
        for (int pi = 0; pi < d->groups[gi].n_planes; ++pi) {
            if (d->planes[d->groups[gi].planes[pi]].process) {
                remaining++;
            }
        }
    }

    for (int gi = 0; gi < d->n_groups; ++gi) {
        const auto & g = d->groups[gi];
        const auto & gf = fr.g[gi];

        // Invariant check: at aggregation time this frame still holds its
        // slots, so each must still record the frame whose stack the
        // aggregation is about to read. A mismatch means the stack belongs
        // to another frame -- the intermittent contaminant band. Radius 0
        // has no window table (its slots are private), so there is nothing
        // to check; the reads need cache_lock like every other table read.
        if (d->trace && r > 0) {
            std::lock_guard lock(d->cache_lock);
            for (int i = 0; i < d->tw; ++i) {
                const int want = std::clamp(n - r + i, 0, nf - 1);
                const int slot = gf.win_slots[i];
                if (g.res_frame[slot] != want) {
                    fprintf(stderr,
                            "[t] n=%d entry %d agg slot %d holds frame %d, "
                            "wants %d (holder %d)\n",
                            n, gi, slot, g.res_frame[slot], want,
                            g.res_writer[slot]);
                }
                if (!g.res_ready[slot]) {
                    fprintf(stderr, "[t] n=%d entry %d agg slot %d NOT READY\n",
                            n, gi, slot);
                }
            }
        }

        for (int pi = 0; pi < g.n_planes; ++pi) {
            const int plane = g.planes[pi];
            const auto & p = d->planes[plane];
            // A joint entry's skipped plane was never computed: its output is
            // the source plane the caller shared in.
            if (!p.process) {
                continue;
            }

            // aggregation: tw stacked slices (clamped frame indices, aggZ blocks)
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          p.agg_pipeline);
            // descriptor bindings do not carry across command buffers: the
            // aggregation is recorded separately from the estimation phase, so
            // without this bind the dispatch runs on undefined descriptor state
            // (black output, and device loss under concurrent submissions)
            bm3d_bind(*d->gpu, cmd, d->pipeline_layout, g.src.buffer,
                      g.res.buffer, dst_planes[plane], d->tags.buffer,
                      d->skipped.buffer, d->refusal.buffer);
            {
                const int clips = d->final ? 2 : 1;
                // The fallback source pixel: the *source* section of this
                // frame's slot in the entry's ring, exactly where
                // record_src_copies put it.
                const int src_slot =
                    (r == 0) ? fr.slot0[gi]
                             : ((n % g.src_ring) + g.src_ring) % g.src_ring;
                const VkDeviceSize src_base =
                    (static_cast<VkDeviceSize>(src_slot) * clips +
                     (clips - 1)) *
                        g.n_planes * g.pe +
                    static_cast<VkDeviceSize>(pi) * g.pe;
                // The kernel rebuilds slot, slice and witness from these; the
                // estimate offsets are then computed in vec4 elements, which is
                // why the plane offset is divided here. Same five ints as the
                // estimation kernel's block, so both share one layout.
                const int32_t pushes[5] { n, nf, (r == 0) ? fr.slot0[gi] : 0,
                                          static_cast<int32_t>(
                                              static_cast<VkDeviceSize>(pi) *
                                              d->tw * 2 * g.pe / 4),
                                          static_cast<int32_t>(src_base) };
                if (d->refusal_mapped && !d->refusal_mapped[0]) {
                    d->refusal_mapped[1] = static_cast<uint32_t>(n);
                }
                gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, pushes,
                                   sizeof(pushes));
            }
            // the estimation kernel's atomic accumulation (and the fill that
            // zeroes the slots) must be visible to the aggregation reads; the
            // aggregation kernel reads with atomic loads, but the RADV driver
            // still needs an explicit barrier for the cross-dispatch visibility
            bm3d_full_barrier(*d->gpu, cmd);
            if (gputrace && first_dispatch) {
                d->gpu->vk->vkCmdWriteTimestamp2(
                    cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, d->probe.query,
                    2);
            }
            d->gpu->vk->vkCmdDispatch(cmd, p.agg_grid_x, p.agg_grid_y, 1);
            if (gputrace && --remaining == 0) {
                d->gpu->vk->vkCmdWriteTimestamp2(
                    cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, d->probe.query,
                    3);
            }
            first_dispatch = false;
        }
    }
}

static const VSFrame * VS_CC BM3DGetFrame(int n, int activationReason,
                                          void * instanceData,
                                          [[maybe_unused]] void ** frameData,
                                          VSFrameContext * frameCtx,
                                          VSCore * core, const VSAPI * vsapi) {

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
        // The centre source frame is the property donor and, for a plane below
        // its sigma threshold, the plane's owner: newVideoFrame2 propagates GPU
        // residency -- and each shared plane's producer pair -- from it. A plane
        // is processed on its own unless chroma=True packs the three of a
        // 4:4:4 clip into one entry.
        const VSFrame * center = vsapi->getFrameFilter(n, d->node, frameCtx);
        bool all_process = true;
        for (int p = 0; p < d->num_planes; ++p) {
            all_process = all_process && d->planes[p].process;
        }
        VSFrame * dst = nullptr;
        if (all_process) {
            dst = d->gpu->api->newGPUVideoFrame(&d->vi->format, d->vi->width,
                                                d->vi->height, center, core);
        } else {
            int pl[3] {};
            const VSFrame * fr[3] {};
            for (int p = 0; p < d->num_planes; ++p) {
                pl[p] = p;
                fr[p] = d->planes[p].process ? nullptr : center;
            }
            dst = vsapi->newVideoFrame2(&d->vi->format, d->vi->width,
                                        d->vi->height, fr, pl, center, core);
        }
        vsapi->freeFrame(center);
        if (!dst) {
            vsapi->setFilterError("BM3D: failed to allocate the output frame",
                                  frameCtx);
            return nullptr;
        }
        // All planes are processed or the node would have been passed through
        // at creation, so there is nothing further to do for a skipped one.

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

        const int n_src = fr.g[0].n_src;
        std::vector<Bm3dWindowCopy> window(static_cast<size_t>(n_src));
        std::vector<const VSFrame *> sources;
        // Chunk 0 is the submission that records the ring copies; until it
        // lands, the src keys this frame committed at acquire are promises, not
        // facts (see fail_pending_frame).
        bool ring_copied = false;

        const auto set_error =
            [&](const std::string & error_message) -> const VSFrame * {
            vsfeel_trace_error("BM3D", n, error_message, d->gpu.get());
            // A reader may be waiting for this frame's estimation; it must be
            // released instead of blocking forever on a signal that is never
            // coming. This frame's own output is an error either way.
            fail_pending_frame(d, fr, ring_copied);
            release_cache(d, fr);
            for (const VSFrame * f : sources) {
                vsapi->freeFrame(f);
            }
            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        // Fault injection for the error-path test: fail after the reservations
        // exist but before anything was recorded, so the test exercises exactly
        // the window where a src key is committed but its copy never lands.
        if (n == d->fault_frame) {
            return set_error("injected estimation fault");
        }

        // Copy only the frames whose cache slots the acquire reserved. The
        // needed range is the union of every window that this record's
        // dispatches may read: [clamp(n-2r), clamp(n+2r)], and every entry
        // reserves the same range, so one upload flag per frame covers them all.
        for (int k = 0; k < n_src; ++k) {
            bool upload = false;
            for (int gi = 0; gi < d->n_groups; ++gi) {
                upload = upload || fr.g[gi].upload_new[k];
            }
            if (!upload) {
                continue;
            }
            const int f = fr.g[0].src_lo + k;
            const VSFrame * src = vsapi->getFrameFilter(f, d->node, frameCtx);
            for (int plane = 0; plane < d->num_planes; ++plane) {
                VSVulkanPlaneInfo plane_info {};
                if (d->gpu->api->getGPUPlane(src, plane, &plane_info)) {
                    vsapi->freeFrame(src);
                    return set_error("clip " + std::to_string(f) + " plane " +
                                     std::to_string(plane) +
                                     " is not GPU resident");
                }
                window[k].source[plane] = plane_info.buffer;
            }
            sources.push_back(src);
            if (d->final) {
                const VSFrame * rsrc =
                    vsapi->getFrameFilter(f, d->ref_node, frameCtx);
                for (int plane = 0; plane < d->num_planes; ++plane) {
                    VSVulkanPlaneInfo plane_info {};
                    if (d->gpu->api->getGPUPlane(rsrc, plane, &plane_info)) {
                        vsapi->freeFrame(rsrc);
                        return set_error("ref clip " + std::to_string(f) +
                                         " plane " + std::to_string(plane) +
                                         " is not GPU resident");
                    }
                    window[k].ref[plane] = plane_info.buffer;
                }
                sources.push_back(rsrc);
            }
        }
        auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        // The destination planes the aggregation writes: the output frame's own
        // storage, so nothing is downloaded.
        std::array<VkBuffer, 3> dst_planes {};
        for (int plane = 0; plane < d->num_planes; ++plane) {
            if (!d->planes[plane].process) {
                continue;
            }
            VSVulkanPlaneInfo plane_info {};
            if (d->gpu->api->getGPUPlane(dst, plane, &plane_info)) {
                return set_error("the output frame is not GPU resident");
            }
            dst_planes[plane] = plane_info.buffer;
        }

        int max_pos = 0;
        for (int gi = 0; gi < d->n_groups; ++gi) {
            collect_est_positions(d, fr.g[gi], n);
            max_pos = std::max(max_pos, fr.g[gi].n_pos);
        }
        const int chunks = (d->split_est && max_pos > 0) ? max_pos : 1;

        if (d->trace) {
            for (int gi = 0; gi < d->n_groups; ++gi) {
                const auto & gf = fr.g[gi];
                for (int k = 0; k < gf.n_src; ++k) {
                    if (!gf.upload_new[k]) {
                        fprintf(stderr,
                                "[t] n=%d entry %d reads src slot %d (copier "
                                "w=%d)\n",
                                n, gi, gf.src_slot[k], gf.src_writer[k]);
                    }
                }
                for (int i = 0; i < d->tw; ++i) {
                    if (!gf.win_recompute[i]) {
                        fprintf(stderr,
                                "[t] n=%d entry %d reads res slot %d (writer "
                                "w=%d)\n",
                                n, gi, gf.win_slots[i], gf.win_writer[i]);
                    }
                }
            }
        }

        // The source window this frame reads may have been copied by other
        // frames; wait host side until those copies have been submitted. This
        // frame holds the slots, so no writer can be re-reserved underneath it.
        wait_src_submitted(d, fr);
        if (frame_failed(d)) {
            return set_error("a source frame's estimation never submitted");
        }

        // The estimation is submitted first, so the GPU stays busy with the
        // heavy kernels while the host waits for the writers of the
        // aggregation slots.
        vsfeel_trace_mark("sub est");
        for (int c = 0; c < chunks; ++c) {
            char errbuf[512] {};
            VSGPUExecContext * ctx =
                d->gpu->api->gpuExecAcquire(d->exec, errbuf, sizeof(errbuf));
            if (!ctx) {
                return set_error("could not acquire a recording context: "s +
                                 errbuf);
            }
            if (c == 0) {
                // The producer pairs of the frames this submission copies from
                // become device-side waits, and the pool keeps the frames alive
                // until the submission completes.
                for (const VSFrame * f : sources) {
                    d->gpu->api->gpuExecReadsFrame(ctx, f);
                }
            }
            record_est_chunk(d, ctx, fr, n, c, chunks, window, dst_planes[0],
                             gputrace);
            uint64_t signaled = 0;
            if (d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf,
                                           sizeof(errbuf))) {
                return set_error("estimation submit failed: "s + errbuf);
            }
            if (c == 0 && n < static_cast<int>(d->chunk0_value.size())) {
                // The ring copy rides in chunk 0; RINGWAIT readers wait this out.
                d->chunk0_value[n] = signaled;
            }
            if (c == 0) {
                ring_copied = true;
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
        if (frame_failed(d)) {
            return set_error("an estimate stack's writer never submitted");
        }

        char aerr[512] {};
        VSGPUExecContext * agg_ctx =
            d->gpu->api->gpuExecAcquire(d->exec, aerr, sizeof(aerr));
        if (!agg_ctx) {
            return set_error("could not acquire a recording context: "s + aerr);
        }
        // The pool publishes the output planes' producer pairs at submit.
        for (int plane = 0; plane < d->num_planes; ++plane) {
            if (d->planes[plane].process) {
                d->gpu->api->gpuExecWritesPlane(agg_ctx, dst, plane);
            }
        }
        vsfeel_trace_mark("sub agg");
        record_bm3d_agg(d, fr, d->gpu->api->gpuExecCommandBuffer(agg_ctx), n,
                        dst_planes, gputrace);
        uint64_t agg_value = 0;
        if (d->gpu->api->gpuExecSubmit(agg_ctx, &agg_value, aerr,
                                       sizeof(aerr))) {
            return set_error("aggregation submit failed: "s + aerr);
        }
        auto t4 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (d->trace) {
            fprintf(stderr, "[t] n=%d submitted (%d est chunks)\n", n, chunks);
        }

        if (gputrace) {
            // The stamps were written by this frame's submissions; waiting the
            // aggregation out leaves the query results final.
            char perr[256] {};
            if (d->gpu->api->gpuExecWaitValue(d->exec, agg_value, perr,
                                              sizeof(perr)) == gdDrained) {
                uint64_t ts[4] {};
                if (d->gpu->vk->vkGetQueryPoolResults(
                        d->gpu->device, d->probe.query, 0, 4, sizeof(ts), ts,
                        sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                    const double period = d->gpu->limits.timestampPeriod;
                    static std::atomic<uint64_t> ts_k {}, ts_a {};
                    static std::atomic<uint32_t> ts_nf {};
                    const uint32_t nq = ts_nf.fetch_add(1) + 1;
                    const auto ns = [period](uint64_t a, uint64_t b) {
                        return static_cast<uint64_t>(
                            std::llround(static_cast<double>(b - a) * period));
                    };
                    if (ts[0] && ts[1]) {
                        ts_k += ns(ts[0], ts[1]);
                    }
                    if (ts[2] && ts[3]) {
                        ts_a += ns(ts[2], ts[3]);
                    }
                    if (nq % 50 == 0) {
                        fprintf(
                            stderr,
                            "[bm3dgpu] n=%u kernel=%.3f agg=%.3f (ms) disp=%d\n",
                            nq, ts_k.load() / double(nq) / 1e6,
                            ts_a.load() / double(nq) / 1e6, max_pos);
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
                    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                        .count());
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

static void VS_CC BM3DFree(void * instanceData, [[maybe_unused]] VSCore * core,
                           const VSAPI * vsapi) {
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

static void VS_CC BM3DCreate(const VSMap * in, VSMap * out,
                             [[maybe_unused]] void * userData, VSCore * core,
                             const VSAPI * vsapi) {

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
    d->fault_frame = env_int("VSFEEL_BM3D_FAULT", -1);
    // Diagnostic: wait the ring copier's submission out host side instead of
    // riding the handoff barrier (see wait_src_submitted).
    d->ring_wait = env_flag("VSFEEL_BM3D_RINGWAIT");

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
            rvi->format.subSamplingH != d->vi->format.subSamplingH) {
            return set_error("\"ref\" must be of the same format as \"clip\"");
        }
        if (rvi->width != d->vi->width || rvi->height != d->vi->height) {
            return set_error(
                "\"ref\" must be of the same dimensions as \"clip\"");
        }
        if (rvi->numFrames != d->vi->numFrames) {
            return set_error(
                "\"ref\" must be of the same number of frames as \"clip\"");
        }
        d->final = true;
    }

    if (d->vi->width <= 0 || d->vi->height <= 0 ||
        d->vi->format.sampleType != stFloat ||
        d->vi->format.bitsPerSample != 32) {
        return set_error("only constant format 32 bit float input supported");
    }

    if (d->vi->format.colorFamily != cfGray &&
        d->vi->format.colorFamily != cfYUV &&
        d->vi->format.colorFamily != cfRGB) {
        return set_error("only Gray, YUV and RGB input are supported");
    }
    d->num_planes = d->vi->format.numPlanes;

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

    // Per-plane sigma, exactly as the reference's perPlane: a missing entry
    // repeats the previous one, the first defaults to 3. A plane whose sigma is
    // below FLT_EPSILON is a bit-exact source copy (the reference's PROC_MASK,
    // and its kernel's epsilon test). With every plane below it BM3Dv2 is the
    // source clip itself, which is the all-zero shortcut both references take.
    std::array<float, 3> sigma {};
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] =
            static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &error));
        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (!std::isfinite(sigma[i]) || sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be finite and non-negative");
        }
    }
    bool any_process = false;
    for (int i = 0; i < d->num_planes; ++i) {
        d->planes[i].process = sigma[i] >= FLT_EPSILON;
        any_process = any_process || d->planes[i].process;
    }
    if (!any_process) {
        // Nothing to denoise: hand the source clip straight back, as the
        // references' BM3Dv2 does. Before any GPU resource exists, so the
        // instance is just the two node references.
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
            d->ref_node = nullptr;
        }
        vsapi->mapConsumeNode(out, "clip", d->node, maReplace);
        return;
    }

    // match the reference sigma scaling exactly (different factor for the
    // final Wiener pass); a skipped plane keeps exactly zero so the kernel's
    // epsilon test cannot disagree with the unscaled decision above
    const float sigma_factor = d->final ? std::bit_cast<float>(0x3e40c0c1u)
                                        : std::bit_cast<float>(0x3f021bb6u);
    for (int i = 0; i < d->num_planes; ++i) {
        d->planes[i].sigma =
            d->planes[i].process ? sigma[i] * sigma_factor : 0.0f;
    }

    std::array<int, 3> block_step {};
    for (int i = 0; i < std::ssize(block_step); ++i) {
        block_step[i] =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "block_step", i, &error));
        if (error) {
            block_step[i] = (i == 0) ? 8 : block_step[i - 1];
        } else if (block_step[i] <= 0 || block_step[i] > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }
    }

    std::array<int, 3> bm_range {};
    for (int i = 0; i < std::ssize(bm_range); ++i) {
        bm_range[i] =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "bm_range", i, &error));
        if (error) {
            bm_range[i] = (i == 0) ? 9 : bm_range[i - 1];
        } else if (bm_range[i] <= 0 || bm_range[i] > kMaxSearchRange) {
            return set_error("\"bm_range\" must be in range [1, 8192]");
        }
    }

    int64_t radius_raw = vsapi->mapGetInt(in, "radius", 0, &error);
    d->radius = vsh::int64ToIntS(radius_raw);
    if (error) {
        d->radius = 0;
    }
    if (d->radius < 0 || d->radius > MAX_RADIUS) {
        return set_error("\"radius\" must be in range [0, "s +
                         std::to_string(MAX_RADIUS) + "]");
    }
    d->tw = 2 * d->radius + 1;

    std::array<int, 3> ps_num {};
    for (int i = 0; i < std::ssize(ps_num); ++i) {
        ps_num[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_num", i, &error));
        if (error) {
            ps_num[i] = (i == 0) ? 2 : ps_num[i - 1];
        } else if (ps_num[i] <= 0 || ps_num[i] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }
    }

    std::array<int, 3> ps_range {};
    for (int i = 0; i < std::ssize(ps_range); ++i) {
        ps_range[i] =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_range", i, &error));
        if (error) {
            ps_range[i] = (i == 0) ? 4 : ps_range[i - 1];
        } else if (ps_range[i] <= 0 || ps_range[i] > kMaxSearchRange) {
            return set_error("\"ps_range\" must be in range [1, 8192]");
        }
    }

    for (int i = 0; i < d->num_planes; ++i) {
        d->planes[i].block_step = block_step[i];
        d->planes[i].bm_range = bm_range[i];
        d->planes[i].ps_num = ps_num[i];
        d->planes[i].ps_range = ps_range[i];
    }

    // chroma=True: one joint entry over the three planes of a 4:4:4 clip, whose
    // groups come from luma. The references require YUV444 for it.
    d->joint = vsapi->mapGetInt(in, "chroma", 0, &error) != 0;
    if (d->joint &&
        (d->vi->format.colorFamily != cfYUV ||
         d->vi->format.subSamplingW != 0 || d->vi->format.subSamplingH != 0)) {
        return set_error("clip format must be YUV444 when \"chroma\" is true");
    }

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
    const int src_ring =
        (d->radius == 0) ? kInflightFrames : 4 * d->radius + kInflightFrames;
    // One in-flight frame needs the stacks of centre frames [n-r, n+r], so
    // kInflightFrames concurrent frames span kInflightFrames + 2r slots. That
    // working set is the default: the estimate cache is the largest allocation,
    // and on an 8 GiB card the slack below is the difference between running
    // radius 4 and failing to allocate. VSFEEL_BM3D_CACHE=1 adds a whole extra
    // window, so an out-of-order (seek) request finds a warm slot instead of
    // waiting in the acquire, for tw/(ns+2r+tw) more VRAM.
    const int res_working_set = kInflightFrames + 2 * d->radius;
    const bool cache_slack = env_int("VSFEEL_BM3D_CACHE", 0) != 0;
    const int res_cap =
        (d->radius == 0)
            ? kInflightFrames
            : (cache_slack ? res_working_set + d->tw : res_working_set);

    const int extractor_exp =
        vsh::int64ToIntS(vsapi->mapGetInt(in, "extractor_exp", 0, &error));
    // bm3dvk (the preferred reference where the two disagree) accepts [0, 127]:
    // a negative exponent pre-rounds the addends below any representable value
    // and 2^128 is already infinite, which turns the aggregation into NaN.
    if (extractor_exp < 0 || extractor_exp > 127) {
        return set_error("\"extractor_exp\" must be in range [0, 127]");
    }
    d->extractor =
        (extractor_exp != 0) ? std::ldexp(1.0f, extractor_exp) : 0.0f;

    d->nframes = d->vi->numFrames;

    {
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

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
    d->cas_atomics =
        env_flag("VSFEEL_BM3D_CAS") || !d->gpu->feat_atomic_float32_add;
    if (vsfeel_device_info_enabled()) {
        fprintf(stderr, "[bm3d] aggregation: %s\n",
                d->cas_atomics
                    ? "CAS loop (no buffer float32 add atomics available)"
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
        const auto result = gpu_push_set_layout(*d->gpu, 6);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        const auto result = // 5 ints, the same block both kernels take
            gpu_pipeline_layout(*d->gpu, d->set_layout, 5 * sizeof(int32_t));
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
    std::array<int, 3> plane_stride_elems {};
    {
        VSFrame * probe = vsapi->newVideoFrame(&d->vi->format, d->vi->width,
                                               d->vi->height, nullptr, core);
        if (probe == nullptr) {
            return set_error(
                "could not allocate a probe frame to read the plane stride");
        }
        for (int plane = 0; plane < d->num_planes; ++plane) {
            plane_stride_elems[plane] = static_cast<int>(
                vsapi->getStride(probe, plane) / sizeof(float));
        }
        vsapi->freeFrame(probe);
    }

    // Per-plane geometry: plane 0 at full size, the chroma planes subsampled by
    // the format (no subsampling for 4:4:4 and RGB).
    const int subW = d->vi->format.subSamplingW;
    const int subH = d->vi->format.subSamplingH;
    for (int plane = 0; plane < d->num_planes; ++plane) {
        auto & p = d->planes[plane];
        p.width = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        p.height = (plane == 0) ? d->vi->height : d->vi->height >> subH;
        p.stride = plane_stride_elems[plane];
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        if (p.stride < p.width) {
            return set_error("the core reported an unexpected plane stride");
        }
        // The kernel clamps block origins to (dimension - 8), so a smaller
        // plane would address before its buffer. It only ever reads processed
        // planes, and a joint entry's luma, which is processed in every joint
        // configuration that gets this far (the all-skipped case returned the
        // source clip above).
        if (p.process && (p.width < 8 || p.height < 8)) {
            char msg[200];
            snprintf(msg, sizeof(msg),
                     "every denoised plane must be at least 8x8; plane %d is "
                     "%dx%d",
                     plane, p.width, p.height);
            return set_error(msg);
        }
    }

    // Entries: one per processed plane, alone with its own geometry and block
    // matching, or one joint entry over all three planes of a 4:4:4 clip under
    // chroma=True (luma's groups, per-plane sigmas). Every plane of an entry
    // shares one geometry, which is what lets one packed buffer hold them.
    d->n_groups = 0;
    if (d->joint) {
        for (int plane = 0; plane < d->num_planes; ++plane) {
            if (d->planes[plane].stride != d->planes[0].stride) {
                return set_error("chroma=True needs a single row stride across "
                                 "the three planes");
            }
        }
        auto & g = d->groups[d->n_groups++];
        g.n_planes = d->num_planes;
        for (int i = 0; i < d->num_planes; ++i) {
            g.planes[i] = i;
        }
        g.pe = d->planes[0].pe;
    } else {
        for (int plane = 0; plane < d->num_planes; ++plane) {
            if (!d->planes[plane].process) {
                continue;
            }
            auto & g = d->groups[d->n_groups++];
            g.n_planes = 1;
            g.planes[0] = plane;
            g.pe = d->planes[plane].pe;
        }
    }

    VkDeviceSize src_total = 0;
    VkDeviceSize res_total = 0;
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        g.src_ring = src_ring;
        g.res_cap = res_cap;
        g.tag_base = gi * res_cap;
        // In final mode each ring slot holds [ref][source], so the ring doubles.
        const int clips = d->final ? 2 : 1;
        g.src_size =
            static_cast<VkDeviceSize>(g.src_ring) * clips * g.n_planes * g.pe;
        g.res_size = static_cast<VkDeviceSize>(g.res_cap) * d->tw * 2 *
                     g.n_planes * g.pe;
        // Both buffers are addressed through signed 32-bit offsets: the
        // estimation kernel takes a slot's result base as a push constant and
        // computes the source ring's through src_search, so a region at or
        // above 2^31 floats wraps and reads or writes outside its slot.
        if (g.res_size > static_cast<VkDeviceSize>(INT32_MAX)) {
            char msg[256];
            snprintf(
                msg, sizeof(msg),
                "frame is too large: the estimate cache needs %llu floats per "
                "entry (radius %d), which overflows the 32-bit kernel "
                "addressing; reduce radius",
                static_cast<unsigned long long>(g.res_size), d->radius);
            return set_error(msg);
        }
        if (g.src_size > static_cast<VkDeviceSize>(INT32_MAX)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "frame is too large: the source ring needs %llu floats "
                     "(radius %d), which overflows the 32-bit kernel "
                     "addressing; reduce radius",
                     static_cast<unsigned long long>(g.src_size), d->radius);
            return set_error(msg);
        }
        src_total += g.src_size;
        res_total += g.res_size;
    }

    // shared buffers
    d->tags_size = static_cast<VkDeviceSize>(d->n_groups) * res_cap * d->tw;
    {
        {
            std::string err =
                gpu_make_buffer(*d->gpu, core, d->tags_size * 4, d->tags,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (!err.empty()) {
                return set_error("the per-slice frame witness could not be "
                                 "allocated: " +
                                 err);
            }
        }
        {
            std::string err =
                gpu_make_buffer(*d->gpu, core, 4, d->skipped,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!err.empty()) {
                return set_error("the skip counter could not be allocated: " +
                                 err);
            }
            d->skipped_mapped =
                static_cast<volatile uint32_t *>(d->skipped.mapped);
            if (d->skipped_mapped) {
                *d->skipped_mapped = 0;
            }
        }
        {
            std::string err =
                gpu_make_buffer(*d->gpu, core, 16, d->refusal,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!err.empty()) {
                return set_error("the refusal report could not be allocated: " +
                                 err);
            }
            d->refusal_mapped =
                static_cast<volatile uint32_t *>(d->refusal.mapped);
            for (int i = 0; i < 4; ++i) {
                if (d->refusal_mapped) {
                    d->refusal_mapped[i] = 0;
                }
            }
        }

        // Both buffers come from the core's pool, so they count against the
        // VRAM budget the frame cache and the thread pool's admission control
        // also see. The source ring only ever receives copies; the estimate
        // cache is filled and read by kernels.
        for (int gi = 0; gi < d->n_groups; ++gi) {
            auto & g = d->groups[gi];
            std::string err =
                gpu_make_buffer(*d->gpu, core, g.src_size * 4, g.src,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (!err.empty()) {
                return set_error(err);
            }
            err = gpu_make_buffer(*d->gpu, core, g.res_size * 4, g.res,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (!err.empty()) {
                // The estimate cache is by far the largest allocation, so an
                // out-of-memory here is the usual "radius too high for this
                // card"; name the size and what shrinks it.
                char msg[320];
                snprintf(msg, sizeof(msg),
                         "%s; the estimate cache needs %.0f MiB (radius %d): "
                         "lower radius",
                         err.c_str(),
                         static_cast<double>(g.res_size) * 4.0 /
                             (1024.0 * 1024.0),
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
        checkVK(d->gpu->vk->vkCreateQueryPool(dev, &qp_info, nullptr,
                                              &d->probe.query));
        d->probe.enabled = true;
    }

    // pipelines: one estimation pipeline per entry (its planes share the
    // geometry and the search parameters, only the sigmas differ) and one
    // aggregation pipeline per processed plane.
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        const auto & first = d->planes[g.planes[0]];
        {
            const uint32_t * code = d->cas_atomics ? bm3d_cas_spv : bm3d_spv;
            const size_t code_size =
                d->cas_atomics ? bm3d_cas_spv_size : bm3d_spv_size;
            const auto result = create_bm3d_pipeline(
                *d->gpu, *d, g, code, code_size, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            g.bm3d_pipeline = std::get<VkPipeline>(result);
        }
        g.bm3d_grid_x = static_cast<uint32_t>(
            (first.width + 4 * first.block_step - 1) / (4 * first.block_step));
        g.bm3d_grid_y = static_cast<uint32_t>(
            (first.height + first.block_step - 1) / first.block_step);
        for (int pi = 0; pi < g.n_planes; ++pi) {
            auto & p = d->planes[g.planes[pi]];
            if (!p.process) {
                continue; // a joint entry's skipped plane is never dispatched
            }
            const auto result =
                create_agg_pipeline(*d->gpu, p, g, *d, bm3d_agg_spv,
                                    bm3d_agg_spv_size, d->pipeline_layout);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            p.agg_pipeline = std::get<VkPipeline>(result);
            p.agg_grid_x = static_cast<uint32_t>((p.stride + 127) / 128);
            p.agg_grid_y = static_cast<uint32_t>((p.height + 7) / 8);
        }
    }

    // One exec pool per instance, on the core's compute queue: it owns the
    // command buffers, the timeline and the backpressure, and the frame path
    // records and submits through it.
    {
        char err[512] {};
        d->exec =
            d->gpu->api->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (!d->exec) {
            return set_error("createGPUExecPool failed: "s + err);
        }
    }

    // Per-instance VRAM budget. Both shared buffers come from the core's pool,
    // so this is the whole footprint now: there is no per-stream staging and no
    // download buffer. Gated by an env flag so a normal creation prints nothing.
    if (vsfeel_debug_flag("VSFEEL_BM3D_VRAM")) {
        const double mib = 1024.0 * 1024.0;
        const double total = static_cast<double>(src_total + res_total) * 4.0;
        fprintf(
            stderr,
            "[bm3d] vram: src=%.1f MiB res=%.1f MiB (%.0f%% of total) -> total=%.1f MiB "
            "(planes=%d entries=%d radius=%d inflight=%d res_cap=%d src_ring=%d "
            "stride=%d)\n",
            static_cast<double>(src_total) * 4.0 / mib,
            static_cast<double>(res_total) * 4.0 / mib,
            100.0 * static_cast<double>(res_total) / (total > 0 ? total : 1.0),
            total / mib, d->num_planes, d->n_groups, d->radius, kInflightFrames,
            res_cap, src_ring, d->planes[0].stride);
    }

    d->chunk0_value.assign(static_cast<size_t>(d->nframes), 0);
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        g.src_frame.assign(g.src_ring, -1);
        g.src_writer.assign(g.src_ring, -1);
        g.src_ready.assign(g.src_ring, 0);
        g.src_holders.resize(g.src_ring);
        g.res_frame.assign(g.res_cap, -1);
        g.res_writer.assign(g.res_cap, -1);
        g.res_ready.assign(g.res_cap, 0);
        g.res_holders.resize(g.res_cap);
        g.r0_free.reserve(kInflightFrames);
        for (int i = 0; i < kInflightFrames; ++i) {
            g.r0_free.push_back(i);
        }
    }

    BM3DData * data = d.release();

    // A temporal filter requests frames outside n, which the strict-spatial
    // policy does not permit; only radius 0 is purely spatial.
    const VSRequestPattern policy =
        data->radius > 0 ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2] = { { data->node, policy },
                                   { data->ref_node, policy } };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        "BM3D", data->vi, BM3DGetFrame, BM3DFree, fmParallel, ffGPUOutput, deps,
        data->ref_node ? 2 : 1, data, core);
    if (result == nullptr) {
        // The core returns nullptr without running the free callback when the
        // node constructor throws, so release the instance through it here.
        BM3DFree(data, core, vsapi);
        vsapi->mapSetError(out, "BM3D: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bm3dv2(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction("BM3Dv2",
                             "clip:vnode:gpu;"
                             "ref:vnode:gpu:opt;"
                             "sigma:float[]:opt;"
                             "block_step:int[]:opt;"
                             "bm_range:int[]:opt;"
                             "radius:int:opt;"
                             "ps_num:int[]:opt;"
                             "ps_range:int[]:opt;"
                             "chroma:int:opt;"
                             "num_streams:int:opt;"
                             "extractor_exp:int:opt;"
                             "device_id:int:opt;",
                             "clip:vnode:gpu;", BM3DCreate, nullptr, plugin);
}
