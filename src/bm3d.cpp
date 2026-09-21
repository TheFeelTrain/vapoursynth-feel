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

struct Bm3dStream {
    VkCommandPool pool {};
    VkCommandBuffer cmd {};       // == est_cmds[0], kept for destroy_common
    VkCommandBuffer cmd_agg {};   // aggregation phase: recorded after the waits
    // One command buffer per recomputed window position, and how many this
    // frame recorded. A single submission holding all of them can run past
    // Windows' TDR watchdog (about 2 s) on a slow card, where the driver's
    // reset fails and the machine freezes instead of reporting a lost device.
    std::array<VkCommandBuffer, 2 * MAX_RADIUS + 1> est_cmds {};
    int est_cb_count {};
    // This stream's own timeline: the device-side clock every cross-frame
    // dependency here is expressed on. One per stream rather than one for the
    // instance, because a timeline's signal values must increase in submission
    // order and only a stream submits in a single order (see gpu_submit).
    VSGPUTimeline * tl {};
    VkSemaphore timeline {};
    VkQueryPool ts_query {};
    int stream_id {};        // index of this stream in the pool
    uint64_t seq {1};        // next monotonic timeline value this stream signals
    // Source frames this stream's in-flight submission copies from. The core
    // recycles a frame once its last reference goes, so they are held until the
    // stream's drain value says the copy is done.
    std::vector<const VSFrame *> held;
    // Timeline value the previous frame on this stream will signal when its
    // work is done; 0 when nothing is in flight. The next take waits it before
    // re-recording this stream's command buffers. This is the whole in-flight
    // gate now that a frame returns with its submission still running.
    uint64_t drain_value {};
    // Dispatches the last frame recorded, for the deferred GPU-timing print.
    int last_ndisp {};
    // cache reservations for the current frame (radius > 0)
    std::array<int, 9> win_slots {};      // res cache slot of each window frame
    std::array<int, 9> win_writers {};    // frame that wrote each window slot
    std::array<int, 9> win_writer_stream {};  // that writer's stream id
    // identity of the slot's writer, captured under the cache lock when this
    // frame reserved the slot: waiting on (semaphore, value) is exact, while
    // recovering it from the frame number (a modulo-64 table) goes stale as
    // soon as the table entry is reused
    std::array<VSGPUTimeline *, 9> win_writer_tl {};
    std::array<uint64_t, 9> win_writer_value {};
    std::array<bool, 9> win_recompute {}; // true where this frame must recompute the slot
    std::array<bool, 4 * MAX_RADIUS + 1> upload_new {};  // src frames this frame copies
    std::array<int, 4 * MAX_RADIUS + 1> src_writers {};  // copier of each window src slot
    std::array<VSGPUTimeline *, 4 * MAX_RADIUS + 1> src_writer_tl {};
    std::array<uint64_t, 4 * MAX_RADIUS + 1> src_writer_value {};
    // unique token identifying this frame's cache reservation. A frame index
    // is not a unique holder identity: the scheduler can process the same
    // frame twice at once, and erasing holders by value would then drop both
    // entries at the first release, freeing a slot another reader still uses.
    uint64_t res_token {};
};


struct BM3DData {
    VSNode * node;
    VSNode * ref_node {};   // optional basic-estimate clip (final/Wiener pass)
    const VSVideoInfo * vi;

    int radius, num_streams = 2;
    int tw;                          // 2 * radius + 1
    float sigma;                     // scaled luma sigma
    float sigma_u, sigma_v;
    int block_step, bm_range, ps_num, ps_range;
    bool process;
    bool chroma;
    bool final {};                   // true when a "ref" clip is given
    float extractor;
    bool cas_atomics {};             // aggregate with the CAS kernel (no float atomics)

    std::shared_ptr<GPUDevice> gpu;
    // Only the destructor needs it, and the destructor has no VSAPI argument.
    const VSAPI * vsapi {};
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    std::array<Bm3dPlane, 3> planes {};
    int n_planes {};

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
    std::vector<int> src_frame {};   // frame index whose data each src slot holds
    std::vector<int> src_writer {};  // frame that reserved each src slot for copying
    // Who last wrote each slot, stored with the reservation itself. The writer
    // is a frame number *and* the (timeline, value) it signals: the frame
    // number alone is not a stable key (a later frame can reuse it), and a
    // stale lookup makes a reader wait on an unrelated frame's unsubmitted
    // value -- a cycle when that frame is itself waiting for the reader.
    std::vector<VSGPUTimeline *> src_writer_tl {};
    std::vector<uint64_t> src_writer_value {};
    std::vector<std::vector<uint64_t>> src_holders {};  // reservation tokens
    std::vector<int> res_frame {};   // frame index whose stack each res slot holds
    std::vector<int> res_writer {};  // frame that computed each res slot's content
    std::vector<VSGPUTimeline *> res_writer_tl {};
    std::vector<uint64_t> res_writer_value {};
    std::vector<int> res_writer_stream {};  // stream that reserved each res slot
    std::vector<std::vector<uint64_t>> res_holders {};  // reservation tokens
    uint64_t next_res_token {1};
    std::mutex cache_lock;
    std::condition_variable cache_cv;
    // Host-side record of which estimation submits have been issued, per stream.
    // A reader's aggregation waits device-side on the estimation timelines of
    // the frames that filled its cache slots; on a shared queue that wait can
    // stall the FIFO ahead of the submit that would signal it, so the reader
    // waits here for the *submission* (not the completion) first.
    std::vector<uint64_t> stream_submitted {};
    FramePool<Bm3dStream> pool;

    // Env-gated host-path probe (VSFEEL_BM3D_TIMING=1). The stage split is the
    // prerequisite for any transfer-path change: kernel time alone says nothing
    // about whether the frame is GPU- or host-bound. Reset the clock after every
    // blocking acquire so a wait never leaks into the next stage.
    bool host_timing { false };
    // Cached at creation: the timestamp query pool only exists when the env
    // var was set then, so recording must not be driven by a frame-time getenv.
    bool gpu_trace { false };
    // Trace/dump flags are read once: the frame path used to call getenv (and
    // the legacy-name fallback) sixteen times per frame on the default path.
    bool trace { false };
    bool dump { false };
    // Submit one command buffer per recomputed window position instead of one
    // holding them all (VSFEEL_BM3D_SPLIT=0 restores the single submission).
    bool split_est { true };
    std::atomic<uint64_t> ht_take_ns {}, ht_acquire_ns {}, ht_upload_ns {},
        ht_record_ns {}, ht_srcwait_ns {}, ht_agg_ns {}, ht_fence_ns {},
        ht_down_ns {}, ht_total_ns {}, ht_n {};

    ~BM3DData() {
        if (host_timing && ht_n.load()) {
            const double n = static_cast<double>(ht_n.load());
            fprintf(stderr,
                "[bm3d-timing] frames=%.0f per-frame us: take=%7.1f acquire=%7.1f "
                "copy=%7.1f record=%7.1f srcwait=%7.1f agg=%7.1f fence=%7.1f "
                "publish=%7.1f total=%7.1f\n",
                n, ht_take_ns.load() / 1000.0 / n, ht_acquire_ns.load() / 1000.0 / n,
                ht_upload_ns.load() / 1000.0 / n, ht_record_ns.load() / 1000.0 / n,
                ht_srcwait_ns.load() / 1000.0 / n, ht_agg_ns.load() / 1000.0 / n,
                ht_fence_ns.load() / 1000.0 / n, ht_down_ns.load() / 1000.0 / n,
                ht_total_ns.load() / 1000.0 / n);
        }
        if (!gpu) {
            return;
        }
        VkDevice dev = gpu->device;
        // Drain every stream's own submissions before tearing anything down.
        // The core keeps the device alive, so there is no device-wide idle to
        // do here and no queue lock to take: each stream's own timeline covers
        // exactly the work this instance submitted.
        for (auto & s : pool.items) {
            // drain_value is what says whether this stream still has work in
            // flight; a stream that was never used has none, which a fence
            // created unsignalled could not express either.
            if (s.drain_value != 0) {
                char derr[256] {};
                gpu->api->gpuTimelineWaitValue(s.tl, s.drain_value, derr, sizeof(derr));
            }
            for (const VSFrame * f : s.held) {
                vsapi->freeFrame(f);
            }
            if (s.ts_query) gpu->vk->vkDestroyQueryPool(dev, s.ts_query, nullptr);
            if (s.cmd_agg) gpu->vk->vkFreeCommandBuffers(dev, s.pool, 1, &s.cmd_agg);
            if (s.pool) gpu->vk->vkDestroyCommandPool(dev, s.pool, nullptr);
            // The timeline is reference counted and planes we published keep
            // their own reference, so dropping ours here cannot invalidate a
            // frame still in flight.
            if (s.tl) gpu->api->freeGPUTimeline(s.tl);
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
        (env_flag("VSFEEL_BM3D_NOSEARCH") || env_flag("BM3D_NOSEARCH")) ? 1 : 0,
        (env_flag("VSFEEL_BM3D_NOESTIMATE") || env_flag("BM3D_NOESTIMATE")) ? 1 : 0,
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
    // The kernel's 8-lane shuffles need a pinned wave width; the core's device
    // baseline guarantees subgroup size control, so 32 is always available.
    // VSFEEL_BM3D_SUBGROUP=64 forces the wave64 path.
    const int forced_subgroup = env_int("VSFEEL_BM3D_SUBGROUP", 0);
    const uint32_t subgroup_size = forced_subgroup > 0
        ? static_cast<uint32_t>(forced_subgroup)
        : 32u;
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(), &spec,
        static_cast<uint32_t>(entries.size()), sizeof(spec), "bm3d",
        subgroup_size);
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

// Reserve the cache slots this frame needs, all-or-nothing: the res slots of
// the temporal window (recomputing the missing stacks) and the src slots of
// the source window (re-uploading the missing frames). Slots held by other
// in-flight frames block until they complete, so no two frames ever touch the
// same slot concurrently; the blocking only happens when the working set
// exceeds the cache (seeks, very out-of-order arrivals) and never holds the
// lock or any stream while waiting.
static void acquire_cache(BM3DData * d, Bm3dStream & stream, int n, uint64_t seq) {
    if (d->radius == 0) {
        // per-frame slots: keyed by the stream so concurrent out-of-order
        // frames never share a slot (a stream processes one frame at a time)
        stream.win_slots.fill(-1);
        stream.win_writers.fill(-1);
        stream.win_writer_stream.fill(-1);
        stream.win_writer_tl.fill(nullptr);
        stream.win_writer_value.fill(0);
        stream.win_recompute.fill(false);
        stream.upload_new.fill(false);
        stream.win_slots[0] = stream.stream_id;
        stream.win_writers[0] = n;
        stream.win_recompute[0] = true;
        stream.upload_new[0] = true;
        return;
    }
    const int r = d->radius;
    const int nf = d->nframes;
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    std::unique_lock lock(d->cache_lock);
    for (;;) {
        bool ok = true;
        stream.win_slots.fill(-1);
        stream.win_writers.fill(-1);
        stream.win_writer_stream.fill(-1);
        stream.win_writer_tl.fill(nullptr);
        stream.win_writer_value.fill(0);
        stream.win_recompute.fill(false);
        stream.upload_new.fill(false);
        stream.src_writers.fill(-1);
        stream.src_writer_tl.fill(nullptr);
        stream.src_writer_value.fill(0);
        // Phase 1: check-only, with no side effects. The failed passes must
        // not leave half-applied reservations behind, or a retry would treat
        // the abandoned slots as cached and never recompute them.
        for (int i = 0; i < d->tw; ++i) {
            const int m = std::clamp(n - r + i, 0, nf - 1);
            const int slot = m % d->res_cap;
            stream.win_slots[i] = slot;
            stream.win_writers[i] = d->res_writer[slot];
            stream.win_writer_stream[i] = d->res_writer_stream[slot];
            stream.win_writer_tl[i] = d->res_writer_tl[slot];
            stream.win_writer_value[i] = d->res_writer_value[slot];
            // A clamped window maps several positions onto one slot, so this
            // must be decided from the state *before* phase 2 mutates it: the
            // later positions would otherwise look cached and keep the stale
            // writer of the slot's previous contents as a dependency.
            stream.win_recompute[i] = (d->res_frame[slot] != m);
            if (stream.win_recompute[i] && !d->res_holders[slot].empty()) {
                ok = false;   // slot in use by an in-flight frame
                break;
            }
        }
        if (ok) {
            for (int f = lo; f <= hi; ++f) {
                const int slot = f % d->src_ring;
                stream.src_writers[f - lo] = d->src_writer[slot];
                stream.src_writer_tl[f - lo] = d->src_writer_tl[slot];
                stream.src_writer_value[f - lo] = d->src_writer_value[slot];
                if (d->src_frame[slot] != f && !d->src_holders[slot].empty()) {
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
        stream.res_token = d->next_res_token++;
        for (int i = 0; i < d->tw; ++i) {
            const int m = std::clamp(n - r + i, 0, nf - 1);
            const int slot = stream.win_slots[i];
            if (stream.win_recompute[i]) {
                if (d->res_frame[slot] != m) {   // first position mapping here
                    d->res_frame[slot] = m;
                    d->res_writer[slot] = n;
                    d->res_writer_stream[slot] = stream.stream_id;
                    d->res_writer_tl[slot] = stream.tl;
                    d->res_writer_value[slot] = seq;
                }
                // every position that maps here drops the previous writer's
                // dependency: this frame overwrites the slot's contents
                stream.win_writers[i] = -1;
                stream.win_writer_stream[i] = -1;
                stream.win_writer_tl[i] = nullptr;
                stream.win_writer_value[i] = 0;
            }
            d->res_holders[slot].push_back(stream.res_token);
        }
        for (int f = lo; f <= hi; ++f) {
            const int slot = f % d->src_ring;
            if (d->src_frame[slot] != f) {
                d->src_frame[slot] = f;
                d->src_writer[slot] = n;
                d->src_writer_tl[slot] = stream.tl;
                d->src_writer_value[slot] = seq;
                stream.upload_new[f - lo] = true;
            }
            d->src_holders[slot].push_back(stream.res_token);
        }
        return;
    }
}

// Release the cache reservations after this frame's aggregation has been
// submitted: the queue order guarantees the later recomputes run after it.
static void release_cache(BM3DData * d, Bm3dStream & stream, int n) {
    if (d->radius == 0) {
        return;
    }
    std::lock_guard lock(d->cache_lock);
    for (int i = 0; i < d->tw; ++i) {
        auto & h = d->res_holders[stream.win_slots[i]];
        h.erase(std::remove(h.begin(), h.end(), stream.res_token), h.end());
    }
    const int r = d->radius;
    const int nf = d->nframes;
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    for (int f = lo; f <= hi; ++f) {
        auto & h = d->src_holders[f % d->src_ring];
        h.erase(std::remove(h.begin(), h.end(), stream.res_token), h.end());
    }
    d->cache_cv.notify_all();
}

// Record the estimation phase of the command buffer (staging copies, the
// zero-fills and the search/estimate dispatches). It has no dependency on the
// other in-flight frames: the src ring is sized for the union of all their
// windows, and each frame only writes its own res slot. It is submitted before
// the cross-frame wait so the GPU is busy with this heavy work while the host
// blocks on the previous frames' timelines.
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
static void record_est_position(BM3DData * d, Bm3dStream & stream, VkCommandBuffer cmd,
                                int n, int i, VkBuffer dst_plane) {
    const int r = d->radius;
    const int nf = d->nframes;
    const int slot = stream.win_slots[i];
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
                static_cast<int32_t>((r == 0) ? stream.stream_id : 0)
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

static int record_bm3d_kernels(BM3DData * d, Bm3dStream & stream, int n,
                     const std::array<bool, 4 * MAX_RADIUS + 1> & copied,
                     const std::vector<Bm3dWindowCopy> & window,
                     VkBuffer dst_plane) {
    const int nf = d->nframes;
    const int r = d->radius;
    const bool gputrace = d->gpu_trace && stream.ts_query != VK_NULL_HANDLE;

    // Which window positions need computing. A clamped window collapses several
    // positions onto one slot and one centre frame, i.e. identical dispatches
    // whose fills wipe each other, so only one of them is kept.
    int pos[2 * MAX_RADIUS + 1] {};
    int n_pos = 0;
    for (int i = 0; i < d->tw; ++i) {
        if (!stream.win_recompute[i]) {
            continue;
        }
        const int m_i = std::clamp(n - r + i, 0, nf - 1);
        const int slot = stream.win_slots[i];
        bool duplicate_later = false;
        for (int j = i + 1; j < d->tw && !duplicate_later; ++j) {
            duplicate_later = stream.win_recompute[j] && stream.win_slots[j] == slot &&
                std::clamp(n - r + j, 0, nf - 1) == m_i;
        }
        if (!duplicate_later) {
            pos[n_pos++] = i;
        }
    }
    // A frame with nothing to recompute still needs a command buffer for its
    // copies and for the timeline signal every consumer waits on.
    stream.est_cb_count = (d->split_est && n_pos > 0) ? n_pos : 1;

    // copy the source window's planes (the union of all windows that this
    // record's dispatches may need, clamped to [n-2r, n+2r]) into the src ring
    const int lo = std::clamp(n - 2 * r, 0, nf - 1);
    const int hi = std::clamp(n + 2 * r, 0, nf - 1);
    const int clips = d->final ? 2 : 1;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    // Every position gets its own submission, so no single one can approach the
    // watchdog. The first carries the uploads and the timeline signal still
    // fires only after the last, which leaves the cross-frame sync unchanged:
    // a consumer waits for the whole estimation exactly as it did before.
    for (int c = 0; c < stream.est_cb_count; ++c) {
        VkCommandBuffer cmd = stream.est_cmds[c];
        d->gpu->vk->vkBeginCommandBuffer(cmd, &begin_info);

        if (c == 0) {
            for (int f = lo; f <= hi; ++f) {
                if (!copied[f - lo]) {
                    continue;
                }
                const int src_slot = (r == 0) ? stream.stream_id : (f % d->src_ring);
                const VkDeviceSize slot_device = static_cast<VkDeviceSize>(src_slot) * clips * d->planes[0].pe;
                for (int plane = 0; plane < d->n_planes; ++plane) {
                    const auto & p = d->planes[plane];
                    const VkDeviceSize pe = p.pe;
                    const VkDeviceSize plane_off = static_cast<VkDeviceSize>(plane) * d->src_size;
                    const Bm3dWindowCopy & w = window[f - lo];
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

            // the estimation dispatches read the freshly copied source/ref
            // frames, so make the transfer writes visible to the compute stage
            // before launching them (and order the res zero-fill ahead of the
            // atomic accumulation)
            bm3d_full_barrier(*d->gpu, cmd);

            if (gputrace) {
                // a query must be reset before first use and before each reuse;
                // doing it inside the command buffer keeps the reset ordered
                // with the stamps (and with the aggregation command buffer
                // submitted after this one)
                d->gpu->vk->vkCmdResetQueryPool(cmd, stream.ts_query, 0, 4);
                d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                    stream.ts_query, 0);
            }
        }

        if (c < n_pos) {
            if (d->split_est) {
                record_est_position(d, stream, cmd, n, pos[c], dst_plane);
            } else {
                for (int k = 0; k < n_pos; ++k) {
                    record_est_position(d, stream, cmd, n, pos[k], dst_plane);
                }
            }
        }

        if (c == stream.est_cb_count - 1) {
            if (gputrace) {
                d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                    stream.ts_query, 1);
            }
            // The estimation kernels' atomic accumulation must be visible to
            // the aggregation reads, which are dispatched from a separate
            // command buffer (submitted later on the same queue): make the
            // writes available to the queue-wide scope so the aggregation sees
            // complete slot contents. The barrier covers the earlier
            // submissions of this frame as well, being later in queue order.
            bm3d_full_barrier(*d->gpu, cmd);
        }

        d->gpu->vk->vkEndCommandBuffer(cmd);
    }
    return n_pos;
}

// Record the aggregation phase of the command buffer. It reads the res slots
// accumulated by the in-flight frames, so the host must have waited for their
// submissions before submitting it. The result goes straight into the output
// plane, which is why nothing is downloaded afterwards.
static void record_bm3d_agg(BM3DData * d, Bm3dStream & stream, int n, VkBuffer dst_plane) {
    VkCommandBuffer cmd = stream.cmd_agg;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };
    d->gpu->vk->vkBeginCommandBuffer(cmd, &begin_info);

    const int nf = d->nframes;
    const int r = d->radius;
    const bool gputrace = d->gpu_trace && stream.ts_query != VK_NULL_HANDLE;

    for (int plane = 0; plane < d->n_planes; ++plane) {
        const auto & p = d->planes[plane];
        const VkDeviceSize pe = p.pe;

        // aggregation: tw stacked slices (clamped frame indices, aggZ blocks)
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.agg_pipeline);
        // descriptor bindings do not carry across command buffers: cmd_agg is
        // recorded separately from the estimation phase, so without this bind
        // the dispatch runs on undefined descriptor state (black output, and
        // device loss under concurrent submissions)
        bm3d_bind(*d->gpu, cmd, d->pipeline_layout, d->src.buffer, d->res.buffer, dst_plane);
        {
            int32_t bases[9] {};
            if (r == 0) {
                // non-temporal: aggregate the single center slice
                const int32_t base = static_cast<int32_t>(
                    static_cast<VkDeviceSize>(stream.win_slots[0]) * 2 * pe +
                    static_cast<VkDeviceSize>(plane) * d->res_size_per_plane);
                for (int i = 0; i < d->tw; ++i) bases[i] = base;
            } else {
                for (int i = 0; i < d->tw; ++i) {
                    const int z = agg_z(i, n, nf, r);
                    bases[i] = static_cast<int32_t>(
                        static_cast<VkDeviceSize>(stream.win_slots[i]) * d->tw * 2 * pe +
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
        if (gputrace) d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, stream.ts_query, 2);
        d->gpu->vk->vkCmdDispatch(cmd, p.agg_grid_x, p.agg_grid_y, 1);
        if (gputrace) d->gpu->vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, stream.ts_query, 3);
    }

    d->gpu->vk->vkEndCommandBuffer(cmd);
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

        auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        vsfeel_trace_frame_begin();
        auto stream = d->pool.take();
        vsfeel_trace_mark("pool");
        auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        // Reuse gate. The previous frame's submission is waited on its own
        // timeline value rather than a fence: a frame leaves this function with
        // its work still running, so the only place its command buffers and
        // cache slots may be touched again is here, once the GPU is past them.
        // Waiting the value also covers the case where this frame's predecessor
        // failed after its estimation was submitted.
        if (stream.drain_value != 0) {
            vsfeel_trace_mark("drain");
            char derr[256] {};
            if (d->gpu->api->gpuTimelineWaitValue(stream.tl, stream.drain_value,
                    derr, sizeof(derr)) != gdDrained) {
                vsapi->setFilterError(("BM3D: the previous frame did not finish: "s + derr).c_str(), frameCtx);
                d->pool.give_back(std::move(stream));
                vsapi->freeFrame(dst);
                return nullptr;
            }
            stream.drain_value = 0;
            // The GPU-timing probe reads the query pool here: the stamps were
            // written by the submission just waited out, and reading them at
            // the end of a frame is no longer possible because the frame leaves
            // with its work in flight.
            if (stream.ts_query && d->gpu_trace) {
                uint64_t ts[4] {};
                if (d->gpu->vk->vkGetQueryPoolResults(d->gpu->device, stream.ts_query, 0, 4,
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
                            stream.last_ndisp);
                    }
                }
            }
        }
        // Only the frame references are deferred: a slot's reservation is
        // handed back as soon as this frame's aggregation is submitted (see
        // below), because the single compute queue already orders a later
        // recompute after it.
        for (const VSFrame * f : stream.held) {
            vsapi->freeFrame(f);
        }
        stream.held.clear();

        if (d->trace) fprintf(stderr, "[t] n=%d acquired\n", n);
        const int my_stream = stream.stream_id;
        // Two values per frame: the estimation signal every cross-frame reader
        // waits on, and the aggregation signal the output plane's producer pair
        // names.
        const uint64_t my_seq = stream.seq++;
        const uint64_t agg_seq = stream.seq++;
        if (d->trace) fprintf(stderr, "[t] n=%d stream=%d\n", n, my_stream);
        // set once the estimation command buffer has been queued; from then on
        // the stream has work in flight that the error path must account for.
        bool estimation_submitted = false;
        bool agg_submitted = false;

        // reserve this frame's cache slots (blocks only when the working set
        // exceeds the cache, e.g. on seeks; never holds a stream while waiting)
        acquire_cache(d, stream, n, my_seq);
        auto t2 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        const auto set_error = [&](const std::string & error_message) {
            vsfeel_trace_error("BM3D", n, error_message, d->gpu.get());
            // Anything a reader could be waiting on must end up signalled even
            // though no submission will ever signal it: host-signal the values
            // this frame had not handed to the queue yet. A value already
            // submitted must not be signalled here -- the device will do it, and
            // a host signal would race it on a timeline that may only increase.
            const auto host_signal = [&](uint64_t value) {
                VkSemaphoreSignalInfo signal_info {};
                signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
                signal_info.semaphore = stream.timeline;
                signal_info.value = value;
                d->gpu->vk->vkSignalSemaphore(d->gpu->device, &signal_info);
            };
            if (!estimation_submitted) {
                host_signal(my_seq);
                host_signal(agg_seq);
                stream.drain_value = 0;
            } else if (!agg_submitted) {
                host_signal(agg_seq);
                stream.drain_value = my_seq;
            } else {
                stream.drain_value = agg_seq;
            }
            // The slots go back at once: the queue orders any later recompute
            // after whatever this frame did manage to submit.
            release_cache(d, stream, n);
            // satisfy any reader blocked on the submission event before
            // dropping the slots
            {
                std::lock_guard lock(d->cache_lock);
                d->stream_submitted[stream.stream_id] = my_seq;
            }
            d->cache_cv.notify_all();
            // Frames this submission reads stay held: the estimation may still
            // be running, and the next take of this stream frees them after it
            // has drained the value above.
            d->pool.give_back(std::move(stream));
            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        // The estimation phase is submitted first so the GPU stays busy with
        // the heavy kernels while this host thread waits for the writers of
        // the aggregation slots before submitting the tiny aggregation.

        // Copy only the frames whose cache slots the acquire reserved. The
        // needed range is the union of every window that this record's
        // dispatches may read: [clamp(n-2r), clamp(n+2r)].
        const int r = d->radius;
        const int lo = std::clamp(n - 2 * r, 0, d->nframes - 1);
        const int hi = std::clamp(n + 2 * r, 0, d->nframes - 1);
        std::vector<Bm3dWindowCopy> window(static_cast<size_t>(hi - lo + 1));
        std::array<bool, 4 * MAX_RADIUS + 1> copied {};
        // Producer pairs of the source planes this frame copies from. They are
        // already-submitted signals (the core handed the frames over), so they
        // are waited device-side on the estimation submit -- no host wait.
        std::vector<VkSemaphore> in_waits;
        std::vector<uint64_t> in_values;
        const auto add_wait = [](std::vector<VkSemaphore> & sems,
                                 std::vector<uint64_t> & values,
                                 VkSemaphore sem, uint64_t value) {
            for (size_t i = 0; i < sems.size(); ++i) {
                if (sems[i] == sem) {
                    values[i] = std::max(values[i], value);
                    return;
                }
            }
            sems.push_back(sem);
            values.push_back(value);
        };
        for (int f = lo; f <= hi; ++f) {
            if (!stream.upload_new[f - lo]) {
                continue;
            }
            const VSFrame * src = vsapi->getFrameFilter(f, d->node, frameCtx);
            for (int plane = 0; plane < d->n_planes; ++plane) {
                VSVulkanPlaneInfo plane_info {};
                if (d->gpu->api->getGPUPlane(src, plane, &plane_info)) {
                    return set_error("clip " + std::to_string(f) + " plane " +
                        std::to_string(plane) + " is not GPU resident");
                }
                window[f - lo].source[plane] = plane_info.buffer;
                if (plane_info.readySemaphore) {
                    add_wait(in_waits, in_values, plane_info.readySemaphore, plane_info.readyValue);
                }
            }
            stream.held.push_back(src);
            if (d->final) {
                const VSFrame * rsrc = vsapi->getFrameFilter(f, d->ref_node, frameCtx);
                for (int plane = 0; plane < d->n_planes; ++plane) {
                    VSVulkanPlaneInfo plane_info {};
                    if (d->gpu->api->getGPUPlane(rsrc, plane, &plane_info)) {
                        return set_error("ref clip " + std::to_string(f) + " plane " +
                            std::to_string(plane) + " is not GPU resident");
                    }
                    window[f - lo].ref[plane] = plane_info.buffer;
                    if (plane_info.readySemaphore) {
                        add_wait(in_waits, in_values, plane_info.readySemaphore, plane_info.readyValue);
                    }
                }
                stream.held.push_back(rsrc);
            }
            copied[f - lo] = true;
        }
        auto t3 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        // The destination plane the aggregation writes. Under the new API this
        // is the output frame's own storage: nothing is downloaded.
        VSVulkanPlaneInfo dst_plane {};
        if (d->gpu->api->getGPUPlane(dst, 0, &dst_plane)) {
            return set_error("the output frame is not GPU resident");
        }

        const int ndisp = record_bm3d_kernels(d, stream, n, copied, window, dst_plane.buffer);
        auto t4 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (d->trace) fprintf(stderr, "[t] n=%d kernels recorded (%d submits)\n", n, stream.est_cb_count);

        // Cross-frame dependencies are expressed on the writers' per-stream
        // timelines, signalled device-side by each writer's kernel submit. A
        // host-side wait alone does not establish device memory visibility, and
        // a single end-of-frame signal would deadlock on out-of-order arrivals
        // (the aggregation of an early frame waits on the kernels of a later
        // frame whose kernels wait on that early frame's source copy). The
        // estimation kernels read the source window (so they wait for the
        // copiers), and the aggregation reads the estimate stacks (so it waits
        // for the stack writers plus its own kernels).
        const int src_lo = std::clamp(n - 2 * r, 0, d->nframes - 1);
        const int src_hi = std::clamp(n + 2 * r, 0, d->nframes - 1);

        std::vector<VkSemaphore> src_waits, res_waits;
        std::vector<uint64_t> src_values, res_values;
        if (d->radius > 0) {
            for (int f = src_lo; f <= src_hi; ++f) {
                if (stream.upload_new[f - src_lo]) {
                    continue;   // own copy: recorded ahead of our dispatches
                }
                const int w = stream.src_writers[f - src_lo];
                VSGPUTimeline * tl = stream.src_writer_tl[f - src_lo];
                if (w < 0 || tl == nullptr) {
                    continue;
                }
                if (tl == stream.tl) {
                    continue;   // same stream: already queue-ordered
                }
                if (d->trace) fprintf(stderr, "[t] n=%d waits on src copier w=%d val=%llu\n", n, w, static_cast<unsigned long long>(stream.src_writer_value[f - src_lo]));
                add_wait(src_waits, src_values,
                    d->gpu->api->getGPUTimelineSemaphore(tl), stream.src_writer_value[f - src_lo]);
            }
            for (int i = 0; i < d->tw; ++i) {
                const int w = stream.win_writers[i];
                VSGPUTimeline * tl = stream.win_writer_tl[i];
                if (w < 0 || tl == nullptr) {
                    continue;   // own work or a slot this frame recomputes
                }
                if (tl == stream.tl) {
                    continue;   // same stream: already queue-ordered
                }
                if (d->trace) fprintf(stderr, "[t] n=%d waits on writer w=%d val=%llu\n", n, w, static_cast<unsigned long long>(stream.win_writer_value[i]));
                add_wait(res_waits, res_values,
                    d->gpu->api->getGPUTimelineSemaphore(tl), stream.win_writer_value[i]);
            }
        }

        // The source window is read by this frame's estimation kernels. The
        // copiers' submissions are ordered ahead of their own kernels on the
        // same queue, so waiting for the copiers' timelines here (host-side) is
        // deadlock-free: a GPU-side wait on a timeline signalled by a later
        // submission would block the whole queue. The host wait alone does not
        // establish the device-side producer->consumer memory dependency, so
        // the same timelines are also fed to the estimation submit as device
        // waits (they are signalled by now, so those waits are immediate).
        if (d->radius > 0) {
            for (int f = src_lo; f <= src_hi; ++f) {
                const int w = stream.src_writers[f - src_lo];
                VSGPUTimeline * tl = stream.src_writer_tl[f - src_lo];
                if (w < 0 || tl == nullptr || tl == stream.tl ||
                    stream.upload_new[f - src_lo]) {
                    continue;
                }
                char werr[256] {};
                if (d->gpu->api->gpuTimelineWaitValue(tl, stream.src_writer_value[f - src_lo],
                        werr, sizeof(werr)) != gdDrained) {
                    return set_error("a source frame's copy did not complete: "s + werr);
                }
            }
        }

        // Everything the first estimation submission waits on: the copiers'
        // estimate handoffs (device-side, for visibility) and the producer
        // pairs of the frames it copies from.
        std::vector<VkSemaphore> first_waits = src_waits;
        std::vector<uint64_t> first_values = src_values;
        for (size_t i = 0; i < in_waits.size(); ++i) {
            add_wait(first_waits, first_values, in_waits[i], in_values[i]);
        }

        /* no fence: completion is expressed on this stream's timeline, which
               the aggregation submit signals and the next reuse drains */
        vsfeel_trace_mark("sub est");
        // One submission per recomputed window position (see
        // record_bm3d_kernels). The queue orders them, so only the first has to
        // wait for the copies and only the last signals the timeline -- which
        // is what every consumer of this frame waits on, so the cross-frame
        // sync is unchanged and a partial failure leaves the signal unsent for
        // the error path to host-signal.
        for (int c = 0; c < stream.est_cb_count; ++c) {
            const bool first = c == 0;
            const bool last = c == stream.est_cb_count - 1;
            const VkResult sub = gpu_submit(*d->gpu, core, stream.est_cmds[c],
                first ? first_waits.data() : nullptr,
                first ? first_values.data() : nullptr,
                first ? static_cast<uint32_t>(first_waits.size()) : 0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                last ? stream.timeline : VK_NULL_HANDLE,
                last ? my_seq : 0, VK_NULL_HANDLE);
            if (sub != VK_SUCCESS) {
                return set_error("estimation submit failed: "s + vk_result_string(sub));
            }
        }
        estimation_submitted = true;
        // Publish the submission event before recording the aggregation: a
        // reader whose aggregation device-waits on this estimation must be able
        // to see that the signal is already on its way (see the host wait below).
        {
            std::lock_guard lock(d->cache_lock);
            d->stream_submitted[stream.stream_id] = my_seq;
        }
        d->cache_cv.notify_all();
        auto t5 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        record_bm3d_agg(d, stream, n, dst_plane.buffer);
        if (d->trace) fprintf(stderr, "[t] n=%d agg recorded\n", n);

        // The aggregation device-waits on the estimate-stack writers'
        // timelines. On a queue shared by several streams, a submit that waits
        // on a value signalled by a *later* submit stalls that queue
        // permanently (the driver does not run past an unsatisfied semaphore
        // wait), so first wait host-side until every writer has *submitted* its
        // estimation. Waiting for submission rather than completion cannot
        // deadlock: the waits follow cache-acquisition order, which is acyclic,
        // and a writer never waits on this frame's aggregation.
        if (d->radius > 0) {
            std::unique_lock lock(d->cache_lock);
            d->cache_cv.wait(lock, [&] {
                for (int i = 0; i < d->tw; ++i) {
                    const int sid = stream.win_writer_stream[i];
                    if (sid < 0 || stream.win_writer_tl[i] == nullptr ||
                        stream.win_writer_tl[i] == stream.tl) {
                        continue;
                    }
                    if (d->stream_submitted[sid] < stream.win_writer_value[i]) {
                        return false;
                    }
                }
                return true;
            });
        }

        {
            // wait on our own kernels (the timeline is signalled by the kernel
            // submit) and on the frames that computed the aggregation slots
            std::vector<VkSemaphore> agg_waits { stream.timeline };
            std::vector<uint64_t> agg_values { my_seq };
            agg_waits.insert(agg_waits.end(), res_waits.begin(), res_waits.end());
            agg_values.insert(agg_values.end(), res_values.begin(), res_values.end());

            vsfeel_trace_mark("sub agg");
            // The aggregation signals agg_seq: that is the value the output
            // plane's producer pair names (so a consumer can wait it device
            // side) and the one the next use of this stream drains before
            // touching its command buffers again.
            const VkResult sub = gpu_submit(*d->gpu, core, stream.cmd_agg,
                agg_waits.data(), agg_values.data(),
                static_cast<uint32_t>(agg_waits.size()),
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                stream.timeline, agg_seq, VK_NULL_HANDLE);
            if (sub != VK_SUCCESS) {
                return set_error("aggregation submit failed: "s + vk_result_string(sub));
            }
        }
        agg_submitted = true;
        auto t6 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        // Publish the output plane's producer pair. The signal is already on
        // its way (the aggregation submit signalled agg_seq), which is exactly
        // the condition setGPUPlaneProducer requires -- publishing a value this
        // host would still have to signal is what can deadlock a consumer.
        vsfeel_trace_mark("publish");
        d->gpu->api->setGPUPlaneProducer(dst, 0, stream.tl, agg_seq);

        // The stream is returned with work in flight; the next take of it
        // drains agg_seq before touching its command buffers or cache slots.
        stream.drain_value = agg_seq;
        stream.last_ndisp = ndisp;
        d->pool.give_back(std::move(stream));
        // Reserved slots go back as soon as the aggregation is queued. Every
        // submission this filter makes lands on the one compute queue the core
        // exposes, so a later frame's recompute is ordered after this
        // aggregation by the queue itself -- the host wait the legacy
        // multi-queue path needed here would only block progress.
        release_cache(d, stream, n);
        auto t7 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        if (d->host_timing) {
            auto t8 = std::chrono::steady_clock::now();
            const auto us = [](auto a, auto b) {
                return static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
            };
            d->ht_take_ns += us(t0, t1);
            d->ht_acquire_ns += us(t1, t2);
            d->ht_upload_ns += us(t2, t3);
            d->ht_record_ns += us(t3, t4);
            d->ht_srcwait_ns += us(t4, t5);
            d->ht_agg_ns += us(t5, t6);
            d->ht_fence_ns += us(t6, t7);
            d->ht_down_ns += us(t7, t8);
            d->ht_total_ns += us(t0, t8);
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
    d->gpu_trace = vsfeel_debug_probe("VSFEEL_BM3D_GPUTRACE") || env_flag("BM3D_GPUTRACE");
    // Cached too: the frame path must not pay a getenv (plus the legacy-name
    // fallback) for flags that are fixed per instance.
    d->trace = vsfeel_debug_trace("VSFEEL_BM3D_TRACE") || env_flag("BM3D_TRACE");
    d->dump = env_flag("VSFEEL_BM3D_DUMP") || env_flag("BM3D_DUMP");
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

    // "num_streams" is accepted for compatibility and no longer selects
    // anything: how many frames are in flight is the core's call now, and the
    // two-stream depth this filter's caches are sized for is fixed below.
    if (vsapi->mapGetInt(in, "num_streams", 0, &error), !error && vsfeel_debug_flag("VSFEEL_BM3D_DEPRECATED")) {
        fprintf(stderr, "[bm3d] num_streams is ignored under the R80 GPU API\n");
    }
    d->num_streams = 2;

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    // Device selection moved to the core (core.set_vulkan_device): one Vulkan
    // device per process, picked before any GPU filter runs. The argument stays
    // accepted so existing scripts keep loading; a negative one is still an
    // error because it never selected anything.
    if (!error && device_id < 0) {
        return set_error("\"device_id\" must be non-negative; under the R80 GPU API "
                         "device selection is core.set_vulkan_device");
    }

    // at radius 0 every frame only touches its own slot and never depends on
    // the previous frames' estimates, so give each in-flight frame its own
    // src/res slot to keep the pipeline full; otherwise the ring of 1 would
    // serialize the frames behind the timeline waits. At radius > 0 the caches
    // are keyed by frame modulo the capacity and reserved all-or-nothing per
    // frame, so they only need to cover the working set of the concurrent
    // frames (like the reference's fused-mode accumulator cache); anything
    // beyond that (e.g. seeking) blocks in the acquire instead of corrupting.
    d->src_ring = (d->radius == 0) ? d->num_streams : 4 * d->radius + d->num_streams;
    // One in-flight frame needs the stacks of centre frames [n-r, n+r], so
    // num_streams concurrent frames span num_streams + 2r slots. That working
    // set is the default: the estimate cache is the largest allocation, and on
    // an 8 GiB card the slack below is the difference between running radius 4
    // and failing to allocate. VSFEEL_BM3D_CACHE=1 adds a whole extra window,
    // so an out-of-order (seek) request finds a warm slot instead of waiting in
    // the acquire, for tw/(ns+2r+tw) more VRAM.
    const int res_working_set = d->num_streams + 2 * d->radius;
    const bool cache_slack = env_int("VSFEEL_BM3D_CACHE", 0) != 0;
    d->res_cap = (d->radius == 0) ? d->num_streams
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

    // The BM3D kernels accumulate into float SSBOs. Hardware buffer float
    // atomics need VK_EXT_shader_atomic_float, which no pre-RDNA3 AMD driver
    // reports (RADV: GFX11+; the Windows driver does not expose it on Polaris
    // either); on anything older the accumulation falls back to the CAS loop
    // the OpenCL reference itself uses (atom_add_f), so the filter runs
    // everywhere instead of failing at creation.
    d->cas_atomics = env_flag("VSFEEL_BM3D_CAS") || !d->gpu->feat_atomic_float32_add;
    if (vsfeel_device_info_enabled()) {
        fprintf(stderr, "[bm3d] aggregation: %s\n",
            d->cas_atomics ? "CAS loop (no buffer float atomics available)"
                           : "hardware buffer float atomics");
    }
    // The 8x8 group transposes and the group-8 reduction are subgroup shuffles.
    // The spec only makes SUBGROUP_FEATURE_BASIC_BIT mandatory, so a device
    // without SHUFFLE would either reject the module or mis-execute.
    if (!d->gpu->subgroup_shuffle) {
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
                "plane (radius %d, num_streams %d), which overflows the 32-bit "
                "kernel addressing; reduce num_streams or radius",
                static_cast<unsigned long long>(res_floats), d->radius,
                d->num_streams);
            return set_error(msg);
        }
    }

    // shared buffers
    {
        VkDeviceSize src_size = 0;
        VkDeviceSize res_size = 0;
        VkDeviceSize dst_size = 0;
        (void)res_size;
        for (int plane = 0; plane < d->n_planes; ++plane) {
            const auto & p = d->planes[plane];
            // in final mode each ring slot holds [ref][source], so the ring
            // doubles in size
            const int clips = d->final ? 2 : 1;
            src_size += static_cast<VkDeviceSize>(d->src_ring) * clips * p.pe;
            res_size += static_cast<VkDeviceSize>(d->res_cap) * d->tw * 2 * p.pe;
            dst_size += p.pe;
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
                    "%s; the estimate cache needs %.0f MiB (radius %d, "
                    "num_streams %d): lower radius or num_streams",
                    err.c_str(),
                    static_cast<double>(res_size) * 4.0 / (1024.0 * 1024.0),
                    d->radius, d->num_streams);
                return set_error(msg);
            }
        }
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

    // streams
    d->pool.semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->pool.reserve(d->num_streams);

    for (int i = 0; i < d->num_streams; ++i) {
        Bm3dStream stream;

        {
            VkCommandPoolCreateInfo pool_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                /* the per-frame recording re-begins the same command buffers;
                   without this flag the implicit reset in vkBeginCommandBuffer
                   is invalid usage (intermittent stale submissions: black
                   output, device loss under load) */
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = d->gpu->queue_family
            };
            checkVK(d->gpu->vk->vkCreateCommandPool(dev, &pool_info, nullptr, &stream.pool));
        }
        {
            VkCommandBufferAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = stream.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 2 * MAX_RADIUS + 2
            };
            VkCommandBuffer cmds[2 * MAX_RADIUS + 2] {};
            checkVK(d->gpu->vk->vkAllocateCommandBuffers(dev, &alloc_info, cmds));
            for (int c = 0; c < 2 * MAX_RADIUS + 1; ++c) {
                stream.est_cmds[c] = cmds[c];
            }
            stream.cmd = stream.est_cmds[0];
            stream.cmd_agg = cmds[2 * MAX_RADIUS + 1];
        }
        {
            // The stream's own timeline. One per stream so the values it
            // signals increase in submission order, which a shared timeline
            // across independently submitting streams could not promise.
            char terr[256] {};
            stream.tl = d->gpu->api->createGPUTimeline(core, terr, sizeof(terr));
            if (stream.tl == nullptr) {
                return set_error(std::string("could not create a timeline: ") + terr);
            }
            stream.timeline = d->gpu->api->getGPUTimelineSemaphore(stream.tl);
        }
        if (d->gpu_trace) {
            VkQueryPoolCreateInfo qp_info {
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = 4
            };
            checkVK(d->gpu->vk->vkCreateQueryPool(dev, &qp_info, nullptr, &stream.ts_query));
        }

        stream.stream_id = i;

        d->pool.push(std::move(stream));
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
            "(radius=%d streams=%d res_cap=%d src_ring=%d stride=%d)\n",
            static_cast<double>(d->src_size) * 4.0 / mib, res_only / mib,
            100.0 * res_only / total, total / mib,
            d->radius, d->num_streams, d->res_cap, d->src_ring, d->planes[0].stride);
    }

    d->src_frame.assign(d->src_ring, -1);
    d->src_writer.assign(d->src_ring, -1);
    d->src_writer_tl.assign(d->src_ring, nullptr);
    d->src_writer_value.assign(d->src_ring, 0);
    d->src_holders.resize(d->src_ring);
    d->res_frame.assign(d->res_cap, -1);
    d->res_writer.assign(d->res_cap, -1);
    d->res_writer_stream.assign(d->res_cap, -1);
    d->res_writer_tl.assign(d->res_cap, nullptr);
    d->res_writer_value.assign(d->res_cap, 0);
    d->res_holders.resize(d->res_cap);
    d->stream_submitted.assign(d->num_streams, 0);

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
