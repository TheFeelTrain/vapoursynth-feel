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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <string_view>
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
// In-flight frames the private caches are sized for. The core's exec pool keeps
// min(threads, 8) recording contexts, and a frame uses two of them (estimation
// and aggregation), so four frames are what the pool can actually overlap; the
// rings are sized for that. Sizing them for two instead leaves the host path
// exactly at the GPU's pace, where every handoff delay stalls the GPU (measured
// 8% run-to-run swing, and 6% slower on average). VSFEEL_BM3D_INFLIGHT lowers
// it to trade pipeline depth back for VRAM on a small card.
constexpr int kInflightFrames = 4;
// The finished-frame cache's depth. The VRAM budget lowers it further at
// creation, so a large frame does not pin forty of them by count alone.
constexpr size_t kEmittedMax = 40;

struct Bm3dPlane {
    int width {};
    int height {};
    int stride {};
    VkDeviceSize pe {}; // plane extent in floats (h * stride)
    // Per-plane temporal radius, like sigma: a missing entry repeats the
    // previous one and the first defaults to 0 (spatial only).
    int radius {};
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

// A native u16 code to the f32 working domain, and the inverse the normalize
// kernel applies. The CPU reference (mawen's VSProcess::Int2Float/Float2Int)
// scales by the frame's _Range: full range spans the whole code range, limited
// range spans 219/224 of it with luma/RGB's 4096 and chroma's 32768 offsets. RGB
// untagged is full; a tagged RGB clip follows its tag. The gain matches the
// reference's RangeConvert exactly, so a native clip and a float conversion of
// the same content carry the same values.
struct Bm3dRange {
    float gain {};
    float neutral {};
};

static Bm3dRange bm3d_range(const VSVideoFormat & format, int plane,
                            bool full) {
    if (format.colorFamily == cfRGB) {
        return { full ? 1.0f / 65535.0f : 1.0f / 56064.0f,
                 full ? 0.0f : 4096.0f };
    }
    const bool chroma = format.colorFamily == cfYUV && plane > 0;
    if (chroma) {
        return { full ? 1.0f / 65535.0f : 1.0f / 57344.0f, 32768.0f };
    }
    return { full ? 1.0f / 65535.0f : 1.0f / 56064.0f, full ? 0.0f : 4096.0f };
}

// The frame's _Range property. A missing one is full on RGB (mawen's
// `full = rgb ? error || _Range != 0 : ...`) and limited on Gray/YUV.
static bool bm3d_full_range(const VSFrame * frame, const VSVideoFormat & format,
                            const VSAPI * vsapi) {
    int error = 0;
    const VSMap * props = vsapi->getFramePropertiesRO(frame);
    const int64_t range = vsapi->mapGetInt(props, "_Range", 0, &error);
    if (format.colorFamily == cfRGB) {
        return error ? true : range != 0;
    }
    return !error && range != 0;
}

// Push blocks of bm3d_opp.comp. The shader declares a packed scalar layout, so
// the host structs must stay 4-byte aligned with no padding.
struct Bm3dOppInPush {
    uint32_t dst_off; // first destination element of this plane
    uint32_t count;   // elements to convert
    int32_t plane;    // 0 = Y, 1 = U, 2 = V
    float neutral;    // Int2Float neutral
    float gain;       // Int2Float gain
};
struct Bm3dOppOutPush {
    uint32_t count; // elements to convert
    uint32_t y_off; // plane bases in the three-plane scratch
    uint32_t u_off;
    uint32_t v_off;
    int32_t plane;  // 0 = R, 1 = G, 2 = B
    float out_gain; // Float2Int scale for this frame's _Range
    float out_bias; // neutral + 0.5
};
static_assert(sizeof(Bm3dOppInPush) == 20, "bm3d_opp in push block");
static_assert(sizeof(Bm3dOppOutPush) == 28, "bm3d_opp out push block");

// One estimation entry: a source ring and an estimate stack that cover the
// planes it filters. The reference's default mode builds one entry per
// processed plane (each plane block-matches on its own content, at its own
// geometry); chroma mode builds one entry over the clip's three 4:4:4 planes,
// whose groups come from luma. Every plane of an entry shares one geometry.
struct Bm3dGroup {
    int n_planes {};
    std::array<int, 3> planes {}; // absolute plane indices, in packing order
    VkDeviceSize pe {};           // per-plane extent in floats
    // The entry's temporal radius. Default mode is one plane per entry, so
    // this is that plane's radius; a joint entry shares one window and runs at
    // the widest of its processed planes.
    int radius {};
    VkPipeline bm3d_pipeline {};
    // The split's two halves (VSFEEL_BM3D_SPLIT_KERNEL); unused when fused.
    VkPipeline bm3d_match_pipeline {};
    VkPipeline bm3d_filter_pipeline {};
    uint32_t bm3d_grid_x {};
    uint32_t bm3d_grid_y {};
    // The copy-and-widen dispatch for this entry's plane extent, folded into X
    // and Y because one dimension's workgroup count can exceed the device's
    // limit at 4K and above.
    uint32_t copy_grid_x {};
    uint32_t copy_grid_y {};

    int src_ring {};          // ring slots for the temporal source window
    int res_cap {};           // accumulator slots of the frame ring
    VkDeviceSize src_size {}; // src_ring * clips * n_planes * pe
    VkDeviceSize res_size {}; // res_cap * 2 * n_planes * pe
    GpuBuffer src;
    GpuBuffer res;
    // A private one-window ring for requests that cannot advance the cursor
    // (a seek, or a frame the cursor has already passed): the frame is computed
    // in these buffers so the main ring keeps its state.
    GpuBuffer scratch_src;
    GpuBuffer scratch_acc;
    // Frame held by each accumulator slot, at radius>0. -1 means the slot holds
    // no frame and must be zeroed before its next writer. The cursor keeps this
    // table coherent: a slot is reassigned only after its frame retired.
    std::vector<int> acc_frame {};

    // Radius 0 has no cross-frame sharing, so each in-flight frame takes one of
    // these private slots for its whole life. Size is the in-flight depth.
    std::vector<int> r0_free {};
};

// Radius 0 bookkeeping, per entry. The exec pool owns the command buffers and
// the in-flight gate, so nothing here outlives the getFrame call.
struct Bm3dGroupFrame {
    // window position maps to a real frame (radius 0 and the clip's ends have
    // fewer than tw valid positions)
    std::array<bool, 2 * MAX_RADIUS + 1> win_valid {};
    // this frame computes it
    std::array<bool, 2 * MAX_RADIUS + 1> win_recompute {};
    // res slot per window position
    std::array<int, 2 * MAX_RADIUS + 1> win_slots {};
    // frame that wrote it (trace)
    std::array<int, 2 * MAX_RADIUS + 1> win_writer {};
    // Window positions this frame must compute, in order. Each valid position
    // is a distinct centre frame, so there is nothing to deduplicate.
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

// Per-dispatch GPU timing (VSFEEL_BM3D_PROFILE): two timestamps around every
// dispatch, fill and copy, reported at free in bm3dvk2's BM3DVK_PROFILE layout
// so the two plugins' kernel tables can be read side by side. Pairs are
// allocated atomically because frames record concurrently; the report runs
// after the exec pool has drained.
struct Bm3dKernelProfile {
    static constexpr uint32_t kQueryCapacity = 1u << 18;
    static constexpr uint32_t kPairs = kQueryCapacity / 2;

    VkDevice device {};
    const VSVulkanFunctions * vk {};
    VkQueryPool pool {};
    std::atomic<uint32_t> next_pair { 0 };
    std::vector<const char *> names;
    double period {};
    uint64_t mask {};

    bool init(const GPUDevice & gpu, const VSAPI & vsapi, VSCore * core) {
        device = gpu.device;
        vk = gpu.vk;
        VkQueryPoolCreateInfo info {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kQueryCapacity
        };
        if (vk->vkCreateQueryPool(device, &info, nullptr, &pool) !=
            VK_SUCCESS) {
            pool = VK_NULL_HANDLE;
            vsapi.logMessage(mtWarning,
                             "BM3D: could not create the timestamp pool; "
                             "VSFEEL_BM3D_PROFILE disabled",
                             core);
            return false;
        }
        vk->vkResetQueryPool(device, pool, 0, kQueryCapacity);
        names.assign(kPairs, nullptr);
        period = gpu.limits.timestampPeriod;
        mask = gpu.timestamp_valid_bits >= 64
                   ? ~static_cast<uint64_t>(0)
                   : (uint64_t { 1 } << gpu.timestamp_valid_bits) - 1;
        return true;
    }

    std::optional<uint32_t> begin(VkCommandBuffer cmd, const char * name) {
        if (pool == VK_NULL_HANDLE) {
            return std::nullopt;
        }
        const uint32_t pair = next_pair.fetch_add(1, std::memory_order_relaxed);
        if (pair >= kPairs) {
            return std::nullopt;
        }
        names[pair] = name;
        vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 pool, pair * 2);
        return pair;
    }

    void end(VkCommandBuffer cmd, std::optional<uint32_t> pair) const {
        if (pair) {
            vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                     pool, *pair * 2 + 1);
        }
    }

    void report(const VSAPI & vsapi, VSCore * core) const {
        if (pool == VK_NULL_HANDLE) {
            return;
        }
        const uint32_t pairs =
            std::min(next_pair.load(std::memory_order_relaxed), kPairs);
        if (pairs == 0) {
            vsapi.logMessage(mtInformation,
                             "BM3D profile: no dispatch was timed", core);
            return;
        }
        struct Result {
            uint64_t value;
            uint64_t available;
        };
        std::vector<Result> results(static_cast<size_t>(pairs) * 2);
        if (vk->vkGetQueryPoolResults(
                device, pool, 0, pairs * 2, results.size() * sizeof(Result),
                results.data(), sizeof(Result),
                VK_QUERY_RESULT_64_BIT |
                    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS) {
            vsapi.logMessage(mtWarning,
                             "BM3D profile: the timestamp queries could not "
                             "be read",
                             core);
            return;
        }
        std::map<std::string_view, std::pair<double, int>> totals;
        for (uint32_t pair = 0; pair < pairs; ++pair) {
            const Result & first = results[static_cast<size_t>(pair) * 2];
            const Result & second = results[static_cast<size_t>(pair) * 2 + 1];
            if (first.available && second.available && names[pair]) {
                auto & [ms, count] = totals[names[pair]];
                ms += static_cast<double>((second.value - first.value) & mask) *
                      period * 1e-6;
                ++count;
            }
        }
        std::string s = "BM3D profile:";
        for (const auto & [name, total] : totals) {
            s += "\n  " + std::string(name) + ": " +
                 std::to_string(total.first / total.second) + " ms x " +
                 std::to_string(total.second) + " = " +
                 std::to_string(total.first) + " ms";
        }
        vsapi.logMessage(mtInformation, s.c_str(), core);
    }

    void destroy() {
        if (pool != VK_NULL_HANDLE) {
            vk->vkDestroyQueryPool(device, pool, nullptr);
            pool = VK_NULL_HANDLE;
        }
    }
};


// One instance per filter node, never an array, so the fields stay grouped the
// way the comments read rather than packed for a padding win that cannot pay.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct BM3DData {
    VSNode * node {};
    VSNode * ref_node {}; // optional basic-estimate clip (final/Wiener pass)
    const VSVideoInfo * vi {};
    // Invoked entry point: BM3D or the BM3Dv2 alias, used in error messages
    // and the node name.
    const char * name { "BM3D" };

    int max_radius {};
    int max_tw {}; // 2 * max_radius + 1
    bool final {}; // true when a "ref" clip is given
    float extractor {};
    // Matching threshold in the shader's SSD domain: the user's 8-bit MSE
    // threshold times the CPU's color-matrix norm and 64/255^2. Exactly zero
    // means the reference-only group, as the CPU's non-positive thMSE does.
    float th_sse { 1.0f };
    // Kept for the match trace's parameter snapshot only.
    float th_mse { 0.0f };
    bool
        cas_atomics {}; // aggregate with the CAS kernel (no float32 add atomics)
    // Stage the group's reference patch in shared memory instead of 64 registers
    // per lane: a backend that promotes the array cannot afford the LDS traffic,
    // one that leaves it in scratch cannot afford the reads. See the driver rule
    // in BM3DCreate and src/bm3d.comp.
    bool patch_lds {};
    // chroma=True: one joint entry over the clip's three 4:4:4 planes, whose
    // groups come from luma (the references' "chroma" mode).
    bool joint {};
    // Launch shape of the estimation kernels, and the group grid the split's
    // handoff is indexed by: chosen from the device's occupancy limits, not
    // fixed. `groups_per_wg` is the same number in 8-lane groups.
    uint32_t workgroup {};
    uint32_t groups_per_wg {};
    // Tile shape of the normalization kernel and the element block of the
    // linear copy/OPP kernels: fitted to the device, so a 128-invocation one
    // runs the aggregation at 16x8 and the copies at 128. Both are
    // specialization constants in the shaders (see vsfeel_fit_workgroup).
    GpuWorkgroup agg_wg {};
    uint32_t copy_wg { 256 };
    // Pinned subgroup width (0 keeps the device default), chosen once so the
    // occupancy calculation and every pipeline agree.
    uint32_t subgroup_size {};

    std::shared_ptr<GPUDevice> gpu;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    int num_planes {}; // planes of the clip's format
    // Sample type of the clip's own planes: 16 for an integer or half clip. The
    // ring, the estimate stacks and every kernel's arithmetic are float either
    // way; only the copy into the ring and the aggregation's store differ.
    int bits {};
    int elem_bytes {};
    // Half input: a 2-byte transport like the integer path, but the sample is
    // already in the ring's float domain, so the copy is an exact widening with
    // no gain or chroma neutral.
    bool half {};
    // Integer or half input only: widens each copied plane into the ring. Null
    // otherwise.
    VkPipeline copy_pipeline {};
    std::array<Bm3dPlane, 3> planes {};
    // One entry per processed plane, or a single one covering all three under
    // chroma=True. Every buffer a kernel reads lives in its entry.
    std::array<Bm3dGroup, 3> groups {};
    int n_groups {};

    // The core's exec pool: one per instance, on the compute queue. It owns the
    // command buffers, the timeline and the backpressure; the filter records and
    // submits, and hands it the frames and scratch a submission must keep alive.
    VSGPUExecPool * exec {};

    // Debug match trace (VSFEEL_BM3D_MATCHTRACE=frame,x,y): the traced dispatch
    // is the one that estimates the requested centre frame itself, and the
    // kernel records the group of the reference block at (x, y) into this
    // host-visible buffer, which the frame path dumps once the submission it
    // rode on has completed.
    int trace_frame { -1 };
    int trace_x {};
    int trace_y {};
    GpuBuffer match_trace {};
    volatile uint32_t * match_trace_mapped {};
    // Match/filter split: the group handoff (one count word per reference block
    // plus 16 words of members per block) and the flag that selects it.
    bool split_kernel {};
    GpuBuffer matches {};
    uint64_t match_words {};
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
    // Set when a temporal frame fails before its submission, so a later frame
    // does not try to build on a partially updated cursor.
    bool failed {};
    // Temporal cursor (radius > 0). Frames are processed in order: one centre
    // is scattered into every accumulator it touches, then the frame whose
    // contributors are all in is normalized. An out-of-order request restarts
    // the ring, which is the only sound eviction for accumulators that mix
    // contributions from several centres.
    struct Cursor {
        bool valid {};
        int next_ref {};        // next centre to scatter
        int retired_up_to {};   // first output frame not yet normalized
        int converted_up_to {}; // first source frame not yet in the ring
    } cursor {};
    // Frames already normalized ahead of their request. vspipe's request order
    // is close to, but not exactly, frame order; serving a cached frame beats
    // restarting the whole window for a request that arrived late. The cap is
    // the count kEmittedMax, lowered by the VRAM budget at creation.
    std::map<int, VSFrame *> emitted {};
    size_t emitted_cap { kEmittedMax };
    // Frames the core has requested but not yet taken. A frame is normalized
    // only if it is needed now or still pending, so a prefetch that jumps ahead
    // costs the window it advances through, not a normalize per frame.
    std::multiset<int> pending {};
    // Source-window lower bound each queued request was given at arInitial.
    // A request whose range only covers its own [n-2r, n+2r] cannot advance a
    // cursor that is further back; the duplicate case keeps the narrower bound.
    std::map<int, int> request_lo {};
    // Keeps the cursor state single-writer while the core may still enter the
    // filter concurrently for different frames (fmParallel).
    std::mutex temporal_lock {};
    std::mutex cache_lock;
    std::condition_variable cache_cv;
    Bm3dProbe probe {};
    Bm3dKernelProfile profile {};
    // RGB is filtered in the opponent space (see bm3d_opp.comp): one input
    // pipeline converts the source planes into the ring, one output pipeline
    // writes the RGB frame from the normalized OPP planes, and the scratch
    // holds those three planes until the output conversion has read them.
    bool rgb {};
    VkPipeline opp_in_pipeline {};
    VkPipeline opp_out_pipeline {};
    VkDeviceSize opp_pe {};
    uint32_t opp_grid_x {};
    uint32_t opp_grid_y {};

    // Drain every submission this instance made; also called from BM3DFree so
    // the profile can read its timestamp queries before the pool goes away.
    void drain() {
        if (gpu && exec) {
            gpu->api->freeGPUExecPool(exec);
            exec = nullptr;
        }
    }

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
    // before its chunk-0 submission, so tests can exercise the private-slot
    // release path (-1 disables).
    int fault_frame { -1 };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_source_ns {}, ht_est_ns {},
        ht_agg_ns {}, ht_release_ns {}, ht_total_ns {}, ht_n {};
    std::atomic<uint64_t> ht_cpu_acquire {}, ht_cpu_source {}, ht_cpu_est {},
        ht_cpu_agg {}, ht_cpu_release {}, ht_cpu_total {};

    ~BM3DData() {
        if (host_timing && ht_n.load()) {
            fprintf(
                stderr,
                "[bm3d-cpu] frames=%.0f per-frame CPU us: acquire=%7.1f "
                "source=%7.1f est=%7.1f agg=%7.1f release=%7.1f total=%7.1f\n",
                static_cast<double>(ht_n.load()),
                ht_cpu_acquire.load() / 1000.0 / ht_n.load(),
                ht_cpu_source.load() / 1000.0 / ht_n.load(),
                ht_cpu_est.load() / 1000.0 / ht_n.load(),
                ht_cpu_agg.load() / 1000.0 / ht_n.load(),
                ht_cpu_release.load() / 1000.0 / ht_n.load(),
                ht_cpu_total.load() / 1000.0 / ht_n.load());
        }
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
        drain();
        if (probe.query) {
            gpu->vk->vkDestroyQueryPool(dev, probe.query, nullptr);
        }
        profile.destroy();
        std::vector<VkPipeline> pipelines { copy_pipeline };
        pipelines.reserve(3 + groups.size() * 3 + planes.size());
        pipelines.push_back(opp_in_pipeline);
        pipelines.push_back(opp_out_pipeline);
        for (auto & g : groups) {
            gpu_destroy_buffer(*gpu, g.scratch_acc);
            gpu_destroy_buffer(*gpu, g.scratch_src);
            gpu_destroy_buffer(*gpu, g.res);
            gpu_destroy_buffer(*gpu, g.src);
            pipelines.push_back(g.bm3d_pipeline);
            pipelines.push_back(g.bm3d_match_pipeline);
            pipelines.push_back(g.bm3d_filter_pipeline);
        }
        for (auto & p : planes) {
            pipelines.push_back(p.agg_pipeline);
        }
        vsfeel_destroy_pipelines(*gpu, pipelines);
        gpu_destroy_buffer(*gpu, match_trace);
        gpu_destroy_buffer(*gpu, matches);
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

// The subgroup width the kernels run at. The 8-lane groups need a width that is
// a multiple of 8, and 32 is the measured best on the target GPU
// (VSFEEL_BM3D_SUBGROUP=64 forces the wave64 path). A device that cannot be
// asked for 32 keeps its own width when that already suits the groups, and is
// otherwise asked for 16; only a width that both is not a multiple of 8 and
// cannot be changed is refused.
static std::variant<uint32_t, std::string>
choose_subgroup(const GPUDevice & gpu) {
    const int forced = env_int("VSFEEL_BM3D_SUBGROUP", 0);
    if (forced > 0) {
        return static_cast<uint32_t>(forced);
    }
    if (gpu.has_subgroup_size(32, 32)) {
        return 32u;
    }
    if (gpu.subgroup_size % 8 == 0) {
        return 0u;
    }
    if (gpu.has_subgroup_size(16, 32)) {
        return 16u;
    }
    return "BM3D needs a subgroup size that is a multiple of 8 lanes "
           "(its 8-lane groups share data with subgroup shuffles), and this "
           "device's is " +
           std::to_string(gpu.subgroup_size) + " and cannot be changed"s;
}

// The estimation kernels launch GROUPS independent 8-lane groups per
// workgroup. Two waves is the floor: at one wave the CU's workgroup slots can
// run out before the register file does, while the per-group shared arrays grow
// with the launch, so a larger CTA does not add resident groups once the
// shared-memory budget binds. The cap still comes from the device: its
// invocations, its subgroup budget and the 256-thread ceiling the shader's
// group table is sized for.
static uint32_t choose_workgroup(const GPUDevice & gpu, uint32_t subgroup,
                                 bool patch_lds) {
    if (const int forced = env_int("VSFEEL_BM3D_WORKGROUP", 0); forced > 0) {
        return static_cast<uint32_t>(forced);
    }
    const uint32_t sg = subgroup != 0 ? subgroup : gpu.subgroup_size;
    // Four 8-int seed rows and, staging the patch, 72 floats per group.
    const uint32_t per_group = 4u * 8u * 4u + (patch_lds ? 72u * 4u : 0u);
    uint32_t cap = std::min<uint32_t>({
        gpu.limits.maxComputeWorkGroupInvocations,
        gpu.limits.maxComputeWorkGroupSize[0],
        sg * std::max(gpu.max_compute_workgroup_subgroups, 1u),
        256u,
    });
    cap = std::max(cap / sg * sg, sg);
    uint32_t wg = std::min(std::max(32u, sg), cap);
    // A small-shared-memory part shrinks a subgroup at a time rather than
    // failing pipeline creation.
    while (wg > sg &&
           (wg / 8u) * per_group > gpu.limits.maxComputeSharedMemorySize) {
        wg -= sg;
    }
    return wg;
}

static std::variant<VkPipeline, std::string>
create_bm3d_pipeline(const GPUDevice & gpu, const BM3DData & d,
                     const Bm3dGroup & group, const uint32_t * code,
                     size_t code_size, VkPipelineLayout layout, bool patch_lds,
                     bool seeds) {

    // One entry's planes share their geometry, and the search parameters are
    // the first plane's (the reference's joint entry does the same); only the
    // per-plane sigmas differ.
    const auto & first = d.planes[group.planes[0]];
    const auto group_sigma = [&](int i) {
        return i < group.n_planes ? d.planes[group.planes[i]].sigma : 0.0f;
    };
    // The block below is the shader's spec-constant ids in order: 0..13 are the
    // original single-plane block, 14..16 the joint-entry additions, 17 the
    // matching threshold, 18 the match trace, and 22 selects the ringed
    // temporal accumulator/source addressing.
    struct Spec {
        int32_t width, height, stride;
        float sigma_y;
        int32_t block_step, bm_range, radius, ps_num, ps_range;
        float extractor;
        int32_t nosearch, noestimate, src_ring, final;
        float sigma_u, sigma_v;
        int32_t nplanes;
        float th_sse;
        int32_t trace;
        int32_t workgroup;
        int32_t groups;
        int32_t res_cap;
        int32_t ringed;
    } spec { first.width,
             first.height,
             first.stride,
             group_sigma(0),
             first.block_step,
             first.bm_range,
             group.radius,
             first.ps_num,
             first.ps_range,
             d.extractor,
             env_flag("VSFEEL_BM3D_NOSEARCH") ? 1 : 0,
             env_flag("VSFEEL_BM3D_NOESTIMATE") ? 1 : 0,
             group.src_ring,
             d.final ? 1 : 0,
             group_sigma(1),
             group_sigma(2),
             group.n_planes,
             d.th_sse,
             d.trace_frame >= 0 ? 1 : 0,
             static_cast<int32_t>(d.workgroup),
             static_cast<int32_t>(d.groups_per_wg),
             static_cast<int32_t>(group.res_cap),
             d.max_radius > 0 ? 1 : 0 };
    const std::array<VkSpecializationMapEntry, 23> entries { {
        { 0, offsetof(Spec, width), sizeof(int32_t) },
        { 1, offsetof(Spec, height), sizeof(int32_t) },
        { 2, offsetof(Spec, stride), sizeof(int32_t) },
        { 3, offsetof(Spec, sigma_y), sizeof(float) },
        { 4, offsetof(Spec, block_step), sizeof(int32_t) },
        { 5, offsetof(Spec, bm_range), sizeof(int32_t) },
        { 6, offsetof(Spec, radius), sizeof(int32_t) },
        { 7, offsetof(Spec, ps_num), sizeof(int32_t) },
        { 8, offsetof(Spec, ps_range), sizeof(int32_t) },
        { 9, offsetof(Spec, extractor), sizeof(float) },
        { 10, offsetof(Spec, nosearch), sizeof(int32_t) },
        { 11, offsetof(Spec, noestimate), sizeof(int32_t) },
        { 12, offsetof(Spec, src_ring), sizeof(int32_t) },
        { 13, offsetof(Spec, final), sizeof(int32_t) },
        { 14, offsetof(Spec, sigma_u), sizeof(float) },
        { 15, offsetof(Spec, sigma_v), sizeof(float) },
        { 16, offsetof(Spec, nplanes), sizeof(int32_t) },
        { 17, offsetof(Spec, th_sse), sizeof(float) },
        { 18, offsetof(Spec, trace), sizeof(int32_t) },
        { 19, offsetof(Spec, workgroup), sizeof(int32_t) },
        { 20, offsetof(Spec, groups), sizeof(int32_t) },
        { 21, offsetof(Spec, res_cap), sizeof(int32_t) },
        { 22, offsetof(Spec, ringed), sizeof(int32_t) },
    } };
    // The subgroup width and the launch size are chosen once in BM3DCreate so
    // the grid, the split's handoff and every pipeline agree on them.
    const uint32_t groups = d.groups_per_wg;
    // LDS: s_x/s_y/s0_x/s0_y hold the prediction seeds of each 8-lane group
    // (four arrays of 8 entries per group), plus the group's reference patch
    // when it is staged there (72 floats; the stride is a multiple of 16 bytes
    // so a patch row reads as vector loads, and it keeps the groups in
    // different banks). The per-lane candidate lists live in registers, so
    // nothing else is shared. The split's filter half uses neither, and
    // declaring LDS it does not use would cost workgroups per SM.
    const uint32_t patch_bytes = patch_lds ? groups * 72 * 4 : 0;
    const uint32_t seed_bytes = seeds ? groups * 8 * 4 * 4 : 0;
    const GpuWorkgroup workgroup { .x = d.workgroup,
                                   .shared_bytes = seed_bytes + patch_bytes };
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(),
                               &spec, static_cast<uint32_t>(entries.size()),
                               sizeof(spec), "bm3d", d.subgroup_size,
                               workgroup);
}

static std::variant<VkPipeline, std::string>
create_agg_pipeline(const GPUDevice & gpu, const Bm3dPlane & plane,
                    const Bm3dGroup & group, const BM3DData & d,
                    const uint32_t * code, size_t code_size,
                    VkPipelineLayout layout, uint32_t wg_x, uint32_t wg_y) {

    struct Spec {
        int32_t height, stride, res_cap, nplanes, ringed;
        // Ids 5 and 6 are the launch shape itself, which the shader reads back
        // as gl_WorkGroupSize. The tile's shared layout is sized for 16 columns
        // and the rows the longest box reaches, so any shape up to 16x16 fits.
        uint32_t wg_x, wg_y;
    } spec { plane.height,
             plane.stride,
             group.res_cap,
             group.n_planes,
             d.max_radius > 0 ? 1 : 0,
             wg_x,
             wg_y };
    const std::array<VkSpecializationMapEntry, 7> entries { {
        { 0, offsetof(Spec, height), sizeof(int32_t) },
        { 1, offsetof(Spec, stride), sizeof(int32_t) },
        { 2, offsetof(Spec, res_cap), sizeof(int32_t) },
        { 3, offsetof(Spec, nplanes), sizeof(int32_t) },
        { 4, offsetof(Spec, ringed), sizeof(int32_t) },
        { 5, offsetof(Spec, wg_x), sizeof(uint32_t) },
        { 6, offsetof(Spec, wg_y), sizeof(uint32_t) },
    } };
    // Normalize is a tiled kernel: it needs a shared tile of the block weights
    // around the output tile for the separable box sum.
    constexpr uint32_t kSpan = 16 + 8 - 1;
    constexpr uint32_t kShared = (kSpan * kSpan + kSpan * 16) * 4;
    const GpuWorkgroup kAggWorkgroup { .x = wg_x,
                                       .y = wg_y,
                                       .shared_bytes = kShared };
    return gpu_create_pipeline(gpu, code, code_size, layout, entries.data(),
                               &spec, static_cast<uint32_t>(entries.size()),
                               sizeof(spec), "bm3d_agg", 0, kAggWorkgroup);
}

// The copy-and-widen kernel an integer input needs: one linear range per source
// plane, no LDS. It declares the first two bindings of the estimation kernels'
// set layout (source plane, ring) and its own block inside their push-constant
// range, so the same pipeline layout serves it. The element block is the
// host's fitted shape (spec constant 0): 256 where the device allows it, 128 on
// the invocation floor.
static std::variant<VkPipeline, std::string>
create_copy_pipeline(const GPUDevice & gpu, VkPipelineLayout layout,
                     const uint32_t * code, size_t code_size,
                     uint32_t threads) {
    const VkSpecializationMapEntry entry { 0, 0, sizeof(uint32_t) };
    // The name is unique per file on purpose: tools/shader_limits.py keys its
    // workgroup table by variable name, so reusing `workgroup` here would
    // shadow the estimation kernel's and report a false mismatch.
    const GpuWorkgroup kCopyWorkgroup { .x = threads };
    return gpu_create_pipeline(gpu, code, code_size, layout, &entry, &threads,
                               1, sizeof(threads), "bm3d_copy", 0,
                               kCopyWorkgroup);
}

// The RGB <-> OPP pair (bm3d_opp.comp): four bindings and a linear index, so it
// shares the estimation kernels' set layout and push-constant block range. It
// uses the copy kernel's element block, and the same grid.
static std::variant<VkPipeline, std::string>
create_opp_pipeline(const GPUDevice & gpu, VkPipelineLayout layout,
                    const uint32_t * code, size_t code_size, bool input,
                    uint32_t threads) {
    const VkSpecializationMapEntry entry { 0, 0, sizeof(uint32_t) };
    const GpuWorkgroup kOppWorkgroup { .x = threads };
    return gpu_create_pipeline(
        gpu, code, code_size, layout, &entry, &threads, 1, sizeof(threads),
        input ? "bm3d_opp_in" : "bm3d_opp_out", 0, kOppWorkgroup);
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// Radius 0 has no cross-frame sharing, so a frame takes one private slot per
// entry for its whole life; it returns it as soon as aggregation is queued.
static void give_r0_slot(BM3DData * d, Bm3dGroup & g, int slot) {
    std::lock_guard lock(d->cache_lock);
    g.r0_free.push_back(slot);
    d->cache_cv.notify_all();
}

// Reserve one private slot per entry for this radius-0 frame. Taking every
// entry's slot in one critical section keeps concurrent frames from waiting on
// each other while holding one, and no recording context is held while blocked.
static void acquire_r0_slots(BM3DData * d, Bm3dFrame & fr, int n) {
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & gf = fr.g[gi];
        gf.win_slots.fill(-1);
        gf.win_writer.fill(-1);
        gf.win_valid.fill(false);
        gf.win_recompute.fill(false);
        gf.upload_new.fill(false);
        gf.src_slot.fill(-1);
        gf.src_writer.fill(-1);
        gf.n_src = 1;
        gf.src_lo = n;
    }

    std::unique_lock lock(d->cache_lock);
    // Every entry's free slot is taken in one step, so no frame waits for
    // another while holding one.
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
        gf.win_valid[0] = true;
        gf.win_recompute[0] = true;
        gf.upload_new[0] = true;
        gf.src_slot[0] = fr.slot0[gi];
        gf.src_writer[0] = n;
    }
}

// Return the private slots once the frame's aggregation has been queued; a
// later frame's leading barrier completes the ordering.
static void release_r0_slots(BM3DData * d, const Bm3dFrame & fr) {
    for (int gi = 0; gi < d->n_groups; ++gi) {
        give_r0_slot(d, d->groups[gi], fr.slot0[gi]);
    }
}

// Window positions whose estimate this frame must compute. Every valid
// position is a distinct centre frame with its own slot, so nothing is
// deduplicated here; positions outside the clip compute nothing at all.
static void collect_est_positions(BM3DData * d, Bm3dGroupFrame & gf) {
    gf.n_pos = 0;
    if (d->max_radius == 0) {
        gf.est_pos[gf.n_pos++] = 0;
        return;
    }
    for (int i = 0; i < d->max_tw; ++i) {
        if (gf.win_valid[i] && gf.win_recompute[i]) {
            gf.est_pos[gf.n_pos++] = i;
        }
    }
}

// The buffers both kernels address, in the order the shader declares them:
// estimate stacks, source ring, destination/trace, and the split's handoff.
static void bm3d_bind(const GPUDevice & gpu, VkCommandBuffer cmd,
                      VkPipelineLayout layout, VkBuffer src, VkBuffer res,
                      VkBuffer dst, VkBuffer aux) {
    const VkBuffer bufs[4] { res, src, dst, aux };
    gpu_push_buffers(gpu, cmd, layout, bufs, 4);
}

// Dump the group the matcher selected for the traced reference block, once the
// submission that recorded it has completed. The layout is the one the kernel
// writes: a marker, the counts, then the eight (x, y, z, error) slots, then the
// per-frame retained/seed counts of both temporal directions.
static void dump_match_trace(const BM3DData * d, int trace_x, int trace_y) {
    const volatile uint32_t * t = d->match_trace_mapped;
    if (!t) {
        return;
    }
    if (t[0] != 0x0B3D0001u) {
        fprintf(stderr,
                "[bm3d-trace] (%d, %d): no group matched -- the origin must be "
                "one of the matcher's reference-block positions\n",
                trace_x, trace_y);
        return;
    }
    const auto as_float = [](uint32_t bits) {
        return std::bit_cast<float>(bits);
    };
    fprintf(stderr,
            "[bm3d-trace] x=%d y=%d th_mse=%.9g th_sse=%.9g n=%u retained=%u "
            "overflow=%u current=%u seeds=%u\n",
            trace_x, trace_y, static_cast<double>(d->th_mse),
            static_cast<double>(d->th_sse), t[1], t[2], t[3], t[4], t[5]);
    for (uint32_t k = 0; k < 8; ++k) {
        fprintf(stderr, "[bm3d-trace] member %u x=%u y=%u z=%u err=%.9g\n", k,
                t[6 + 4 * k], t[7 + 4 * k], t[8 + 4 * k],
                static_cast<double>(as_float(t[9 + 4 * k])));
    }
    // The frame counters only exist when the kernel ran the predictive chain;
    // a zero threshold (or the no-search diagnostic) stops it before the first
    // neighbour, and the words are then just the buffer's zero fill.
    if (d->th_sse <= 0.0f || env_flag("VSFEEL_BM3D_NOSEARCH")) {
        return;
    }
    for (int direction = -1; direction <= 1; direction += 2) {
        const int nt = std::min(
            d->max_radius,
            direction < 0 ? d->trace_frame : d->nframes - 1 - d->trace_frame);
        for (int step = 1; step <= nt; ++step) {
            const int base =
                38 + ((direction < 0 ? 0 : d->max_radius) + step - 1) * 2;
            fprintf(stderr,
                    "[bm3d-trace] %s step=%d frame=%d retained=%u seeds=%u\n",
                    direction < 0 ? "bwd" : "fwd", step,
                    d->trace_frame + direction * step, t[base], t[base + 1]);
        }
    }
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

// Compute-to-compute handoff for the split: the search half's group writes must
// be visible to the filter half's reads. The fused path's ALL_COMMANDS barrier
// orders far more than this dependency needs.
static void bm3d_compute_barrier(const GPUDevice & gpu, VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
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
    const int r = d->max_radius;
    const int nf = d->nframes;
    const auto & g = d->groups[gi];
    const auto & gf = fr.g[gi];
    const int i = gf.est_pos[c];
    const int slot = gf.win_slots[i];
    const int m_i = n - r + i;
    // The split shares one handoff buffer across entries. The match half of
    // the next entry would otherwise overwrite it while this entry's filter
    // dispatch is still reading; the match->filter barrier below only orders
    // one entry's pair.
    if (d->split_kernel && gi > 0) {
        bm3d_compute_barrier(*d->gpu, cmd);
    }
    if (d->dump) {
        fprintf(stderr, "[d] n=%d entry %d computes slot %d for frame %d\n", n,
                gi, slot, m_i);
    }

    // The whole slot is zeroed in one fill: every plane of the entry packs
    // into it, and one plane's estimation run owns all of them.
    const VkDeviceSize slot_elems =
        static_cast<VkDeviceSize>(g.n_planes) * 2 * g.pe;
    const VkDeviceSize res_off = static_cast<VkDeviceSize>(slot) * slot_elems;
    const auto zero_timer = d->profile.begin(cmd, "BM3D zero");
    d->gpu->vk->vkCmdFillBuffer(cmd, g.res.buffer, res_off * 4, slot_elems * 4,
                                0);
    d->profile.end(cmd, zero_timer);

    // The zero-fill precedes the estimation dispatch.
    bm3d_full_barrier(*d->gpu, cmd);

    // The match trace rides on the destination binding, which the estimation
    // kernel never writes: only the dispatch that estimates the traced centre
    // frame itself records, and only the group holding the requested reference
    // block matches inside the kernel.
    //
    // The traced centre is estimated once and then served from the cache, so
    // whichever output frame computes it has to carry the trace: requiring the
    // output frame to be the traced one too left the trace unrecorded whenever
    // an earlier frame (n - radius) got there first, which is what a sequential
    // request load does. The dump still waits for the traced frame's own
    // aggregation, by which point the holder's estimation has been waited out.
    int32_t trace_xy = -1;
    if (d->trace_frame >= 0 && m_i == d->trace_frame) {
        trace_xy =
            static_cast<int32_t>((d->trace_x & 0xFFFF) | (d->trace_y << 16));
    }
    const VkBuffer dst_bind = trace_xy >= 0 ? d->match_trace.buffer : dst_plane;

    // With the split, the same bindings and push constants serve both halves:
    // the match buffer is binding 3 and only the pipeline changes between the
    // two dispatches.
    const VkBuffer aux_bind =
        d->split_kernel ? d->matches.buffer : g.src.buffer;
    d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  d->split_kernel ? g.bm3d_match_pipeline
                                                  : g.bm3d_pipeline);
    // The estimation kernel reads only the estimate stacks and the source ring;
    // the destination binding carries the output plane and is unused here.
    bm3d_bind(*d->gpu, cmd, d->pipeline_layout, g.src.buffer, g.res.buffer,
              dst_bind, aux_bind);
    {
        const int32_t slot0_push =
            static_cast<int32_t>((r == 0) ? fr.slot0[gi] : 0);
        const int32_t pushes[7] {
            0, // acc_base: the accumulator slot is derived in the shader
            m_i,      nf, slot0_push,
            0, // src_base: unused with RINGED=0 (a private slot)
            trace_xy,
            -1 // focus: radius 0 scatters every member
        };
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, pushes,
                           sizeof(pushes));
    }
    const auto est_timer = d->profile.begin(
        cmd, d->split_kernel ? "BM3D blockMatch" : "BM3D blockMatch+filter");
    d->gpu->vk->vkCmdDispatch(cmd, g.bm3d_grid_x, g.bm3d_grid_y, 1);
    d->profile.end(cmd, est_timer);
    if (d->split_kernel) {
        // The groups the search half wrote must be visible to the filter half:
        // the barrier is the whole handoff.
        bm3d_compute_barrier(*d->gpu, cmd);
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      g.bm3d_filter_pipeline);
        const auto filter_timer = d->profile.begin(cmd, "BM3D filter");
        d->gpu->vk->vkCmdDispatch(cmd, g.bm3d_grid_x, g.bm3d_grid_y, 1);
        d->profile.end(cmd, filter_timer);
    }
}

// The input planes one source frame contributes to the ring copy: the frame
// being denoised, plus the basic-estimate clip in final mode. The range is the
// frame's own _Range, so each clip and each frame keeps its own code mapping.
struct Bm3dWindowCopy {
    VkBuffer source[3] {};
    VkBuffer ref[3] {};
    std::array<Bm3dRange, 3> src_range {};
    std::array<Bm3dRange, 3> ref_range {};
};

// Copy the source window's planes (the union of all windows that this record's
// dispatches may need, clamped to [n-2r, n+2r]) into each entry's src ring.
// Every entry packs [clip][plane] per slot, so a plane is one `pe` hop away and
// the slot stride spans the entry's planes.
//
// The ring is float at every input depth: a 2-byte clip's samples are widened
// once, here, by bm3d_copy_u16 / bm3d_copy_f16 rather than on every load in the
// matcher (see that shader). Offsets are in elements, so the destination stride
// stays the source plane's element stride and the copy is one linear range
// either way.
static void record_src_copies(BM3DData * d, const Bm3dFrame & fr,
                              VkCommandBuffer cmd, bool scratch,
                              const std::vector<Bm3dWindowCopy> & window) {
    const int clips = d->final ? 2 : 1;
    const auto copy_plane = [&](VkBuffer src, VkBuffer dst, VkDeviceSize offset,
                                VkDeviceSize bytes, const Bm3dGroup & g,
                                const Bm3dRange & range) {
        const auto timer = d->profile.begin(cmd, "BM3D convert");
        if (d->bits == 16) {
            struct CopyPush {
                uint32_t off;
                uint32_t count;
                float neutral;
                float gain;
            } push { static_cast<uint32_t>(offset),
                     static_cast<uint32_t>(bytes), range.neutral, range.gain };
            const VkBuffer bufs[2] { src, dst };
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          d->copy_pipeline);
            gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, bufs, 2);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &push,
                               sizeof(push));
            // The grid is the entry's extent at creation, so it covers this
            // plane's (never larger) count; past pc.count a thread returns.
            d->gpu->vk->vkCmdDispatch(cmd, g.copy_grid_x, g.copy_grid_y, 1);
            d->profile.end(cmd, timer);
            return;
        }
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
        d->profile.end(cmd, timer);
    };
    // RGB is filtered in OPP: one dispatch reads the three source planes and
    // writes this entry's OPP plane into the ring, the same linear range the
    // copy path handles.
    const auto convert_plane = [&](const VkBuffer rgb[3], VkBuffer dst,
                                   VkDeviceSize offset, VkDeviceSize bytes,
                                   const Bm3dGroup & g, const Bm3dRange & range,
                                   int plane) {
        const auto timer = d->profile.begin(cmd, "BM3D opp in");
        Bm3dOppInPush push { static_cast<uint32_t>(offset),
                             static_cast<uint32_t>(bytes), plane, range.neutral,
                             range.gain };
        const VkBuffer bufs[4] { rgb[0], rgb[1], rgb[2], dst };
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      d->opp_in_pipeline);
        gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, bufs, 4);
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &push,
                           sizeof(push));
        d->gpu->vk->vkCmdDispatch(cmd, g.copy_grid_x, g.copy_grid_y, 1);
        d->profile.end(cmd, timer);
    };
    for (int gi = 0; gi < d->n_groups; ++gi) {
        const auto & g = d->groups[gi];
        const auto & gf = fr.g[gi];
        const VkBuffer src_buf = scratch ? g.scratch_src.buffer : g.src.buffer;
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
                const VkDeviceSize src_off =
                    slot_base +
                    static_cast<VkDeviceSize>(clips - 1) * g.n_planes * g.pe +
                    static_cast<VkDeviceSize>(pi) * g.pe;
                if (d->rgb) {
                    convert_plane(w.source, src_buf, src_off, g.pe, g,
                                  w.src_range[plane], plane);
                } else {
                    copy_plane(w.source[plane], src_buf, src_off, g.pe, g,
                               w.src_range[plane]);
                }
                if (d->final) {
                    const VkDeviceSize ref_off =
                        slot_base + static_cast<VkDeviceSize>(pi) * g.pe;
                    if (d->rgb) {
                        convert_plane(w.ref, src_buf, ref_off, g.pe, g,
                                      w.ref_range[plane], plane);
                    } else {
                        copy_plane(w.ref[plane], src_buf, ref_off, g.pe, g,
                                   w.ref_range[plane]);
                    }
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
        record_src_copies(d, fr, cmd, false, window);
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

// Record the normalize submission into the context's command buffer. It reads
// the frame's accumulator and writes the output plane, so nothing is
// downloaded afterwards. One dispatch per processed plane, each with its own
// geometry and its entry's accumulator ring.
static void record_bm3d_agg(BM3DData * d, const Bm3dFrame * fr,
                            VkCommandBuffer cmd, int n, int base, bool isolated,
                            const std::array<VkBuffer, 3> & dst_planes,
                            const std::array<Bm3dRange, 3> & out_range,
                            bool gputrace, const GpuBuffer & opp) {
    const int r = d->max_radius;
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

        for (int pi = 0; pi < g.n_planes; ++pi) {
            const int plane = g.planes[pi];
            const auto & p = d->planes[plane];
            // A joint entry's skipped plane was never computed: its output is
            // the source plane the caller shared in.
            if (!p.process) {
                continue;
            }

            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          p.agg_pipeline);
            // Descriptor bindings do not carry across command buffers: the
            // normalization is recorded separately from the estimation phase,
            // so without this bind the dispatch runs on undefined descriptor
            // state (black output, and device loss under concurrent
            // submissions).
            // An RGB plane is normalized into the OPP scratch; the output
            // conversion below turns the three of them into the RGB frame.
            const VkBuffer dst_buf = d->rgb ? opp.buffer : dst_planes[plane];
            const int32_t dst_base =
                d->rgb ? static_cast<int32_t>(static_cast<VkDeviceSize>(plane) *
                                              d->opp_pe)
                       : 0;
            bm3d_bind(*d->gpu, cmd, d->pipeline_layout,
                      isolated ? g.scratch_src.buffer : g.src.buffer,
                      isolated ? g.scratch_acc.buffer : g.res.buffer, dst_buf,
                      isolated ? g.scratch_src.buffer : g.src.buffer);
            {
                const int clips = d->final ? 2 : 1;
                // The fallback source pixel: the *source* section of this
                // frame's slot in the entry's ring, exactly where the ring
                // copy put it.
                const int src_slot =
                    (r == 0)
                        ? fr->slot0[gi]
                        : (isolated ? (n % g.src_ring)
                                    : (((n - base) % g.src_ring) + g.src_ring) %
                                          g.src_ring);
                const VkDeviceSize src_base =
                    (static_cast<VkDeviceSize>(src_slot) * clips +
                     (clips - 1)) *
                        g.n_planes * g.pe +
                    static_cast<VkDeviceSize>(pi) * g.pe;
                // The kernel rebuilds the accumulator slot from these; the
                // plane offset is in floats within the slot.
                const int32_t slot0_push = (r == 0) ? fr->slot0[gi] : 0;
                const int32_t plane_off = static_cast<int32_t>(
                    static_cast<VkDeviceSize>(pi) * 2 * g.pe);
                // Named rather than cast inside the list: cppcheck reads a cast
                // alone on an initializer's continuation line as a discarded
                // expression statement (constStatement), in some versions only.
                const int32_t src_base_push = static_cast<int32_t>(src_base);
                // The layout is five ints followed by the two floats the u16
                // store inverts; 28 bytes is the pipeline layout's whole block.
                struct AggPush {
                    int32_t n;
                    int32_t acc_base;
                    int32_t slot0;
                    int32_t plane_off;
                    int32_t src_base;
                    float gain;
                    float bias;
                    int32_t dst_base;
                } push { n,
                         isolated ? n : base,
                         slot0_push,
                         plane_off,
                         src_base_push,
                         out_range[plane].gain != 0.0f
                             ? 1.0f / out_range[plane].gain
                             : 0.0f,
                         out_range[plane].neutral + 0.5f,
                         dst_base };
                gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &push,
                                   sizeof(push));
            }
            // The estimation kernel's atomic accumulation (and the fill that
            // zeroed the slot) must be visible to the normalize reads; the
            // reads are plain loads, so this explicit cross-dispatch barrier is
            // still required. Later planes read their own accumulator and write
            // their own output plane, disjoint from the first one's.
            if (first_dispatch) {
                bm3d_full_barrier(*d->gpu, cmd);
            }
            if (gputrace && first_dispatch) {
                d->gpu->vk->vkCmdWriteTimestamp2(
                    cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, d->probe.query,
                    2);
            }
            const auto agg_timer = d->profile.begin(cmd, "BM3D normalize");
            d->gpu->vk->vkCmdDispatch(cmd, p.agg_grid_x, p.agg_grid_y, 1);
            d->profile.end(cmd, agg_timer);
            if (gputrace && --remaining == 0) {
                d->gpu->vk->vkCmdWriteTimestamp2(
                    cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, d->probe.query,
                    3);
            }
            first_dispatch = false;
        }
    }
    if (d->rgb) {
        // The three OPP planes are in the scratch; a barrier makes them visible
        // to the output conversion, which writes each RGB plane of the frame.
        bm3d_full_barrier(*d->gpu, cmd);
        for (int plane = 0; plane < 3; ++plane) {
            const VkBuffer bufs[4] { opp.buffer, opp.buffer, opp.buffer,
                                     dst_planes[plane] };
            const Bm3dOppOutPush push { static_cast<uint32_t>(d->opp_pe),
                                        0,
                                        static_cast<uint32_t>(d->opp_pe),
                                        static_cast<uint32_t>(2 * d->opp_pe),
                                        plane,
                                        out_range[plane].gain != 0.0f
                                            ? 1.0f / out_range[plane].gain
                                            : 0.0f,
                                        out_range[plane].neutral + 0.5f };
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          d->opp_out_pipeline);
            gpu_push_buffers(*d->gpu, cmd, d->pipeline_layout, bufs, 4);
            gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, &push,
                               sizeof(push));
            const auto timer = d->profile.begin(cmd, "BM3D opp out");
            d->gpu->vk->vkCmdDispatch(cmd, d->opp_grid_x, d->opp_grid_y, 1);
            d->profile.end(cmd, timer);
        }
    }
}

// Record one centre's scatter into the request's command buffer. The centre's
// source window [m-r, m+r] is copied into the ring first (each frame once, as
// the cursor advances), the accumulator slots the window's frames occupy are
// zeroed when they are first assigned, and one estimation dispatch per entry
// scatters the estimates. The leading barrier orders the previous centre's
// reads behind these writes; the first normalize's barrier carries this
// centre's atomics across.
static void record_center_cmd(BM3DData * d, VkCommandBuffer cmd, int m,
                              VSFrameContext * frameCtx, const VSAPI * vsapi,
                              int acc_base, int focus, int & converted,
                              bool isolated,
                              std::vector<const VSFrame *> & holds) {
    const int r = d->max_radius;
    const int last = d->nframes - 1;
    const int first = std::max(0, m - r);
    const int lastFrame = std::min(last, m + r);
    const int count = lastFrame - first + 1;
    const int copyFrom = std::max(converted, first);

    std::vector<Bm3dWindowCopy> window(static_cast<size_t>(count));
    for (int f = copyFrom; f <= lastFrame; ++f) {
        const VSFrame * src = vsapi->getFrameFilter(f, d->node, frameCtx);
        if (!src) {
            throw std::runtime_error("source frame " + std::to_string(f) +
                                     " could not be fetched");
        }
        holds.push_back(src);
        auto & w = window[static_cast<size_t>(f - first)];
        for (int plane = 0; plane < d->num_planes; ++plane) {
            VSVulkanPlaneInfo info {};
            if (d->gpu->api->getGPUPlane(src, plane, &info)) {
                throw std::runtime_error("clip " + std::to_string(f) +
                                         " plane " + std::to_string(plane) +
                                         " is not GPU resident");
            }
            if (auto e =
                    vsfeel_plane_range_error(info, *d->gpu, "the source plane");
                !e.empty()) {
                throw std::runtime_error(e);
            }
            w.source[plane] = info.buffer;
            w.src_range[plane] =
                bm3d_range(d->vi->format, plane,
                           bm3d_full_range(src, d->vi->format, vsapi));
        }
        if (d->ref_node) {
            const VSFrame * rsrc =
                vsapi->getFrameFilter(f, d->ref_node, frameCtx);
            if (!rsrc) {
                throw std::runtime_error("ref frame " + std::to_string(f) +
                                         " could not be fetched");
            }
            holds.push_back(rsrc);
            for (int plane = 0; plane < d->num_planes; ++plane) {
                VSVulkanPlaneInfo info {};
                if (d->gpu->api->getGPUPlane(rsrc, plane, &info)) {
                    throw std::runtime_error("ref clip " + std::to_string(f) +
                                             " plane " + std::to_string(plane) +
                                             " is not GPU resident");
                }
                if (auto e = vsfeel_plane_range_error(info, *d->gpu,
                                                      "the reference plane");
                    !e.empty()) {
                    throw std::runtime_error(e);
                }
                w.ref[plane] = info.buffer;
                w.ref_range[plane] =
                    bm3d_range(d->vi->format, plane,
                               bm3d_full_range(rsrc, d->vi->format, vsapi));
            }
        }
    }

    // The per-entry copy table: frames [first, lastFrame] map to ring slots,
    // and only those at or past copyFrom are new.
    Bm3dFrame copyFrame {};
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & gf = copyFrame.g[gi];
        gf.src_lo = first;
        gf.n_src = count;
        for (int k = 0; k < count; ++k) {
            const int f = first + k;
            gf.src_slot[k] = f % d->groups[gi].src_ring;
            gf.upload_new[k] = (f >= copyFrom);
        }
    }

    bm3d_full_barrier(*d->gpu, cmd);

    // Zero every accumulator slot whose frame changes in this window. The
    // cursor only reassigns a slot after its frame retired, so a slot that
    // already holds f carries contributions from earlier centres and is left
    // alone.
    for (int gi = 0; gi < d->n_groups && !isolated; ++gi) {
        auto & g = d->groups[gi];
        const VkDeviceSize slot_bytes =
            static_cast<VkDeviceSize>(g.n_planes) * 2 * g.pe * 4;
        for (int f = first; f <= lastFrame; ++f) {
            int slot = f % g.res_cap;
            if (slot < 0) {
                slot += g.res_cap;
            }
            if (g.acc_frame[slot] == f) {
                continue;
            }
            const auto timer = d->profile.begin(cmd, "BM3D zero");
            d->gpu->vk->vkCmdFillBuffer(
                cmd, g.res.buffer, static_cast<VkDeviceSize>(slot) * slot_bytes,
                slot_bytes, 0);
            d->profile.end(cmd, timer);
            g.acc_frame[slot] = f;
        }
    }

    record_src_copies(d, copyFrame, cmd, isolated, window);
    bm3d_full_barrier(*d->gpu, cmd);

    const int32_t trace_xy =
        (d->trace_frame >= 0 && m == d->trace_frame)
            ? static_cast<int32_t>((d->trace_x & 0xFFFF) | (d->trace_y << 16))
            : -1;
    const VkBuffer dst_bind =
        trace_xy >= 0 ? d->match_trace.buffer : d->groups[0].src.buffer;
    const VkBuffer aux_bind =
        d->split_kernel ? d->matches.buffer : d->groups[0].src.buffer;

    for (int gi = 0; gi < d->n_groups; ++gi) {
        // Same write-after-read hazard as record_est_position: one split
        // handoff buffer is shared by every entry.
        if (d->split_kernel && gi > 0) {
            bm3d_compute_barrier(*d->gpu, cmd);
        }
        const auto & g = d->groups[gi];
        const VkBuffer src_buf = isolated ? g.scratch_src.buffer : g.src.buffer;
        const VkBuffer acc_buf = isolated ? g.scratch_acc.buffer : g.res.buffer;
        d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      d->split_kernel ? g.bm3d_match_pipeline
                                                      : g.bm3d_pipeline);
        bm3d_bind(*d->gpu, cmd, d->pipeline_layout, src_buf, acc_buf, dst_bind,
                  aux_bind);
        const int32_t pushes[7] {
            acc_base, m,    d->nframes, 0,
            0, // src_base: slots are frame % SRC_RING, so 0 is the ring base
            trace_xy, focus
        };
        gpu_push_constants(*d->gpu, cmd, d->pipeline_layout, pushes,
                           sizeof(pushes));
        const auto est_timer =
            d->profile.begin(cmd, d->split_kernel ? "BM3D blockMatch"
                                                  : "BM3D blockMatch+filter");
        d->gpu->vk->vkCmdDispatch(cmd, g.bm3d_grid_x, g.bm3d_grid_y, 1);
        d->profile.end(cmd, est_timer);
        if (d->split_kernel) {
            bm3d_compute_barrier(*d->gpu, cmd);
            d->gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          g.bm3d_filter_pipeline);
            const auto filter_timer = d->profile.begin(cmd, "BM3D filter");
            d->gpu->vk->vkCmdDispatch(cmd, g.bm3d_grid_x, g.bm3d_grid_y, 1);
            d->profile.end(cmd, filter_timer);
        }
    }
    converted = lastFrame + 1;
}

// Allocate the output frame for f and record its normalization into the
// request's command buffer. The frame's unprocessed planes are shared from the
// centre source frame, as in the radius-0 path.
static VSFrame * normalize_cmd(BM3DData * d, VkCommandBuffer cmd,
                               VSGPUExecContext * ctx, int f,
                               VSFrameContext * frameCtx, VSCore * core,
                               const VSAPI * vsapi, int base, bool isolated) {
    const VSFrame * center = vsapi->getFrameFilter(f, d->node, frameCtx);
    if (!center) {
        throw std::runtime_error("centre frame " + std::to_string(f) +
                                 " could not be fetched");
    }
    // The output store inverts the centre frame's own _Range.
    std::array<Bm3dRange, 3> out_range {};
    for (int p = 0; p < d->num_planes; ++p) {
        out_range[p] = bm3d_range(
            d->vi->format, p, bm3d_full_range(center, d->vi->format, vsapi));
    }
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
        dst = vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height,
                                    fr, pl, center, core);
    }
    vsapi->freeFrame(center);
    if (!dst) {
        throw std::runtime_error("failed to allocate the output frame");
    }

    std::array<VkBuffer, 3> dst_planes {};
    for (int plane = 0; plane < d->num_planes; ++plane) {
        if (!d->planes[plane].process) {
            continue;
        }
        VSVulkanPlaneInfo info {};
        if (d->gpu->api->getGPUPlane(dst, plane, &info)) {
            vsapi->freeFrame(dst);
            throw std::runtime_error("the output frame is not GPU resident");
        }
        if (auto e =
                vsfeel_plane_range_error(info, *d->gpu, "the output plane");
            !e.empty()) {
            vsapi->freeFrame(dst);
            throw std::runtime_error(e);
        }
        dst_planes[plane] = info.buffer;
    }

    // RGB: the aggregation writes the three OPP planes here and the output
    // conversion below reads them. The buffer lives as long as this
    // submission's context, so concurrent frames never share it.
    GpuBuffer opp;
    if (d->rgb) {
        if (auto e =
                gpu_frame_buffer(*d->gpu, core, ctx, 3 * d->opp_pe * 4, opp);
            !e.empty()) {
            vsapi->freeFrame(dst);
            throw std::runtime_error("the opponent output scratch could not "
                                     "be allocated: " +
                                     e);
        }
    }
    record_bm3d_agg(d, nullptr, cmd, f, base, isolated, dst_planes, out_range,
                    false, opp);
    return dst;
}

// Reap finished frames beyond the cache's cap, keeping the one just served and
// anything the core has requested but not taken. Both returns from the temporal
// path go through this: a request served by the isolated ring used to skip it,
// so a reverse or random-access run grew the cache by one frame per request.
static void reap_emitted(BM3DData * d, const VSAPI * vsapi, int keep) {
    for (auto it = d->emitted.begin();
         d->emitted.size() > d->emitted_cap && it != d->emitted.end();) {
        if (it->first == keep || d->pending.count(it->first) != 0) {
            ++it;
            continue;
        }
        vsapi->freeFrame(it->second);
        it = d->emitted.erase(it);
    }
}

// A request the cursor cannot advance to (a seek, or a frame it has already
// passed) is computed in its own one-window ring: the centres around it are
// scattered with only the requested frame kept, and the main ring keeps its
// state. This is what lets the cursor stay monotonic when vspipe asks for one
// frame far ahead and then returns to the frames in between.
static VSFrame * record_isolated(BM3DData * d, int n, VSFrameContext * frameCtx,
                                 VSCore * core, const VSAPI * vsapi) {
    const int r = d->max_radius;
    const int last = d->nframes - 1;
    const int first = std::max(0, n - r);
    const int top = std::min(last, n + r);
    if (d->trace) {
        fprintf(stderr, "[t] n=%d isolated cached=%zu\n", n, d->emitted.size());
    }
    std::vector<const VSFrame *> holds;
    VSGPUExecContext * ctx = nullptr;
    char errbuf[512] {};
    try {
        ctx = d->gpu->api->gpuExecAcquire(d->exec, errbuf, sizeof(errbuf));
        if (!ctx) {
            throw std::runtime_error("could not acquire a recording context: " +
                                     std::string(errbuf));
        }
        VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
        // Allocate the private source ring on the first isolated request and
        // keep it: a linear playback never pays it, and a seek-heavy consumer
        // pays one allocation instead of one per request.
        for (int gi = 0; gi < d->n_groups; ++gi) {
            auto & g = d->groups[gi];
            if (g.scratch_src.handle) {
                continue;
            }
            if (auto e = gpu_make_buffer(*d->gpu, core, g.src_size * 4,
                                         g.scratch_src,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                !e.empty()) {
                throw std::runtime_error(
                    "the isolated source ring could not be allocated: " + e);
            }
        }
        bm3d_full_barrier(*d->gpu, cmd);
        for (int gi = 0; gi < d->n_groups; ++gi) {
            const auto & g = d->groups[gi];
            const VkDeviceSize slot_bytes =
                static_cast<VkDeviceSize>(g.n_planes) * 2 * g.pe * 4;
            const auto timer = d->profile.begin(cmd, "BM3D zero");
            d->gpu->vk->vkCmdFillBuffer(cmd, g.scratch_acc.buffer, 0,
                                        slot_bytes, 0);
            d->profile.end(cmd, timer);
        }
        int converted = std::max(0, n - 2 * r);
        for (int m = first; m <= top; ++m) {
            record_center_cmd(d, cmd, m, frameCtx, vsapi, n, n, converted, true,
                              holds);
        }
        VSFrame * out =
            normalize_cmd(d, cmd, ctx, n, frameCtx, core, vsapi, n, true);
        for (const VSFrame * h : holds) {
            d->gpu->api->gpuExecReadsFrame(ctx, h);
        }
        for (int plane = 0; plane < d->num_planes; ++plane) {
            if (d->planes[plane].process) {
                d->gpu->api->gpuExecWritesPlane(ctx, out, plane);
            }
        }
        uint64_t value = 0;
        const int submit_err =
            d->gpu->api->gpuExecSubmit(ctx, &value, errbuf, sizeof(errbuf));
        ctx = nullptr;
        if (submit_err) {
            vsapi->freeFrame(out);
            throw std::runtime_error("isolated submit failed: " +
                                     std::string(errbuf));
        }
        for (const VSFrame * h : holds) {
            vsapi->freeFrame(h);
        }
        if (d->trace_frame >= 0 && n == d->trace_frame) {
            char terr[256] {};
            if (d->gpu->api->gpuExecWaitValue(d->exec, value, terr,
                                              sizeof(terr)) == gdDrained) {
                dump_match_trace(d, d->trace_x, d->trace_y);
            }
        }
        return out;
    } catch (...) {
        if (ctx) {
            d->gpu->api->gpuExecAbandon(ctx);
        }
        for (const VSFrame * h : holds) {
            vsapi->freeFrame(h);
        }
        throw;
    }
}

// The temporal frame path: process centres in order, scatter each into the
// per-frame accumulators, and normalize every frame whose last contributor is
// in. One command buffer per request carries the whole advance; the pool then
// overlaps it with the next request.
static const VSFrame * temporalFrame(BM3DData * d, int n,
                                     VSFrameContext * frameCtx, VSCore * core,
                                     const VSAPI * vsapi) {
    std::lock_guard lock(d->temporal_lock);
    vsfeel_trace_frame_begin();
    if (d->failed) {
        vsfeel_trace_error(d->name, n, "a previous frame failed", d->gpu.get());
        vsapi->setFilterError(
            (std::string(d->name) + ": a previous frame failed").c_str(),
            frameCtx);
        return nullptr;
    }
    if (n == d->fault_frame) {
        d->failed = true;
        d->cursor.valid = false;
        vsfeel_trace_error(d->name, n, "injected estimation fault",
                           d->gpu.get());
        vsapi->setFilterError(
            (std::string(d->name) + ": injected estimation fault").c_str(),
            frameCtx);
        return nullptr;
    }
    if (n < 0 || n >= d->nframes) {
        return nullptr;
    }
    const int r = d->max_radius;
    int req_lo = std::max(0, n - 2 * r);
    if (const auto it = d->request_lo.find(n); it != d->request_lo.end()) {
        req_lo = it->second;
        d->request_lo.erase(it);
    }
    // A frame normalized earlier (this request is late, or another consumer
    // asked for it) is served from the emitted ring.
    if (const auto it = d->emitted.find(n); it != d->emitted.end()) {
        if (const auto p = d->pending.find(n); p != d->pending.end()) {
            d->pending.erase(p);
        }
        return vsapi->addFrameRef(it->second);
    }
    const int last = d->nframes - 1;
    auto & c = d->cursor;
    // Continue while the request is at or a little ahead of the cursor: the
    // intermediate frames are normalized into the emitted ring, which is what
    // keeps a small out-of-order step from costing a restart. A seek, or a
    // request for a frame the ring has already dropped, restarts.
    constexpr int kMaxJump = 24;
    // What the cursor would have to copy for this request: only source frames
    // at or above this one are fetched. The arInitial range the request was
    // given must cover that, or advancing would fetch frames nobody requested.
    const int copy_from = std::max(c.converted_up_to, c.next_ref - r);
    const bool sources_ok = copy_from >= req_lo;
    if (n < c.retired_up_to || n - c.retired_up_to > kMaxJump || !sources_ok) {
        // A seek or a frame the cursor has passed: compute it in its own ring
        // so the main cursor is not disturbed.
        try {
            VSFrame * out = record_isolated(d, n, frameCtx, core, vsapi);
            // The isolated ring leaves the cursor where it was, so a forward
            // seek would otherwise pin every later frame to this same
            // full-window recompute. Re-anchoring skips the frames in between,
            // which is only free when none of them is still waiting at a
            // reachable distance: an out-of-order prefetch inside a batch must
            // keep catching up through the cursor, or each waiting frame
            // becomes its own one-window recompute. Re-anchor at the first
            // centre whose window covers the next frame, and forget the
            // accumulator tags so the next centre zeroes the slots it uses
            // instead of building on the old ring's contents.
            bool waiting = false;
            for (auto it = d->pending.lower_bound(c.retired_up_to);
                 it != d->pending.end() && *it < n; ++it) {
                if (*it - c.retired_up_to <= kMaxJump) {
                    waiting = true;
                    break;
                }
            }
            if (n >= c.retired_up_to && !waiting) {
                c.next_ref = std::max(0, n + 1 - r);
                c.retired_up_to = n + 1;
                c.converted_up_to = std::max(0, n + 1 - 2 * r);
                for (int gi = 0; gi < d->n_groups; ++gi) {
                    std::fill(d->groups[gi].acc_frame.begin(),
                              d->groups[gi].acc_frame.end(), -1);
                }
            }
            const auto [it, inserted] = d->emitted.emplace(n, out);
            if (!inserted) {
                vsapi->freeFrame(it->second);
                it->second = out;
            }
            if (const auto p = d->pending.find(n); p != d->pending.end()) {
                d->pending.erase(p);
            }
            reap_emitted(d, vsapi, n);
            return vsapi->addFrameRef(it->second);
        } catch (const std::exception & e) {
            d->failed = true;
            c.valid = false;
            if (const auto p = d->pending.find(n); p != d->pending.end()) {
                d->pending.erase(p);
            }
            vsfeel_trace_error(d->name, n, e.what(), d->gpu.get());
            vsapi->setFilterError(
                (std::string(d->name) + ": " + e.what()).c_str(), frameCtx);
            return nullptr;
        }
    }

    std::vector<const VSFrame *> holds;
    std::vector<VSFrame *> outputs;
    VSGPUExecContext * ctx = nullptr;
    char errbuf[512] {};
    try {
        ctx = d->gpu->api->gpuExecAcquire(d->exec, errbuf, sizeof(errbuf));
        if (!ctx) {
            throw std::runtime_error("could not acquire a recording context: " +
                                     std::string(errbuf));
        }
        VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
        // No leading barrier: the first centre records one before it writes the
        // ring, and the first normalize records one before it reads, so the
        // cross-submission dependency is already carried by the first command
        // that touches shared state.
        while (c.retired_up_to <= n) {
            const int m = c.next_ref;
            if (m <= last) {
                vsfeel_trace_mark("est");
                record_center_cmd(d, cmd, m, frameCtx, vsapi, 0, -1,
                                  c.converted_up_to, false, holds);
                c.next_ref = m + 1;
            }
            // Frames up to m-r have every contributor in. Normalize the ones
            // this request needs (and any it skipped past), keeping a small
            // ring of finished frames for late requesters.
            int ready = (m > last) ? last : m - r;
            ready = std::min(ready, n);
            while (c.retired_up_to <= ready) {
                const int f = c.retired_up_to;
                if (f != n && d->pending.count(f) == 0) {
                    // Nobody has asked for it: its accumulator can be reused
                    // without producing a frame.
                    ++c.retired_up_to;
                    continue;
                }
                vsfeel_trace_mark("normalize");
                VSFrame * out = normalize_cmd(d, cmd, ctx, f, frameCtx, core,
                                              vsapi, 0, false);
                const auto [it, inserted] = d->emitted.emplace(f, out);
                if (!inserted) {
                    vsapi->freeFrame(it->second);
                    it->second = out;
                }
                outputs.push_back(out);
                ++c.retired_up_to;
            }
            if (m > last && c.retired_up_to > n) {
                break;
            }
        }
        for (const VSFrame * h : holds) {
            d->gpu->api->gpuExecReadsFrame(ctx, h);
        }
        for (VSFrame * out : outputs) {
            for (int plane = 0; plane < d->num_planes; ++plane) {
                if (d->planes[plane].process) {
                    d->gpu->api->gpuExecWritesPlane(ctx, out, plane);
                }
            }
        }
        uint64_t value = 0;
        if (d->gpu->api->gpuExecSubmit(ctx, &value, errbuf, sizeof(errbuf))) {
            ctx = nullptr;
            throw std::runtime_error("temporal submit failed: " +
                                     std::string(errbuf));
        }
        ctx = nullptr;
        if (d->trace_frame >= 0 && n == d->trace_frame) {
            // The trace rode on this request's estimation submission, or an
            // earlier one the cache served, so waiting this submission out
            // leaves it final. Debug only: this is the one blocking
            // readback, and only for the traced frame.
            char terr[256] {};
            if (d->gpu->api->gpuExecWaitValue(d->exec, value, terr,
                                              sizeof(terr)) == gdDrained) {
                dump_match_trace(d, d->trace_x, d->trace_y);
            }
        }
    } catch (const std::exception & e) {
        if (ctx) {
            d->gpu->api->gpuExecAbandon(ctx);
        }
        for (const VSFrame * h : holds) {
            vsapi->freeFrame(h);
        }
        d->failed = true;
        c.valid = false;
        for (auto & [frame, node] : d->emitted) {
            (void)frame;
            vsapi->freeFrame(node);
        }
        d->emitted.clear();
        if (const auto it = d->pending.find(n); it != d->pending.end()) {
            d->pending.erase(it);
        }
        d->request_lo.erase(n);
        vsfeel_trace_error(d->name, n, e.what(), d->gpu.get());
        vsapi->setFilterError((std::string(d->name) + ": " + e.what()).c_str(),
                              frameCtx);
        return nullptr;
    }
    for (const VSFrame * h : holds) {
        vsapi->freeFrame(h);
    }

    // Reap emitted frames beyond the cache cap, keeping anything the core has
    // requested but not taken yet.
    reap_emitted(d, vsapi, n);
    if (const auto it = d->emitted.find(n); it != d->emitted.end()) {
        if (const auto p = d->pending.find(n); p != d->pending.end()) {
            d->pending.erase(p);
        }
        return vsapi->addFrameRef(it->second);
    }
    if (const auto p = d->pending.find(n); p != d->pending.end()) {
        d->pending.erase(p);
    }
    d->request_lo.erase(n);
    d->failed = true;
    c.valid = false;
    vsapi->setFilterError(
        (std::string(d->name) + ": the requested frame did not normalize")
            .c_str(),
        frameCtx);
    return nullptr;
}

static const VSFrame * VS_CC BM3DGetFrame(int n, int activationReason,
                                          void * instanceData,
                                          [[maybe_unused]] void ** frameData,
                                          VSFrameContext * frameCtx,
                                          VSCore * core, const VSAPI * vsapi) {

    BM3DData * d = static_cast<BM3DData *>(instanceData);

    if (activationReason == arInitial) {
        const int r = d->max_radius;
        int lo = std::clamp(n - 2 * r, 0, d->nframes - 1);
        const int hi = std::clamp(n + 2 * r, 0, d->nframes - 1);
        if (r > 0) {
            // A request the cursor can advance to needs the sources of every
            // centre it will pass, not just its own window. The cursor only
            // moves forward (an unreachable request is computed in its own
            // ring), so the range the cursor had here still covers the work.
            constexpr int kMaxJump = 24;
            std::lock_guard lock(d->temporal_lock);
            d->pending.insert(n);
            if (!d->cursor.valid) {
                // Start the cursor here rather than at the first execution: the
                // requests behind this one then get source ranges relative to a
                // position the cursor can actually advance from.
                auto & c = d->cursor;
                c.valid = true;
                c.next_ref = std::max(0, n - r);
                c.retired_up_to = n;
                c.converted_up_to = std::max(0, n - 2 * r);
                for (int gi = 0; gi < d->n_groups; ++gi) {
                    std::fill(d->groups[gi].acc_frame.begin(),
                              d->groups[gi].acc_frame.end(), -1);
                }
            }
            if (n >= d->cursor.retired_up_to &&
                n - d->cursor.retired_up_to <= kMaxJump) {
                lo = std::clamp(d->cursor.retired_up_to - 2 * r, 0,
                                d->nframes - 1);
            }
            d->request_lo[n] = lo;
        }
        for (int idx = lo; idx <= hi; ++idx) {
            vsapi->requestFrameFilter(idx, d->node, frameCtx);
            if (d->ref_node) {
                vsapi->requestFrameFilter(idx, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        if (d->max_radius > 0) {
            return temporalFrame(d, n, frameCtx, core, vsapi);
        }
        // The centre source frame is the property donor and, for a plane below
        // its sigma threshold, the plane's owner: newVideoFrame2 propagates GPU
        // residency -- and each shared plane's producer pair -- from it. A plane
        // is processed on its own unless chroma=True packs the three of a
        // 4:4:4 clip into one entry.
        const VSFrame * center = vsapi->getFrameFilter(n, d->node, frameCtx);
        if (!center) {
            vsapi->setFilterError((std::string(d->name) +
                                   ": the centre frame could not be fetched")
                                      .c_str(),
                                  frameCtx);
            return nullptr;
        }
        // The output store inverts the centre frame's own _Range.
        std::array<Bm3dRange, 3> out_range {};
        for (int p = 0; p < d->num_planes; ++p) {
            out_range[p] =
                bm3d_range(d->vi->format, p,
                           bm3d_full_range(center, d->vi->format, vsapi));
        }
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
            vsapi->setFilterError(
                (std::string(d->name) + ": failed to allocate the output frame")
                    .c_str(),
                frameCtx);
            return nullptr;
        }
        // All planes are processed or the node would have been passed through
        // at creation, so there is nothing further to do for a skipped one.

        vsfeel_trace_frame_begin();
        const auto cpu_now = [] {
            struct timespec ts {};
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
            return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
                   static_cast<uint64_t>(ts.tv_nsec);
        };
        auto c0 = d->host_timing ? cpu_now() : 0ull;
        auto t0 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        vsfeel_trace_mark("acquire");

        // The GPU-timing probe serializes instrumented frames: the one query
        // pool is shared, so the probe lock is taken before recording.
        bool gputrace = false;
        std::unique_lock<std::mutex> probe_lock(d->probe.lock, std::defer_lock);
        if (d->probe.enabled) {
            probe_lock.lock();
            gputrace = true;
        }

        // Reserve this frame's private radius-0 slots; blocks until another
        // in-flight frame returns one, holding no recording context meanwhile.
        Bm3dFrame fr;
        acquire_r0_slots(d, fr, n);
        auto c_t1 = d->host_timing ? cpu_now() : 0ull;
        auto t1 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

        const int n_src = fr.g[0].n_src;
        std::vector<Bm3dWindowCopy> window(static_cast<size_t>(n_src));
        std::vector<const VSFrame *> sources;

        const auto set_error =
            [&](const std::string & error_message) -> const VSFrame * {
            vsfeel_trace_error(d->name, n, error_message, d->gpu.get());
            release_r0_slots(d, fr);
            for (const VSFrame * f : sources) {
                vsapi->freeFrame(f);
            }
            vsapi->setFilterError(
                (std::string(d->name) + ": " + error_message).c_str(),
                frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        // Fault injection for the error-path test: fail after the private
        // slots exist but before anything was recorded, so the test exercises
        // the release path.
        if (n == d->fault_frame) {
            return set_error("injected estimation fault");
        }

        // Copy only the frames whose private slots the acquire reserved. The
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
            if (!src) {
                return set_error("clip frame " + std::to_string(f) +
                                 " could not be fetched");
            }
            for (int plane = 0; plane < d->num_planes; ++plane) {
                VSVulkanPlaneInfo plane_info {};
                if (d->gpu->api->getGPUPlane(src, plane, &plane_info)) {
                    vsapi->freeFrame(src);
                    return set_error("clip " + std::to_string(f) + " plane " +
                                     std::to_string(plane) +
                                     " is not GPU resident");
                }
                if (auto e = vsfeel_plane_range_error(plane_info, *d->gpu,
                                                      "the source plane");
                    !e.empty()) {
                    vsapi->freeFrame(src);
                    return set_error(e);
                }
                window[k].source[plane] = plane_info.buffer;
                window[k].src_range[plane] =
                    bm3d_range(d->vi->format, plane,
                               bm3d_full_range(src, d->vi->format, vsapi));
            }
            sources.push_back(src);
            if (d->final) {
                const VSFrame * rsrc =
                    vsapi->getFrameFilter(f, d->ref_node, frameCtx);
                if (!rsrc) {
                    return set_error("ref clip frame " + std::to_string(f) +
                                     " could not be fetched");
                }
                for (int plane = 0; plane < d->num_planes; ++plane) {
                    VSVulkanPlaneInfo plane_info {};
                    if (d->gpu->api->getGPUPlane(rsrc, plane, &plane_info)) {
                        vsapi->freeFrame(rsrc);
                        return set_error("ref clip " + std::to_string(f) +
                                         " plane " + std::to_string(plane) +
                                         " is not GPU resident");
                    }
                    if (auto e = vsfeel_plane_range_error(
                            plane_info, *d->gpu, "the reference plane");
                        !e.empty()) {
                        vsapi->freeFrame(rsrc);
                        return set_error(e);
                    }
                    window[k].ref[plane] = plane_info.buffer;
                    window[k].ref_range[plane] =
                        bm3d_range(d->vi->format, plane,
                                   bm3d_full_range(rsrc, d->vi->format, vsapi));
                }
                sources.push_back(rsrc);
            }
        }
        auto c_t2 = d->host_timing ? cpu_now() : 0ull;
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
        // The estimation kernel's destination binding is unused unless it is
        // tracing (which binds the trace buffer instead), but an unused binding
        // still may not be null: chroma-only processing leaves plane 0
        // unprocessed, so bind any processed plane's output there.
        VkBuffer est_dst = VK_NULL_HANDLE;
        for (int plane = 0; plane < d->num_planes && !est_dst; ++plane) {
            est_dst = dst_planes[plane];
        }

        int max_pos = 0;
        for (int gi = 0; gi < d->n_groups; ++gi) {
            collect_est_positions(d, fr.g[gi]);
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
                for (int i = 0; i < d->max_tw; ++i) {
                    if (gf.win_valid[i] && !gf.win_recompute[i]) {
                        fprintf(
                            stderr,
                            "[t] n=%d entry %d reads res slot %d for centre "
                            "%d (writer w=%d)\n",
                            n, gi, gf.win_slots[i], n - d->max_radius + i,
                            gf.win_writer[i]);
                    }
                }
            }
        }

        // The estimation is submitted first, so the GPU stays busy with the
        // heavy kernels.
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
            record_est_chunk(d, ctx, fr, n, c, chunks, window, est_dst,
                             gputrace);
            uint64_t signaled = 0;
            if (d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf,
                                           sizeof(errbuf))) {
                return set_error("estimation submit failed: "s + errbuf);
            }
        }
        for (const VSFrame * f : sources) {
            vsapi->freeFrame(f);
        }
        sources.clear();
        auto c_t3 = d->host_timing ? cpu_now() : 0ull;
        auto t3 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};

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
        GpuBuffer opp;
        if (d->rgb) {
            if (auto e = gpu_frame_buffer(*d->gpu, core, agg_ctx,
                                          3 * d->opp_pe * 4, opp);
                !e.empty()) {
                d->gpu->api->gpuExecAbandon(agg_ctx);
                return set_error("the opponent output scratch could not be "
                                 "allocated: " +
                                 e);
            }
        }
        record_bm3d_agg(d, &fr, d->gpu->api->gpuExecCommandBuffer(agg_ctx), n,
                        0, false, dst_planes, out_range, gputrace, opp);
        uint64_t agg_value = 0;
        if (d->gpu->api->gpuExecSubmit(agg_ctx, &agg_value, aerr,
                                       sizeof(aerr))) {
            return set_error("aggregation submit failed: "s + aerr);
        }
        auto c_t4 = d->host_timing ? cpu_now() : 0ull;
        auto t4 = d->host_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        if (d->trace) {
            fprintf(stderr, "[t] n=%d submitted (%d est chunks)\n", n, chunks);
        }

        if (d->trace_frame >= 0 && n == d->trace_frame) {
            // The trace rode on this frame's estimation submission, so waiting
            // the aggregation out leaves it final. Debug only: this is the one
            // blocking readback, and only for the traced frame.
            char terr[256] {};
            if (d->gpu->api->gpuExecWaitValue(d->exec, agg_value, terr,
                                              sizeof(terr)) == gdDrained) {
                dump_match_trace(d, d->trace_x, d->trace_y);
            }
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

        // Private slots go back as soon as the aggregation is queued.
        release_r0_slots(d, fr);
        auto c_t5 = d->host_timing ? cpu_now() : 0ull;
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
            d->ht_cpu_acquire += c_t1 - c0;
            d->ht_cpu_source += c_t2 - c_t1;
            d->ht_cpu_est += c_t3 - c_t2;
            d->ht_cpu_agg += c_t4 - c_t3;
            d->ht_cpu_release += c_t5 - c_t4;
            d->ht_cpu_total += c_t5 - c0;
            d->ht_n.fetch_add(1, std::memory_order_relaxed);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC BM3DFree(void * instanceData, VSCore * core,
                           const VSAPI * vsapi) {
    BM3DData * d = static_cast<BM3DData *>(instanceData);
    // Drain first: the profile reads timestamp queries, which are only valid
    // once every submission that wrote them has completed.
    d->drain();
    d->profile.report(*vsapi, core);
    for (auto & [frame, node] : d->emitted) {
        (void)frame;
        vsapi->freeFrame(node);
    }
    d->emitted.clear();
    vsapi->freeNode(d->node);
    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    delete d;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC BM3DCreate(const VSMap * in, VSMap * out, void * userData,
                             VSCore * core, const VSAPI * vsapi) {

    auto d { std::make_unique<BM3DData>() };

    d->name = userData ? static_cast<const char *>(userData) : "BM3D";

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
    // Debug oracle: VSFEEL_BM3D_MATCHTRACE=<frame>,<x>,<y> records the group
    // the matcher selects for one reference block of one centre frame.

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error(d->name, -1, error_message, d->gpu.get());
        vsapi->mapSetError(
            out, (std::string(d->name) + ": " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    if (!vsh::isConstantVideoFormat(d->vi)) {
        return set_error("only constant format clips are supported.");
    }

    if (const char * spec = env_str("VSFEEL_BM3D_MATCHTRACE")) {
        // <frame>,<x>,<y>, parsed with strtol so a malformed value is an error
        // rather than a silently truncated one.
        std::array<int, 3> vals {};
        const char * p = spec;
        bool ok = true;
        for (size_t i = 0; i < vals.size() && ok; ++i) {
            char * end = nullptr;
            const long v = std::strtol(p, &end, 10);
            ok = end != p && v >= 0 && v <= INT32_MAX;
            vals[i] = static_cast<int>(v);
            const bool last = i + 1 == vals.size();
            ok = ok && *end == (last ? '\0' : ',');
            p = end + 1;
        }
        if (!ok) {
            return set_error("VSFEEL_BM3D_MATCHTRACE must be "
                             "<frame>,<x>,<y>.");
        }
        d->trace_frame = vals[0];
        d->trace_x = vals[1];
        d->trace_y = vals[2];
    }

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
            return set_error("\"ref\" must be of the same format as \"clip\".");
        }
        if (rvi->width != d->vi->width || rvi->height != d->vi->height) {
            return set_error(
                "\"ref\" must be of the same dimensions as \"clip\".");
        }
        if (rvi->numFrames != d->vi->numFrames) {
            return set_error(
                "\"ref\" must be of the same number of frames as \"clip\".");
        }
        d->final = true;
    }

    if (d->vi->width <= 0 || d->vi->height <= 0 ||
        !((d->vi->format.sampleType == stFloat &&
           (d->vi->format.bitsPerSample == 32 ||
            d->vi->format.bitsPerSample == 16)) ||
          (d->vi->format.sampleType == stInteger &&
           d->vi->format.bitsPerSample == 16))) {
        return set_error(
            "only constant format 16-bit integer (u16), 16-bit float (f16) or "
            "32-bit float (f32) input supported.");
    }
    d->bits = d->vi->format.bitsPerSample;
    d->half = d->vi->format.sampleType == stFloat && d->bits == 16;
    d->elem_bytes = d->bits / 8;

    if (d->vi->format.colorFamily != cfGray &&
        d->vi->format.colorFamily != cfYUV &&
        d->vi->format.colorFamily != cfRGB) {
        return set_error("only Gray, YUV and RGB input are supported.");
    }
    // RGB is filtered in the reference's opponent space: the ring conversion
    // and the output conversion below are that space's transport.
    d->rgb = d->vi->format.colorFamily == cfRGB;
    d->num_planes = d->vi->format.numPlanes;

    // The block-matching kernel always loads unconditional 8x8 patches and
    // clamps its block origin to (dimension - 8), so anything smaller would
    // index negative buffer offsets. Reject it before allocating GPU
    // resources rather than relying on robust buffer access (which is not
    // enabled).
    if (d->vi->width < 8 || d->vi->height < 8) {
        return set_error("clip dimensions must be at least 8x8.");
    }

    // The scan lists pack a candidate's (x, y) into one 32-bit word (16 + 15
    // bits) so the top-8 insert shifts three arrays instead of four.
    if (d->vi->width > 0xFFFF || d->vi->height > 0x7FFF) {
        return set_error("clip dimensions must not exceed 65535x32767.");
    }

    // The window clamps and the cache keys assume [0, numFrames-1]; an empty
    // or unknown length would index the slot tables before their base.
    if (d->vi->numFrames <= 0) {
        return set_error("clip frame count must be known and positive.");
    }

    // The reference's color matrix: th_mse is scaled by its luma-row norm and
    // each plane's sigma by that plane's own row norm. `matrix` takes the CPU
    // plugin's enum; 2 (unspecified) follows the reference's resolution rule
    // (SD smpte170m, HD bt709, UHD bt2020nc). 100 is the opponent matrix,
    // forced for RGB input and available for a clip converted with RGB2OPP.
    std::array<double, 3> matrix_norm {};
    // Every accepted matrix sets it; the switch's default returns.
    const char * matrix_name;
    {
        int matrix =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "matrix", 0, &error));
        if (d->rgb) {
            // The references force the opponent space for RGB, whatever
            // `matrix` says.
            matrix = 100;
        } else if (error || matrix == 2) {
            const bool uhd = d->vi->width > 2048 || d->vi->height > 1536;
            const bool hd = d->vi->width > 1024 || d->vi->height > 576;
            matrix = uhd ? 9 : (hd ? 1 : 6);
        }
        double yr {}, yg {}, yb {}, ur {}, ug {}, ub {}, vr {}, vg {}, vb {};
        if (matrix == 0) {
            // GBR: Y=G, U=B, V=R.
            matrix_name = "gbr";
            yg = 1.0;
            ub = 1.0;
            vr = 1.0;
        } else if (matrix == 100) {
            matrix_name = "opp";
            yr = yg = yb = 1.0 / 3.0;
            ur = 0.5;
            ub = -0.5;
            vr = 0.25;
            vg = -0.5;
            vb = 0.25;
        } else {
            double kr {}, kg {}, kb {};
            switch (matrix) {
            case 1:
                matrix_name = "bt709";
                kr = 0.2126;
                kg = 0.7152;
                kb = 0.0722;
                break;
            case 4:
                matrix_name = "fcc";
                kr = 0.30;
                kg = 0.59;
                kb = 0.11;
                break;
            case 5:
                matrix_name = "bt470bg";
                kr = 0.299;
                kg = 0.587;
                kb = 0.114;
                break;
            case 6:
                matrix_name = "smpte170m";
                kr = 0.299;
                kg = 0.587;
                kb = 0.114;
                break;
            case 7:
                matrix_name = "smpte240m";
                kr = 0.212;
                kg = 0.701;
                kb = 0.087;
                break;
            case 9:
                matrix_name = "bt2020nc";
                kr = 0.2627;
                kg = 0.6780;
                kb = 0.0593;
                break;
            case 10:
                matrix_name = "bt2020c";
                kr = 0.2627;
                kg = 0.6780;
                kb = 0.0593;
                break;
            default:
                return set_error(
                    "\"matrix\" must be 0 (GBR), 1 (bt709), 2 (auto), 4 "
                    "(fcc), 5 (bt470bg), 6 (smpte170m), 7 (smpte240m), 9 "
                    "(bt2020nc), 10 (bt2020c) or 100 (opponent).");
            }
            yr = kr;
            yg = kg;
            yb = kb;
            ur = -yr * 0.5 / (1.0 - yb);
            ug = -yg * 0.5 / (1.0 - yb);
            ub = 0.5;
            vr = 0.5;
            vg = -yg * 0.5 / (1.0 - yr);
            vb = -yb * 0.5 / (1.0 - yr);
        }
        // The rows are summed explicitly: GCC 16 miscompiles the same three
        // scalar sqrt expressions at -O3 (the U term is dropped).
        const double rows[3][3] = { { yr, yg, yb },
                                    { ur, ug, ub },
                                    { vr, vg, vb } };
        for (int i = 0; i < 3; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 3; ++j) {
                sum += rows[i][j] * rows[i][j];
            }
            matrix_norm[i] = std::sqrt(sum);
        }
    }

    // Per-plane sigma, exactly as the reference's perPlane: a missing entry
    // repeats the previous one, the first defaults to 3. A plane whose sigma is
    // below FLT_EPSILON is a bit-exact source copy (the reference's PROC_MASK,
    // and its kernel's epsilon test). With every plane below it BM3D is the
    // source clip itself, which is the all-zero shortcut both references take.
    std::array<float, 3> sigma {};
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] =
            static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &error));
        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (!std::isfinite(sigma[i]) || sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be finite and non-negative.");
        }
    }
    bool any_process = false;
    for (int i = 0; i < d->num_planes; ++i) {
        // The reference processes all three RGB planes in OPP, even at sigma 0.
        d->planes[i].process = d->rgb || sigma[i] >= FLT_EPSILON;
        any_process = any_process || d->planes[i].process;
    }
    // th_mse: the CPU matcher's threshold in 8-bit MSE units, derived from the
    // shared sigma[0] when omitted (mawen's fast profile). It is validated
    // before the all-zero-sigma shortcut so an invalid value is never
    // conditionally accepted, and converted to the shader's SSD domain once:
    // MSE_8bit = SSD * 255^2 / 64.
    //
    // The reference scales both its threshold and its per-plane sigmas by the
    // norms of its color matrix rows; th_mse carries the luma-row norm.
    {
        double th_mse = vsapi->mapGetFloat(in, "th_mse", 0, &error);
        if (error) {
            th_mse = d->final ? (200.0 + sigma[0] * 10.0)
                              : (400.0 + sigma[0] * 80.0);
        } else if (!std::isfinite(th_mse) || th_mse < 0.0) {
            return set_error(
                "\"th_mse\" must be a finite non-negative number of 8-bit MSE "
                "units (0 selects the reference block alone).");
        }
        double th_sse = th_mse * matrix_norm[0] * (64.0 / (255.0 * 255.0));
        if (th_sse > static_cast<double>(FLT_MAX)) {
            // A finite but enormous threshold accepts every finite distance; a
            // saturated float keeps that meaning instead of narrowing to inf.
            th_sse = static_cast<double>(FLT_MAX);
        }
        d->th_mse = static_cast<float>(th_mse);
        d->th_sse = static_cast<float>(th_sse);
        if (th_mse > 0.0 && d->th_sse == 0.0f) {
            // Underflowed to zero, which the kernel reads as "reference only";
            // the smallest denormal keeps the search on, as the CPU's positive
            // threshold does.
            d->th_sse = FLT_TRUE_MIN;
        }
    }

    // extractor_exp: validated with the other scalars, before the all-zero
    // shortcut. bm3dvk (the preferred reference where the two disagree)
    // accepts [0, 127]: a negative exponent pre-rounds the addends below any
    // representable value and 2^128 is already infinite, which turns the
    // aggregation into NaN.
    {
        const int extractor_exp =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "extractor_exp", 0, &error));
        if (extractor_exp < 0 || extractor_exp > 127) {
            return set_error("\"extractor_exp\" must be in range [0, 127].");
        }
        d->extractor =
            (extractor_exp != 0) ? std::ldexp(1.0f, extractor_exp) : 0.0f;
    }

    // Match the reference's sigma scaling exactly: each plane's sigma carries
    // its own color-matrix row norm, and the final Wiener pass uses 1x instead
    // of the basic stage's 2.7x. A skipped plane keeps exactly zero so the
    // kernel's epsilon test cannot disagree with the unscaled decision above.
    const double sigma_scale = d->final ? 1.0 : 2.7;
    for (int i = 0; i < d->num_planes; ++i) {
        const float sigma_factor =
            static_cast<float>(sigma_scale * matrix_norm[i] * 64.0 / 255.0);
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
            return set_error("\"block_step\" must be in range [1, 8].");
        }
    }

    std::array<int, 3> bm_range {};
    for (int i = 0; i < std::ssize(bm_range); ++i) {
        bm_range[i] =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "bm_range", i, &error));
        if (error) {
            bm_range[i] = (i == 0) ? 9 : bm_range[i - 1];
        } else if (bm_range[i] <= 0 || bm_range[i] > kMaxSearchRange) {
            return set_error("\"bm_range\" must be in range [1, "s +
                             std::to_string(kMaxSearchRange) + "].");
        }
    }

    // Per-plane temporal radius, exactly like sigma: a missing entry repeats
    // the previous one and the first defaults to 0 (spatial only). The widest
    // entry's window drives the frame requests and the shared cache sizing;
    // each entry's own radius drives its kernel and scatter range.
    std::array<int, 3> radius {};
    for (int i = 0; i < std::ssize(radius); ++i) {
        radius[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", i, &error));
        if (error) {
            radius[i] = (i == 0) ? 0 : radius[i - 1];
        } else if (radius[i] < 0 || radius[i] > MAX_RADIUS) {
            return set_error("\"radius\" must be in range [0, "s +
                             std::to_string(MAX_RADIUS) + "].");
        }
    }
    for (int i = 0; i < d->num_planes; ++i) {
        d->planes[i].radius = radius[i];
    }

    std::array<int, 3> ps_num {};
    for (int i = 0; i < std::ssize(ps_num); ++i) {
        ps_num[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_num", i, &error));
        if (error) {
            ps_num[i] = (i == 0) ? 2 : ps_num[i - 1];
        } else if (ps_num[i] <= 0 || ps_num[i] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8].");
        }
    }

    std::array<int, 3> ps_range {};
    for (int i = 0; i < std::ssize(ps_range); ++i) {
        ps_range[i] =
            vsh::int64ToIntS(vsapi->mapGetInt(in, "ps_range", i, &error));
        if (error) {
            ps_range[i] = (i == 0) ? 4 : ps_range[i - 1];
        } else if (ps_range[i] <= 0 || ps_range[i] > kMaxSearchRange) {
            return set_error("\"ps_range\" must be in range [1, "s +
                             std::to_string(kMaxSearchRange) + "].");
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
        return set_error("clip format must be YUV444 when \"chroma\" is true.");
    }

    // Nothing to denoise: hand the source clip straight back, as the
    // references' BM3Dv2 does. This is after every argument has been parsed and
    // validated, so an invalid block_step/radius/chroma is still an error when
    // all sigmas are zero; it is before any GPU resource exists, so the
    // instance is just the node references.
    if (!any_process) {
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
            d->ref_node = nullptr;
        }
        vsapi->mapConsumeNode(out, "clip", d->node, maReplace);
        return;
    }

    // device_id and num_streams are registered but never read: the core owns
    // the one device and sizes in-flight depth itself (exec pool ring).

    // Radius 0 gives every in-flight frame its own src/res slot so concurrent
    // frames never serialize on a one-deep ring. Radius > 0 uses the temporal
    // cursor and its separate ring sizes.
    int inflight =
        std::clamp(env_int("VSFEEL_BM3D_INFLIGHT", kInflightFrames), 1, 8);

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

    // The estimation kernel accumulates into float SSBOs with atomicAdd, which
    // needs shaderBufferFloat32AtomicAdd (the base extension's load/store/
    // exchange bit alone is not enough); anything else runs the reference's
    // atom_add_f CAS loop. VSFEEL_BM3D_CAS forces the CAS build.
    d->cas_atomics =
        env_flag("VSFEEL_BM3D_CAS") || !d->gpu->feat_atomic_float32_add;
    // Where the group's reference patch lives is a codegen decision, not a
    // register-file or cache one: the 64 values are sixteen vec4 locals (a
    // private array is what a backend that will not promote it stages in
    // scratch, and then every SSD read per candidate is a memory access). A
    // backend that keeps them in registers (RADV/ACO: 168 VGPRs, zero scratch,
    // the same count the vectors get) pays nothing, and shared memory is 8-10%
    // worse there; a backend that does not loses ~3x without it. Every compiler
    // except the one measured to keep them gets shared memory.
    // VSFEEL_BM3D_PATCH_LDS=0/1 overrides either way.
    const char * patch_lds_env = std::getenv("VSFEEL_BM3D_PATCH_LDS");
    d->patch_lds = patch_lds_env && *patch_lds_env
                       ? env_flag("VSFEEL_BM3D_PATCH_LDS")
                       : d->gpu->driver_id != VK_DRIVER_ID_MESA_RADV;
    // The split runs the search and the estimate as two dispatches, with the
    // group handed over through a buffer. The search then carries no estimate
    // stack and the filter no search state, which is where the search's
    // occupancy comes from: 120 -> 96 VGPRs and 12 -> 16 waves/SIMD. A 4070
    // reads that as +8.8%, RADV as -13% (at 12 waves/SIMD it was never
    // occupancy-bound), so it follows the same driver rule as the patch above.
    // VSFEEL_BM3D_SPLIT_KERNEL=0/1 overrides.
    const char * split_env = std::getenv("VSFEEL_BM3D_SPLIT_KERNEL");
    d->split_kernel = split_env && *split_env
                          ? env_flag("VSFEEL_BM3D_SPLIT_KERNEL")
                          : d->gpu->driver_id != VK_DRIVER_ID_MESA_RADV;
    // One launch shape for every pipeline of this instance: the split's
    // handoff and the per-plane grid are indexed by it.
    const auto subgroup = choose_subgroup(*d->gpu);
    if (std::holds_alternative<std::string>(subgroup)) {
        return set_error(std::get<std::string>(subgroup));
    }
    d->subgroup_size = std::get<uint32_t>(subgroup);
    d->workgroup = choose_workgroup(*d->gpu, d->subgroup_size, d->patch_lds);
    if (d->workgroup % 8 != 0) {
        return set_error("VSFEEL_BM3D_WORKGROUP must be a multiple of 8.");
    }
    d->groups_per_wg = d->workgroup / 8;
    // The normalization tile and the copy/OPP element blocks come from the
    // device: 16x16 and 256 where it allows them, 16x8 and 128 on the
    // 128-invocation floor. The grid divisors below and in the frame path are
    // computed from the same two shapes the shaders launch with.
    d->agg_wg = vsfeel_fit_workgroup(d->gpu->limits, 16, 16);
    d->copy_wg = vsfeel_fit_workgroup(d->gpu->limits, 256).x;
    if (vsfeel_device_info_enabled()) {
        fprintf(stderr, "[bm3d] aggregation: %s\n",
                d->cas_atomics ? "CAS loop (atom_add_f fallback)"
                               : "hardware buffer float atomics");
        fprintf(stderr, "[bm3d] estimation: %s\n",
                d->split_kernel ? "match + filter split"
                                : "fused match/estimate");
        fprintf(stderr, "[bm3d] reference patch: %s (driver %u)\n",
                d->patch_lds ? "shared memory" : "per-lane registers",
                d->gpu->driver_id);
        fprintf(stderr, "[bm3d] workgroup: %u threads, %u groups\n",
                d->workgroup, d->groups_per_wg);
        fprintf(stderr, "[bm3d] shapes: normalize %ux%u, copies %u\n",
                d->agg_wg.x, d->agg_wg.y, d->copy_wg);
        fprintf(stderr,
                "[bm3d] color matrix: %s (norms y=%.6f u=%.6f v=%.6f)\n",
                matrix_name, matrix_norm[0], matrix_norm[1], matrix_norm[2]);
    }
    // The 8x8 group transposes and the group-8 reduction are subgroup shuffles,
    // and the kernel's per-lane layout puts each 8-lane group inside one
    // subgroup, so a device without the shuffle intrinsics cannot run it. BASIC
    // is the only operation Vulkan mandates, so this is a real check.
    if (!d->gpu->has_subgroup_ops(VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) {
        return set_error("subgroup shuffle is not supported by this device "
                         "(VK_SUBGROUP_FEATURE_SHUFFLE_BIT is required).");
    }

    VkDevice dev = d->gpu->device;

    {
        // Push descriptors: the bindings change every frame (the destination
        // plane is the frame's own storage), so nothing is allocated from a
        // pool and nothing survives the command buffer.
        const auto result = gpu_push_set_layout(*d->gpu, 4);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(result);
    }
    {
        // The largest block is the aggregation's eight ints (the OPP output
        // base included); the estimation kernel declares the first seven.
        const auto result =
            gpu_pipeline_layout(*d->gpu, d->set_layout, 8 * sizeof(int32_t));
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(result);
    }

    // Per-plane geometry: plane 0 at full size, the chroma planes subsampled by
    // the format (no subsampling for 4:4:4 and RGB).
    const int subW = d->vi->format.subSamplingW;
    const int subH = d->vi->format.subSamplingH;
    for (int plane = 0; plane < d->num_planes; ++plane) {
        auto & p = d->planes[plane];
        p.width = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        p.height = (plane == 0) ? d->vi->height : d->vi->height >> subH;
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

    // The split's handoff is one flat word array that both halves index with
    // 32-bit words, so its size is checked from the geometry alone before the
    // probe frame below: a truncated word count would let the matcher write
    // past the allocation.
    if (d->split_kernel) {
        const uint64_t groups = d->groups_per_wg;
        const auto handoff_words = [&](int plane) {
            const auto & p = d->planes[plane];
            const uint64_t step = static_cast<uint64_t>(p.block_step);
            const uint64_t gx =
                (static_cast<uint64_t>(p.width) + groups * step - 1) /
                (groups * step);
            const uint64_t gy =
                (static_cast<uint64_t>(p.height) + step - 1) / step;
            const uint64_t blocks = gx * gy * groups;
            return ((blocks + 1) & ~uint64_t(1)) + blocks * 16;
        };
        uint64_t words = 0;
        int widest = -1;
        const auto consider = [&](int plane) {
            const uint64_t w = handoff_words(plane);
            if (w > words) {
                words = w;
                widest = plane;
            }
        };
        if (d->joint) {
            // A joint entry's groups come from luma, processed or not.
            consider(0);
        } else {
            for (int plane = 0; plane < d->num_planes; ++plane) {
                if (d->planes[plane].process) {
                    consider(plane);
                }
            }
        }
        // The shader's highest word is MATCH_MEM_OFF + bid * 16 + 15, which
        // is the last word of the buffer.
        if (words > uint64_t(UINT32_MAX) + 1) {
            const auto & p = d->planes[widest];
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "the split match handoff needs %llu words for a %dx%d "
                     "plane at block_step %d, past the kernel's 32-bit word "
                     "indexing; increase block_step or lower the resolution",
                     static_cast<unsigned long long>(words), p.width, p.height,
                     p.block_step);
            return set_error(msg);
        }
        if (words * 4 > d->gpu->limits.maxStorageBufferRange) {
            return set_error(
                "the split match handoff (" + std::to_string(words * 4) +
                " bytes) exceeds this device's maxStorageBufferRange (" +
                std::to_string(d->gpu->limits.maxStorageBufferRange) +
                " bytes); increase block_step or lower the resolution");
        }
        d->match_words = words;
    }

    // The plane layout has to be the one the core's GPU frames carry, because
    // a source frame's plane is copied into the ring with a single flat
    // buffer-to-buffer region (see vsfeel_probe_strides).
    std::array<int, 3> plane_stride_elems {};
    if (auto e = vsfeel_probe_strides(vsapi, core, d->vi->format, d->vi->width,
                                      d->elem_bytes, plane_stride_elems.data());
        !e.empty()) {
        return set_error(e);
    }
    for (int plane = 0; plane < d->num_planes; ++plane) {
        auto & p = d->planes[plane];
        p.stride = plane_stride_elems[plane];
        p.pe = static_cast<VkDeviceSize>(p.stride) * p.height;
        if (p.stride < p.width) {
            return set_error("the core reported an unexpected plane stride");
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
                return set_error("\"chroma\" needs a single row stride across "
                                 "the three planes.");
            }
        }
        auto & g = d->groups[d->n_groups++];
        g.n_planes = d->num_planes;
        for (int i = 0; i < d->num_planes; ++i) {
            g.planes[i] = i;
        }
        g.pe = d->planes[0].pe;
        // A joint entry shares one window, so run it at the widest processed
        // plane rather than narrowing any plane's requested radius.
        for (int i = 0; i < d->num_planes; ++i) {
            if (d->planes[i].process) {
                g.radius = std::max(g.radius, d->planes[i].radius);
            }
        }
    } else {
        for (int plane = 0; plane < d->num_planes; ++plane) {
            if (!d->planes[plane].process) {
                continue;
            }
            auto & g = d->groups[d->n_groups++];
            g.n_planes = 1;
            g.planes[0] = plane;
            g.pe = d->planes[plane].pe;
            g.radius = d->planes[plane].radius;
        }
    }

    // One window drives the frame requests, the shared floor for every
    // entry's accumulator ring and the union source copy. An entry with a
    // smaller radius still uses that floor so its output frame survives until
    // every plane is ready to normalize; its own kernel only touches its own
    // window.
    for (int gi = 0; gi < d->n_groups; ++gi) {
        d->max_radius = std::max(d->max_radius, d->groups[gi].radius);
    }
    d->max_tw = 2 * d->max_radius + 1;

    // The CAS fallback (src/bm3d.comp) bounds its compare-exchange loop with a
    // 32-bit counter. In temporal mode one accumulator element is written by
    // 2*radius+1 centres, each scattering 8 patches from every reference block
    // whose window reaches it, so the exact bound is
    // 8*(2*radius+1)*cells^2 with
    // cells = ceil((2*bm_range + 8)/block_step) + 1. Reject configs whose bound
    // would overflow the counter; the shader keeps a 32-bit loop, so this needs
    // no shaderInt64 feature or extra variant.
    uint64_t cas_retries = 0;
    int cas_gi = -1;
    for (int gi = 0; gi < d->n_groups; ++gi) {
        const auto & g = d->groups[gi];
        const auto & first = d->planes[g.planes[0]];
        const uint64_t step = static_cast<uint64_t>(first.block_step);
        const uint64_t range = static_cast<uint64_t>(first.bm_range);
        const uint64_t cells = (2ull * range + 8 + step - 1) / step + 1;
        const uint64_t temporal = 2ull * g.radius + 1;
        const uint64_t bound = 8ull * temporal * cells * cells;
        if (bound > cas_retries) {
            cas_retries = bound;
            cas_gi = gi;
        }
    }
    if (cas_retries > UINT32_MAX) {
        const auto & g = d->groups[cas_gi];
        const auto & first = d->planes[g.planes[0]];
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "the temporal CAS retry bound (%llu) overflows the kernel's "
                 "32-bit loop counter (bm_range %d, block_step %d, radius %d); "
                 "lower bm_range, raise block_step or lower radius",
                 static_cast<unsigned long long>(cas_retries), first.bm_range,
                 first.block_step, g.radius);
        return set_error(msg);
    }

    // The temporal cursor copies each frame into the ring once as the centre
    // advances, so the source ring only has to hold one search window plus the
    // frame the next centre is about to add. Radius 0 keeps a private slot per
    // in-flight frame.
    const auto src_ring_of = [r = d->max_radius](int n) {
        return (r == 0) ? n : 2 * r + 3;
    };
    // Accumulators are keyed by output frame: the ring holds the 2r+1 frames
    // whose contributions are in flight. Radius 0's slots are private to one
    // frame. The ring is exactly one window wide: a slot is reassigned only
    // after the frame it held has been normalized by the previous request.
    const auto res_cap_of = [r = d->max_radius, tw = d->max_tw,
                             nf = d->vi->numFrames](int n) {
        return (r == 0) ? n : std::min(tw, nf);
    };
    int src_ring = src_ring_of(inflight);
    int res_cap = res_cap_of(inflight);

    // The ring copy's 1D grid, folded into X and Y: a 4096x4096 plane alone
    // needs 65536 workgroups, one past the X limit Vulkan only guarantees.
    // Every copy of an entry uses its pe, so one folded grid per entry covers
    // both the source and the ref ring sections.
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        std::string err;
        if (!vsfeel_fold_grid(
                d->gpu->limits,
                (static_cast<uint64_t>(g.pe) + d->copy_wg - 1) / d->copy_wg,
                "bm3d's ring copy", g.copy_grid_x, g.copy_grid_y, err)) {
            return set_error(err);
        }
    }

    VkDeviceSize src_total = 0;
    VkDeviceSize res_total = 0;
    // One entry's buffers, sized from the current pipeline depth and checked
    // against a device's maxStorageBufferRange, which can bind before VRAM does
    // at the top of the radius range (1080p radius 7 on a 4 GiB-range card).
    const auto size_groups = [&]() {
        bool fits = true;
        src_total = 0;
        res_total = 0;
        for (int gi = 0; gi < d->n_groups; ++gi) {
            auto & g = d->groups[gi];
            g.src_ring = src_ring;
            g.res_cap = res_cap;
            // In final mode each ring slot holds [ref][source], so the ring
            // doubles.
            const int clips = d->final ? 2 : 1;
            g.src_size = static_cast<VkDeviceSize>(g.src_ring) * clips *
                         g.n_planes * g.pe;
            g.res_size =
                static_cast<VkDeviceSize>(g.res_cap) * 2 * g.n_planes * g.pe;
            fits = fits &&
                   g.res_size * 4 <= d->gpu->limits.maxStorageBufferRange &&
                   g.src_size * 4 <= d->gpu->limits.maxStorageBufferRange;
            src_total += g.src_size;
            res_total += g.res_size;
        }
        return fits;
    };
    // Radius 0's rings scale with the in-flight depth, so the depth can give
    // way: two frames are the floor the recording pipeline needs to overlap
    // with execution at all, and the deeper default is worth having only while
    // it fits. A temporal ring is fixed at one window (shrinking either ring
    // would break the cursor), so it has nothing to give way; the budget check
    // below rejects it instead.
    bool buffers_fit = false;
    if (d->max_radius == 0) {
        while (!(buffers_fit = size_groups()) && inflight > 2) {
            --inflight;
            src_ring = src_ring_of(inflight);
            res_cap = res_cap_of(inflight);
        }
    } else {
        buffers_fit = size_groups();
    }
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
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
                static_cast<unsigned long long>(g.res_size), d->max_radius);
            return set_error(msg);
        }
        if (g.src_size > static_cast<VkDeviceSize>(INT32_MAX)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "frame is too large: the source ring needs %llu samples "
                     "(radius %d), which overflows the 32-bit kernel "
                     "addressing; reduce radius",
                     static_cast<unsigned long long>(g.src_size),
                     d->max_radius);
            return set_error(msg);
        }
    }

    // Persistent VRAM: the rings just sized, the one-slot isolated accumulator,
    // the split handoff and the finished-frame cache. The isolated source ring
    // is outside this on purpose: it is allocated on the first seek and a
    // linear playback never pays it. A quarter of the core's allowance is the
    // budget, which leaves room for that duplicate ring plus the frame being
    // normalized. An unknown allowance (0) skips the check, and a buffer past
    // maxStorageBufferRange is left to the allocation's own error, which names
    // the device limit rather than the budget.
    VkDeviceSize frame_bytes = 0;
    for (int p = 0; p < d->num_planes; ++p) {
        if (d->planes[p].process) {
            frame_bytes += d->planes[p].pe * d->elem_bytes;
        }
    }
    VkDeviceSize ring_bytes = 0;
    for (int gi = 0; gi < d->n_groups; ++gi) {
        const auto & g = d->groups[gi];
        ring_bytes += (g.src_size + g.res_size) * 4;
        if (d->max_radius > 0) {
            ring_bytes += static_cast<VkDeviceSize>(2) * g.n_planes * g.pe * 4;
        }
    }
    if (d->split_kernel) {
        ring_bytes += static_cast<VkDeviceSize>(d->match_words) * 4;
    }
    const VkDeviceSize vram = vsfeel_vram_limit(*d->gpu, core);
    const VkDeviceSize budget = vram / 4;
    if (budget > 0 && d->max_radius > 0 && buffers_fit) {
        const double mib = 1024.0 * 1024.0;
        if (ring_bytes > budget) {
            char msg[320];
            snprintf(
                msg, sizeof(msg),
                "the temporal rings need %.0f MiB (radius %d, %d planes, "
                "%dx%d) but the VRAM allowance is %.0f MiB and BM3D keeps "
                "them within a quarter of it; lower radius or use a smaller "
                "clip",
                static_cast<double>(ring_bytes) / mib, d->max_radius,
                d->num_planes, d->vi->width, d->vi->height,
                static_cast<double>(vram) / mib);
            return set_error(msg);
        }
        const VkDeviceSize room = budget - ring_bytes;
        const VkDeviceSize frames =
            room / std::max<VkDeviceSize>(frame_bytes, 1);
        d->emitted_cap = static_cast<size_t>(std::min<VkDeviceSize>(
            frames, static_cast<VkDeviceSize>(kEmittedMax)));
    }

    if (d->trace_frame >= 0) {
        if (d->trace_frame >= d->nframes) {
            return set_error(
                "VSFEEL_BM3D_MATCHTRACE frame is outside the clip");
        }
        // The kernel only visits reference-block origins: multiples of the
        // block step, plus the clamped final origin at (dimension - 8) when it
        // is not already on that grid. Snapping would silently trace a
        // different block than the one asked for, which is exactly the kind of
        // quiet substitution a matching oracle must not do.
        const int bs = d->planes[0].block_step;
        const auto is_origin = [bs](int v, int dim) {
            const int last = dim - 8;
            return v >= 0 && v <= last &&
                   (v == last || (v % bs == 0 && v < last + bs));
        };
        if (!is_origin(d->trace_x, d->planes[0].width) ||
            !is_origin(d->trace_y, d->planes[0].height)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "VSFEEL_BM3D_MATCHTRACE origin (%d, %d) is not a "
                     "reference-block position: block_step is %d and the plane "
                     "is %dx%d",
                     d->trace_x, d->trace_y, bs, d->planes[0].width,
                     d->planes[0].height);
            return set_error(msg);
        }
    }

    // The split's handoff, sized and validated from the geometry above.
    if (d->split_kernel) {
        std::string err = gpu_make_buffer(
            *d->gpu, core, static_cast<VkDeviceSize>(d->match_words) * 4,
            d->matches, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!err.empty()) {
            return set_error("the match handoff could not be allocated: " +
                             err);
        }
    }
    if (d->trace_frame >= 0) {
        // One record is the marker, the five counts that follow it, eight
        // (x, y, z, error) slots and two counts per temporal frame and
        // direction: 6 + 32 + 4 * radius words, all of which the kernel
        // writes (the last temporal word of a radius-16 trace is index 101).
        constexpr VkDeviceSize kTraceWords = 6 + 8 * 4 + 2 * 2 * MAX_RADIUS;
        std::string err =
            gpu_make_buffer(*d->gpu, core, kTraceWords * 4, d->match_trace,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!err.empty()) {
            return set_error("the match trace buffer could not be "
                             "allocated: " +
                             err);
        }
        d->match_trace_mapped =
            static_cast<volatile uint32_t *>(d->match_trace.mapped);
        if (d->match_trace_mapped) {
            for (VkDeviceSize i = 0; i < kTraceWords; ++i) {
                d->match_trace_mapped[i] = 0;
            }
        }
    }
    // The kernels read or atomically update these host-initialised words.
    store_fence();

    // Both buffers come from the core's pool, so they count against the
    // VRAM budget the frame cache and the thread pool's admission control
    // also see. The source ring only ever receives copies; the estimate
    // cache is filled and read by kernels.
    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        std::string err = gpu_make_buffer(*d->gpu, core, g.src_size * 4, g.src,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                          0, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
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
                     static_cast<double>(g.res_size) * 4.0 / (1024.0 * 1024.0),
                     d->max_radius);
            return set_error(msg);
        }
        if (d->max_radius > 0) {
            // The isolated source ring is allocated by record_isolated, on the
            // first request the cursor cannot serve, so a linear playback does
            // not reserve a duplicate of the source ring.
            const VkDeviceSize one_slot =
                static_cast<VkDeviceSize>(2) * g.n_planes * g.pe * 4;
            err = gpu_make_buffer(*d->gpu, core, one_slot, g.scratch_acc,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (!err.empty()) {
                return set_error("the isolated accumulator ring could not "
                                 "be allocated: " +
                                 err);
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
    // Per-dispatch kernel profile (VSFEEL_BM3D_PROFILE). Unlike GPUTRACE this
    // does not serialize frames: every dispatch takes a timestamp pair and the
    // whole table is read once, after the pool drains at free.
    if (vsfeel_debug_probe("VSFEEL_BM3D_PROFILE") &&
        vsfeel_probe_timestamps(*d->gpu, "BM3D")) {
        d->profile.init(*d->gpu, *vsapi, core);
    }

    // pipelines: one estimation pipeline per entry (its planes share the
    // geometry and the search parameters, only the sigmas differ) and one
    // aggregation pipeline per processed plane.
    if (d->bits == 16) {
        const auto result =
            d->half ? create_copy_pipeline(*d->gpu, d->pipeline_layout,
                                           bm3d_copy_f16_spv,
                                           bm3d_copy_f16_spv_size, d->copy_wg)
                    : create_copy_pipeline(*d->gpu, d->pipeline_layout,
                                           bm3d_copy_u16_spv,
                                           bm3d_copy_u16_spv_size, d->copy_wg);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->copy_pipeline = std::get<VkPipeline>(result);
    }

    // RGB: the input conversion widens and mixes each source frame into OPP,
    // and the output conversion mixes the three normalized OPP planes back
    // into the RGB frame. Both follow the clip's depth at the outside and f32
    // in the middle (the ring and the scratch).
    if (d->rgb) {
        const uint32_t * in_code = bm3d_opp_in_f32_spv;
        size_t in_size = bm3d_opp_in_f32_spv_size;
        const uint32_t * out_code = bm3d_opp_out_f32_spv;
        size_t out_size = bm3d_opp_out_f32_spv_size;
        if (d->bits == 16) {
            if (d->half) {
                in_code = bm3d_opp_in_f16_spv;
                in_size = bm3d_opp_in_f16_spv_size;
                out_code = bm3d_opp_out_f16_spv;
                out_size = bm3d_opp_out_f16_spv_size;
            } else {
                in_code = bm3d_opp_in_u16_spv;
                in_size = bm3d_opp_in_u16_spv_size;
                out_code = bm3d_opp_out_u16_spv;
                out_size = bm3d_opp_out_u16_spv_size;
            }
        }
        const auto in_pipe = create_opp_pipeline(
            *d->gpu, d->pipeline_layout, in_code, in_size, true, d->copy_wg);
        if (std::holds_alternative<std::string>(in_pipe)) {
            return set_error(std::get<std::string>(in_pipe));
        }
        d->opp_in_pipeline = std::get<VkPipeline>(in_pipe);
        const auto out_pipe = create_opp_pipeline(
            *d->gpu, d->pipeline_layout, out_code, out_size, false, d->copy_wg);
        if (std::holds_alternative<std::string>(out_pipe)) {
            return set_error(std::get<std::string>(out_pipe));
        }
        d->opp_out_pipeline = std::get<VkPipeline>(out_pipe);
        d->opp_pe = d->planes[0].pe;
        std::string grid_err;
        if (!vsfeel_fold_grid(
                d->gpu->limits,
                (static_cast<uint64_t>(d->opp_pe) + d->copy_wg - 1) /
                    d->copy_wg,
                "bm3d's opponent output", d->opp_grid_x, d->opp_grid_y,
                grid_err)) {
            return set_error(grid_err);
        }
    }

    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        const auto & first = d->planes[g.planes[0]];
        {
            // A one-plane entry gets the loop-free build (a one-trip loop over
            // the NPLANES spec constant still costs a backend that will not
            // hoist the body out of it about 2x), and the arm the driver rule
            // picked decides whether the reference patch is staged. See
            // src/bm3d.comp for both.
            const bool one_plane = g.n_planes == 1;
            if (d->split_kernel) {
                // The search half stages the patch but has no estimate stack;
                // the filter half is the other way round. That asymmetry is the
                // point, so the two are built from different arms.
                const uint32_t * mcode =
                    d->patch_lds ? bm3d_match_lds_spv : bm3d_match_spv;
                const size_t msize = d->patch_lds ? bm3d_match_lds_spv_size
                                                  : bm3d_match_spv_size;
                const auto mres = create_bm3d_pipeline(
                    *d->gpu, *d, g, mcode, msize, d->pipeline_layout,
                    d->patch_lds, true);
                if (std::holds_alternative<std::string>(mres)) {
                    return set_error(std::get<std::string>(mres));
                }
                g.bm3d_match_pipeline = std::get<VkPipeline>(mres);

                const uint32_t * fcode = nullptr;
                size_t fsize = 0;
                if (d->cas_atomics) {
                    fcode = one_plane ? bm3d_filter_cas_1plane_spv
                                      : bm3d_filter_cas_spv;
                    fsize = one_plane ? bm3d_filter_cas_1plane_spv_size
                                      : bm3d_filter_cas_spv_size;
                } else {
                    fcode =
                        one_plane ? bm3d_filter_1plane_spv : bm3d_filter_spv;
                    fsize = one_plane ? bm3d_filter_1plane_spv_size
                                      : bm3d_filter_spv_size;
                }
                const auto fres =
                    create_bm3d_pipeline(*d->gpu, *d, g, fcode, fsize,
                                         d->pipeline_layout, false, false);
                if (std::holds_alternative<std::string>(fres)) {
                    return set_error(std::get<std::string>(fres));
                }
                g.bm3d_filter_pipeline = std::get<VkPipeline>(fres);
            } else {
                const uint32_t * code = nullptr;
                size_t code_size = 0;
                if (d->cas_atomics) {
                    code = d->patch_lds ? (one_plane ? bm3d_cas_lds_1plane_spv
                                                     : bm3d_cas_lds_spv)
                                        : (one_plane ? bm3d_cas_1plane_spv
                                                     : bm3d_cas_spv);
                    code_size = d->patch_lds
                                    ? (one_plane ? bm3d_cas_lds_1plane_spv_size
                                                 : bm3d_cas_lds_spv_size)
                                    : (one_plane ? bm3d_cas_1plane_spv_size
                                                 : bm3d_cas_spv_size);
                } else {
                    code =
                        d->patch_lds
                            ? (one_plane ? bm3d_lds_1plane_spv : bm3d_lds_spv)
                            : (one_plane ? bm3d_1plane_spv : bm3d_spv);
                    code_size = d->patch_lds
                                    ? (one_plane ? bm3d_lds_1plane_spv_size
                                                 : bm3d_lds_spv_size)
                                    : (one_plane ? bm3d_1plane_spv_size
                                                 : bm3d_spv_size);
                }
                const auto result = create_bm3d_pipeline(
                    *d->gpu, *d, g, code, code_size, d->pipeline_layout,
                    d->patch_lds, true);
                if (std::holds_alternative<std::string>(result)) {
                    return set_error(std::get<std::string>(result));
                }
                g.bm3d_pipeline = std::get<VkPipeline>(result);
            }
        }
        const int groups = static_cast<int>(d->groups_per_wg);
        g.bm3d_grid_x = static_cast<uint32_t>(
            (first.width + groups * first.block_step - 1) /
            (groups * first.block_step));
        g.bm3d_grid_y = static_cast<uint32_t>(
            (first.height + first.block_step - 1) / first.block_step);
        for (int pi = 0; pi < g.n_planes; ++pi) {
            auto & p = d->planes[g.planes[pi]];
            if (!p.process) {
                continue; // a joint entry's skipped plane is never dispatched
            }
            // The code and its size stay inline at the call site: assigning
            // them to locals first is what tools/shader_limits.py resolves, and
            // an identifier carrying the io stems attaches them to every other
            // helper it is passed through.
            const auto result =
                (d->rgb || d->bits != 16)
                    ? create_agg_pipeline(*d->gpu, p, g, *d, bm3d_agg_f32_spv,
                                          bm3d_agg_f32_spv_size,
                                          d->pipeline_layout, d->agg_wg.x,
                                          d->agg_wg.y)
                    : (d->half ? create_agg_pipeline(
                                     *d->gpu, p, g, *d, bm3d_agg_f16_spv,
                                     bm3d_agg_f16_spv_size, d->pipeline_layout,
                                     d->agg_wg.x, d->agg_wg.y)
                               : create_agg_pipeline(
                                     *d->gpu, p, g, *d, bm3d_agg_u16_spv,
                                     bm3d_agg_u16_spv_size, d->pipeline_layout,
                                     d->agg_wg.x, d->agg_wg.y));
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            p.agg_pipeline = std::get<VkPipeline>(result);
            p.agg_grid_x = (p.stride + d->agg_wg.x - 1) / d->agg_wg.x;
            p.agg_grid_y = (p.height + d->agg_wg.y - 1) / d->agg_wg.y;
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

    // Per-instance VRAM budget. The rings and the handoff come from the core's
    // pool, the finished-frame cache from the core's frame allocator; the
    // isolated source ring is not in the total because it does not exist yet.
    // Gated by an env flag so a normal creation prints nothing.
    if (vsfeel_debug_flag("VSFEEL_BM3D_VRAM")) {
        const double mib = 1024.0 * 1024.0;
        const double src_bytes = static_cast<double>(src_total) * 4.0;
        const double res_bytes = static_cast<double>(res_total) * 4.0;
        double iso_acc_bytes = 0.0;
        for (int gi = 0; gi < d->n_groups; ++gi) {
            const auto & g = d->groups[gi];
            if (d->max_radius > 0) {
                iso_acc_bytes +=
                    static_cast<double>(2) * g.n_planes * g.pe * 4.0;
            }
        }
        const double handoff_bytes =
            d->split_kernel ? static_cast<double>(d->match_words) * 4.0 : 0.0;
        const double total =
            src_bytes + res_bytes + iso_acc_bytes + handoff_bytes;
        char budget_txt[48];
        if (budget > 0) {
            snprintf(budget_txt, sizeof(budget_txt), "%.0f MiB",
                     static_cast<double>(budget) / mib);
        } else {
            snprintf(budget_txt, sizeof(budget_txt), "unknown");
        }
        fprintf(
            stderr,
            "[bm3d] vram: src=%.1f MiB acc=%.1f MiB iso_acc=%.1f MiB "
            "handoff=%.1f MiB iso_src=lazy -> total=%.1f MiB "
            "(budget=%s, emitted_cap=%zu) "
            "(planes=%d entries=%d radius=%d inflight=%d res_cap=%d src_ring=%d "
            "stride=%d bits=%d)\n",
            src_bytes / mib, res_bytes / mib, iso_acc_bytes / mib,
            handoff_bytes / mib, total / mib, budget_txt, d->emitted_cap,
            d->num_planes, d->n_groups, d->max_radius, inflight, res_cap,
            src_ring, d->planes[0].stride, d->bits);
    }

    for (int gi = 0; gi < d->n_groups; ++gi) {
        auto & g = d->groups[gi];
        g.acc_frame.assign(g.res_cap, -1);
        g.r0_free.reserve(inflight);
        for (int i = 0; i < inflight; ++i) {
            g.r0_free.push_back(i);
        }
    }

    BM3DData * data = d.release();

    // A temporal filter requests frames outside n, which the strict-spatial
    // policy does not permit; only radius 0 is purely spatial.
    const VSRequestPattern policy =
        data->max_radius > 0 ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2] = { { data->node, policy },
                                   { data->ref_node, policy } };

    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    // The temporal path keeps one cursor and advances it in frame order; the
    // core serializes the calls so the cursor always belongs to the frame it
    // is working on. Radius 0 has no cross-frame state and stays parallel.
    VSNode * result = vsapi->createVideoFilterEx2(
        data->name, data->vi, BM3DGetFrame, BM3DFree,
        data->max_radius > 0 ? fmUnordered : fmParallel, ffGPUOutput, deps,
        data->ref_node ? 2 : 1, data, core);
    if (result == nullptr) {
        // The core returns nullptr without running the free callback when the
        // node constructor throws, so release the instance through it here.
        vsfeel_trace_error(data->name, -1, "filter creation failed",
                           data->gpu.get());
        const char * name = data->name;
        BM3DFree(data, core, vsapi);
        vsapi->mapSetError(
            out, (std::string(name) + ": filter creation failed").c_str());
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_bm3d(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    // The documented argument order, with the ignored compatibility pair last
    // like every other filter. BM3Dv2 is the same filter under its former name;
    // the invoked name reaches the instance so its errors and node say BM3Dv2.
    static constexpr const char * argspec = "clip:vnode:gpu;"
                                            "ref:vnode:gpu:opt;"
                                            "sigma:float[]:opt;"
                                            "block_step:int[]:opt;"
                                            "bm_range:int[]:opt;"
                                            "radius:int[]:opt;"
                                            "ps_num:int[]:opt;"
                                            "ps_range:int[]:opt;"
                                            "th_mse:float:opt;"
                                            "chroma:int:opt;"
                                            "extractor_exp:int:opt;"
                                            "matrix:int:opt;"
                                            "device_id:int:opt;"
                                            "num_streams:int:opt;";
    vspapi->registerFunction("BM3D", argspec, "clip:vnode:gpu;", BM3DCreate,
                             const_cast<char *>("BM3D"), plugin);
    vspapi->registerFunction("BM3Dv2", argspec, "clip:vnode:gpu;", BM3DCreate,
                             const_cast<char *>("BM3Dv2"), plugin);
}
