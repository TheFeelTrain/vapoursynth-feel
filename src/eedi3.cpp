#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <volk.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// EEDI3 - full-pel edge-directed line interpolation on the core's Vulkan device
// (the R80 GPU API: vnode:gpu inputs, ffGPUOutput, one exec pool).
//
// Family-A semantics: reproduces HolyWu's eedi3m (the CPU ground truth) and
// agrees with eedi3vk2 (same family, GPU). EEDI3, EEDI3H (native
// transposed-plane, no std.Transpose nodes) and EEDI3AA (fused based_aa chain)
// share every kernel, geometry and pipeline.
//
// Everything the filter once did on the host is gone: the input planes are read
// straight out of the core's GPU frames at their own row pitch, the mask
// predicate and its copyMask dilation are GPU kernels, and every kernel writes
// into the output frame's own memory. One command buffer per output frame,
// submitted through the exec pool that also carries the frames' producer pairs.
//
// Kernels per plane (entry points in src/eedi3.comp):
//   ENTRY_MASKDILATE mask predicate + copyMask dilation -> packed u32 rows
//   ENTRY_MASKPACK   horizontal only: transpose the predicate bit matrix
//   ENTRY_XPOSE      horizontal only: kept columns -> packed R'/B'
//   ENTRY_PAD        mirror-pad builder
//   ENTRY_ROW        one workgroup per interp row; DP + backtrack + interpolate
//   ENTRY_VCHECK     serial (or parallel) interp-row vcheck
//   ENTRY_VCOPY      fully-masked rows copied in parallel
//   ENTRY_BLIT       vertical output: interp + kept rows into the frame plane
//   ENTRY_COMPOSE    horizontal output assembly (+ EEDI3AA's fused 50/50 merge)
//   ENTRY_ASSEMBLEV  EEDI3AA vertical merge into the intermediate frame v
// ---------------------------------------------------------------------------

constexpr int MARGIN_H = 12;
constexpr int MARGIN_V = 4;
constexpr int MAX_PLANES = 3;
// One 32-lane subgroup per interp row (see the row kernel).
constexpr int SGSIZE = 32;

// Compile-time max plane width for the shared-memory (LDS) vcheck variant. The
// single source is EEDI3_MAXW in CMakeLists.txt, which passes -DMAXW to the
// vcheck_lds shader rule and -DEEDI3_MAXW_LDS here; wider planes fall back to
// the global-read vcheck.
#ifndef EEDI3_MAXW_LDS
#define EEDI3_MAXW_LDS 4096
#endif
static constexpr int MAXW_LDS = EEDI3_MAXW_LDS;

// Pad element size: native u16 for 16-bit input (halves the pad traffic vs
// float), float for 32-bit. u16 values are exact in float so all downstream
// cost/interp math is bit-identical to a float pad.
static int pad_elem_bytes(int bits) {
    return (bits == 16) ? 2 : 4;
}

// Push constant layout; must match the PC block in eedi3.comp exactly.
struct Eedi3PushConstants {
    int32_t pad_base;
    int32_t dst_base;
    int32_t pbt_base;
    int32_t dmap_base;
    int32_t bmask_base;
    int32_t sclip_base;
    int32_t cint_base;
    int32_t vout_base;
    int32_t pad_stride;
    int32_t pad_height;
    int32_t field;
    int32_t rows;
    float alpha;
    float beta;
    float gamma;
    float rw;
    float vth0r;
    float vth1r;
    float vth2r;
    float vth2;
    int32_t raw_base;
    int32_t rempty_base;
    int32_t dst_stride;
    int32_t src_stride;
    int32_t sclip_stride;
    int32_t src_first;
    int32_t src_step;
    int32_t out_base;
    int32_t vout2_base;
    int32_t comp_fuse;
    int32_t pad_src_pitched;
};
static_assert(sizeof(Eedi3PushConstants) == 31 * sizeof(int32_t),
              "push constants size");

// Row/vcheck specialization block (constant ids 0-5 and the two local sizes).
struct Eedi3RowSpec {
    int32_t width;
    int32_t nrad;
    int32_t mdis;
    int32_t has_mclip;
    int32_t has_sclip;
    int32_t vcheck;
    int32_t lsz_row;
    int32_t lsz_vcheck;
};

static constexpr std::array<VkSpecializationMapEntry, 8> row_entries {{
    { 0,  0, sizeof(int32_t) },
    { 1,  4, sizeof(int32_t) },
    { 2,  8, sizeof(int32_t) },
    { 3, 12, sizeof(int32_t) },
    { 4, 16, sizeof(int32_t) },
    { 5, 20, sizeof(int32_t) },
    { 6, 24, sizeof(int32_t) },
    { 7, 28, sizeof(int32_t) },
}};

struct Eedi3Pipelines {
    VkPipeline row {};
    VkPipeline vcheck {};
    VkPipeline pad {};
    VkPipeline vcopy {};
    VkPipeline blit {};
    VkPipeline xpose {};
    VkPipeline compose {};
    VkPipeline maskpack {};
    VkPipeline maskdilate_raw {};
    VkPipeline maskdilate_tr {};
    VkPipeline assemble {};
    bool vcheck_lds {};
    bool vcheck_para {};
};

// Per-geometry plane description: kernel dims, region offsets into the per
// frame scratch buffer, and the pipelines that run it.
struct Eedi3PlaneConfig {
    int width {};                     // kernel plane width (row kernel's WIDTH)
    int height {};                    // kernel plane height
    int src_h {};                     // source plane height (gather geometry)
    int out_w {};                     // output (frame order) plane dims
    int out_h {};
    int rows {};                      // interp rows == height / 2
    int pad_stride {};                // pad elements per padded row
    int pad_height {};                // padded rows == height + 2*MARGIN_V
    int tpitch {};                    // 2*mdis + 1
    Eedi3Pipelines pipes {};          // shared per geometry (see width_pipes)

    // region offsets (bytes into the per-frame scratch buffer) and sizes
    VkDeviceSize pad_off {}, pad_bytes {};
    VkDeviceSize dst_off {}, dst_bytes {};
    VkDeviceSize dst2_off {}, dst2_bytes {};
    VkDeviceSize pbt_off {}, pbt_bytes {};
    VkDeviceSize dmap_off {}, dmap_bytes {};
    VkDeviceSize cint_off {}, cint_bytes {};
    VkDeviceSize vout_off {}, vout_bytes {};
    VkDeviceSize vout2_off {}, vout2_bytes {};
    VkDeviceSize rempty_off {}, rempty_bytes {};
    VkDeviceSize bits_off {}, bits_bytes {};
    VkDeviceSize pred_off {}, pred_bytes {};
    VkDeviceSize rt_off {}, rt_bytes {};
    VkDeviceSize rtS_off {}, rtS_bytes {};
    VkDeviceSize o0_off {}, o0_bytes {};
    VkDeviceSize v_off {}, v_bytes {};
};

// What a recorded pass does after the row kernel + vcheck.
enum class PassTail {
    kBlit,        // vertical: write interp + kept rows into the output plane
    kCompose,     // horizontal: assemble the frame-order plane
    kAssembleV,   // EEDI3AA vertical merge into the intermediate frame v
    kNone,        // EEDI3AA's first vertical sub-pass
};

// Everything a recorded pass needs to address: the pass's source planes, the
// mask/sclip planes, the output frame planes and the shared scratch buffer.
struct Eedi3PassInputs {
    VkBuffer src[MAX_PLANES] {};       // source for xpose/pad/blit
    int src_base[MAX_PLANES] {};       // element base into src[]
    int src_stride[MAX_PLANES] {};     // source row pitch, io elements
    VkBuffer mask {};                  // single Gray mask plane
    int mask_stride {};                // mask row pitch, mask elements
    VkBuffer sclip[MAX_PLANES] {};     // sclip planes (vertical xpose/vcheck)
    int sclip_stride[MAX_PLANES] {};   // sclip row pitch, io elements
    VkBuffer out[MAX_PLANES] {};       // output frame planes
    int out_stride[MAX_PLANES] {};     // output row pitch, io elements
    VkBuffer scratch {};               // per-frame scratch buffer
};

struct Eedi3Data {
    VSNode * node {};
    VSNode * sclip_node {};
    VSNode * mclip_node {};
    const VSVideoInfo * vi {};

    int bits {}, elem_bytes {};
    bool process[MAX_PLANES] { true, true, true };

    int field {}, nrad { 2 }, mdis { 20 }, vcheck { 2 };
    bool dh {};
    bool horiz { false };   // run the whole pipeline on the transposed plane
    bool aa { false };      // fused based_aa vertical-then-horizontal chain
    std::array<Eedi3PlaneConfig, MAX_PLANES> aplanes {};

    // vcheck d2p source. Levels trade accuracy for work: 1 = the un-vchecked
    // dst row, 2..6 = +an increasing number of Jacobi steps. The shipped
    // default reproduces the serial walk exactly on the whole tested surface;
    // VSFEEL_EEDI3_VPARA=0 restores the serial walk; VSFEEL_EEDI3_VCLDS=1
    // selects the shared-memory ping-pong walk instead.
    static constexpr int VCHECK_PARA_LEVELS = 6;
    int vcheck_para { VCHECK_PARA_LEVELS };

    // Native mask handling: Gray16 integer and Gray32 float masks are read
    // straight from the GPU plane (no conversion node); other depths keep the
    // reference SetFrameProps(_Range=1) + resize.Point -> Gray8 path.
    int mclip_bits { 8 };
    bool mclip_native16 {}, mclip_native32 {};

    float alpha { 0.2f }, beta { 0.25f }, gamma { 20.0f };
    float vthresh2 { 4.0f };
    float rw {}, rcp_vth0 {}, rcp_vth1 {}, rcp_vth2 {};

    // one pipeline set per distinct plane geometry (like eedi3vk2); pipelines
    // for the same geometry are shared by all planes. HORIZ is a spec constant
    // (the vcheck's sclip layout), so it is part of the key.
    struct WidthKey {
        int width, rows, tpitch, pad_stride, pad_height;
        bool horiz;
        bool operator==(const WidthKey & o) const {
            return width == o.width && rows == o.rows && tpitch == o.tpitch &&
                   pad_stride == o.pad_stride && pad_height == o.pad_height &&
                   horiz == o.horiz;
        }
    };

    std::shared_ptr<GPUDevice> gpu;
    VSCore * core {};
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VSGPUExecPool * pool {};

    // shader blobs for this bit depth
    const uint32_t * row_code {};
    size_t row_size {};
    const uint32_t * vcheck_code {};
    size_t vcheck_size {};
    const uint32_t * vcheck_lds_code {};
    size_t vcheck_lds_size {};
    std::array<const uint32_t *, VCHECK_PARA_LEVELS> vcheck_para_code {};
    std::array<size_t, VCHECK_PARA_LEVELS> vcheck_para_size {};
    const uint32_t * pad_code {};
    size_t pad_size {};
    const uint32_t * vcopy_code {};
    size_t vcopy_size {};
    const uint32_t * blit_code {};
    size_t blit_size {};
    const uint32_t * xpose_code {};
    size_t xpose_size {};
    const uint32_t * compose_code {};
    size_t compose_size {};
    const uint32_t * assemble_code {};
    size_t assemble_size {};
    // mask kernels, indexed by mask format (0 = 8 bit, 1 = 16, 2 = 32)
    std::array<const uint32_t *, 3> maskpack_code {};
    std::array<size_t, 3> maskpack_size {};
    std::array<const uint32_t *, 3> maskdilate_raw_code {};
    std::array<size_t, 3> maskdilate_raw_size {};
    std::array<const uint32_t *, 3> maskdilate_tr_code {};
    std::array<size_t, 3> maskdilate_tr_size {};
    bool have_lds {};

    VkDeviceSize scratch_bytes { 1 };
    std::array<Eedi3PlaneConfig, MAX_PLANES> planes {};
    std::vector<std::pair<WidthKey, Eedi3Pipelines>> width_pipes {};

    // VSFEEL_EEDI3_TRACE only.
    bool trace { false };

    // VSFEEL_EEDI3_TIMING=1: per-frame host stage split. The prerequisite for
    // any transfer/allocation change -- kernel time alone says nothing about
    // whether the frame is GPU- or host-bound. Every stage is clocked after the
    // previous blocking call so a wait never leaks into the next stage.
    bool host_timing { false };
    // VSFEEL_EEDI3_SYNC=1: wait the submission out and report its wall time. It
    // serializes the pipeline, so it measures one frame's GPU time alone.
    bool sync_wait { false };
    std::atomic<uint64_t> ht_acquire_ns {}, ht_alloc_ns {}, ht_record_ns {},
        ht_submit_ns {}, ht_total_ns {}, ht_n {}, ht_fence_ns {};

    // Output-frame batching (see the batching section above).
    int batch_size { 4 };
    int max_out { -1 };            // last valid output frame index, -1 if unknown
    std::mutex cache_lock;
    std::condition_variable cache_cv;
    std::vector<std::pair<int, VSFrame *>> cache;   // computed frames awaiting a caller
    std::vector<std::pair<int, int>> claims;        // ranges being recorded

    ~Eedi3Data();
};

Eedi3Data::~Eedi3Data() {
    if (host_timing && ht_n.load()) {
        const double n = static_cast<double>(ht_n.load());
        fprintf(stderr,
            "[eedi3-timing] frames=%.0f per-frame us: acquire=%7.1f alloc=%7.1f "
            "record=%7.1f submit=%7.1f wait=%7.1f total=%7.1f\n",
            n, ht_acquire_ns.load() / 1000.0 / n, ht_alloc_ns.load() / 1000.0 / n,
            ht_record_ns.load() / 1000.0 / n, ht_submit_ns.load() / 1000.0 / n,
            ht_fence_ns.load() / 1000.0 / n, ht_total_ns.load() / 1000.0 / n);
    }
    if (!gpu) {
        return;
    }
    // The pool owns every scratch buffer and drains its submissions first, so
    // the pipelines and layouts below are safe to destroy once it is gone.
    if (pool) {
        gpu->api->freeGPUExecPool(pool);
        pool = nullptr;
    }
    const GPUDevice & g = *gpu;
    std::vector<VkPipeline> destroyed;
    for (auto & [key, p] : width_pipes) {
        const VkPipeline all[] {
            p.row, p.vcheck, p.pad, p.vcopy, p.blit, p.xpose, p.compose,
            p.maskpack, p.maskdilate_raw, p.maskdilate_tr, p.assemble
        };
        for (VkPipeline pipe : all) {
            if (!pipe) {
                continue;
            }
            if (std::find(destroyed.begin(), destroyed.end(), pipe) == destroyed.end()) {
                destroyed.push_back(pipe);
                g.vk->vkDestroyPipeline(g.device, pipe, nullptr);
            }
        }
    }
    if (pipeline_layout) {
        g.vk->vkDestroyPipelineLayout(g.device, pipeline_layout, nullptr);
    }
    if (set_layout) {
        g.vk->vkDestroyDescriptorSetLayout(g.device, set_layout, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Recording helpers
// ---------------------------------------------------------------------------

static void pc_push(const Eedi3Data & d, VkCommandBuffer cmd,
                    const Eedi3PushConstants & pc) {
    gpu_push_constants(*d.gpu, cmd, d.pipeline_layout, &pc, sizeof(pc));
}

static void bind_pipe(const Eedi3Data & d, VkCommandBuffer cmd, VkPipeline pipe) {
    d.gpu->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
}

static void bind_bufs(const Eedi3Data & d, VkCommandBuffer cmd,
                      std::initializer_list<VkBuffer> bufs) {
    std::array<VkBuffer, GPU_MAX_BINDINGS> arr {};
    uint32_t n = 0;
    for (VkBuffer b : bufs) {
        arr[n++] = b;
    }
    gpu_push_buffers(*d.gpu, cmd, d.pipeline_layout, arr.data(), n);
}

static void dispatch(const Eedi3Data & d, VkCommandBuffer cmd, int x, int y = 1,
                     int z = 1) {
    // Every Vulkan entry point goes through the core's loaded table: nothing
    // links or initializes the loader on this path.
    d.gpu->vk->vkCmdDispatch(cmd, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                             static_cast<uint32_t>(z));
}

// Host-stage accumulation for VSFEEL_EEDI3_TIMING (see the member comment).
static void eedi3_add_timing(Eedi3Data & d, uint64_t signaled,
                             std::chrono::steady_clock::time_point t0,
                             std::chrono::steady_clock::time_point t1,
                             std::chrono::steady_clock::time_point t2,
                             std::chrono::steady_clock::time_point t3,
                             std::chrono::steady_clock::time_point t4) {
    if (!d.host_timing) {
        return;
    }
    const auto ns = [](auto a, auto b) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    d.ht_acquire_ns += ns(t0, t1);
    d.ht_alloc_ns += ns(t1, t2);
    d.ht_record_ns += ns(t2, t3);
    d.ht_submit_ns += ns(t3, t4);
    d.ht_total_ns += ns(t0, t4);
    d.ht_n += 1;
    if (d.sync_wait) {
        char werr[256] {};
        const auto w0 = std::chrono::steady_clock::now();
        d.gpu->api->gpuExecWaitValue(d.pool, signaled, werr, sizeof(werr));
        d.ht_fence_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - w0).count());
    }
}

// One EEDI3 sub-pass: mask prep -> (horizontal transpose) -> pad build -> row
// kernel -> vcheck, plus the tail.
//
// Recording is split into phases because Vulkan orders storage accesses only
// where a barrier says so, and a barrier's second scope is everything after it
// in the command buffer. Running one whole frame per barrier therefore
// serializes frames; instead a batch runs *all* of its pad builders, then all
// of its row kernels with no barrier between them, and so on. That is what
// lets several frames' latency-bound row kernels share the one compute queue
// the core exposes (a single frame's `rows` workgroups cannot fill the GPU).
enum class PassPhase { kPrep, kRow, kVcheck, kTail };

// One frame's inputs for one sub-pass.
struct Eedi3Job {
    const Eedi3PlaneConfig * planes {};
    Eedi3PassInputs in {};
    int field {};
    bool horiz {}, second {};
    PassTail tail { PassTail::kNone };
};

static void record_pass(const Eedi3Data & d, VkCommandBuffer cmd,
                        const Eedi3Job * jobs, const int njobs,
                        const PassPhase phase) {

    const int32_t elem = d.elem_bytes;
    const int32_t pad_elem = pad_elem_bytes(d.bits);
    const int numPlanes = d.vi->format.numPlanes;

    for (int job = 0; job < njobs; ++job) {
        const Eedi3Job & jb = jobs[job];
        const Eedi3PlaneConfig * planes = jb.planes;
        const Eedi3PassInputs & in = jb.in;
        const int field = jb.field;
        const bool horiz = jb.horiz;
        const bool second = jb.second;
        const PassTail tail = jb.tail;
        // The kept parity of a horizontal pass's columns: the row kernel's
        // interp rows are the source columns of parity `field`, so the
        // transposed kept columns are `1 - field`. Under dh every source column
        // is kept.
        const int32_t kept_first = d.dh ? 0 : (1 - field);
        const int32_t kept_step = d.dh ? 1 : 2;

        if (phase == PassPhase::kPrep) {
            // Mask preparation: the row kernel reads packed dilated predicate
            // bits. Vertically the mask plane's interp-parity rows are dilated
            // along x; horizontally its interp-parity columns are first
            // transposed into the predicate bit matrix (ENTRY_MASKPACK) and
            // then dilated along y, which is the same matrix transposed.
            if (d.mclip_node) {
                const int32_t mfirst = d.dh ? 0 : field;
                const int32_t mstep = d.dh ? 1 : 2;
                for (int plane = 0; plane < numPlanes; ++plane) {
                    if (!d.process[plane]) {
                        continue;
                    }
                    const auto & cfg = planes[plane];
                    const auto & pipes = cfg.pipes;
                    Eedi3PushConstants pc {};
                    if (horiz) {
                        bind_pipe(d, cmd, pipes.maskpack);
                        bind_bufs(d, cmd, { in.mask, in.scratch });
                        pc.raw_base = static_cast<int32_t>(cfg.pred_off / 4);
                        pc.src_stride = in.mask_stride;
                        pc.rows = cfg.rows;
                        pc.src_first = mfirst;
                        pc.src_step = mstep;
                        pc_push(d, cmd, pc);
                        const int mask_w = cfg.rows * mstep;
                        dispatch(d, cmd, (mask_w + 31) / 32, (cfg.width + 31) / 32);
                        gpu_barrier(*d.gpu, cmd);

                        bind_pipe(d, cmd, pipes.maskdilate_tr);
                        bind_bufs(d, cmd, { in.scratch, in.scratch });
                        Eedi3PushConstants dpc {};
                        dpc.raw_base = static_cast<int32_t>(cfg.pred_off / 4);
                        dpc.bmask_base = static_cast<int32_t>(cfg.bits_off / 4);
                        dpc.rows = cfg.rows;
                        pc_push(d, cmd, dpc);
                    } else {
                        bind_pipe(d, cmd, pipes.maskdilate_raw);
                        bind_bufs(d, cmd, { in.mask, in.scratch });
                        pc.raw_base = 0;
                        pc.src_stride = in.mask_stride;
                        pc.rows = cfg.rows;
                        pc.src_first = mfirst;
                        pc.src_step = mstep;
                        pc.bmask_base = static_cast<int32_t>(cfg.bits_off / 4);
                        pc_push(d, cmd, pc);
                    }
                    dispatch(d, cmd, cfg.rows);
                    gpu_barrier(*d.gpu, cmd);
                }
            }

            // Horizontal passes: transpose the kept source columns into R' (and
            // the sclip's interp columns into B') so the pad builder and vcheck
            // see their usual packed layout.
            if (horiz) {
                for (int plane = 0; plane < numPlanes; ++plane) {
                    if (!d.process[plane]) {
                        continue;
                    }
                    const auto & cfg = planes[plane];
                    bind_pipe(d, cmd, cfg.pipes.xpose);
                    bind_bufs(d, cmd, { in.src[plane], in.scratch });
                    Eedi3PushConstants pc {};
                    pc.raw_base = in.src_base[plane];
                    pc.src_stride = in.src_stride[plane];
                    pc.pad_base = static_cast<int32_t>(cfg.rt_off / pad_elem);
                    pc.rows = cfg.rows;
                    pc.src_first = kept_first;
                    pc.src_step = kept_step;
                    pc_push(d, cmd, pc);
                    dispatch(d, cmd, (cfg.rows + 15) / 16, (cfg.width + 15) / 16);

                    if (d.vcheck > 0 && d.sclip_node) {
                        bind_pipe(d, cmd, cfg.pipes.xpose);
                        bind_bufs(d, cmd, { in.sclip[plane], in.scratch });
                        Eedi3PushConstants spc {};
                        spc.raw_base = 0;
                        spc.src_stride = in.sclip_stride[plane];
                        spc.pad_base = static_cast<int32_t>(cfg.rtS_off / elem);
                        spc.rows = cfg.rows;
                        // The output doubles the transposed width under dh, so
                        // the sclip's interp columns are field+2k even there.
                        spc.src_first = field;
                        spc.src_step = 2;
                        pc_push(d, cmd, spc);
                        dispatch(d, cmd, (cfg.rows + 15) / 16, (cfg.width + 15) / 16);
                    }
                    gpu_barrier(*d.gpu, cmd);
                }
            }

            // Mirror-pad builder: one thread per padded element.
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!d.process[plane]) {
                    continue;
                }
                const auto & cfg = planes[plane];
                bind_pipe(d, cmd, cfg.pipes.pad);
                // Horizontally the pad builder reads the packed R' in the
                // scratch buffer; vertically it reads the source frame plane.
                bind_bufs(d, cmd, { horiz ? in.scratch : in.src[plane], in.scratch });
                Eedi3PushConstants pc {};
                pc.raw_base = horiz ? static_cast<int32_t>(cfg.rt_off / pad_elem)
                                    : in.src_base[plane];
                pc.src_stride = horiz ? 0 : in.src_stride[plane];
                pc.src_first = kept_first;
                pc.src_step = kept_step;
                pc.pad_base = static_cast<int32_t>(cfg.pad_off / pad_elem);
                pc.pad_stride = cfg.pad_stride;
                pc.pad_height = cfg.pad_height;
                pc.field = field;
                pc.rows = cfg.rows;
                pc.pad_src_pitched = horiz ? 0 : 1;
                pc_push(d, cmd, pc);
                const int total = cfg.pad_stride * cfg.pad_height;
                dispatch(d, cmd, (total + 255) / 256);
            }
        }

        // The per-plane push constant block the row and vcheck kernels share.
        auto row_pc = [&](const Eedi3PlaneConfig & cfg, const int plane) {
            Eedi3PushConstants pc {};
            pc.pad_base = static_cast<int32_t>(cfg.pad_off / pad_elem);
            pc.dst_base = static_cast<int32_t>(
                (second ? cfg.dst2_off : cfg.dst_off) / elem);
            pc.pbt_base = static_cast<int32_t>(cfg.pbt_off);
            pc.dmap_base = static_cast<int32_t>(cfg.dmap_off);
            pc.bmask_base = static_cast<int32_t>(cfg.bits_off / 4);
            pc.sclip_base = horiz ? static_cast<int32_t>(cfg.rtS_off / elem) : 0;
            pc.sclip_stride = horiz ? 0 : in.sclip_stride[plane];
            pc.cint_base = static_cast<int32_t>(cfg.cint_off / elem);
            pc.vout_base = static_cast<int32_t>(
                (second ? cfg.vout2_off : cfg.vout_off) / elem);
            pc.pad_stride = cfg.pad_stride;
            pc.pad_height = cfg.pad_height;
            pc.field = field;
            pc.rows = cfg.rows;
            pc.alpha = d.alpha;
            pc.beta = d.beta;
            pc.gamma = d.gamma;
            pc.rw = d.rw;
            pc.vth0r = d.rcp_vth0;
            pc.vth1r = d.rcp_vth1;
            pc.vth2r = d.rcp_vth2;
            pc.vth2 = d.vthresh2;
            pc.rempty_base = static_cast<int32_t>(cfg.rempty_off);
            // The vcheck's sclip source layout: the packed transposed B' in the
            // scratch buffer (horizontal) or the sclip frame plane (vertical).
            pc.pad_src_pitched = horiz ? 1 : 0;
            return pc;
        };

        if (phase == PassPhase::kRow) {
            // Row kernel: one workgroup per interp row. Recorded with no
            // barrier between jobs so the batch's rows all share the queue.
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!d.process[plane]) {
                    continue;
                }
                const auto & cfg = planes[plane];
                const VkBuffer sclip_buf = (horiz || !in.sclip[plane])
                    ? in.scratch : in.sclip[plane];
                bind_pipe(d, cmd, cfg.pipes.row);
                bind_bufs(d, cmd, {
                    in.scratch, in.scratch, in.scratch, in.scratch,
                    in.scratch, sclip_buf, in.scratch, in.scratch });
                pc_push(d, cmd, row_pc(cfg, plane));
                dispatch(d, cmd, 1, cfg.rows);
            }
        }

        if (phase == PassPhase::kVcheck && d.vcheck > 0) {
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!d.process[plane]) {
                    continue;
                }
                const auto & cfg = planes[plane];
                const VkBuffer sclip_buf = (horiz || !in.sclip[plane])
                    ? in.scratch : in.sclip[plane];
                const Eedi3PushConstants pc = row_pc(cfg, plane);
                // Fully-masked rows are copied in parallel so the serial walk
                // only visits non-empty rows (the LDS and parallel forms carry
                // every row and need no copy pass).
                if (d.mclip_node && !cfg.pipes.vcheck_lds && !cfg.pipes.vcheck_para) {
                    bind_pipe(d, cmd, cfg.pipes.vcopy);
                    bind_bufs(d, cmd, {
                        in.scratch, in.scratch, in.scratch, in.scratch,
                        in.scratch, sclip_buf, in.scratch, in.scratch });
                    pc_push(d, cmd, pc);
                    const int total = cfg.rows * cfg.width;
                    dispatch(d, cmd, (total + 255) / 256);
                    gpu_barrier(*d.gpu, cmd);
                }
                bind_pipe(d, cmd, cfg.pipes.vcheck);
                bind_bufs(d, cmd, {
                    in.scratch, in.scratch, in.scratch, in.scratch,
                    in.scratch, sclip_buf, in.scratch, in.scratch });
                pc_push(d, cmd, pc);
                dispatch(d, cmd, 1, cfg.pipes.vcheck_para ? cfg.rows : 1);
            }
        }

        if (phase != PassPhase::kTail || tail == PassTail::kNone) {
            continue;
        }

        if (tail == PassTail::kAssembleV) {
            // EEDI3AA vertical merge: v[y] = (src[y] + vout_{y&1}[y>>1] + 1) >> 1.
            // Recorded with the SECOND sub-pass's parity, so the assembler's
            // `pc.field` (which vout region belongs to which parity) is its
            // complement.
            const int32_t first_field = 1 - field;
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!d.process[plane]) {
                    continue;
                }
                const auto & cfg = planes[plane];
                Eedi3PushConstants pc {};
                pc.raw_base = in.src_base[plane];
                pc.src_stride = in.src_stride[plane];
                pc.vout_base = static_cast<int32_t>(cfg.vout_off / elem);
                pc.vout2_base = static_cast<int32_t>(cfg.vout2_off / elem);
                pc.out_base = static_cast<int32_t>(cfg.v_off / elem);
                pc.field = first_field;
                pc.rows = cfg.rows;
                bind_pipe(d, cmd, cfg.pipes.assemble);
                bind_bufs(d, cmd, { in.src[plane], in.scratch, in.scratch, in.scratch });
                pc_push(d, cmd, pc);
                dispatch(d, cmd, (2 * cfg.rows * cfg.width + 255) / 256);
            }
            continue;
        }

        if (tail == PassTail::kCompose) {
            // EEDI3H / EEDI3AA horizontal output assembly. EEDI3AA's fused form
            // parks the first sub-pass's O_0 in the device-local o0 region and
            // the second merges against it straight into the output frame.
            // The interp source lives on its own binding, so the fused merge
            // works with or without a vcheck.
            const bool fuse = d.aa && planes[0].o0_bytes > 0;
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!d.process[plane]) {
                    continue;
                }
                const auto & cfg = planes[plane];
                Eedi3PushConstants pc {};
                pc.pad_base = static_cast<int32_t>(cfg.rt_off / elem);
                pc.vout_base = static_cast<int32_t>(
                    (second ? cfg.vout2_off : cfg.vout_off) / elem);
                pc.dst_base = static_cast<int32_t>(
                    (second ? cfg.dst2_off : cfg.dst_off) / elem);
                pc.out_base = fuse ? static_cast<int32_t>(cfg.o0_off / elem) : 0;
                pc.dst_stride = in.out_stride[plane];
                pc.field = field;
                pc.rows = cfg.rows;
                pc.comp_fuse = fuse ? (second ? 2 : 1) : 0;
                bind_pipe(d, cmd, cfg.pipes.compose);
                bind_bufs(d, cmd, { in.scratch, in.scratch, in.out[plane], in.scratch });
                pc_push(d, cmd, pc);
                dispatch(d, cmd, (cfg.rows + 15) / 16, (cfg.width + 15) / 16);
            }
            continue;
        }

        // kBlit: every output row is either an interp row or a kept source row.
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d.process[plane]) {
                continue;
            }
            const auto & cfg = planes[plane];
            Eedi3PushConstants pc {};
            pc.vout_base = static_cast<int32_t>(cfg.vout_off / elem);
            pc.dst_base = static_cast<int32_t>(cfg.dst_off / elem);
            pc.raw_base = in.src_base[plane];
            pc.src_stride = in.src_stride[plane];
            pc.dst_stride = in.out_stride[plane];
            pc.field = field;
            pc.rows = cfg.rows;
            pc.src_step = kept_step;
            bind_pipe(d, cmd, cfg.pipes.blit);
            bind_bufs(d, cmd, { in.scratch, in.src[plane], in.out[plane] });
            pc_push(d, cmd, pc);
            const int total = 2 * cfg.rows * cfg.width;
            dispatch(d, cmd, (total + 255) / 256);
        }
    }
}

// ---------------------------------------------------------------------------
// Output-frame batching
// ---------------------------------------------------------------------------
//
// The core exposes ONE compute queue, and a single EEDI3 frame launches only
// `rows` workgroups of a latency-bound row kernel -- nowhere near enough to
// fill the GPU. The pre-R80 filter ran up to eight streams on eight Vulkan
// queues, whose submissions overlapped; with one queue that overlap has to
// come from inside a submission, so a submission now carries a batch of output
// frames recorded phase by phase (see record_pass). Measured on a 1080p
// GRAY16 blank at field=3, mdis=20: 2.25 ms/frame with one frame per
// submission, 1.45 at two, 1.10 at four -- against 1.01 for the old
// eight-queue build.
//
// getFrame asks for a batch starting at its own frame, computes whatever part
// of it nobody else is computing, and caches the rest for the sibling requests
// that follow. The cache owns its frames; a caller takes one out (ownership
// transfers to it), and leftover entries are dropped once they are older than
// three batches.

// Take the cached output frame `n`, transferring ownership. NULL when absent.
static VSFrame * eedi3_take_cached(Eedi3Data * d, const int n) {
    std::lock_guard guard(d->cache_lock);
    for (auto it = d->cache.begin(); it != d->cache.end(); ++it) {
        if (it->first == n) {
            VSFrame * frame = it->second;
            d->cache.erase(it);
            return frame;
        }
    }
    return nullptr;
}

static bool eedi3_cache_has(const Eedi3Data * d, const int n) {
    for (const auto & e : d->cache) {
        if (e.first == n) {
            return true;
        }
    }
    return false;
}

// Claim [n, last] so only this thread records it. False when another thread
// published frame n while we waited: the caller then takes it.
static bool eedi3_claim_batch(Eedi3Data * d, const int n, const int last) {
    std::unique_lock lock(d->cache_lock);
    for (;;) {
        if (eedi3_cache_has(d, n)) {
            return false;
        }
        bool busy = false;
        for (const auto & c : d->claims) {
            if (n >= c.first && n <= c.second) {
                busy = true;
                break;
            }
        }
        if (!busy) {
            d->claims.emplace_back(n, last);
            return true;
        }
        d->cache_cv.wait(lock, [&] {
            if (eedi3_cache_has(d, n)) {
                return true;
            }
            for (const auto & c : d->claims) {
                if (n >= c.first && n <= c.second) {
                    return false;
                }
            }
            return true;
        });
    }
}

static void eedi3_release_claim(Eedi3Data * d, const int first, const int last) {
    std::lock_guard guard(d->cache_lock);
    for (auto it = d->claims.begin(); it != d->claims.end(); ++it) {
        if (it->first == first && it->second == last) {
            d->claims.erase(it);
            break;
        }
    }
    d->cache_cv.notify_all();
}

// Publish freshly computed frames and drop the claim. The oldest entries are
// evicted past three batches' worth; their frames are freed outside the lock.
static void eedi3_publish(Eedi3Data * d, const int first, const int last,
                          std::vector<std::pair<int, VSFrame *>> & produced,
                          const VSAPI * vsapi) {
    std::vector<VSFrame *> victims;
    {
        std::lock_guard guard(d->cache_lock);
        for (auto & p : produced) {
            d->cache.push_back(p);
        }
        produced.clear();
        const size_t cap = static_cast<size_t>(d->batch_size) * 3;
        while (d->cache.size() > cap) {
            victims.push_back(d->cache.front().second);
            d->cache.erase(d->cache.begin());
        }
    }
    eedi3_release_claim(d, first, last);
    for (VSFrame * v : victims) {
        vsapi->freeFrame(v);
    }
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// Everything that must happen on every exit path after the exec context is
// held: abandon the recording (or not, once submitted), free the frames and
// the output frame, and report the error.
struct Eedi3FrameGuard {
    const Eedi3Data & d;
    VSGPUExecContext * ctx {};
    bool submitted { false };
    int frame {};
};

static const VSFrame * eedi3_error(Eedi3FrameGuard & guard, VSFrameContext * frameCtx,
                                   const VSAPI * vsapi, VSFrame * dst, const char * what,
                                   const std::string & message) {
    if (guard.ctx && !guard.submitted) {
        guard.d.gpu->api->gpuExecAbandon(guard.ctx);
    }
    vsfeel_trace_error(guard.d.aa ? "EEDI3AA" : "EEDI3VK", guard.frame, message,
                       guard.d.gpu.get());
    vsapi->setFilterError((std::string(what) + ": " + message).c_str(), frameCtx);
    if (dst) {
        vsapi->freeFrame(dst);
    }
    return nullptr;
}

// Fill the shared mask/sclip planes once (they are the same for every pass).
static bool fill_mask_sclip(const Eedi3Data & d, const VSAPI * vsapi,
                            const VSFrame * scp, const VSFrame * mcp,
                            VkBuffer sclip[MAX_PLANES], int sclip_stride[MAX_PLANES],
                            VkBuffer & mask, int & mask_stride,
                            std::string & err) {
    const int numPlanes = d.vi->format.numPlanes;
    for (int plane = 0; plane < numPlanes; ++plane) {
        sclip[plane] = VK_NULL_HANDLE;
        sclip_stride[plane] = 0;
        if (scp && d.vcheck > 0 && d.process[plane]) {
            VSVulkanPlaneInfo pi {};
            if (d.gpu->api->getGPUPlane(scp, plane, &pi)) {
                err = "sclip plane " + std::to_string(plane) + " is not GPU resident";
                return false;
            }
            sclip[plane] = pi.buffer;
            sclip_stride[plane] = static_cast<int>(
                vsapi->getStride(scp, plane) / d.elem_bytes);
        }
    }
    mask = VK_NULL_HANDLE;
    mask_stride = 0;
    if (mcp) {
        VSVulkanPlaneInfo pi {};
        if (d.gpu->api->getGPUPlane(mcp, 0, &pi)) {
            err = "mclip is not GPU resident";
            return false;
        }
        mask = pi.buffer;
        mask_stride = static_cast<int>(
            vsapi->getStride(mcp, 0) / (d.mclip_bits / 8));
    }
    return true;
}

static bool fill_out_planes(const Eedi3Data & d, const VSAPI * vsapi, const VSFrame * dst,
                            VkBuffer out[MAX_PLANES], int out_stride[MAX_PLANES],
                            std::string & err) {
    const int numPlanes = d.vi->format.numPlanes;
    for (int plane = 0; plane < numPlanes; ++plane) {
        out[plane] = VK_NULL_HANDLE;
        out_stride[plane] = 0;
        if (!d.process[plane]) {
            continue;
        }
        VSVulkanPlaneInfo pi {};
        if (d.gpu->api->getGPUPlane(dst, plane, &pi)) {
            err = "the output plane " + std::to_string(plane) + " is not GPU resident";
            return false;
        }
        out[plane] = pi.buffer;
        out_stride[plane] = static_cast<int>(
            vsapi->getStride(dst, plane) / d.elem_bytes);
    }
    return true;
}

// Clip/mask input frame index for one output frame (sclip describes the output,
// so it is indexed by the output frame directly).
static inline int eedi3_clip_index(const Eedi3Data & d, const int n) {
    return (d.field > 1 && !d.aa) ? n / 2 : n;
}

// Last output frame of the batch starting at n.
static inline int eedi3_batch_last(const Eedi3Data & d, const int n) {
    if (d.max_out < 0) {
        return n;   // unknown length: never request past the end
    }
    return std::min(n + d.batch_size - 1, d.max_out);
}

static const VSFrame *VS_CC Eedi3GetFrame(
    int n, int activationReason, void *instanceData,
    [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core,
    const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    const int last = eedi3_batch_last(*d, n);

    if (activationReason == arInitial) {
        for (int j = n; j <= last; ++j) {
            const int sn = eedi3_clip_index(*d, j);
            vsapi->requestFrameFilter(sn, d->node, frameCtx);
            if (d->vcheck > 0 && d->sclip_node) {
                vsapi->requestFrameFilter(j, d->sclip_node, frameCtx);
            }
            if (d->mclip_node) {
                vsapi->requestFrameFilter(sn, d->mclip_node, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    if (VSFrame * hit = eedi3_take_cached(d, n)) {
        return hit;
    }
    if (!eedi3_claim_batch(d, n, last)) {
        return eedi3_take_cached(d, n);   // published while we waited
    }

    vsfeel_trace_frame_begin();
    vsfeel_trace_mark("frames");

    const int numPlanes = d->vi->format.numPlanes;
    const int out_height = (d->dh && !d->horiz) ? d->vi->height * 2 : d->vi->height;
    const int out_width = (d->dh && d->horiz) ? d->vi->width * 2 : d->vi->width;

    // Processed planes are freshly allocated; unprocessed ones may share the
    // source plane data when the dims match (not under dh, which doubles one
    // axis). newVideoFrame2/newGPUVideoFrame propagate GPU residency.
    bool all_processed = true;
    for (int plane = 0; plane < numPlanes; ++plane) {
        all_processed = all_processed && d->process[plane];
    }

    char errbuf[512] {};
    const bool timing = d->host_timing;
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto t0 = timing ? now() : std::chrono::steady_clock::time_point {};
    VSGPUExecContext * ctx = d->gpu->api->gpuExecAcquire(d->pool, errbuf, sizeof(errbuf));
    Eedi3FrameGuard guard { *d, ctx, false, n };
    std::vector<const VSFrame *> inputs;
    std::vector<std::pair<int, VSFrame *>> produced;
    std::vector<Eedi3Job> jobs;
    auto fail = [&](const std::string & message) -> const VSFrame * {
        for (auto & p : produced) {
            vsapi->freeFrame(p.second);
        }
        for (const VSFrame * f : inputs) {
            vsapi->freeFrame(f);
        }
        eedi3_release_claim(d, n, last);
        return eedi3_error(guard, frameCtx, vsapi, nullptr, "EEDI3VK", message);
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
    const auto t1 = timing ? now() : std::chrono::steady_clock::time_point {};

    for (int j = n; j <= last; ++j) {
        const int sn = eedi3_clip_index(*d, j);
        const VSFrame * src = vsapi->getFrameFilter(sn, d->node, frameCtx);
        const VSFrame * scp = (d->vcheck > 0 && d->sclip_node)
            ? vsapi->getFrameFilter(j, d->sclip_node, frameCtx) : nullptr;
        const VSFrame * mcp = d->mclip_node
            ? vsapi->getFrameFilter(sn, d->mclip_node, frameCtx) : nullptr;
        inputs.push_back(src);
        if (scp) {
            inputs.push_back(scp);
        }
        if (mcp) {
            inputs.push_back(mcp);
        }

        const int pl[] = { 0, 1, 2 };
        const VSFrame * fr[] = {
            (!d->dh && !d->process[0]) ? src : nullptr,
            (!d->dh && !d->process[1]) ? src : nullptr,
            (!d->dh && !d->process[2]) ? src : nullptr
        };
        VSFrame * dst = all_processed
            ? d->gpu->api->newGPUVideoFrame(&d->vi->format, out_width, out_height, src, core)
            : vsapi->newVideoFrame2(&d->vi->format, out_width, out_height, fr, pl, src, core);
        if (!dst) {
            return fail("failed to allocate an output frame");
        }

        int fb_err = 0;
        const int fieldBased = vsapi->mapGetIntSaturated(
            vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &fb_err);
        int field = d->field & 1;
        if (fieldBased == VSC_FIELD_BOTTOM) {
            field = 0;
        } else if (fieldBased == VSC_FIELD_TOP) {
            field = 1;
        }
        if (d->field > 1) {
            field = (j & 1) ^ field;
        }

        GpuBuffer scratch;
        if (const std::string e = gpu_make_buffer(*d->gpu, core, d->scratch_bytes, scratch,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); !e.empty()) {
            vsapi->freeFrame(dst);
            return fail("scratch allocation failed: " + e);
        }
        d->gpu->api->gpuExecUsesBuffer(ctx, scratch.handle);

        Eedi3Job job {};
        job.planes = d->planes.data();
        job.field = field;
        job.horiz = d->horiz;
        job.second = false;
        job.tail = d->horiz ? PassTail::kCompose : PassTail::kBlit;
        Eedi3PassInputs & in = job.in;
        in.scratch = scratch.buffer;
        std::string err;
        if (!fill_mask_sclip(*d, vsapi, scp, mcp, in.sclip, in.sclip_stride,
                             in.mask, in.mask_stride, err)) {
            vsapi->freeFrame(dst);
            return fail(err);
        }
        if (!fill_out_planes(*d, vsapi, dst, in.out, in.out_stride, err)) {
            vsapi->freeFrame(dst);
            return fail(err);
        }
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            VSVulkanPlaneInfo pi {};
            if (d->gpu->api->getGPUPlane(src, plane, &pi)) {
                vsapi->freeFrame(dst);
                return fail("clip plane " + std::to_string(plane) + " is not GPU resident");
            }
            in.src[plane] = pi.buffer;
            in.src_base[plane] = 0;
            in.src_stride[plane] = static_cast<int>(
                vsapi->getStride(src, plane) / d->elem_bytes);
        }
        jobs.push_back(job);
        produced.emplace_back(j, dst);
    }

    vsfeel_trace_mark("record");
    const int njobs = static_cast<int>(jobs.size());
    record_pass(*d, cmd, jobs.data(), njobs, PassPhase::kPrep);
    gpu_barrier(*d->gpu, cmd);
    record_pass(*d, cmd, jobs.data(), njobs, PassPhase::kRow);
    gpu_barrier(*d->gpu, cmd);
    record_pass(*d, cmd, jobs.data(), njobs, PassPhase::kVcheck);
    gpu_barrier(*d->gpu, cmd);
    record_pass(*d, cmd, jobs.data(), njobs, PassPhase::kTail);
    vsfeel_trace_mark("submit");
    const auto t3 = timing ? now() : std::chrono::steady_clock::time_point {};

    for (const VSFrame * f : inputs) {
        d->gpu->api->gpuExecReadsFrame(ctx, f);
    }
    for (auto & p : produced) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (d->process[plane]) {
                d->gpu->api->gpuExecWritesPlane(ctx, p.second, plane);
            }
        }
    }

    uint64_t signaled = 0;
    errbuf[0] = '\0';
    if (d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf, sizeof(errbuf))) {
        guard.ctx = nullptr;   // consumed either way
        return fail("submit failed: "s + errbuf);
    }
    guard.submitted = true;
    eedi3_add_timing(*d, signaled, t0, t1, t1, t3, now());

    for (const VSFrame * f : inputs) {
        vsapi->freeFrame(f);
    }

    for (auto & p : produced) {
        VSMap * props = vsapi->getFramePropertiesRW(p.second);
        vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
        if (d->field > 1) {
            int errNum, errDen;
            int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
            int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
            if (!errNum && !errDen) {
                vsh::muldivRational(&durationNum, &durationDen, 1, 2);
                vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
                vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
            }
        }
    }
    eedi3_publish(d, n, last, produced, vsapi);
    return eedi3_take_cached(d, n);
}

// EEDI3AA: the whole based_aa EEDI3 chain in one submission. The vertical
// stage runs the transposed-parity source twice and merges into the
// intermediate frame v in VRAM; the horizontal stage then runs twice on v, with
// the second compose merging the two planes straight into the output frame.
static const VSFrame *VS_CC Eedi3AaGetFrame(
    int n, int activationReason, void *instanceData,
    [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core,
    const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    const int last = eedi3_batch_last(*d, n);

    if (activationReason == arInitial) {
        for (int j = n; j <= last; ++j) {
            vsapi->requestFrameFilter(j, d->node, frameCtx);
            if (d->vcheck > 0 && d->sclip_node) {
                vsapi->requestFrameFilter(2 * j, d->sclip_node, frameCtx);
                vsapi->requestFrameFilter(2 * j + 1, d->sclip_node, frameCtx);
            }
            if (d->mclip_node) {
                vsapi->requestFrameFilter(j, d->mclip_node, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    if (VSFrame * hit = eedi3_take_cached(d, n)) {
        return hit;
    }
    if (!eedi3_claim_batch(d, n, last)) {
        return eedi3_take_cached(d, n);   // published while we waited
    }

    vsfeel_trace_frame_begin();
    vsfeel_trace_mark("frames");

    const int numPlanes = d->vi->format.numPlanes;
    bool all_processed = true;
    for (int plane = 0; plane < numPlanes; ++plane) {
        all_processed = all_processed && d->process[plane];
    }

    char errbuf[512] {};
    const bool timing = d->host_timing;
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto t0 = timing ? now() : std::chrono::steady_clock::time_point {};
    VSGPUExecContext * ctx = d->gpu->api->gpuExecAcquire(d->pool, errbuf, sizeof(errbuf));
    Eedi3FrameGuard guard { *d, ctx, false, n };
    std::vector<const VSFrame *> inputs;
    std::vector<std::pair<int, VSFrame *>> produced;
    // jobs[sub-pass][frame]: vertical fv0, vertical fv1, horizontal fh0, fh1.
    std::array<std::vector<Eedi3Job>, 4> jobs;
    auto fail = [&](const std::string & message) -> const VSFrame * {
        for (auto & p : produced) {
            vsapi->freeFrame(p.second);
        }
        for (const VSFrame * f : inputs) {
            vsapi->freeFrame(f);
        }
        eedi3_release_claim(d, n, last);
        return eedi3_error(guard, frameCtx, vsapi, nullptr, "EEDI3AA", message);
    };
    if (!ctx) {
        return fail("could not acquire a recording context: "s + errbuf);
    }
    VkCommandBuffer cmd = d->gpu->api->gpuExecCommandBuffer(ctx);
    const auto t1 = timing ? now() : std::chrono::steady_clock::time_point {};

    for (int j = n; j <= last; ++j) {
        const VSFrame * src = vsapi->getFrameFilter(j, d->node, frameCtx);
        const VSFrame * scp0 = (d->vcheck > 0 && d->sclip_node)
            ? vsapi->getFrameFilter(2 * j, d->sclip_node, frameCtx) : nullptr;
        const VSFrame * scp1 = (d->vcheck > 0 && d->sclip_node)
            ? vsapi->getFrameFilter(2 * j + 1, d->sclip_node, frameCtx) : nullptr;
        const VSFrame * mcp = d->mclip_node
            ? vsapi->getFrameFilter(j, d->mclip_node, frameCtx) : nullptr;
        inputs.push_back(src);
        if (scp0) {
            inputs.push_back(scp0);
        }
        if (scp1) {
            inputs.push_back(scp1);
        }
        if (mcp) {
            inputs.push_back(mcp);
        }

        const int pl[] = { 0, 1, 2 };
        const VSFrame * fr[] = {
            !d->process[0] ? src : nullptr,
            !d->process[1] ? src : nullptr,
            !d->process[2] ? src : nullptr
        };
        VSFrame * dst = all_processed
            ? d->gpu->api->newGPUVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core)
            : vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);
        if (!dst) {
            return fail("failed to allocate an output frame");
        }

        // The doubled stream's n=2k takes the input's _FieldBased (or field&1),
        // n=2k+1 its complement. The horizontal pass runs on v, an EEDI3 output
        // frame, which is always progressive, so it takes field&1 unmodified.
        int base = d->field & 1;
        int fb_err = 0;
        const int fieldBased = vsapi->mapGetIntSaturated(
            vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &fb_err);
        if (fieldBased == VSC_FIELD_BOTTOM) {
            base = 0;
        } else if (fieldBased == VSC_FIELD_TOP) {
            base = 1;
        }
        const int fv0 = base, fv1 = 1 - base;
        const int fh0 = d->field & 1, fh1 = 1 - fh0;

        GpuBuffer scratch;
        if (const std::string e = gpu_make_buffer(*d->gpu, core, d->scratch_bytes, scratch,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); !e.empty()) {
            vsapi->freeFrame(dst);
            return fail("scratch allocation failed: " + e);
        }
        d->gpu->api->gpuExecUsesBuffer(ctx, scratch.handle);

        VkBuffer mask {};
        int mask_stride = 0;
        if (mcp) {
            VSVulkanPlaneInfo pi {};
            if (d->gpu->api->getGPUPlane(mcp, 0, &pi)) {
                vsapi->freeFrame(dst);
                return fail("mclip is not GPU resident");
            }
            mask = pi.buffer;
            mask_stride = static_cast<int>(
                vsapi->getStride(mcp, 0) / (d->mclip_bits / 8));
        }
        auto sclip_planes = [&](const VSFrame * scf, VkBuffer * out, int * strides) {
            for (int plane = 0; plane < numPlanes; ++plane) {
                out[plane] = VK_NULL_HANDLE;
                strides[plane] = 0;
                if (!scf || !d->process[plane]) {
                    continue;
                }
                VSVulkanPlaneInfo pi {};
                if (d->gpu->api->getGPUPlane(scf, plane, &pi)) {
                    return false;
                }
                out[plane] = pi.buffer;
                strides[plane] = static_cast<int>(
                    vsapi->getStride(scf, plane) / d->elem_bytes);
            }
            return true;
        };
        VkBuffer sc0[MAX_PLANES] {}, sc1[MAX_PLANES] {};
        int sc0s[MAX_PLANES] {}, sc1s[MAX_PLANES] {};
        if (!sclip_planes(scp0, sc0, sc0s) || !sclip_planes(scp1, sc1, sc1s)) {
            vsapi->freeFrame(dst);
            return fail("sclip is not GPU resident");
        }

        std::string err;
        Eedi3Job v0 {}, v1 {}, h0 {}, h1 {};
        v0.planes = d->planes.data();
        v0.field = fv0;
        v0.tail = PassTail::kNone;
        v1.planes = d->planes.data();
        v1.field = fv1;
        v1.second = true;
        v1.tail = PassTail::kAssembleV;
        h0.planes = d->aplanes.data();
        h0.field = fh0;
        h0.horiz = true;
        h0.tail = PassTail::kCompose;
        h1.planes = d->aplanes.data();
        h1.field = fh1;
        h1.horiz = true;
        h1.second = true;
        h1.tail = PassTail::kCompose;

        for (auto * job : { &v0, &v1, &h0, &h1 }) {
            Eedi3PassInputs & in = job->in;
            in.scratch = scratch.buffer;
            in.mask = mask;
            in.mask_stride = mask_stride;
            for (int plane = 0; plane < numPlanes; ++plane) {
                in.out[plane] = VK_NULL_HANDLE;
                in.out_stride[plane] = 0;
            }
            const bool horiz = job->horiz;
            const VkBuffer * sc = job->second ? sc1 : sc0;
            const int * scs = job->second ? sc1s : sc0s;
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }
                in.sclip[plane] = sc[plane];
                in.sclip_stride[plane] = scs[plane];
            }
            if (horiz) {
                for (int plane = 0; plane < numPlanes; ++plane) {
                    if (!d->process[plane]) {
                        continue;
                    }
                    in.src[plane] = scratch.buffer;
                    in.src_base[plane] = static_cast<int>(
                        d->planes[plane].v_off / d->elem_bytes);
                    in.src_stride[plane] = d->planes[plane].out_w;
                    in.out[plane] = VK_NULL_HANDLE;
                    in.out_stride[plane] = 0;
                }
            } else {
                for (int plane = 0; plane < numPlanes; ++plane) {
                    if (!d->process[plane]) {
                        continue;
                    }
                    VSVulkanPlaneInfo pi {};
                    if (d->gpu->api->getGPUPlane(src, plane, &pi)) {
                        vsapi->freeFrame(dst);
                        return fail("clip plane " + std::to_string(plane) + " is not GPU resident");
                    }
                    in.src[plane] = pi.buffer;
                    in.src_base[plane] = 0;
                    in.src_stride[plane] = static_cast<int>(
                        vsapi->getStride(src, plane) / d->elem_bytes);
                }
            }
        }
        // Output planes (used by the horizontal compose tail).
        if (!fill_out_planes(*d, vsapi, dst, h0.in.out, h0.in.out_stride, err) ||
            !fill_out_planes(*d, vsapi, dst, h1.in.out, h1.in.out_stride, err)) {
            vsapi->freeFrame(dst);
            return fail(err);
        }

        jobs[0].push_back(v0);
        jobs[1].push_back(v1);
        jobs[2].push_back(h0);
        jobs[3].push_back(h1);
        produced.emplace_back(j, dst);
    }

    vsfeel_trace_mark("record");
    // One sub-pass group at a time: a frame's later stages read what its earlier
    // ones wrote, so the groups stay ordered, but within a group every frame's
    // dispatch runs back to back with no barrier between them.
    for (int sub = 0; sub < 4; ++sub) {
        auto & J = jobs[sub];
        if (J.empty()) {
            continue;
        }
        const int nj = static_cast<int>(J.size());
        record_pass(*d, cmd, J.data(), nj, PassPhase::kPrep);
        gpu_barrier(*d->gpu, cmd);
        record_pass(*d, cmd, J.data(), nj, PassPhase::kRow);
        gpu_barrier(*d->gpu, cmd);
        record_pass(*d, cmd, J.data(), nj, PassPhase::kVcheck);
        gpu_barrier(*d->gpu, cmd);
        record_pass(*d, cmd, J.data(), nj, PassPhase::kTail);
        if (sub < 3) {
            gpu_barrier(*d->gpu, cmd);
        }
    }
    vsfeel_trace_mark("submit");
    const auto t3 = timing ? now() : std::chrono::steady_clock::time_point {};

    for (const VSFrame * f : inputs) {
        d->gpu->api->gpuExecReadsFrame(ctx, f);
    }
    for (auto & p : produced) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (d->process[plane]) {
                d->gpu->api->gpuExecWritesPlane(ctx, p.second, plane);
            }
        }
    }

    uint64_t signaled = 0;
    errbuf[0] = '\0';
    if (d->gpu->api->gpuExecSubmit(ctx, &signaled, errbuf, sizeof(errbuf))) {
        guard.ctx = nullptr;
        return fail("submit failed: "s + errbuf);
    }
    guard.submitted = true;
    eedi3_add_timing(*d, signaled, t0, t1, t1, t3, now());

    for (const VSFrame * f : inputs) {
        vsapi->freeFrame(f);
    }
    for (auto & p : produced) {
        VSMap * props = vsapi->getFramePropertiesRW(p.second);
        vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
    }
    eedi3_publish(d, n, last, produced, vsapi);
    return eedi3_take_cached(d, n);
}

static void VS_CC Eedi3Free(void *instanceData, [[maybe_unused]] VSCore *core,
                            const VSAPI *vsapi) {
    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sclip_node);
    vsapi->freeNode(d->mclip_node);
    // The batch cache owns every frame no caller has taken yet.
    for (auto & entry : d->cache) {
        vsapi->freeFrame(entry.second);
    }
    d->cache.clear();
    delete d;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void vsfeel_eedi3_create(
    const VSMap *in, VSMap *out, [[maybe_unused]] void *userData,
    VSCore *core, const VSAPI *vsapi, bool horiz, bool aa) {

    auto d { std::make_unique<Eedi3Data>() };
    int err = 0;
    d->horiz = horiz;
    d->aa = aa;
    d->core = core;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    d->sclip_node = vsapi->mapGetNode(in, "sclip", 0, &err);
    bool has_sclip = d->sclip_node != nullptr;
    err = 0;
    d->mclip_node = vsapi->mapGetNode(in, "mclip", 0, &err);
    bool has_mclip = d->mclip_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
        vsfeel_trace_error("EEDI3VK", -1, error_message, d->gpu.get());
        vsapi->mapSetError(out, ("EEDI3VK: " + error_message).c_str());
        vsapi->freeNode(d->node);
        if (has_sclip) {
            vsapi->freeNode(d->sclip_node);
        }
        if (has_mclip) {
            vsapi->freeNode(d->mclip_node);
        }
    };

    if (auto [bps, sample] = std::pair{
            d->vi->format.bitsPerSample, d->vi->format.sampleType };
        !vsh::isConstantVideoFormat(d->vi) ||
        (sample == stInteger && bps != 16) ||
        (sample == stFloat && bps != 32)
    ) {
        return set_error("input bitdepth must be 16 (integer) or 32 (float).");
    }

    d->bits = d->vi->format.bitsPerSample;
    d->elem_bytes = d->bits / 8;

    d->field = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "field", 0, nullptr));
    d->dh = !!vsapi->mapGetInt(in, "dh", 0, &err);
    err = 0;

    const int m = vsapi->mapNumElements(in, "planes");
    for (int i = 0; i < MAX_PLANES; ++i) {
        d->process[i] = (m <= 0);
    }
    for (int i = 0; i < m; ++i) {
        const int n = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "planes", i, nullptr));
        if (n < 0 || n >= d->vi->format.numPlanes) {
            return set_error("plane index out of range");
        }
        if (d->process[n]) {
            return set_error("plane specified twice");
        }
        d->process[n] = true;
    }

    auto get_float = [&](const char * key, float def) {
        err = 0;
        const float v = static_cast<float>(vsapi->mapGetFloatSaturated(in, key, 0, &err));
        return err ? def : v;
    };
    auto get_int = [&](const char * key, int def) {
        err = 0;
        const int v = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, key, 0, &err));
        return err ? def : v;
    };

    d->alpha = get_float("alpha", 0.2f);
    d->beta = get_float("beta", 0.25f);
    d->gamma = get_float("gamma", 20.0f);
    d->nrad = get_int("nrad", 2);
    d->mdis = get_int("mdis", 20);
    d->vcheck = get_int("vcheck", 2);
    float vthresh0 = get_float("vthresh0", 32.0f);
    float vthresh1 = get_float("vthresh1", 64.0f);
    d->vthresh2 = get_float("vthresh2", 4.0f);

    // opt accepted for eedi3m parity but ignored (GPU is always AVX2-class)
    (void)get_int("opt", 0);

    if (d->field < 0 || d->field > 3) {
        return set_error("field must be 0, 1, 2, or 3");
    }
    if (d->aa) {
        if (d->field < 2) {
            return set_error("field must be 2 or 3 for EEDI3AA");
        }
        if (d->dh) {
            return set_error("dh is not supported by EEDI3AA");
        }
    }
    if (!d->dh) {
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            const int axis = d->horiz
                ? (d->vi->width >> (plane > 0 ? d->vi->format.subSamplingW : 0))
                : (d->vi->height >> (plane > 0 ? d->vi->format.subSamplingH : 0));
            if (d->process[plane] && (axis & 1)) {
                return set_error(d->horiz
                    ? "plane's width must be mod 2 when dh=False"
                    : "plane's height must be mod 2 when dh=False");
            }
        }
    }
    if (d->dh && d->field > 1) {
        return set_error("field must be 0 or 1 when dh=True");
    }
    if (d->alpha < 0.0f || d->alpha > 1.0f) {
        return set_error("alpha must be between 0.0 and 1.0 (inclusive)");
    }
    if (d->beta < 0.0f || d->beta > 1.0f) {
        return set_error("beta must be between 0.0 and 1.0 (inclusive)");
    }
    if (d->alpha + d->beta > 1.0f) {
        return set_error("alpha+beta must be between 0.0 and 1.0 (inclusive)");
    }
    if (d->gamma < 0.0f) {
        return set_error("gamma must be greater than or equal to 0.0");
    }
    if (d->nrad < 0 || d->nrad > 3) {
        return set_error("nrad must be between 0 and 3 (inclusive)");
    }
    if (d->mdis < 1 || d->mdis > 40) {
        return set_error("mdis must be between 1 and 40 (inclusive)");
    }
    if (d->vcheck < 0 || d->vcheck > 3) {
        return set_error("vcheck must be 0, 1, 2, or 3");
    }
    if (d->vcheck > 0 && (vthresh0 <= 0.0f || vthresh1 <= 0.0f || d->vthresh2 <= 0.0f)) {
        return set_error("vthresh0, vthresh1 and vthresh2 must be greater than 0.0");
    }

    // mclip must be a single Gray plane (vszip CPU semantic). Gray16 integer and
    // Gray32 float masks are consumed natively by the GPU predicate; other
    // depths keep the reference SetFrameProps(_Range=1) + resize.Point -> Gray8
    // conversion, which the core then uploads like any other CPU clip.
    if (d->mclip_node) {
        const auto mvi = vsapi->getVideoInfo(d->mclip_node);
        if (mvi->format.colorFamily != cfGray) {
            return set_error("mclip must be Gray");
        }
        if (mvi->width != d->vi->width || mvi->height != d->vi->height) {
            return set_error("mclip's dimensions don't match");
        }
        if (mvi->numFrames != d->vi->numFrames) {
            return set_error("mclip's number of frames doesn't match");
        }

        if (mvi->format.bitsPerSample == 16 && mvi->format.sampleType == stInteger) {
            d->mclip_native16 = true;
            d->mclip_bits = 16;
        } else if (mvi->format.bitsPerSample == 32 && mvi->format.sampleType == stFloat) {
            d->mclip_native32 = true;
            d->mclip_bits = 32;
        } else if (mvi->format.bitsPerSample != 8 || mvi->format.sampleType != stInteger) {
            VSMap * args = vsapi->createMap();
            vsapi->mapConsumeNode(args, "clip", d->mclip_node, maReplace);
            d->mclip_node = nullptr;  // ownership moved into args

            vsapi->mapSetInt(args, "_Range", 1, maReplace);
            VSMap * ret = vsapi->invoke(
                vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "SetFrameProps", args);
            if (vsapi->mapGetError(ret)) {
                vsfeel_trace_error("EEDI3VK", -1, vsapi->mapGetError(ret), d->gpu.get());
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                vsapi->freeNode(d->node);
                if (d->sclip_node) {
                    vsapi->freeNode(d->sclip_node);
                }
                return;
            }
            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->freeMap(ret);

            vsapi->mapSetInt(args, "format", vsapi->queryVideoFormatID(
                cfGray, stInteger, 8, 0, 0, core), maReplace);
            ret = vsapi->invoke(
                vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args);
            vsapi->freeMap(args);
            if (vsapi->mapGetError(ret)) {
                vsfeel_trace_error("EEDI3VK", -1, vsapi->mapGetError(ret), d->gpu.get());
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(ret);
                vsapi->freeNode(d->node);
                if (d->sclip_node) {
                    vsapi->freeNode(d->sclip_node);
                }
                return;
            }
            d->mclip_node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(ret);
            d->mclip_bits = 8;
        }
    }

    // sclip only validated when vcheck > 0 (eedi3m semantics). Under field > 1
    // the output doubles the frame count and under dh it doubles one axis, so
    // the sclip describes the OUTPUT.
    if (d->vcheck > 0 && d->sclip_node) {
        const auto svi = vsapi->getVideoInfo(d->sclip_node);
        VSVideoInfo out_vi = *d->vi;
        if (d->field > 1) {
            if (d->vi->numFrames > INT32_MAX / 2) {
                return set_error("resulting clip is too long");
            }
            if (d->vi->numFrames > 0) {
                out_vi.numFrames *= 2;
            }
        }
        if (d->dh) {
            if (d->horiz) {
                out_vi.width *= 2;
            } else {
                out_vi.height *= 2;
            }
        }
        if (!vsh::isSameVideoInfo(svi, &out_vi)) {
            return set_error("sclip's format and dimensions don't match");
        }
        if (svi->numFrames != out_vi.numFrames) {
            return set_error("sclip's number of frames doesn't match");
        }
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &err));
    if (err) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    // device_id is accepted for compatibility: the core owns the one device.
    (void)device_id;
    // num_streams is a registered no-op: in-flight depth is the core's
    // (exec pool ring) call, so the argument is accepted and never read.

    if (const char * vp = env_str("VSFEEL_EEDI3_VPARA")) {
        const int v = atoi(vp);
        d->vcheck_para = (v >= 0 && v <= Eedi3Data::VCHECK_PARA_LEVELS) ? v : 0;
    }
    const bool want_lds = env_flag("VSFEEL_EEDI3_VCLDS");
    d->trace = vsfeel_debug_flag("VSFEEL_EEDI3_TRACE");
    d->host_timing = vsfeel_debug_probe("VSFEEL_EEDI3_TIMING");
    d->sync_wait = vsfeel_debug_probe("VSFEEL_EEDI3_SYNC");
    d->host_timing = d->host_timing || d->sync_wait;

    {
        const auto result = get_gpu_device(core, vsapi);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->gpu = std::get<std::shared_ptr<GPUDevice>>(result);
    }

    {
        const auto layout = gpu_push_set_layout(*d->gpu, 8);
        if (std::holds_alternative<std::string>(layout)) {
            return set_error(std::get<std::string>(layout));
        }
        d->set_layout = std::get<VkDescriptorSetLayout>(layout);
        const auto pl = gpu_pipeline_layout(*d->gpu, d->set_layout,
                                            sizeof(Eedi3PushConstants));
        if (std::holds_alternative<std::string>(pl)) {
            return set_error(std::get<std::string>(pl));
        }
        d->pipeline_layout = std::get<VkPipelineLayout>(pl);
    }
    {
        char errbuf[512] {};
        d->pool = d->gpu->api->createGPUExecPool(core, vqCompute, errbuf, sizeof(errbuf));
        if (!d->pool) {
            return set_error("createGPUExecPool failed: "s + errbuf);
        }
    }

    // eedi3m scaling, with cost3 always on (GPU family semantics):
    //   remainingWeight = 1 - alpha - beta; alpha /= 3
    //   int:   beta/gamma/vthresh0/vthresh1 *= 2^(bits-8)
    //   float: beta/gamma/vthresh0/vthresh1 /= 255
    //   vthresh2 is never scaled; alpha is never /255.
    d->rw = 1.0f - d->alpha - d->beta;
    d->alpha /= 3.0f;
    if (d->vi->format.sampleType == stInteger) {
        const int scale = 1 << (d->bits - 8);
        d->beta *= static_cast<float>(scale);
        d->gamma *= static_cast<float>(scale);
        vthresh0 *= static_cast<float>(scale);
        vthresh1 *= static_cast<float>(scale);
    } else {
        d->beta /= 255.0f;
        d->gamma /= 255.0f;
        vthresh0 /= 255.0f;
        vthresh1 /= 255.0f;
    }
    d->rcp_vth0 = 1.0f / vthresh0;
    d->rcp_vth1 = 1.0f / vthresh1;
    d->rcp_vth2 = 1.0f / d->vthresh2;

    const auto & lim = d->gpu->limits;
    const int max_invoc = static_cast<int>(lim.maxComputeWorkGroupInvocations);
    const int max_x = static_cast<int>(lim.maxComputeWorkGroupSize[0]);
    const int lsz_vcheck = std::min({ 1024, max_invoc, max_x });
    if (!d->gpu->has_subgroup_size(SGSIZE)) {
        return set_error("device cannot run the EEDI3 row kernel (needs 32-lane "
                         "subgroups, natively or via subgroup size control)");
    }

    // Shader blobs for this io depth.
    switch (d->bits) {
        case 16:
            d->row_code = eedi3_16_row_spv; d->row_size = eedi3_16_row_spv_size;
            d->vcheck_code = eedi3_16_vcheck_spv; d->vcheck_size = eedi3_16_vcheck_spv_size;
            d->vcheck_lds_code = eedi3_16_vcheck_lds_spv;
            d->vcheck_lds_size = eedi3_16_vcheck_lds_spv_size;
            d->vcheck_para_code = { eedi3_16_vcheck_para_spv, eedi3_16_vcheck_para_j1_spv,
                eedi3_16_vcheck_para_j2_spv, eedi3_16_vcheck_para_j3_spv,
                eedi3_16_vcheck_para_j4_spv, eedi3_16_vcheck_para_j5_spv };
            d->vcheck_para_size = { eedi3_16_vcheck_para_spv_size,
                eedi3_16_vcheck_para_j1_spv_size, eedi3_16_vcheck_para_j2_spv_size,
                eedi3_16_vcheck_para_j3_spv_size, eedi3_16_vcheck_para_j4_spv_size,
                eedi3_16_vcheck_para_j5_spv_size };
            d->pad_code = eedi3_16_pad_spv; d->pad_size = eedi3_16_pad_spv_size;
            d->vcopy_code = eedi3_16_vcopy_spv; d->vcopy_size = eedi3_16_vcopy_spv_size;
            d->blit_code = eedi3_16_blit_spv; d->blit_size = eedi3_16_blit_spv_size;
            d->xpose_code = eedi3_16_xpose_spv; d->xpose_size = eedi3_16_xpose_spv_size;
            d->compose_code = eedi3_16_compose_spv; d->compose_size = eedi3_16_compose_spv_size;
            d->assemble_code = eedi3_16_assemblev_spv;
            d->assemble_size = eedi3_16_assemblev_spv_size;
            break;
        case 32:
            d->row_code = eedi3_32_row_spv; d->row_size = eedi3_32_row_spv_size;
            d->vcheck_code = eedi3_32_vcheck_spv; d->vcheck_size = eedi3_32_vcheck_spv_size;
            d->vcheck_lds_code = eedi3_32_vcheck_lds_spv;
            d->vcheck_lds_size = eedi3_32_vcheck_lds_spv_size;
            d->vcheck_para_code = { eedi3_32_vcheck_para_spv, eedi3_32_vcheck_para_j1_spv,
                eedi3_32_vcheck_para_j2_spv, eedi3_32_vcheck_para_j3_spv,
                eedi3_32_vcheck_para_j4_spv, eedi3_32_vcheck_para_j5_spv };
            d->vcheck_para_size = { eedi3_32_vcheck_para_spv_size,
                eedi3_32_vcheck_para_j1_spv_size, eedi3_32_vcheck_para_j2_spv_size,
                eedi3_32_vcheck_para_j3_spv_size, eedi3_32_vcheck_para_j4_spv_size,
                eedi3_32_vcheck_para_j5_spv_size };
            d->pad_code = eedi3_32_pad_spv; d->pad_size = eedi3_32_pad_spv_size;
            d->vcopy_code = eedi3_32_vcopy_spv; d->vcopy_size = eedi3_32_vcopy_spv_size;
            d->blit_code = eedi3_32_blit_spv; d->blit_size = eedi3_32_blit_spv_size;
            d->xpose_code = eedi3_32_xpose_spv; d->xpose_size = eedi3_32_xpose_spv_size;
            d->compose_code = eedi3_32_compose_spv; d->compose_size = eedi3_32_compose_spv_size;
            d->assemble_code = eedi3_32_assemblev_spv;
            d->assemble_size = eedi3_32_assemblev_spv_size;
            break;
        default:
            return set_error("unsupported bit depth");
    }
    d->maskpack_code = { eedi3_maskpack_8_spv, eedi3_maskpack_16_spv,
                         eedi3_maskpack_32_spv };
    d->maskpack_size = { eedi3_maskpack_8_spv_size, eedi3_maskpack_16_spv_size,
                         eedi3_maskpack_32_spv_size };
    d->maskdilate_raw_code = { eedi3_maskdilate_8_raw_spv, eedi3_maskdilate_16_raw_spv,
                               eedi3_maskdilate_32_raw_spv };
    d->maskdilate_raw_size = { eedi3_maskdilate_8_raw_spv_size,
                               eedi3_maskdilate_16_raw_spv_size,
                               eedi3_maskdilate_32_raw_spv_size };
    d->maskdilate_tr_code = { eedi3_maskdilate_8_tr_spv, eedi3_maskdilate_16_tr_spv,
                              eedi3_maskdilate_32_tr_spv };
    d->maskdilate_tr_size = { eedi3_maskdilate_8_tr_spv_size,
                              eedi3_maskdilate_16_tr_spv_size,
                              eedi3_maskdilate_32_tr_spv_size };
    d->have_lds = d->vcheck > 0 && d->vcheck_lds_code &&
        d->gpu->limits.maxComputeSharedMemorySize >= 2 * MAXW_LDS * sizeof(float);

    const int numPlanes = d->vi->format.numPlanes;
    const int subW = d->vi->format.subSamplingW;
    const int subH = d->vi->format.subSamplingH;

    // Output video info: field > 1 doubles the frame rate (not for the fused
    // EEDI3AA, which consumes the doubled stream internally), dh doubles the
    // interpolated axis (height vertically, width horizontally).
    VSVideoInfo out_video = *d->vi;
    if (d->field > 1 && !d->aa) {
        if (d->vi->numFrames > INT32_MAX / 2) {
            return set_error("resulting clip is too long");
        }
        if (d->vi->numFrames > 0) {
            out_video.numFrames *= 2;
        }
        vsh::muldivRational(&out_video.fpsNum, &out_video.fpsDen, 2, 1);
    }
    if (d->dh) {
        if (d->horiz) {
            out_video.width *= 2;
        } else {
            out_video.height *= 2;
        }
    }

    // Last valid output frame index for the batch range (unknown length means
    // -1, and then a batch is always one frame: requesting past the end fails).
    d->max_out = (out_video.numFrames > 0) ? out_video.numFrames - 1 : -1;

    const int tpitch = 2 * d->mdis + 1;
    const int pad_elem = pad_elem_bytes(d->bits);
    const int elem = d->elem_bytes;

    // EEDI3AA's horizontal geometry (the transpose of the vertical one) is
    // computed up front so the shared regions can be sized for the larger.
    if (d->aa) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const int in_w = (plane == 0) ? d->vi->width : d->vi->width >> subW;
            const int in_h = (plane == 0) ? d->vi->height : d->vi->height >> subH;
            auto & a = d->aplanes[plane];
            a.src_h = in_h;
            a.width = in_h;
            a.height = in_w;
            a.rows = in_w / 2;
            a.out_w = in_w;
            a.out_h = in_h;
            a.tpitch = tpitch;
            a.pad_stride = (a.width + MARGIN_H * 2 + 15) & ~15;
            a.pad_height = a.height + MARGIN_V * 2;
        }
    }
    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = d->planes[plane];
        const int in_w = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        const int in_h = (plane == 0) ? d->vi->height : d->vi->height >> subH;
        const int kw = d->horiz ? in_h : in_w;
        const int kh = d->horiz ? (d->dh ? 2 * in_w : in_w)
                                : (d->dh ? 2 * in_h : in_h);
        cfg.src_h = in_h;
        cfg.width = kw;
        cfg.height = kh;
        cfg.rows = kh / 2;
        cfg.out_w = d->horiz ? kh : kw;
        cfg.out_h = d->horiz ? kw : kh;
        cfg.tpitch = tpitch;
        cfg.pad_stride = (kw + MARGIN_H * 2 + 15) & ~15;
        cfg.pad_height = kh + MARGIN_V * 2;
    }

    // Region layout. The vertical and horizontal geometries of an AA instance
    // run in separate passes, so every region they share is placed once at the
    // larger of the two sizes.
    VkDeviceSize total = 0;
    auto place = [&](VkDeviceSize bytes) {
        const VkDeviceSize off = align32(total);
        total = align32(off + bytes);
        return off;
    };
    auto sz_io = [&](const Eedi3PlaneConfig & c) {
        return static_cast<VkDeviceSize>(c.rows) * c.width * elem;
    };
    auto sz_pbt = [](const Eedi3PlaneConfig & c) {
        // Must match the shader's pbt layout: with two directions per lane
        // (tpitch 33..64) the deltas pack to a nibble each, one byte per lane
        // pair -- 16 bytes per column -- and everything else stays int8.
        const VkDeviceSize stride =
            (c.tpitch > 32 && c.tpitch <= 64) ? 16 : static_cast<VkDeviceSize>(c.tpitch);
        return static_cast<VkDeviceSize>(c.rows) * c.width * stride;
    };
    auto sz_pad = [&](const Eedi3PlaneConfig & c) {
        return static_cast<VkDeviceSize>(c.pad_stride) * c.pad_height * pad_elem;
    };
    auto sz_bits = [&](const Eedi3PlaneConfig & c) {
        return static_cast<VkDeviceSize>((c.width + 31) / 32) * c.rows * 4;
    };
    auto sz_rt = [&](const Eedi3PlaneConfig & c) {
        return static_cast<VkDeviceSize>(c.rows) * c.width * pad_elem;
    };
    auto shared = [&](VkDeviceSize va, VkDeviceSize ha,
                      VkDeviceSize & voff, VkDeviceSize & vlen,
                      VkDeviceSize & hoff, VkDeviceSize & hlen) {
        const VkDeviceSize bytes = std::max(va, ha);
        const VkDeviceSize off = place(bytes);
        voff = off; vlen = bytes;
        hoff = off; hlen = bytes;
    };

    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & V = d->planes[plane];
        auto & H = d->aa ? d->aplanes[plane] : d->planes[plane];
        const bool has_io = true;
        shared(sz_pad(V), sz_pad(H), V.pad_off, V.pad_bytes, H.pad_off, H.pad_bytes);
        shared(sz_io(V), sz_io(H), V.dst_off, V.dst_bytes, H.dst_off, H.dst_bytes);
        shared(d->aa ? sz_io(V) : 0, d->aa ? sz_io(H) : 0,
               V.dst2_off, V.dst2_bytes, H.dst2_off, H.dst2_bytes);
        shared(sz_pbt(V), sz_pbt(H), V.pbt_off, V.pbt_bytes, H.pbt_off, H.pbt_bytes);
        shared(d->vcheck > 0 ? sz_io(V) : 0, d->vcheck > 0 ? sz_io(H) : 0,
               V.dmap_off, V.dmap_bytes, H.dmap_off, H.dmap_bytes);
        shared((d->vcheck > 0 && !d->sclip_node) ? sz_io(V) : 0,
               (d->vcheck > 0 && !d->sclip_node) ? sz_io(H) : 0,
               V.cint_off, V.cint_bytes, H.cint_off, H.cint_bytes);
        shared((d->vcheck > 0 || d->aa) ? sz_io(V) : 0,
               (d->vcheck > 0 || d->aa) ? sz_io(H) : 0,
               V.vout_off, V.vout_bytes, H.vout_off, H.vout_bytes);
        shared(d->aa ? sz_io(V) : 0, d->aa ? sz_io(H) : 0,
               V.vout2_off, V.vout2_bytes, H.vout2_off, H.vout2_bytes);
        shared(static_cast<VkDeviceSize>(V.rows), static_cast<VkDeviceSize>(H.rows),
               V.rempty_off, V.rempty_bytes, H.rempty_off, H.rempty_bytes);
        shared(d->mclip_node ? sz_bits(V) : 0, d->mclip_node ? sz_bits(H) : 0,
               V.bits_off, V.bits_bytes, H.bits_off, H.bits_bytes);
        shared(0, d->mclip_node ? sz_bits(H) : 0,
               V.pred_off, V.pred_bytes, H.pred_off, H.pred_bytes);
        shared(0, sz_rt(H), V.rt_off, V.rt_bytes, H.rt_off, H.rt_bytes);
        shared(0, (d->vcheck > 0 && d->sclip_node) ? sz_io(H) : 0,
               V.rtS_off, V.rtS_bytes, H.rtS_off, H.rtS_bytes);
        const VkDeviceSize plane_io = static_cast<VkDeviceSize>(V.out_w) *
            V.out_h * elem;
        // Without a vcheck the row kernel's dst IS the interp value, so point
        // the assembler's vout/vout2 regions at it (bit-exact: the elements are
        // the same, only the name differs).
        if (d->aa && d->vcheck == 0) {
            V.dst_off = V.vout_off;
            V.dst_bytes = V.vout_bytes;
            V.dst2_off = V.vout2_off;
            V.dst2_bytes = V.vout2_bytes;
            H.dst_off = H.vout_off;
            H.dst_bytes = H.vout_bytes;
            H.dst2_off = H.vout2_off;
            H.dst2_bytes = H.vout2_bytes;
        }
        shared(d->aa ? plane_io : 0,
               d->aa ? plane_io : 0,
               V.o0_off, V.o0_bytes, H.o0_off, H.o0_bytes);
        shared(d->aa ? plane_io : 0, d->aa ? plane_io : 0,
               V.v_off, V.v_bytes, H.v_off, H.v_bytes);
        (void)has_io;
    }
    d->scratch_bytes = std::max(total, VkDeviceSize(4));

    if (d->trace) {
        VSVulkanCoreInfo info {};
        char verr[256] {};
        if (d->gpu->api->getVulkanCoreInfo(core, &info, verr, sizeof(verr)) == 0) {
            fprintf(stderr, "[eedi3] vram allocated=%.0f budget=%.0f limit=%.0f MiB\n",
                static_cast<double>(info.allocated) / (1024.0 * 1024.0),
                static_cast<double>(info.budget) / (1024.0 * 1024.0),
                static_cast<double>(info.limit) / (1024.0 * 1024.0));
        }
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & c = d->planes[plane];
            fprintf(stderr,
                "[eedi3] plane %d w=%d rows=%d scratch=%.1f MiB pbt=%.1f MiB "
                "pad=%.1f MiB\n",
                plane, c.width, c.rows,
                static_cast<double>(d->scratch_bytes) / (1024.0 * 1024.0),
                static_cast<double>(c.pbt_bytes) / (1024.0 * 1024.0),
                static_cast<double>(c.pad_bytes) / (1024.0 * 1024.0));
        }
    }

    // Batch size. A submission's frames overlap only inside it (one compute
    // queue), so a bigger batch overlaps more -- but only up to a knee that
    // moves with the frame size. Swept on the graded 2x2160p EEDI3 workload
    // (masked, so most rows take the cheap cubic branch): with the packed pbt
    // B=2 387, B=4 381, B=6 323, B=8 277 fps, and the same sweep with a 3.7x
    // smaller scratch (mdis=5) still preferred 4 over 8, so the knee is not
    // memory -- it is VRAM-independent and the batch simply must not grow with
    // the frame. At 1080p, where one frame is 4x cheaper, B=8 wins (1273 vs
    // 892 fps). That is what the byte target below encodes: ~512 MiB of
    // scratch per submission lands on 4 at 2x2160p and 8 at 1080p, and never
    // below two frames so a submission can overlap at all.
    // VSFEEL_EEDI3_BATCH overrides.
    {
        VSVulkanCoreInfo info {};
        char verr[256] {};
        // EEDI3AA's four sub-passes make a frame ~4x heavier, so it drains a
        // batch slower and its knee is 4 where the single-stage filters' is 2
        // (swept on the graded workload: EEDI3 2->406/4->389, EEDI3AA 2->142
        // /4->157 fps at 2x2160p).
        VkDeviceSize target = VkDeviceSize(d->aa ? 512 : 256) << 20;
        if (d->gpu->api->getVulkanCoreInfo(core, &info, verr, sizeof(verr)) == 0 &&
            info.limit > 0) {
            target = std::min(target,
                              static_cast<VkDeviceSize>(info.limit) / 16);
        }
        target = std::max(target, VkDeviceSize(256) << 20);
        const int auto_batch = static_cast<int>(std::clamp<VkDeviceSize>(
            target / std::max<VkDeviceSize>(d->scratch_bytes, 1), 2, 8));
        d->batch_size = std::clamp(env_int("VSFEEL_EEDI3_BATCH", auto_batch), 1, 8);
    }

    // Pipelines, one set per distinct geometry.
    const bool mclip_on = d->mclip_node != nullptr;
    const bool sclip_on = d->vcheck > 0 && d->sclip_node != nullptr;
    const int fmt = (d->mclip_bits == 16) ? 1 : (d->mclip_bits == 32 ? 2 : 0);
    const bool para_ok = d->vcheck > 0 && d->vcheck_para >= 1 &&
        d->vcheck_para <= Eedi3Data::VCHECK_PARA_LEVELS &&
        d->vcheck_para_code[d->vcheck_para - 1] != nullptr;

    auto get_pipes = [&](const Eedi3Data::WidthKey & key,
                         Eedi3Pipelines & result) -> std::optional<std::string> {
        for (auto & [k, p] : d->width_pipes) {
            if (k == key) {
                result = p;
                return std::nullopt;
            }
        }
        Eedi3RowSpec spec {
            key.width, d->nrad, d->mdis, mclip_on ? 1 : 0, sclip_on ? 1 : 0,
            d->vcheck, SGSIZE, lsz_vcheck
        };
        if (d->trace) {
            fprintf(stderr, "[eedi3] spec w=%d nrad=%d mdis=%d mclip=%d sclip=%d "
                            "vcheck=%d lszr=%d lszv=%d horiz=%d batch=%d\n",
                    spec.width, spec.nrad, spec.mdis, spec.has_mclip,
                    spec.has_sclip, spec.vcheck, spec.lsz_row, spec.lsz_vcheck,
                    key.horiz ? 1 : 0, d->batch_size);
        }
        Eedi3Pipelines p;
        auto add = [&](const uint32_t * code, size_t size, const char * tag,
                       uint32_t subgroup, VkPipeline * dst) -> std::optional<std::string> {
            if (!code) {
                return std::nullopt;
            }
            auto r = gpu_create_pipeline(*d->gpu, code, size, d->pipeline_layout,
                row_entries.data(), &spec, static_cast<uint32_t>(row_entries.size()),
                sizeof(spec), tag, subgroup);
            if (std::holds_alternative<std::string>(r)) {
                return std::get<std::string>(r);
            }
            *dst = std::get<VkPipeline>(r);
            return std::nullopt;
        };
        if (auto e = add(d->row_code, d->row_size, "eedi3-row", SGSIZE, &p.row)) {
            return e;
        }
        p.vcheck_para = para_ok;
        p.vcheck_lds = !p.vcheck_para && want_lds && d->have_lds &&
            key.width <= MAXW_LDS;
        if (d->vcheck > 0) {
            const uint32_t * vc_code = p.vcheck_para
                ? d->vcheck_para_code[d->vcheck_para - 1]
                : (p.vcheck_lds ? d->vcheck_lds_code : d->vcheck_code);
            const size_t vc_size = p.vcheck_para
                ? d->vcheck_para_size[d->vcheck_para - 1]
                : (p.vcheck_lds ? d->vcheck_lds_size : d->vcheck_size);
            if (auto e = add(vc_code, vc_size, "eedi3-vcheck", 0, &p.vcheck)) {
                return e;
            }
            if (mclip_on && !p.vcheck_lds && !p.vcheck_para) {
                if (auto e = add(d->vcopy_code, d->vcopy_size, "eedi3-vcopy", 0,
                                 &p.vcopy)) {
                    return e;
                }
            }
        }
        if (auto e = add(d->pad_code, d->pad_size, "eedi3-pad", 0, &p.pad)) {
            return e;
        }
        if (auto e = add(d->blit_code, d->blit_size, "eedi3-blit", 0, &p.blit)) {
            return e;
        }
        if (auto e = add(d->xpose_code, d->xpose_size, "eedi3-xpose", 0, &p.xpose)) {
            return e;
        }
        if (auto e = add(d->compose_code, d->compose_size, "eedi3-compose", 0,
                         &p.compose)) {
            return e;
        }
        if (auto e = add(d->assemble_code, d->assemble_size, "eedi3-assemblev", 0,
                         &p.assemble)) {
            return e;
        }
        if (mclip_on) {
            if (auto e = add(d->maskpack_code[fmt], d->maskpack_size[fmt],
                             "eedi3-maskpack", 0, &p.maskpack)) {
                return e;
            }
            if (auto e = add(d->maskdilate_raw_code[fmt], d->maskdilate_raw_size[fmt],
                             "eedi3-maskdilate", 0, &p.maskdilate_raw)) {
                return e;
            }
            if (auto e = add(d->maskdilate_tr_code[fmt], d->maskdilate_tr_size[fmt],
                             "eedi3-maskdilate-tr", 0, &p.maskdilate_tr)) {
                return e;
            }
        }
        d->width_pipes.emplace_back(key, p);
        result = p;
        return std::nullopt;
    };

    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = d->planes[plane];
        Eedi3Data::WidthKey key { cfg.width, cfg.rows, cfg.tpitch, cfg.pad_stride,
                                  cfg.pad_height, d->horiz };
        if (auto e = get_pipes(key, cfg.pipes)) {
            return set_error(*e);
        }
        if (d->aa) {
            auto & a = d->aplanes[plane];
            Eedi3Data::WidthKey akey { a.width, a.rows, a.tpitch, a.pad_stride,
                                       a.pad_height, true };
            if (auto e = get_pipes(akey, a.pipes)) {
                return set_error(*e);
            }
        }
    }

    std::vector<VSFilterDependency> deps;
    const bool general = d->field > 1 && !d->aa;
    deps.push_back({ d->node, general ? rpGeneral : rpStrictSpatial });
    if (d->vcheck > 0 && d->sclip_node) {
        deps.push_back({ d->sclip_node, rpStrictSpatial });
    }
    if (d->mclip_node) {
        deps.push_back({ d->mclip_node, general ? rpGeneral : rpStrictSpatial });
    }

    Eedi3Data * data = d.release();
    // ffGPUOutput: the frames this filter returns live in VRAM and carry their
    // own producer pairs, so the core never downloads them for a consumer that
    // does not need host pixels.
    VSNode * result = vsapi->createVideoFilterEx2(
        aa ? "EEDI3AA" : (horiz ? "EEDI3H" : "EEDI3"), &out_video,
        aa ? Eedi3AaGetFrame : Eedi3GetFrame, Eedi3Free,
        fmParallel, ffGPUOutput, deps.data(), static_cast<int>(deps.size()), data, core);
    if (result == nullptr) {
        vsapi->mapSetError(out, "EEDI3: filter creation failed");
        return;
    }
    vsapi->mapConsumeNode(out, "clip", result, maAppend);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

static void VS_CC Eedi3Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {
    vsfeel_eedi3_create(in, out, userData, core, vsapi, false, false);
}

static void VS_CC Eedi3HCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {
    vsfeel_eedi3_create(in, out, userData, core, vsapi, true, false);
}

static void VS_CC Eedi3AaCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {
    vsfeel_eedi3_create(in, out, userData, core, vsapi, false, true);
}

void vsfeel_register_eedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    // Under the R80 GPU API every input and the output are GPU resident: the
    // core inserts the upload for a CPU clip and a GPUDownload for a CPU
    // consumer, so the filter itself never moves a frame.
    const char * eedi3_args =
        "clip:vnode:gpu;"
        "field:int;"
        "dh:int:opt;"
        "planes:int[]:opt;"
        "alpha:float:opt;"
        "beta:float:opt;"
        "gamma:float:opt;"
        "nrad:int:opt;"
        "mdis:int:opt;"
        "vcheck:int:opt;"
        "vthresh0:float:opt;"
        "vthresh1:float:opt;"
        "vthresh2:float:opt;"
        "sclip:vnode:gpu:opt;"
        "mclip:vnode:gpu:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;";
    vspapi->registerFunction(
        "EEDI3", eedi3_args, "clip:vnode:gpu;", Eedi3Create, nullptr, plugin);
    vspapi->registerFunction(
        "EEDI3H", eedi3_args, "clip:vnode:gpu;", Eedi3HCreate, nullptr, plugin);
    vspapi->registerFunction(
        "EEDI3AA", eedi3_args, "clip:vnode:gpu;", Eedi3AaCreate, nullptr, plugin);
}



