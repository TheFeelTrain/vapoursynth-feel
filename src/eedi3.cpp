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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <immintrin.h>

#include <vulkan/vulkan.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// EEDI3 — full-pel edge-directed line interpolation.
//
// Family-A semantics: reproduces HolyWu's eedi3m (the CPU ground truth) and
// agrees with eedi3vk2 (same family, GPU). This file follows the vsfeel host
// idiom (gaussblur/bilateral): a per-frame resource with a staging buffer for
// uploads/downloads and device-local buffers for the interleaved kernels.
//
// Supported formats: u16 integer and f32 float only (per the vsfeel scope).
// mclip is a SINGLE Gray plane driving every processed plane (vszip CPU
// semantic; non-Gray8 masks are converted to Gray8 internally). hp is not
// implemented (parameter absent).
//
// Kernel pipeline per plane:
//   ENTRY_ROW    one workgroup per interp row; DP + backtrack + interpolate.
//                Writes interp rows (rows = dstH/2, tight) to b1, pbt to b2,
//                dmap/cint to b3/b6 (vcheck only).
//   ENTRY_VCHECK single workgroup per plane; serial interp-row walk; writes
//                the vchecked interp rows back to b1.
// After the GPU, the host blits the interp rows from the download region into
// the dst frame at field+2r and copies the kept rows (parity 1-field) from
// src, exactly like eedi3vk2.
// ---------------------------------------------------------------------------

constexpr int MARGIN_H = 12;
constexpr int MARGIN_V = 4;
constexpr int MAX_PLANES = 3;

// Out-of-place transpose with 64x64 blocking (both sides stay cache-friendly:
// every tile-row segment is contiguous on read and write):
// dst[x * dstride + y] = src[y * sstride + x], x in [0,W), y in [0,H).
// Strides and dims are in ELEMENTS; elem is the byte size.
static void transpose_plane(const uint8_t * src, ptrdiff_t sstride_elems,
                            const int W, const int H,
                            uint8_t * dst, ptrdiff_t dstride_elems,
                            const int elem) {
    constexpr int TILE = 64;
    const size_t seg_bytes = static_cast<size_t>(TILE) * elem;
    for (int x0 = 0; x0 < W; x0 += TILE) {
        const int xn = std::min(TILE, W - x0);
        for (int y0 = 0; y0 < H; y0 += TILE) {
            const int yn = std::min(TILE, H - y0);
            for (int y = 0; y < yn; ++y) {
                const uint8_t * s = src + (static_cast<int64_t>(y0 + y) * sstride_elems + x0) * elem;
                uint8_t * d = dst + (static_cast<int64_t>(x0) * dstride_elems + y0 + y) * elem;
                if (xn == TILE) {
                    copy_stream_out(d, s, seg_bytes);
                } else {
                    std::memcpy(d, s, static_cast<size_t>(xn) * elem);
                }
            }
            (void)yn;
        }
    }
}

// Pad element size: native u16 for 16-bit input (halves the pad upload, the
// H2D copy and the kernel's pad read traffic vs float), float for 32-bit.
// u16 values are exact in float so all downstream math is bit-identical.
static int pad_elem_bytes(int bits) {
    return (bits == 16) ? 2 : 4;
}

// Copy dispatch for the per-frame host staging work. The NT load/store pair
// (copy_stream_*) bypasses the CPU cache, which is right for buffers the CPU
// touches exactly once — but under 8 concurrent streams the aggregate NT
// store bandwidth on this platform is what limits the frame, so the choice is
// a measured per-call-site trade (VSFEEL_EEDI3_COPY: bit0 = upload gathers,
// bit1 = the dst blit).
static inline void frame_copy_out(void * dst, const void * src, size_t bytes,
                                  const bool nt) {
    if (nt) {
        copy_stream_out(dst, src, bytes);
    } else {
        std::memcpy(dst, src, bytes);
    }
}

static inline void frame_copy_stream_read(void * dst, const void * src,
                                          size_t bytes, const bool nt) {
    if (nt) {
        copy_stream_read(dst, src, bytes);
    } else {
        std::memcpy(dst, src, bytes);
    }
}

// Host-side pad upload is always float (see eedi3.comp header): u16 native
// values are exact integers < 2^24 so float cost/interp math is bit-exact.
enum Binding : uint32_t {
    B_PAD = 0,
    B_DST = 1,
    B_PBT = 2,
    B_DMAP = 3,
    B_BMASK = 4,
    B_SCLIP = 5,
    B_CINT = 6,
    B_VOUT = 7,       // vcheck output rows (io type); vcheck reads taps from
                      // B_DST (untouched) and writes the vchecked row here
    B_RAW = 8,        // ENTRY_PAD's extra view of pad_dev (built pad out)
    BIND_COUNT = 9
};

struct PlaneConfig {
    int width {};                     // output plane width (pixels)
    int height {};                    // output plane height (pixels)
    int rows {};                      // number of interp rows == height / 2
    int pad_stride {};                // pad elements per padded row
    int pad_height {};                // padded rows == height + 2*MARGIN_V
    int tpitch {};                    // 2*mdis + 1

    VkPipeline row_pipeline {};
    VkPipeline vcheck_pipeline {};    // only when vcheck > 0
    VkPipeline pad_pipeline {};       // mirror-pad builder (always)
    VkPipeline vcopy_pipeline {};     // empty-row vcheck copy (vcheck+mclip)

    // staging (host-visible) regions, byte offsets: tight kept source rows,
    // gathered sclip rows (sclip only). The shared tight mask rows live in
    // Eedi3Data (maskraw_offset/bytes).
    VkDeviceSize raw_offset {};       // kept src rows (upload)
    VkDeviceSize raw_bytes {};
    VkDeviceSize sclip_offset {};     // sclip interp rows (upload; sclip only)
    VkDeviceSize sclip_bytes {};
    VkDeviceSize dl_offset {};        // interp row download (device -> host)
    VkDeviceSize dl_bytes {};

    // device-mirror (pad_dev) regions: 1:1 copy of the upload [0,upload_total)
    // plus kernel-built regions (never staged)
    VkDeviceSize built_offset {};     // built padded plane (pad kernel out)
    VkDeviceSize built_bytes {};
    VkDeviceSize bits_offset {};      // packed dilation bits (bmask kernel out)
    VkDeviceSize bits_bytes {};

    // device-local buffer regions, byte offsets (offsets into d->dev_buf)
    VkDeviceSize dst_offset {};       // interp rows (io type)
    VkDeviceSize dst_bytes {};
    VkDeviceSize rempty_offset {};    // per-row empty flags, int8 (always)
    VkDeviceSize rempty_bytes {};
    VkDeviceSize pbt_offset {};       // int8 rows*width*tpitch
    VkDeviceSize pbt_bytes {};
    VkDeviceSize dmap_offset {};      // int8 rows*width (vcheck only)
    VkDeviceSize dmap_bytes {};
    VkDeviceSize cint_offset {};      // io rows*width (vcheck && !sclip)
    VkDeviceSize cint_bytes {};
    VkDeviceSize vout_offset {};      // io rows*width (vcheck output; dev)
    VkDeviceSize vout_bytes {};
};

struct Eedi3Resource {
    VkBuffer staging {};
    VkDeviceMemory staging_mem {};
    VkBuffer pad_dev {};          // device-local built pads (pad kernel out)
    VkDeviceMemory pad_dev_mem {};
    VkBuffer up_dev {};           // ReBAR upload: CPU NT-stores land in VRAM
    VkDeviceMemory up_dev_mem {};
    uint8_t * up_map {};          // mapped view of up_dev
    uint32_t up_type_index {};
    VkDescriptorSet desc_set_pad {};   // pad kernel (b0 = up_dev, b8 = pad_dev)
    VkBuffer dev_buf {};          // device-local kernels' buffers
    VkDeviceMemory dev_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    float * map {};
    uint32_t staging_type_index {};
    uint32_t dev_type_index {};
};

struct Eedi3Data {
    VSNode * node {};
    VSNode * sclip_node {};
    VSNode * mclip_node {};
    const VSVideoInfo * vi {};

    int device_id {}, num_streams {};
    int bits {}, elem_bytes {};
    bool process[MAX_PLANES] { true, true, true };

    int field {}, nrad { 2 }, mdis { 20 }, vcheck { 2 };
    bool dh {};
    // Which copies use non-temporal load/store (bit0 = upload gathers, bit1 =
    // the final blit). Swept with VSFEEL_EEDI3_COPY; the default is measured.
    int copy_mode { 3 };
    // Diagnostics-only host-path ablations (env-gated; all default off).
    bool skip_blit {}, skip_sclip {}, skip_raw {}, skip_h2d {}, skip_vcheck {};
    bool raw_stage {};   // diagnostics: raw gather -> cached staging (plain stores)
    bool blit_contig {}; // diagnostics: one contiguous copy instead of per-row
    // Gray16 mclip handled natively (no SetFrameProps+resize.Point->Gray8 node).
    bool mclip_native16 {};
    bool skip_pad {};    // diagnostics: skip the pad-builder dispatch
    // ReBAR direct upload: the CPU NT-stores the upload straight into
    // host-visible VRAM (nnedi3's proven path) instead of writing system-RAM
    // staging and DMA'ing it. VSFEEL_EEDI3_NOREBAR=1 forces the old path.
    bool rebar_up { true };  // VSFEEL_EEDI3_NOREBAR=1 forces the H2D path
    // Download structure. false (default): the vcheck/vcopy kernels write the
    // vout rows straight into the host staging (no D2H copy, but every GPU
    // store targets system RAM and must snoop the CPU caches). true: vout goes
    // to a device-local region and one vkCmdCopyBuffer (SDMA) brings it back.
    // VSFEEL_EEDI3_VOUTDEV=1 selects the DMA form.
    bool vout_dev {};
    float alpha { 0.2f }, beta { 0.25f }, gamma { 20.0f };
    float vthresh2 { 4.0f };
    float rw {}, rcp_vth0 {}, rcp_vth1 {}, rcp_vth2 {};
    int peak {};

    // one pipeline set per distinct plane width (like eedi3vk2); pipelines
    // for the same width are shared by all planes of that width
    struct WidthKey {
        int width, rows, tpitch, pad_stride, pad_height;
        int has_mclip, has_sclip, vcheck;
        bool operator==(const WidthKey & o) const {
            return width == o.width && rows == o.rows && tpitch == o.tpitch &&
                   pad_stride == o.pad_stride && pad_height == o.pad_height &&
                   has_mclip == o.has_mclip && has_sclip == o.has_sclip &&
                   vcheck == o.vcheck;
        }
    };

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule row_module {};
    VkShaderModule vcheck_module {};
    VkShaderModule pad_module {};
    VkShaderModule vcopy_module {};
    VkDeviceSize upload_total {}, download_total {}, dev_total {};
    std::array<PlaneConfig, MAX_PLANES> planes {};
    FramePool<Eedi3Resource> pool;

    // module/pipeline cache keyed by width (values owned by the pipelines
    // map): [row, vcheck, pad]
    std::vector<std::pair<WidthKey, std::array<VkPipeline, 4>>> width_pipes {};

    ~Eedi3Data() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);

        for (auto & resource : pool.items) {
            if (resource.map) {
                vkUnmapMemory(dev, resource.staging_mem);
            }
            if (resource.pad_dev_mem) {
                vkFreeMemory(dev, resource.pad_dev_mem, nullptr);
            }
            if (resource.pad_dev) {
                vkDestroyBuffer(dev, resource.pad_dev, nullptr);
            }
            if (resource.up_map) {
                vkUnmapMemory(dev, resource.up_dev_mem);
            }
            if (resource.up_dev_mem) {
                vkFreeMemory(dev, resource.up_dev_mem, nullptr);
            }
            if (resource.up_dev) {
                vkDestroyBuffer(dev, resource.up_dev, nullptr);
            }
            if (resource.dev_mem) {
                vkFreeMemory(dev, resource.dev_mem, nullptr);
            }
            if (resource.dev_buf) {
                vkDestroyBuffer(dev, resource.dev_buf, nullptr);
            }
            destroy_common(dev, resource);
        }

        VkPipeline destroyed[4 * 3] {};
        int nd = 0;
        for (auto & [key, quad] : width_pipes) {
            for (int i = 0; i < 4; ++i) {
                if (!quad[i]) {
                    continue;
                }
                bool seen = false;
                for (int j = 0; j < nd; ++j) {
                    if (destroyed[j] == quad[i]) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    destroyed[nd++] = quad[i];
                    vkDestroyPipeline(dev, quad[i], nullptr);
                }
            }
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
        if (row_module) {
            vkDestroyShaderModule(dev, row_module, nullptr);
        }
        if (vcheck_module) {
            vkDestroyShaderModule(dev, vcheck_module, nullptr);
        }
        if (pad_module) {
            vkDestroyShaderModule(dev, pad_module, nullptr);
        }
        if (vcopy_module) {
            vkDestroyShaderModule(dev, vcopy_module, nullptr);
        }

        release_device(device);
    }
};

// ---------------------------------------------------------------------------
// CPU-side staging (tight kept-row / sclip gathers live inline in GetFrame;
// the GPU pad kernel expands mirrors, replicating eedi3m's copyPad exactly)
// ---------------------------------------------------------------------------

// Row dilation + bit packing: out[x>>5] bit (x&31) is set iff any mask byte in
// [x-mdis, x+mdis] is non-zero (bytes outside [0,width) count as zero) — the
// same window-OR the scalar last-propagation scan below computes, proved
// equivalent incl. edges when the GPU bmask kernel was prototyped (notes 8.2).
// Writes PACKED BITS (one uint32 per 32 columns): 8x fewer bytes than the byte
// mask (smaller H2D + the shader loads words instead of packing).
//
// Why bits and not the scan: bits are a *dilation by a box*, which composes
// (dil_a ∘ dil_b == dil_(a+b)) and therefore collapses to a handful of
// whole-array shift-ORs. The scan's serial `last` dependency is exactly what
// made the scalar version ~6 ms/frame at 3840x1080 (microbenchmark) — i.e. the
// single largest CPU cost in the whole filter.
//
// Native 16-bit mask variant: the host used to force the mask through a
// SetFrameProps(_Range=1) + resize.Point -> Gray8 node and then pack "byte != 0".
// That conversion is a whole extra full-frame pass in the graph (16.6 MB read +
// 8.3 MB write at the bench geometry) and it is NOT needed to recover the same
// boolean: zimg's full-range 16->8 bit reduction is monotonic and
// round(v * 255 / 65535), so "converted != 0" is exactly "v >= 129" — verified
// exhaustively over all 65536 u16 values (tmp/thresh_full.py, 0 mismatches).
// Reading the u16 mask directly skips the conversion node entirely and halves
// the mask read; only the needed (kept-parity) rows are touched.
//
// scratch must hold 2 * ((width + mdis + 63) / 64) uint64 words.
static void build_bmask_row_scalar(const uint8_t * maskp, const uint16_t * mask16,
                                   uint32_t * out, const int width, const int mdis) {
    // Native u16 masks use the exact bit-depth-reduction threshold (see
    // build_bmask_row): nonzero-after-conversion == (v >= 129).
    auto nz = [&](int x) {
        return mask16 ? (mask16[x] >= 129u) : (maskp[x] != 0);
    };
    const int minmdis = std::min(width, mdis);
    int last = -666999;

    for (int x = 0; x < minmdis; ++x) {
        if (nz(x)) {
            last = x + mdis;
        }
    }
    for (int x = 0; x < width - minmdis; ++x) {
        if (nz(x + mdis)) {
            last = x + mdis * 2;
        }
        if (x <= last) {
            out[x >> 5] |= 1u << (x & 31);
        }
    }
    for (int x = std::max(width - minmdis, 0); x < width; ++x) {
        if (x <= last) {
            out[x >> 5] |= 1u << (x & 31);
        }
    }
}

// 16 u16 lanes -> 16 packed bits (bit k = mask[x+k] >= 129). Biasing by 0x8000
// turns the unsigned compare into a signed one. cmpgt is STRICT, so the
// constant must be 128-32768: biased > -32640 <=> v-32768 > 128-32768 <=> v > 128
// <=> v >= 129. (Writing 129-32768 would silently test v >= 130 — a bug that
// only shows up on masks whose values straddle the threshold.) The movemask
// then has two identical bits per lane; the compress folds the even ones to 16.
static inline uint32_t bmask_bits16(const __m256i v) {
    const __m256i biased = _mm256_xor_si256(
        v, _mm256_set1_epi16(static_cast<short>(0x8000)));
    const __m256i cmp = _mm256_cmpgt_epi16(
        biased, _mm256_set1_epi16(static_cast<short>(128 - 32768)));
    uint32_t m = static_cast<uint32_t>(_mm256_movemask_epi8(cmp)) & 0x55555555u;
    m = (m | (m >> 1)) & 0x33333333u;
    m = (m | (m >> 2)) & 0x0F0F0F0Fu;
    m = (m | (m >> 4)) & 0x00FF00FFu;
    m = (m | (m >> 8)) & 0x0000FFFFu;
    return m;
}

static void build_bmask_row(const uint8_t * maskp, const uint16_t * mask16,
                            uint32_t * out,
                            const int width, const int mdis,
                            uint64_t * scratch) {
    // The shift/dilate form below equals the scalar scan only when the scan's
    // two coverage loops are contiguous, i.e. width >= 2*mdis (validation caps
    // mdis at 40). Narrower rows keep the exact legacy scalar path; so does
    // mdis >= 64, which the shift form cannot express.
    if (mdis >= 64 || width < 2 * mdis) {
        const int nwords = (width + 31) / 32;
        std::memset(out, 0, static_cast<size_t>(nwords) * sizeof(uint32_t));
        build_bmask_row_scalar(maskp, mask16, out, width, mdis);
        return;
    }

    // The accumulator must span [0, width-1+mdis]: B = b << mdis puts set bits
    // up to width-1+mdis, and the dilation reads B at x+t for x < width and
    // t <= 2*mdis only through those (bits at p > width-1+mdis map to mask
    // bytes past the row, which are zero) -- but the SHIFT itself needs the
    // extra words to exist, and the dilation needs them to hold B's high bits.
    const int nw_b = (width + 63) / 64;                  // packed mask words
    const int nw = (width + mdis + 63) / 64;             // accumulator words

    // 1. pack: bit x = mask non-zero; words past nw_b stay zero
    std::memset(scratch, 0, static_cast<size_t>(nw) * sizeof(uint64_t));
    const __m256i zero = _mm256_setzero_si256();
    if (mask16) {
        for (int i = 0; i < nw_b; ++i) {
            const int x = i * 64;
            uint64_t w = 0;
            for (int q = 0; q < 4; ++q) {
                const int xb = x + q * 16;
                uint32_t m = 0;
                if (xb + 16 <= width) {
                    m = bmask_bits16(_mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(mask16 + xb)));
                } else if (xb < width) {
                    uint16_t tmp[16] = {};
                    std::memcpy(tmp, mask16 + xb,
                                static_cast<size_t>(width - xb) * sizeof(uint16_t));
                    m = bmask_bits16(_mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(tmp)));
                }
                w |= static_cast<uint64_t>(m) << (q * 16);
            }
            scratch[i] = w;
        }
    } else {
        for (int i = 0; i < nw_b; ++i) {
            const int x = i * 64;
            uint64_t w = 0;
            for (int h = 0; h < 2; ++h) {
                const int xb = x + h * 32;
                uint32_t m = 0;
                if (xb + 32 <= width) {
                    const __m256i v = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(maskp + xb));
                    m = ~static_cast<uint32_t>(_mm256_movemask_epi8(
                        _mm256_cmpeq_epi8(v, zero)));
                } else if (xb < width) {
                    uint8_t tmp[32] = {};
                    std::memcpy(tmp, maskp + xb, static_cast<size_t>(width - xb));
                    const __m256i v = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(tmp));
                    m = ~static_cast<uint32_t>(_mm256_movemask_epi8(
                        _mm256_cmpeq_epi8(v, zero)));
                }
                w |= static_cast<uint64_t>(m) << (h * 32);
            }
            scratch[i] = w;
        }
    }

    // 2. B[x] = b[x - mdis]: shift the bit array toward higher x by mdis
    //    (descending in place: src[i-1] is still untouched when i is written).
    {
        const int k = mdis;
        for (int i = nw - 1; i >= 0; --i) {
            const uint64_t hi = scratch[i] << k;
            const uint64_t lo = (i > 0) ? (scratch[i - 1] >> (64 - k)) : 0;
            scratch[i] = hi | lo;
        }
    }

    // 3. out[x] = OR_{t=0}^{2*mdis} B[x+t]. In bit-index terms this is
    //    `acc >> t` ((a>>t)[p] == a[p+t]), so each B bit at p covers out bits
    //    [p-2mdis, p]. Compose by doubling: if acc == dil_r then
    //    acc | (acc >> s) == dil_{r+s} for any s <= r+1, so 2*mdis is reached
    //    in O(log mdis) whole-array passes instead of 2*mdis passes.
    if (mdis > 0) {
        const int r = 2 * mdis;
        uint64_t * const shifted = scratch + nw;   // caller-provided second half
        int radius = 0;
        for (int step = 1; radius < r; step <<= 1) {
            const int s = std::min(step, r - radius);
            for (int i = 0; i < nw; ++i) {
                const uint64_t hi = scratch[i] >> s;
                const uint64_t lo = (i + 1 < nw)
                    ? (scratch[i + 1] << (64 - s)) : 0;
                shifted[i] = hi | lo;
            }
            for (int i = 0; i < nw; ++i) {
                scratch[i] |= shifted[i];
            }
            radius += s;
        }
    }

    // 4. store as packed uint32 words (row stride is (width+31)/32 words);
    //    the bits past `width` in the last word are cleared (the shader never
    //    reads them, but the upload stays byte-reproducible).
    const int nwords = (width + 31) / 32;
    for (int i = 0; i < nwords; ++i) {
        out[i] = static_cast<uint32_t>(scratch[i >> 1] >> ((i & 1) * 32));
    }
    const int tail = width & 31;
    if (tail != 0) {
        out[nwords - 1] &= (1u << tail) - 1u;
    }
}

// ---------------------------------------------------------------------------
// Shader module + pipeline helpers (vsfeel idiom)
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

struct RowSpecData {
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

static std::variant<VkPipeline, std::string> create_pipeline(
    const VK_Device & dev, const RowSpecData & spec, VkShaderModule module,
    VkPipelineLayout layout, uint32_t required_subgroup_size = 0) {

    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(row_entries.size()),
        .pMapEntries = row_entries.data(),
        .dataSize = sizeof(spec),
        .pData = &spec
    };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = required_subgroup_size
    };

    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = required_subgroup_size ? &subgroup_size_info : nullptr,
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

// Push constant layout must match the shader's PC struct (int block then
// float block; see eedi3.comp).
struct PushConstants {
    int32_t pad_base;       // pad-elem base of the BUILT pad (b0 reads it)
    int32_t dst_base;
    int32_t pbt_base;
    int32_t dmap_base;
    int32_t bmask_base;     // word base of the packed bits (b4)
    int32_t sclip_base;
    int32_t cint_base;
    int32_t vout_base;    // vcheck output rows (element base into dev_buf)
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
    // upload-kernel field (ENTRY_PAD; ignored by row/vcheck).
    // NOTE: ENTRY_PAD writes the built pad through b8 at pc.pad_base (same
    // numeric base as the row kernel's b0 reads — same region, same buffer).
    int32_t raw_base;       // pad-elem base of tight kept rows (H2D mirror)
    int32_t rempty_base;    // int8 base of per-row empty flags (dev_buf via b3)
};
static_assert(sizeof(PushConstants) == 14 * 4 + 8 * 4, "push constants size");

// base offsets in ELEMENTS for each binding of a plane's regions (element
// type per binding; the descriptors range the whole buffer so the shader
// indexes base + offset).
struct PlaneBases {
    int32_t pad;       // float elements into staging (pad region)
    int32_t dst;       // io elements into dev_buf (interp rows)
    int32_t pbt;       // int8 elements into dev_buf
    int32_t dmap;      // int8 elements into dev_buf
    int32_t bmask;     // uint8 elements into staging
    int32_t sclip;     // io elements into staging
    int32_t cint;      // io elements into dev_buf
};

static std::optional<std::string> record_command_buffer(
    const Eedi3Data & d, Eedi3Resource & resource, const int field) {

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

    const int32_t elem = d.elem_bytes;

    // Upload: with the ReBAR path the CPU has ALREADY NT-stored the upload
    // region straight into host-visible VRAM (resource.up_dev), so there is
    // nothing to copy here and no transfer->compute barrier to pay -- the
    // kernels read up_dev directly. The host-side stores are drained by an
    // _mm_sfence() before submit. VSFEEL_EEDI3_NOREBAR=1 restores the old
    // staging -> pad_dev DMA.
    if (d.upload_total > 0 && !d.skip_h2d && !d.rebar_up) {
        const VkBufferCopy region {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = d.upload_total
        };
        vkCmdCopyBuffer(resource.cmd, resource.staging, resource.pad_dev, 1, &region);

        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
    }

    // Upload kernel: build the padded planes from the raw upload (per plane).
    // The command buffer is recorded per frame because the interp-row parity
    // `field` varies (field > 1 doubles frames and _FieldBased sources).
    (void)field;
    const int32_t pad_elem = pad_elem_bytes(d.bits);
    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane] || d.skip_pad) {
            continue;
        }
        const auto & cfg = d.planes[plane];

        PushConstants ppc {};
        ppc.pad_base = static_cast<int32_t>(cfg.built_offset / pad_elem);
        ppc.pad_stride = cfg.pad_stride;
        ppc.pad_height = cfg.pad_height;
        ppc.field = field;
        ppc.rows = cfg.rows;
        ppc.raw_base = static_cast<int32_t>(cfg.raw_offset / pad_elem);

        // pad builder: one thread per padded element
        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pad_pipeline);
        // The pad kernel reads the raw upload at binding 0; with the ReBAR
        // path that is up_dev and needs its own set (the row kernel reads the
        // built pad from pad_dev at the same binding index).
        VkDescriptorSet const pad_set = resource.desc_set_pad
            ? resource.desc_set_pad : resource.desc_set;
        vkCmdBindDescriptorSets(
            resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &pad_set, 0, nullptr);
        vkCmdPushConstants(
            resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(ppc), &ppc);
        const uint32_t pad_total = static_cast<uint32_t>(cfg.pad_stride) *
            static_cast<uint32_t>(cfg.pad_height);
        vkCmdDispatch(resource.cmd, (pad_total + 255) / 256, 1, 1);
    }
    {
        // the pad builder's writes must be visible to the row kernel's reads
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
    }

    // The row kernel reads the built pad / packed bmask / sclip and writes
    // dst / pbt / dmap / cint to dev_buf; a D2H copy then brings the interp
    // rows back into the staging download area.
    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
            continue;
        }
        const auto & cfg = d.planes[plane];
        const int has_mclip = d.mclip_node ? 1 : 0;
        const int has_sclip = (d.vcheck > 0 && d.sclip_node) ? 1 : 0;

        PushConstants pc {
            .pad_base = static_cast<int32_t>(cfg.built_offset / pad_elem_bytes(d.bits)),
            .dst_base = static_cast<int32_t>(cfg.dst_offset / elem),
            .pbt_base = static_cast<int32_t>(cfg.pbt_offset),
            .dmap_base = static_cast<int32_t>(cfg.dmap_offset),
            .bmask_base = static_cast<int32_t>(cfg.bits_offset / sizeof(uint32_t)),
            .sclip_base = static_cast<int32_t>(cfg.sclip_offset / elem),
            .cint_base = static_cast<int32_t>(cfg.cint_offset / elem),
            // vout either lands DIRECTLY in the staging download region (b7
            // views staging: no D2H copy, but GPU stores go to system RAM) or
            // in a device-local region that one SDMA copy brings back.
            .vout_base = static_cast<int32_t>(d.vout_dev
                ? (cfg.vout_offset / elem)
                : ((d.upload_total + cfg.dl_offset) / elem)),
            .pad_stride = cfg.pad_stride,
            .pad_height = cfg.pad_height,
            .field = field,
            .rows = cfg.rows,
            .alpha = d.alpha,
            .beta = d.beta,
            .gamma = d.gamma,
            .rw = d.rw,
            .vth0r = d.rcp_vth0,
            .vth1r = d.rcp_vth1,
            .vth2r = d.rcp_vth2,
            .vth2 = d.vthresh2,
            .raw_base = 0,
            .rempty_base = static_cast<int32_t>(cfg.rempty_offset),
        };

        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.row_pipeline);
        vkCmdBindDescriptorSets(
            resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
        vkCmdPushConstants(
            resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);
        vkCmdDispatch(resource.cmd, 1, static_cast<uint32_t>(cfg.rows), 1);

        if (d.vcheck > 0 && !d.skip_vcheck) {
            // the vcheck passes read the row kernel's writes (dst/dmap/cint
            // rempty flags)
            VkMemoryBarrier mem_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

            // empty-row fast pass (mclip only; without a mask no row is
            // empty and the walk below handles everything): fully-masked
            // rows copied in parallel, so the serial walk only iterates
            // non-empty rows and pays ~1/3 of the barriers.
            if (d.mclip_node) {
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.vcopy_pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
                vkCmdPushConstants(
                    resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(pc), &pc);
                const uint32_t vcopy_total = static_cast<uint32_t>(cfg.rows) *
                    static_cast<uint32_t>(cfg.width);
                vkCmdDispatch(resource.cmd, (vcopy_total + 255) / 256, 1, 1);

                VkMemoryBarrier vcopy_barrier {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                };
                vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &vcopy_barrier, 0, nullptr, 0, nullptr);
            }

            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.vcheck_pipeline);
            vkCmdBindDescriptorSets(
                resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdPushConstants(
                resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(pc), &pc);
            vkCmdDispatch(resource.cmd, 1, 1, 1);
        }
    }

    // download: vcheck writes vout either into staging directly (no copy) or
    // into dev_buf (one SDMA copy back); without vcheck the row kernel's dst
    // (dev_buf) always needs the copy.
    const bool need_d2h = (d.vcheck > 0) ? d.vout_dev : (d.download_total > 0);
    if (need_d2h) {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
            if (!d.process[plane]) {
                continue;
            }
            const auto & cfg = d.planes[plane];
            const VkBufferCopy region {
                .srcOffset = (d.vcheck > 0) ? cfg.vout_offset : cfg.dst_offset,
                .dstOffset = d.upload_total + cfg.dl_offset,
                .size = cfg.dl_bytes
            };
            vkCmdCopyBuffer(resource.cmd, resource.dev_buf, resource.staging, 1, &region);
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

static const VSFrame *VS_CC Eedi3GetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    const int sn = (d->field > 1) ? n / 2 : n;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
        if (d->vcheck > 0 && d->sclip_node) {
            vsapi->requestFrameFilter(n, d->sclip_node, frameCtx);
        }
        if (d->mclip_node) {
            vsapi->requestFrameFilter(sn, d->mclip_node, frameCtx);
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const VSFrame * src = vsapi->getFrameFilter(sn, d->node, frameCtx);

    // Plane passthrough. newVideoFrame2 requires a non-null planeSrc array but
    // tolerates null entries (those planes are freshly allocated). Processed
    // planes are always freshly allocated. Unprocessed planes may share src's
    // plane data when !dh (dims match); under dh the output is 2x tall so no
    // plane can be shared (eedi3m/eedi3vk2 also leave unprocessed planes
    // undefined there).
    const int pl[] = { 0, 1, 2 };
    const VSFrame * fr[] = {
        (!d->dh && !d->process[0]) ? src : nullptr,
        (!d->dh && !d->process[1]) ? src : nullptr,
        (!d->dh && !d->process[2]) ? src : nullptr
    };
    // dh doubles the output height (width is never changed by EEDI3)
    const int out_height = d->dh ? d->vi->height * 2 : d->vi->height;
    VSFrame * dst = vsapi->newVideoFrame2(
        &d->vi->format, d->vi->width, out_height, fr, pl, src, core);

    auto resource = d->pool.take();

    auto set_error = [&](const std::string & error_message) {
        d->pool.give_back(std::move(resource));
        vsapi->setFilterError(("EEDI3VK: " + error_message).c_str(), frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    };

    VkDevice dev = d->device->device;
    float * map = resource.map;

    const VSFrame * scp = nullptr;
    if (d->vcheck > 0 && d->sclip_node) {
        scp = vsapi->getFrameFilter(n, d->sclip_node, frameCtx);
    }
    const VSFrame * mcp = nullptr;
    if (d->mclip_node) {
        mcp = vsapi->getFrameFilter(sn, d->mclip_node, frameCtx);
    }

    int field = d->field & 1;
    int err;
    const int fieldBased = vsapi->mapGetIntSaturated(
        vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err);
    if (fieldBased == VSC_FIELD_BOTTOM) {
        field = 0;
    } else if (fieldBased == VSC_FIELD_TOP) {
        field = 1;
    }
    if (d->field > 1) {
        field = (n & 1) ^ field;
    }
    const int off = 1 - field;

    static const int hframe = getenv("VSFEEL_EEDI3_HFRAME") ? atoi(getenv("VSFEEL_EEDI3_HFRAME")) : 0;
    const bool hbench = getenv("VSFEEL_EEDI3_HBENCH") && sn == hframe;
    const auto h_t0 = std::chrono::steady_clock::now();
    auto h_tMaskEnd = h_t0, h_tGatherEnd = h_t0, h_tRawEnd = h_t0;
    auto h_tRecEnd = h_t0;

    // Re-record the command buffer with this frame's interp-row parity (the
    // previous submit on this resource was waited on before give_back, so the
    // pool reset is safe).
    checkVK(vkResetCommandPool(dev, resource.pool, 0));
    if (const auto err = record_command_buffer(*d, resource, field)) {
        return set_error(*err);
    }
    if (hbench) { h_tRecEnd = std::chrono::steady_clock::now(); }

    const bool coherent =
        !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // CPU: gather tight kept source rows, CPU-packed dilation bits, and
    // sclip interp rows into the upload region (plain streaming copies for
    // the rows; the GPU pad kernel expands mirrors, the row kernel reads the
    // bits). With the ReBAR path the upload region lives in host-visible
    // VRAM (resource.up_map) and is NT-stored directly; the old path writes
    // system-RAM staging and lets the DMA mirror it.
    uint8_t * const staging = static_cast<uint8_t *>(static_cast<void *>(map));
    uint8_t * const upload = resource.up_map
        ? resource.up_map : staging;

    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        const auto & cfg = d->planes[plane];

        const auto srcp = vsapi->getReadPtr(src, plane);
        const ptrdiff_t src_stride = vsapi->getStride(src, plane);
        const size_t raw_row_bytes = static_cast<size_t>(cfg.width) * pad_elem_bytes(d->bits);
        uint8_t * rawp = (d->raw_stage ? staging : upload) + cfg.raw_offset;
        // The tight kept-row gather and the destination's kept-row copy read
        // exactly the same source rows in the same order (raw row k -> kept
        // row k -> dst row `off + 2k`), so both are written here while the
        // source row is hot. Doing the dst half later in the blit re-read the
        // whole source a second time (8.3 MB/frame at the bench geometry) —
        // measured worth ~5-7% when the host copies are the limiter.
        auto dstp = vsapi->getWritePtr(dst, plane);
        const ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
        const bool nt_raw = (d->copy_mode & 1) != 0 && !d->raw_stage;
        const bool nt_kept = (d->copy_mode & 2) != 0;
        if (d->skip_raw) {
            // diagnostics-only
        } else if (!d->dh) {
            for (int k = 0; k < cfg.rows; ++k) {
                const uint8_t * const srow = srcp + src_stride * (off + 2 * k);
                frame_copy_out(rawp + static_cast<size_t>(k) * raw_row_bytes,
                               srow, raw_row_bytes, nt_raw);
                if (!d->skip_blit) {
                    frame_copy_out(dstp + dst_stride * (off + 2 * k),
                                   srow, raw_row_bytes, nt_kept);
                }
            }
        } else {
            for (int k = 0; k < cfg.rows; ++k) {
                const uint8_t * const srow = srcp + src_stride * k;
                frame_copy_out(rawp + static_cast<size_t>(k) * raw_row_bytes,
                               srow, raw_row_bytes, nt_raw);
                if (!d->skip_blit) {
                    frame_copy_out(dstp + dst_stride * (2 * k + off),
                                   srow, raw_row_bytes, nt_kept);
                }
            }
        }
        if (hbench) { h_tRawEnd = std::chrono::steady_clock::now(); }
        if (d->mclip_node && mcp) {
            // single Gray mask drives every processed plane: mask row for
            // interp row r is (dh ? r : field + 2r) of the mclip frame.
            // Stored as packed bits (words per row); zeroed first since the
            // builder ORs bits in.
            const uint8_t * maskp = vsapi->getReadPtr(mcp, 0);
            const ptrdiff_t mask_stride = vsapi->getStride(mcp, 0);
            const uint16_t * mask16 = d->mclip_native16
                ? reinterpret_cast<const uint16_t *>(maskp) : nullptr;
            uint8_t * bm = upload + cfg.bits_offset;
            const size_t bm_row_bytes =
                static_cast<size_t>((cfg.width + 31) / 32) * sizeof(uint32_t);
            std::vector<uint64_t> bmask_scratch(
                2 * static_cast<size_t>((cfg.width + d->mdis + 63) / 64));
            for (int r = 0; r < cfg.rows; ++r) {
                const int mrow = d->dh ? r : field + 2 * r;
                uint8_t * bmr = bm + static_cast<int64_t>(r) * bm_row_bytes;
                const uint8_t * const mrow_p = maskp + mask_stride * mrow;
                build_bmask_row(mrow_p,
                                mask16 ? reinterpret_cast<const uint16_t *>(mrow_p)
                                       : nullptr,
                                reinterpret_cast<uint32_t *>(bmr),
                                cfg.width, d->mdis, bmask_scratch.data());
            }
        }
        if (hbench) { h_tMaskEnd = std::chrono::steady_clock::now(); }

        if (d->vcheck > 0 && d->sclip_node && scp && !d->skip_sclip) {
            const auto scpp = vsapi->getReadPtr(scp, plane);
            const ptrdiff_t scp_stride = vsapi->getStride(scp, plane);
            const size_t row_bytes = static_cast<size_t>(cfg.width) * d->elem_bytes;
            uint8_t * sc = upload + cfg.sclip_offset;
            for (int r = 0; r < cfg.rows; ++r) {
                frame_copy_out(sc + static_cast<size_t>(r) * row_bytes,
                               scpp + scp_stride * (field + 2 * r), row_bytes,
                               (d->copy_mode & 1) != 0);
            }
        }
    }
    if (hbench) { h_tGatherEnd = std::chrono::steady_clock::now(); }

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
                .offset = 0,
                .size = d->upload_total,
            });
        }
        checkVK(vkFlushMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
    }

    // Drain the CPU store buffer so no NT upload write is still in flight when
    // the GPU reads up_dev (required for the ReBAR path; harmless otherwise).
    _mm_sfence();

    checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
        resource.cmd, resource.fence));

    const auto h_t1 = std::chrono::steady_clock::now();
    checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));
    const auto h_t2 = std::chrono::steady_clock::now();

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
                .offset = d->upload_total + cfg.dl_offset,
                .size = cfg.dl_bytes,
            });
        }
        checkVK(vkInvalidateMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
    }

    // Copy results into dst: interp rows from the download region, kept rows
    // straight from src (like eedi3vk2).
    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        const auto & cfg = d->planes[plane];
        const size_t row_bytes = static_cast<size_t>(cfg.width) * d->elem_bytes;
        const uint8_t * dl = static_cast<const uint8_t *>(static_cast<const void *>(map)) +
                             d->upload_total + cfg.dl_offset;

        auto dstp = vsapi->getWritePtr(dst, plane);
        const ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
        const auto srcp = vsapi->getReadPtr(src, plane);
        const ptrdiff_t src_stride = vsapi->getStride(src, plane);

        // interp rows (the row kernel wrote rows r at dst rows field+2r).
        // Streaming: the staging download is never re-read by the CPU and
        // the frame is written once — bypass the cache both ways. Staging
        // download rows are 32-byte aligned when row_bytes is (the common
        // case); otherwise fall back to memcpy (movntdqa faults unaligned).
        // NOTE: a plain-memcpy variant was tried and is SLOWER in situ
        // (blit 6.1 -> 10.4 ms/frame, vc0 481 -> 401 fps) even though the
        // isolated microbenchmark prefers it — do not "fix" this back.
        if (d->skip_blit) {
            // diagnostics-only
        } else if (d->blit_contig) {
            // diagnostics-only: measures the ceiling of a fully contiguous copy
            std::memcpy(dstp, dl, static_cast<size_t>(cfg.rows) * row_bytes);
        } else if ((row_bytes & 31) == 0 && (d->copy_mode & 2)) {
            // bit2 picks the load flavor: NT load (copy_stream_read, right for
            // WC memory) or an ordinary cached load (copy_stream_out, right
            // for the WB staging the GPU just wrote). Both use an NT store.
            if (d->copy_mode & 4) {
                for (int r = 0; r < cfg.rows; ++r) {
                    copy_stream_out(dstp + dst_stride * (field + 2 * r),
                                    dl + static_cast<size_t>(r) * row_bytes, row_bytes);
                }
            } else {
                for (int r = 0; r < cfg.rows; ++r) {
                    copy_stream_read(dstp + dst_stride * (field + 2 * r),
                                     dl + static_cast<size_t>(r) * row_bytes, row_bytes);
                }
            }
        } else {
            for (int r = 0; r < cfg.rows; ++r) {
                std::memcpy(dstp + dst_stride * (field + 2 * r),
                            dl + static_cast<size_t>(r) * row_bytes, row_bytes);
            }
        }

        // kept rows were already written during the upload gather above
        // (same source rows, same order) — nothing left to do here.
    }

    d->pool.give_back(std::move(resource));

    if (hbench) {
        const auto h_t3 = std::chrono::steady_clock::now();
        const auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        fprintf(stderr, "[eedi3-hbench] cpu_stage=%.3fms gpu_submit_wait=%.3fms blit=%.3fms\n",
                us(h_t0, h_t1), us(h_t1, h_t2), us(h_t2, h_t3));
        fprintf(stderr, "[eedi3-hbench]   of cpu_stage: record=%.3fms raw=%.3fms bits=%.3fms sclip=%.3fms\n",
                us(h_t0, h_tRecEnd), us(h_tRecEnd, h_tRawEnd),
                us(h_tRawEnd, h_tMaskEnd), us(h_tMaskEnd, h_tGatherEnd));
    }

    vsapi->freeFrame(src);
    vsapi->freeFrame(scp);
    vsapi->freeFrame(mcp);

    VSMap * props = vsapi->getFramePropertiesRW(dst);
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

    return dst;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC Eedi3Free(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sclip_node);
    vsapi->freeNode(d->mclip_node);
    delete d;
}

static void VS_CC Eedi3Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<Eedi3Data>() };
    int err = 0;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    d->sclip_node = vsapi->mapGetNode(in, "sclip", 0, &err);
    bool has_sclip = d->sclip_node != nullptr;
    err = 0;
    d->mclip_node = vsapi->mapGetNode(in, "mclip", 0, &err);
    bool has_mclip = d->mclip_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
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
    if (!d->dh) {
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (d->process[plane] &&
                ((d->vi->height >> (plane > 0 ? d->vi->format.subSamplingH : 0)) & 1)) {
                return set_error("plane's height must be mod 2 when dh=False");
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

    // mclip must be a single Gray plane (vszip CPU semantic); non-Gray8 masks
    // are converted to Gray8 internally (SetFrameProps _Range=1 + resize.Point,
    // like vszip CPU).
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

        // Gray16 integer masks are handled natively: the exact boolean the
        // Gray8 conversion would produce is (v >= 129), verified exhaustively,
        // so the extra full-frame graph node is pure overhead. Everything else
        // (float, 10/12/14-bit, ...) keeps the reference conversion.
        if (mvi->format.bitsPerSample == 16 && mvi->format.sampleType == stInteger) {
            d->mclip_native16 = true;
        } else if (mvi->format.bitsPerSample != 8 || mvi->format.sampleType != stInteger) {
            VSMap * args = vsapi->createMap();
            vsapi->mapConsumeNode(args, "clip", d->mclip_node, maReplace);
            d->mclip_node = nullptr;  // ownership moved into args

            // std.SetFrameProps(_Range=1)
            vsapi->mapSetInt(args, "_Range", 1, maReplace);
            VSMap * ret = vsapi->invoke(
                vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "SetFrameProps", args);
            if (vsapi->mapGetError(ret)) {
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

            // resize.Point -> Gray8
            vsapi->mapSetInt(args, "format", vsapi->queryVideoFormatID(
                cfGray, stInteger, 8, 0, 0, core), maReplace);
            ret = vsapi->invoke(
                vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args);
            vsapi->freeMap(args);
            if (vsapi->mapGetError(ret)) {
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
        }
    }

    // sclip only validated when vcheck > 0 (eedi3m semantics). Like eedi3m /
    // eedi3vk2 / vszip, the sclip describes the OUTPUT: under field > 1 the
    // output doubles the frame count (sclip supplies one frame per output
    // frame, i.e. 2N frames — based_aa builds it via Interleave([s, s])), and
    // under dh it doubles the height. Validating against the pre-doubling vi
    // would wrongly accept an N-frame sclip (whose frames n >= N would be
    // requested out of range) and reject the correct 2N one.
    if (d->vcheck > 0 && d->sclip_node) {
        const auto svi = vsapi->getVideoInfo(d->sclip_node);
        VSVideoInfo out_vi = *d->vi;
        if (d->field > 1) {
            if (d->vi->numFrames > INT32_MAX / 2) {
                return set_error("resulting clip is too long");
            }
            out_vi.numFrames = d->vi->numFrames * 2;
        }
        if (d->dh) {
            out_vi.height *= 2;
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

    int num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &err));
    if (err) {
        // 8 is the overlap knee on the target GPU (4 streams starves the
        // queue; 16+ plateaus). ~250MB VRAM per stream.
        num_streams = 8;
    }
    if (num_streams < 1 || num_streams > 32) {
        return set_error("num_streams must be 1..32.");
    }
    d->num_streams = num_streams;

    if (const char * rb = std::getenv("VSFEEL_EEDI3_NOREBAR")) {
        d->rebar_up = (atoi(rb) == 0);
    }
    if (const char * vd = std::getenv("VSFEEL_EEDI3_VOUTDEV")) {
        d->vout_dev = (atoi(vd) != 0);
    }
    d->raw_stage = std::getenv("VSFEEL_EEDI3_RAWSTAGE") != nullptr;
    d->blit_contig = std::getenv("VSFEEL_EEDI3_BLITCONTIG") != nullptr;
    d->skip_pad = std::getenv("VSFEEL_EEDI3_NOPAD") != nullptr;
    d->skip_blit = std::getenv("VSFEEL_EEDI3_NOBLIT") != nullptr;
    d->skip_sclip = std::getenv("VSFEEL_EEDI3_NOSCLIP") != nullptr;
    d->skip_raw = std::getenv("VSFEEL_EEDI3_NORAW") != nullptr;
    d->skip_h2d = std::getenv("VSFEEL_EEDI3_NOH2D") != nullptr;
    d->skip_vcheck = std::getenv("VSFEEL_EEDI3_NOVC") != nullptr;

    if (const char * cm = std::getenv("VSFEEL_EEDI3_COPY")) {
        const int v = atoi(cm);
        if (v >= 0 && v <= 7) {
            d->copy_mode = v;
        }
    }
    d->device_id = device_id;

    {
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
    }

    // eedi3m scaling (see EEDI3.cpp create), with cost3 always on (GPU family
    // semantics — eedi3vk2/vszip* have no cost3 switch and always do alpha/3
    // and sum s0+s1+s2):
    //   remainingWeight = 1 - alpha - beta   (raw, before alpha/3)
    //   alpha /= 3   (cost3)
    //   int:  beta *= 2^(bits-8), gamma *= 2^(bits-8),
    //         vthresh0 *= 2^(bits-8), vthresh1 *= 2^(bits-8)
    //   float: beta /= 255, gamma /= 255, vthresh0 /= 255, vthresh1 /= 255
    //   vthresh2 is never scaled; alpha is never /255.
    d->rw = 1.0f - d->alpha - d->beta;
    d->alpha /= 3.0f;
    if (d->vi->format.sampleType == stInteger) {
        d->peak = (1 << d->vi->format.bitsPerSample) - 1;
        const int scale = 1 << (d->vi->format.bitsPerSample - 8);
        d->beta *= static_cast<float>(scale);
        d->gamma *= static_cast<float>(scale);
        vthresh0 *= static_cast<float>(scale);
        vthresh1 *= static_cast<float>(scale);
    } else {
        d->peak = 1;
        d->beta /= 255.0f;
        d->gamma /= 255.0f;
        vthresh0 /= 255.0f;
        vthresh1 /= 255.0f;
    }
    d->rcp_vth0 = 1.0f / vthresh0;
    d->rcp_vth1 = 1.0f / vthresh1;
    d->rcp_vth2 = 1.0f / d->vthresh2;

    // Workgroup sizes (spec constants 6/7 in the shaders). The row kernel is
    // a subgroup-register DP: one workgroup of SGSIZE=32 lanes per interp row,
    // exactly one subgroup, with the host requesting requiredSubgroupSize=32
    // on the row pipeline (RDNA3's native 64-lane wavefront is split via the
    // subgroup size control feature). The vcheck kernel is a single WG
    // striding over columns (up to WIDTH).
    constexpr int SGSIZE = 32;   // must match the shader
    const auto & lim = d->device->limits;
    const int max_invoc = static_cast<int>(lim.maxComputeWorkGroupInvocations);
    const int max_x = static_cast<int>(lim.maxComputeWorkGroupSize[0]);
    int lsz_vcheck = std::min({ 1024, max_invoc, max_x });
    if (!d->device->subgroup_size_control ||
        d->device->min_subgroup_size > SGSIZE || SGSIZE > d->device->max_subgroup_size) {
        return set_error("device cannot run the EEDI3 row kernel (needs a 32-lane "
                         "subgroup via subgroup size control)");
    }
    const int lsz_row = SGSIZE;

    VkDevice dev = d->device->device;

    // Pipeline layout / descriptors: 7 storage buffers.
    {
        VkDescriptorSetLayoutBinding bindings[BIND_COUNT];
        for (uint32_t i = 0; i < BIND_COUNT; ++i) {
            bindings[i] = VkDescriptorSetLayoutBinding {
                i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr
            };
        }
        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = BIND_COUNT,
            .pBindings = bindings
        };
        checkVK(vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &d->set_layout));
    }
    {
        VkPushConstantRange pcr {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        VkPipelineLayoutCreateInfo plci {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &d->set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcr
        };
        checkVK(vkCreatePipelineLayout(dev, &plci, nullptr, &d->pipeline_layout));
    }
    {
        // Two sets per stream when the ReBAR upload is active (the pad kernel
        // needs its own set because its binding 0 is the raw upload while the
        // row kernel's binding 0 is the built pad).
        const uint32_t sets_per_stream = (d->rebar_up) ? 2u : 1u;
        VkDescriptorPoolSize pool_size {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            BIND_COUNT * static_cast<uint32_t>(d->num_streams) * sets_per_stream
        };
        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(d->num_streams) * sets_per_stream,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size
        };
        checkVK(vkCreateDescriptorPool(dev, &pool_info, nullptr, &d->desc_pool));
    }

    const int elem_bytes = d->elem_bytes;
    const int pad_elem = pad_elem_bytes(d->bits);

    // Shader modules for this bit depth (row/vcheck/pad per depth; the
    // bmask dilate+pack kernel is depth-independent).
    {
        const uint32_t * row_code = nullptr;
        size_t row_size = 0;
        const uint32_t * vc_code = nullptr;
        size_t vc_size = 0;
        const uint32_t * pad_code = nullptr;
        size_t pad_size = 0;
        switch (d->bits) {
            case 16:
                row_code = eedi3_16_row_spv; row_size = eedi3_16_row_spv_size;
                vc_code = eedi3_16_vcheck_spv; vc_size = eedi3_16_vcheck_spv_size;
                pad_code = eedi3_16_pad_spv; pad_size = eedi3_16_pad_spv_size;
                break;
            case 32:
                row_code = eedi3_32_row_spv; row_size = eedi3_32_row_spv_size;
                vc_code = eedi3_32_vcheck_spv; vc_size = eedi3_32_vcheck_spv_size;
                pad_code = eedi3_32_pad_spv; pad_size = eedi3_32_pad_spv_size;
                break;
            default:
                return set_error("unsupported bit depth");
        }
        auto r1 = create_shader_module(*d->device, row_code, row_size);
        if (std::holds_alternative<std::string>(r1)) {
            return set_error(std::get<std::string>(r1));
        }
        d->row_module = std::get<VkShaderModule>(r1);
        if (d->vcheck > 0) {
            auto r2 = create_shader_module(*d->device, vc_code, vc_size);
            if (std::holds_alternative<std::string>(r2)) {
                return set_error(std::get<std::string>(r2));
            }
            d->vcheck_module = std::get<VkShaderModule>(r2);
        }
        auto r3 = create_shader_module(*d->device, pad_code, pad_size);
        if (std::holds_alternative<std::string>(r3)) {
            return set_error(std::get<std::string>(r3));
        }
        d->pad_module = std::get<VkShaderModule>(r3);
        if (d->vcheck > 0 && d->mclip_node) {
            const uint32_t * vcopy_code = nullptr;
            size_t vcopy_size = 0;
            switch (d->bits) {
                case 16:
                    vcopy_code = eedi3_16_vcopy_spv; vcopy_size = eedi3_16_vcopy_spv_size;
                    break;
                case 32:
                    vcopy_code = eedi3_32_vcopy_spv; vcopy_size = eedi3_32_vcopy_spv_size;
                    break;
                default:
                    return set_error("unsupported bit depth");
            }
            auto r4 = create_shader_module(*d->device, vcopy_code, vcopy_size);
            if (std::holds_alternative<std::string>(r4)) {
                return set_error(std::get<std::string>(r4));
            }
            d->vcopy_module = std::get<VkShaderModule>(r4);
        }
    }

    const int numPlanes = d->vi->format.numPlanes;
    const int subW = d->vi->format.subSamplingW;
    const int subH = d->vi->format.subSamplingH;

    // Plane geometry (output dims; dh doubles the height BEFORE this: d->vi
    // already doubled? No - the filter doubles vi at create AFTER validation.
    // Do it now:
    VSVideoInfo * out_vi = const_cast<VSVideoInfo *>(d->vi);
    // We must NOT modify the const node videoInfo in place (it is shared with
    // the upstream node). Instead build an output vi copy.
    VSVideoInfo out_video = *d->vi;
    if (d->field > 1) {
        if (d->vi->numFrames > INT32_MAX / 2) {
            return set_error("resulting clip is too long");
        }
        out_video.numFrames *= 2;
        vsh::muldivRational(&out_video.fpsNum, &out_video.fpsDen, 2, 1);
    }
    if (d->dh) {
        out_video.height *= 2;
    }

    auto align32 = [](VkDeviceSize v) { return (v + 31) & ~VkDeviceSize(31); };

    const int tpitch = 2 * d->mdis + 1;

    VkDeviceSize upload_total = 0;
    VkDeviceSize download_total = 0;
    VkDeviceSize dev_total = 0;

    auto & planes = d->planes;
    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        const int pw = (plane == 0) ? out_video.width : out_video.width >> subW;
        const int ph = (plane == 0) ? out_video.height : out_video.height >> subH;

        cfg.width = pw;
        cfg.height = ph;
        cfg.rows = ph / 2;
        cfg.tpitch = tpitch;
        cfg.pad_stride = (pw + MARGIN_H * 2 + 15) & ~15;   // pad elements
        cfg.pad_height = ph + MARGIN_V * 2;

        // staging/upload regions: tight kept source rows (the pad kernel
        // expands mirrors), gathered sclip rows, and CPU-packed dilation bits
        cfg.raw_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * pad_elem;
        cfg.raw_offset = align32(upload_total);
        upload_total = align32(cfg.raw_offset + cfg.raw_bytes);

        if (d->vcheck > 0 && d->sclip_node) {
            cfg.sclip_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * elem_bytes;
            cfg.sclip_offset = align32(upload_total);
            upload_total = align32(cfg.sclip_offset + cfg.sclip_bytes);
        }

        if (d->mclip_node) {
            cfg.bits_bytes = static_cast<VkDeviceSize>((pw + 31) / 32) * cfg.rows * 4;
            cfg.bits_offset = align32(upload_total);
            upload_total = align32(cfg.bits_offset + cfg.bits_bytes);
        }

        // download region (host staging): interp rows only (tight)
        cfg.dl_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * elem_bytes;
        cfg.dl_offset = align32(download_total);
        download_total = align32(cfg.dl_offset + cfg.dl_bytes);

        // device-local regions
        cfg.dst_bytes = cfg.dl_bytes;
        cfg.dst_offset = align32(dev_total);
        dev_total = align32(cfg.dst_offset + cfg.dst_bytes);

        // per-interp-row empty flags for the vcheck split (1 byte/row;
        // written by the row kernel, read by vcopy + the walk)
        cfg.rempty_bytes = static_cast<VkDeviceSize>(cfg.rows);
        cfg.rempty_offset = align32(dev_total);
        dev_total = align32(cfg.rempty_offset + cfg.rempty_bytes);

        cfg.pbt_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * tpitch;  // int8
        cfg.pbt_offset = align32(dev_total);
        dev_total = align32(cfg.pbt_offset + cfg.pbt_bytes);

        if (d->vcheck > 0) {
            cfg.dmap_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows;      // int8
            cfg.dmap_offset = align32(dev_total);
            dev_total = align32(cfg.dmap_offset + cfg.dmap_bytes);

            if (!(d->vcheck > 0 && d->sclip_node)) {
                cfg.cint_bytes = cfg.dl_bytes;  // io
                cfg.cint_offset = align32(dev_total);
                dev_total = align32(cfg.cint_offset + cfg.cint_bytes);
            }

            // vout region (dev_buf): RETIRED — vcheck/vcopy write vout
            // directly into the staging download region (b7 views staging),
            // so no D2H copy is needed. Kept allocated to avoid layout churn.
            cfg.vout_bytes = cfg.dl_bytes;
            cfg.vout_offset = align32(dev_total);
            dev_total = align32(cfg.vout_offset + cfg.vout_bytes);
        }
    }

    d->upload_total = upload_total;
    d->download_total = download_total;
    d->dev_total = dev_total;

    // pad_dev (device-only, never staged): per-plane built padded planes
    // produced by the pad kernel from the mirrored upload. The mirror
    // occupies [0, upload_total), so the tail starts after it.
    VkDeviceSize tail_total = upload_total;
    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        cfg.built_bytes = static_cast<VkDeviceSize>(cfg.pad_stride) * cfg.pad_height * pad_elem;
        cfg.built_offset = align32(tail_total);
        tail_total = align32(cfg.built_offset + cfg.built_bytes);
    }

    const VkDeviceSize staging_size = std::max(upload_total + download_total, VkDeviceSize(4));
    const VkDeviceSize dev_size = std::max(dev_total, VkDeviceSize(4));
    const VkDeviceSize mirror_size = std::max(tail_total, VkDeviceSize(4));

    // Per-width pipelines (deduplicated).
    RowSpecData base_spec {
        .width = 0,
        .nrad = d->nrad,
        .mdis = d->mdis,
        .has_mclip = d->mclip_node ? 1 : 0,
        .has_sclip = (d->vcheck > 0 && d->sclip_node) ? 1 : 0,
        .vcheck = d->vcheck,
        .lsz_row = lsz_row,
        .lsz_vcheck = lsz_vcheck,
    };

    // helper to fetch-or-create the (row, vcheck, pad, vcopy) pipelines for
    // a width key; uses Eedi3Data::WidthKey (all filter-level params like
    // vcheck/mclip are identical across planes, so the geometry fields
    // dominate). The pad/vcopy upload kernels use fixed local sizes, so the
    // row/vcheck workgroup-size spec entries are ignored by their pipelines.
    using WidthKey = Eedi3Data::WidthKey;
    auto get_pipelines = [&](const WidthKey & key, VkPipeline & row_pipe,
                             VkPipeline & vc_pipe, VkPipeline & pad_pipe,
                             VkPipeline & vcopy_pipe) -> std::optional<std::string> {
        for (auto & [k, quad] : d->width_pipes) {
            if (k == key) {
                row_pipe = quad[0];
                vc_pipe = quad[1];
                pad_pipe = quad[2];
                vcopy_pipe = quad[3];
                return std::nullopt;
            }
        }
        RowSpecData spec = base_spec;
        spec.width = key.width;

        auto r1 = create_pipeline(*d->device, spec, d->row_module, d->pipeline_layout,
                                  d->device->subgroup_size_control ? SGSIZE : 0);
        if (std::holds_alternative<std::string>(r1)) {
            return std::get<std::string>(r1);
        }
        VkPipeline rowp = std::get<VkPipeline>(r1);
        VkPipeline vcp = VK_NULL_HANDLE;
        if (d->vcheck > 0) {
            auto r2 = create_pipeline(*d->device, spec, d->vcheck_module, d->pipeline_layout, 0);
            if (std::holds_alternative<std::string>(r2)) {
                vkDestroyPipeline(dev, rowp, nullptr);
                return std::get<std::string>(r2);
            }
            vcp = std::get<VkPipeline>(r2);
        }
        auto r3 = create_pipeline(*d->device, spec, d->pad_module, d->pipeline_layout, 0);
        if (std::holds_alternative<std::string>(r3)) {
            vkDestroyPipeline(dev, rowp, nullptr);
            if (vcp) {
                vkDestroyPipeline(dev, vcp, nullptr);
            }
            return std::get<std::string>(r3);
        }
        VkPipeline padp = std::get<VkPipeline>(r3);
        VkPipeline vcopyp = VK_NULL_HANDLE;
        if (d->vcheck > 0 && d->mclip_node) {
            auto r4 = create_pipeline(*d->device, spec, d->vcopy_module, d->pipeline_layout, 0);
            if (std::holds_alternative<std::string>(r4)) {
                vkDestroyPipeline(dev, rowp, nullptr);
                if (vcp) {
                    vkDestroyPipeline(dev, vcp, nullptr);
                }
                vkDestroyPipeline(dev, padp, nullptr);
                return std::get<std::string>(r4);
            }
            vcopyp = std::get<VkPipeline>(r4);
        }
        d->width_pipes.emplace_back(key, std::array<VkPipeline, 4>{ rowp, vcp, padp, vcopyp });
        row_pipe = rowp;
        vc_pipe = vcp;
        pad_pipe = padp;
        vcopy_pipe = vcopyp;
        return std::nullopt;
    };

    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        WidthKey key { cfg.width, cfg.rows, cfg.tpitch, cfg.pad_stride, cfg.pad_height };
        if (auto err = get_pipelines(key, cfg.row_pipeline, cfg.vcheck_pipeline,
                                     cfg.pad_pipeline, cfg.vcopy_pipeline)) {
            return set_error(*err);
        }
    }

    // Resources (one per stream)
    d->pool.semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->pool.reserve(d->num_streams);

    // Queue sharing: min(num_streams, queue_count) is the starting point. With
    // one stream per queue each queue drains while its worker does the
    // post-fence CPU work (blit + bookkeeping + next upload) before the next
    // submit, leaving idle bubbles; sharing a queue across streams keeps a
    // next CB queued. Sweep with VSFEEL_EEDI3_QUEUES=N (durable tuning knob,
    // sibling of VSFEEL_BILAT_QUEUES).
    uint32_t num_queues = std::min(
        d->num_streams, static_cast<int>(d->device->queue_count));
    if (const char * qn = std::getenv("VSFEEL_EEDI3_QUEUES")) {
        const int q = atoi(qn);
        if (q > 0) {
            num_queues = std::min<uint32_t>(
                static_cast<uint32_t>(d->num_streams),
                std::min<uint32_t>(static_cast<uint32_t>(q),
                    d->device->queue_count));
        }
    }

    for (int i = 0; i < d->num_streams; ++i) {
        Eedi3Resource resource;

        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.staging));
        }
        {
            // Cached host-visible staging: CPU writes use streaming stores
            // and the download uses streaming loads (both bypass the cache
            // anyway), but cached memory is REQUIRED for sane NT-load and
            // DMA behavior — uncached staging collapsed throughput (43fps).
            // GPU GTT reads of this memory snoop-stall, so everything the
            // kernels reuse goes through the H2D mirror instead.
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
                .size = dev_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.dev_buf));
            const auto result = allocate_memory(
                *d->device, resource.dev_buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.dev_mem = std::get<AllocatedMemory>(result).memory;
            resource.dev_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        {
            // Device-local kernel-built regions (built pads, packed bits).
            // The upload itself is read straight from staging (no H2D).
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = mirror_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.pad_dev));
            const auto result = allocate_memory(
                *d->device, resource.pad_dev, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.pad_dev_mem = std::get<AllocatedMemory>(result).memory;
        }
        if (d->rebar_up && upload_total > 0) {
            // ReBAR direct-upload buffer: DEVICE_LOCAL|HOST_VISIBLE|COHERENT
            // (types 3/4 on this 7900XTX). The CPU NT-stores the upload into
            // VRAM through the BAR (the type is uncached, so an ordinary
            // cached store would be pathologically slow -- this is exactly why
            // an earlier attempt at host-visible staging collapsed; NT stores
            // bypass the cache and are fine, the nnedi3 precedent). The pad /
            // row / vcheck kernels then read it at full VRAM speed with no
            // DMA and no transfer barrier.
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = std::max(upload_total, VkDeviceSize(4)),
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.up_dev));
            const auto result = allocate_memory(
                *d->device, resource.up_dev,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.up_dev_mem = std::get<AllocatedMemory>(result).memory;
            resource.up_type_index = std::get<AllocatedMemory>(result).type_index;
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
            if (d->rebar_up && upload_total > 0) {
                checkVK(vkAllocateDescriptorSets(dev, &alloc_info, &resource.desc_set_pad));
            }
        }
        {
            // Heavily-reused kernel data (built pads, packed bits, sclip) is
            // read from the device-local H2D mirror; dst/pbt/dmap/cint/vout
            // live in dev_buf. Binding 8 is ENTRY_PAD's built-pad output view
            // of pad_dev.
            VkDescriptorBufferInfo pad_info {
                .buffer = resource.pad_dev, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo dst_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo pbt_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo dmap_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            // With the ReBAR upload the packed bits and sclip live in up_dev
            // (the CPU wrote them there); otherwise they sit in the pad_dev
            // mirror like the raw rows.
            VkBuffer const bmask_buf = (d->rebar_up && upload_total > 0)
                ? resource.up_dev : resource.pad_dev;
            VkDescriptorBufferInfo bmask_info {
                .buffer = bmask_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo sclip_info {
                .buffer = bmask_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo cint_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo vout_info {
                .buffer = d->vout_dev ? resource.dev_buf : resource.staging,
                .offset = 0, .range = VK_WHOLE_SIZE
            };
            // The PAD kernel reads the raw upload at binding 0 while the row
            // kernel reads the BUILT pad at binding 0 -- different buffers on
            // the same binding, which is exactly the aliasing hazard the notes
            // warn about. With ReBAR they need separate sets: desc_set_pad has
            // b0 = up_dev (raw), desc_set has b0 = pad_dev (built pad).
            VkDescriptorBufferInfo raw_info {
                .buffer = (d->rebar_up && upload_total > 0)
                    ? resource.up_dev : resource.pad_dev,
                .offset = 0, .range = VK_WHOLE_SIZE
            };
            const VkDescriptorBufferInfo * infos[BIND_COUNT] {
                &pad_info, &dst_info, &pbt_info, &dmap_info,
                &bmask_info, &sclip_info, &cint_info, &vout_info,
                &raw_info
            };
            VkWriteDescriptorSet writes[BIND_COUNT];
            for (uint32_t b = 0; b < BIND_COUNT; ++b) {
                writes[b] = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = b,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = infos[b],
                    .pTexelBufferView = nullptr
                };
            }
            vkUpdateDescriptorSets(dev, BIND_COUNT, writes, 0, nullptr);

            if (d->rebar_up && upload_total > 0) {
                // Same set as above except binding 0 = the raw upload (the
                // pad kernel's input) and binding 8 = the built-pad output
                // (desc_set leaves b8 == raw_info because raw and built both
                // used to live in pad_dev).
                VkWriteDescriptorSet pad_writes[BIND_COUNT];
                for (uint32_t b = 0; b < BIND_COUNT; ++b) {
                    pad_writes[b] = writes[b];
                    pad_writes[b].dstSet = resource.desc_set_pad;
                    pad_writes[b].pBufferInfo = (b == 0) ? &raw_info
                        : (b == 8) ? &pad_info : infos[b];
                }
                vkUpdateDescriptorSets(dev, BIND_COUNT, pad_writes, 0, nullptr);
            }
        }

        checkVK(vkMapMemory(dev, resource.staging_mem, 0, staging_size, 0,
                            reinterpret_cast<void **>(&resource.map)));
        if (d->rebar_up && upload_total > 0) {
            checkVK(vkMapMemory(dev, resource.up_dev_mem, 0, upload_total, 0,
                                reinterpret_cast<void **>(&resource.up_map)));
        }

        resource.queue = d->device->queues[i % num_queues].queue;
        resource.queue_lock = d->device->queues[i % num_queues].lock.get();

        // The command buffer is recorded per frame (interp-row parity varies),
        // so the resource is pushed empty; record_command_buffer runs in
        // GetFrame before each submit.
        d->pool.push(std::move(resource));
    }

    // Dependencies: node (strict spatial, or general when field > 1), sclip
    // (strict, only when vcheck > 0), mclip (strict/general).
    std::vector<VSFilterDependency> deps;
    deps.push_back({ d->node, d->field > 1 ? rpGeneral : rpStrictSpatial });
    if (d->vcheck > 0 && d->sclip_node) {
        deps.push_back({ d->sclip_node, rpStrictSpatial });
    }
    if (d->mclip_node) {
        deps.push_back({ d->mclip_node, d->field > 1 ? rpGeneral : rpStrictSpatial });
    }

    // Store the output dims into d->vi? No: createVideoFilter reads the local
    // `out_video`. The data struct keeps a pointer to the ORIGINAL vi (input
    // dims) — the frame processing reads src planes (original dims) and writes
    // the dst frame (output dims). Keep both.
    Eedi3Data * data = d.release();

    vsapi->createVideoFilter(
        out, "EEDI3", &out_video,
        Eedi3GetFrame, Eedi3Free,
        fmParallel, deps.data(), static_cast<int>(deps.size()), data, core);
}

// ---------------------------------------------------------------------------
// EEDI3H — horizontal EEDI3 (vszipcl parity).
//
// Pure composition: Transpose clip/sclip/mclip in, run EEDI3, Transpose back.
// No instance state, no new kernels, no new host paths — every behavior
// (validation, numerics, mclip/sclip/dh/field semantics) is EEDI3's, applied
// to the transposed geometry exactly like vszipcl's EEDI3H does on the GPU:
// field selects transposed-row (= original column) parity, dh doubles the
// transposed height (= the original width). By construction,
//   EEDI3H(x) == Transpose(EEDI3(Transpose(x)))
// bit-exactly (same kernels, same params, transposed data flow), which is
// also the primary test oracle (tests/test_eedi3h.py).
// ---------------------------------------------------------------------------

static void VS_CC Eedi3HCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    VSPlugin * self = static_cast<VSPlugin *>(userData);
    std::vector<VSNode *> owned;   // transposed intermediates we must release
    auto fail = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("EEDI3H: " + error_message).c_str());
        for (VSNode * n : owned) {
            vsapi->freeNode(n);
        }
    };

    // Transpose one optional input clip (absent -> leave *slot null).
    auto transpose_opt = [&](const char * key, VSNode ** slot) -> bool {
        int e = 0;
        VSNode * src = vsapi->mapGetNode(in, key, 0, &e);
        if (e || !src) {
            return true;
        }
        VSMap * args = vsapi->createMap();
        vsapi->mapConsumeNode(args, "clip", src, maReplace);
        VSMap * ret = vsapi->invoke(
            vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Transpose", args);
        vsapi->freeMap(args);
        if (vsapi->mapGetError(ret)) {
            fail(vsapi->mapGetError(ret));
            vsapi->freeMap(ret);
            return false;
        }
        *slot = vsapi->mapGetNode(ret, "clip", 0, nullptr);
        owned.push_back(*slot);
        vsapi->freeMap(ret);
        return true;
    };

    int e = 0;
    VSNode * clip = vsapi->mapGetNode(in, "clip", 0, &e);
    if (e || !clip) {
        return fail("clip is required");
    }
    VSNode *tclip = nullptr, *tsclip = nullptr, *tmclip = nullptr;
    {
        VSMap * args = vsapi->createMap();
        vsapi->mapConsumeNode(args, "clip", clip, maReplace);
        VSMap * ret = vsapi->invoke(
            vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Transpose", args);
        vsapi->freeMap(args);
        if (vsapi->mapGetError(ret)) {
            fail(vsapi->mapGetError(ret));
            vsapi->freeMap(ret);
            return;
        }
        tclip = vsapi->mapGetNode(ret, "clip", 0, nullptr);
        owned.push_back(tclip);
        vsapi->freeMap(ret);
    }
    if (!transpose_opt("sclip", &tsclip) || !transpose_opt("mclip", &tmclip)) {
        return;  // fail() already recorded the error and freed `owned`
    }

    // Forward every scalar arg verbatim; clips go in transposed.
    VSMap * args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", tclip, maReplace);
    if (tsclip) {
        vsapi->mapConsumeNode(args, "sclip", tsclip, maReplace);
    }
    if (tmclip) {
        vsapi->mapConsumeNode(args, "mclip", tmclip, maReplace);
    }
    // owned refs moved into args; the inner filter takes its own references.
    owned.clear();
    auto fwd_int = [&](const char * key) {
        int ee = 0;
        const int64_t v = vsapi->mapGetInt(in, key, 0, &ee);
        if (!ee) {
            vsapi->mapSetInt(args, key, v, maReplace);
        }
    };
    auto fwd_int_arr = [&](const char * key) {
        const int n = vsapi->mapNumElements(in, key);
        for (int i = 0; i < n; ++i) {
            int ee = 0;
            const int64_t v = vsapi->mapGetInt(in, key, i, &ee);
            if (!ee) {
                vsapi->mapSetInt(args, key, v, maAppend);
            }
        }
    };
    auto fwd_float = [&](const char * key) {
        int ee = 0;
        const double v = vsapi->mapGetFloat(in, key, 0, &ee);
        if (!ee) {
            vsapi->mapSetFloat(args, key, v, maReplace);
        }
    };
    fwd_int("field");
    fwd_int("dh");
    fwd_int_arr("planes");
    fwd_float("alpha");
    fwd_float("beta");
    fwd_float("gamma");
    fwd_int("nrad");
    fwd_int("mdis");
    fwd_int("vcheck");
    fwd_float("vthresh0");
    fwd_float("vthresh1");
    fwd_float("vthresh2");
    fwd_int("device_id");
    fwd_int("num_streams");

    VSMap * ret = vsapi->invoke(self, "EEDI3", args);
    vsapi->freeMap(args);
    if (vsapi->mapGetError(ret)) {
        // Our transposed refs died with `args`; the inner filter cleans up
        // its own references on create-failure. Just propagate the error.
        fail(vsapi->mapGetError(ret));
        vsapi->freeMap(ret);
        return;
    }
    VSNode * eedi3 = vsapi->mapGetNode(ret, "clip", 0, nullptr);
    vsapi->freeMap(ret);

    VSMap * args2 = vsapi->createMap();
    vsapi->mapConsumeNode(args2, "clip", eedi3, maReplace);
    VSMap * ret2 = vsapi->invoke(
        vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Transpose", args2);
    vsapi->freeMap(args2);
    if (vsapi->mapGetError(ret2)) {
        fail(vsapi->mapGetError(ret2));
        vsapi->freeMap(ret2);
        return;
    }
    VSNode * final = vsapi->mapGetNode(ret2, "clip", 0, nullptr);
    vsapi->freeMap(ret2);
    vsapi->mapConsumeNode(out, "clip", final, maReplace);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_eedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    const char * eedi3_args =
        "clip:vnode;"
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
        "sclip:vnode:opt;"
        "mclip:vnode:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;";
    vspapi->registerFunction(
        "EEDI3",
        eedi3_args,
        "clip:vnode;",
        Eedi3Create, nullptr, plugin
    );
    vspapi->registerFunction(
        "EEDI3H",
        eedi3_args,
        "clip:vnode;",
        Eedi3HCreate, plugin, plugin
    );
}
