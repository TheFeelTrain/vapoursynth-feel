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

// 16x16 byte transpose in place (SSE2, four unpack stages: 8/16/32/64-bit).
static inline void transpose16x16_u8(__m128i * r) {
    __m128i t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i]     = _mm_unpacklo_epi8(r[i], r[i + 1]);
        t[i + 1] = _mm_unpackhi_epi8(r[i], r[i + 1]);
    }
    for (int i = 0; i < 16; i += 4) {
        const __m128i u0 = _mm_unpacklo_epi16(t[i], t[i + 2]);
        const __m128i u1 = _mm_unpackhi_epi16(t[i], t[i + 2]);
        const __m128i u2 = _mm_unpacklo_epi16(t[i + 1], t[i + 3]);
        const __m128i u3 = _mm_unpackhi_epi16(t[i + 1], t[i + 3]);
        t[i] = u0; t[i + 1] = u1; t[i + 2] = u2; t[i + 3] = u3;
    }
    for (int i = 0; i < 16; i += 8) {
        const __m128i u0 = _mm_unpacklo_epi32(t[i], t[i + 4]);
        const __m128i u1 = _mm_unpackhi_epi32(t[i], t[i + 4]);
        const __m128i u2 = _mm_unpacklo_epi32(t[i + 1], t[i + 5]);
        const __m128i u3 = _mm_unpackhi_epi32(t[i + 1], t[i + 5]);
        const __m128i u4 = _mm_unpacklo_epi32(t[i + 2], t[i + 6]);
        const __m128i u5 = _mm_unpackhi_epi32(t[i + 2], t[i + 6]);
        const __m128i u6 = _mm_unpacklo_epi32(t[i + 3], t[i + 7]);
        const __m128i u7 = _mm_unpackhi_epi32(t[i + 3], t[i + 7]);
        t[i] = u0; t[i + 1] = u1; t[i + 2] = u2; t[i + 3] = u3;
        t[i + 4] = u4; t[i + 5] = u5; t[i + 6] = u6; t[i + 7] = u7;
    }
    for (int i = 0; i < 8; ++i) {
        r[2 * i]     = _mm_unpacklo_epi64(t[i], t[i + 8]);
        r[2 * i + 1] = _mm_unpackhi_epi64(t[i], t[i + 8]);
    }
}

// Out-of-place transpose: dst[x * dstride + y] = src[y * sstride + x], x in
// [0,W), y in [0,H). Strides and dims are in ELEMENTS; elem is the byte size.
// Only the 1-byte case (EEDI3H's transposed mask) is on a hot path: it runs
// 16x16 byte-transpose tiles, so both the load and the store are contiguous.
// Wider elements take a scalar fallback (correct, and unused in practice).
static void transpose_plane(const uint8_t * src, ptrdiff_t sstride_elems,
                            const int W, const int H,
                            uint8_t * dst, ptrdiff_t dstride_elems,
                            const int elem) {
    if (elem != 1) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                std::memcpy(dst + (static_cast<ptrdiff_t>(x) * dstride_elems + y) * elem,
                            src + (static_cast<ptrdiff_t>(y) * sstride_elems + x) * elem,
                            static_cast<size_t>(elem));
            }
        }
        return;
    }
    for (int x0 = 0; x0 < W; x0 += 16) {
        const int xn = std::min(16, W - x0);
        for (int y0 = 0; y0 < H; y0 += 16) {
            const int yn = std::min(16, H - y0);
            if (xn == 16 && yn == 16) {
                __m128i r[16];
                for (int y = 0; y < 16; ++y) {
                    r[y] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                        src + static_cast<size_t>(y0 + y) * sstride_elems + x0));
                }
                transpose16x16_u8(r);
                for (int x = 0; x < 16; ++x) {
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(
                        dst + static_cast<size_t>(x0 + x) * dstride_elems + y0), r[x]);
                }
            } else {
                for (int y = 0; y < yn; ++y) {
                    for (int x = 0; x < xn; ++x) {
                        dst[static_cast<size_t>(x0 + x) * dstride_elems + y0 + y] =
                            src[static_cast<size_t>(y0 + y) * sstride_elems + x0 + x];
                    }
                }
            }
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

// ---------------------------------------------------------------------------
// EEDI3H host gathers.
//
// The GPU kernels are orientation-agnostic, so EEDI3H simply feeds them the
// transposed plane: the host reads the source ROWS and writes the selected
// column parity compactly (K), and one GPU pass (ENTRY_XPOSE) transposes K into
// the pad builder's own layout. Every access stays a contiguous stream: the
// selected source columns are read at line granularity and the K rows are
// written whole.
// ---------------------------------------------------------------------------

// The even u16/f32 lanes of two 256-bit vectors, packed into one (16 or 8
// elements). Selecting every other source column is exactly what the parity
// gathers need, and it keeps the read side a full-width linear stream.
static inline __m256i deint_even_u16(const __m256i a, const __m256i b) {
    const __m256i m = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i sa = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(a, m), 0x08);
    const __m256i sb = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(b, m), 0x08);
    return _mm256_set_m128i(_mm256_castsi256_si128(sb), _mm256_castsi256_si128(sa));
}

static inline __m256i deint_even_f32(const __m256i a, const __m256i b) {
    const __m256i s = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0x88));
    return _mm256_permute4x64_epi64(s, 0xD8);
}

// The odd u16/f32 lanes of two 256-bit vectors (the complementary parity: the
// fused pair gather below needs both halves of one 64-byte row chunk).
static inline __m256i deint_odd_u16(const __m256i a, const __m256i b) {
    const __m256i m = _mm256_setr_epi8(
        2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i sa = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(a, m), 0x08);
    const __m256i sb = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(b, m), 0x08);
    return _mm256_set_m128i(_mm256_castsi256_si128(sb), _mm256_castsi256_si128(sa));
}

static inline __m256i deint_odd_f32(const __m256i a, const __m256i b) {
    const __m256i s = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0xDD));
    return _mm256_permute4x64_epi64(s, 0xD8);
}

// One deinterleaved row: d[k] = s[2*k + parity], k in [0,rows), with `s`
// pointing at the ROW START. Selecting the parity with the deinterleave (rather
// than offsetting the pointer) keeps every 32-byte load 32-byte aligned — a
// 2-byte offset makes each load straddle two cache lines. `nt` streams the
// stores (required for the uncached host-visible VRAM the ReBAR upload targets;
// the 32-byte-aligned body is what makes that worth doing).
static inline void deint_row_u16(const uint16_t * s, uint16_t * d, int rows,
                                 int simd_lim, bool nt, const bool odd) {
    int k = 0;
    if (nt) {
        const int head = std::min<int>(
            rows, static_cast<int>(((32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31) / 2));
        for (; k < head; ++k) {
            d[k] = s[2 * k + (odd ? 1 : 0)];
        }
    }
    for (; k + 16 <= simd_lim; k += 16) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + 2 * k));
        const __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(s + 2 * k + 16));
        const __m256i v = odd ? deint_odd_u16(va, vb) : deint_even_u16(va, vb);
        if (nt && (k & 15) == 0) {
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d + k), v);
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + k), v);
        }
    }
    for (; k < rows; ++k) {
        d[k] = s[2 * k + (odd ? 1 : 0)];
    }
}

static inline void deint_row_f32(const float * s, float * d, int rows,
                                 int simd_lim, bool nt, const bool odd) {
    int k = 0;
    if (nt) {
        const int head = std::min<int>(
            rows, static_cast<int>(((32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31) / 4));
        for (; k < head; ++k) {
            d[k] = s[2 * k + (odd ? 1 : 0)];
        }
    }
    for (; k + 8 <= simd_lim; k += 8) {
        const __m256i va = _mm256_castps_si256(_mm256_loadu_ps(s + 2 * k));
        const __m256i vb = _mm256_castps_si256(_mm256_loadu_ps(s + 2 * k + 8));
        const __m256i v = odd ? deint_odd_f32(va, vb) : deint_even_f32(va, vb);
        if (nt && (k & 7) == 0) {
            _mm256_stream_ps(d + k, _mm256_castsi256_ps(v));
        } else {
            _mm256_storeu_ps(d + k, _mm256_castsi256_ps(v));
        }
    }
    for (; k < rows; ++k) {
        d[k] = s[2 * k + (odd ? 1 : 0)];
    }
}

// dst[y*rows + k] = src[y*src_stride + first + step*k] for y in [0,H).
// step == 2 is the column-parity deinterleave; step == 1 (dh: every source
// column is kept) degenerates to a plain row copy.
static void gather_columns(const uint8_t * src, ptrdiff_t src_stride,
                           const int H, uint8_t * dst, const int rows,
                           const int step, const int first, const int elem,
                           const bool nt) {
    const size_t dst_row = static_cast<size_t>(rows) * elem;
    if (step == 1) {
        for (int y = 0; y < H; ++y) {
            frame_copy_out(dst + static_cast<size_t>(y) * dst_row,
                           src + static_cast<size_t>(y) * src_stride,
                           dst_row, nt);
        }
    } else if (elem == 2) {
        for (int y = 0; y < H; ++y) {
            const uint16_t * s = reinterpret_cast<const uint16_t *>(
                src + static_cast<size_t>(y) * src_stride);
            deint_row_u16(s, reinterpret_cast<uint16_t *>(dst + static_cast<size_t>(y) * dst_row),
                          rows, rows, nt, first != 0);
        }
    } else {
        for (int y = 0; y < H; ++y) {
            const float * s = reinterpret_cast<const float *>(
                src + static_cast<size_t>(y) * src_stride);
            deint_row_f32(s, reinterpret_cast<float *>(dst + static_cast<size_t>(y) * dst_row),
                          rows, rows, nt, first != 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused kept/interp column gather (EEDI3H, when the sclip frame IS the clip
// frame -- based_aa and the benchmark both pass Interleave([clip, clip]), so
// vsapi hands out the identical plane pointer).
//
// The separate gathers read every source row twice: once for the kept columns
// (parity `off`) and once for the interp columns (parity `field`). Both
// parities live in the same 64-byte row chunk, so one pass can produce both,
// halving the source read (measured +8% end-to-end at ns=8).
//
// Deinterleave from element 0 (even lanes = parity 0, odd = parity 1) so the
// loads are aligned, and require the destination rows to be 32-byte aligned
// and a multiple of the vector width, which makes every NT store aligned (the
// call site checks this and falls back to the two-pass path otherwise).
// ---------------------------------------------------------------------------

static void gather_pair_row_u16(const uint16_t * s, uint16_t * da, uint16_t * db,
                                const int rows, const bool a_even, const bool nt) {
    for (int k = 0; k < rows; k += 16) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + 2 * k));
        const __m256i b = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(s + 2 * k + 16));
        const __m256i ve = deint_even_u16(a, b);
        const __m256i vo = deint_odd_u16(a, b);
        if (nt) {
            _mm256_stream_si256(reinterpret_cast<__m256i *>(da + k), a_even ? ve : vo);
            _mm256_stream_si256(reinterpret_cast<__m256i *>(db + k), a_even ? vo : ve);
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(da + k), a_even ? ve : vo);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(db + k), a_even ? vo : ve);
        }
    }
}

static void gather_pair_row_f32(const float * s, float * da, float * db,
                                const int rows, const bool a_even, const bool nt) {
    for (int k = 0; k < rows; k += 8) {
        const __m256i a = _mm256_castps_si256(
            _mm256_loadu_ps(s + 2 * k));
        const __m256i b = _mm256_castps_si256(
            _mm256_loadu_ps(s + 2 * k + 8));
        const __m256i ve = deint_even_f32(a, b);
        const __m256i vo = deint_odd_f32(a, b);
        if (nt) {
            _mm256_stream_ps(da + k, _mm256_castsi256_ps(a_even ? ve : vo));
            _mm256_stream_ps(db + k, _mm256_castsi256_ps(a_even ? vo : ve));
        } else {
            _mm256_storeu_ps(da + k, _mm256_castsi256_ps(a_even ? ve : vo));
            _mm256_storeu_ps(db + k, _mm256_castsi256_ps(a_even ? vo : ve));
        }
    }
}

// dst_kept[y*rows + k] = src[y][first + 2k], dst_interp[y*rows + k] =
// src[y][1-first + 2k] (both compacted). Returns false when the geometry does
// not allow the aligned fast path (caller falls back to two gather_columns).
static bool gather_columns_pair(const uint8_t * src, ptrdiff_t src_stride,
                                const int H, uint8_t * dst_kept, uint8_t * dst_interp,
                                const int rows, const int first, const int elem,
                                const bool nt) {
    const size_t dst_row = static_cast<size_t>(rows) * elem;
    const int vec = (elem == 2) ? 16 : 8;
    if (rows % vec != 0 || (dst_row & 31) != 0 ||
        (reinterpret_cast<uintptr_t>(dst_kept) & 31) != 0 ||
        (reinterpret_cast<uintptr_t>(dst_interp) & 31) != 0) {
        return false;
    }
    const bool a_even = (first == 0);
    for (int y = 0; y < H; ++y) {
        const uint8_t * const row = src + static_cast<size_t>(y) * src_stride;
        uint8_t * const da = dst_kept + static_cast<size_t>(y) * dst_row;
        uint8_t * const db = dst_interp + static_cast<size_t>(y) * dst_row;
        if (elem == 2) {
            gather_pair_row_u16(reinterpret_cast<const uint16_t *>(row),
                                reinterpret_cast<uint16_t *>(da),
                                reinterpret_cast<uint16_t *>(db), rows, a_even, nt);
        } else {
            gather_pair_row_f32(reinterpret_cast<const float *>(row),
                                reinterpret_cast<float *>(da),
                                reinterpret_cast<float *>(db), rows, a_even, nt);
        }
    }
    return true;
}


// The even bytes of two 16-byte vectors, packed into one.
static inline __m128i deint_even_u8(const __m128i a, const __m128i b) {
    const __m128i m = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                    -1, -1, -1, -1, -1, -1, -1, -1);
    return _mm_unpacklo_epi64(_mm_shuffle_epi8(a, m), _mm_shuffle_epi8(b, m));
}

// Nonzero-byte predicate (0xFF/0x00), used for the reference's native Gray8
// mask form.
static inline __m128i nz_u8_mask(const __m128i v) {
    const __m128i zero = _mm_setzero_si128();
    return _mm_andnot_si128(_mm_cmpeq_epi8(v, zero), _mm_set1_epi8(static_cast<char>(0xFF)));
}

// Mask gather: dst[y*rows + k] = predicate(src[y*src_stride + first + step*k])
// as one byte per element (0xFF/0x00 for the vector path, 0/1 for the tail).
// The predicate matches build_bmask_row_scalar's nz() exactly (u16 >= 129,
// f32 > 0.5/255, byte != 0) so it is the same boolean the vertical path builds
// from the mask's native representation; the GPU never sees this matrix.
// step == 2 is the column-parity deinterleave, step == 1 (dh) keeps every
// column.
static void gather_mask_u8(const uint8_t * src, ptrdiff_t src_stride,
                           const int H, uint8_t * dst, const int rows,
                           const int first, const int step, const int bits) {
    const __m256i bias = _mm256_set1_epi16(static_cast<short>(0x8000));
    const __m256i thr16 = _mm256_set1_epi16(static_cast<short>(128 - 32768));
    for (int y = 0; y < H; ++y) {
        const uint8_t * const row = src + static_cast<size_t>(y) * src_stride;
        uint8_t * const d = dst + static_cast<size_t>(y) * rows;
        int k = 0;
        if (bits == 16) {
            const uint16_t * s = reinterpret_cast<const uint16_t *>(row) + first;
            const int lim = rows - (first ? 1 : 0);
            for (; k + 16 <= lim; k += 16) {
                // xor bias turns the unsigned >= 129 test into a signed compare
                // (cmpgt is strict: 128-32768 as the biased constant <=> v >= 129).
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + step * k));
                __m256i b = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(s + step * k + 16));
                const __m256i v = (step == 2) ? deint_even_u16(a, b) : a;
                const __m256i cmp = _mm256_cmpgt_epi16(_mm256_xor_si256(v, bias), thr16);
                // signed pack: cmp is all-ones (true) or zero, so the bytes
                // come out 0xFF/0x00 -- what build_bmask_row's byte path tests.
                const __m256i packed = _mm256_packs_epi16(cmp, cmp);
                const __m128i bytes = _mm256_castsi256_si128(
                    _mm256_permute4x64_epi64(packed, 0x08));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(d + k), bytes);
            }
            for (; k < rows; ++k) {
                d[k] = s[step * k] >= 129u ? 0xFF : 0x00;
            }
        } else if (bits == 32) {
            const float * s = reinterpret_cast<const float *>(row) + first;
            const __m256i thr = _mm256_castps_si256(_mm256_set1_ps(0.5f / 255.0f));
            const int lim = rows - (first ? 1 : 0);
            for (; k + 8 <= lim; k += 8) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + step * k));
                __m256i b = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(s + step * k + 8));
                const __m256i v = (step == 2) ? deint_even_f32(a, b) : a;
                const __m256i cmp = _mm256_castps_si256(_mm256_cmp_ps(
                    _mm256_castsi256_ps(v), _mm256_castsi256_ps(thr), _CMP_GT_OQ));
                const uint32_t mask = static_cast<uint32_t>(_mm256_movemask_ps(
                    _mm256_castsi256_ps(cmp)));
                uint64_t bytes = 0;
                for (int i = 0; i < 8; ++i) {
                    bytes |= static_cast<uint64_t>((mask >> i) & 1u) << (i * 8);
                }
                std::memcpy(d + k, &bytes, 8);
            }
            for (; k < rows; ++k) {
                d[k] = s[step * k] > 0.5f / 255.0f ? 0xFF : 0x00;
            }
        } else {
            // Gray8 (or a mask the reference converted to Gray8): the exact
            // predicate is "byte != 0".
            const uint8_t * s = row + first;
            const int lim = rows - (first ? 1 : 0);
            for (; k + 16 <= lim; k += 16) {
                const __m128i a = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(s + step * k));
                const __m128i b = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(s + step * k + 16));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(d + k),
                                 nz_u8_mask(step == 2 ? deint_even_u8(a, b) : a));
            }
            for (; k < rows; ++k) {
                d[k] = s[step * k] != 0 ? 0xFF : 0x00;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EEDI3H fused mask path.
//
// The transposed plane's mask row k is the source mask COLUMN
// `first + step*k`, so the byte matrix the vertical path feeds
// build_bmask_row has to be transposed first. Building that matrix and then
// transposing it moves ~4x more bytes than the bits the row kernel actually
// reads (a 4.15 MB byte matrix + a second 4.15 MB transpose copy in the 2x
// upscaled benchmark), and it was the single largest EEDI3H host stage
// (measured: skipping it entirely, +33% end-to-end). The pair below builds
// the PACKED bit matrix in one pass instead: threshold 16 mask elements into
// 16 predicate bytes, transpose the 16x16 byte tile in registers, and
// movemask it into a 64-bit word of the transposed bit row. Both global
// streams stay linear (mask rows in, bit rows out).
// ---------------------------------------------------------------------------

// 8 float predicates as 8 bytes (0xFF/0x00), the exact strict "> 0.5/255" test.
static inline __m128i nz_bytes_f32(const __m256 v) {
    const __m256 thr = _mm256_set1_ps(0.5f / 255.0f);
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(v, thr, _CMP_GT_OQ));
    const __m256i p16 = _mm256_packs_epi32(c, c);
    const __m256i p8 = _mm256_packs_epi16(p16, p16);
    return _mm256_castsi256_si128(p8);   // low 8 bytes = the 8 predicates
}

// 16 mask u16 -> 16 predicate bytes (bit-identical to bmask_bits16's test).
// `s` is the row start and `odd` selects the source parity via the
// deinterleave, so the loads stay aligned.
static inline __m128i nz_bytes_u16(const uint16_t * s, const int step, const bool odd) {
    const __m256i bias = _mm256_set1_epi16(static_cast<short>(0x8000));
    const __m256i thr = _mm256_set1_epi16(static_cast<short>(128 - 32768));
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s));
    __m256i v = a;
    if (step == 2) {
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + 16));
        v = odd ? deint_odd_u16(a, b) : deint_even_u16(a, b);
    }
    const __m256i cmp = _mm256_cmpgt_epi16(_mm256_xor_si256(v, bias), thr);
    const __m256i packed = _mm256_packs_epi16(cmp, cmp);
    return _mm256_castsi256_si128(_mm256_permute4x64_epi64(packed, 0x08));
}

// 16 mask f32 -> 16 predicate bytes.
static inline __m128i nz_bytes_f32x16(const float * s, const int step, const bool odd) {
    if (step == 2) {
        const __m256 a = _mm256_loadu_ps(s);
        const __m256 b = _mm256_loadu_ps(s + 8);
        const __m256 c = _mm256_loadu_ps(s + 16);
        const __m256 d = _mm256_loadu_ps(s + 24);
        const __m256i va = _mm256_castps_si256(a);
        const __m256i vb = _mm256_castps_si256(b);
        const __m256i vc = _mm256_castps_si256(c);
        const __m256i vd = _mm256_castps_si256(d);
        const __m256 e0 = _mm256_castsi256_ps(odd ? deint_odd_f32(va, vb)
                                                  : deint_even_f32(va, vb));
        const __m256 e1 = _mm256_castsi256_ps(odd ? deint_odd_f32(vc, vd)
                                                  : deint_even_f32(vc, vd));
        return _mm_unpacklo_epi64(nz_bytes_f32(e0), nz_bytes_f32(e1));
    }
    return _mm_unpacklo_epi64(nz_bytes_f32(_mm256_loadu_ps(s)),
                              nz_bytes_f32(_mm256_loadu_ps(s + 8)));
}

// 16 mask bytes -> 16 predicate bytes.
static inline __m128i nz_bytes_u8(const uint8_t * s, const int step, const bool odd) {
    if (step == 2) {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + 16));
        if (odd) {
            const __m128i m = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15,
                                            -1, -1, -1, -1, -1, -1, -1, -1);
            return nz_u8_mask(_mm_unpacklo_epi64(_mm_shuffle_epi8(a, m),
                                                 _mm_shuffle_epi8(b, m)));
        }
        return nz_u8_mask(deint_even_u8(a, b));
    }
    return nz_u8_mask(_mm_loadu_si128(reinterpret_cast<const __m128i *>(s)));
}

// One element, for the k-block that does not fit the 16-wide vector path.
static inline bool mask_nz_elem(const uint8_t * row, const int idx, const int bits) {
    if (bits == 16) {
        return reinterpret_cast<const uint16_t *>(row)[idx] >= 129u;
    }
    if (bits == 32) {
        return reinterpret_cast<const float *>(row)[idx] > 0.5f / 255.0f;
    }
    return row[idx] != 0;
}

// bitmat[k * bw + (y >> 6)] bit (y & 63) = predicate(mask[y][first + step*k])
// for k in [0, rows), y in [0, H). The bitmat row length is H bits; bw must be
// (H + 63) / 64 (the caller allocates rows * bw words). `bits` is the MASK's
// representation (16/32/8), not the clip's.
//
// wacc must hold `rows` words. The 64 rows of one y block are accumulated in
// four 16-row groups, and within a group the k sweep walks each row LINEARLY
// (only 16 row streams are live, which the hardware prefetcher tracks); the
// obvious k-outermost order keeps 64 streams open and measured 2x slower.
static void gather_mask_bitmat(const uint8_t * src, ptrdiff_t src_stride,
                               const int H, const int rows, const int first,
                               const int step, const int bits, uint64_t * bitmat,
                               const int bw, uint64_t * wacc) {
    // For step == 2 the source row is the kept/interp parity pair, i.e. 2*rows
    // elements wide; for step == 1 (dh: every column survives) it is `rows`.
    const int in_w = (step == 2) ? 2 * rows : rows;
    const int load_span = (step == 2) ? 32 : 16;
    for (int y0 = 0; y0 < H; y0 += 64) {
        std::memset(wacc, 0, static_cast<size_t>(rows) * sizeof(uint64_t));
        for (int g = 0; g < 4; ++g) {
            if (y0 + g * 16 >= H) {
                break;
            }
            for (int k0 = 0; k0 < rows; k0 += 16) {
                const int kn = std::min(16, rows - k0);
                // The 16-wide vector load spans step * k0 + load_span elements
                // from the ROW START (the parity is selected by the
                // deinterleave, so the loads stay aligned and the last block of
                // an aligned row is exactly in bounds); anything else falls
                // back to the scalar loop.
                const bool vec = step * k0 + load_span <= in_w;
                uint32_t m[16] = {};
                if (vec) {
                    __m128i r[16];
                    for (int i = 0; i < 16; ++i) {
                        const int y = y0 + g * 16 + i;
                        if (y >= H) {
                            r[i] = _mm_setzero_si128();
                            continue;
                        }
                        const uint8_t * const row =
                            src + static_cast<size_t>(y) * src_stride;
                        const int c = step * k0;
                        const bool odd = first != 0;
                        r[i] = (bits == 16)
                            ? nz_bytes_u16(reinterpret_cast<const uint16_t *>(row) + c, step, odd)
                            : (bits == 32)
                                ? nz_bytes_f32x16(reinterpret_cast<const float *>(row) + c, step, odd)
                                : nz_bytes_u8(row + c, step, odd);
                    }
                    transpose16x16_u8(r);
                    for (int j = 0; j < 16; ++j) {
                        m[j] = static_cast<uint32_t>(_mm_movemask_epi8(r[j]));
                    }
                } else {
                    for (int i = 0; i < 16; ++i) {
                        const int y = y0 + g * 16 + i;
                        if (y >= H) {
                            break;
                        }
                        const uint8_t * const row =
                            src + static_cast<size_t>(y) * src_stride;
                        for (int j = 0; j < kn; ++j) {
                            if (mask_nz_elem(row, first + step * (k0 + j), bits)) {
                                m[j] |= 1u << i;
                            }
                        }
                    }
                }
                for (int j = 0; j < kn; ++j) {
                    wacc[k0 + j] |= static_cast<uint64_t>(m[j]) << (g * 16);
                }
            }
        }
        const int yvalid = std::min(64, H - y0);
        const uint64_t ymask = (yvalid >= 64) ? ~0ull : ((1ull << yvalid) - 1);
        for (int k = 0; k < rows; ++k) {
            bitmat[static_cast<size_t>(k) * bw + (y0 >> 6)] = wacc[k] & ymask;
        }
    }
}

// The shift+dilate+pack step of build_bmask_row, shared with the EEDI3H fused
// path (which fills `scratch` from the transposed bit matrix instead of from
// mask bytes). scratch holds 2 * ((width + mdis + 63) / 64) words: the first
// nw are the input bits (zero past `width`), the second nw the dilation
// temporaries.
static void bmask_dilate_store(uint64_t * scratch, uint32_t * out,
                               const int width, const int mdis) {
    const int nw = (width + mdis + 63) / 64;             // accumulator words
    // 2. B[x] = b[x - mdis]: shift the bit array toward higher x by mdis
    //    (descending in place: src[i-1] is still untouched when i is written).
    if (mdis > 0) {
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
        uint64_t * const shifted = scratch + nw;
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

// Transposed bit row -> the row kernel's packed dilated mask row.
static void build_bmask_row_from_bits(const uint64_t * bits, uint32_t * out,
                                      const int width, const int mdis,
                                      uint64_t * scratch) {
    const int nw = (width + mdis + 63) / 64;
    const int bw = (width + 63) / 64;
    std::memcpy(scratch, bits, static_cast<size_t>(bw) * sizeof(uint64_t));
    std::memset(scratch + bw, 0, static_cast<size_t>(nw - bw) * sizeof(uint64_t));
    bmask_dilate_store(scratch, out, width, mdis);
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
    B_PADSRC = 9,     // ENTRY_PAD's source plane: the raw upload (vertical) or
                      // the GPU-transposed R'/B' in pad_dev (EEDI3H). The pad
                      // builder always reads binding 9, so switching the layout
                      // never needs a second pad-kernel variant.
    B_OUT = 10,       // ENTRY_COMPOSE's assembled output plane (staging); a
                      // separate binding from vout (b7) so vout can sit in
                      // device-local memory without dragging the output there.
    BIND_COUNT = 11
};

// Compile-time max plane width for the shared-memory (LDS) vcheck variant.
// Must match MAXW in src/eedi3.comp and the -DMAXW passed by the CMake
// vcheck_lds rule; wider planes fall back to the global-read vcheck.
static constexpr int MAXW_LDS = 4096;

struct PlaneConfig {
    int width {};                     // KERNEL plane width: the row kernel's
                                      // WIDTH spec constant and the EEDI3H
                                      // transposed row length
    int height {};                    // output plane height (vertical) / kernel
                                      // plane height (EEDI3H)
    int src_w {};                     // source plane dims (gather geometry)
    int src_h {};
    int out_w {};                     // output (frame order) plane dims
    int out_h {};
    int rows {};                      // number of interp rows == height / 2
    int pad_stride {};                // pad elements per padded row
    int pad_height {};                // padded rows == height + 2*MARGIN_V
    int tpitch {};                    // 2*mdis + 1

    VkPipeline row_pipeline {};
    VkPipeline vcheck_pipeline {};    // only when vcheck > 0
    VkPipeline pad_pipeline {};       // mirror-pad builder (always)
    VkPipeline vcopy_pipeline {};     // empty-row vcheck copy (vcheck+mclip)
    VkPipeline blit_pipeline {};      // direct-to-frame row spread (import path)
    VkPipeline xpose_pipeline {};     // compact upload -> R'/B' (EEDI3H only)
    VkPipeline compose_pipeline {};   // R' + vout -> output plane (EEDI3H only)
    // true when vcheck_pipeline is the shared-memory d2p variant: it carries
    // every row (no empty-row skip) and therefore needs no vcopy dispatch.
    bool vcheck_lds {};

    // staging (host-visible) regions, byte offsets: tight kept source rows,
    // gathered sclip rows (sclip only). The shared tight mask rows live in
    // Eedi3Data (maskraw_offset/bytes).
    VkDeviceSize raw_offset {};       // kept src rows (upload)
    VkDeviceSize raw_bytes {};
    VkDeviceSize sclip_offset {};     // sclip interp rows (upload; sclip only)
    VkDeviceSize sclip_bytes {};
    VkDeviceSize dl_offset {};        // interp row download (device -> host)
    VkDeviceSize dl_bytes {};
    // EEDI3H staging tail: the composed output plane and the CPU mask scratch
    // (deinterleaved rows + their transpose; never touched by the GPU).
    VkDeviceSize out_offset {};       // composed output plane (io elements)
    VkDeviceSize out_bytes {};
    VkDeviceSize ms_offset {};        // mask scratch, bytes (2 * rows * width)
    VkDeviceSize ms_bytes {};

    // device-mirror (pad_dev) regions: 1:1 copy of the upload [0,upload_total)
    // plus kernel-built regions (never staged)
    VkDeviceSize built_offset {};     // built padded plane (pad kernel out)
    VkDeviceSize built_bytes {};
    VkDeviceSize bits_offset {};      // packed dilation bits (bmask kernel out)
    VkDeviceSize bits_bytes {};
    // EEDI3H: transposed kept rows R' (pad kernel source) and transposed sclip
    // B' (vcheck source), both written by the xpose kernel in pad_dev.
    VkDeviceSize rt_offset {};        // R' (pad elements)
    VkDeviceSize rt_bytes {};
    VkDeviceSize rtS_offset {};       // B' (io elements)
    VkDeviceSize rtS_bytes {};

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
    // ------------------------------------------------------------------
    // EEDI3AA additions. The fused filter runs a vertical sub-pass twice (one
    // per field of the doubled output) and then a horizontal one twice, reusing
    // the same buffers; everything here is only allocated in aa mode.
    // ------------------------------------------------------------------
    VkPipeline assemble_pipeline {};  // ENTRY_ASSEMBLEV (vertical geometry)
    VkDeviceSize raw2_offset {};      // second compacted raw upload (other parity)
    VkDeviceSize raw2_bytes {};
    VkDeviceSize sclip2_offset {};    // second sub-pass's compacted sclip
    VkDeviceSize sclip2_bytes {};
    VkDeviceSize bits2_offset {};     // second sub-pass's packed mask bits
    VkDeviceSize bits2_bytes {};
    VkDeviceSize dst2_offset {};      // second sub-pass's row-kernel output (dev)
    VkDeviceSize dst2_bytes {};
    VkDeviceSize vout2_offset {};     // second sub-pass's vcheck output (dev)
    VkDeviceSize vout2_bytes {};
    VkDeviceSize v_offset {};         // merged vertical frame (staging scratch)
    VkDeviceSize v_bytes {};
    VkDeviceSize out2_offset {};      // second composed horizontal plane (staging;
                                      // the first one reuses out_offset above)
    VkDeviceSize out2_bytes {};
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
    // EEDI3AA needs BOTH geometries' views of the shared buffers in one filter:
    //   desc_set_h  — horizontal row/vcheck/compose: b0/b5/b9 all view pad_dev
    //                 (the built pad, the transposed sclip B' and R');
    //   desc_set_xp — xpose + the horizontal pad builder: b0 = the raw upload
    //                 (K), b8 = pad_dev (built pad), b9 = pad_dev (R').
    // The historical single-geometry filters get the same effect from
    // d->horiz, which is fixed per filter instance.
    VkDescriptorSet desc_set_h {};
    VkDescriptorSet desc_set_xp {};
    VkDescriptorSet desc_set_blit[MAX_PLANES] {};  // b1 = imported output frame
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
    // EEDI3H: run the whole pipeline on the TRANSPOSED plane. The EEDI3
    // kernels are orientation-agnostic (the pad builder, row kernel, vcheck and
    // vcopy all just see a padded plane), so nothing in the kernels changes:
    // the host deinterleaves the kept source columns into K, one GPU transpose
    // (ENTRY_XPOSE) puts them in the pad builder's own layout, and one GPU pass
    // (ENTRY_COMPOSE) reassembles the output plane. This removes the four
    // std.Transpose passes that used to be the entire EEDI3H penalty.
    bool horiz { false };
    // EEDI3AA: fused vertical-then-horizontal EEDI3 with the two 50/50 merges
    // folded in (the exact based_aa chain, one plugin call per output frame).
    // The vertical geometry lives in `planes` (the historical layout, now the
    // buffer set both passes reuse), the horizontal pass's geometry in
    // `aplanes`. Specialised: dh=false, field>1, single-rate output.
    bool aa { false };
    std::array<PlaneConfig, MAX_PLANES> aplanes {};

    // Which copies use non-temporal load/store (bit0 = upload gathers, bit1 =
    // the final blit). Swept with VSFEEL_EEDI3_COPY; the default is measured.
    int copy_mode { 3 };
    // Diagnostics-only host-path ablations (env-gated; all default off).
    bool skip_blit {}, skip_sclip {}, skip_raw {}, skip_h2d {}, skip_vcheck {};
    bool skip_xfer {};   // diagnostics: import path without the blit dispatch
    bool skip_xpose {}, skip_compose {}, skip_maskx {};  // diagnostics (EEDI3H)
    // EEDI3H only: build the transposed mask's PACKED bit matrix in one pass
    // (threshold + 16x16 byte transpose + movemask) instead of the byte matrix
    // plus its whole-plane transpose. VSFEEL_EEDI3_MASKFUSE=0 restores the old
    // two-pass path for A/B.
    bool mask_fuse { true };
    // EEDI3H only: when the sclip frame aliases the clip frame (based_aa's
    // Interleave([clip, clip])), gather the kept AND interp column parities in
    // one pass over the source rows instead of reading the whole frame twice.
    // VSFEEL_EEDI3_PAIR=0 forces the two-pass form.
    bool skip_pair { false };

    bool raw_stage {};   // diagnostics: raw gather -> cached staging (plain stores)
    bool blit_contig {}; // diagnostics: one contiguous copy instead of per-row
    // Gray16 mclip handled natively (no SetFrameProps+resize.Point->Gray8 node).
    bool mclip_native16 {};
    // Gray32 float mclip handled natively too. based_aa passes mclip in the
    // CLIP's format, so a float clip gets a float mask; eedi3vk2 consumes that
    // directly, and forcing it through the reference conversion node made
    // vsfeel pay a whole extra full-frame pass that based_aa does not require.
    // The exact predicate is in build_bmask_row.
    bool mclip_native32 {};
    bool skip_pad {};    // diagnostics: skip the pad-builder dispatch
    // ENTRY_PAD stores only the pad rows pad_get can read (parity 1-field:
    // every interp row's +/-{1,3} taps). The other parity is written and never
    // read, so skipping it halves the pad kernel's stores and mirror math.
    // Bit-exact by construction; VSFEEL_EEDI3_PADPAR=0 builds the whole plane
    // for A/B.
    bool pad_skip_parity { true };
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
    // Direct-to-frame output: write the interp rows into the output frame's
    // own memory, imported with VK_EXT_external_memory_host, instead of
    // downloading them and blitting on the CPU. It removes the whole ~20% dst
    // blit (a staging read plus a frame write) at the cost of one host pointer
    // import per frame, and is BIT-EXACT (tests pass either way) -- but it is
    // DEFAULT OFF because the import is not cheap on this platform: RADV/
    // amdgpu does GPU-visible VM work for every import and destroy, and at
    // num_streams=8 that saturates the GPU. Measured on the 2x-upscaled jpbd AA
    // benchmark (field=3, 2000f, ns=8, harness cache 500 frames):
    //   staged 608 | staged + vout in dev_buf (no import) 588 |
    //   per-frame import, GPU never touches the import 487 | import + blit 390
    // i.e. the import alone costs ~20%, more than the 20% blit it removes. It
    // only wins at low frame-buffer churn (harness cache 50: 697 vs 641) and
    // at low stream counts (ns=2: 324 vs 287). Re-test if RADV's import path
    // gets cheaper. VSFEEL_EEDI3_DSTHOST=1 enables it, VSFEEL_EEDI3_DSTHOST=0
    // forces the CPU blit.
    bool dst_host { false };
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
    VkShaderModule vcheck_lds_module {};  // shared-memory d2p variant (optional)
    VkShaderModule pad_module {};
    VkShaderModule vcopy_module {};
    VkShaderModule blit_module {};    // direct-to-frame row spread (import path)
    VkShaderModule xpose_module {};   // EEDI3H upload transpose (K -> R'/B')
    VkShaderModule compose_module {}; // EEDI3H output assembly (R' + vout)
    VkShaderModule assemble_module {}; // EEDI3AA vertical merge (src + vout -> v)
    VkDeviceSize upload_total {}, download_total {}, dev_total {};
    // EEDI3H staging tail (composed plane + CPU mask scratch), after the
    // upload and download regions.
    VkDeviceSize scratch_total {};
    std::array<PlaneConfig, MAX_PLANES> planes {};
    FramePool<Eedi3Resource> pool;

    // module/pipeline cache keyed by width (values owned by the pipelines
    // map): [row, vcheck, pad, vcopy, blit, _, xpose, compose]
    std::vector<std::pair<WidthKey, std::array<VkPipeline, 8>>> width_pipes {};
    // EEDI3AA: the vertical-merge pipeline depends only on the vertical width,
    // so it is cached here rather than in the (rows/pad-shaped) WidthKey map.
    std::vector<std::pair<int, VkPipeline>> assemble_pipes {};
    // vout lives in dev_buf whenever the direct-to-frame path is compiled in
    // (the transfer needs a device-local source), which also makes the
    // pre-existing VOUTDEV layout the fallback for frames that cannot be
    // imported. Kept consistent between descriptor creation and recording.
    // EEDI3H sets vout_dev by default: its only consumer is ENTRY_COMPOSE, and
    // reading vout back over PCIe from the staging download costs ~10% of the
    // frame. The assembled plane has its own binding, so it is unaffected.
    bool vout_in_dev() const {
        return vout_dev || (!horiz && dst_host && device && device->host_import);
    }

    ~Eedi3Data() {
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        // retire this instance's own submissions (per queue) instead of
        // idling the whole device, which other filters may be sharing
        retire_instance(pool);

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

        VkPipeline destroyed[8 * 3] {};
        int nd = 0;
        for (auto & [key, quad] : width_pipes) {
            for (int i = 0; i < 8; ++i) {
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
        if (vcheck_lds_module) {
            vkDestroyShaderModule(dev, vcheck_lds_module, nullptr);
        }
        if (pad_module) {
            vkDestroyShaderModule(dev, pad_module, nullptr);
        }
        if (vcopy_module) {
            vkDestroyShaderModule(dev, vcopy_module, nullptr);
        }
        if (blit_module) {
            vkDestroyShaderModule(dev, blit_module, nullptr);
        }
        if (xpose_module) {
            vkDestroyShaderModule(dev, xpose_module, nullptr);
        }
        if (compose_module) {
            vkDestroyShaderModule(dev, compose_module, nullptr);
        }
        if (assemble_module) {
            vkDestroyShaderModule(dev, assemble_module, nullptr);
        }
        // EEDI3AA assemble pipelines (keyed by vertical width, so not part of
        // the shared row/vcheck/pad pipeline cache).
        for (auto & [w, p] : assemble_pipes) {
            if (p) {
                vkDestroyPipeline(dev, p, nullptr);
            }
        }

        release_device(device);
    }
};

// ---------------------------------------------------------------------------
// Direct-to-frame output (VK_EXT_external_memory_host)
//
// The final interp rows are written by ENTRY_BLIT straight into the output
// frame, instead of being downloaded to staging and blitted row by row by the
// CPU. Both legs of the old CPU blit (a staging read plus a dst write, ~20% of
// the frame) disappear; the new cost is one host-pointer import of the plane
// per frame plus the blit dispatch itself.
//
// VapourSynth plane pointers are 64-byte aligned but not page aligned (they
// sit 128 bytes into their page on this build) while
// minImportedHostPointerAlignment is the page size (4096). The import
// therefore starts at the enclosing page and the buffer is bound at the
// plane's byte offset inside that region: the offset stays a multiple of the
// buffer's alignment requirement, and every byte the GPU touches lies inside
// the plane itself, so no page outside the frame allocation is ever used.
// ---------------------------------------------------------------------------

struct Eedi3HostImport {
    VkBuffer buffer {};
    VkDeviceMemory memory {};
};

static std::optional<std::string> import_plane_host_memory(
    const Eedi3Data & d, void * plane, const VkDeviceSize plane_bytes,
    Eedi3HostImport & out) {

    const VK_Device & dev = *d.device;
    if (!dev.host_import || dev.host_pointer_alignment == 0 ||
        !dev.get_memory_host_pointer_properties) {
        return "host pointer import unavailable";
    }
    const VkDeviceSize align = dev.host_pointer_alignment;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(plane);
    const uintptr_t base = addr & ~(static_cast<uintptr_t>(align) - 1);
    const VkDeviceSize offset = addr - base;
    // both the address and the size must be a multiple of the alignment
    const VkDeviceSize region = ((offset + plane_bytes) + align - 1) / align * align;

    VkMemoryHostPointerPropertiesEXT props {
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
        .pNext = nullptr,
        .memoryTypeBits = 0
    };
    if (dev.get_memory_host_pointer_properties(dev.device,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            reinterpret_cast<void *>(base), &props) != VK_SUCCESS) {
        return "vkGetMemoryHostPointerPropertiesEXT failed";
    }

    uint32_t type_index = ~0u;
    for (uint32_t i = 0; i < dev.mem_props.memoryTypeCount; ++i) {
        if (!(props.memoryTypeBits & (1u << i))) {
            continue;
        }
        const auto flags = dev.mem_props.memoryTypes[i].propertyFlags;
        if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            type_index = i;
            break;
        }
    }
    if (type_index == ~0u) {
        return "no coherent host-visible memory type accepts the plane pointer";
    }

    VkBufferCreateInfo buffer_info {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = plane_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr
    };
    if (vkCreateBuffer(dev.device, &buffer_info, nullptr, &out.buffer) != VK_SUCCESS) {
        return "vkCreateBuffer for the imported plane failed";
    }

    VkImportMemoryHostPointerInfoEXT import_info {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = reinterpret_cast<void *>(base)
    };
    VkMemoryAllocateInfo alloc_info {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = region,
        .memoryTypeIndex = type_index
    };
    VkResult result = vkAllocateMemory(dev.device, &alloc_info, nullptr, &out.memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(dev.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return "vkAllocateMemory (host import) failed: "s + vk_result_string(result);
    }
    result = vkBindBufferMemory(dev.device, out.buffer, out.memory, offset);
    if (result != VK_SUCCESS) {
        vkFreeMemory(dev.device, out.memory, nullptr);
        vkDestroyBuffer(dev.device, out.buffer, nullptr);
        out = {};
        return "vkBindBufferMemory (host import) failed: "s + vk_result_string(result);
    }
    return std::nullopt;
}

static void destroy_host_import(const Eedi3Data & d, Eedi3HostImport & imp) {
    VkDevice dev = d.device->device;
    if (imp.memory) {
        vkFreeMemory(dev, imp.memory, nullptr);
    }
    if (imp.buffer) {
        vkDestroyBuffer(dev, imp.buffer, nullptr);
    }
    imp = {};
}

// Per-frame direct-output targets: the imported output frame of each processed
// plane plus the frame's own row pitch in elements.
struct Eedi3Direct {
    Eedi3HostImport plane[MAX_PLANES] {};
    int stride[MAX_PLANES] {};                 // frame row pitch in elements
    VkDeviceSize vout_offset[MAX_PLANES] {};   // tight source rows in dev_buf
    bool active[MAX_PLANES] {};

    void destroy(const Eedi3Data & d) {
        for (int p = 0; p < MAX_PLANES; ++p) {
            destroy_host_import(d, plane[p]);
            active[p] = false;
        }
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
                                   const float * maskf,
                                   uint32_t * out, const int width, const int mdis) {
    // Native masks use the exact bit-depth-reduction threshold (see
    // build_bmask_row): nonzero-after-conversion == (v >= 129) for Gray16 and
    // == (v > 0.5/255) for a full-range Gray32 float mask.
    //
    // Float masks are only required to live in [0,1] (based_aa's are exactly
    // 0.0/1.0), and the predicate is exact there. One deliberate divergence:
    // zimg's float->8-bit reduction is a saturating SIMD convert, so a value
    // above ~8.42e6 (where v*255+0.5 overflows int32) converts to byte 0.
    // Reproducing that would make vsfeel depend on an unrelated library's
    // integer-overflow behavior, and eedi3vk2 -- which consumes the float mask
    // directly -- treats such a value as nonzero. The native path follows
    // eedi3vk2.
    auto nz = [&](int x) {
        if (maskf) {
            // STRICT >: measured on the real conversion, the first float whose
            // converted byte is nonzero is 0x1.01104p-9 = nextafter(0.5f/255),
            // so ">= 0.5f/255" is off by one ULP (round 11's cmpgt lesson).
            return maskf[x] > 0.5f / 255.0f;
        }
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
                            const float * maskf,
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
        build_bmask_row_scalar(maskp, mask16, maskf, out, width, mdis);
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
    if (maskf) {
        // Float mask: the conversion node computed round(clamp(v,0,1)*255) and
        // tested != 0, which holds exactly for v > 0.5f/255 (the boundary float
        // is nextafter(0.5f/255) — see build_bmask_row_scalar). Strict compare,
        // so no multiply: the threshold IS the comparison constant.
        const __m256 thresh = _mm256_set1_ps(0.5f / 255.0f);
        for (int i = 0; i < nw_b; ++i) {
            const int x = i * 64;
            uint64_t w = 0;
            for (int g = 0; g < 8; ++g) {
                const int xb = x + g * 8;
                uint32_t m = 0;
                if (xb + 8 <= width) {
                    m = static_cast<uint32_t>(_mm256_movemask_ps(
                        _mm256_cmp_ps(_mm256_loadu_ps(maskf + xb), thresh,
                                      _CMP_GT_OQ)));
                } else if (xb < width) {
                    float tmp[8] = {};
                    std::memcpy(tmp, maskf + xb,
                                static_cast<size_t>(width - xb) * sizeof(float));
                    m = static_cast<uint32_t>(_mm256_movemask_ps(
                        _mm256_cmp_ps(_mm256_loadu_ps(tmp), thresh,
                                      _CMP_GT_OQ)));
                }
                w |= static_cast<uint64_t>(m) << (g * 8);
            }
            scratch[i] = w;
        }
    } else if (mask16) {
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

    bmask_dilate_store(scratch, out, width, mdis);
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
        dev.device, dev.pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
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
    // ENTRY_BLIT only: output frame row pitch in io elements (b1 is the
    // imported output frame, b7 holds the tight interp rows).
    int32_t dst_stride;
    // ENTRY_PAD only: skip pad rows of parity `field` (never read; see the pad
    // kernel). 1 = skip (default), 0 = build the whole plane.
    int32_t pad_skip_parity;
    // ENTRY_COMPOSE only: io-element base of the assembled output plane in the
    // staging download region (binding 10; vout is binding 7).
    int32_t out_base;
    // ENTRY_ASSEMBLEV only (EEDI3AA): the second sub-pass's vout rows (b7) and
    // the second compacted raw upload (b9). Both sub-passes of the vertical
    // pass share the buffers, so their bases are push constants rather than
    // extra bindings.
    int32_t vout2_base;
    int32_t raw2_base;
};
static_assert(sizeof(PushConstants) == 19 * 4 + 8 * 4, "push constants size");

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

// What a recorded pass does after the row kernel + vcheck:
//   kTransfer — copy the interp rows back to staging (the historical default,
//               direct-to-frame disabled);
//   kDirect   — ENTRY_BLIT straight into imported output frames;
//   kCompose  — EEDI3H: assemble the frame-order plane from R' + vout;
//   kAssembleV— EEDI3AA: merge src + both fields' vout into the intermediate
//               vertical frame v (one dispatch per plane);
//   kNone     — EEDI3AA's first vertical sub-pass: nothing (the second one
//               assembles).
enum class PassTail { kTransfer, kDirect, kCompose, kAssembleV, kNone };

// One EEDI3 sub-pass: pad build -> row kernel -> vcheck, plus the tail above.
// The command buffer must already be recording; the caller owns begin/end and
// the (optional) H2D mirror copy, so a fused filter can chain several passes
// into one submission. `planes` is the geometry set to run (vertical or the
// EEDI3AA horizontal one), `second` selects the second sub-pass's dst/raw
// buffers (EEDI3AA only).
static std::optional<std::string> record_pass(
    const Eedi3Data & d, Eedi3Resource & resource, const int field,
    const std::array<PlaneConfig, MAX_PLANES> & planes, const bool horiz,
    const bool second, const PassTail tail, const Eedi3Direct & direct) {

    const int32_t elem = d.elem_bytes;

    // Sub-passes reuse pbt / dst / built pad / R', so a full compute barrier
    // separates this pass from the previous one's reads and writes.
    {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
    }

    // Upload kernel: build the padded planes from the raw upload (per plane).
    // The command buffer is recorded per frame because the interp-row parity
    // `field` varies (field > 1 doubles frames and _FieldBased sources).
    const int32_t pad_elem = pad_elem_bytes(d.bits);
    // The row/vcheck/compose kernels read the BUILT pad at b0 and, on the
    // horizontal geometry, the transposed sclip/R' at b5/b9 (all pad_dev).
    VkDescriptorSet const row_set = (horiz && d.aa)
        ? resource.desc_set_h : resource.desc_set;
    // The pad kernel (and EEDI3AA's merge writer) reads its source at
    // binding 9: the host-uploaded raw kept rows in the ReBAR/non-ReBAR upload
    // buffer vertically, or the device-local R'/B' regions EEDI3H produces.
    // Under ReBAR that needs the pad kernel's own set (the row kernel reads the
    // BUILT pad from pad_dev at the same binding index as the row kernel's
    // pad); the AA horizontal geometry needs desc_set_xp (b9 = pad_dev).
    VkDescriptorSet const pad_set = (horiz && d.aa)
        ? resource.desc_set_xp
        : (resource.desc_set_pad ? resource.desc_set_pad : resource.desc_set);
    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane] || d.skip_pad) {
            continue;
        }
        const auto & cfg = planes[plane];

        if (horiz) {
            // ENTRY_XPOSE: the host deinterleaved the source columns into K
            // (binding 0, `rows` elements per row, `width` rows); transpose it
            // into R' (binding 9, pad_dev) so the pad builder sees exactly the
            // layout it has always read. Same transpose for the sclip K -> B'
            // when the vcheck needs one.
            auto xpose = [&](VkDeviceSize src_off, VkDeviceSize dst_off) {
                PushConstants xpc {};
                xpc.raw_base = static_cast<int32_t>(src_off / pad_elem);
                xpc.pad_base = static_cast<int32_t>(dst_off / pad_elem);
                xpc.rows = cfg.rows;
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  cfg.xpose_pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &pad_set, 0, nullptr);
                vkCmdPushConstants(
                    resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(xpc), &xpc);
                const uint32_t gx = (static_cast<uint32_t>(cfg.rows) + 15) / 16;
                const uint32_t gy = (static_cast<uint32_t>(cfg.width) + 15) / 16;
                vkCmdDispatch(resource.cmd, gx, gy, 1);
            };
            if (!d.skip_xpose) {
                // `second` selects the second sub-pass's K / sclip compaction;
                // R' itself is a single scratch region (the sub-passes are
                // strictly sequential inside the pass).
                xpose(second ? cfg.raw2_offset : cfg.raw_offset, cfg.rt_offset);
                if (cfg.rtS_bytes > 0) {
                    xpose(second ? cfg.sclip2_offset : cfg.sclip_offset, cfg.rtS_offset);
                }
            }
            VkMemoryBarrier xbarrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xbarrier,
                0, nullptr, 0, nullptr);
        }

        PushConstants ppc {};
        ppc.pad_base = static_cast<int32_t>(cfg.built_offset / pad_elem);
        ppc.pad_stride = cfg.pad_stride;
        ppc.pad_height = cfg.pad_height;
        ppc.field = field;
        ppc.rows = cfg.rows;
        ppc.raw_base = static_cast<int32_t>(
            (horiz ? cfg.rt_offset : (second ? cfg.raw2_offset : cfg.raw_offset)) / pad_elem);
        ppc.pad_skip_parity = d.pad_skip_parity ? 1 : 0;

        // pad builder: one thread per padded element
        vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pad_pipeline);
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
        const auto & cfg = planes[plane];

        PushConstants pc {
            .pad_base = static_cast<int32_t>(cfg.built_offset / pad_elem_bytes(d.bits)),
            .dst_base = static_cast<int32_t>(
                (second ? cfg.dst2_offset : cfg.dst_offset) / elem),
            .pbt_base = static_cast<int32_t>(cfg.pbt_offset),
            .dmap_base = static_cast<int32_t>(cfg.dmap_offset),
            .bmask_base = static_cast<int32_t>(
                (second ? cfg.bits2_offset : cfg.bits_offset) / sizeof(uint32_t)),
            // EEDI3H keeps the vcheck's sclip in the GPU-transposed B' region
            // (binding 5 points at pad_dev for that mode); vertically it is the
            // host-gathered compact sclip in the upload mirror.
            .sclip_base = static_cast<int32_t>(
                (horiz ? cfg.rtS_offset
                       : (second ? cfg.sclip2_offset : cfg.sclip_offset)) / elem),
            .cint_base = static_cast<int32_t>(cfg.cint_offset / elem),
            // vout either lands DIRECTLY in the staging download region (b7
            // views staging: no D2H copy, but GPU stores go to system RAM) or
            // in a device-local region that a transfer brings back (SDMA to
            // staging, or the direct-to-frame copy below).
            .vout_base = static_cast<int32_t>(d.vout_in_dev()
                ? ((second ? cfg.vout2_offset : cfg.vout_offset) / elem)
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
            d.pipeline_layout, 0, 1, &row_set, 0, nullptr);
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
            // non-empty rows and pays ~1/3 of the barriers. The LDS walk
            // carries every row (it needs the whole chain), so it needs no
            // vcopy pass at all.
            if (d.mclip_node && !cfg.vcheck_lds) {
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.vcopy_pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &row_set, 0, nullptr);
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
                d.pipeline_layout, 0, 1, &row_set, 0, nullptr);
            vkCmdPushConstants(
                resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(pc), &pc);
            vkCmdDispatch(resource.cmd, 1, 1, 1);
        }
    }

    // Tail. Shapes per plane:
    //   direct       -> an ENTRY_BLIT dispatch spreads vout (dev_buf, tight,
    //                   b7) into the imported output frame (b1) at the frame's
    //                   own row pitch and interp parity;
    //   vout_dev     -> vout (dev_buf) is copied back to staging as one region;
    //   staging vout -> the vcheck kernels already wrote staging: nothing here.
    // Without vcheck the row kernel's dst (dev_buf) is the transfer source.
    // EEDI3H: the interp values are rows of the TRANSPOSED plane, so they are
    // reassembled into the frame-order output plane by ENTRY_COMPOSE.
    if (tail == PassTail::kNone) {
        return std::nullopt;
    }

    if (tail == PassTail::kAssembleV) {
        // EEDI3AA: merge the two sub-passes' interp rows with the source rows
        // into the intermediate frame v, which the host then column-gathers for
        // the horizontal pass. This pass was recorded with the SECOND
        // sub-pass's parity, so the first sub-pass's field is its complement.
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
        // vout is always device-local on this path (vout_dev is forced on).
        const int32_t field0 = 1 - field;
        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
            if (!d.process[plane] || !planes[plane].assemble_pipeline) {
                continue;
            }
            const auto & cfg = planes[plane];
            PushConstants apc {};
            apc.raw_base = static_cast<int32_t>(cfg.raw_offset / pad_elem);
            apc.raw2_base = static_cast<int32_t>(cfg.raw2_offset / pad_elem);
            apc.vout_base = static_cast<int32_t>(cfg.vout_offset / elem);
            apc.vout2_base = static_cast<int32_t>(cfg.vout2_offset / elem);
            apc.out_base = static_cast<int32_t>(
                (d.upload_total + d.download_total + cfg.v_offset) / elem);
            apc.field = field0;
            apc.rows = cfg.rows;
            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              cfg.assemble_pipeline);
            vkCmdBindDescriptorSets(
                resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &pad_set, 0, nullptr);
            vkCmdPushConstants(
                resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(apc), &apc);
            const uint32_t total = 2u * static_cast<uint32_t>(cfg.rows) *
                static_cast<uint32_t>(cfg.width);
            vkCmdDispatch(resource.cmd, (total + 255) / 256, 1, 1);
        }
        return std::nullopt;
    }

    if (tail == PassTail::kCompose) {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
            if (!d.process[plane] || d.skip_compose) {
                continue;
            }
            const auto & cfg = planes[plane];
            PushConstants cpc {};
            // b9 = R' (device-local), b7 = vout, b10 = the assembled plane
            // (the staging download). Without a vcheck the interp values are
            // the row kernel's dst (b1) instead -- a spec constant selects it.
            cpc.pad_base = static_cast<int32_t>(cfg.rt_offset / elem);
            // Same rule as the row/vcheck push constants: vout is either the
            // device-local region or the staging download, and binding 7 was
            // created for exactly one of them.
            cpc.vout_base = static_cast<int32_t>(d.vout_in_dev()
                ? ((second ? cfg.vout2_offset : cfg.vout_offset) / elem)
                : ((d.upload_total + cfg.dl_offset) / elem));
            cpc.dst_base = static_cast<int32_t>(
                (second ? cfg.dst2_offset : cfg.dst_offset) / elem);
            cpc.out_base = static_cast<int32_t>(
                (d.upload_total + d.download_total +
                 (second ? cfg.out2_offset : cfg.out_offset)) / elem);
            cpc.field = field;
            cpc.rows = cfg.rows;
            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              cfg.compose_pipeline);
            vkCmdBindDescriptorSets(
                resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &row_set, 0, nullptr);
            vkCmdPushConstants(
                resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(cpc), &cpc);
            const uint32_t gx = (static_cast<uint32_t>(cfg.rows) + 15) / 16;
            const uint32_t gy = (static_cast<uint32_t>(cfg.width) + 15) / 16;
            vkCmdDispatch(resource.cmd, gx, gy, 1);
        }
        return std::nullopt;
    }

    const bool direct_tail = (tail == PassTail::kDirect);
    bool any_direct = false, any_d2h = false;
    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
            continue;
        }
        if (direct_tail && direct.active[plane] && !d.skip_xfer &&
            planes[plane].blit_pipeline) {
            any_direct = true;
        } else if ((!direct_tail || !direct.active[plane]) &&
                   ((d.vcheck > 0) ? d.vout_in_dev() : (d.download_total > 0))) {
            any_d2h = true;
        }
    }
    if (any_direct || any_d2h) {
        VkMemoryBarrier mem_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = static_cast<VkAccessFlags>(
                (any_direct ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) : 0) |
                (any_d2h ? VK_ACCESS_TRANSFER_READ_BIT : 0))
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            static_cast<VkPipelineStageFlags>(
                (any_direct ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0) |
                (any_d2h ? VK_PIPELINE_STAGE_TRANSFER_BIT : 0)),
            0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

        for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
            if (!d.process[plane]) {
                continue;
            }
            const auto & cfg = planes[plane];
            if (direct_tail && direct.active[plane] && !d.skip_xfer &&
                cfg.blit_pipeline) {
                PushConstants bpc {};
                bpc.dst_base = 0;   // the imported buffer starts at the plane
                bpc.dst_stride = direct.stride[plane];
                bpc.vout_base = static_cast<int32_t>(direct.vout_offset[plane] / elem);
                bpc.field = field;
                bpc.rows = cfg.rows;
                vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  cfg.blit_pipeline);
                vkCmdBindDescriptorSets(
                    resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    d.pipeline_layout, 0, 1, &resource.desc_set_blit[plane], 0, nullptr);
                vkCmdPushConstants(
                    resource.cmd, d.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(bpc), &bpc);
                const uint32_t total = static_cast<uint32_t>(cfg.rows) *
                    static_cast<uint32_t>(cfg.width);
                vkCmdDispatch(resource.cmd, (total + 255) / 256, 1, 1);
                continue;
            }
            if ((direct_tail && direct.active[plane]) ||
                !((d.vcheck > 0) ? d.vout_in_dev() : (d.download_total > 0))) {
                continue;
            }
            const VkBufferCopy region {
                .srcOffset = (d.vcheck > 0) ? cfg.vout_offset : cfg.dst_offset,
                .dstOffset = d.upload_total + cfg.dl_offset,
                .size = cfg.dl_bytes
            };
            vkCmdCopyBuffer(resource.cmd, resource.dev_buf, resource.staging, 1, &region);
        }
    }

    return std::nullopt;
}

// The historical single-pass entry point: one EEDI3 sub-pass per frame. The
// EEDI3AA path hoists the begin/end and the H2D mirror copy so it can chain
// several sub-passes into one submission; this wrapper keeps EEDI3/EEDI3H
// exactly as they were.
static void record_h2d_copy(const Eedi3Data & d, Eedi3Resource & resource) {
    // Upload: with the ReBAR path the CPU has ALREADY NT-stored the upload
    // region straight into host-visible VRAM (resource.up_dev), so there is
    // nothing to copy here and no transfer->compute barrier to pay -- the
    // kernels read up_dev directly. The host-side stores are drained by an
    // _mm_sfence() before submit. VSFEEL_EEDI3_NOREBAR=1 restores the old
    // staging -> pad_dev DMA.
    if (d.upload_total == 0 || d.skip_h2d || d.rebar_up) {
        return;
    }
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

// The mode's tail: EEDI3H assembles; the direct-to-frame path (only selected
// when dst_host imported at least one plane) blits; everything else downloads
// the interp rows.
static PassTail pass_tail(const Eedi3Data & d) {
    if (d.horiz) {
        return PassTail::kCompose;
    }
    if (d.dst_host && d.device->host_import && !d.skip_blit) {
        return PassTail::kDirect;
    }
    return PassTail::kTransfer;
}

static std::optional<std::string> record_command_buffer(
    const Eedi3Data & d, Eedi3Resource & resource, const int field,
    Eedi3Direct & direct) {

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };

    if (vkBeginCommandBuffer(resource.cmd, &begin_info) != VK_SUCCESS) {
        return "vkBeginCommandBuffer failed";
    }

    record_h2d_copy(d, resource);

    if (const auto err = record_pass(d, resource, field, d.planes, d.horiz,
                                     false, pass_tail(d), direct)) {
        return err;
    }

    if (vkEndCommandBuffer(resource.cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer failed";
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// EEDI3AA host gathers
//
// The fused filter runs the vertical pass on the source clip and the horizontal
// pass on the merged vertical frame `v` it produces internally. Both gather
// shapes already exist in EEDI3/EEDI3H; these two take raw pointers so the same
// code feeds either a VSFrame plane or the internal v buffer, and a `second`
// flag so the two sub-passes of one stage (recorded into a single command
// buffer) write disjoint upload regions.
// ---------------------------------------------------------------------------

// Vertical sub-pass upload: the compacted kept source rows for interp parity
// `field`, the packed dilation bits of the mask rows field+2r, and (when the
// vcheck consumes one) the compacted sclip rows. Mirrors EEDI3's vertical
// gather minus the dst kept-row copy -- the fused filter builds the merged
// frame on the GPU instead.
static void aa_gather_vertical(
    const Eedi3Data & d, const PlaneConfig & cfg, uint8_t * upload,
    uint8_t * staging, const uint8_t * srcp, ptrdiff_t src_stride,
    const uint8_t * scpp, ptrdiff_t scp_stride,
    const uint8_t * maskp, ptrdiff_t mask_stride,
    const int field, const bool second) {

    const bool nt = (d.copy_mode & 1) != 0;
    const int pw = cfg.width;
    const size_t raw_row_bytes = static_cast<size_t>(pw) * pad_elem_bytes(d.bits);
    uint8_t * const rawp = upload + (second ? cfg.raw2_offset : cfg.raw_offset);
    const int off = 1 - field;

    for (int k = 0; !d.skip_raw && k < cfg.rows; ++k) {
        frame_copy_out(rawp + static_cast<size_t>(k) * raw_row_bytes,
                       srcp + src_stride * (off + 2 * k), raw_row_bytes, nt);
    }

    if (maskp) {
        // The packed bits are built with ordinary cached stores, which are
        // catastrophic into the uncached host-visible VRAM the ReBAR upload
        // uses: keep them in the cached staging mirror (binding 4).
        uint8_t * const bm = staging + (second ? cfg.bits2_offset : cfg.bits_offset);
        const size_t bm_row_bytes =
            static_cast<size_t>((pw + 31) / 32) * sizeof(uint32_t);
        std::vector<uint64_t> scratch(
            2 * static_cast<size_t>((pw + d.mdis + 63) / 64));
        for (int r = 0; r < cfg.rows; ++r) {
            const uint8_t * const mrow_p = maskp + mask_stride * (field + 2 * r);
            build_bmask_row(mrow_p,
                            d.mclip_native16
                                ? reinterpret_cast<const uint16_t *>(mrow_p) : nullptr,
                            d.mclip_native32
                                ? reinterpret_cast<const float *>(mrow_p) : nullptr,
                            reinterpret_cast<uint32_t *>(
                                bm + static_cast<int64_t>(r) * bm_row_bytes),
                            pw, d.mdis, scratch.data());
        }
    }

    if (scpp && d.vcheck > 0) {
        const size_t row_bytes = static_cast<size_t>(pw) * d.elem_bytes;
        uint8_t * const sc = upload + (second ? cfg.sclip2_offset : cfg.sclip_offset);
        for (int r = 0; !d.skip_sclip && r < cfg.rows; ++r) {
            frame_copy_out(sc + static_cast<size_t>(r) * row_bytes,
                           scpp + scp_stride * (field + 2 * r), row_bytes, nt);
        }
    }
}

// Horizontal sub-pass upload against the transposed geometry `acfg`: the kept
// columns of the packed source (`srcp`, either the internal v staging buffer or
// the sclip frame) into K, the interp-parity sclip columns, and the transposed
// mask bit matrix. Exactly EEDI3H's gather, with the source pointer supplied by
// the caller.
static void aa_gather_horizontal(
    const Eedi3Data & d, const PlaneConfig & acfg, uint8_t * upload,
    uint8_t * staging, const uint8_t * srcp, ptrdiff_t src_stride,
    const uint8_t * scpp, ptrdiff_t scp_stride,
    const uint8_t * maskp, ptrdiff_t mask_stride,
    const int field, const bool second) {

    const bool nt = (d.copy_mode & 1) != 0;
    const int rows = acfg.rows;    // kept-column count (kernel rows)
    const int H = acfg.src_h;      // source rows == kernel plane width

    if (!d.skip_raw) {
        gather_columns(srcp, src_stride, H,
                       upload + (second ? acfg.raw2_offset : acfg.raw_offset),
                       rows, 2, 1 - field, d.elem_bytes, nt);
    }

    if (maskp) {
        const int mbits = d.mclip_native16 ? 16 : (d.mclip_native32 ? 32 : 8);
        uint8_t * const ms = staging + d.upload_total + d.download_total + acfg.ms_offset;
        uint8_t * const bm = staging + (second ? acfg.bits2_offset : acfg.bits_offset);
        const int nwords = (acfg.width + 31) / 32;
        const int nw64 = (acfg.width + d.mdis + 63) / 64;
        std::vector<uint64_t> scratch(
            2 * static_cast<size_t>(nw64) + acfg.rows);
        if (d.mask_fuse && d.mdis < 64 && acfg.width >= 2 * d.mdis) {
            const int bw = (acfg.width + 63) / 64;   // words per bit row
            uint64_t * const bitmat = reinterpret_cast<uint64_t *>(ms);
            gather_mask_bitmat(maskp, mask_stride, H, rows, field, 2, mbits,
                               bitmat, bw, scratch.data() + 2 * nw64);
            for (int r = 0; r < rows; ++r) {
                build_bmask_row_from_bits(
                    bitmat + static_cast<size_t>(r) * bw,
                    reinterpret_cast<uint32_t *>(
                        bm + static_cast<size_t>(r) * nwords * 4),
                    acfg.width, d.mdis, scratch.data());
            }
        } else {
            uint8_t * const m2 = ms + static_cast<size_t>(rows) * acfg.width;
            gather_mask_u8(maskp, mask_stride, H, ms, rows, field, 2, mbits);
            transpose_plane(ms, rows, rows, acfg.width, m2, acfg.width, 1);
            for (int r = 0; r < rows; ++r) {
                build_bmask_row(m2 + static_cast<size_t>(r) * acfg.width,
                                nullptr, nullptr,
                                reinterpret_cast<uint32_t *>(
                                    bm + static_cast<size_t>(r) * nwords * 4),
                                acfg.width, d.mdis, scratch.data());
            }
        }
    }

    if (scpp && d.vcheck > 0 && !d.skip_sclip) {
        gather_columns(scpp, scp_stride, H,
                       upload + (second ? acfg.sclip2_offset : acfg.sclip_offset),
                       rows, 2, field, d.elem_bytes, nt);
    }
}

// Final 50/50 merge of the two composed horizontal planes into the output
// frame: std.Merge's default u16 arithmetic is exactly (a+b+1)>>1 (==
// _mm256_avg_epu16), f32 is 0.5f*a + 0.5f*b bitwise.
static void merge_pair_rows(uint8_t * dst, ptrdiff_t dst_stride,
                            const uint8_t * a, const uint8_t * b,
                            const size_t row_bytes, const int rows,
                            const int bits, const bool nt) {
    for (int y = 0; y < rows; ++y) {
        uint8_t * const dp = dst + static_cast<ptrdiff_t>(y) * dst_stride;
        const uint8_t * const ap = a + static_cast<size_t>(y) * row_bytes;
        const uint8_t * const bp = b + static_cast<size_t>(y) * row_bytes;
        if (bits == 16) {
            const int n = static_cast<int>(row_bytes / 2);
            const uint16_t * pa = reinterpret_cast<const uint16_t *>(ap);
            const uint16_t * pb = reinterpret_cast<const uint16_t *>(bp);
            uint16_t * pd = reinterpret_cast<uint16_t *>(dp);
            int i = 0;
            for (; i + 16 <= n; i += 16) {
                const __m256i va = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(pa + i));
                const __m256i vb = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(pb + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(pd + i),
                                    _mm256_avg_epu16(va, vb));
            }
            for (; i < n; ++i) {
                pd[i] = static_cast<uint16_t>((pa[i] + pb[i] + 1u) >> 1);
            }
        } else {
            const int n = static_cast<int>(row_bytes / 4);
            const float * pa = reinterpret_cast<const float *>(ap);
            const float * pb = reinterpret_cast<const float *>(bp);
            float * pd = reinterpret_cast<float *>(dp);
            const __m256 half = _mm256_set1_ps(0.5f);
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                const __m256 va = _mm256_loadu_ps(pa + i);
                const __m256 vb = _mm256_loadu_ps(pb + i);
                _mm256_storeu_ps(pd + i, _mm256_add_ps(
                    _mm256_mul_ps(va, half), _mm256_mul_ps(vb, half)));
            }
            for (; i < n; ++i) {
                pd[i] = 0.5f * pa[i] + 0.5f * pb[i];
            }
        }
    }
    (void)nt;
}

// ---------------------------------------------------------------------------
// EEDI3AA frame handler
//
// One output frame consumes input frame n (single rate!) and does the whole
// based_aa EEDI3 chain: vertical sub-frames n=2k (parity fv0) and n=2k+1 (fv1)
// are merged into the intermediate frame v, which is then column-gathered and
// run through the horizontal pass twice (fh0/fh1); the two composed planes are
// 50/50-merged into the output. Two submissions: the vertical stage and the
// horizontal stage, with the host column gather of v in between.
// ---------------------------------------------------------------------------

static const VSFrame *VS_CC Eedi3AaGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    Eedi3Data * d = static_cast<Eedi3Data *>(instanceData);
    const int sn = n;   // single-rate

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
        if (d->vcheck > 0 && d->sclip_node) {
            vsapi->requestFrameFilter(2 * sn, d->sclip_node, frameCtx);
            vsapi->requestFrameFilter(2 * sn + 1, d->sclip_node, frameCtx);
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

    // Unprocessed planes may share src's plane data (dims always match here).
    const int pl[] = { 0, 1, 2 };
    const VSFrame * fr[] = {
        !d->process[0] ? src : nullptr,
        !d->process[1] ? src : nullptr,
        !d->process[2] ? src : nullptr
    };
    VSFrame * dst = vsapi->newVideoFrame2(
        &d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

    auto resource = d->pool.take();
    Eedi3Direct direct;   // the fused path never imports output planes

    VkDevice dev = d->device->device;
    float * map = resource.map;

    const VSFrame * scp0 = nullptr, * scp1 = nullptr;
    if (d->vcheck > 0 && d->sclip_node) {
        scp0 = vsapi->getFrameFilter(2 * sn, d->sclip_node, frameCtx);
        scp1 = vsapi->getFrameFilter(2 * sn + 1, d->sclip_node, frameCtx);
    }
    const VSFrame * mcp = nullptr;
    if (d->mclip_node) {
        mcp = vsapi->getFrameFilter(sn, d->mclip_node, frameCtx);
    }

    auto set_error = [&](const std::string & error_message) {
        d->pool.give_back(std::move(resource));
        vsapi->setFilterError(("EEDI3AA: " + error_message).c_str(), frameCtx);
        vsapi->freeFrame(src);
        vsapi->freeFrame(scp0);
        vsapi->freeFrame(scp1);
        vsapi->freeFrame(mcp);
        return nullptr;
    };

    // Vertical sub-frame parities: the doubled stream's n=2k takes the input's
    // _FieldBased (or field&1), n=2k+1 its complement. The horizontal pass runs
    // on v, an EEDI3 output frame, which is always _FieldBased=PROGRESSIVE, so
    // it takes field&1 with no override -- the asymmetry §2.3 of the design
    // notes calls out as the easiest silent bug.
    int base = d->field & 1;
    int err;
    const int fieldBased = vsapi->mapGetIntSaturated(
        vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err);
    if (fieldBased == VSC_FIELD_BOTTOM) {
        base = 0;
    } else if (fieldBased == VSC_FIELD_TOP) {
        base = 1;
    }
    const int fv0 = base, fv1 = 1 - base;
    const int fh0 = d->field & 1, fh1 = 1 - fh0;

    const bool coherent =
        !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    uint8_t * const staging = static_cast<uint8_t *>(static_cast<void *>(map));
    uint8_t * const upload = resource.up_map ? resource.up_map : staging;

    const uint8_t * maskp = nullptr;
    ptrdiff_t mask_stride = 0;
    if (mcp) {
        maskp = vsapi->getReadPtr(mcp, 0);
        mask_stride = vsapi->getStride(mcp, 0);
    }

    const int numPlanes = d->vi->format.numPlanes;

    // Host-stage probe: sample one frame with VSFEEL_EEDI3AA_HFRAME=<n>, or
    // every 200th frame from sn=100 when unset (a single sample is
    // clock-noise-prone; <0 selects the periodic form). Durable tuning probe,
    // same shape as EEDI3's VSFEEL_EEDI3_HBENCH.
    static const int hframe = getenv("VSFEEL_EEDI3AA_HFRAME")
        ? atoi(getenv("VSFEEL_EEDI3AA_HFRAME")) : -1;
    const bool hbench = getenv("VSFEEL_EEDI3AA_HBENCH") &&
        (hframe >= 0 ? sn == hframe : (sn >= 100 && sn % 200 == 0));
    const auto h_t0 = std::chrono::steady_clock::now();
    auto h_tvGather = h_t0, h_tvRec = h_t0, h_tvWait = h_t0;
    auto h_thGather = h_t0, h_thRec = h_t0, h_thWait = h_t0;

    auto gather_vertical = [&](const int field, const bool second) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & cfg = d->planes[plane];
            const VSFrame * const scf = second ? scp1 : scp0;
            const uint8_t * scpp = nullptr;
            ptrdiff_t scp_stride = 0;
            if (scf) {
                scpp = vsapi->getReadPtr(scf, plane);
                scp_stride = vsapi->getStride(scf, plane);
            }
            aa_gather_vertical(*d, cfg, upload, staging,
                               vsapi->getReadPtr(src, plane),
                               vsapi->getStride(src, plane),
                               scpp, scp_stride, maskp, mask_stride,
                               field, second);
        }
    };

    auto gather_horizontal = [&](const int field, const bool second) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & vcfg = d->planes[plane];
            const auto & acfg = d->aplanes[plane];
            const VSFrame * const scf = second ? scp1 : scp0;
            const uint8_t * scpp = nullptr;
            ptrdiff_t scp_stride = 0;
            if (scf) {
                scpp = vsapi->getReadPtr(scf, plane);
                scp_stride = vsapi->getStride(scf, plane);
            }
            const uint8_t * const vp = staging + d->upload_total +
                d->download_total + vcfg.v_offset;
            const ptrdiff_t v_stride =
                static_cast<ptrdiff_t>(vcfg.out_w) * d->elem_bytes;
            aa_gather_horizontal(*d, acfg, upload, staging, vp, v_stride,
                                 scpp, scp_stride, maskp, mask_stride,
                                 field, second);
        }
    };

    // ------------------------------------------------------------------
    // Submission 1: the vertical stage (both sub-frames).
    // ------------------------------------------------------------------
    checkVK(vkResetCommandPool(dev, resource.pool, 0));
    {
        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pInheritanceInfo = nullptr
        };
        if (vkBeginCommandBuffer(resource.cmd, &begin_info) != VK_SUCCESS) {
            return set_error("vkBeginCommandBuffer failed");
        }
    }
    record_h2d_copy(*d, resource);
    gather_vertical(fv0, false);
    gather_vertical(fv1, true);
    if (hbench) { h_tvGather = std::chrono::steady_clock::now(); }
    if (const auto e = record_pass(*d, resource, fv0, d->planes, false, false,
                                   PassTail::kNone, direct)) {
        return set_error(*e);
    }
    if (const auto e = record_pass(*d, resource, fv1, d->planes, false, true,
                                   PassTail::kAssembleV, direct)) {
        return set_error(*e);
    }
    if (hbench) { h_tvRec = std::chrono::steady_clock::now(); }
    if (vkEndCommandBuffer(resource.cmd) != VK_SUCCESS) {
        return set_error("vkEndCommandBuffer failed");
    }

    if (!coherent) {
        std::vector<VkMappedMemoryRange> ranges;
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            ranges.push_back(VkMappedMemoryRange {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = resource.staging_mem,
                .offset = 0,
                .size = d->upload_total,
            });
        }
        checkVK(vkFlushMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()),
                                          ranges.data()));
    }
    _mm_sfence();
    checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
        resource.cmd, resource.fence));
    checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));
    if (hbench) { h_tvWait = std::chrono::steady_clock::now(); }


    // ------------------------------------------------------------------
    // Host: column-gather the merged v for both horizontal sub-passes.
    // ------------------------------------------------------------------
    gather_horizontal(fh0, false);
    gather_horizontal(fh1, true);
    if (hbench) { h_thGather = std::chrono::steady_clock::now(); }


    // ------------------------------------------------------------------
    // Submission 2: the horizontal stage (both sub-frames compose their plane).
    // ------------------------------------------------------------------
    checkVK(vkResetCommandPool(dev, resource.pool, 0));
    {
        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pInheritanceInfo = nullptr
        };
        if (vkBeginCommandBuffer(resource.cmd, &begin_info) != VK_SUCCESS) {
            return set_error("vkBeginCommandBuffer failed");
        }
    }
    record_h2d_copy(*d, resource);
    if (const auto e = record_pass(*d, resource, fh0, d->aplanes, true, false,
                                   PassTail::kCompose, direct)) {
        return set_error(*e);
    }
    if (const auto e = record_pass(*d, resource, fh1, d->aplanes, true, true,
                                   PassTail::kCompose, direct)) {
        return set_error(*e);
    }
    if (vkEndCommandBuffer(resource.cmd) != VK_SUCCESS) {
        return set_error("vkEndCommandBuffer failed");
    }
    if (hbench) { h_thRec = std::chrono::steady_clock::now(); }

    if (!coherent) {
        std::vector<VkMappedMemoryRange> ranges;
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            ranges.push_back(VkMappedMemoryRange {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = resource.staging_mem,
                .offset = 0,
                .size = d->upload_total,
            });
        }
        checkVK(vkFlushMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()),
                                          ranges.data()));
    }
    _mm_sfence();
    checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
        resource.cmd, resource.fence));
    checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));
    if (hbench) { h_thWait = std::chrono::steady_clock::now(); }

    // ------------------------------------------------------------------
    // Final 50/50 merge of the two composed planes into the output frame.
    // ------------------------------------------------------------------
    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        const auto & cfg = d->planes[plane];
        const size_t out_row = static_cast<size_t>(cfg.out_w) * d->elem_bytes;
        const uint8_t * const o0 = staging + d->upload_total +
            d->download_total + cfg.out_offset;
        const uint8_t * const o1 = staging + d->upload_total +
            d->download_total + cfg.out2_offset;
        merge_pair_rows(vsapi->getWritePtr(dst, plane),
                        vsapi->getStride(dst, plane), o0, o1, out_row,
                        cfg.out_h, d->bits, (d->copy_mode & 2) != 0);
    }

    if (hbench) {
        const auto h_t3 = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        fprintf(stderr, "[eedi3aa-hbench] sn=%d vGather=%.3fms vRec=%.3fms "
                        "vWait=%.3fms hGather=%.3fms hRec=%.3fms hWait=%.3fms "
                        "merge=%.3fms total=%.3fms\n",
                sn, ms(h_t0, h_tvGather), ms(h_tvGather, h_tvRec),
                ms(h_tvRec, h_tvWait), ms(h_tvWait, h_thGather),
                ms(h_thGather, h_thRec), ms(h_thRec, h_thWait),
                ms(h_thWait, h_t3), ms(h_t0, h_t3));
    }

    d->pool.give_back(std::move(resource));

    vsapi->freeFrame(src);
    vsapi->freeFrame(scp0);
    vsapi->freeFrame(scp1);
    vsapi->freeFrame(mcp);

    VSMap * props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);

    return dst;
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
    // dh doubles the interpolated axis: height vertically, width in EEDI3H.
    const int out_height = (d->dh && !d->horiz) ? d->vi->height * 2 : d->vi->height;
    const int out_width = (d->dh && d->horiz) ? d->vi->width * 2 : d->vi->width;
    VSFrame * dst = vsapi->newVideoFrame2(
        &d->vi->format, out_width, out_height, fr, pl, src, core);

    auto resource = d->pool.take();

    // per-frame direct-to-frame output imports; released on every exit path
    Eedi3Direct direct;

    auto set_error = [&](const std::string & error_message) {
        direct.destroy(*d);
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
    if (d->horiz && scp && getenv("VSFEEL_EEDI3_PTRTRACE") && (sn < 3)) {
        fprintf(stderr, "[eedi3-ptr] sn=%d src=%p scp=%p mcp=%p\n", sn,
                vsapi->getReadPtr(src, 0), vsapi->getReadPtr(scp, 0),
                mcp ? vsapi->getReadPtr(mcp, 0) : nullptr);
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

    // Host-path probe: sample one frame with VSFEEL_EEDI3_HFRAME=<n>, or every
    // 200th frame from sn=100 when the variable is unset (a single sample is
    // clock-noise-prone; <0 selects the periodic form).
    static const int hframe = getenv("VSFEEL_EEDI3_HFRAME") ? atoi(getenv("VSFEEL_EEDI3_HFRAME")) : -1;
    const bool hbench = getenv("VSFEEL_EEDI3_HBENCH") &&
        (hframe >= 0 ? sn == hframe : (sn >= 100 && sn % 200 == 0));
    const auto h_t0 = std::chrono::steady_clock::now();
    auto h_tMaskEnd = h_t0, h_tGatherEnd = h_t0, h_tRawEnd = h_t0;
    auto h_tMaskMid = h_t0;
    auto h_tRecEnd = h_t0, h_tImportEnd = h_t0;

    // Re-record the command buffer with this frame's interp-row parity (the
    // previous submit on this resource was waited on before give_back, so the
    // pool reset is safe).
    checkVK(vkResetCommandPool(dev, resource.pool, 0));

    // Direct-to-frame output: import each processed plane's memory before
    // recording, because the blit dispatch needs the buffer handle. All planes
    // must be importable: vout_base is a single push constant per dispatch, but
    // the fallback (staging + CPU blit) needs vout in staging, so a mixed frame
    // is not expressible. Any failure falls back to the CPU blit for the whole
    // frame.
    if (d->dst_host && d->device->host_import && !d->skip_blit) {
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & cfg = d->planes[plane];
            void * const dstp = vsapi->getWritePtr(dst, plane);
            const ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
            const VkDeviceSize plane_bytes =
                static_cast<VkDeviceSize>(dst_stride) * cfg.height;
            if (auto err = import_plane_host_memory(*d, dstp, plane_bytes,
                                                    direct.plane[plane])) {
                direct.destroy(*d);
                if (trace_on("VSFEEL_EEDI3_TRACE")) {
                    fprintf(stderr, "[eedi3] direct-to-frame import failed: %s\n",
                            err->c_str());
                }
                break;
            }
            direct.stride[plane] = static_cast<int>(dst_stride / d->elem_bytes);
            direct.vout_offset[plane] = (d->vcheck > 0) ? cfg.vout_offset
                                                        : cfg.dst_offset;
            direct.active[plane] = true;
            // point the blit set's b1 (the output frame) at this plane's import
            VkDescriptorBufferInfo blit_info {
                .buffer = direct.plane[plane].buffer,
                .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkWriteDescriptorSet blit_write {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = resource.desc_set_blit[plane],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &blit_info,
                .pTexelBufferView = nullptr
            };
            vkUpdateDescriptorSets(dev, 1, &blit_write, 0, nullptr);
        }
    }
    if (hbench) { h_tImportEnd = std::chrono::steady_clock::now(); }

    if (const auto err = record_command_buffer(*d, resource, field, direct)) {
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
        auto dstp = vsapi->getWritePtr(dst, plane);
        const ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
        const bool nt_raw = (d->copy_mode & 1) != 0 && !d->raw_stage;
        const bool nt_kept = (d->copy_mode & 2) != 0;

        if (d->horiz) {
            // EEDI3H: the upload is the transposed plane's kept rows, i.e. the
            // source COLUMNS of the kept parity read out of the source ROWS.
            // dh keeps every source column (the transposed height doubles, so
            // every input column survives as a kept row).
            //
            // When the sclip frame IS this frame (based_aa's Interleave([clip,
            // clip]), which vsapi hands out as the same plane pointer) the
            // interp-parity columns are the other half of the very same row
            // chunks, so both compactions come out of one pass: the second
            // full-frame read is pure waste. VSFEEL_EEDI3_PAIR=0 restores the
            // two-pass form for A/B.
            const bool need_sclip = d->vcheck > 0 && d->sclip_node && scp &&
                !d->skip_sclip;
            bool pair_done = false;
            if (need_sclip && !d->skip_raw && !d->skip_pair && !d->dh &&
                vsapi->getReadPtr(scp, plane) == srcp &&
                nt_raw == ((d->copy_mode & 1) != 0)) {
                pair_done = gather_columns_pair(
                    srcp, src_stride, cfg.src_h, rawp, upload + cfg.sclip_offset,
                    cfg.rows, off, d->elem_bytes, nt_raw);
            }
            if (!pair_done && !d->skip_raw) {
                gather_columns(srcp, src_stride, cfg.src_h, rawp, cfg.rows,
                               d->dh ? 1 : 2, d->dh ? 0 : off,
                               d->elem_bytes, nt_raw);
            }
            if (hbench) { h_tRawEnd = std::chrono::steady_clock::now(); }

            if (d->mclip_node && mcp) {
                // The row kernel needs, for transposed interp row r, the mask
                // dilated along the transposed row (the source COLUMN r): build
                // the mask's transposed byte image and run the existing
                // build_bmask_row on its rows -- identical math to the vertical
                // path, just applied to the transposed plane.
                const uint8_t * maskp = vsapi->getReadPtr(mcp, 0);
                const ptrdiff_t mask_stride = vsapi->getStride(mcp, 0);
                uint8_t * const ms = staging + d->upload_total + d->download_total + cfg.ms_offset;
                uint8_t * const m2 = ms + static_cast<size_t>(cfg.rows) * cfg.width;
                // The mask's own representation (the reference's native forms
                // and the Gray8 fallback), not the clip's depth.
                const int mbits = d->mclip_native16 ? 16
                    : (d->mclip_native32 ? 32 : 8);
                if (!d->skip_maskx) {
                    gather_mask_u8(maskp, mask_stride, cfg.src_h, ms, cfg.rows,
                                   d->dh ? 0 : field, d->dh ? 1 : 2, mbits);
                    transpose_plane(ms, cfg.rows, cfg.rows, cfg.width, m2,
                                    cfg.width, 1);
                }
                // The packed bits are built by ordinary (cached) stores, which
                // are catastrophic into the uncached host-visible VRAM the
                // ReBAR upload uses: keep them in the cached staging mirror and
                // let the row kernel read them from there (binding 4).
                uint8_t * bm = staging + cfg.bits_offset;
                const int nwords = (cfg.width + 31) / 32;
                const int nw64 = (cfg.width + d->mdis + 63) / 64;
                // [0, 2*nw64) = the dilation scratch, then the fused gather's
                // per-y-block accumulator (one u64 per transposed row).
                std::vector<uint64_t> bmask_scratch(
                    2 * static_cast<size_t>(nw64) + cfg.rows);
                if (d->mask_fuse && !d->skip_maskx && d->mdis < 64 &&
                    cfg.width >= 2 * d->mdis) {
                    // Fused: threshold+transpose+pack straight into the
                    // transposed bit matrix, then dilate each transposed row.
                    const int bw = (cfg.width + 63) / 64;   // words per bit row
                    uint64_t * const bitmat = reinterpret_cast<uint64_t *>(ms);
                    gather_mask_bitmat(maskp, mask_stride, cfg.src_h, cfg.rows,
                                       d->dh ? 0 : field, d->dh ? 1 : 2, mbits,
                                       bitmat, bw, bmask_scratch.data() + 2 * nw64);
                    if (hbench) { h_tMaskMid = std::chrono::steady_clock::now(); }
                    for (int r = 0; r < cfg.rows; ++r) {
                        build_bmask_row_from_bits(
                            bitmat + static_cast<size_t>(r) * bw,
                            reinterpret_cast<uint32_t *>(
                                bm + static_cast<size_t>(r) * nwords * 4),
                            cfg.width, d->mdis, bmask_scratch.data());
                    }
                } else {
                    if (!d->skip_maskx) {
                        gather_mask_u8(maskp, mask_stride, cfg.src_h, ms, cfg.rows,
                                       d->dh ? 0 : field, d->dh ? 1 : 2, mbits);
                        transpose_plane(ms, cfg.rows, cfg.rows, cfg.width, m2,
                                        cfg.width, 1);
                    }
                    if (hbench) { h_tMaskMid = std::chrono::steady_clock::now(); }
                    for (int r = 0; r < cfg.rows; ++r) {
                        build_bmask_row(m2 + static_cast<size_t>(r) * cfg.width,
                                        nullptr, nullptr,
                                        reinterpret_cast<uint32_t *>(
                                            bm + static_cast<size_t>(r) * nwords * 4),
                                        cfg.width, d->mdis, bmask_scratch.data());
                    }
                }
            }
            if (hbench) { h_tMaskEnd = std::chrono::steady_clock::now(); }

            if (d->vcheck > 0 && d->sclip_node && scp && !d->skip_sclip &&
                !pair_done) {
                // The vcheck reads the transposed sclip's interp rows, i.e. the
                // sclip's interp columns (field + 2r) transposed.
                const auto scpp = vsapi->getReadPtr(scp, plane);
                const ptrdiff_t scp_stride = vsapi->getStride(scp, plane);
                gather_columns(scpp, scp_stride, cfg.src_h,
                               upload + cfg.sclip_offset, cfg.rows,
                               2, field, d->elem_bytes, (d->copy_mode & 1) != 0);
            }
            continue;
        }

        // The tight kept-row gather and the destination's kept-row copy read
        // exactly the same source rows in the same order (raw row k -> kept
        // row k -> dst row `off + 2k`), so both are written here while the
        // source row is hot. Doing the dst half later in the blit re-read the
        // whole source a second time (8.3 MB/frame at the bench geometry) —
        // measured worth ~5-7% when the host copies are the limiter.
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
            const float * maskf = d->mclip_native32
                ? reinterpret_cast<const float *>(maskp) : nullptr;
            // Same as the EEDI3H gather: the packed bits go to the cached
            // staging mirror, never straight into the uncached ReBAR VRAM.
            uint8_t * bm = staging + cfg.bits_offset;
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
                                maskf ? reinterpret_cast<const float *>(mrow_p)
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

    const auto h_tSub0 = std::chrono::steady_clock::now();
    checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
        resource.cmd, resource.fence));
    const auto h_tSub1 = std::chrono::steady_clock::now();

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
                .offset = d->upload_total + (d->horiz
                    ? d->download_total + cfg.out_offset : cfg.dl_offset),
                .size = d->horiz ? cfg.out_bytes : cfg.dl_bytes,
            });
        }
        checkVK(vkInvalidateMappedMemoryRanges(dev, static_cast<uint32_t>(ranges.size()), ranges.data()));
    }

    // Copy results into dst: interp rows from the download region, kept rows
    // straight from src (like eedi3vk2). Planes whose memory was imported
    // (direct-to-frame) were already written by the blit kernel.
    for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
        if (!d->process[plane] || direct.active[plane]) {
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

        if (d->horiz) {
            // ENTRY_COMPOSE wrote the whole frame-order plane, so the CPU blit
            // is a plain strided row copy (the kept columns came back through
            // the GPU with the interpolated ones).
            if (!d->skip_blit) {
                const size_t out_row = static_cast<size_t>(cfg.out_w) * d->elem_bytes;
                const uint8_t * const src_out =
                    static_cast<const uint8_t *>(static_cast<const void *>(map)) +
                    d->upload_total + d->download_total + cfg.out_offset;
                for (int y = 0; y < cfg.out_h; ++y) {
                    frame_copy_out(dstp + dst_stride * y,
                                   src_out + static_cast<size_t>(y) * out_row,
                                   out_row, (d->copy_mode & 2) != 0);
                }
            }
            continue;
        }

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

    // The GPU is done with the imported planes (the fence above), so the
    // pinned pages can be released before the frame is handed to the consumer.
    direct.destroy(*d);

    d->pool.give_back(std::move(resource));

    if (hbench) {
        const auto h_t3 = std::chrono::steady_clock::now();
        const auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        fprintf(stderr, "[eedi3-hbench] cpu_stage=%.3fms submit=%.3fms fence_wait=%.3fms blit=%.3fms\n",
                us(h_t0, h_t1), us(h_tSub0, h_tSub1), us(h_t1, h_t2), us(h_t2, h_t3));
        fprintf(stderr, "[eedi3-hbench]   of cpu_stage: import=%.3fms record=%.3fms raw=%.3fms bits=%.3fms(maskx=%.3f bmask=%.3f) sclip=%.3fms\n",
                us(h_t0, h_tImportEnd), us(h_tImportEnd, h_tRecEnd), us(h_tRecEnd, h_tRawEnd),
                us(h_tRawEnd, h_tMaskEnd), us(h_tRawEnd, h_tMaskMid),
                us(h_tMaskMid, h_tMaskEnd), us(h_tMaskEnd, h_tGatherEnd));
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

static void vsfeel_eedi3_create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi, bool horiz, bool aa) {

    auto d { std::make_unique<Eedi3Data>() };
    int err = 0;
    d->horiz = horiz;
    d->aa = aa;

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
    if (d->aa) {
        // EEDI3AA is the fused based_aa chain: double-rate vertical then
        // horizontal, both merged 50/50. Only that shape is expressible.
        if (d->field < 2) {
            return set_error("field must be 2 or 3 for EEDI3AA");
        }
        if (d->dh) {
            return set_error("dh is not supported by EEDI3AA");
        }
    }
    if (!d->dh) {
        // The interpolated axis must divide into interp/kept lines: the height
        // vertically, the width in EEDI3H (which interpolates columns).
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

        // Gray16 integer and Gray32 float masks are handled natively: the exact
        // boolean the Gray8 conversion would produce is (v >= 129) for Gray16
        // (verified exhaustively) and (v * 255 >= 0.5) for a full-range float
        // mask, so the extra full-frame graph node is pure overhead. This
        // matters for based_aa, which passes mclip in the CLIP's format: a
        // float clip gets a float mask, and eedi3vk2 consumes it directly.
        // Other depths (10/12/14-bit) keep the reference conversion.
        if (mvi->format.bitsPerSample == 16 && mvi->format.sampleType == stInteger) {
            d->mclip_native16 = true;
        } else if (mvi->format.bitsPerSample == 32 && mvi->format.sampleType == stFloat) {
            d->mclip_native32 = true;
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
    } else if (d->horiz) {
        // EEDI3H default: the vcheck output is read back by the compose pass,
        // and reading it over PCIe from the staging download costs ~10% of the
        // frame (measured 328 -> 360 fps at ns=8). Device-local vout is free
        // here because the assembled plane has its own staging binding.
        d->vout_dev = true;
    }
    if (const char * dh = std::getenv("VSFEEL_EEDI3_DSTHOST")) {
        d->dst_host = (atoi(dh) != 0);
    }
    if (d->horiz) {
        // The direct-to-frame import path is vertical-only: EEDI3H's output is
        // assembled by ENTRY_COMPOSE into the staging download and blitted by
        // the CPU, and importing the frame anyway would only add the (measured
        // very expensive) per-frame host-pointer import for nothing.
        d->dst_host = false;
    }
    if (d->aa) {
        // EEDI3AA consumes vout with a GPU kernel (the vertical merge, then the
        // horizontal compose), so it must be device-local; the final CPU blit
        // merges the two composed planes into the frame, so no host import.
        d->vout_dev = true;
        d->dst_host = false;
    }
    d->raw_stage = std::getenv("VSFEEL_EEDI3_RAWSTAGE") != nullptr;
    d->blit_contig = std::getenv("VSFEEL_EEDI3_BLITCONTIG") != nullptr;
    d->skip_pad = std::getenv("VSFEEL_EEDI3_NOPAD") != nullptr;
    if (const char * pp = std::getenv("VSFEEL_EEDI3_PADPAR")) {
        d->pad_skip_parity = atoi(pp) != 0;
    }
    d->skip_blit = std::getenv("VSFEEL_EEDI3_NOBLIT") != nullptr;
    d->skip_sclip = std::getenv("VSFEEL_EEDI3_NOSCLIP") != nullptr;
    d->skip_raw = std::getenv("VSFEEL_EEDI3_NORAW") != nullptr;
    d->skip_h2d = std::getenv("VSFEEL_EEDI3_NOH2D") != nullptr;
    d->skip_vcheck = std::getenv("VSFEEL_EEDI3_NOVC") != nullptr;
    d->skip_xfer = std::getenv("VSFEEL_EEDI3_NOXFER") != nullptr;
    d->skip_xpose = std::getenv("VSFEEL_EEDI3_NOXPOSE") != nullptr;
    d->skip_compose = std::getenv("VSFEEL_EEDI3_NOCOMPOSE") != nullptr;
    d->skip_maskx = std::getenv("VSFEEL_EEDI3_NOMASKX") != nullptr;
    if (const char * mf = std::getenv("VSFEEL_EEDI3_MASKFUSE")) {
        d->mask_fuse = atoi(mf) != 0;
    }
    if (const char * pr = std::getenv("VSFEEL_EEDI3_PAIR")) {
        d->skip_pair = atoi(pr) == 0;
    }

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

    // The row/vcheck shaders compile to `OpMemoryModel Logical Vulkan`
    // (GL_KHR_memory_scope_semantics) and use `local_size_x_id` (LocalSizeId,
    // which needs maintenance4); all EEDI3 shaders also use int8/int16 SSBO
    // storage. Report a precise creation error instead of relying on the
    // driver accepting a pipeline whose features were never enabled.
    if (!d->device->feat_vulkan_memory_model) {
        return set_error("vulkanMemoryModel is not enabled on this device");
    }
    if (!d->device->feat_maintenance4) {
        return set_error("maintenance4 (LocalSizeId) is not enabled on this device");
    }
    if (!d->device->feat_8bit_storage) {
        return set_error("shaderInt8/storageBuffer8BitAccess is not enabled on this device");
    }
    if (!d->device->feat_16bit_storage) {
        return set_error("storageBuffer16BitAccess is not enabled on this device");
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
        // Per stream: the shared row/vcheck/vcopy set, a second set when the
        // ReBAR upload is active (the pad kernel needs its own set because its
        // binding 0 is the raw upload while the row kernel's binding 0 is the
        // built pad), and one blit set per plane (its b1 is that plane's
        // imported output frame, which differs per dispatch in a frame).
        const uint32_t sets_per_stream =
            ((d->rebar_up) ? 2u : 1u) +
            (d->aa ? 2u : 0u) +
            ((d->dst_host && d->device->host_import) ? MAX_PLANES : 0u);
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
        const uint32_t * vclds_code = nullptr;
        size_t vclds_size = 0;
        const uint32_t * pad_code = nullptr;
        size_t pad_size = 0;
        switch (d->bits) {
            case 16:
                row_code = eedi3_16_row_spv; row_size = eedi3_16_row_spv_size;
                vc_code = eedi3_16_vcheck_spv; vc_size = eedi3_16_vcheck_spv_size;
                vclds_code = eedi3_16_vcheck_lds_spv; vclds_size = eedi3_16_vcheck_lds_spv_size;
                pad_code = eedi3_16_pad_spv; pad_size = eedi3_16_pad_spv_size;
                break;
            case 32:
                row_code = eedi3_32_row_spv; row_size = eedi3_32_row_spv_size;
                vc_code = eedi3_32_vcheck_spv; vc_size = eedi3_32_vcheck_spv_size;
                vclds_code = eedi3_32_vcheck_lds_spv; vclds_size = eedi3_32_vcheck_lds_spv_size;
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
            if (d->device->limits.maxComputeSharedMemorySize >=
                    2 * MAXW_LDS * sizeof(float)) {
                auto r2b = create_shader_module(*d->device, vclds_code, vclds_size);
                if (std::holds_alternative<std::string>(r2b)) {
                    return set_error(std::get<std::string>(r2b));
                }
                d->vcheck_lds_module = std::get<VkShaderModule>(r2b);
            }
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
        if (d->dst_host && d->device->host_import) {
            const uint32_t * blit_code = nullptr;
            size_t blit_size = 0;
            switch (d->bits) {
                case 16:
                    blit_code = eedi3_16_blit_spv; blit_size = eedi3_16_blit_spv_size;
                    break;
                case 32:
                    blit_code = eedi3_32_blit_spv; blit_size = eedi3_32_blit_spv_size;
                    break;
                default:
                    return set_error("unsupported bit depth");
            }
            auto r5 = create_shader_module(*d->device, blit_code, blit_size);
            if (std::holds_alternative<std::string>(r5)) {
                return set_error(std::get<std::string>(r5));
            }
            d->blit_module = std::get<VkShaderModule>(r5);
        }
        if (d->aa) {
            const uint32_t * av_code = nullptr;
            size_t av_size = 0;
            switch (d->bits) {
                case 16:
                    av_code = eedi3_16_assemblev_spv; av_size = eedi3_16_assemblev_spv_size;
                    break;
                case 32:
                    av_code = eedi3_32_assemblev_spv; av_size = eedi3_32_assemblev_spv_size;
                    break;
                default:
                    return set_error("unsupported bit depth");
            }
            auto ra = create_shader_module(*d->device, av_code, av_size);
            if (std::holds_alternative<std::string>(ra)) {
                return set_error(std::get<std::string>(ra));
            }
            d->assemble_module = std::get<VkShaderModule>(ra);
        }
        if (d->horiz || d->aa) {
            const uint32_t * xp_code = nullptr;
            size_t xp_size = 0;
            const uint32_t * cp_code = nullptr;
            size_t cp_size = 0;
            switch (d->bits) {
                case 16:
                    xp_code = eedi3_16_xpose_spv; xp_size = eedi3_16_xpose_spv_size;
                    cp_code = eedi3_16_compose_spv; cp_size = eedi3_16_compose_spv_size;
                    break;
                case 32:
                    xp_code = eedi3_32_xpose_spv; xp_size = eedi3_32_xpose_spv_size;
                    cp_code = eedi3_32_compose_spv; cp_size = eedi3_32_compose_spv_size;
                    break;
                default:
                    return set_error("unsupported bit depth");
            }
            auto rx = create_shader_module(*d->device, xp_code, xp_size);
            if (std::holds_alternative<std::string>(rx)) {
                return set_error(std::get<std::string>(rx));
            }
            d->xpose_module = std::get<VkShaderModule>(rx);
            auto rc = create_shader_module(*d->device, cp_code, cp_size);
            if (std::holds_alternative<std::string>(rc)) {
                return set_error(std::get<std::string>(rc));
            }
            d->compose_module = std::get<VkShaderModule>(rc);
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
    // EEDI3AA is single-rate: it consumes the two doubled sub-frames of every
    // input frame internally and emits one frame per input frame, so the
    // historical *2 (rate doubling) does not apply.
    if (d->field > 1 && !d->aa) {
        if (d->vi->numFrames > INT32_MAX / 2) {
            return set_error("resulting clip is too long");
        }
        out_video.numFrames *= 2;
        vsh::muldivRational(&out_video.fpsNum, &out_video.fpsDen, 2, 1);
    }
    // dh doubles the interpolated axis: the height vertically, the width in
    // EEDI3H (EEDI3H = T(EEDI3(T(x))), so EEDI3's height doubling becomes a
    // width doubling here).
    if (d->dh) {
        if (d->horiz) {
            out_video.width *= 2;
        } else {
            out_video.height *= 2;
        }
    }

    auto align32 = [](VkDeviceSize v) { return (v + 31) & ~VkDeviceSize(31); };

    const int tpitch = 2 * d->mdis + 1;

    VkDeviceSize upload_total = 0;
    VkDeviceSize download_total = 0;
    VkDeviceSize dev_total = 0;
    VkDeviceSize scratch_total = 0;

    auto & planes = d->planes;

    // EEDI3AA: the horizontal pass runs on the merged vertical frame v, whose
    // dims equal the input plane's, so its kernel geometry is the input
    // transposed. Computing it up front lets the layout below take the max of
    // the two shapes where they differ (the padded plane, the packed mask bits,
    // the per-row empty flags) and share offsets where the two are symmetric.
    if (d->aa) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const int in_w = (plane == 0) ? d->vi->width : d->vi->width >> subW;
            const int in_h = (plane == 0) ? d->vi->height : d->vi->height >> subH;
            auto & a = d->aplanes[plane];
            a.src_w = in_w;
            a.src_h = in_h;
            a.width = in_h;                 // kernel plane width = frame height
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
        auto & cfg = planes[plane];
        // Input plane dims, then the kernel's view of them: vertically the
        // kernel plane is the input plane with dh doubling the height;
        // horizontally it is the transpose (width <-> height, dh doubling the
        // transposed height = the output width).
        const int in_w = (plane == 0) ? d->vi->width : d->vi->width >> subW;
        const int in_h = (plane == 0) ? d->vi->height : d->vi->height >> subH;
        const int kw = d->horiz ? in_h : in_w;
        const int kh = d->horiz ? (d->dh ? 2 * in_w : in_w)
                                : (d->dh ? 2 * in_h : in_h);

        cfg.src_w = in_w;
        cfg.src_h = in_h;
        cfg.width = kw;
        cfg.height = kh;
        cfg.rows = kh / 2;
        cfg.out_w = d->horiz ? kh : kw;
        cfg.out_h = d->horiz ? kw : kh;
        cfg.tpitch = tpitch;
        cfg.pad_stride = (kw + MARGIN_H * 2 + 15) & ~15;   // pad elements
        cfg.pad_height = kh + MARGIN_V * 2;
        const int pw = kw;

        // staging/upload regions: tight kept source rows (the pad kernel
        // expands mirrors), gathered sclip rows, and CPU-packed dilation bits
        cfg.raw_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * pad_elem;
        cfg.raw_offset = align32(upload_total);
        upload_total = align32(cfg.raw_offset + cfg.raw_bytes);

        if (d->aa) {
            // EEDI3AA: the two vertical sub-passes keep complementary row
            // parities, so both compacted worlds coexist until the assemble
            // kernel merges them. The horizontal pass then reuses the pair as
            // its per-sub-pass kept-column uploads (same byte count).
            cfg.raw2_bytes = cfg.raw_bytes;
            cfg.raw2_offset = align32(upload_total);
            upload_total = align32(cfg.raw2_offset + cfg.raw2_bytes);
        }

        if (d->vcheck > 0 && d->sclip_node) {
            cfg.sclip_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * elem_bytes;
            cfg.sclip_offset = align32(upload_total);
            upload_total = align32(cfg.sclip_offset + cfg.sclip_bytes);
            if (d->aa) {
                // The two sub-passes of a stage are recorded into one command
                // buffer, so both sclip compactions (complementary row parities)
                // must be in the upload region before the submit.
                cfg.sclip2_bytes = cfg.sclip_bytes;
                cfg.sclip2_offset = align32(upload_total);
                upload_total = align32(cfg.sclip2_offset + cfg.sclip2_bytes);
            }
        }

        if (d->mclip_node) {
            cfg.bits_bytes = static_cast<VkDeviceSize>((pw + 31) / 32) * cfg.rows * 4;
            if (d->aa) {
                // The horizontal pass's mask is the transposed one: `rows_h`
                // packed rows of `width_h` bits. One region serves both
                // geometries (the vertical and horizontal stages are separate
                // submissions, so their bits never coexist); size it for the
                // larger.
                const auto & a = d->aplanes[plane];
                cfg.bits_bytes = std::max(cfg.bits_bytes,
                    static_cast<VkDeviceSize>((a.width + 31) / 32) * a.rows * 4);
            }
            cfg.bits_offset = align32(upload_total);
            upload_total = align32(cfg.bits_offset + cfg.bits_bytes);
            if (d->aa) {
                cfg.bits2_bytes = cfg.bits_bytes;
                cfg.bits2_offset = align32(upload_total);
                upload_total = align32(cfg.bits2_offset + cfg.bits2_bytes);
            }
        }

        // download region (host staging): interp rows only (tight)
        cfg.dl_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * elem_bytes;
        cfg.dl_offset = align32(download_total);
        download_total = align32(cfg.dl_offset + cfg.dl_bytes);

        // device-local regions
        cfg.dst_bytes = cfg.dl_bytes;
        cfg.dst_offset = align32(dev_total);
        dev_total = align32(cfg.dst_offset + cfg.dst_bytes);
        if (d->aa) {
            // Second sub-pass's row-kernel output. Kept separate rather than
            // reusing dst across the two sub-passes because the assemble kernel
            // needs both fields' interp values simultaneously when vcheck == 0
            // (with a vcheck they land in vout/vout2 and dst is only a
            // scratch between the row kernel and the vcheck).
            cfg.dst2_bytes = cfg.dst_bytes;
            cfg.dst2_offset = align32(dev_total);
            dev_total = align32(cfg.dst2_offset + cfg.dst2_bytes);
        }

        // per-interp-row empty flags for the vcheck split (1 byte/row;
        // written by the row kernel, read by vcopy + the walk)
        cfg.rempty_bytes = static_cast<VkDeviceSize>(cfg.rows);
        if (d->aa) {
            cfg.rempty_bytes = std::max(cfg.rempty_bytes,
                static_cast<VkDeviceSize>(d->aplanes[plane].rows));
        }
        cfg.rempty_offset = align32(dev_total);
        dev_total = align32(cfg.rempty_offset + cfg.rempty_bytes);

        cfg.pbt_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows * tpitch;  // int8
        cfg.pbt_offset = align32(dev_total);
        dev_total = align32(cfg.pbt_offset + cfg.pbt_bytes);

        if (d->vcheck > 0 || d->aa) {
            if (d->vcheck > 0) {
                cfg.dmap_bytes = static_cast<VkDeviceSize>(pw) * cfg.rows;      // int8
                cfg.dmap_offset = align32(dev_total);
                dev_total = align32(cfg.dmap_offset + cfg.dmap_bytes);

                if (!(d->vcheck > 0 && d->sclip_node)) {
                    cfg.cint_bytes = cfg.dl_bytes;  // io
                    cfg.cint_offset = align32(dev_total);
                    dev_total = align32(cfg.cint_offset + cfg.cint_bytes);
                }
            }

            // vout region (dev_buf): RETIRED — vcheck/vcopy write vout
            // directly into the staging download region (b7 views staging),
            // so no D2H copy is needed. Kept allocated to avoid layout churn.
            // EEDI3AA forces vout device-local: the assemble/compose kernels
            // read it on the GPU, and with vcheck == 0 the row kernel writes
            // into this very region (dst is aliased onto it below), which is
            // how the assembler sees both fields' interp values.
            cfg.vout_bytes = cfg.dl_bytes;
            cfg.vout_offset = align32(dev_total);
            dev_total = align32(cfg.vout_offset + cfg.vout_bytes);
            if (d->aa) {
                cfg.vout2_bytes = cfg.vout_bytes;
                cfg.vout2_offset = align32(dev_total);
                dev_total = align32(cfg.vout2_offset + cfg.vout2_bytes);
                if (d->vcheck == 0) {
                    // No vcheck: the row kernel's output is the interp value,
                    // so make dst/dst2 BE the two vout regions (bit-exact: the
                    // assembler reads b7 at the same element offsets).
                    cfg.dst_offset = cfg.vout_offset;
                    cfg.dst2_offset = cfg.vout2_offset;
                }
            }
        }

        if (d->horiz || d->aa) {
            // Staging tail: the assembled output plane (written by ENTRY_
            // COMPOSE, read back by the CPU) and the CPU-only mask scratch
            // (the deinterleaved mask rows plus their transpose; the GPU never
            // sees either). Both sit after the download region so the
            // non-coherent invalidate ranges stay well defined.
            cfg.out_bytes = static_cast<VkDeviceSize>(cfg.out_w) * cfg.out_h * elem_bytes;
            cfg.out_offset = align32(scratch_total);
            scratch_total = align32(cfg.out_offset + cfg.out_bytes);

            cfg.ms_bytes = 2 * static_cast<VkDeviceSize>(cfg.rows) * cfg.width;
            // The fused path uses this scratch for the PACKED transposed bit
            // matrix (rows * (width+63)/64 u64 words), which is ~8x smaller
            // than the byte matrix -- max() keeps both paths in bounds. The
            // horizontal geometry's byte matrix is the same size (rows*width is
            // symmetric), so one region serves both passes.
            cfg.ms_bytes = std::max(cfg.ms_bytes,
                static_cast<VkDeviceSize>(cfg.rows) * ((cfg.width + 63) / 64) * 8);
            cfg.ms_offset = align32(scratch_total);
            scratch_total = align32(cfg.ms_offset + cfg.ms_bytes);

            if (d->aa) {
                // The two horizontal sub-passes compose their full planes here;
                // the host then 50/50-merges them into the output frame.
                cfg.out2_bytes = cfg.out_bytes;
                cfg.out2_offset = align32(scratch_total);
                scratch_total = align32(cfg.out2_offset + cfg.out2_bytes);
                // The merged vertical frame v, which the CPU column-gathers for
                // the horizontal pass. Full frame, frame order.
                cfg.v_bytes = static_cast<VkDeviceSize>(cfg.out_w) * cfg.out_h * elem_bytes;
                cfg.v_offset = align32(scratch_total);
                scratch_total = align32(cfg.v_offset + cfg.v_bytes);
            }
        }
    }

    d->upload_total = upload_total;
    d->download_total = download_total;
    d->dev_total = dev_total;
    d->scratch_total = scratch_total;


    // pad_dev (device-only, never staged): per-plane built padded planes
    // produced by the pad kernel from the mirrored upload. The mirror
    // occupies [0, upload_total), so the tail starts after it. EEDI3H adds the
    // transposed upload R' and transposed sclip B' there as well: both are
    // GPU-produced and only ever read by kernels, so they must be
    // device-local, and both are read by ENTRY_PAD / the vcheck through
    // binding 9 (and 5), which point at pad_dev.
    VkDeviceSize tail_total = upload_total;
    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        cfg.built_bytes = static_cast<VkDeviceSize>(cfg.pad_stride) * cfg.pad_height * pad_elem;
        if (d->aa) {
            // The vertical and horizontal passes run sequentially, so they
            // share one built-pad region; size it for the larger shape.
            const auto & a = d->aplanes[plane];
            cfg.built_bytes = std::max(cfg.built_bytes,
                static_cast<VkDeviceSize>(a.pad_stride) * a.pad_height * pad_elem);
        }
        cfg.built_offset = align32(tail_total);
        tail_total = align32(cfg.built_offset + cfg.built_bytes);
        if (d->horiz || d->aa) {
            cfg.rt_bytes = static_cast<VkDeviceSize>(cfg.rows) * cfg.width * pad_elem;
            cfg.rt_offset = align32(tail_total);
            tail_total = align32(cfg.rt_offset + cfg.rt_bytes);
            if (d->vcheck > 0 && d->sclip_node) {
                cfg.rtS_bytes = static_cast<VkDeviceSize>(cfg.rows) * cfg.width * elem_bytes;
                cfg.rtS_offset = align32(tail_total);
                tail_total = align32(cfg.rtS_offset + cfg.rtS_bytes);
            }
        }
    }

    // EEDI3AA: point the horizontal geometry at the shared buffer regions.
    // Every region the two passes share is byte-identical in the two shapes
    // (the horizontal kernel plane is the vertical one transposed), so the
    // offsets are copied rather than re-derived; the horizontal-only regions
    // (packed mask bits, R'/B', output planes) have their own offsets.
    if (d->aa) {
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const auto & c = planes[plane];
            auto & a = d->aplanes[plane];
            a.raw_offset = c.raw_offset;   a.raw_bytes = c.raw_bytes;
            a.raw2_offset = c.raw2_offset; a.raw2_bytes = c.raw2_bytes;
            a.sclip_offset = c.sclip_offset; a.sclip_bytes = c.sclip_bytes;
            a.sclip2_offset = c.sclip2_offset; a.sclip2_bytes = c.sclip2_bytes;
            a.bits_offset = c.bits_offset;  a.bits_bytes = c.bits_bytes;
            a.bits2_offset = c.bits2_offset; a.bits2_bytes = c.bits2_bytes;
            a.dl_offset = c.dl_offset;     a.dl_bytes = c.dl_bytes;
            a.dst_offset = c.dst_offset;   a.dst_bytes = c.dst_bytes;
            a.dst2_offset = c.dst2_offset; a.dst2_bytes = c.dst2_bytes;
            a.rempty_offset = c.rempty_offset; a.rempty_bytes = c.rempty_bytes;
            a.pbt_offset = c.pbt_offset;   a.pbt_bytes = c.pbt_bytes;
            a.dmap_offset = c.dmap_offset; a.dmap_bytes = c.dmap_bytes;
            a.cint_offset = c.cint_offset; a.cint_bytes = c.cint_bytes;
            a.vout_offset = c.vout_offset; a.vout_bytes = c.vout_bytes;
            a.vout2_offset = c.vout2_offset; a.vout2_bytes = c.vout2_bytes;
            a.built_offset = c.built_offset; a.built_bytes = c.built_bytes;
            a.rt_offset = c.rt_offset;     a.rt_bytes = c.rt_bytes;
            a.rtS_offset = c.rtS_offset;   a.rtS_bytes = c.rtS_bytes;
            a.v_offset = c.v_offset;       a.v_bytes = c.v_bytes;
            a.out_offset = c.out_offset;   a.out_bytes = c.out_bytes;
            a.out2_offset = c.out2_offset; a.out2_bytes = c.out2_bytes;
            a.ms_offset = c.ms_offset;     a.ms_bytes = c.ms_bytes;
        }
    }

    const VkDeviceSize staging_size = std::max(
        upload_total + download_total + scratch_total, VkDeviceSize(4));
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

    // Shared-memory vcheck opt-out (durable tuning knob and A/B switch): the
    // LDS path is strictly less memory traffic, but VSFEEL_EEDI3_VCLDS=0
    // forces the global-read form so the two can be measured in one binary.
    bool lds_ok = d->vcheck > 0 && d->vcheck_lds_module;
    if (const char * lv = std::getenv("VSFEEL_EEDI3_VCLDS")) {
        if (atoi(lv) == 0) {
            lds_ok = false;
        }
    }

    // helper to fetch-or-create the (row, vcheck, pad, vcopy, blit, vcheck_lds)
    // pipelines for a width key; uses Eedi3Data::WidthKey (all filter-level
    // params like vcheck/mclip are identical across planes, so the geometry
    // fields dominate). The pad/vcopy/blit upload kernels use fixed local
    // sizes, so the row/vcheck workgroup-size spec entries are ignored by
    // their pipelines. Widths that fit the shared-memory vcheck take the LDS
    // pipeline as their `vcheck` and leave vcopy unused.
    using WidthKey = Eedi3Data::WidthKey;
    auto get_pipelines = [&](const WidthKey & key, VkPipeline & row_pipe,
                             VkPipeline & vc_pipe, VkPipeline & pad_pipe,
                             VkPipeline & vcopy_pipe,
                             VkPipeline & blit_pipe,
                             VkPipeline & xpose_pipe,
                             VkPipeline & compose_pipe,
                             bool & use_lds) -> std::optional<std::string> {
        use_lds = lds_ok && key.width <= MAXW_LDS;
        for (auto & [k, quad] : d->width_pipes) {
            if (k == key) {
                row_pipe = quad[0];
                vc_pipe = quad[1];
                pad_pipe = quad[2];
                vcopy_pipe = quad[3];
                blit_pipe = quad[4];
                xpose_pipe = quad[6];
                compose_pipe = quad[7];
                // A cached entry wins only if it was built for the same d2p
                // path; entries are keyed by width so they are consistent.
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
            auto r2 = create_pipeline(*d->device, spec,
                                      use_lds ? d->vcheck_lds_module : d->vcheck_module,
                                      d->pipeline_layout, 0);
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
        if (d->vcheck > 0 && d->mclip_node && !use_lds) {
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
        VkPipeline blitp = VK_NULL_HANDLE;
        if (d->dst_host && d->device->host_import) {
            auto r5 = create_pipeline(*d->device, spec, d->blit_module, d->pipeline_layout, 0);
            if (std::holds_alternative<std::string>(r5)) {
                vkDestroyPipeline(dev, rowp, nullptr);
                if (vcp) {
                    vkDestroyPipeline(dev, vcp, nullptr);
                }
                vkDestroyPipeline(dev, padp, nullptr);
                if (vcopyp) {
                    vkDestroyPipeline(dev, vcopyp, nullptr);
                }
                return std::get<std::string>(r5);
            }
            blitp = std::get<VkPipeline>(r5);
        }
        // EEDI3H passes: the upload transpose and the output reassembly. Both
        // are per-width like everything else (WIDTH is the spec constant that
        // sizes their tiles), so they ride the same cache key.
        VkPipeline xposep = VK_NULL_HANDLE, composep = VK_NULL_HANDLE;
        if (d->horiz || d->aa) {
            auto r6 = create_pipeline(*d->device, spec, d->xpose_module,
                                      d->pipeline_layout, 0);
            if (std::holds_alternative<std::string>(r6)) {
                vkDestroyPipeline(dev, rowp, nullptr);
                if (vcp) { vkDestroyPipeline(dev, vcp, nullptr); }
                vkDestroyPipeline(dev, padp, nullptr);
                if (vcopyp) { vkDestroyPipeline(dev, vcopyp, nullptr); }
                if (blitp) { vkDestroyPipeline(dev, blitp, nullptr); }
                return std::get<std::string>(r6);
            }
            xposep = std::get<VkPipeline>(r6);
            auto r7 = create_pipeline(*d->device, spec, d->compose_module,
                                      d->pipeline_layout, 0);
            if (std::holds_alternative<std::string>(r7)) {
                vkDestroyPipeline(dev, rowp, nullptr);
                if (vcp) { vkDestroyPipeline(dev, vcp, nullptr); }
                vkDestroyPipeline(dev, padp, nullptr);
                if (vcopyp) { vkDestroyPipeline(dev, vcopyp, nullptr); }
                if (blitp) { vkDestroyPipeline(dev, blitp, nullptr); }
                vkDestroyPipeline(dev, xposep, nullptr);
                return std::get<std::string>(r7);
            }
            composep = std::get<VkPipeline>(r7);
        }
        d->width_pipes.emplace_back(key, std::array<VkPipeline, 8>{
            rowp, vcp, padp, vcopyp, blitp, VK_NULL_HANDLE, xposep, composep });
        row_pipe = rowp;
        vc_pipe = vcp;
        pad_pipe = padp;
        vcopy_pipe = vcopyp;
        blit_pipe = blitp;
        xpose_pipe = xposep;
        compose_pipe = composep;
        return std::nullopt;
    };

    for (int plane = 0; plane < numPlanes; ++plane) {
        if (!d->process[plane]) {
            continue;
        }
        auto & cfg = planes[plane];
        WidthKey key { cfg.width, cfg.rows, cfg.tpitch, cfg.pad_stride, cfg.pad_height };
        if (auto err = get_pipelines(key, cfg.row_pipeline, cfg.vcheck_pipeline,
                                     cfg.pad_pipeline, cfg.vcopy_pipeline,
                                     cfg.blit_pipeline, cfg.xpose_pipeline,
                                     cfg.compose_pipeline, cfg.vcheck_lds)) {
            return set_error(*err);
        }
        if (d->aa) {
            // EEDI3AA: the horizontal pass is a second geometry over the same
            // buffers, and the vertical merge is its own kernel.
            auto & a = d->aplanes[plane];
            WidthKey akey { a.width, a.rows, a.tpitch, a.pad_stride, a.pad_height };
            if (auto err = get_pipelines(akey, a.row_pipeline, a.vcheck_pipeline,
                                         a.pad_pipeline, a.vcopy_pipeline,
                                         a.blit_pipeline, a.xpose_pipeline,
                                         a.compose_pipeline, a.vcheck_lds)) {
                return set_error(*err);
            }
            VkPipeline asm_pipe = VK_NULL_HANDLE;
            for (auto & [w, p] : d->assemble_pipes) {
                if (w == cfg.width) {
                    asm_pipe = p;
                    break;
                }
            }
            if (!asm_pipe) {
                RowSpecData aspec = base_spec;
                aspec.width = cfg.width;
                auto ra = create_pipeline(*d->device, aspec, d->assemble_module,
                                          d->pipeline_layout, 0);
                if (std::holds_alternative<std::string>(ra)) {
                    return set_error(std::get<std::string>(ra));
                }
                asm_pipe = std::get<VkPipeline>(ra);
                d->assemble_pipes.emplace_back(cfg.width, asm_pipe);
            }
            cfg.assemble_pipeline = asm_pipe;
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
            if (d->aa) {
                checkVK(vkAllocateDescriptorSets(dev, &alloc_info, &resource.desc_set_h));
                checkVK(vkAllocateDescriptorSets(dev, &alloc_info, &resource.desc_set_xp));
            }
            if (d->dst_host && d->device->host_import) {
                for (int p = 0; p < MAX_PLANES; ++p) {
                    checkVK(vkAllocateDescriptorSets(dev, &alloc_info,
                                                     &resource.desc_set_blit[p]));
                }
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
            // The upload mirror (raw rows + sclip) lives in up_dev under
            // ReBAR and in pad_dev otherwise.
            VkBuffer const bmask_buf = (d->rebar_up && upload_total > 0)
                ? resource.up_dev : resource.pad_dev;
            // Packed bits are always written to the cached staging mirror (see
            // get_frame) -- ordinary stores into the uncached host-visible VRAM
            // are one transaction per word, and that showed up as the single
            // largest host stage. Under ReBAR the kernel reads them from
            // staging; otherwise the mirror copy in pad_dev already has them.
            VkDescriptorBufferInfo bmask_info {
                .buffer = (d->rebar_up && upload_total > 0)
                    ? resource.staging : resource.pad_dev,
                .offset = 0, .range = VK_WHOLE_SIZE
            };
            // EEDI3H keeps the transposed sclip (binding 9's sibling, written
            // by the xpose pass) in pad_dev; the host-uploaded compact sclip K
            // is only the xpose pass's source (binding 0).
            VkDescriptorBufferInfo sclip_info {
                .buffer = d->horiz ? resource.pad_dev : bmask_buf,
                .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo cint_info {
                .buffer = resource.dev_buf, .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo vout_info {
                .buffer = d->vout_in_dev() ? resource.dev_buf : resource.staging,
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
            // Binding 9 is the pad builder's source plane. Vertically that is
            // the host-uploaded raw (up_dev under ReBAR, the pad_dev mirror
            // otherwise, i.e. exactly binding 0's buffer); EEDI3H always uses
            // the device-local R'/B' regions instead.
            VkDescriptorBufferInfo padsrc_info {
                .buffer = d->horiz ? resource.pad_dev : raw_info.buffer,
                .offset = 0, .range = VK_WHOLE_SIZE
            };
            VkDescriptorBufferInfo out_info {
                .buffer = resource.staging, .offset = 0, .range = VK_WHOLE_SIZE
            };
            const VkDescriptorBufferInfo * infos[BIND_COUNT] {
                &pad_info, &dst_info, &pbt_info, &dmap_info,
                &bmask_info, &sclip_info, &cint_info, &vout_info,
                &raw_info, &padsrc_info, &out_info
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

            if (d->aa) {
                // Every region the horizontal geometry reads through a
                // geometry-dependent binding (the built pad at b0, the
                // transposed sclip B' at b5, R' at b9) lives in pad_dev.
                VkDescriptorBufferInfo pad_h_info {
                    .buffer = resource.pad_dev, .offset = 0, .range = VK_WHOLE_SIZE
                };
                VkWriteDescriptorSet h_writes[BIND_COUNT];
                for (uint32_t b = 0; b < BIND_COUNT; ++b) {
                    h_writes[b] = writes[b];
                    h_writes[b].dstSet = resource.desc_set_h;
                    h_writes[b].pBufferInfo = (b == 5 || b == 9) ? &pad_h_info
                                                                 : infos[b];
                }
                vkUpdateDescriptorSets(dev, BIND_COUNT, h_writes, 0, nullptr);

                // xpose + the horizontal pad builder: b0 = the raw/K upload
                // (up_dev under ReBAR, the pad_dev mirror otherwise), b8 =
                // the built pad, b9 = R'.
                VkWriteDescriptorSet xp_writes[BIND_COUNT];
                for (uint32_t b = 0; b < BIND_COUNT; ++b) {
                    xp_writes[b] = writes[b];
                    xp_writes[b].dstSet = resource.desc_set_xp;
                    xp_writes[b].pBufferInfo = (b == 0) ? &raw_info
                        : (b == 8) ? &pad_info
                        : (b == 9) ? &pad_h_info : infos[b];
                }
                vkUpdateDescriptorSets(dev, BIND_COUNT, xp_writes, 0, nullptr);
            }

            if (d->dst_host && d->device->host_import) {
                // One set per plane for the direct-to-frame blit: the shader
                // only reads b7 (tight vout rows) and writes b1, whose buffer
                // is re-pointed at that plane's imported frame every frame
                // before the command buffer is recorded. All other bindings
                // are written with valid defaults so no binding is left
                // undefined.
                for (int p = 0; p < MAX_PLANES; ++p) {
                    VkWriteDescriptorSet blit_writes[BIND_COUNT];
                    for (uint32_t b = 0; b < BIND_COUNT; ++b) {
                        blit_writes[b] = writes[b];
                        blit_writes[b].dstSet = resource.desc_set_blit[p];
                    }
                    vkUpdateDescriptorSets(dev, BIND_COUNT, blit_writes, 0, nullptr);
                }
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

    // Dependencies. EEDI3/EEDI3H are temporal when field > 1 (each output frame
    // pairs two input frames... actually it re-reads per sub-frame, so general).
    // EEDI3AA consumes only input frame n plus sclip[2n], sclip[2n+1] and
    // mclip[n], so it stays strictly spatial and lets the scheduler parallelise.
    std::vector<VSFilterDependency> deps;
    const bool general = d->field > 1 && !d->aa;
    deps.push_back({ d->node, general ? rpGeneral : rpStrictSpatial });
    if (d->vcheck > 0 && d->sclip_node) {
        deps.push_back({ d->sclip_node, rpStrictSpatial });
    }
    if (d->mclip_node) {
        deps.push_back({ d->mclip_node, general ? rpGeneral : rpStrictSpatial });
    }

    // Store the output dims into d->vi? No: createVideoFilter reads the local
    // `out_video`. The data struct keeps a pointer to the ORIGINAL vi (input
    // dims) — the frame processing reads src planes (original dims) and writes
    // the dst frame (output dims). Keep both.
    Eedi3Data * data = d.release();

    vsapi->createVideoFilter(
        out, aa ? "EEDI3AA" : (horiz ? "EEDI3H" : "EEDI3"), &out_video,
        aa ? Eedi3AaGetFrame : Eedi3GetFrame, Eedi3Free,
        fmParallel, deps.data(), static_cast<int>(deps.size()), data, core);
}

// ---------------------------------------------------------------------------
// EEDI3H — horizontal EEDI3 (vszipcl parity).
//
// Thin wrapper over the same implementation with the transposed geometry: the
// kernels are orientation-agnostic, so EEDI3H only changes what the host
// gathers (source columns instead of source rows) and how the result is
// reassembled (one GPU pass instead of four std.Transpose nodes). Everything
// else -- validation, numerics, mclip/sclip/dh/field semantics -- is EEDI3's,
// and by construction
//   EEDI3H(x) == Transpose(EEDI3(Transpose(x)))
// bit-exactly, which is also the primary test oracle (tests/test_eedi3h.py).
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

// ---------------------------------------------------------------------------
// EEDI3AA — fused based_aa EEDI3 chain.
//
// One call runs the whole `Merge(H(Merge(V(clip))))` chain: the vertical pass
// on the source, the 50/50 merge into the intermediate frame v, the horizontal
// pass on v, and the final 50/50 merge. Single-rate (N in, N out), dh disabled,
// field 2 or 3 (the doubled sub-frame parities). It shares every kernel, host
// gather and buffer with EEDI3/EEDI3H, so those stay the reference filters and
// a live regression net for the shared code.
// ---------------------------------------------------------------------------

static void VS_CC Eedi3AaCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {
    vsfeel_eedi3_create(in, out, userData, core, vsapi, false, true);
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
        Eedi3HCreate, nullptr, plugin
    );
    vspapi->registerFunction(
        "EEDI3AA",
        eedi3_args,
        "clip:vnode;",
        Eedi3AaCreate, nullptr, plugin
    );
}
