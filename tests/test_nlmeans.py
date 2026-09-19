"""Unit tests for core.vsfeel.NLMeans.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The noise content makes every parameter axis observable: NLM must
actually alter it, and any misindexing of the padded window or sweep tables
shows up as a large diff against the reference implementation.

Run from the repository root:  python -m pytest tests/test_nlmeans.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_changes_on_noise, assert_gray32, assert_preserves_frame_props,
    assert_temporal_order_consistent, eval_parallel, max_diff, plane as _plane,
    reference_compare, reference_or_skip, reference_spec,
)

pytestmark = pytest.mark.usefixtures("noise_gray")

H_PARAM = 1.2


def _run(clip, num_streams=1, **kwargs):
    return vs.core.vsfeel.NLMeans(clip, num_streams=num_streams, **kwargs)


def _ref_compare(fmt, frames, params, planes=None, guide=None, crop=None):
    """Worst diff vs vszipcl over ``frames`` (subprocess; skips if absent).

    ``fmt`` is a format name understood by ``conftest.REFERENCE_SCRIPT``;
    ``guide`` describes the rclip when the case uses one.
    """
    reference_or_skip("vszipcl", "NLMeans")
    spec = reference_spec("vszipcl", "NLMeans", fmt, frames=frames,
                          planes=planes, kwargs=params, guide=guide,
                          guide_kwarg="rclip", crop=crop)
    return reference_compare(spec)["maxdiff"]


# --- basic behaviour ---------------------------------------------------------


def test_output_format_and_frames_preserved(noise_gray):
    out = _run(noise_gray, d=0)
    assert out.format.id == noise_gray.format.id
    assert (out.width, out.height) == (noise_gray.width, noise_gray.height)
    assert out.num_frames == noise_gray.num_frames


def test_nlmeans_preserves_frame_props(noise_gray):
    """NLMeans must republish the source frame's properties."""
    assert_preserves_frame_props(_run, noise_gray, d=0, num_streams=1)


def test_denoise_changes_output_32bit(noise_gray):
    src = noise_gray
    out = _run(src, d=0)
    assert np.abs(_plane(out.get_frame(5), 0) - _plane(src.get_frame(5), 0)).max() > 0.0


def test_higher_h_smooths_more_32bit(noise_gray):
    src = noise_gray
    weak = _run(src, d=0, h=0.3)
    strong = _run(src, d=0, h=4.0)
    dw = np.abs(_plane(weak.get_frame(5), 0) - _plane(src.get_frame(5), 0)).max()
    ds = np.abs(_plane(strong.get_frame(5), 0) - _plane(src.get_frame(5), 0)).max()
    assert ds > dw


def test_search_radius_changes_output_32bit(noise_gray):
    a1 = _run(noise_gray, d=0, a=1)
    a4 = _run(noise_gray, d=0, a=4)
    assert max_diff(a1, a4, frames=(5,)) > 0.0


def test_patch_size_changes_output_32bit(noise_gray):
    s1 = _run(noise_gray, d=0, s=1)
    s3 = _run(noise_gray, d=0, s=3)
    assert max_diff(s1, s3, frames=(5,)) > 0.0


def test_wmode_changes_output_32bit(noise_gray):
    w0 = _run(noise_gray, d=0, wmode=0)
    w3 = _run(noise_gray, d=0, wmode=3)
    assert max_diff(w0, w3, frames=(5,)) > 0.0


def test_wref_changes_output_32bit(noise_gray):
    w1 = _run(noise_gray, d=0, wref=1.0)
    w0 = _run(noise_gray, d=0, wref=0.0)
    assert max_diff(w1, w0, frames=(5,)) > 0.0


def test_temporal_differs_from_spatial_32bit(noise_gray):
    # the noise clip is independent per frame, so a temporal window must
    # produce a different result than spatial-only
    spatial = _run(noise_gray, d=0)
    temporal = _run(noise_gray, d=2)
    assert max_diff(spatial, temporal) > 0.0


# --- correctness vs the reference --------------------------------------------

# Parameter sweep vs vszipcl. Tolerance set from measurement: diffs come
# from fp32 accumulation order in the weighted average (more taps at larger
# a/s accumulate more), landing in the 0 .. 6e-5 band across all configs and
# all special paths (UV/RGB joint, rclip guide, cropped stride). The bound
# keeps ~40% headroom over the worst measurement while still catching real
# misindexing (which produces O(0.1..1) errors).
NLMEANS_REF_TOL = 1e-4

REFERENCE_CASES = [
    {"d": 0},
    {"d": 2},
    {"d": 1, "a": 3, "s": 3, "h": 3.0, "wref": 0.4},
    {"d": 0, "wmode": 1},
    {"d": 0, "wmode": 2, "h": 2.0},
    {"d": 0, "wmode": 3},
    {"d": 0, "a": 1},
    {"d": 0, "a": 4},
    {"d": 0, "s": 1},
    {"d": 0, "s": 8},
    {"d": 0, "h": 0.3},
    {"d": 0, "h": 6.0},
    # wref=0 at the default h is outside the fp16 weight ring's envelope (see
    # the envelope tests below); at h=3.0 both sides agree to ~1 code.
    {"d": 0, "wref": 0.0, "h": 3.0},
]


@pytest.mark.parametrize("kwargs", REFERENCE_CASES, ids=lambda kw: str(kw))
def test_matches_reference_32bit(noise_gray, kwargs):
    worst = _ref_compare("gray32", (0, 11, 23), dict(num_streams=1, **kwargs))
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl {kwargs}: {worst}"


GRAY16_CASES = [
    {"d": 1},
    {"d": 2},
    {"d": 0, "wmode": 2, "h": 2.0},
    {"d": 0, "wmode": 3},
    {"d": 1, "a": 4},
]


@pytest.mark.parametrize("kwargs", GRAY16_CASES, ids=lambda kw: str(kw))
def test_matches_reference_16bit(noise_16bit, kwargs):
    """Integer rounding path: both sides round nearly identical fp32 results
    once, so codes differ by at most one step."""
    worst = _ref_compare("gray16", (0, 11, 23), dict(num_streams=1, **kwargs))
    assert worst <= 1.0, f"max LSB diff vs vszipcl {kwargs}: {worst}"


# Positive maxima of the three radii (a=64, s=8, d=16): the m=0..2
# sweep-table variants and run-group boundaries the a<=4 / d<=2 sweep misses.
# At the default h=1.2 the largest push the fp16 weight ring to its envelope
# (a=64 2.02e-3, s=8+a=64 1.42e-2) while the same configs at h=3.0 land at
# 1.6e-6 and wmode 1/2/3 at a=64 stays ~1e-6, so the drift is weight
# quantisation, not indexing. d=16 with s=8 and a=64 together is omitted: it
# hard-recovers the GPU.
POSITIVE_MAX_CASES = [
    # (kwargs, bound, measured)
    ({"d": 0, "a": 64}, 3e-3, 2.02e-3),
    ({"d": 0, "s": 8, "a": 64}, 2e-2, 1.42e-2),
    ({"d": 0, "a": 64, "h": 3.0}, NLMEANS_REF_TOL, 1.61e-6),
    ({"d": 0, "s": 8, "a": 64, "h": 3.0}, NLMEANS_REF_TOL, 1.46e-6),
    ({"d": 16}, NLMEANS_REF_TOL, 4.22e-6),
    ({"d": 16, "a": 64}, NLMEANS_REF_TOL, 6.97e-5),
    ({"d": 16, "a": 64, "h": 3.0}, NLMEANS_REF_TOL, 1.54e-5),
]


@pytest.mark.parametrize("kwargs,bound,measured", POSITIVE_MAX_CASES,
                         ids=[str(kw) for kw, _, _ in POSITIVE_MAX_CASES])
def test_matches_reference_positive_maxima_32bit(noise_gray, kwargs, bound,
                                                 measured):
    worst = _ref_compare("gray32", (0, 11, 23), dict(num_streams=1, **kwargs))
    assert worst < bound, \
        f"max diff vs vszipcl {kwargs}: {worst} (bound {bound}, was {measured})"


POSITIVE_MAX_16_CASES = [
    # (kwargs, bound, measured codes)
    ({"d": 16}, 1.0, 1),
    ({"d": 16, "s": 8}, 1.0, 1),
    ({"d": 16, "a": 64}, 8.0, 5),
    ({"d": 16, "a": 64, "h": 3.0}, 1.0, 1),
    ({"d": 0, "a": 64}, 200.0, 133),          # fp16 weight envelope at h=1.2
    ({"d": 0, "a": 64, "h": 3.0}, 1.0, 1),    # same config, weights representable
    ({"d": 0, "s": 8, "a": 64, "h": 3.0}, 1.0, 1),
]


@pytest.mark.parametrize("kwargs,bound,measured", POSITIVE_MAX_16_CASES,
                         ids=[str(kw) for kw, _, _ in POSITIVE_MAX_16_CASES])
def test_matches_reference_positive_maxima_16bit(noise_16bit, kwargs, bound,
                                                 measured):
    """16-bit mirror: whole output codes. The a=64/h=1.2 entry carries the
    same fp16-weight envelope (133 codes) and is 1 code at h=3.0."""
    worst = _ref_compare("gray16", (0, 11, 23), dict(num_streams=1, **kwargs))
    assert worst <= bound, \
        f"max LSB diff vs vszipcl {kwargs}: {worst} (bound {bound}, was {measured})"


def test_yuv_default_denises_luma_copies_chroma_32bit(noise_yuv32):
    src = noise_yuv32
    out = _run(src, d=0)
    assert max_diff(out, src, planes=(0,), frames=(5,)) > 0.0
    for p in (1, 2):
        fa = _plane(out.get_frame(5), p)
        fb = _plane(src.get_frame(5), p)
        assert np.array_equal(fa, fb), f"chroma{p} changed"


def test_yuv_default_denises_luma_copies_chroma_16bit(noise_yuv420_16):
    """16-bit mirror of test_yuv_default_denises_luma_copies_chroma."""
    src = noise_yuv420_16
    out = _run(src, d=0)
    assert max_diff(out, src, planes=(0,), frames=(5,)) > 0.0
    for p in (1, 2):
        fa = _plane(out.get_frame(5), p)
        fb = _plane(src.get_frame(5), p)
        assert np.array_equal(fa, fb), f"chroma{p} changed"
    worst = _ref_compare("yuv420_16", (5,), dict(num_streams=1, d=0),
                         planes=(0,))
    assert worst <= 1.0


def test_yuv_channels_uv_matches_reference_32bit(noise_yuv32):
    """channels='UV' on a subsampled YUV clip: chroma denoised (subsampled
    lattice), luma passed through bit-exactly."""
    src = noise_yuv32
    assert src.format.subsampling_w == 1 and src.format.subsampling_h == 1

    out = _run(src, d=0, channels="UV", h=1.5)
    assert max_diff(out, src, planes=(0,), frames=(5,)) == 0.0
    assert max_diff(out, src, planes=(1, 2), frames=(5,)) > 0.0

    worst = _ref_compare("yuv32", (0, 11, 23),
                         dict(num_streams=1, d=0, channels="UV", h=1.5),
                         planes=(1, 2))
    assert worst < NLMEANS_REF_TOL


# --- UV (chroma-only, 2-channel) sweep at both depths vs the reference ------
#
# The luma sweeps above only exercise 1 channel; the 16-bit 'UV' path was once
# broken while every other depth/channel-count combination stayed correct, so
# the 2-channel sweep must run at BOTH depths. Parameter list mirrors
# REFERENCE_CASES. Tolerances measured on the noise clip:
#   - 32-bit: worst 8.31e-5 across the sweep (fp32 accumulation order),
#     bound NLMEANS_REF_TOL as for the luma sweep.
#   - 16-bit: worst 4 codes (integer rounding + fp32 order on the subsampled
#     lattice), bound 8.0 with 2x headroom. The wref=0 case is run with h=3.0
#     instead of the default h=1.2: with wref=0 the tiny exp() weights of a
#     noise clip fall into fp16 subnormals (x4096 store), which drifts the
#     weighted average by thousands of codes at h=1.2 (measured 4421 LSB for
#     UV16 and 4619 LSB for GRAY16 identically - a general 16-bit property of
#     the fp16 weight ring, not a per-channel indexing fault), and drops to
#     1 LSB at h=3.0.

UV32_CASES = [
    {"d": 0},
    {"d": 2},
    {"d": 1, "a": 3, "s": 3, "h": 3.0, "wref": 0.4},
    {"d": 0, "wmode": 1},
    {"d": 0, "wmode": 2, "h": 2.0},
    {"d": 0, "wmode": 3},
    {"d": 0, "a": 1},
    {"d": 0, "a": 4},
    {"d": 0, "s": 1},
    {"d": 0, "s": 8},
    {"d": 0, "h": 0.3},
    {"d": 0, "h": 6.0},
    # No wref=0 entry: at fp32 vszipcl's chroma path emits non-finite pixels
    # (~16/plane/frame) and a few wild ones on the last frame, so an equality
    # comparison is not meaningful. vsfeel's finiteness is pinned below.
]

UV16_CASES = [
    {"d": 0},
    {"d": 2},
    {"d": 1, "a": 3, "s": 3, "h": 3.0, "wref": 0.4},
    {"d": 0, "wmode": 1},
    {"d": 0, "wmode": 2, "h": 2.0},
    {"d": 0, "wmode": 3},
    {"d": 0, "a": 1},
    {"d": 0, "a": 4},
    {"d": 0, "s": 1},
    {"d": 0, "s": 8},
    {"d": 0, "h": 0.3},
    {"d": 0, "h": 6.0},
    {"d": 0, "wref": 0.0, "h": 3.0},  # see note above
]

UV16_REF_TOL = 8.0


@pytest.mark.parametrize("kwargs", UV32_CASES, ids=lambda kw: str(kw))
def test_uv_matches_reference_32bit(noise_yuv32, kwargs):
    worst = _ref_compare("yuv32", (0, 11, 23),
                         dict(num_streams=1, channels="UV", **kwargs),
                         planes=(1, 2))
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl {kwargs}: {worst}"


@pytest.mark.parametrize("kwargs", UV16_CASES, ids=lambda kw: str(kw))
def test_uv_matches_reference_16bit(noise_yuv420_16, kwargs):
    worst = _ref_compare("yuv420_16", (0, 11, 23),
                         dict(num_streams=1, channels="UV", **kwargs),
                         planes=(1, 2))
    assert worst <= UV16_REF_TOL, f"max LSB diff vs vszipcl {kwargs}: {worst}"

ENVELOPE_FRAMES = (0, 11, 23)

def test_wref0_low_h_is_finite_32bit(noise_gray):
    """A fully flushed weight ring must fall back to the centre sample, not 0/0.

    Covers both the exp() ring (wmode 0, fp16 flush) and the truncated modes
    (wmode 1-3, max(1-arg,0) == 0 for every tap).
    """
    for h in (0.6, 1.0, 1.2):
        for wmode in (0, 1, 2, 3):
            out = _run(noise_gray, d=0, h=h, wmode=wmode, wref=0.0)
            for n in ENVELOPE_FRAMES:
                got = _plane(out.get_frame(n), 0)
                assert np.isfinite(got).all(), f"h={h} wmode={wmode} n={n}"


def test_wref0_low_h_is_finite_uv_32bit(noise_yuv32):
    """Same guard on the 2-channel path."""
    out = _run(noise_yuv32, channels="UV", d=0, wref=0.0)
    for n in ENVELOPE_FRAMES:
        for p in (1, 2):
            assert np.isfinite(_plane(out.get_frame(n), p)).all(), f"n={n} p={p}"


def test_wref0_low_h_envelope_16bit(noise_16bit):
    """Pin the measured deviation at the envelope edge (h=1.2, wref=0)."""
    worst = _ref_compare("gray16", ENVELOPE_FRAMES,
                         dict(num_streams=1, d=0, wref=0.0))
    assert 1000.0 < worst < 6000.0, f"fp16 weight-ring envelope moved: {worst}"


def test_wref0_low_h_envelope_32bit(noise_gray):
    """Pin the same edge at fp32 (float units, ~4573 codes measured)."""
    worst = _ref_compare("gray32", ENVELOPE_FRAMES,
                         dict(num_streams=1, d=0, wref=0.0))
    assert 0.03 < worst < 0.12, f"fp16 weight-ring envelope moved: {worst}"


def test_yuv_channels_uv_temporal_matches_reference_32bit(noise_yuv32):
    worst = _ref_compare("yuv32", (0, 11, 23),
                         dict(num_streams=2, d=1, channels="UV", h=1.5),
                         planes=(1, 2))
    assert worst < NLMEANS_REF_TOL


def test_yuv444_joint_matches_reference_16bit(noise_yuv444_16):
    # joint processing sums distances across three planes before rounding,
    # so the fp divergence reaches two output codes (measured); single-plane
    # paths stay within one
    worst = _ref_compare("yuv444_16", (0, 11, 23),
                         dict(num_streams=1, d=0, channels="YUV", h=1.0))
    assert worst <= 2.0


def test_rgb_joint_matches_reference_32bit(noise_rgb32):
    worst = _ref_compare("rgb32", (0, 11, 23),
                         dict(num_streams=1, d=1, h=1.0))
    assert worst < NLMEANS_REF_TOL


def test_rgb_joint_matches_reference_16bit(noise_rgb16):
    """16-bit mirror of test_rgb_joint_matches_reference (whole codes)."""
    worst = _ref_compare("rgb16", (0, 11, 23),
                         dict(num_streams=1, d=1, h=1.0))
    assert worst <= 1.0


def test_rclip_self_is_identity_32bit(noise_gray):
    src = noise_gray
    plain = _run(src, d=0, h=1.5)
    withref = _run(src, d=0, h=1.5, rclip=src)
    assert max_diff(plain, withref, frames=(5,)) == 0.0


def test_rclip_self_is_identity_16bit(noise_16bit):
    """16-bit mirror of test_rclip_self_is_identity."""
    src = noise_16bit
    plain = _run(src, d=0, h=1.5)
    withref = _run(src, d=0, h=1.5, rclip=src)
    assert max_diff(plain, withref, frames=(5,)) == 0.0


def test_rclip_guide_matches_reference_32bit(noise_gray):
    src = noise_gray
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    mine = _run(src, d=0, h=1.5, rclip=guide)
    plain = _run(src, d=0, h=1.5)
    assert max_diff(mine, plain, frames=(5,)) > 0.0
    worst = _ref_compare("gray32", (0, 11, 23),
                         dict(num_streams=1, d=0, h=1.5),
                         guide={"kind": "boxblur", "hradius": 5, "vradius": 5})
    assert worst < NLMEANS_REF_TOL


def test_rclip_guide_matches_reference_16bit(noise_16bit):
    """16-bit mirror of test_rclip_guide_matches_reference (whole codes)."""
    src = noise_16bit
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    mine = _run(src, d=0, h=1.5, rclip=guide)
    plain = _run(src, d=0, h=1.5)
    assert max_diff(mine, plain, frames=(5,)) > 0.0
    worst = _ref_compare("gray16", (0, 11, 23),
                         dict(num_streams=1, d=0, h=1.5),
                         guide={"kind": "boxblur", "hradius": 5, "vradius": 5})
    assert worst <= 1.0


@pytest.mark.parametrize("d", [1, 2])
def test_rclip_temporal_matches_reference_early_frames(noise_gray, d):
    """Frames n < d use a shorter window (2*min(d,n)+1 layers), where the
    guide clip's slot table used to be indexed with the full-window stride
    and read past win_slots. Pre-fix diff vs vszipcl was 2.4e-2 (d=1) and
    6.7e-3 (d=2); frames n >= d matched to ~1e-6."""
    worst = _ref_compare("gray32", (0, 1, 2, 3),
                         dict(num_streams=1, d=d, h=1.5),
                         guide={"kind": "boxblur", "hradius": 5, "vradius": 5})
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl (d={d}): {worst}"


def test_stride_handling_matches_reference_32bit(noise_gray):
    # a cropped frame keeps its parent's (wider) stride; the filter must
    # handle non-tight pitches identically to the reference
    assert noise_gray.std.Crop(left=27).width < noise_gray.width
    worst = _ref_compare("gray32", (0, 11, 23), dict(num_streams=1, d=1),
                         crop={"left": 27})
    assert worst < NLMEANS_REF_TOL


def test_stride_handling_matches_reference_16bit(noise_16bit):
    """16-bit mirror of test_stride_handling_matches_reference (whole
    codes)."""
    assert noise_16bit.std.Crop(left=27).width < noise_16bit.width
    worst = _ref_compare("gray16", (0, 11, 23), dict(num_streams=1, d=1),
                         crop={"left": 27})
    assert worst <= 1.0


# --- determinism / streams ---------------------------------------------------


def test_deterministic_serial_32bit(noise_gray):
    a = _run(noise_gray, d=2, h=1.5)
    b = _run(noise_gray, d=2, h=1.5)
    assert max_diff(a, b) == 0.0


def test_deterministic_serial_16bit(noise_16bit):
    a = _run(noise_16bit, d=2, h=1.5)
    b = _run(noise_16bit, d=2, h=1.5)
    assert max_diff(a, b) == 0.0


def test_multi_stream_matches_single_32bit(noise_gray):
    a = _run(noise_gray, d=2, num_streams=4)
    b = _run(noise_gray, d=2, num_streams=1)
    assert max_diff(a, b) == 0.0


def test_multi_stream_matches_single_16bit(noise_16bit):
    a = _run(noise_16bit, d=2, num_streams=4)
    b = _run(noise_16bit, d=2, num_streams=1)
    assert max_diff(a, b) == 0.0


def test_multi_stream_temporal_matches_single_32bit(noise_yuv32):
    a = _run(noise_yuv32, d=1, channels="UV", num_streams=4)
    b = _run(noise_yuv32, d=1, channels="UV", num_streams=1)
    assert max_diff(a, b, planes=(1, 2)) == 0.0


def test_multi_stream_uv_matches_single_16bit(noise_yuv420_16):
    a = _run(noise_yuv420_16, d=1, channels="UV", num_streams=4)
    b = _run(noise_yuv420_16, d=1, channels="UV", num_streams=1)
    assert max_diff(a, b, planes=(1, 2)) == 0.0


def test_parallel_load_matches_serial_32bit(noise_gray):
    par = eval_parallel(_run, noise_gray, d=2, num_streams=4)
    ref = _run(noise_gray, d=2, num_streams=1)
    for n in range(noise_gray.num_frames):
        fb = _plane(ref.get_frame(n), 0)
        assert np.abs(par[n].astype(np.float64) - fb).max() == 0.0, \
            f"parallel/serial mismatch at frame {n}"


def test_parallel_load_matches_serial_16bit(noise_16bit):
    par = eval_parallel(_run, noise_16bit, d=2, num_streams=4)
    ref = _run(noise_16bit, d=2, num_streams=1)
    for n in range(noise_16bit.num_frames):
        fb = _plane(ref.get_frame(n), 0)
        assert np.abs(par[n].astype(np.float64) - fb).max() == 0.0, \
            f"parallel/serial mismatch at frame {n}"


def test_parallel_load_deterministic_32bit(noise_gray):
    a = eval_parallel(_run, noise_gray, d=2, num_streams=4)
    b = eval_parallel(_run, noise_gray, d=2, num_streams=4)
    for n in range(noise_gray.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


def test_parallel_load_deterministic_16bit(noise_16bit):
    a = eval_parallel(_run, noise_16bit, d=2, num_streams=4)
    b = eval_parallel(_run, noise_16bit, d=2, num_streams=4)
    for n in range(noise_16bit.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


# --- frame-request order / short clips ---------------------------------------

@pytest.mark.parametrize("d", [2, 16])
def test_frame_request_order_matches_serial(d):
    """The tile cache must return the same pixels whatever order frames arrive.

    A cached slot keyed and released by frame index instead of a per-request
    token can be freed while another request still reads it; only an
    out-of-sequence request pattern makes that observable.
    """
    assert_temporal_order_consistent(
        "NLMeans", {"d": d, "num_streams": 4}, tol=0.0, timeout=300)


@pytest.mark.parametrize("nframes", [1, 2])
def test_short_clip_temporal_window(nframes):
    """A clip shorter than the temporal window is order-independent too."""
    assert_temporal_order_consistent(
        "NLMeans", {"d": 2, "num_streams": 4}, tol=0.0, nframes=nframes,
        timeout=300)


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_no_nan_all_frames_32bit(noise_gray, radius):
    out = _run(noise_gray, d=radius)
    assert_gray32(out)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"
        assert (a >= 0.0).all() and (a <= 1.0).all(), f"out-of-range output at frame {n}"


@pytest.mark.parametrize("num_streams", [2, 4])
def test_no_nan_all_frames_multi_stream_32bit(noise_gray, num_streams):
    out = _run(noise_gray, d=2, num_streams=num_streams)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_no_nan_all_frames_16bit(noise_16bit, radius):
    """Finiteness on the integer path is trivially true (uint16), so the
    meaningful half is the anti-vacuity check: the filter must alter the
    noise rather than pass it through."""
    out = _run(noise_16bit, d=radius)
    assert out.format.id == noise_16bit.format.id
    assert_changes_on_noise(out, noise_16bit, what="NLMeans")


# --- validation errors -------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(d=-1), r"d must be 0\.\.16"),
        (dict(d=17), r"d must be 0\.\.16"),
        (dict(a=0), r"a must be 1\.\.64"),
        (dict(a=65), r"a must be 1\.\.64"),
        (dict(s=-1), r"s must be 0\.\.8"),
        (dict(s=9), r"s must be 0\.\.8"),
        (dict(h=0), r"h must be > 0"),
        (dict(h=-1.0), r"h must be > 0"),
        (dict(wmode=-1), r"wmode must be 0\.\.3"),
        (dict(wmode=4), r"wmode must be 0\.\.3"),
        (dict(wref=-0.5), r"wref must be >= 0"),
        (dict(num_streams=0), r"num_streams must be 1\.\.32"),
        (dict(num_streams=33), r"num_streams must be 1\.\.32"),
        (dict(device_id=-1), r"invalid device ID"),
        (dict(channels="bogus"), r"'channels' must be 'Y' with Gray"),
        (dict(channels="UV"), r"'channels' must be 'Y' with Gray"),
    ],
)
def test_validation_errors_gray(noise_gray, args, msg):
    with pytest.raises(vs.Error, match=msg):
        _run(noise_gray, **args)


def test_rejects_8bit(noise_8bit):
    with pytest.raises(vs.Error, match=r"input bitdepth must be"):
        _run(noise_8bit)


def test_channels_yuv_requires_444(noise_yuv32):
    with pytest.raises(vs.Error, match=r"'channels'='YUV' requires 4:4:4"):
        _run(noise_yuv32, channels="YUV")


def test_channels_invalid_on_yuv(noise_yuv32):
    with pytest.raises(vs.Error, match=r"'channels' must be 'YUV', 'Y' or 'UV' with YUV"):
        _run(noise_yuv32, channels="RGB")


def test_channels_invalid_on_rgb(noise_rgb32):
    with pytest.raises(vs.Error, match=r"'channels' must be 'RGB' with RGB"):
        _run(noise_rgb32, channels="Y")


def test_reject_search_window_larger_than_frame():
    core = vs.core
    src = core.std.BlankClip(None, 8, 8, vs.GRAYS, length=1, color=[0.5])
    with pytest.raises(vs.Error, match=r"research window \(2\*a\+1\) larger than the frame"):
        _run(src, a=10)


def test_reject_rclip_dimension_mismatch(noise_gray):
    bad = noise_gray.std.Crop(left=16)
    with pytest.raises(vs.Error, match=r"'rclip' must match the source clip"):
        _run(noise_gray, d=0, rclip=bad)


def test_reject_rclip_format_mismatch(noise_gray, noise_16bit):
    with pytest.raises(vs.Error, match=r"'rclip' must match the source clip"):
        _run(noise_gray, d=0, rclip=noise_16bit)


def test_reject_window_larger_than_slot_pool():
    """A frame at n >= d must hold one full window (clips*C*(2d+1) slots) at
    once. When the 512 MiB pool cap cannot cover that, creation must reject
    the configuration: the all-or-nothing acquire would otherwise wait on
    cache_cv forever (it holds no slots and never submits, so nothing can
    notify it). 1080p f32 YUV444 at d=16 is the minimal case: pool 70,
    needs 99."""
    core = vs.core
    src = core.std.BlankClip(None, 1920, 1080, vs.YUV444PS, length=40,
                             color=[0.5, 0.5, 0.5])
    with pytest.raises(vs.Error,
                       match=r"needs 99 cache slots .* budget allows 70"):
        _run(src, channels="YUV", d=16)


def test_accept_window_that_fits_the_slot_pool():
    """Control for the case above: the same geometry in luma at d=16 needs
    only 33 slots and must still create and evaluate frame n = d."""
    core = vs.core
    src = core.std.BlankClip(None, 1920, 1080, vs.GRAYS, length=40,
                             color=[0.5])
    out = _run(src, d=16)
    assert out.get_frame(16) is not None
