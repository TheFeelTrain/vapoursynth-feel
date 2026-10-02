"""Unit tests for core.vsfeel.NLMeans.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is used as the input. The noise content makes every parameter axis observable: NLM must
actually alter it, and any misindexing of the padded window or sweep tables
shows up as a large diff against the reference implementation.

Run from the repository root:  uv run python -m pytest tests/test_nlmeans.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_changes_on_clip,
    assert_gray32,
    assert_preserves_frame_props,
    assert_temporal_order_consistent,
    cpu_node,
    eval_parallel,
    max_diff,
    plane as _plane,
    reference_compare,
    reference_or_skip,
    reference_spec,
)

pytestmark = pytest.mark.usefixtures("clip_gray")

H_PARAM = 1.2


def _run(clip, **kwargs):
    """NLMeans as a clip the test can read pixels from.

    Under the R80 GPU API the filter takes and returns ``vnode:gpu`` frames, so
    a CPU clip is auto-uploaded by the core and the output is downloaded before
    the host reads it (see conftest.cpu_node).
    """
    return cpu_node(vs.core.vsfeel.NLMeans(clip, **kwargs))


def _ref_compare(fmt, frames, params, planes=None, guide=None, crop=None):
    """Worst diff vs vszipcl over ``frames`` (subprocess; skips if absent).

    ``fmt`` is a format name understood by ``conftest.REFERENCE_SCRIPT``;
    ``guide`` describes the rclip when the case uses one.
    """
    reference_or_skip("vszipcl", "NLMeans")
    spec = reference_spec(
        "vszipcl",
        "NLMeans",
        fmt,
        frames=frames,
        planes=planes,
        kwargs=params,
        guide=guide,
        guide_kwarg="rclip",
        crop=crop,
    )
    return reference_compare(spec)["maxdiff"]


# --- basic behaviour ---------------------------------------------------------


def test_output_format_and_frames_preserved(clip_gray):
    out = _run(clip_gray, d=0)
    assert out.format.id == clip_gray.format.id
    assert (out.width, out.height) == (clip_gray.width, clip_gray.height)
    assert out.num_frames == clip_gray.num_frames


def test_nlmeans_preserves_frame_props(clip_gray):
    """NLMeans must republish the source frame's properties."""
    assert_preserves_frame_props(_run, clip_gray, d=0)


def test_denoise_changes_output_32bit(clip_gray):
    src = clip_gray
    out = _run(src, d=0)
    assert np.abs(_plane(out.get_frame(5), 0) - _plane(src.get_frame(5), 0)).max() > 0.0


def test_higher_h_smooths_more_32bit(clip_gray):
    src = clip_gray
    weak = _run(src, d=0, h=0.3)
    strong = _run(src, d=0, h=4.0)
    dw = np.abs(_plane(weak.get_frame(5), 0) - _plane(src.get_frame(5), 0)).max()
    ds = np.abs(_plane(strong.get_frame(5), 0) - _plane(src.get_frame(5), 0)).max()
    assert ds > dw


def test_search_radius_changes_output_32bit(clip_gray):
    a1 = _run(clip_gray, d=0, a=1)
    a4 = _run(clip_gray, d=0, a=4)
    assert max_diff(a1, a4, frames=(5,)) > 0.0


def test_patch_size_changes_output_32bit(clip_gray):
    s1 = _run(clip_gray, d=0, s=1)
    s3 = _run(clip_gray, d=0, s=3)
    assert max_diff(s1, s3, frames=(5,)) > 0.0


def test_wmode_changes_output_32bit(clip_gray):
    w0 = _run(clip_gray, d=0, wmode=0)
    w3 = _run(clip_gray, d=0, wmode=3)
    assert max_diff(w0, w3, frames=(5,)) > 0.0


def test_wref_changes_output_32bit(clip_gray):
    w1 = _run(clip_gray, d=0, wref=1.0)
    w0 = _run(clip_gray, d=0, wref=0.0)
    assert max_diff(w1, w0, frames=(5,)) > 0.0


def test_temporal_differs_from_spatial_32bit(clip_gray):
    # the noise clip is independent per frame, so a temporal window must
    # produce a different result than spatial-only
    spatial = _run(clip_gray, d=0)
    temporal = _run(clip_gray, d=2)
    assert max_diff(spatial, temporal) > 0.0


# --- correctness vs the reference --------------------------------------------

# Parameter sweep vs vszipcl. Tolerance set from measurement: diffs come from
# fp32 accumulation order in the weighted average (more taps at larger a/s
# accumulate more), landing at <= 4.4e-6 across all configs and all special
# paths (UV/RGB joint, rclip guide, cropped stride, a=64, d=16); the rest of
# the surface is <= 1e-6. The bound keeps 2.3x headroom over the worst
# measurement while still catching real misindexing (O(0.1..1) errors) and a
# return of the reduced-precision weight ring (which measured 2.1e-3 at a=64).
NLMEANS_REF_TOL = 1e-5

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
    # No wref=0 entry: the reference's own weighted average is degenerate there
    # and it emits NaN (1533 samples on frame 0 at h=3.0, 53983 at h=1.2; nlm_hip
    # agrees), so there is no oracle to compare against. vsfeel's finiteness and
    # its centre-sample fallback are pinned by the wref0 tests below.
]


@pytest.mark.parametrize("kwargs", REFERENCE_CASES, ids=lambda kw: str(kw))
def test_matches_reference_32bit(clip_gray, kwargs):
    worst = _ref_compare("gray32", (0, 11, 23), dict(**kwargs))
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl {kwargs}: {worst}"


GRAY16_CASES = [
    {"d": 1},
    {"d": 2},
    {"d": 0, "wmode": 2, "h": 2.0},
    {"d": 0, "wmode": 3},
    {"d": 1, "a": 4},
]


@pytest.mark.parametrize("kwargs", GRAY16_CASES, ids=lambda kw: str(kw))
def test_matches_reference_16bit(clip_16bit, kwargs):
    """Integer rounding path: both sides round nearly identical fp32 results
    once, so codes differ by at most one step."""
    worst = _ref_compare("gray16", (0, 11, 23), dict(**kwargs))
    assert worst <= 1.0, f"max LSB diff vs vszipcl {kwargs}: {worst}"


# Second, independent implementation (KNLMeansCL's Vulkan port) on the spatial
# surface, where it agrees with vsfeel bit-for-bit at f32 and within 1 LSB at
# u16: a cross-check that the vszipcl agreement is not a shared quirk. Only d=0
# configs - knlmvk's temporal pairing differs (measured 3.7e-2 at d=2, 1.4e-1
# for d=1 a=3 s=3 h=3.0 wref=0.4) and it is that far from vszipcl too, so it is
# not an oracle for the temporal path or for wref=0 (where it is 1.4e-1 from
# vsfeel while vszipcl evaluates 0/0).
KNLMVK_CASES = [
    {"d": 0},
    {"d": 0, "a": 4},
    {"d": 0, "s": 8},
    {"d": 0, "wmode": 2, "h": 2.0},
    {"d": 0, "wmode": 3},
    {"d": 0, "h": 0.3},
]


def _knlmvk_compare(fmt, kwargs, frames=(0, 11, 23)):
    """Worst diff vs knlmvk (subprocess; skips if it is not installed)."""
    reference_or_skip("knlmvk", "KNLMeans")
    spec = reference_spec(
        "knlmvk", "KNLMeans", fmt, frames=frames, kwargs=kwargs, vsfeel_filter="NLMeans"
    )
    return reference_compare(spec)["maxdiff"]


@pytest.mark.parametrize("kwargs", KNLMVK_CASES, ids=lambda kw: str(kw))
def test_spatial_matches_knlmvk_32bit(clip_gray, kwargs):
    worst = _knlmvk_compare("gray32", dict(**kwargs))
    assert worst <= 1e-6, f"max diff vs knlmvk {kwargs}: {worst}"


@pytest.mark.parametrize("kwargs", KNLMVK_CASES, ids=lambda kw: str(kw))
def test_spatial_matches_knlmvk_16bit(clip_16bit, kwargs):
    worst = _knlmvk_compare("gray16", dict(**kwargs))
    assert worst <= 1.0, f"max LSB diff vs knlmvk {kwargs}: {worst}"


# Positive maxima of the three radii (a=64, s=8, d=16): the m=0..2
# sweep-table variants and run-group boundaries the a<=4 / d<=2 sweep misses.
# Every entry is at the shared NLMEANS_REF_TOL now: the fp32 weight ring holds
# the whole surface at <= 4.4e-6 (measured). d=16 with s=8 and a=64 together is
# omitted: it hard-recovers the GPU.
POSITIVE_MAX_CASES = [
    # (kwargs, bound, measured)
    ({"d": 0, "a": 64}, NLMEANS_REF_TOL, 7.2e-7),
    ({"d": 0, "s": 8, "a": 64}, NLMEANS_REF_TOL, 4.2e-7),
    ({"d": 0, "a": 64, "h": 3.0}, NLMEANS_REF_TOL, 4.35e-6),
    ({"d": 0, "s": 8, "a": 64, "h": 3.0}, NLMEANS_REF_TOL, 4.10e-6),
    ({"d": 16}, NLMEANS_REF_TOL, 1.55e-6),
    ({"d": 16, "s": 8}, NLMEANS_REF_TOL, 1.43e-6),
    # d=16 together with a=64 is omitted: it hard-recovers the GPU.
]


@pytest.mark.parametrize(
    "kwargs,bound,measured", POSITIVE_MAX_CASES, ids=[str(kw) for kw, _, _ in POSITIVE_MAX_CASES]
)
def test_matches_reference_positive_maxima_32bit(clip_gray, kwargs, bound, measured):
    worst = _ref_compare("gray32", (0, 11, 23), dict(**kwargs))
    assert worst < bound, f"max diff vs vszipcl {kwargs}: {worst} (bound {bound}, was {measured})"


POSITIVE_MAX_16_CASES = [
    # (kwargs, bound, measured codes)
    ({"d": 16}, 1.0, 1),
    ({"d": 16, "s": 8}, 1.0, 1),
    # d=16 with a=64 omitted for the same reason as the 32-bit list.
    ({"d": 0, "a": 64}, 1.0, 1),
    ({"d": 0, "s": 8, "a": 64}, 1.0, 1),
    ({"d": 0, "a": 64, "h": 3.0}, 1.0, 1),
    ({"d": 0, "s": 8, "a": 64, "h": 3.0}, 1.0, 1),
]


@pytest.mark.parametrize(
    "kwargs,bound,measured",
    POSITIVE_MAX_16_CASES,
    ids=[str(kw) for kw, _, _ in POSITIVE_MAX_16_CASES],
)
def test_matches_reference_positive_maxima_16bit(clip_16bit, kwargs, bound, measured):
    """16-bit mirror: whole output codes, every entry at the 1 LSB floor."""
    worst = _ref_compare("gray16", (0, 11, 23), dict(**kwargs))
    assert worst <= bound, (
        f"max LSB diff vs vszipcl {kwargs}: {worst} (bound {bound}, was {measured})"
    )


def test_yuv_default_denises_luma_copies_chroma_32bit(clip_yuv32):
    src = clip_yuv32
    out = _run(src, d=0)
    assert max_diff(out, src, planes=(0,), frames=(5,)) > 0.0
    for p in (1, 2):
        fa = _plane(out.get_frame(5), p)
        fb = _plane(src.get_frame(5), p)
        assert np.array_equal(fa, fb), f"chroma{p} changed"


def test_yuv_default_denises_luma_copies_chroma_16bit(clip_yuv420_16):
    """16-bit mirror of test_yuv_default_denises_luma_copies_chroma."""
    src = clip_yuv420_16
    out = _run(src, d=0)
    assert max_diff(out, src, planes=(0,), frames=(5,)) > 0.0
    for p in (1, 2):
        fa = _plane(out.get_frame(5), p)
        fb = _plane(src.get_frame(5), p)
        assert np.array_equal(fa, fb), f"chroma{p} changed"
    worst = _ref_compare("yuv420_16", (5,), dict(d=0), planes=(0,))
    assert worst <= 1.0


def test_yuv_channels_uv_matches_reference_32bit(clip_yuv32):
    """channels='UV' on a subsampled YUV clip: chroma denoised (subsampled
    lattice), luma passed through bit-exactly."""
    src = clip_yuv32
    assert src.format.subsampling_w == 1 and src.format.subsampling_h == 1

    out = _run(src, d=0, channels="UV", h=1.5)
    assert max_diff(out, src, planes=(0,), frames=(5,)) == 0.0
    assert max_diff(out, src, planes=(1, 2), frames=(5,)) > 0.0

    worst = _ref_compare("yuv32", (0, 11, 23), dict(d=0, channels="UV", h=1.5), planes=(1, 2))
    assert worst < NLMEANS_REF_TOL


# --- UV (chroma-only, 2-channel) sweep at both depths vs the reference ------
#
# The luma sweeps above only exercise 1 channel; the 16-bit 'UV' path was once
# broken while every other depth/channel-count combination stayed correct, so
# the 2-channel sweep must run at BOTH depths. Parameter list mirrors
# REFERENCE_CASES. Tolerances measured on this clip with the fp32 weight ring:
#   - 32-bit: worst 2.4e-7 across the sweep (fp32 accumulation order), bound
#     NLMEANS_REF_TOL as for the luma sweep.
#   - 16-bit: worst 1 code (integer rounding + fp32 order on the subsampled
#     lattice), i.e. the whole list sits at the 1 LSB floor.

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
    # No wref=0 entry, same reason as the luma sweep above (the reference emits
    # non-finite pixels there in both the f32 and the saturated-65535 u16 path).
]

UV16_REF_TOL = 1.0


@pytest.mark.parametrize("kwargs", UV32_CASES, ids=lambda kw: str(kw))
def test_uv_matches_reference_32bit(clip_yuv32, kwargs):
    worst = _ref_compare("yuv32", (0, 11, 23), dict(channels="UV", **kwargs), planes=(1, 2))
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl {kwargs}: {worst}"


@pytest.mark.parametrize("kwargs", UV16_CASES, ids=lambda kw: str(kw))
def test_uv_matches_reference_16bit(clip_yuv420_16, kwargs):
    worst = _ref_compare("yuv420_16", (0, 11, 23), dict(channels="UV", **kwargs), planes=(1, 2))
    assert worst <= UV16_REF_TOL, f"max LSB diff vs vszipcl {kwargs}: {worst}"


ENVELOPE_FRAMES = (0, 11, 23)


def test_wref0_low_h_is_finite_32bit(clip_gray):
    """Nothing in the wref=0 envelope may produce 0/0.

    Covers both the exp() ring (wmode 0) and the truncated modes (wmode 1-3,
    whose weights are all exactly 0 for arg >= 1).
    """
    for h in (0.6, 1.0, 1.2):
        for wmode in (0, 1, 2, 3):
            out = _run(clip_gray, d=0, h=h, wmode=wmode, wref=0.0)
            for n in ENVELOPE_FRAMES:
                got = _plane(out.get_frame(n), 0)
                assert np.isfinite(got).all(), f"h={h} wmode={wmode} n={n}"


def test_wref0_low_h_is_finite_uv_32bit(clip_yuv32):
    """Same guard on the 2-channel path."""
    out = _run(clip_yuv32, channels="UV", d=0, wref=0.0)
    for n in ENVELOPE_FRAMES:
        for p in (1, 2):
            assert np.isfinite(_plane(out.get_frame(n), p)).all(), f"n={n} p={p}"


def test_wref0_flushed_ring_returns_centre_sample(clip_gray, clip_16bit):
    """A fully flushed weight ring must return the centre sample exactly.

    wref=0 with a small h drives every tap's weight below the ring's range
    (wmode 0, measured: bit-exact from h <= 0.1) or truncates every weight to
    zero (wmode 1-3), so the total weight is 0 and the finish must fall back to
    the centre sample instead of evaluating 0/0. The reference cannot bound
    this: vszipcl and nlm_hip both emit NaN in the same regime (1533 samples on
    frame 0 at h=3.0, 53983 at h=1.2), which is why the oracle here is the
    source clip.
    """
    for clip, dt in ((clip_gray, np.float32), (clip_16bit, np.uint16)):
        for h in (1e-4, 1e-2, 0.1):
            for wmode in (0, 1, 2, 3):
                out = _run(clip, d=0, h=h, wmode=wmode, wref=0.0)
                for n in ENVELOPE_FRAMES:
                    got = _plane(out.get_frame(n), 0, dtype=dt)
                    src = _plane(clip.get_frame(n), 0, dtype=dt)
                    assert np.array_equal(got, src), f"h={h} wmode={wmode} n={n}"


def test_yuv_channels_uv_temporal_matches_reference_32bit(clip_yuv32):
    worst = _ref_compare("yuv32", (0, 11, 23), dict(d=1, channels="UV", h=1.5), planes=(1, 2))
    assert worst < NLMEANS_REF_TOL


def test_yuv444_joint_matches_reference_16bit(clip_yuv444_16):
    # joint processing sums distances across three planes before rounding,
    # so the fp divergence reaches two output codes (measured); single-plane
    # paths stay within one
    worst = _ref_compare("yuv444_16", (0, 11, 23), dict(d=0, channels="YUV", h=1.0))
    assert worst <= 2.0


def test_rgb_joint_matches_reference_32bit(clip_rgb32):
    worst = _ref_compare("rgb32", (0, 11, 23), dict(d=1, h=1.0))
    assert worst < NLMEANS_REF_TOL


def test_rgb_joint_matches_reference_16bit(clip_rgb16):
    """16-bit mirror of test_rgb_joint_matches_reference (whole codes)."""
    worst = _ref_compare("rgb16", (0, 11, 23), dict(d=1, h=1.0))
    assert worst <= 1.0


def test_rclip_self_is_identity_32bit(clip_gray):
    src = clip_gray
    plain = _run(src, d=0, h=1.5)
    withref = _run(src, d=0, h=1.5, rclip=src)
    assert max_diff(plain, withref, frames=(5,)) == 0.0


def test_rclip_self_is_identity_16bit(clip_16bit):
    """16-bit mirror of test_rclip_self_is_identity."""
    src = clip_16bit
    plain = _run(src, d=0, h=1.5)
    withref = _run(src, d=0, h=1.5, rclip=src)
    assert max_diff(plain, withref, frames=(5,)) == 0.0


def test_rclip_guide_matches_reference_32bit(clip_gray):
    src = clip_gray
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    mine = _run(src, d=0, h=1.5, rclip=guide)
    plain = _run(src, d=0, h=1.5)
    assert max_diff(mine, plain, frames=(5,)) > 0.0
    worst = _ref_compare(
        "gray32",
        (0, 11, 23),
        dict(d=0, h=1.5),
        guide={"kind": "boxblur", "hradius": 5, "vradius": 5},
    )
    assert worst < NLMEANS_REF_TOL


def test_rclip_guide_matches_reference_16bit(clip_16bit):
    """16-bit mirror of test_rclip_guide_matches_reference (whole codes)."""
    src = clip_16bit
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    mine = _run(src, d=0, h=1.5, rclip=guide)
    plain = _run(src, d=0, h=1.5)
    assert max_diff(mine, plain, frames=(5,)) > 0.0
    worst = _ref_compare(
        "gray16",
        (0, 11, 23),
        dict(d=0, h=1.5),
        guide={"kind": "boxblur", "hradius": 5, "vradius": 5},
    )
    assert worst <= 1.0


@pytest.mark.parametrize("d", [1, 2])
def test_rclip_temporal_matches_reference_early_frames(clip_gray, d):
    """Frames n < d use a shorter window (2*min(d,n)+1 layers), where the
    guide clip's slot table used to be indexed with the full-window stride
    and read past win_slots. Pre-fix diff vs vszipcl was 2.4e-2 (d=1) and
    6.7e-3 (d=2); frames n >= d matched to ~1e-6."""
    worst = _ref_compare(
        "gray32",
        (0, 1, 2, 3),
        dict(d=d, h=1.5),
        guide={"kind": "boxblur", "hradius": 5, "vradius": 5},
    )
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl (d={d}): {worst}"


def test_stride_handling_matches_reference_32bit(clip_gray):
    # a cropped frame keeps its parent's (wider) stride; the filter must
    # handle non-tight pitches identically to the reference
    assert clip_gray.std.Crop(left=27).width < clip_gray.width
    worst = _ref_compare("gray32", (0, 11, 23), dict(d=1), crop={"left": 27})
    assert worst < NLMEANS_REF_TOL


def test_stride_handling_matches_reference_16bit(clip_16bit):
    """16-bit mirror of test_stride_handling_matches_reference (whole
    codes)."""
    assert clip_16bit.std.Crop(left=27).width < clip_16bit.width
    worst = _ref_compare("gray16", (0, 11, 23), dict(d=1), crop={"left": 27})
    assert worst <= 1.0


# --- determinism / streams ---------------------------------------------------


def test_deterministic_serial_32bit(clip_gray):
    a = _run(clip_gray, d=2, h=1.5)
    b = _run(clip_gray, d=2, h=1.5)
    assert max_diff(a, b) == 0.0


def test_deterministic_serial_16bit(clip_16bit):
    a = _run(clip_16bit, d=2, h=1.5)
    b = _run(clip_16bit, d=2, h=1.5)
    assert max_diff(a, b) == 0.0


def test_parallel_load_matches_serial_32bit(clip_gray):
    par = eval_parallel(_run, clip_gray, d=2)
    ref = _run(clip_gray, d=2)
    for n in range(clip_gray.num_frames):
        fb = _plane(ref.get_frame(n), 0)
        assert np.abs(par[n].astype(np.float64) - fb).max() == 0.0, (
            f"parallel/serial mismatch at frame {n}"
        )


def test_parallel_load_matches_serial_16bit(clip_16bit):
    par = eval_parallel(_run, clip_16bit, d=2)
    ref = _run(clip_16bit, d=2)
    for n in range(clip_16bit.num_frames):
        fb = _plane(ref.get_frame(n), 0)
        assert np.abs(par[n].astype(np.float64) - fb).max() == 0.0, (
            f"parallel/serial mismatch at frame {n}"
        )


def test_parallel_load_deterministic_32bit(clip_gray):
    a = eval_parallel(_run, clip_gray, d=2)
    b = eval_parallel(_run, clip_gray, d=2)
    for n in range(clip_gray.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


def test_parallel_load_deterministic_16bit(clip_16bit):
    a = eval_parallel(_run, clip_16bit, d=2)
    b = eval_parallel(_run, clip_16bit, d=2)
    for n in range(clip_16bit.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


# --- frame-request order / short clips ---------------------------------------


@pytest.mark.parametrize("d", [2, 16])
def test_frame_request_order_matches_serial(d):
    """The tile cache must return the same pixels whatever order frames arrive.

    A cached slot keyed and released by frame index instead of a per-request
    token can be freed while another request still reads it; only an
    out-of-sequence request pattern makes that observable.
    """
    assert_temporal_order_consistent("NLMeans", {"d": d}, tol=0.0, timeout=300)


@pytest.mark.parametrize("nframes", [1, 2])
def test_short_clip_temporal_window(nframes):
    """A clip shorter than the temporal window is order-independent too."""
    assert_temporal_order_consistent("NLMeans", {"d": 2}, tol=0.0, nframes=nframes, timeout=300)


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_no_nan_all_frames_32bit(clip_gray, radius):
    out = _run(clip_gray, d=radius)
    assert_gray32(out)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"
        assert (a >= 0.0).all() and (a <= 1.0).all(), f"out-of-range output at frame {n}"


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_no_nan_all_frames_16bit(clip_16bit, radius):
    """Finiteness on the integer path is trivially true (uint16), so the
    meaningful half is the anti-vacuity check: the filter must alter the
    noise rather than pass it through."""
    out = _run(clip_16bit, d=radius)
    assert out.format.id == clip_16bit.format.id
    assert_changes_on_clip(out, clip_16bit, what="NLMeans")


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
        (dict(h=0), r"h must be finite and > 0"),
        (dict(h=-1.0), r"h must be finite and > 0"),
        (dict(h=float("inf")), r"h must be finite and > 0"),
        (dict(h=float("nan")), r"h must be finite and > 0"),
        (dict(wmode=-1), r"wmode must be 0\.\.3"),
        (dict(wmode=4), r"wmode must be 0\.\.3"),
        (dict(wref=-0.5), r"wref must be >= 0"),
        (dict(channels="bogus"), r"'channels' must be 'Y' with Gray"),
        (dict(channels="UV"), r"'channels' must be 'Y' with Gray"),
    ],
)
def test_validation_errors_gray(clip_gray, args, msg):
    with pytest.raises(vs.Error, match=msg):
        _run(clip_gray, **args)


def test_rejects_8bit(clip_8bit):
    with pytest.raises(vs.Error, match=r"input bitdepth must be"):
        _run(clip_8bit)


def test_channels_yuv_requires_444(clip_yuv32):
    with pytest.raises(vs.Error, match=r"'channels'='YUV' requires 4:4:4"):
        _run(clip_yuv32, channels="YUV")


def test_channels_invalid_on_yuv(clip_yuv32):
    with pytest.raises(vs.Error, match=r"'channels' must be 'YUV', 'Y' or 'UV' with YUV"):
        _run(clip_yuv32, channels="RGB")


def test_channels_invalid_on_rgb(clip_rgb32):
    with pytest.raises(vs.Error, match=r"'channels' must be 'RGB' with RGB"):
        _run(clip_rgb32, channels="Y")


def test_reject_search_window_larger_than_frame():
    core = vs.core
    src = core.std.BlankClip(None, 8, 8, vs.GRAYS, length=1, color=[0.5])
    with pytest.raises(vs.Error, match=r"research window \(2\*a\+1\) larger than the frame"):
        _run(src, a=10)


def test_reject_rclip_dimension_mismatch(clip_gray):
    bad = clip_gray.std.Crop(left=16)
    with pytest.raises(vs.Error, match=r"'rclip' must match the source clip"):
        _run(clip_gray, d=0, rclip=bad)


def test_reject_rclip_format_mismatch(clip_gray, clip_16bit):
    with pytest.raises(vs.Error, match=r"'rclip' must match the source clip"):
        _run(clip_gray, d=0, rclip=clip_16bit)


def test_window_larger_than_old_slot_pool_runs():
    """A window larger than the removed 512 MiB slot pool must now run.

    The old design kept its own padded slot pool and rejected any config whose
    full window did not fit 512 MiB (1080p f32 YUV444 d=16 was the minimal
    case). Under the R80 GPU API the temporal frames live in the core's GPU
    frame cache and the per-frame scratch is transient, so the same *shape*
    must create and evaluate. The geometry is shrunk to keep the test cheap;
    the noise-clip d=16 case above already covers correctness.
    """
    core = vs.core
    src = core.std.BlankClip(None, 320, 240, vs.YUV444PS, length=20, color=[0.5, 0.5, 0.5])
    out = _run(src, channels="YUV", d=16)
    assert out.get_frame(16) is not None


def test_cpu_and_gpu_input_match(clip_gray):
    """The core's auto-upload of a CPU clip must match an explicit GPUUpload.

    This is the filter-level mirror of the benchmark's ``--gpu-cache`` arm: the
    same node fed a CPU clip (auto GPUUpload) and a resident GPU clip has to
    produce identical pixels.
    """
    cpu = _run(clip_gray, d=2)
    gpu = cpu_node(vs.core.vsfeel.NLMeans(vs.core.std.GPUUpload(clip=clip_gray), d=2))
    assert max_diff(cpu, gpu) == 0.0
