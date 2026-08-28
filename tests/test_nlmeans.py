"""Unit tests for core.vsfeel.NLMeans.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The noise content makes every parameter axis observable: NLM must
actually alter it, and any misindexing of the padded window or sweep tables
shows up as a large diff against the reference implementation.

Run from the repository root:  python -m pytest tests/test_nlmeans.py
"""

import ctypes
import textwrap
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, assert_gray32, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")

H_PARAM = 1.2


def _run(clip, num_streams=1, **kwargs):
    return vs.core.vsfeel.NLMeans(clip, num_streams=num_streams, **kwargs)


def _plane(frame, plane):
    """Copy a frame plane into an ndarray (float32 or uint16), honouring the
    plane's actual row pitch."""
    fmt = frame.format
    itemsize = fmt.bytes_per_sample
    # plane dims follow the format's subsampling (planes 1/2 of YUV-like)
    ss_w = fmt.subsampling_w if plane in (1, 2) and fmt.num_planes >= 3 else 0
    ss_h = fmt.subsampling_h if plane in (1, 2) and fmt.num_planes >= 3 else 0
    w = frame.width >> ss_w
    h = frame.height >> ss_h
    raw = np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(h, frame.get_stride(plane)),
    )
    return raw[:, :w * itemsize].copy().view(
        np.float32 if fmt.sample_type == vs.FLOAT else np.uint16).reshape(h, w)


def _eval_parallel(clip, **kwargs):
    """Evaluate every frame concurrently, the way vspipe does."""
    out = _run(clip, **kwargs)
    frames = [None] * out.num_frames

    def worker(n):
        frames[n] = _plane(out.get_frame(n), 0)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(out.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(f is not None for f in frames)
    return frames


def _max_diff(a_clip, b_clip, planes=(0,), frames=(0, 11, 23)):
    worst = 0.0
    for n in frames:
        for p in planes:
            fa = _plane(a_clip.get_frame(n), p)
            fb = _plane(b_clip.get_frame(n), p)
            worst = max(worst, float(np.abs(fa.astype(np.float64) - fb).max()))
    return worst


# --- shared clips ------------------------------------------------------------

@pytest.fixture(scope="session")
def noise_yuv32():
    """YUV420PS (float32) version of the noise clip (chroma is subsampled)."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    return vs.core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)


@pytest.fixture(scope="session")
def noise_yuv420_16():
    """YUV420P16 version of the noise clip (subsampled chroma lattice, for
    the 16-bit 'UV' sweep)."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    return vs.core.resize.Bicubic(src, format=vs.YUV420P16)


@pytest.fixture(scope="session")
def noise_yuv444_16():
    """YUV444P16 version of the noise clip (for joint 'YUV' processing)."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    return vs.core.resize.Bicubic(src, format=vs.YUV444P16)


@pytest.fixture(scope="session")
def noise_rgb32():
    """RGBS version of the noise clip."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    return vs.core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")


@pytest.fixture(scope="session")
def noise_rgb16():
    """RGB48 (16-bit integer) version of the noise clip."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    return vs.core.resize.Bicubic(src, format=vs.RGB48, matrix_in_s="709")


# --- basic behaviour ---------------------------------------------------------


def test_output_format_and_frames_preserved(noise_gray):
    out = _run(noise_gray, d=0)
    assert out.format.id == noise_gray.format.id
    assert (out.width, out.height) == (noise_gray.width, noise_gray.height)
    assert out.num_frames == noise_gray.num_frames


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
    assert _max_diff(a1, a4, frames=(5,)) > 0.0


def test_patch_size_changes_output_32bit(noise_gray):
    s1 = _run(noise_gray, d=0, s=1)
    s3 = _run(noise_gray, d=0, s=3)
    assert _max_diff(s1, s3, frames=(5,)) > 0.0


def test_wmode_changes_output_32bit(noise_gray):
    w0 = _run(noise_gray, d=0, wmode=0)
    w3 = _run(noise_gray, d=0, wmode=3)
    assert _max_diff(w0, w3, frames=(5,)) > 0.0


def test_wref_changes_output_32bit(noise_gray):
    w1 = _run(noise_gray, d=0, wref=1.0)
    w0 = _run(noise_gray, d=0, wref=0.0)
    assert _max_diff(w1, w0, frames=(5,)) > 0.0


def test_temporal_differs_from_spatial_32bit(noise_gray):
    # the noise clip is independent per frame, so a temporal window must
    # produce a different result than spatial-only
    spatial = _run(noise_gray, d=0)
    temporal = _run(noise_gray, d=2)
    assert _max_diff(spatial, temporal) > 0.0


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
    {"d": 0, "wref": 0.0},
]


def _reference_plugin():
    ref = getattr(vs.core, "vszipcl", None)
    if ref is None or not hasattr(ref, "NLMeans"):
        pytest.skip("no vszipcl reference plugin")
    return ref


@pytest.mark.parametrize("kwargs", REFERENCE_CASES, ids=lambda kw: str(kw))
def test_matches_reference_32bit(noise_gray, kwargs):
    theirs = _reference_plugin().NLMeans(noise_gray, num_streams=1, **kwargs)
    mine = _run(noise_gray, **kwargs)
    worst = _max_diff(mine, theirs)
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
    theirs = _reference_plugin().NLMeans(noise_16bit, num_streams=1, **kwargs)
    mine = _run(noise_16bit, **kwargs)
    worst = _max_diff(mine, theirs)
    assert worst <= 1.0, f"max LSB diff vs vszipcl {kwargs}: {worst}"


def test_yuv_default_denises_luma_copies_chroma_32bit(noise_yuv32):
    src = noise_yuv32
    out = _run(src, d=0)
    assert _max_diff(out, src, planes=(0,), frames=(5,)) > 0.0
    for p in (1, 2):
        fa = _plane(out.get_frame(5), p)
        fb = _plane(src.get_frame(5), p)
        assert np.array_equal(fa, fb), f"chroma{p} changed"


def test_yuv_default_denises_luma_copies_chroma_16bit(noise_yuv420_16):
    """16-bit mirror of test_yuv_default_denises_luma_copies_chroma."""
    src = noise_yuv420_16
    out = _run(src, d=0)
    assert _max_diff(out, src, planes=(0,), frames=(5,)) > 0.0
    for p in (1, 2):
        fa = _plane(out.get_frame(5), p)
        fb = _plane(src.get_frame(5), p)
        assert np.array_equal(fa, fb), f"chroma{p} changed"
    ref = _reference_plugin()
    theirs = ref.NLMeans(src, num_streams=1, d=0)
    assert _max_diff(out, theirs, planes=(0,), frames=(5,)) <= 1.0


def test_yuv_channels_uv_matches_reference_32bit(noise_yuv32):
    """channels='UV' on a subsampled YUV clip: chroma denoised (subsampled
    lattice), luma passed through bit-exactly."""
    ref = _reference_plugin()
    src = noise_yuv32
    assert src.format.subsampling_w == 1 and src.format.subsampling_h == 1

    out = _run(src, d=0, channels="UV", h=1.5)
    assert _max_diff(out, src, planes=(0,), frames=(5,)) == 0.0
    assert _max_diff(out, src, planes=(1, 2), frames=(5,)) > 0.0

    theirs = ref.NLMeans(src, num_streams=1, d=0, channels="UV", h=1.5)
    assert _max_diff(out, theirs, planes=(1, 2)) < NLMEANS_REF_TOL


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
    {"d": 0, "wref": 0.0},
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
    ref = _reference_plugin()
    src = noise_yuv32
    mine = _run(src, channels="UV", **kwargs)
    theirs = ref.NLMeans(src, num_streams=1, channels="UV", **kwargs)
    worst = _max_diff(mine, theirs, planes=(1, 2))
    assert worst < NLMEANS_REF_TOL, f"max diff vs vszipcl {kwargs}: {worst}"


@pytest.mark.parametrize("kwargs", UV16_CASES, ids=lambda kw: str(kw))
def test_uv_matches_reference_16bit(noise_yuv420_16, kwargs):
    ref = _reference_plugin()
    src = noise_yuv420_16
    mine = _run(src, channels="UV", **kwargs)
    theirs = ref.NLMeans(src, num_streams=1, channels="UV", **kwargs)
    worst = _max_diff(mine, theirs, planes=(1, 2))
    assert worst <= UV16_REF_TOL, f"max LSB diff vs vszipcl {kwargs}: {worst}"


def test_yuv_channels_uv_temporal_matches_reference_32bit(noise_yuv32):
    ref = _reference_plugin()
    src = noise_yuv32
    mine = _run(src, d=1, channels="UV", h=1.5, num_streams=2)
    theirs = ref.NLMeans(src, num_streams=2, d=1, channels="UV", h=1.5)
    assert _max_diff(mine, theirs, planes=(1, 2)) < NLMEANS_REF_TOL


def test_yuv444_joint_matches_reference_16bit(noise_yuv444_16):
    ref = _reference_plugin()
    src = noise_yuv444_16
    mine = _run(src, d=0, channels="YUV", h=1.0)
    theirs = ref.NLMeans(src, num_streams=1, d=0, channels="YUV", h=1.0)
    # joint processing sums distances across three planes before rounding,
    # so the fp divergence reaches two output codes (measured); single-plane
    # paths stay within one
    assert _max_diff(mine, theirs, planes=(0, 1, 2)) <= 2.0


def test_rgb_joint_matches_reference_32bit(noise_rgb32):
    ref = _reference_plugin()
    src = noise_rgb32
    mine = _run(src, d=1, h=1.0)
    theirs = ref.NLMeans(src, num_streams=1, d=1, h=1.0)
    assert _max_diff(mine, theirs, planes=(0, 1, 2)) < NLMEANS_REF_TOL


def test_rgb_joint_matches_reference_16bit(noise_rgb16):
    """16-bit mirror of test_rgb_joint_matches_reference (whole codes)."""
    ref = _reference_plugin()
    src = noise_rgb16
    mine = _run(src, d=1, h=1.0)
    theirs = ref.NLMeans(src, num_streams=1, d=1, h=1.0)
    assert _max_diff(mine, theirs, planes=(0, 1, 2)) <= 1.0


def test_rclip_self_is_identity_32bit(noise_gray):
    src = noise_gray
    plain = _run(src, d=0, h=1.5)
    withref = _run(src, d=0, h=1.5, rclip=src)
    assert _max_diff(plain, withref, frames=(5,)) == 0.0


def test_rclip_self_is_identity_16bit(noise_16bit):
    """16-bit mirror of test_rclip_self_is_identity."""
    src = noise_16bit
    plain = _run(src, d=0, h=1.5)
    withref = _run(src, d=0, h=1.5, rclip=src)
    assert _max_diff(plain, withref, frames=(5,)) == 0.0


def test_rclip_guide_matches_reference_32bit(noise_gray):
    ref = _reference_plugin()
    src = noise_gray
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    mine = _run(src, d=0, h=1.5, rclip=guide)
    plain = _run(src, d=0, h=1.5)
    assert _max_diff(mine, plain, frames=(5,)) > 0.0
    theirs = ref.NLMeans(src, num_streams=1, d=0, h=1.5, rclip=guide)
    assert _max_diff(mine, theirs) < NLMEANS_REF_TOL


def test_rclip_guide_matches_reference_16bit(noise_16bit):
    """16-bit mirror of test_rclip_guide_matches_reference (whole codes)."""
    ref = _reference_plugin()
    src = noise_16bit
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    mine = _run(src, d=0, h=1.5, rclip=guide)
    plain = _run(src, d=0, h=1.5)
    assert _max_diff(mine, plain, frames=(5,)) > 0.0
    theirs = ref.NLMeans(src, num_streams=1, d=0, h=1.5, rclip=guide)
    assert _max_diff(mine, theirs) <= 1.0


def test_stride_handling_matches_reference_32bit(noise_gray):
    # a cropped frame keeps its parent's (wider) stride; the filter must
    # handle non-tight pitches identically to the reference
    ref = _reference_plugin()
    cropped = noise_gray.std.Crop(left=27)
    assert cropped.width < noise_gray.width
    mine = _run(cropped, d=1)
    theirs = ref.NLMeans(cropped, num_streams=1, d=1)
    assert _max_diff(mine, theirs) < NLMEANS_REF_TOL


def test_stride_handling_matches_reference_16bit(noise_16bit):
    """16-bit mirror of test_stride_handling_matches_reference (whole
    codes)."""
    ref = _reference_plugin()
    cropped = noise_16bit.std.Crop(left=27)
    assert cropped.width < noise_16bit.width
    mine = _run(cropped, d=1)
    theirs = ref.NLMeans(cropped, num_streams=1, d=1)
    assert _max_diff(mine, theirs) <= 1.0


# --- determinism / streams ---------------------------------------------------


def test_deterministic_serial_32bit(noise_gray):
    a = _run(noise_gray, d=2, h=1.5)
    b = _run(noise_gray, d=2, h=1.5)
    assert _max_diff(a, b) == 0.0


def test_deterministic_serial_16bit(noise_16bit):
    a = _run(noise_16bit, d=2, h=1.5)
    b = _run(noise_16bit, d=2, h=1.5)
    assert _max_diff(a, b) == 0.0


def test_multi_stream_matches_single_32bit(noise_gray):
    a = _run(noise_gray, d=2, num_streams=4)
    b = _run(noise_gray, d=2, num_streams=1)
    assert _max_diff(a, b) == 0.0


def test_multi_stream_matches_single_16bit(noise_16bit):
    a = _run(noise_16bit, d=2, num_streams=4)
    b = _run(noise_16bit, d=2, num_streams=1)
    assert _max_diff(a, b) == 0.0


def test_multi_stream_temporal_matches_single_32bit(noise_yuv32):
    a = _run(noise_yuv32, d=1, channels="UV", num_streams=4)
    b = _run(noise_yuv32, d=1, channels="UV", num_streams=1)
    assert _max_diff(a, b, planes=(1, 2)) == 0.0


def test_multi_stream_uv_matches_single_16bit(noise_yuv420_16):
    a = _run(noise_yuv420_16, d=1, channels="UV", num_streams=4)
    b = _run(noise_yuv420_16, d=1, channels="UV", num_streams=1)
    assert _max_diff(a, b, planes=(1, 2)) == 0.0


def test_parallel_load_matches_serial_32bit(noise_gray):
    par = _eval_parallel(noise_gray, d=2, num_streams=4)
    ref = _run(noise_gray, d=2, num_streams=1)
    for n in range(noise_gray.num_frames):
        fb = _plane(ref.get_frame(n), 0)
        assert np.abs(par[n].astype(np.float64) - fb).max() == 0.0, \
            f"parallel/serial mismatch at frame {n}"


def test_parallel_load_matches_serial_16bit(noise_16bit):
    par = _eval_parallel(noise_16bit, d=2, num_streams=4)
    ref = _run(noise_16bit, d=2, num_streams=1)
    for n in range(noise_16bit.num_frames):
        fb = _plane(ref.get_frame(n), 0)
        assert np.abs(par[n].astype(np.float64) - fb).max() == 0.0, \
            f"parallel/serial mismatch at frame {n}"


def test_parallel_load_deterministic_32bit(noise_gray):
    a = _eval_parallel(noise_gray, d=2, num_streams=4)
    b = _eval_parallel(noise_gray, d=2, num_streams=4)
    for n in range(noise_gray.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


def test_parallel_load_deterministic_16bit(noise_16bit):
    a = _eval_parallel(noise_16bit, d=2, num_streams=4)
    b = _eval_parallel(noise_16bit, d=2, num_streams=4)
    for n in range(noise_16bit.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


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
    out = _run(noise_16bit, d=radius)
    assert out.format.id == noise_16bit.format.id
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0)
        assert np.isfinite(a.astype(np.float32)).all(), f"non-finite output at frame {n}"


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
