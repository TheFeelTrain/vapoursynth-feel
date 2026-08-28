"""Unit tests for core.vsfeel.Bilateral.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input: it exercises the filter on high-frequency content without any flat
or black regions.

Reference comparisons run against core.vszipcl.Bilateral when that plugin is
installed, with a two-tier "close enough" policy:

- REF_TOL (1e-6 on the normalized [0, 1] scale) for everything where both
  sides compute the same math — measured diffs are float32 noise (~1e-8).
- BORDER_TOL (0.01) only for configs where the staging tile outgrows the
  shared-memory budget: there the reference's own kernel variants disagree
  at frame borders (edge-clamp vs window truncation), a semantic difference
  worth up to ~7e-3 on noise input.

16-bit integer output is compared in whole output codes (<= 1 LSB): both
sides round nearly identical fp32 results once, so codes differ by at most
one rounding step. Self-consistency checks (determinism) remain exact.

Run from the repository root:  python -m pytest tests/test_bilateral.py
"""

import ctypes

import numpy as np
import pytest
import vapoursynth as vs

from conftest import HEIGHT, WIDTH, assert_gray32, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")

SIGMA_SPATIAL = 3.0
SIGMA_COLOR = 0.05
REF_TOL = 1e-6
BORDER_TOL = 0.01


def _run(clip, **kwargs):
    return vs.core.vsfeel.Bilateral(clip, **kwargs)


def _ref(clip, **kwargs):
    """Reference output, skipping the test if vszipcl is not installed."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "Bilateral"):
        pytest.skip("vszipcl not installed")
    return vs.core.vszipcl.Bilateral(clip, **kwargs)


def _plane(frame, plane):
    """Copy a frame plane into an ndarray, honouring the row pitch."""
    fmt = frame.format
    itemsize = fmt.bytes_per_sample
    ss_w = fmt.subsampling_w if plane in (1, 2) and fmt.num_planes >= 3 else 0
    ss_h = fmt.subsampling_h if plane in (1, 2) and fmt.num_planes >= 3 else 0
    w = frame.width >> ss_w
    h = frame.height >> ss_h
    raw = np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(h, frame.get_stride(plane)),
    )
    dtype = np.float32 if fmt.sample_type == vs.FLOAT else np.uint16
    return raw[:, :w * itemsize].copy().view(dtype).reshape(h, w)


def _max_diff(a_node, b_node, frames=(0, 11, 23)):
    worst = 0.0
    for n in frames:
        fa, fb = a_node.get_frame(n), b_node.get_frame(n)
        for p in range(fa.format.num_planes):
            d = np.abs(
                _plane(fa, p).astype(np.float64) - _plane(fb, p).astype(np.float64)
            )
            worst = max(worst, float(d.max()))
    return worst


def test_bilateral_output_finite_32bit(noise_gray):
    for num_streams in (1, 2):
        out = _run(
            noise_gray,
            sigma_spatial=SIGMA_SPATIAL,
            sigma_color=SIGMA_COLOR,
            num_streams=num_streams,
        )
        assert_gray32(out)
        for n in (0, 10, 23):
            frame = out.get_frame(n)
            a = frame_to_ndarray(frame)
            assert np.isfinite(a).all(), f"non-finite output at frame {n}"
            assert a.min() >= 0.0 and a.max() <= 1.0


def test_bilateral_defaults_run_32bit(noise_gray):
    out = _run(noise_gray)
    assert_gray32(out)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bilateral_deterministic_32bit(noise_gray):
    a = _run(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(da).max() == 0.0


def test_bilateral_rejects_8bit(noise_8bit):
    """Only 16-bit integer and 32-bit float input is supported."""
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_bilateral_rejects_10bit(noise_8bit):
    """Formats other than 16-bit int and 32-bit float must be rejected."""
    clip10 = vs.core.fmtc.bitdepth(noise_8bit, bits=10)
    with pytest.raises(vs.Error):
        _run(clip10)


def test_bilateral_matches_reference_32bit(noise_gray):
    """Default parameters stay close to the reference implementation."""
    for n in (0, 11, 23):
        a = frame_to_ndarray(
            _run(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR, num_streams=2).get_frame(n)
        )
        b = frame_to_ndarray(
            _ref(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR, num_streams=2).get_frame(n)
        )
        d = np.abs(a - b)
        assert d.max() < REF_TOL, f"max diff {d.max()} at frame {n}"


# ---------------------------------------------------------------------------
# Reference sweeps (both bit depths, parameter grid)
# ---------------------------------------------------------------------------

def test_bilateral_matches_reference_16bit(noise_16bit):
    """16-bit integer input: within one output-code rounding step."""
    out = _run(noise_16bit, sigma_spatial=2.0, sigma_color=0.05)
    ref = _ref(noise_16bit, sigma_spatial=2.0, sigma_color=0.05)
    worst = _max_diff(out, ref, frames=(0, 7, 23))
    assert worst <= 1.0, f"int16 max diff {worst} LSB"


# The 32-bit sweeps above are mirrored at 16-bit (whole-code comparison).
# Measured on the noise clip: every non-border config lands at 1 LSB; the
# wide-sigma border case (staging tile > LDS budget, where the reference's
# own kernel variants disagree at borders) measured 328 codes, bound by the
# BORDER_TOL equivalent in 16-bit codes (0.01 * 65535).
BORDER_TOL_CODES = 655.0


@pytest.mark.parametrize("sigma_spatial,sigma_color,tol", [
    (1.0, 0.02, 1.0),
    (4.0, 0.05, 1.0),
    (8.0, 0.15, BORDER_TOL_CODES),
], ids=["small-sigma", "default-like", "wide-sigma"])
def test_bilateral_sigma_sweep_matches_reference_16bit(
        noise_16bit, sigma_spatial, sigma_color, tol):
    """16-bit mirror of test_bilateral_sigma_sweep_matches_reference."""
    kwargs = dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color)
    out = _run(noise_16bit, **kwargs)
    ref = _ref(noise_16bit, **kwargs)
    worst = _max_diff(out, ref, frames=(3, 17))
    assert worst <= tol, f"max diff {worst} LSB (tol {tol})"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_16bit(noise_16bit, radius):
    """16-bit mirror of test_bilateral_radius_sweep_matches_reference."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius)
    out = _run(noise_16bit, **kwargs)
    ref = _ref(noise_16bit, **kwargs)
    worst = _max_diff(out, ref, frames=(3, 17))
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_16bit(
        noise_16bit, use_shared_memory):
    """16-bit mirror of test_bilateral_shader_variants_match_vszipcl."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05,
                  use_shared_memory=use_shared_memory)
    out = _run(noise_16bit, **kwargs)
    ref = _ref(noise_16bit, **kwargs)
    worst = _max_diff(out, ref, frames=(0, 23))
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_16bit(noise_gray, use_shared_memory):
    """16-bit YUV420: all planes must track the reference in both shader
    variants (mirror of the 32-bit YUV test)."""
    if hasattr(vs.core, "bs"):
        source = vs.core.bs.VideoSource("tests/noise_24f.mkv")
    else:
        source = vs.core.ffms2.Source("tests/noise_24f.mkv")
    yuv = vs.core.resize.Bicubic(source, format=vs.YUV420P16)
    kwargs = dict(use_shared_memory=use_shared_memory)
    out = _run(yuv, **kwargs)
    ref = _ref(yuv, **kwargs)
    worst = _max_diff(out, ref, frames=(0, 11))
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("sigma_spatial,sigma_color,tol", [
    (1.0, 0.02, REF_TOL),
    (4.0, 0.05, REF_TOL),
    # radius 24: the staging tile exceeds the 48 KiB shared-memory budget,
    # where the reference's own kernel variants diverge at borders.
    (8.0, 0.15, BORDER_TOL),
], ids=["small-sigma", "default-like", "wide-sigma"])
def test_bilateral_sigma_sweep_matches_reference_32bit(noise_gray, sigma_spatial, sigma_color, tol):
    """Sigma grid on float input; radius is auto-derived from sigma_spatial."""
    kwargs = dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color)
    out = _run(noise_gray, **kwargs)
    ref = _ref(noise_gray, **kwargs)
    worst = _max_diff(out, ref, frames=(3, 17))
    assert worst < tol, f"max diff {worst} (tol {tol})"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_32bit(noise_gray, radius):
    """Explicit small and large radii override the auto-derived window."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius)
    out = _run(noise_gray, **kwargs)
    ref = _ref(noise_gray, **kwargs)
    worst = _max_diff(out, ref, frames=(3, 17))
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_32bit(noise_gray, use_shared_memory):
    """Both shader variants track the reference (borders may differ between
    variants, exactly like the reference's own _sm/_gl pair)."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05,
                  use_shared_memory=use_shared_memory)
    out = _run(noise_gray, **kwargs)
    ref = _ref(noise_gray, **kwargs)
    worst = _max_diff(out, ref, frames=(0, 23))
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_32bit(noise_gray, use_shared_memory):
    """YUV float32: all planes (incl. subsampled chroma with scaled default
    sigmas) must match the reference, in both shader variants."""
    if hasattr(vs.core, "bs"):
        source = vs.core.bs.VideoSource("tests/noise_24f.mkv")
    else:
        source = vs.core.ffms2.Source("tests/noise_24f.mkv")
    yuv = vs.core.fmtc.bitdepth(source, bits=32, fulls=True, fulld=True)
    kwargs = dict(use_shared_memory=use_shared_memory)
    out = _run(yuv, **kwargs)
    ref = _ref(yuv, **kwargs)
    worst = _max_diff(out, ref, frames=(0, 11))
    assert worst < REF_TOL, f"max diff {worst}"


def test_bilateral_ref_clip_matches_reference_32bit(noise_gray):
    """Joint filtering through a guide clip matches the reference."""
    guide = noise_gray.std.FlipVertical()
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05, ref=guide)
    out = _run(noise_gray, **kwargs)
    ref = _ref(noise_gray, **kwargs)
    worst = _max_diff(out, ref, frames=(0, 11, 23))
    assert worst < REF_TOL, f"max diff {worst}"
    assert_gray32(out)


def test_bilateral_output_finite_16bit(noise_16bit):
    for num_streams in (1, 2):
        out = _run(
            noise_16bit,
            sigma_spatial=SIGMA_SPATIAL,
            sigma_color=SIGMA_COLOR,
            num_streams=num_streams,
        )
        assert out.format.id == noise_16bit.format.id
        a = _plane(out.get_frame(10), 0)
        assert np.isfinite(a.astype(np.float32)).all()
        assert a.min() >= 0 and a.max() <= 65535


def test_bilateral_defaults_run_16bit(noise_16bit):
    out = _run(noise_16bit)
    assert out.format.id == noise_16bit.format.id
    a = _plane(out.get_frame(0), 0)
    assert np.isfinite(a.astype(np.float32)).all()


def test_bilateral_deterministic_16bit(noise_16bit):
    a = _run(noise_16bit, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(noise_16bit, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = _plane(a.get_frame(n), 0) - _plane(b.get_frame(n), 0)
        assert np.abs(da).max() == 0.0


def test_bilateral_ref_clip_matches_reference_16bit(noise_16bit):
    """16-bit mirror of test_bilateral_ref_clip_matches_reference."""
    guide = noise_16bit.std.FlipVertical()
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05, ref=guide)
    out = _run(noise_16bit, **kwargs)
    ref = _ref(noise_16bit, **kwargs)
    worst = _max_diff(out, ref, frames=(0, 11, 23))
    assert worst <= 1.0, f"max diff {worst} LSB"
