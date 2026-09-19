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

Every reference comparison runs in a subprocess (``reference_compare``) so a
reference crash cannot take the pytest process down with it.

Run from the repository root:  python -m pytest tests/test_bilateral.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_changes_on_noise, assert_gray32, format_dtype, frame_to_ndarray,
    plane_to_ndarray, reference_compare, reference_or_skip, reference_spec,
)

pytestmark = pytest.mark.usefixtures("noise_gray")

SIGMA_SPATIAL = 3.0
SIGMA_COLOR = 0.05
REF_TOL = 1e-6
BORDER_TOL = 0.01


def _run(clip, **kwargs):
    return vs.core.vsfeel.Bilateral(clip, **kwargs)


def _plane(frame, plane):
    """Copy a frame plane into an ndarray, honouring the row pitch.

    Delegates to the shared stride-aware reader (geometry derived from the
    frame, result always a fresh copy).
    """
    return plane_to_ndarray(frame, plane, format_dtype(frame.format))


def _compare(fmt, frames, params, guide=None):
    """Worst diff vs vszipcl over ``frames`` (subprocess; skips if absent)."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec("vszipcl", "Bilateral", fmt, frames=frames,
                          kwargs=params, guide=guide)
    return reference_compare(spec)["maxdiff"]


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
    params = dict(sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR,
                  num_streams=2)
    worst = _compare("gray32", (0, 11, 23), params)
    assert worst < REF_TOL, f"max diff {worst}"


# ---------------------------------------------------------------------------
# Reference sweeps (both bit depths, parameter grid)
# ---------------------------------------------------------------------------

def test_bilateral_matches_reference_16bit(noise_16bit):
    """16-bit integer input: within one output-code rounding step."""
    params = dict(sigma_spatial=2.0, sigma_color=0.05)
    worst = _compare("gray16", (0, 7, 23), params)
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
    worst = _compare("gray16", (3, 17),
                     dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color))
    assert worst <= tol, f"max diff {worst} LSB (tol {tol})"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_16bit(noise_16bit, radius):
    """16-bit mirror of test_bilateral_radius_sweep_matches_reference."""
    worst = _compare("gray16", (3, 17),
                     dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius))
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_16bit(
        noise_16bit, use_shared_memory):
    """16-bit mirror of test_bilateral_shader_variants_match_vszipcl."""
    worst = _compare("gray16", (0, 23),
                     dict(sigma_spatial=2.0, sigma_color=0.05,
                          use_shared_memory=use_shared_memory))
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_16bit(noise_gray, use_shared_memory):
    """16-bit YUV420: all planes must track the reference in both shader
    variants (mirror of the 32-bit YUV test)."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec(
        "vszipcl", "Bilateral", "yuv420_16", frames=(0, 11),
        kwargs=dict(use_shared_memory=use_shared_memory))
    worst = reference_compare(spec)["maxdiff"]
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
    worst = _compare("gray32", (3, 17),
                     dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color))
    assert worst < tol, f"max diff {worst} (tol {tol})"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_32bit(noise_gray, radius):
    """Explicit small and large radii override the auto-derived window."""
    worst = _compare("gray32", (3, 17),
                     dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius))
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_32bit(noise_gray, use_shared_memory):
    """Both shader variants track the reference (borders may differ between
    variants, exactly like the reference's own _sm/_gl pair)."""
    worst = _compare("gray32", (0, 23),
                     dict(sigma_spatial=2.0, sigma_color=0.05,
                          use_shared_memory=use_shared_memory))
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_32bit(noise_gray, use_shared_memory):
    """YUV float32: all planes (incl. subsampled chroma with scaled default
    sigmas) must match the reference, in both shader variants."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec(
        "vszipcl", "Bilateral", "yuv32", frames=(0, 11),
        kwargs=dict(use_shared_memory=use_shared_memory))
    worst = reference_compare(spec)["maxdiff"]
    assert worst < REF_TOL, f"max diff {worst}"


def test_bilateral_ref_clip_matches_reference_32bit(noise_gray):
    """Joint filtering through a guide clip matches the reference."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05)
    out = _run(noise_gray, ref=noise_gray.std.FlipVertical(), **kwargs)
    assert_gray32(out)
    worst = _compare("gray32", (0, 11, 23), kwargs,
                     guide={"kind": "flipvertical"})
    assert worst < REF_TOL, f"max diff {worst}"


def test_bilateral_output_finite_16bit(noise_16bit):
    """The 16-bit output must be finite AND actually filtered: an identity
    implementation used to pass the uint16 finiteness/range check."""
    for num_streams in (1, 2):
        out = _run(
            noise_16bit,
            sigma_spatial=SIGMA_SPATIAL,
            sigma_color=SIGMA_COLOR,
            num_streams=num_streams,
        )
        assert out.format.id == noise_16bit.format.id
        assert_changes_on_noise(out, noise_16bit, frames=(10,),
                                what="bilateral")


def test_bilateral_defaults_run_16bit(noise_16bit):
    out = _run(noise_16bit)
    assert out.format.id == noise_16bit.format.id
    assert_changes_on_noise(out, noise_16bit, frames=(0,), what="bilateral")


def test_bilateral_deterministic_16bit(noise_16bit):
    a = _run(noise_16bit, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(noise_16bit, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = _plane(a.get_frame(n), 0) - _plane(b.get_frame(n), 0)
        assert np.abs(da).max() == 0.0


def test_bilateral_ref_clip_matches_reference_16bit(noise_16bit):
    """16-bit mirror of test_bilateral_ref_clip_matches_reference."""
    worst = _compare("gray16", (0, 11, 23),
                     dict(sigma_spatial=2.0, sigma_color=0.05),
                     guide={"kind": "flipvertical"})
    assert worst <= 1.0, f"max diff {worst} LSB"


# ---------------------------------------------------------------------------
# Workgroup shape (block_x / block_y)
# ---------------------------------------------------------------------------
#
# vszipcl hardcodes 16x8 and takes no block args, so these are checked for
# exact shape-invariance instead (the kernel is a per-pixel gather, so the
# shape is a pure launch parameter); the default shape is anchored to vszipcl
# by the sweeps above.

BLOCK_SHAPES = [(16, 16), (8, 32), (32, 8), (64, 1), (1, 64), (16, 8), (4, 4)]


@pytest.mark.parametrize("block_x,block_y", BLOCK_SHAPES,
                         ids=[f"{x}x{y}" for x, y in BLOCK_SHAPES])
def test_bilateral_block_shape_does_not_change_output_32bit(
        noise_gray, block_x, block_y):
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05, num_streams=1)
    default = _run(noise_gray, **kwargs)
    shaped = _run(noise_gray, block_x=block_x, block_y=block_y, **kwargs)
    for n in (0, 11, 23):
        a = frame_to_ndarray(default.get_frame(n))
        b = frame_to_ndarray(shaped.get_frame(n))
        assert np.array_equal(a, b), \
            f"block {block_x}x{block_y} changed frame {n}"


@pytest.mark.parametrize("block_x,block_y", BLOCK_SHAPES,
                         ids=[f"{x}x{y}" for x, y in BLOCK_SHAPES])
def test_bilateral_block_shape_does_not_change_output_16bit(
        noise_16bit, block_x, block_y):
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05, num_streams=1)
    default = _run(noise_16bit, **kwargs)
    shaped = _run(noise_16bit, block_x=block_x, block_y=block_y, **kwargs)
    for n in (0, 11, 23):
        a = _plane(default.get_frame(n), 0)
        b = _plane(shaped.get_frame(n), 0)
        assert np.array_equal(a, b), \
            f"block {block_x}x{block_y} changed frame {n}"


def test_bilateral_rejects_zero_block(noise_gray):
    """A zero workgroup dimension cannot be launched and must be rejected."""
    for bx, by in [(0, 8), (8, 0), (0, 0)]:
        with pytest.raises(vs.Error):
            _run(noise_gray, sigma_spatial=3.0, sigma_color=0.05,
                 block_x=bx, block_y=by)


def test_bilateral_negative_block_is_clamped_not_rejected(noise_gray):
    """Pins current behaviour: a negative block dimension is silently clamped
    to the device default, not rejected.

    ``bilateral.cpp:726`` casts to ``uint32_t`` before the limit comparison,
    so -1 wraps and takes the shrink branch before the ``block_x <= 0`` check
    at :733. Output is still correct (the shape has no semantic effect); the
    validation is just inconsistent with ``block_x=0``, which errors.
    """
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05, num_streams=1)
    default = _run(noise_gray, **kwargs)
    for bx, by in [(-1, 8), (8, -1), (-1, -1)]:
        shaped = _run(noise_gray, block_x=bx, block_y=by, **kwargs)
        for n in (0, 11):
            a = frame_to_ndarray(default.get_frame(n))
            b = frame_to_ndarray(shaped.get_frame(n))
            assert np.array_equal(a, b), \
                f"negative block {bx}x{by} changed frame {n}"


# ---------------------------------------------------------------------------
# Per-plane parameter arrays
# ---------------------------------------------------------------------------
#
# Explicit three-element sigma_spatial / sigma_color / radius arrays (incl.
# the documented chroma default) against vszipcl, which the scalar YUV tests
# only covered implicitly.


@pytest.mark.parametrize("kwargs", [
    {"sigma_spatial": [2.0, 1.0, 1.0]},                    # the chroma rule
    {"sigma_spatial": [2.0, 1.5, 1.5], "sigma_color": [0.05, 0.02, 0.02]},
    {"sigma_spatial": [2.0, 1.0, 3.0],
     "sigma_color": [0.05, 0.02, 0.1], "radius": [2, 1, 3]},
    {"sigma_spatial": [1.0, 0.5, 0.5], "radius": [1, 1, 1]},
], ids=["chroma-rule", "spatial+color", "all-three", "radius-1"])
def test_bilateral_per_plane_arrays_match_reference_32bit(noise_gray, kwargs):
    """YUV420 float32 with explicit per-plane arrays must track vszipcl on
    all three planes (incl. the subsampled chroma lattice)."""
    worst = _compare("yuv32", (0, 11, 23), kwargs)
    assert worst < REF_TOL, f"max diff {worst} ({kwargs})"


@pytest.mark.parametrize("kwargs", [
    {"sigma_spatial": [2.0, 1.0, 1.0]},
    {"sigma_spatial": [2.0, 1.5, 1.5], "sigma_color": [0.05, 0.02, 0.02]},
    {"sigma_spatial": [2.0, 1.0, 3.0], "radius": [2, 1, 3]},
], ids=["chroma-rule", "spatial+color", "spatial+radius"])
def test_bilateral_per_plane_arrays_match_reference_16bit(noise_16bit, kwargs):
    """16-bit mirror of the per-plane array sweep (whole output codes)."""
    worst = _compare("yuv420_16", (0, 11, 23), kwargs)
    assert worst <= 1.0, f"max diff {worst} LSB ({kwargs})"


def test_bilateral_per_plane_arrays_gray_32bit(noise_gray):
    """A three-element array on a single-plane clip: only element 0 is used,
    and the result must equal the scalar-parameter run exactly."""
    scalar = _run(noise_gray, sigma_spatial=2.0, sigma_color=0.05,
                  num_streams=1)
    array = _run(noise_gray, sigma_spatial=[2.0, 1.5, 1.0],
                 sigma_color=[0.05, 0.03, 0.02], num_streams=1)
    for n in (0, 11):
        a = frame_to_ndarray(scalar.get_frame(n))
        b = frame_to_ndarray(array.get_frame(n))
        assert np.array_equal(a, b), f"extra array elements changed frame {n}"
    worst = _compare("gray32", (0, 11, 23),
                     dict(sigma_spatial=[2.0, 1.5, 1.0],
                          sigma_color=[0.05, 0.03, 0.02]))
    assert worst < REF_TOL, f"max diff {worst}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_bilateral_rejects_negative_sigma(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma_spatial=-1.0)
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma_color=-1.0)


def test_bilateral_rejects_nonpositive_radius(noise_gray):
    for bad in (0, -1):
        with pytest.raises(vs.Error):
            _run(noise_gray, radius=bad)
