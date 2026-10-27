"""Unit tests for core.vsfeel.Bilateral.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is used as the input: high-frequency noise on top of real
structure, edges and flat regions.

Reference comparisons run against core.vszipcl.Bilateral when that plugin is
installed: REF_TOL (1e-6 on the normalized [0, 1] scale) for float32, where
both sides compute the same math and measured diffs are float32 noise (~1e-8,
including the wide-sigma configs measured at 5.6e-9). u16 output is
compared in whole output codes (<= 1 LSB, 2 for the widest sigma), since both
sides round nearly identical f32 results once. Half input/output is a storage
width in the same f32 pipeline, so its agreement is bounded by one half ulp
(F16_TOL, measured 4.88e-4; see the note on the constant). Self-consistency
checks (determinism) remain exact at every depth.

Every reference comparison runs in a subprocess (``reference_compare``) so a
reference crash cannot take the pytest process down with it.

Run from the repository root:  uv run python -m pytest tests/test_bilateral.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_changes_on_clip,
    assert_grays,
    assert_preserves_frame_props,
    cpu_node,
    eval_parallel,
    frame_to_ndarray,
    plane_to_ndarray,
    plane as _plane,
    reference_compare,
    reference_or_skip,
    reference_spec,
)

pytestmark = pytest.mark.usefixtures("clip_grays")

SIGMA_SPATIAL = 3.0
SIGMA_COLOR = 0.05
REF_TOL = 1e-6

# Half is a storage width, not a precision: the f16 variant runs the f32
# arithmetic and narrows once at its store (round to nearest even, the
# reference's vstore_half_rte). Measured on the noise clip, every f16 config
# below lands at exactly 4.88e-4 (one half step at |v| in [0.5, 1)) on a few
# tens of samples per frame; the two runs' f32 sums differ in their last bits,
# and the half store surfaces that as one step. 1e-3 leaves headroom for a
# second step on another driver's conversion.
F16_TOL = 1e-3


def _run(clip, **kwargs):
    """Bilateral as a clip the test can read pixels from (see conftest.cpu_node)."""
    return cpu_node(vs.core.vsfeel.Bilateral(clip, **kwargs))


def _compare(fmt, frames, params, guide=None):
    """Worst diff vs vszipcl over ``frames`` (subprocess; skips if absent)."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec("vszipcl", "Bilateral", fmt, frames=frames, kwargs=params, guide=guide)
    return reference_compare(spec)["maxdiff"]


def test_bilateral_output_finite_f32(clip_grays):
    out = _run(
        clip_grays,
        sigma_spatial=SIGMA_SPATIAL,
        sigma_color=SIGMA_COLOR,
    )
    assert_grays(out)
    for n in (0, 10, 23):
        frame = out.get_frame(n)
        a = frame_to_ndarray(frame)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"
        assert a.min() >= 0.0 and a.max() <= 1.0


def test_bilateral_defaults_run_f32(clip_grays):
    out = _run(clip_grays)
    assert_grays(out)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bilateral_deterministic_f32(clip_grays):
    a = _run(clip_grays, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(clip_grays, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(da).max() == 0.0


def test_bilateral_rejects_8bit(clip_gray8):
    """Only u16 and f32 input is supported."""
    with pytest.raises(vs.Error):
        _run(clip_gray8)


def test_bilateral_rejects_10bit(clip_gray8):
    """Formats other than u16 int and f32 must be rejected."""
    clip10 = vs.core.fmtc.bitdepth(clip_gray8, bits=10)
    with pytest.raises(vs.Error):
        _run(clip10)


def test_bilateral_matches_reference_f32(clip_grays):
    """Default parameters stay close to the reference implementation."""
    params = dict(sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    worst = _compare("grays", (0, 11, 23), params)
    assert worst < REF_TOL, f"max diff {worst}"


# ---------------------------------------------------------------------------
# f16 mirrors (half is a storage width in the same f32 pipeline)
# ---------------------------------------------------------------------------


def test_bilateral_matches_reference_f16(clip_grayh):
    """Half input: within one half step of the reference's f16 output."""
    params = dict(sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    worst = _compare("grayh", (0, 11, 23), params)
    assert worst <= F16_TOL, f"max diff {worst}"


def test_bilateral_output_finite_f16(clip_grayh):
    """The half output must be finite and inside the [0, 1] sample range."""
    out = _run(clip_grayh, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 10, 23):
        a = _plane(out.get_frame(n), 0, dtype=np.float16).astype(np.float64)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"
        assert a.min() >= 0.0 and a.max() <= 1.0


def test_bilateral_defaults_run_f16(clip_grayh):
    out = _run(clip_grayh)
    assert_changes_on_clip(out, clip_grayh, frames=(0,), what="bilateral f16")


def test_bilateral_deterministic_f16(clip_grayh):
    a = _run(clip_grayh, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(clip_grayh, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = _plane(a.get_frame(n), 0, dtype=np.float16) - _plane(
            b.get_frame(n), 0, dtype=np.float16
        )
        assert np.abs(da).max() == 0.0


def test_bilateral_accepts_f16(clip_grayh):
    """A half clip is processed with its own format, not rejected as an
    unsupported depth or silently routed through the u16 kernel.

    A u16 kernel fed half samples would reinterpret the bit patterns and
    produce a wildly different picture, so the reference comparison above is
    what actually anchors the storage type; this pins the format contract.
    """
    out = _run(clip_grayh, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    assert out.format.id == clip_grayh.format.id
    assert_changes_on_clip(out, clip_grayh, frames=(10,), what="bilateral f16")


# ---------------------------------------------------------------------------
# Reference sweeps (both bit depths, parameter grid)
# ---------------------------------------------------------------------------


def test_bilateral_matches_reference_u16(clip_gray16):
    """u16 input: within one output-code rounding step."""
    params = dict(sigma_spatial=2.0, sigma_color=0.05)
    worst = _compare("gray16", (0, 7, 23), params)
    assert worst <= 1.0, f"u16 max diff {worst} LSB"


# The f32 sweeps above are mirrored at u16 (whole-code comparison).
# Measured on the noise clip: every config lands within one output code; the
# widest sigma gets 2 for rounding headroom.
WIDE_SIGMA_TOL_CODES = 2.0


@pytest.mark.parametrize(
    "sigma_spatial,sigma_color,tol",
    [
        (1.0, 0.02, 1.0),
        (4.0, 0.05, 1.0),
        (8.0, 0.15, WIDE_SIGMA_TOL_CODES),
    ],
    ids=["small-sigma", "default-like", "wide-sigma"],
)
def test_bilateral_sigma_sweep_matches_reference_u16(clip_gray16, sigma_spatial, sigma_color, tol):
    """u16 mirror of test_bilateral_sigma_sweep_matches_reference."""
    worst = _compare("gray16", (3, 17), dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color))
    assert worst <= tol, f"max diff {worst} LSB (tol {tol})"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_u16(clip_gray16, radius):
    """u16 mirror of test_bilateral_radius_sweep_matches_reference."""
    worst = _compare("gray16", (3, 17), dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius))
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_u16(clip_gray16, use_shared_memory):
    """u16 mirror of test_bilateral_shader_variants_match_vszipcl."""
    worst = _compare(
        "gray16",
        (0, 23),
        dict(sigma_spatial=2.0, sigma_color=0.05, use_shared_memory=use_shared_memory),
    )
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_u16(clip_grays, use_shared_memory):
    """u16 YUV420: all planes must track the reference in both shader
    variants (mirror of the f32 YUV test)."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec(
        "vszipcl",
        "Bilateral",
        "yuv420p16",
        frames=(0, 11),
        kwargs=dict(use_shared_memory=use_shared_memory),
    )
    worst = reference_compare(spec)["maxdiff"]
    assert worst <= 1.0, f"max diff {worst} LSB"


@pytest.mark.parametrize(
    "sigma_spatial,sigma_color,tol",
    [
        (1.0, 0.02, REF_TOL),
        (4.0, 0.05, REF_TOL),
        (8.0, 0.15, REF_TOL),
    ],
    ids=["small-sigma", "default-like", "wide-sigma"],
)
def test_bilateral_sigma_sweep_matches_reference_f32(clip_grays, sigma_spatial, sigma_color, tol):
    """Sigma grid on float input; radius is auto-derived from sigma_spatial."""
    worst = _compare("grays", (3, 17), dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color))
    assert worst < tol, f"max diff {worst} (tol {tol})"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_f32(clip_grays, radius):
    """Explicit small and large radii override the auto-derived window."""
    worst = _compare("grays", (3, 17), dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius))
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_f32(clip_grays, use_shared_memory):
    """Both shader variants (shared LDS tile and plain gather) track the
    reference to f32 noise."""
    worst = _compare(
        "grays",
        (0, 23),
        dict(sigma_spatial=2.0, sigma_color=0.05, use_shared_memory=use_shared_memory),
    )
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_f32(clip_grays, use_shared_memory):
    """YUV float32: all planes (incl. subsampled chroma with scaled default
    sigmas) must match the reference, in both shader variants."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec(
        "vszipcl",
        "Bilateral",
        "yuv420ps",
        frames=(0, 11),
        kwargs=dict(use_shared_memory=use_shared_memory),
    )
    worst = reference_compare(spec)["maxdiff"]
    assert worst < REF_TOL, f"max diff {worst}"


def test_bilateral_ref_clip_matches_reference_f32(clip_grays):
    """Joint filtering through a guide clip matches the reference."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05)
    out = _run(clip_grays, ref=clip_grays.std.FlipVertical(), **kwargs)
    assert_grays(out)
    worst = _compare("grays", (0, 11, 23), kwargs, guide={"kind": "flipvertical"})
    assert worst < REF_TOL, f"max diff {worst}"


def test_bilateral_output_finite_u16(clip_gray16):
    """The u16 output must be finite AND actually filtered: an identity
    implementation used to pass the uint16 finiteness/range check."""
    out = _run(
        clip_gray16,
        sigma_spatial=SIGMA_SPATIAL,
        sigma_color=SIGMA_COLOR,
    )
    assert out.format.id == clip_gray16.format.id
    assert_changes_on_clip(out, clip_gray16, frames=(10,), what="bilateral")


def test_bilateral_defaults_run_u16(clip_gray16):
    out = _run(clip_gray16)
    assert out.format.id == clip_gray16.format.id
    assert_changes_on_clip(out, clip_gray16, frames=(0,), what="bilateral")


def test_bilateral_preserves_frame_props(clip_grays):
    """Bilateral must republish the source frame's properties."""
    assert_preserves_frame_props(
        _run, clip_grays, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR
    )


def test_bilateral_preserves_frame_props_f16(clip_grayh):
    """The half path must republish props too (the store narrows pixels, not
    frame props)."""
    assert_preserves_frame_props(
        _run, clip_grayh, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR
    )


def test_bilateral_deterministic_u16(clip_gray16):
    a = _run(clip_gray16, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(clip_gray16, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = _plane(a.get_frame(n), 0) - _plane(b.get_frame(n), 0)
        assert np.abs(da).max() == 0.0


def test_bilateral_ref_clip_matches_reference_u16(clip_gray16):
    """u16 mirror of test_bilateral_ref_clip_matches_reference."""
    worst = _compare(
        "gray16",
        (0, 11, 23),
        dict(sigma_spatial=2.0, sigma_color=0.05),
        guide={"kind": "flipvertical"},
    )
    assert worst <= 1.0, f"max diff {worst} LSB"


# --- f16 sweeps, mirroring the u16 and f32 sets above -----------------------
#
# Same parameter grid, same shader variants, same joint-filter and YUV cases as
# the u16 set, against vszipcl's f16 output. The bound is one half step plus
# headroom (see F16_TOL).


@pytest.mark.parametrize(
    "sigma_spatial,sigma_color",
    [
        (1.0, 0.02),
        (4.0, 0.05),
        (8.0, 0.15),
    ],
    ids=["small-sigma", "default-like", "wide-sigma"],
)
def test_bilateral_sigma_sweep_matches_reference_f16(clip_grayh, sigma_spatial, sigma_color):
    """Sigma grid on half input; radius is auto-derived from sigma_spatial."""
    worst = _compare("grayh", (3, 17), dict(sigma_spatial=sigma_spatial, sigma_color=sigma_color))
    assert worst <= F16_TOL, f"max diff {worst}"


@pytest.mark.parametrize("radius", [1, 5])
def test_bilateral_radius_sweep_matches_reference_f16(clip_grayh, radius):
    """Explicit small and large radii on half input."""
    worst = _compare("grayh", (3, 17), dict(sigma_spatial=2.0, sigma_color=0.05, radius=radius))
    assert worst <= F16_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_shader_variants_match_vszipcl_f16(clip_grayh, use_shared_memory):
    """Both half shader variants (shared LDS tile and plain gather) track the
    reference; the plain run passes the same flag to vszipcl, so this is the
    plain-vs-plain comparison (the two kernels' border semantics differ)."""
    worst = _compare(
        "grayh",
        (0, 23),
        dict(sigma_spatial=2.0, sigma_color=0.05, use_shared_memory=use_shared_memory),
    )
    assert worst <= F16_TOL, f"max diff {worst}"


@pytest.mark.parametrize("use_shared_memory", [True, False], ids=["shared", "plain"])
def test_bilateral_yuv_matches_reference_f16(clip_grayh, use_shared_memory):
    """YUV420PH: all planes (incl. subsampled chroma with scaled default
    sigmas) must track the reference, in both shader variants."""
    reference_or_skip("vszipcl", "Bilateral")
    spec = reference_spec(
        "vszipcl",
        "Bilateral",
        "yuv420ph",
        frames=(0, 11),
        kwargs=dict(use_shared_memory=use_shared_memory),
    )
    worst = reference_compare(spec)["maxdiff"]
    assert worst <= F16_TOL, f"max diff {worst}"


def test_bilateral_ref_clip_matches_reference_f16(clip_grayh):
    """Joint filtering through a guide clip, half variant."""
    kwargs = dict(sigma_spatial=2.0, sigma_color=0.05)
    out = _run(clip_grayh, ref=clip_grayh.std.FlipVertical(), **kwargs)
    assert out.format.id == clip_grayh.format.id
    worst = _compare("grayh", (0, 11, 23), kwargs, guide={"kind": "flipvertical"})
    assert worst <= F16_TOL, f"max diff {worst}"


def test_bilateral_per_plane_lds_tiers_match_reference_f32(clip_grays, monkeypatch):
    """Planes of one instance may take different shared kernels.

    The shared kernel computes two output columns per thread when its wider
    tile fits, and falls back to one column otherwise; the choice is per plane.
    With radius 24 on luma and 3 on chroma at the auto 32x16 block, a 24576-byte
    limit rejects luma's two-column tile (28672 B) but not its one-column one
    (20480 B), while chroma's two-column tile (6160 B) still fits: the two tiers
    run side by side in one filter. Both must track vszipcl.
    """
    monkeypatch.setenv("VSFEEL_LIMIT_SHARED_MEMORY", "24576")
    worst = _compare(
        "yuv420ps",
        (0, 11, 23),
        dict(sigma_spatial=[8.0, 1.0, 1.0], sigma_color=[0.15, 0.05, 0.05]),
    )
    assert worst < REF_TOL, f"max diff {worst}"


def test_bilateral_f16_is_f32_rounded(clip_grayh):
    """The half path is the f32 pipeline plus one store rounding.

    ``gh`` widened to GRAYS is a container change (half -> float is exact), so
    the two runs carry byte-identical input. Measured: one half step on 125 of
    2.07M samples (the half variant's extra OpFConvert shifts the driver's
    contraction, and the store rounds that last-bit difference), so the bound is
    the f16 constant rather than 0. This oracle is device-independent and fails
    first if the half variant ever grows arithmetic of its own — e.g. a half LDS
    tile or a half accumulator, which the reference comparison could hide
    behind its tolerance.
    """
    widened = vs.core.resize.Point(clip_grayh, format=vs.GRAYS)
    half = _run(clip_grayh, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    full = _run(widened, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (3, 11, 23):
        a = _plane(half.get_frame(n), 0, dtype=np.float16).astype(np.float64)
        b = _plane(full.get_frame(n), 0, dtype=np.float32).astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all(), f"non-finite at frame {n}"
        worst = float(np.abs(a - b).max())
        assert worst <= F16_TOL, f"f16 vs f32 max diff {worst} at frame {n}"


def test_bilateral_f16_is_f32_rounded_yuv(clip_yuv420ph):
    """YUV420PH mirror of the rounded-once oracle: every plane, subsampled
    chroma included."""
    widened = vs.core.resize.Point(clip_yuv420ph, format=vs.YUV420PS)
    half = _run(clip_yuv420ph, sigma_spatial=2.0, sigma_color=0.05)
    full = _run(widened, sigma_spatial=2.0, sigma_color=0.05)
    for n in (3, 11, 23):
        fa, fb = half.get_frame(n), full.get_frame(n)
        for p in range(fa.format.num_planes):
            a = _plane(fa, p, dtype=np.float16).astype(np.float64)
            b = _plane(fb, p, dtype=np.float32).astype(np.float64)
            worst = float(np.abs(a - b).max())
            assert worst <= F16_TOL, f"plane {p}: f16 vs f32 max diff {worst} at frame {n}"


# ---------------------------------------------------------------------------
# Workgroup shape (block_x / block_y)
# ---------------------------------------------------------------------------
#
# vszipcl hardcodes 16x8 and takes no block args, so these are checked for
# exact shape-invariance instead (the kernel is a per-pixel gather, so the
# shape is a pure launch parameter); the default shape is anchored to vszipcl
# by the sweeps above.

BLOCK_SHAPES = [(16, 16), (8, 32), (32, 8), (64, 1), (1, 64), (16, 8), (4, 4)]


@pytest.mark.parametrize("block_x,block_y", BLOCK_SHAPES, ids=[f"{x}x{y}" for x, y in BLOCK_SHAPES])
def test_bilateral_block_shape_does_not_change_output_f32(clip_grays, block_x, block_y):
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05)
    default = _run(clip_grays, **kwargs)
    shaped = _run(clip_grays, block_x=block_x, block_y=block_y, **kwargs)
    for n in (0, 11, 23):
        a = frame_to_ndarray(default.get_frame(n))
        b = frame_to_ndarray(shaped.get_frame(n))
        assert np.array_equal(a, b), f"block {block_x}x{block_y} changed frame {n}"


@pytest.mark.parametrize("block_x,block_y", BLOCK_SHAPES, ids=[f"{x}x{y}" for x, y in BLOCK_SHAPES])
def test_bilateral_block_shape_does_not_change_output_u16(clip_gray16, block_x, block_y):
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05)
    default = _run(clip_gray16, **kwargs)
    shaped = _run(clip_gray16, block_x=block_x, block_y=block_y, **kwargs)
    for n in (0, 11, 23):
        a = _plane(default.get_frame(n), 0)
        b = _plane(shaped.get_frame(n), 0)
        assert np.array_equal(a, b), f"block {block_x}x{block_y} changed frame {n}"


@pytest.mark.parametrize("block_x,block_y", BLOCK_SHAPES, ids=[f"{x}x{y}" for x, y in BLOCK_SHAPES])
def test_bilateral_block_shape_does_not_change_output_f16(clip_grayh, block_x, block_y):
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05)
    default = _run(clip_grayh, **kwargs)
    shaped = _run(clip_grayh, block_x=block_x, block_y=block_y, **kwargs)
    for n in (0, 11, 23):
        a = _plane(default.get_frame(n), 0, dtype=np.float16)
        b = _plane(shaped.get_frame(n), 0, dtype=np.float16)
        assert np.array_equal(a, b), f"block {block_x}x{block_y} changed frame {n}"


def test_bilateral_rejects_zero_block(clip_grays):
    """A zero workgroup dimension cannot be launched and must be rejected."""
    for bx, by in [(0, 8), (8, 0), (0, 0)]:
        with pytest.raises(vs.Error):
            _run(clip_grays, sigma_spatial=3.0, sigma_color=0.05, block_x=bx, block_y=by)


def test_bilateral_rejects_negative_block_dimensions(clip_grays):
    """A negative block dimension is an error, like block_x=0.

    It used to wrap in the unsigned device-limit comparison and silently become
    the 16x16 default, which made a typo a no-op.
    """
    kwargs = dict(sigma_spatial=3.0, sigma_color=0.05)
    for bx, by in [(-1, 8), (8, -1), (-1, -1)]:
        with pytest.raises(vs.Error, match="must be positive"):
            _run(clip_grays, block_x=bx, block_y=by, **kwargs)


# ---------------------------------------------------------------------------
# Per-plane parameter arrays
# ---------------------------------------------------------------------------
#
# Explicit three-element sigma_spatial / sigma_color / radius arrays (incl.
# the documented chroma default) against vszipcl, which the scalar YUV tests
# only covered implicitly.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_spatial": [2.0, 1.0, 1.0]},  # the chroma rule
        {"sigma_spatial": [2.0, 1.5, 1.5], "sigma_color": [0.05, 0.02, 0.02]},
        {"sigma_spatial": [2.0, 1.0, 3.0], "sigma_color": [0.05, 0.02, 0.1], "radius": [2, 1, 3]},
        {"sigma_spatial": [1.0, 0.5, 0.5], "radius": [1, 1, 1]},
    ],
    ids=["chroma-rule", "spatial+color", "all-three", "radius-1"],
)
def test_bilateral_per_plane_arrays_match_reference_f32(clip_grays, kwargs):
    """YUV420 float32 with explicit per-plane arrays must track vszipcl on
    all three planes (incl. the subsampled chroma lattice)."""
    worst = _compare("yuv420ps", (0, 11, 23), kwargs)
    assert worst < REF_TOL, f"max diff {worst} ({kwargs})"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_spatial": [2.0, 1.0, 1.0]},
        {"sigma_spatial": [2.0, 1.5, 1.5], "sigma_color": [0.05, 0.02, 0.02]},
        {"sigma_spatial": [2.0, 1.0, 3.0], "radius": [2, 1, 3]},
    ],
    ids=["chroma-rule", "spatial+color", "spatial+radius"],
)
def test_bilateral_per_plane_arrays_match_reference_u16(clip_gray16, kwargs):
    """u16 mirror of the per-plane array sweep (whole output codes)."""
    worst = _compare("yuv420p16", (0, 11, 23), kwargs)
    assert worst <= 1.0, f"max diff {worst} LSB ({kwargs})"


def test_bilateral_per_plane_arrays_gray_f32(clip_grays):
    """A three-element array on a single-plane clip: only element 0 is used,
    and the result must equal the scalar-parameter run exactly."""
    scalar = _run(clip_grays, sigma_spatial=2.0, sigma_color=0.05)
    array = _run(clip_grays, sigma_spatial=[2.0, 1.5, 1.0], sigma_color=[0.05, 0.03, 0.02])
    for n in (0, 11):
        a = frame_to_ndarray(scalar.get_frame(n))
        b = frame_to_ndarray(array.get_frame(n))
        assert np.array_equal(a, b), f"extra array elements changed frame {n}"
    worst = _compare(
        "grays", (0, 11, 23), dict(sigma_spatial=[2.0, 1.5, 1.0], sigma_color=[0.05, 0.03, 0.02])
    )
    assert worst < REF_TOL, f"max diff {worst}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_spatial": [2.0, 1.0, 1.0]},  # the chroma rule
        {"sigma_spatial": [2.0, 1.5, 1.5], "sigma_color": [0.05, 0.02, 0.02]},
        {"sigma_spatial": [2.0, 1.0, 3.0], "sigma_color": [0.05, 0.02, 0.1], "radius": [2, 1, 3]},
        {"sigma_spatial": [1.0, 0.5, 0.5], "radius": [1, 1, 1]},
    ],
    ids=["chroma-rule", "spatial+color", "all-three", "radius-1"],
)
def test_bilateral_per_plane_arrays_match_reference_f16(clip_grayh, kwargs):
    """YUV420PH with explicit per-plane arrays must track vszipcl on all three
    planes (incl. the subsampled chroma lattice)."""
    worst = _compare("yuv420ph", (0, 11, 23), kwargs)
    assert worst <= F16_TOL, f"max diff {worst} ({kwargs})"


def test_bilateral_per_plane_arrays_gray_f16(clip_grayh):
    """f16 mirror: on a single-plane clip only element 0 is used and extra
    elements must not change the result."""
    scalar = _run(clip_grayh, sigma_spatial=2.0, sigma_color=0.05)
    array = _run(clip_grayh, sigma_spatial=[2.0, 1.5, 1.0], sigma_color=[0.05, 0.03, 0.02])
    for n in (0, 11):
        a = _plane(scalar.get_frame(n), 0, dtype=np.float16)
        b = _plane(array.get_frame(n), 0, dtype=np.float16)
        assert np.array_equal(a, b), f"extra array elements changed frame {n}"
    worst = _compare(
        "grayh", (0, 11, 23), dict(sigma_spatial=[2.0, 1.5, 1.0], sigma_color=[0.05, 0.03, 0.02])
    )
    assert worst <= F16_TOL, f"max diff {worst}"


# ---------------------------------------------------------------------------
# Per-plane processing mask (unprocessed planes share from the source)
# ---------------------------------------------------------------------------
#
# A zero sigma_spatial/sigma_color turns a plane off. On YUV that leaves the
# output allocated through newVideoFrame2 with the source chroma planes, so the
# unprocessed chroma must be passed through bit-identically while the luma is
# still filtered. No other test reaches that share path.


def test_bilateral_zero_chroma_sigma_shares_chroma_f32(clip_yuv420ps):
    src = clip_yuv420ps
    out = _run(src, sigma_spatial=[2.0, 0.0, 0.0])
    assert out.format.id == src.format.id
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = src.get_frame(n)
        assert not np.array_equal(
            _plane(f, 0, src.width, src.height), _plane(s, 0, src.width, src.height)
        ), f"luma was not filtered at frame {n}"
        for p in (1, 2):
            w, h = src.width >> 1, src.height >> 1
            assert np.array_equal(_plane(f, p, w, h), _plane(s, p, w, h)), (
                f"chroma{p} changed at frame {n}"
            )


def test_bilateral_zero_chroma_sigma_shares_chroma_u16(clip_yuv420p16):
    """u16 mirror of the share-path test (whole codes)."""
    src = clip_yuv420p16
    out = _run(src, sigma_spatial=[2.0, 0.0, 0.0])
    assert out.format.id == src.format.id
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = src.get_frame(n)
        d = _plane(f, 0, src.width, src.height, np.uint16).astype(np.int64) - _plane(
            s, 0, src.width, src.height, np.uint16
        ).astype(np.int64)
        assert np.abs(d).max() > 0, f"luma was not filtered at frame {n}"
        for p in (1, 2):
            w, h = src.width >> 1, src.height >> 1
            assert np.array_equal(_plane(f, p, w, h, np.uint16), _plane(s, p, w, h, np.uint16)), (
                f"chroma{p} changed at frame {n}"
            )


def test_bilateral_zero_chroma_sigma_shares_chroma_f16(clip_yuv420ph):
    """f16 mirror of the share path: half chroma must pass through
    bit-identically (not narrowed through f32 and back)."""
    src = clip_yuv420ph
    out = _run(src, sigma_spatial=[2.0, 0.0, 0.0])
    assert out.format.id == src.format.id
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = src.get_frame(n)
        assert not np.array_equal(_plane(f, 0, dtype=np.float16), _plane(s, 0, dtype=np.float16)), (
            f"luma was not filtered at frame {n}"
        )
        for p in (1, 2):
            assert np.array_equal(_plane(f, p, dtype=np.float16), _plane(s, p, dtype=np.float16)), (
                f"chroma{p} changed at frame {n}"
            )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_bilateral_rejects_negative_sigma(clip_grays):
    with pytest.raises(vs.Error):
        _run(clip_grays, sigma_spatial=-1.0)
    with pytest.raises(vs.Error):
        _run(clip_grays, sigma_color=-1.0)


def test_bilateral_rejects_nonpositive_radius(clip_grays):
    for bad in (0, -1):
        with pytest.raises(vs.Error):
            _run(clip_grays, radius=bad)


@pytest.mark.parametrize("bad", [32768, 46341, 2**31 - 1])
def test_bilateral_rejects_oversized_radius(clip_grays, bad):
    """A radius past 32767 wraps the signed-32-bit tile and distance math; it
    must be refused at creation. Only construction happens here, so a missing
    check fails instead of dispatching a kernel that never finishes.
    """
    with pytest.raises(vs.Error):
        vs.core.vsfeel.Bilateral(clip_grays, sigma_spatial=3.0, sigma_color=0.05, radius=bad)


def test_bilateral_rejects_sigma_spatial_that_derives_oversized_radius(clip_grays):
    """The auto-derived radius (``round(3*sigma_spatial)``, capped at 1e6) must
    hit the same bound as an explicit one."""
    with pytest.raises(vs.Error):
        vs.core.vsfeel.Bilateral(clip_grays, sigma_spatial=1e6, sigma_color=0.05)


def test_bilateral_parallel_load_matches_serial(clip_grays):
    """The pipelined multi-frame request path must match sequential requests."""
    params = dict(sigma_spatial=3.0, sigma_color=0.05)
    par = eval_parallel(_run, clip_grays, dtype=np.float32, **params)
    ref = cpu_node(_run(clip_grays, **params))
    for n in range(clip_grays.num_frames):
        seq = plane_to_ndarray(ref.get_frame(n), 0, np.float32)
        assert np.array_equal(par[n], seq), f"parallel/serial mismatch at frame {n}"
