"""Unit tests for core.vsfeel.WNNM.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is the input. The reference WNNM is float-only, so reference
comparisons run on f32 clips in a subprocess against the installed
``vszip.WNNM`` (the only implementation: vszipcl/vszipcu have no WNNM); the
native u16/f16 transport is graded against the same filter on the widened f32
clip, which is the chain a caller would otherwise write by hand.

The f32 port matches vszip to 2.9e-3 max on the default 640x360 GRAYS case
(0.09% of samples over 1e-5). A shrink decision that sits on the f32 rounding
boundary can flip: `block_size=4, group_size=4` is the only case here that
reaches 1.4e-2 (four samples over 1e-2), and every other case stays under
6e-3. At most 0.8% of the samples in a case sit over 1e-5. The normal ceiling
is therefore 8e-3, with a 1.5e-2 cap for that small-group flip and a 1% cap on
the fraction over 1e-5. A broken path moves far more samples (the old
group_size=33 barrier bug 97%, the win overrun 27%).

Self-consistency is separate: the scatter accumulates through buffer atomics, so
two runs differ by the float-atomic ordering floor (measured <= 3.6e-7); those
assertions use 1e-5 like BM3D's.

Run from the repository root:  uv run python -m pytest tests/test_wnnm.py
"""

import gc
import json
import os
import re
import subprocess
import sys
import weakref

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_all_frames_finite,
    assert_preserves_frame_props,
    assert_temporal_order_consistent,
    cpu_node,
    eval_parallel,
    max_diff,
    plane,
    reference_compare,
    reference_or_skip,
    reference_spec,
    source_clip,
)

pytestmark = pytest.mark.usefixtures("clip_grays")

# The eigen step is f32, so a shrink decision flips where an eigenvalue sits
# within ~1e-5 of the threshold. Only block_size=4/group_size=4 reaches 1.4e-2
# (four samples); every other case stays under 6e-3. REF_TOL is the normal
# ceiling, FLIP_TOL the small-group outlier, and REF_NDIFF catches a broken
# path, which moves a much larger fraction of samples.
REF_TOL = 8e-3
FLIP_TOL = 1.5e-2
REF_NDIFF = 0.01
# float atomic ordering floor: measured <= 3.6e-7 between two runs.
SELF_TOL = 1e-5
# Above SELCAP=32 the matcher feeds the group serially. Measured after the
# barrier was made uniform: <= 2.1e-3 max diff and <= 0.5% of samples over 1e-5
# on the 512x264 crop (1.5e-2 / 1.9e-2 and 97% before it).
SERIAL_TOL = 5e-3
SERIAL_NDIFF = 0.01


def WNNM(*args, **kwargs):
    """WNNM as a clip the test can read pixels from (see conftest.cpu_node)."""
    return cpu_node(vs.core.vsfeel.WNNM(*args, **kwargs))


def _ref_payload(
    fmt, frames, params, planes=None, guide=None, crop=None, nframes=None, diff_tol=None
):
    reference_or_skip("vszip", "WNNM")
    spec = reference_spec(
        "vszip",
        "WNNM",
        fmt,
        frames=frames,
        planes=planes,
        kwargs=params,
        guide=guide,
        guide_kwarg="rclip",
        crop=crop,
        diff_tol=diff_tol,
    )
    if nframes is not None:
        spec["nframes"] = int(nframes)
    return reference_compare(spec)


def _ref_compare(fmt, frames, params, planes=None, guide=None, crop=None, nframes=None):
    payload = _ref_payload(
        fmt,
        frames,
        params,
        planes=planes,
        guide=guide,
        crop=crop,
        nframes=nframes,
        diff_tol=1e-5,
    )
    assert payload["ndiff_frac"] <= REF_NDIFF, (
        f"{payload['ndiff_frac']:.2%} of samples differ (params {params})"
    )
    return payload["maxdiff"]


# ---------------------------------------------------------------------------
# Reference parity: spatial
# ---------------------------------------------------------------------------

SPATIAL_CASES = [
    dict(sigma=3),
    dict(sigma=1.5),
    dict(sigma=10),
    dict(sigma=3, group_size=4),
    dict(sigma=3, group_size=16, block_size=6, block_step=3),
    dict(sigma=3, block_size=4, block_step=4, group_size=4),
    # bs8/step3 exercises overlapping blocks; the same group contributes to
    # several pixels of one collaboration
    dict(sigma=3, block_step=3),
    dict(sigma=3, bm_range=2),
    # one candidate per window: every group is starved (n < group_size)
    dict(sigma=3, bm_range=0),
    dict(sigma=5, residual=True),
    dict(sigma=3, adaptive_aggregation=False),
    dict(sigma=3, block_size=16, block_step=16, group_size=16),
    # wide patch matrix (n > m): reconWide's geometry
    dict(sigma=3, block_size=4, block_step=4, group_size=32),
]


@pytest.mark.parametrize(
    "kwargs", SPATIAL_CASES, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items())
)
def test_spatial_matches_vszip(clip_grays, kwargs):
    worst = _ref_compare("grays", (0, 11, 23), kwargs)
    tol = FLIP_TOL if kwargs.get("block_size") == 4 and kwargs.get("group_size") == 4 else REF_TOL
    assert worst <= tol, f"max abs diff {worst:.3e} (params {kwargs})"


@pytest.mark.parametrize(
    "sigma",
    [
        pytest.param([3, 0, 5], id="middle-plane-skipped"),
        pytest.param([0, 5], id="luma-skipped"),
        pytest.param([3, 2, 2], id="all-planes"),
    ],
)
def test_per_plane_sigma_matches_vszip(sigma):
    """sigma is per plane: 0 skips a plane (copied through), others inherit."""
    worst = _ref_compare("yuv420ps", (0, 11, 23), dict(sigma=sigma))
    assert worst <= REF_TOL, f"max abs diff {worst:.3e}"


def test_rclip_matches_vszip(clip_grays):
    """Block matching on a blurred reference, patches from the source."""
    worst = _ref_compare(
        "grays", (0, 11, 23), dict(sigma=3), guide=dict(kind="boxblur", hradius=2, vradius=2)
    )
    assert worst <= REF_TOL, f"max abs diff {worst:.3e}"


def test_odd_geometry_matches_vszip(clip_grays):
    """Non-multiple dimensions clamp the last lattice row/column."""
    worst = _ref_compare("grays", (0, 11, 23), dict(sigma=3), crop={"left": 61, "top": 47})
    assert worst <= REF_TOL, f"max abs diff {worst:.3e}"


def test_starved_tiny_frame_matches_vszip(clip_grays):
    """13x7 with 4x4 blocks and one candidate per window: every group is starved."""
    crop = {"left": 13, "top": 7, "right": 614, "bottom": 346}
    worst = _ref_compare(
        "grays",
        (0, 11, 23),
        dict(sigma=3, block_size=4, block_step=4, group_size=8, bm_range=0),
        crop=crop,
    )
    assert worst <= REF_TOL, f"max abs diff {worst:.3e}"


@pytest.mark.parametrize("group_size", [32, 33, 64])
def test_serial_group_merge_matches_vszip(clip_grays, group_size):
    """group_size above SELCAP=32 uses the serial merge; every thread must reach
    its tile barrier, or the next tile overwrites `cand` under thread 0."""
    payload = _ref_payload(
        "grays",
        (0, 11, 23),
        dict(sigma=3, block_size=4, group_size=group_size),
        crop={"left": 64, "top": 48, "right": 64, "bottom": 48},
        diff_tol=1e-5,
    )
    assert payload["maxdiff"] <= SERIAL_TOL, (
        f"max abs diff {payload['maxdiff']:.3e} (group_size {group_size})"
    )
    assert payload["ndiff_frac"] <= SERIAL_NDIFF, (
        f"{payload['ndiff_frac']:.2%} of samples differ (group_size {group_size})"
    )


@pytest.mark.parametrize("block_size", [8, 9, 16, 32])
def test_split_residual_matches_fused(clip_grays, monkeypatch, block_size):
    """The split mean vector is M floats, not the fixed 64 that block_size=8
    needs; writing into the next group's Gram used to corrupt the frame."""
    params = dict(sigma=3, block_size=block_size, group_size=8, residual=True)
    split = WNNM(clip_grays, **params)
    monkeypatch.setenv("VSFEEL_WNNM_NOSPLIT", "1")
    fused = WNNM(clip_grays, **params)
    worst = max_diff(split, fused, frames=(0,))
    # block_size=9's uneven eight-lane columns flip one f32 shrink decision
    # (2.4e-3), the same split/fused spread the residual=0 path has.
    tol = 5e-3 if block_size == 9 else 1e-5
    assert worst <= tol, f"split vs fused {worst:.3e} (block_size {block_size})"


@pytest.mark.parametrize("block_size", [8, 9, 16, 32])
def test_residual_large_block_matches_vszip(block_size):
    """residual with block_size above 8 must match the reference, not just the
    fused path (the old split mean overran its 64-float slot)."""
    payload = _ref_payload(
        "grays",
        (0, 11, 23),
        dict(sigma=3, block_size=block_size, group_size=8, residual=True),
        diff_tol=1e-5,
    )
    assert payload["maxdiff"] <= REF_TOL, (
        f"max abs diff {payload['maxdiff']:.3e} (block_size {block_size})"
    )
    assert payload["ndiff_frac"] <= REF_NDIFF, (
        f"{payload['ndiff_frac']:.2%} of samples differ (block_size {block_size})"
    )


@pytest.mark.parametrize("bm_range", [7, 12, 13, 20])
def test_narrow_plane_window_matches_fused(clip_grays, monkeypatch, bm_range):
    """On a narrow plane the window's row stride is clamped by the width but
    its staged height by the height; bm_range 13+ needs more rows than
    win[1024] and used to run past the LDS."""
    crop = dict(left=100, top=50, right=508, bottom=110)  # 32x200
    narrow = vs.core.std.Crop(clip_grays, **crop)
    params = dict(sigma=3, block_size=8, group_size=8, bm_range=bm_range)
    split = WNNM(narrow, **params)
    monkeypatch.setenv("VSFEEL_WNNM_NOSPLIT", "1")
    fused = WNNM(narrow, **params)
    worst = max_diff(split, fused, frames=(0,))
    assert worst <= 1e-5, f"split vs fused {worst:.3e} (bm_range {bm_range})"


@pytest.mark.parametrize("bm_range", [12, 13, 20])
def test_narrow_plane_window_matches_vszip(bm_range):
    """bm_range=12 fills win[] exactly; 13 and 20 must take the global path."""
    payload = _ref_payload(
        "grays",
        (0,),
        dict(sigma=3, block_size=8, group_size=8, bm_range=bm_range),
        crop={"left": 100, "top": 50, "right": 508, "bottom": 110},
        diff_tol=1e-5,
    )
    assert payload["maxdiff"] <= 1e-5, (
        f"max abs diff {payload['maxdiff']:.3e} (bm_range {bm_range})"
    )
    assert payload["ndiff_frac"] <= REF_NDIFF, (
        f"{payload['ndiff_frac']:.2%} of samples differ (bm_range {bm_range})"
    )


# ---------------------------------------------------------------------------
# Reference parity: temporal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("radius", "frames"),
    [(1, (0, 1, 2)), (2, (0, 2)), (2, (0, 1, 2))],
)
def test_temporal_matches_vszip(clip_grays, radius, frames):
    """The VAggregate fold, including the clipped boundaries' duplicate slabs.

    A 3-frame clip makes every frame a boundary, which is where the reference's
    slab clamp reads some (frame, slab) pairs twice.
    """
    worst = _ref_compare(
        "grays", frames, dict(sigma=3, radius=radius), nframes=3, crop={"left": 64, "top": 48}
    )
    assert worst <= REF_TOL, f"max abs diff {worst:.3e} (radius {radius})"


def test_temporal_residual_matches_vszip(clip_grays):
    worst = _ref_compare(
        "grays",
        (1,),
        dict(sigma=3, radius=1, residual=True, block_size=6, block_step=3, group_size=16),
        nframes=3,
        crop={"left": 64, "top": 48},
    )
    assert worst <= REF_TOL, f"max abs diff {worst:.3e}"


def test_temporal_per_plane_matches_vszip():
    worst = _ref_compare(
        "yuv420ps", (1,), dict(sigma=[3, 2, 2], radius=1), nframes=3, crop={"left": 64, "top": 48}
    )
    assert worst <= REF_TOL, f"max abs diff {worst:.3e}"


# ---------------------------------------------------------------------------
# Per-plane radius
# ---------------------------------------------------------------------------
#
# The reference's radius is one value, so the oracle is the filter itself run
# one plane at a time: a plane's estimate depends only on its own pixels and
# window, which is what per-plane radius relies on. The atomic-ordering floor
# (SELF_TOL) is the only difference allowed between the two.

PER_PLANE_RADIUS_CASES = [
    pytest.param([3, 2, 3], [2, 0, 1], id="mixed-spatial-temporal"),
    pytest.param([3, 3, 3], [0, 3, 2], id="luma-spatial"),
    pytest.param([3, 2], [1, 2, 0], id="last-plane-spatial"),
    pytest.param([3, 0, 5], [2, 3, 0], id="unprocessed-plane"),
]


@pytest.mark.parametrize(("sigma", "radius"), PER_PLANE_RADIUS_CASES)
def test_per_plane_radius_matches_scalar_runs(clip_yuv420ps, sigma, radius):
    """Each plane must equal a single-plane run at that plane's radius."""
    src = vs.core.std.Crop(clip_yuv420ps, left=64, top=48)
    out = WNNM(src, sigma=sigma, radius=radius)
    for p in range(3):
        # A short list repeats its last entry, as sigma does.
        s = sigma[min(p, len(sigma) - 1)]
        r = radius[min(p, len(radius) - 1)]
        gray = vs.core.std.ShufflePlanes(src, p, vs.GRAY)
        ref = WNNM(gray, sigma=s, radius=r)
        for n in (0, 11, 23):
            a, b = plane(out.get_frame(n), p), plane(ref.get_frame(n), 0)
            worst = float(np.abs(a - b).max())
            assert worst < SELF_TOL, f"plane {p} (radius {r}) at frame {n}: {worst:.3e}"


def test_per_plane_radius_fused_matches_scalar_runs(clip_yuv420ps, monkeypatch):
    """The fused kernel takes each plane's own radius as well."""
    monkeypatch.setenv("VSFEEL_WNNM_NOSPLIT", "1")
    src = vs.core.std.Crop(clip_yuv420ps, left=64, top=48)
    sigma, radius = [3, 2, 2], [1, 0, 2]
    out = WNNM(src, sigma=sigma, radius=radius)
    for p in range(3):
        gray = vs.core.std.ShufflePlanes(src, p, vs.GRAY)
        ref = WNNM(gray, sigma=sigma[p], radius=radius[p])
        for n in (0, 23):
            a, b = plane(out.get_frame(n), p), plane(ref.get_frame(n), 0)
            worst = float(np.abs(a - b).max())
            assert worst < SELF_TOL, f"plane {p} at frame {n}: {worst:.3e}"


def test_per_plane_radius_repeats_last_entry(clip_yuv420ps):
    """A short radius list repeats its last entry, like sigma; a scalar still
    sets the first slot and repeats from there."""
    src = vs.core.std.Crop(clip_yuv420ps, left=64, top=48)
    full = WNNM(src, sigma=[3, 2, 2], radius=[2, 2, 2])
    for kwargs in (dict(radius=[2]), dict(radius=2)):
        out = WNNM(src, sigma=[3, 2, 2], **kwargs)
        worst = max_diff(out, full, frames=(0, 11, 23))
        assert worst < SELF_TOL, f"{kwargs}: {worst:.3e}"


@pytest.mark.parametrize("env", [None, {"VSFEEL_WNNM_NOCACHE": "1"}], ids=["cache", "nocache"])
def test_per_plane_radius_request_order(env, monkeypatch):
    """The per-plane match caches (and the per-frame slots without them) must
    not depend on frame-request order."""
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    assert_temporal_order_consistent(
        "WNNM", dict(sigma=[3, 3, 3], radius=[1, 0, 2]), tol=SELF_TOL, nframes=6, clip="yuv420ps"
    )


def test_u16_per_plane_radius_matches_the_float_chain(clip_yuv420p16):
    """The conversion ring stores each plane's own window; the mixed radii must
    still match the same filter on the widened clip to a code."""
    from vstools import depth

    src = vs.core.std.Crop(clip_yuv420p16, left=64, top=48)
    params = dict(sigma=[3, 2, 2], radius=[1, 0, 2])
    manual = depth(WNNM(depth(src, 32), **params), 16)
    native = WNNM(src, **params)
    for n in (0, 7, 23):
        worst = _code_diff(native.get_frame(n), manual.get_frame(n), (0, 1, 2))
        assert worst <= 1, f"frame {n}: {worst} codes apart"


@pytest.mark.parametrize("ps_num", [33, 64])
def test_temporal_large_ps_num_matches_vszip(clip_grays, ps_num):
    """ps_num above SELCAP=32 uses the serial temporal merge, which shares the
    same required-uniform barrier."""
    payload = _ref_payload(
        "grays",
        (1,),
        dict(sigma=3, radius=1, ps_num=ps_num),
        nframes=3,
        crop={"left": 64, "top": 48},
        diff_tol=1e-5,
    )
    assert payload["maxdiff"] <= SERIAL_TOL, (
        f"max abs diff {payload['maxdiff']:.3e} (ps_num {ps_num})"
    )
    assert payload["ndiff_frac"] <= SERIAL_NDIFF, (
        f"{payload['ndiff_frac']:.2%} of samples differ (ps_num {ps_num})"
    )


# ---------------------------------------------------------------------------
# Passthrough and unprocessed planes
# ---------------------------------------------------------------------------


def test_sigma_zero_is_passthrough(clip_grays):
    """sigma below FLT_EPSILON on every plane hands the input node back."""
    out = vs.core.vsfeel.WNNM(clip_grays, sigma=0)
    a = plane(clip_grays.get_frame(3), 0)
    b = plane(cpu_node(out).get_frame(3), 0)
    assert np.array_equal(a, b), "sigma=0 must return the source pixels untouched"


def test_unprocessed_plane_is_an_exact_copy():
    src = cpu_node(vs.core.fmtc.bitdepth(source_clip(), bits=32, fulls=True, fulld=True))
    src = vs.core.std.Crop(src, left=200, top=100, right=200, bottom=100)
    out = WNNM(src, sigma=[3, 0, 5])
    for n in (0, 7):
        a, b = src.get_frame(n), out.get_frame(n)
        assert np.array_equal(plane(a, 1), plane(b, 1)), f"unprocessed plane changed at frame {n}"
        assert not np.array_equal(plane(a, 0), plane(b, 0))


# ---------------------------------------------------------------------------
# Native 16-bit transport
# ---------------------------------------------------------------------------
#
# A u16 or f16 clip is widened into the f32 working domain and narrowed on the
# way out; the oracle is the same filter on the manually widened clip. The
# widening reproduces VapourSynth's `depth` bit for bit (multiply, then
# subtract the pre-scaled neutral, in f32), which is what makes the two chains
# comparable at the code level instead of a handful of shrink decisions apart.


def _code_diff(a_frame, b_frame, planes):
    return max(
        int(np.abs(plane(a_frame, p).astype(np.int64) - plane(b_frame, p).astype(np.int64)).max())
        for p in planes
    )


@pytest.mark.parametrize("full", [False, True], ids=["limited", "full"])
def test_u16_matches_the_float_chain(clip_yuv420p16, full):
    """Native u16 is `depth(WNNM(depth(clip, 32)), 16)` to the last code."""
    from vstools import depth

    clip = clip_yuv420p16
    if full:
        clip = vs.core.std.SetFrameProp(clip, prop="_Range", intval=1)
    manual = depth(WNNM(depth(clip, 32)), 16)
    native = WNNM(clip)
    for n in (0, 7, 23):
        worst = _code_diff(native.get_frame(n), manual.get_frame(n), (0, 1, 2))
        assert worst <= 1, f"frame {n}: {worst} codes apart (full={full})"


def test_u16_unprocessed_plane_is_an_exact_copy(clip_yuv420p16):
    """A sigma-0 plane rides through without a conversion round trip."""
    out = WNNM(clip_yuv420p16, sigma=[3, 0, 5])
    for n in (0, 7):
        a, b = clip_yuv420p16.get_frame(n), out.get_frame(n)
        assert np.array_equal(plane(a, 1), plane(b, 1)), f"plane 1 changed at frame {n}"


def test_u16_rclip_matches_the_float_chain(clip_yuv420p16):
    """The reference clip gets its own conversion slots."""
    from vstools import depth

    f32 = depth(clip_yuv420p16, 32)
    manual = depth(WNNM(f32, rclip=f32), 16)
    native = WNNM(clip_yuv420p16, rclip=clip_yuv420p16)
    for n in (0, 7):
        worst = _code_diff(native.get_frame(n), manual.get_frame(n), (1, 2))
        assert worst <= 1, f"frame {n}: {worst} codes apart"


def test_f16_matches_the_float_chain(clip_yuv420ph):
    """The half output is the f32 result rounded to half, to one half ulp."""
    from vstools import depth

    manual = WNNM(depth(clip_yuv420ph, 32))
    native = WNNM(clip_yuv420ph)
    for n in (0, 7, 23):
        a, b = native.get_frame(n), manual.get_frame(n)
        for p in (0, 1, 2):
            x = plane(a, p).astype(np.float32)
            y = plane(b, p).astype(np.float16).astype(np.float32)
            worst = float(np.abs(x - y).max())
            assert worst <= 1e-3, f"frame {n} plane {p}: {worst:.3e} apart"


@pytest.mark.parametrize(
    ("kind", "tol"),
    [("yuv420p16", 1.0), ("yuv420ph", 1e-3)],
)
def test_native_depth_request_order(kind, tol):
    """The conversion ring must not care which frame asked for what first."""
    assert_temporal_order_consistent("WNNM", dict(sigma=3, radius=1), tol=tol, nframes=6, clip=kind)


_COPY_COUNT_SCRIPT = r"""
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=64, height=48, format=vs.YUV420P16, length=1)
node = core.vsfeel.WNNM(clip, sigma=3, radius=15)
core.std.GPUDownload(node).get_frame(0)
"""


def test_u16_boundary_copies_are_deduplicated():
    """A one-frame clip's whole radius-15 window clamps to frame 0.

    Every processed plane must be converted once, not once per window position:
    the duplicate copies would write one ring slot before any tag is published.
    """
    env = {**os.environ, "MANGOHUD": "0", "VSFEEL_WNNM": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", _COPY_COUNT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert re.search(r"n=0: 3 conversions", proc.stderr), proc.stderr[-2000:]


def test_u16_max_radius_matches_the_float_chain(clip_yuv420p16):
    """Every window position of a short clip clamps to an existing frame."""
    from vstools import depth

    short = vs.core.std.Trim(clip_yuv420p16, length=3)
    manual = depth(WNNM(depth(short, 32), radius=15), 16)
    native = WNNM(short, radius=15)
    for n in range(3):
        worst = _code_diff(native.get_frame(n), manual.get_frame(n), (0, 1, 2))
        assert worst <= 1, f"frame {n}: {worst} codes apart"


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [dict(sigma=3), dict(sigma=3, radius=1), dict(sigma=3, residual=True)],
    ids=["spatial", "temporal", "residual"],
)
def test_deterministic(clip_grays, kwargs):
    a, b = WNNM(clip_grays, **kwargs), WNNM(clip_grays, **kwargs)
    frames = (0, 5, 11) if kwargs.get("radius") else (0, 11, 23)
    worst = max(
        float(np.abs(plane(a.get_frame(n), 0) - plane(b.get_frame(n), 0)).max()) for n in frames
    )
    assert worst < SELF_TOL, f"nondeterministic output ({worst:.3e})"


def test_parallel_load_matches_serial(clip_grays):
    """Every frame requested at once must match the serial run."""
    serial = WNNM(clip_grays, sigma=3)
    expected = [plane(serial.get_frame(n), 0) for n in range(serial.num_frames)]
    got = eval_parallel(lambda clip, **kw: vs.core.vsfeel.WNNM(clip, **kw), clip_grays, sigma=3)
    worst = max(float(np.abs(a - b).max()) for a, b in zip(expected, got))
    assert worst < SELF_TOL, f"parallel/serial disagreement ({worst:.3e})"


def _parallel_vs_serial(clip, **kwargs):
    serial = WNNM(clip, **kwargs)
    expected = [plane(serial.get_frame(n), 0) for n in range(serial.num_frames)]
    got = eval_parallel(lambda c, **kw: vs.core.vsfeel.WNNM(c, **kw), clip, **kwargs)
    return max(
        float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        for a, b in zip(expected, got)
    )


@pytest.mark.parametrize(
    ("fixture", "tol"),
    [("clip_yuv420p16", 1.0), ("clip_yuv420ph", 1e-3)],
)
def test_native_depth_parallel_load_matches_serial(request, fixture, tol):
    """A 16-bit ring slot can be overwritten by a later frame; requesting every
    frame at once must still reproduce the serial run. One code / one half ulp
    is the atomic-ordering floor; a stale slot is far larger."""
    clip = request.getfixturevalue(fixture)
    worst = _parallel_vs_serial(clip, sigma=3)
    assert worst <= tol, f"parallel/serial disagreement ({worst:.3e})"


def test_native_depth_nocache_parallel_load(monkeypatch, clip_yuv420p16):
    """The temporal path without the match cache still needs the barrier."""
    monkeypatch.setenv("VSFEEL_WNNM_NOCACHE", "1")
    worst = _parallel_vs_serial(clip_yuv420p16, sigma=3, radius=1)
    assert worst <= 1.0, f"parallel/serial disagreement ({worst:.3e})"


def test_estimate_cache_matches_nocache(clip_grays, monkeypatch):
    """The cached solution M must equal the recomputed one on the same frame."""
    cached = WNNM(clip_grays, sigma=3, radius=1)
    monkeypatch.setenv("VSFEEL_WNNM_NOCACHE", "1")
    plain = WNNM(clip_grays, sigma=3, radius=1)
    worst = max_diff(cached, plain, frames=(0, 1, 2))
    assert worst < SELF_TOL, f"cached vs recomputed {worst:.3e}"


def test_ring_allocation_failure_matches_nocache(clip_grays, monkeypatch):
    """A failed match-ring allocation must land on the uncached layout.

    The ring is allocated before the pipelines are specialized, so the fallback
    rebuilds the layout they compile against; a later fallback would leave them
    indexing ring slots while the host binds per-frame storage.
    """
    monkeypatch.setenv("VSFEEL_WNNM_NOCACHE", "1")
    plain = WNNM(clip_grays, sigma=3, radius=1)
    monkeypatch.delenv("VSFEEL_WNNM_NOCACHE")
    monkeypatch.setenv("VSFEEL_WNNM_NORING", "1")
    forced = WNNM(clip_grays, sigma=3, radius=1)
    worst = max_diff(plain, forced, frames=(0, 1, 2))
    assert worst < SELF_TOL, f"forced ring fallback vs nocache {worst:.3e}"


@pytest.mark.parametrize(
    "params",
    [dict(sigma=3), dict(sigma=3, radius=1)],
    ids=["spatial", "temporal"],
)
def test_request_order_consistent(params):
    assert_temporal_order_consistent("WNNM", params, tol=SELF_TOL, nframes=6)


def test_all_frames_finite(clip_grays):
    out = WNNM(clip_grays, sigma=3, radius=1)
    assert_all_frames_finite(out, frames=range(0, out.num_frames, 5))


def test_frame_properties_preserved(clip_grays):
    assert_preserves_frame_props(vs.core.vsfeel.WNNM, clip_grays, sigma=3)


def test_gpu_output_chains(clip_grays):
    """The output is GPU resident and feeds another GPU filter directly."""
    first = vs.core.vsfeel.WNNM(clip_grays, sigma=3)
    assert first.gpu_resident
    second = vs.core.vsfeel.WNNM(first, sigma=3)
    assert second.gpu_resident
    got = plane(cpu_node(second).get_frame(0), 0)
    assert np.isfinite(got).all()


def test_cropped_frame_stride(clip_grays):
    """A cropped clip's plane keeps the parent's wide stride; the kernel's
    plane addressing is STRIDE elements per row, so it has to be read from the
    frame, not assumed equal to the width."""
    src = vs.core.std.Crop(clip_grays, left=200, top=100, right=100, bottom=100)
    assert src.get_frame(0).get_stride(0) // 4 > src.width
    out = WNNM(src, sigma=3, block_size=8)
    got = plane(out.get_frame(0), 0)
    assert got.shape == (src.height, src.width)
    assert np.isfinite(got).all()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_DUMP_SCRIPT = r"""
import json
import sys

import vapoursynth as vs

core = vs.core
src = core.bs.VideoSource(sys.argv[3])
src = core.std.Crop(src, left=0, top=0, right=608, bottom=328)  # 32x32
clip = core.std.ShufflePlanes(src, 0, vs.GRAY)
clip = core.fmtc.bitdepth(clip, bits=32, fulls=True, fulld=True)
clip = core.std.Trim(clip, length=3)
node = core.vsfeel.WNNM(clip, **json.loads(sys.argv[1]))
out = core.std.GPUDownload(clip=node)
for n in range(int(sys.argv[2]) + 1):
    out.get_frame(n)
"""

_DUMP_CLIP = os.path.join(os.path.dirname(__file__), "bigbuckbunny_360p_grain.mp4")


def _dump_records(params, frame, env=None):
    """Parse VSFEEL_WNNM_DUMP for one frame: (p, gx, gy, td) -> fields."""
    run_env = {
        **os.environ,
        "MANGOHUD": "0",
        "VSFEEL_DEBUG": "1",
        "VSFEEL_WNNM_DUMP": str(frame),
        **(env or {}),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _DUMP_SCRIPT, json.dumps(params), str(frame), _DUMP_CLIP],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    records = {}
    current = None
    for line in proc.stderr.splitlines():
        head = re.match(
            r"\[wnnm-dump\] p=(\d+) gx=(\d+) gy=(\d+) td=(\d+) n=(\d+) "
            r"kept=(\d+) w=(\S+) nl=(\d+)$",
            line,
        )
        if head:
            key = tuple(int(x) for x in head.groups()[:4])
            current = {
                "n": int(head.group(5)),
                "kept": int(head.group(6)),
                "w": float(head.group(7)),
                "nl": int(head.group(8)),
                "top": [],
            }
            records[key] = current
            continue
        item = re.match(r"\[wnnm-dump\]\s+j=(\d+) err=(\S+) xy=0x([0-9a-f]+) t=(\d+)$", line)
        if item and current is not None:
            current["top"].append(
                (float(item.group(2)), int(item.group(3), 16), int(item.group(4)))
            )
    return records


def _check_dump_record(rec, block_size, group_size, size=32):
    assert rec["n"] == len(rec["top"]), rec
    assert 1 <= rec["n"] <= group_size, rec
    assert rec["kept"] <= min(block_size * block_size, rec["n"]), rec
    if rec["kept"]:
        assert rec["w"] * rec["kept"] == pytest.approx(1.0), rec
    else:
        assert rec["w"] == 1.0, rec
    errors = [entry[0] for entry in rec["top"]]
    assert errors == sorted(errors), rec
    for _, xy, _ in rec["top"]:
        assert 0 <= (xy & 0xFFFF) < size and 0 <= (xy >> 16) < size, rec


def test_dump_split_spatial_matches_fused():
    """The split matcher's trace must carry the same selection as the fused
    kernel, and the split solve's kept/weight must be filled."""
    params = dict(sigma=3, block_size=4, block_step=4, group_size=4, bm_range=2)
    split = _dump_records(params, 0)
    fused = _dump_records(params, 0, {"VSFEEL_WNNM_NOSPLIT": "1"})
    assert split
    assert set(split) == set(fused)
    assert {key[3] for key in split} == {0}
    for key, rec in split.items():
        _check_dump_record(rec, 4, 4)
        assert rec["nl"] == 0
        assert rec == fused[key], key


def test_dump_temporal_cache_matches_nocache():
    """A cached fold was not searched, so it is not dumped; the searched fold's
    record must match the no-cache run."""
    params = dict(sigma=3, radius=1, block_size=4, group_size=4, ps_num=2)
    cached = _dump_records(params, 1)
    all_folds = _dump_records(params, 1, {"VSFEEL_WNNM_NOCACHE": "1"})
    assert cached
    assert set(cached) < set(all_folds)
    assert len({key[3] for key in cached}) < 3
    for key, rec in cached.items():
        _check_dump_record(rec, 4, 4)
        assert rec["nl"] > 0
        assert rec == all_folds[key], key


# ---------------------------------------------------------------------------
# Creation-time variants
# ---------------------------------------------------------------------------

_CREATE_SCRIPT = r"""
import json
import sys

import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=640, height=360, format=vs.GRAYS, length=2)
node = core.vsfeel.WNNM(clip, **json.loads(sys.argv[1]))
core.std.GPUDownload(clip=node).get_frame(0)
"""


def _creation_stderr(args_expr, env=None):
    """Stderr of a process that creates one WNNM instance.

    The QMAX variant and the eigen precision are invisible in the pixels (a
    silent fallback passes every correctness test), so the creation banner is
    where they are pinned.
    """
    run_env = {**os.environ, "MANGOHUD": "0", "VSFEEL_DEBUG": "1", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _CREATE_SCRIPT, json.dumps(eval(args_expr))],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stderr


@pytest.mark.parametrize(
    ("kwargs", "qmax"),
    [
        ("dict(sigma=3)", 8),
        ("dict(sigma=3, group_size=16)", 16),
        ("dict(sigma=3, block_size=8, group_size=32)", 32),
    ],
)
def test_qmax_variant_pinned(kwargs, qmax):
    err = _creation_stderr(kwargs)
    match = re.search(r"qmax=(\d+)", err)
    assert match, err[-2000:]
    assert int(match.group(1)) == qmax, err[-2000:]


_CACHE_LIMIT_SCRIPT = r"""
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=1920, height=1080, format=vs.YUV420PS, length=24)
core.vsfeel.WNNM(clip, sigma=[3, 0], radius=1, ps_num=1, ps_range=0)
"""


def test_cache_budget_respects_storage_range():
    """A ring above maxStorageBufferRange must halve, not drop the cache."""
    cap = 70 << 20  # the luma-only 1080p ring is 134.5 MiB unbounded here
    env = {
        **os.environ,
        "MANGOHUD": "0",
        "VSFEEL_WNNM": "1",
        "VSFEEL_LIMIT_STORAGE_RANGE": str(cap),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _CACHE_LIMIT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    match = re.search(r"cache=(\d+) ring=(\d+) ring=([\d.]+) MiB", proc.stderr)
    assert match, proc.stderr[-2000:]
    cache, ring, mib = int(match.group(1)), int(match.group(2)), float(match.group(3))
    assert cache == 1, proc.stderr[-2000:]
    assert ring <= 32, proc.stderr[-2000:]
    assert mib <= cap / (1 << 20), proc.stderr[-2000:]


_PER_BUFFER_LIMIT_SCRIPT = r"""
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAYS, length=24)
core.vsfeel.WNNM(clip, sigma=3, radius=1)
"""


def test_storage_range_is_per_scratch_buffer():
    """maxStorageBufferRange caps one buffer, not the per-frame scratch sum.

    1080p luma, radius=1 at the defaults: the scratch sum is ~126 MiB but the
    largest single buffer (locs) is ~60 MiB, so a 70 MiB range must be
    accepted.
    """
    cap = 70 << 20
    env = {
        **os.environ,
        "MANGOHUD": "0",
        "VSFEEL_WNNM": "1",
        "VSFEEL_LIMIT_STORAGE_RANGE": str(cap),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PER_BUFFER_LIMIT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    banner = re.search(r"acc=([\d.]+) MiB pmat=([\d.]+) MiB locs=([\d.]+) MiB", proc.stderr)
    assert banner, proc.stderr[-2000:]
    acc, pmat, locs = (float(v) for v in banner.groups())
    assert max(acc, pmat, locs) < cap / (1 << 20) < acc + pmat + locs, banner.group(0)
    assert re.search(r"cache=1 ring=\d+", proc.stderr), proc.stderr[-2000:]


@pytest.mark.parametrize(
    "knob",
    [
        {"VSFEEL_LIMIT_SUBGROUP_OPS": "0"},
        {"VSFEEL_LIMIT_SUBGROUP_SIZE": "4"},
    ],
    ids=["no-shuffle", "narrow-subgroup"],
)
def test_split_falls_back_without_eight_lane_shuffle(knob):
    """The split Gram groups lanes by subgroup ID and XOR-reduces them.

    Without shuffle, or with subgroups narrower than one eight-lane group, the
    fused kernel must run instead of a reduction that crosses subgroup
    boundaries. The banner's `split` field is the only place the choice shows.
    """
    assert "split=1" in _creation_stderr("dict(sigma=3)"), "split path not default"
    err = _creation_stderr("dict(sigma=3)", knob)
    assert "split=0" in err, err[-2000:]
    assert "fused kernel" in err, err[-2000:]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _blank():
    return vs.core.std.BlankClip(width=64, height=48, format=vs.GRAYS, length=1)


_FIRST_PIPELINE_FAILURE_SCRIPT = r"""
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=64, height=48, format=vs.GRAYS, length=2)
try:
    core.vsfeel.WNNM(clip, sigma=3)
except vs.Error as e:
    print("ERROR:", e, flush=True)
"""


def test_pipeline_failure_stops_creation_at_the_first_error():
    """A failed pipeline must release the input nodes once and return.

    The 64-invocation cap makes the 128-lane Gram fail before solve (256) and
    apply (128); without an early return those failures call set_error again,
    which frees the already-freed input nodes.
    """
    env = {
        **os.environ,
        "MANGOHUD": "0",
        "VSFEEL_DEBUG": "1",
        "VSFEEL_LIMIT_INVOCATIONS": "64",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _FIRST_PIPELINE_FAILURE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ERROR:" in proc.stdout, proc.stderr[-2000:]
    assert "gram" in proc.stdout, proc.stdout[-2000:]
    assert "wnnm group p0 solve" not in proc.stderr, "creation continued past the first failure"


@pytest.mark.parametrize(
    ("fmt", "msg"),
    [
        (vs.GRAY8, "only 16 bit integer"),
        (vs.YUV420P8, "only 16 bit integer"),
        (vs.RGB24, "only 16 bit integer"),
    ],
    ids=lambda f: vs.PresetVideoFormat(f).name if isinstance(f, int) else None,
)
def test_8bit_format_rejected(fmt, msg):
    src = vs.core.std.BlankClip(width=64, height=48, format=fmt, length=1)
    with pytest.raises(vs.Error, match=msg):
        vs.core.vsfeel.WNNM(src)


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        (dict(block_size=0), r'"block_size" must be in \[1, 64\]'),
        (dict(block_size=65), r'"block_size" must be in \[1, 64\]'),
        (
            dict(block_size=50),
            r'"block_size" must not exceed the dimensions of any processed plane',
        ),
        (
            dict(block_size=16, group_size=64),
            r"min\(block_size\^2, group_size\) must be <= 32",
        ),
        (
            dict(block_size=8, block_step=9),
            r'"block_step" must be positive and no larger than "block_size"',
        ),
        (dict(block_step=0), r'"block_step" must be positive and no larger than'),
        (dict(sigma=-1), r'"sigma" must be finite and non-negative'),
        (dict(group_size=0), r'"group_size" must be in \[1, 64\]'),
        (dict(group_size=65), r'"group_size" must be in \[1, 64\]'),
        (dict(group_size=256), r'"group_size" must be in \[1, 64\]'),
        (dict(bm_range=-1), r'"bm_range" must be in \[0, 4096\]'),
        (dict(bm_range=4097), r'"bm_range" must be in \[0, 4096\]'),
        (dict(bm_range=2147483647), r'"bm_range" must be in \[0, 4096\]'),
        (dict(radius=16), r'"radius" must be in \[0, 15\]'),
        (dict(radius=-1), r'"radius" must be in \[0, 15\]'),
        (dict(radius=[0, 16]), r'"radius" must be in \[0, 15\]'),
        (dict(ps_num=0), r'"ps_num" must be in \[1, 64\]'),
        (dict(ps_num=65), r'"ps_num" must be in \[1, 64\]'),
        (dict(ps_range=-1), r'"ps_range" must be in \[0, 4096\]'),
        (dict(ps_range=4097), r'"ps_range" must be in \[0, 4096\]'),
        (dict(ps_range=2147483647), r'"ps_range" must be in \[0, 4096\]'),
    ],
)
def test_validation_errors(kwargs, msg):
    with pytest.raises(vs.Error, match=msg):
        vs.core.vsfeel.WNNM(_blank(), **kwargs)


def test_parameter_caps_are_inclusive():
    """The list caps (64) and the search-radius cap (4096) accept their edge."""
    vs.core.vsfeel.WNNM(_blank(), block_size=4, group_size=64, ps_num=64)
    vs.core.vsfeel.WNNM(_blank(), bm_range=4096, ps_range=4096)


def test_block_size_checked_per_processed_plane():
    """YUV420PS 64x48 has 32x24 chroma: bs=32 fits luma only."""
    src = vs.core.std.BlankClip(width=64, height=48, format=vs.YUV420PS, length=1)
    with pytest.raises(vs.Error, match=r'"block_size" must not exceed the dimensions'):
        vs.core.vsfeel.WNNM(src, block_size=32)
    vs.core.vsfeel.WNNM(src, block_size=32, sigma=[3, 0])


def test_rclip_mismatch_rejected():
    a = vs.core.std.BlankClip(width=64, height=48, format=vs.GRAYS, length=1)
    b = vs.core.std.BlankClip(width=48, height=48, format=vs.GRAYS, length=1)
    with pytest.raises(vs.Error, match=r'"rclip" must be of the same format'):
        vs.core.vsfeel.WNNM(a, rclip=b)
    two = vs.core.std.BlankClip(width=64, height=48, format=vs.GRAYS, length=2)
    with pytest.raises(vs.Error, match=r'"rclip" must be of the same format'):
        vs.core.vsfeel.WNNM(a, rclip=two)
    with pytest.raises(vs.Error, match=r'"rclip" must be of the same format'):
        vs.core.vsfeel.WNNM(
            a, rclip=vs.core.std.BlankClip(width=64, height=48, format=vs.GRAY16, length=1)
        )


def test_short_clip_with_max_radius_reads_only_existing_frames(clip_grays):
    """radius=15 on a 3-frame clip must clamp every window, not read past the end."""
    short = vs.core.std.Trim(clip_grays, length=3)
    out = WNNM(short, sigma=[3.0], radius=15)
    assert out.num_frames == 3
    for n in range(3):
        a = plane(out.get_frame(n), 0, dtype=np.float32)
        assert np.isfinite(a).all(), f"non-finite frame {n}"


def test_filter_releases_its_input_nodes():
    """Freeing the filter must release the `clip` and `rclip` references.

    Both upstreams are Python filters, so a leaked node reference keeps their
    callables alive past the filter's destruction; that is what the weakrefs
    observe.
    """

    class Marker:
        def __call__(self, f, **kw):
            return f

    def wrap(clip):
        marker = Marker()
        node = vs.core.std.ModifyFrame(clip, [clip], marker)
        return node, weakref.ref(marker)

    clip, clip_ref = wrap(_blank())
    rclip, rclip_ref = wrap(_blank())
    out = vs.core.vsfeel.WNNM(clip, rclip=rclip)
    out.get_frame(0)
    del clip, rclip
    gc.collect()
    assert clip_ref() is not None and rclip_ref() is not None

    del out
    vs.core.clear_cache()
    gc.collect()
    assert clip_ref() is None, "WNNM retained its `clip` node"
    assert rclip_ref() is None, "WNNM retained its `rclip` node"
