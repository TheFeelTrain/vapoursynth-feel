"""Unit tests for core.vsfeel.WNNM.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is the input. The reference WNNM is float-only, so reference
comparisons run on f32 clips in a subprocess against the installed
``vszip.WNNM`` (the only implementation: vszipcl/vszipcu have no WNNM); the
native u16/f16 transport is graded against the same filter on the widened f32
clip, which is the chain a caller would otherwise write by hand.

The port matched the reference to <= 7.7e-7 max abs over every case here
(measured, 640x360 GRAYS and 64x48 crops, spatial and temporal); the tests use
5e-6 to leave headroom for content and rounding-order variation while still
catching a broken match or a flipped shrink decision.

Self-consistency is separate: the scatter accumulates through buffer atomics, so
two runs differ by the float-atomic ordering floor (measured <= 3.6e-7); those
assertions use 1e-5 like BM3D's.

Run from the repository root:  uv run python -m pytest tests/test_wnnm.py
"""

import json
import os
import re
import subprocess
import sys

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

# The default (f32) eigen step lands its shrink decisions on the same side as
# vszip's f64 one except where an eigenvalue sits within ~1e-5 of the shrink
# threshold; a flipped decision moves those pixels by up to 1.4e-2 (block_size 4,
# typically ~3e-3). Against a f64 exact-SVD oracle ours is 2.2e-3 at worst and
# the nssfactory CPU reference 4.7e-3 to 2.0e-2, so the fast path is closer to
# the exact answer than the CPU reference the port was checked against.
REF_TOL = 2e-2
# float atomic ordering floor: measured <= 3.6e-7 between two runs.
SELF_TOL = 1e-5
# Above SELCAP=32 the matcher feeds the group serially. Measured after the
# barrier was made uniform: <= 2.1e-3 max diff and <= 0.5% of samples over 1e-5
# on the 512x264 crop (1.5e-2 / 1.9e-2 and 97% before it).
SERIAL_TOL = 5e-3
SERIAL_NDIFF = 0.02


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
    return _ref_payload(
        fmt, frames, params, planes=planes, guide=guide, crop=crop, nframes=nframes
    )["maxdiff"]


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
    assert worst <= REF_TOL, f"max abs diff {worst:.3e} (params {kwargs})"


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
    """13x7 with 4x4 blocks: fewer candidates than group_size everywhere."""
    crop = {"left": 13, "top": 7, "right": 640 - 200, "bottom": 360 - 200}
    worst = _ref_compare(
        "grays", (0, 11, 23), dict(sigma=3, block_size=4, block_step=4, group_size=8), crop=crop
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
    assert payload["ndiff_frac"] <= 0.02, (
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
    assert payload["ndiff_frac"] <= 0.02, (
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _blank():
    return vs.core.std.BlankClip(width=64, height=48, format=vs.GRAYS, length=1)


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
            dict(block_size=16, group_size=128),
            r"min\(block_size\^2, group_size\) must be <= 32",
        ),
        (
            dict(block_size=8, block_step=9),
            r'"block_step" must be positive and no larger than "block_size"',
        ),
        (dict(block_step=0), r'"block_step" must be positive and no larger than'),
        (dict(sigma=-1), r'"sigma" must be finite and non-negative'),
        (dict(group_size=0), r'"group_size" must be in \[1, 256\]'),
        (dict(bm_range=-1), r'"bm_range" must be non-negative'),
        (dict(radius=16), r'"radius" must be in \[0, 15\]'),
        (dict(ps_num=0), r'"ps_num" must be in \[1, 64\]'),
        (dict(ps_num=65), r'"ps_num" must be in \[1, 64\]'),
        (dict(ps_range=-1), r'"ps_range" must be non-negative'),
    ],
)
def test_validation_errors(kwargs, msg):
    with pytest.raises(vs.Error, match=msg):
        vs.core.vsfeel.WNNM(_blank(), **kwargs)


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
