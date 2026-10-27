"""Unit tests for core.vsfeel.Box.

The oracle is ``vskernels.Box`` (resize2's ``resize2.Custom`` with the box
kernel), which is also what ``vsaa.based_aa`` uses for an integer ``rfactor``.
Every depth (u16, f16, f32) and every color family are compared on the committed
bigbuckbunny_360p_grain.mp4 clip, whose grain exercises every tap window.

Run from the repository root:  uv run python -m pytest tests/test_box.py
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_preserves_frame_props,
    cpu_node,
    eval_parallel,
    plane_to_ndarray,
    plane as _plane,
)

pytest.importorskip("vskernels")
from vskernels import Box

# Measured agreement with resize2, per downscale ratio, on the 640x360 fixture:
#
#   u16: bit-exact whenever every tap count is a power of two, which is also
#        every fractional ratio that only ever pairs one or two source pixels.
#        Where resize2's own float32 quantisation of 1/count is inexact (the tap
#        counts 3, 5, 6, 7) a handful of codes move, never more than 2.
#   f32: within 1-2 ulp of the peak everywhere, and bit-exact for most ratios.
#
# (ratio, u16 bound in whole codes, f32 bound as a fraction of the peak)
RATIOS = [
    (2, 0, 0.0),
    (4, 0, 1.2e-7),
    (8, 0, 1.2e-7),
    (1.5, 0, 0.0),
    (1.25, 0, 0.0),
    (1.75, 0, 0.0),
    (1.1, 0, 0.0),
    (4 / 3, 0, 0.0),
    (3, 2, 2.4e-7),
    (2.5, 2, 2.4e-7),
    (3.5, 1, 2.4e-7),
    (5, 2, 2.4e-7),
    (6, 1, 1.8e-7),
]

# Subsampled formats need even chroma planes, so their fractional ratios are
# limited to the ones that keep the lattice legal (resize2 rejects the rest).
COLOR_CASES = [
    ("yuv420p16", 2),
    ("yuv420p16", 4),
    ("yuv420ps", 2),
    ("yuv420ps", 4),
    ("yuv444p16", 2),
    ("yuv444p16", 4),
    ("yuv444p16", 1.5),
    ("rgb48", 2),
    ("rgb48", 4),
    ("rgb48", 1.5),
    ("rgbs", 2),
    ("rgbs", 4),
    ("rgbs", 1.5),
]

# Half is a storage width, not a precision: the f16 path reads and writes the
# clip's own 2-byte samples, every accumulator is f32, and the two passes round
# the intermediate and the output to half (OpFConvert, round-to-nearest-even,
# the reference's vstore_half_rte). Measured against resize2 on the GRAYH
# fixture, every plane, all 24 frames: bit-exact at 2, 4, 1.5, 1.25, 1.75, 1.1
# and 4/3, and one half step (4.88e-4 at the unit peak) at 3, 2.5, 3.5, 5 (our
# f32 divide can flip a half tie) and at 6, 8 (resize2's own f16 tap window
# straddles two 8-tap blocks).
F16_STEP = 4.8828125e-4  # one half step at the unit peak

# (ratio, worst absolute difference)
F16_RATIOS = [
    (2, 0.0),
    (4, 0.0),
    (8, F16_STEP),
    (1.5, 0.0),
    (1.25, 0.0),
    (1.75, 0.0),
    (1.1, 0.0),
    (4 / 3, 0.0),
    (3, F16_STEP),
    (2.5, F16_STEP),
    (3.5, F16_STEP),
    (5, F16_STEP),
    (6, F16_STEP),
]

# Half mirrors of COLOR_CASES, on the fixtures built from their f32 siblings.
COLOR_CASES_F16 = [
    ("yuv420ph", 2),
    ("yuv420ph", 4),
    ("yuv444ph", 2),
    ("yuv444ph", 4),
    ("yuv444ph", 1.5),
    ("rgbh", 2),
    ("rgbh", 4),
    ("rgbh", 1.5),
]


def _px(node, p=0, n=0):
    """One plane of one frame as int64.

    Widened deliberately: the u16 operands are whole integer codes and
    subtracting two uint16 arrays wraps (``0 - 1 == 65535``), which would turn a
    one-code difference into a full-scale one.
    """
    return _plane(cpu_node(node).get_frame(n), p).astype(np.int64)


def _box(clip, width, height):
    return vs.core.vsfeel.Box(clip, width=width, height=height)


def _ref(clip, width, height):
    return Box().scale(clip, width, height, (0, 0))


def _worst(clip, width, height, planes):
    """(worst absolute, worst peak-relative) difference over `planes`."""
    worst_abs = worst_rel = 0.0
    for p in planes:
        ref, out = _px(_ref(clip, width, height), p), _px(_box(clip, width, height), p)
        d = float(np.abs(ref - out).max())
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, d / max(1.0, float(np.abs(ref).max())))
    return worst_abs, worst_rel


def _target(clip, ratio):
    return round(clip.width / ratio), round(clip.height / ratio)


def _pxf(node, p=0, n=0):
    """One half plane of one frame as float64, read in its native dtype."""
    return _plane(cpu_node(node).get_frame(n), p, dtype=np.float16).astype(np.float64)


def _worst_f16(clip, width, height, planes):
    """Worst absolute difference over `planes` on the half path."""
    worst = 0.0
    for p in planes:
        ref = _pxf(_ref(clip, width, height), p)
        out = _pxf(_box(clip, width, height), p)
        worst = max(worst, float(np.abs(ref - out).max()))
    return worst


def _transfers(build):
    """The implicit transfers the core inserted while building this graph.

    The core logs them when the consuming node is *created*, so the handler has
    to wrap the construction, not the frame pull.
    """
    messages: list[str] = []
    handle = vs.core.add_log_handler(lambda level, msg: messages.append(msg))
    try:
        node = build()
    finally:
        vs.core.remove_log_handler(handle)
    return node, [m for m in messages if "automatically inserted" in m]


# ---------------------------------------------------------------------------
# Reference comparison: u16 and f32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,abs_bound,rel_bound", RATIOS)
def test_agrees_with_resize2_u16(clip_gray16, ratio, abs_bound, rel_bound):
    """The integer path reproduces resize2's Box within the measured bound."""
    w, h = _target(clip_gray16, ratio)
    abs_d, _ = _worst(clip_gray16, w, h, [0])
    assert abs_d <= abs_bound, f"{ratio}: max|d|={abs_d} > {abs_bound}"


@pytest.mark.parametrize("ratio,abs_bound,rel_bound", RATIOS)
def test_agrees_with_resize2_float(clip_grays, ratio, abs_bound, rel_bound):
    """The float path reproduces resize2's Box to float32 rounding."""
    w, h = _target(clip_grays, ratio)
    _, rel_d = _worst(clip_grays, w, h, [0])
    assert rel_d <= max(rel_bound, 1e-6), f"{ratio}: rel={rel_d} > {rel_bound}"


@pytest.mark.parametrize("fmt,ratio", COLOR_CASES)
def test_agrees_with_resize2_color(request, fmt, ratio):
    """Every plane agrees, with the subsampled planes on their own lattice."""
    clip = request.getfixturevalue(f"clip_{fmt}")
    w, h = _target(clip, ratio)
    out = _box(clip, w, h)
    assert out.format.num_planes == clip.format.num_planes
    assert (out.width, out.height) == (w, h)
    planes = list(range(clip.format.num_planes))
    if clip.format.sample_type == vs.INTEGER:
        abs_d, _ = _worst(clip, w, h, planes)
        assert abs_d <= 2, f"max|d|={abs_d}"
    else:
        _, rel_d = _worst(clip, w, h, planes)
        assert rel_d <= 2.4e-7, f"rel={rel_d}"


def test_chroma_planes_are_scaled(clip_yuv420p16):
    """A subsampled plane is averaged on its own lattice, not the luma's."""
    w, h = clip_yuv420p16.width // 2, clip_yuv420p16.height // 2
    out = cpu_node(_box(clip_yuv420p16, w, h))
    f = out.get_frame(0)
    assert (f.width, f.height) == (w, h)
    assert f.format.num_planes == 3
    # 640x360 -> 320x180 means chroma 320x180 -> 160x90
    assert _px(out, p=1).shape == (90, 160)
    assert _px(out, p=2).shape == (90, 160)


# ---------------------------------------------------------------------------
# Reference comparison: f16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,bound", F16_RATIOS)
def test_agrees_with_resize2_f16(clip_grayh, ratio, bound):
    """The half path reproduces resize2's Box, bit-exact where its taps are."""
    w, h = _target(clip_grayh, ratio)
    worst = _worst_f16(clip_grayh, w, h, [0])
    assert worst <= bound, f"{ratio}: max|d|={worst} > {bound}"


@pytest.mark.parametrize("fmt,ratio", COLOR_CASES_F16)
def test_agrees_with_resize2_color_f16(request, fmt, ratio):
    """Every half plane agrees, with the subsampled planes on their own lattice."""
    clip = request.getfixturevalue(f"clip_{fmt}")
    w, h = _target(clip, ratio)
    out = _box(clip, w, h)
    assert out.format.id == clip.format.id
    assert (out.width, out.height) == (w, h)
    planes = list(range(clip.format.num_planes))
    worst = _worst_f16(clip, w, h, planes)
    assert worst <= F16_STEP, f"max|d|={worst}"


def test_f16_keeps_the_half_format(clip_grayh):
    """Half in, half out: the depth is neither widened nor reinterpreted."""
    w, h = _target(clip_grayh, 2)
    out = _box(clip_grayh, w, h)
    assert out.format.id == clip_grayh.format.id
    assert out.format.sample_type == vs.FLOAT
    assert out.format.bits_per_sample == 16


def test_chroma_planes_are_scaled_f16(clip_yuv420ph):
    """The subsampled half planes are averaged on their own lattice."""
    w, h = clip_yuv420ph.width // 2, clip_yuv420ph.height // 2
    out = cpu_node(_box(clip_yuv420ph, w, h))
    f = out.get_frame(0)
    assert (f.width, f.height) == (w, h)
    assert f.format.num_planes == 3
    assert _pxf(out, p=1).shape == (90, 160)
    assert _pxf(out, p=2).shape == (90, 160)


def test_gray_has_one_plane_f16(clip_grayh):
    w, h = _target(clip_grayh, 2)
    assert _pxf(_box(clip_grayh, w, h)).shape == (h, w)


def test_gray_has_one_plane(clip_grays):
    w, h = _target(clip_grays, 2)
    assert _px(_box(clip_grays, w, h)).shape == (h, w)


# ---------------------------------------------------------------------------
# Residency
# ---------------------------------------------------------------------------


def test_is_resident(clip_gray16):
    resident = vs.core.std.GPUUpload(clip=clip_gray16)
    node, transfers = _transfers(lambda: _box(resident, resident.width // 2, resident.height // 2))
    assert node.gpu_resident
    assert transfers == []


def test_is_resident_f16(clip_grayh):
    """The half path is GPU-resident too, with no inserted transfer."""
    resident = vs.core.std.GPUUpload(clip=clip_grayh)
    node, transfers = _transfers(lambda: _box(resident, resident.width // 2, resident.height // 2))
    assert node.gpu_resident
    assert transfers == []


def test_reference_is_not_resident(clip_gray16):
    """The CPU-only oracle this replaces is exactly why the filter exists."""
    resident = vs.core.std.GPUUpload(clip=clip_gray16)
    _, transfers = _transfers(
        lambda: Box().scale(resident, resident.width // 2, resident.height // 2, (0, 0))
    )
    assert transfers


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


def test_identity(clip_gray16):
    """Omitting the size, or asking for the input size, copies the plane."""
    nodes = [
        _box(clip_gray16, clip_gray16.width, clip_gray16.height),
        vs.core.vsfeel.Box(clip_gray16),
    ]
    for node in nodes:
        assert np.array_equal(_px(node), _px(clip_gray16))


def test_deterministic(clip_gray16):
    w, h = _target(clip_gray16, 2)
    a, b = _box(clip_gray16, w, h), _box(clip_gray16, w, h)
    for n in (0, 7, 23):
        assert np.array_equal(_px(a, n=n), _px(b, n=n))


def test_preserves_frame_props(clip_gray16):
    assert_preserves_frame_props(
        lambda clip, **kw: _box(clip, clip.width // 2, clip.height // 2), clip_gray16
    )


def test_identity_f16(clip_grayh):
    """The half plane copies exactly when neither axis is scaled."""
    nodes = [
        _box(clip_grayh, clip_grayh.width, clip_grayh.height),
        vs.core.vsfeel.Box(clip_grayh),
    ]
    for node in nodes:
        assert np.array_equal(_pxf(node), _pxf(clip_grayh))


def test_deterministic_f16(clip_grayh):
    w, h = _target(clip_grayh, 2)
    a, b = _box(clip_grayh, w, h), _box(clip_grayh, w, h)
    for n in (0, 7, 23):
        assert np.array_equal(_pxf(a, n=n), _pxf(b, n=n))


def test_f16_output_finite(clip_grayh):
    """Every half sample is finite and the filter is not a pixel drop."""
    w, h = _target(clip_grayh, 2)
    out = _box(clip_grayh, w, h)
    for n in range(out.num_frames):
        assert np.isfinite(_pxf(out, n=n)).all(), f"non-finite output at frame {n}"
    naive = cpu_node(vs.core.resize.Point(clip_grayh, w, h))
    assert not np.array_equal(_pxf(out), _pxf(naive))


def test_preserves_frame_props_f16(clip_grayh):
    assert_preserves_frame_props(
        lambda clip, **kw: _box(clip, clip.width // 2, clip.height // 2), clip_grayh
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_upscale(clip_grays):
    with pytest.raises(Exception, match="downscales"):
        _box(clip_grays, clip_grays.width * 2, clip_grays.height)


def test_rejects_8bit(clip_gray8):
    with pytest.raises(Exception, match="bitdepth"):
        _box(clip_gray8, clip_gray8.width // 2, clip_gray8.height)


def test_rejects_non_positive(clip_grays):
    with pytest.raises(Exception, match="positive"):
        _box(clip_grays, 0, clip_grays.height)


# ---------------------------------------------------------------------------
# zimg boundary cases (odd / cropped source dimensions)
# ---------------------------------------------------------------------------
#
# resize2's tap set is decided by zimg's coordinate loop, not by a cell formula;
# at a boundary that lands exactly on a sample centre the two disagree. Odd
# source dimensions make that common (measured: 613 -> 306 and 613 -> 612), and
# the 640x360 fixtures used above never hit it.


def test_odd_width_matches_resize2_u16(clip_gray16):
    """613 -> {306,612} and 639 -> 319: one or two columns at the boundary."""
    for right, width in ((27, 306), (27, 612), (1, 319)):
        cropped = clip_gray16.std.Crop(right=right)
        worst, _ = _worst(cropped, width, 180, [0])
        # The copy's f32 divide is allowed 2.5 ULP, so a 3-tap boundary run can
        # move a code; the old cell formula moved thousands.
        assert worst <= 2, f"613->{width}: {worst} codes"


def test_odd_width_matches_resize2_f32(clip_grays):
    for right, width in ((27, 306), (27, 612), (1, 319)):
        cropped = clip_grays.std.Crop(right=right)
        _, worst = _worst(cropped, width, 180, [0])
        assert worst <= 2.4e-7, f"613->{width}: {worst} of peak"


def test_odd_width_matches_resize2_f16(clip_grayh):
    for right, width in ((27, 306), (27, 612), (1, 319)):
        cropped = clip_grayh.std.Crop(right=right)
        worst = _worst_f16(cropped, width, 180, [0])
        assert worst <= F16_STEP, f"613->{width}: {worst}"


def test_odd_height_matches_resize2_f32(clip_grays):
    """The vertical pass has the same boundary rule."""
    cropped = clip_grays.std.Crop(bottom=7)  # 353 high
    _, worst = _worst(cropped, 640, 176, [0])
    assert worst <= 2.4e-7, f"353->176: {worst} of peak"


def test_one_axis_omitted_matches_resize2(clip_gray16):
    """Width-only and height-only requests exercise skip_vert in one pass."""
    for kwargs, width, height in (
        ({"width": 320}, 320, clip_gray16.height),
        ({"height": 180}, clip_gray16.width, 180),
    ):
        out = cpu_node(vs.core.vsfeel.Box(clip_gray16, **kwargs))
        ref = _ref(clip_gray16, width, height)
        for p in (0,):
            d = np.abs(_px(out, p) - _px(ref, p))
            assert d.max() <= 1, f"{kwargs}: {d.max()} codes"


def test_tiny_subsampled_target(clip_yuv420p16):
    """A legal small target (chroma 2x2/1x1) must match resize2 on every plane."""
    small = clip_yuv420p16.std.Crop(right=632, bottom=352)  # 8x8 luma
    for width, height in ((4, 4), (2, 2)):
        out = _box(small, width, height)
        ref = _ref(small, width, height)
        assert out.format.id == small.format.id
        for p in range(3):
            d = np.abs(_px(out, p) - _px(ref, p))
            assert d.max() <= 1, f"{width}x{height} plane {p}: {d.max()} codes"


_BAD_DIM_SCRIPT = """
import sys
import vapoursynth as vs

core = vs.core
fmt = getattr(vs, sys.argv[1])
clip = core.std.BlankClip(width=8, height=8, format=fmt, length=2)
try:
    core.vsfeel.Box(clip, width=int(sys.argv[2]), height=int(sys.argv[3]))
except vs.Error as exc:
    print("rejected: %s" % exc)
    raise SystemExit(0)
print("accepted")
raise SystemExit(1)
"""


def test_rejects_illegal_subsampled_output_dimensions():
    """An output size the format cannot store must be a vs.Error, not a crash.

    The output-stride probe frame is created at the requested size, and
    VapourSynth treats an illegal subsampled size as a fatal error (printing
    and terminating the process). This has to run in a subprocess because the
    pre-fix behaviour aborts the interpreter.
    """
    cases = (
        ("YUV420P16", 5, 4),
        ("YUV420P16", 4, 3),
        ("YUV422P16", 5, 4),
        ("YUV420PH", 5, 4),
    )
    for fmt, width, height in cases:
        proc = subprocess.run(
            [sys.executable, "-c", _BAD_DIM_SCRIPT, fmt, str(width), str(height)],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "MANGOHUD": "0"},
        )
        assert proc.returncode == 0, (
            f"{fmt} {width}x{height} did not return a normal error: "
            f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )
        assert "rejected:" in proc.stdout, proc.stdout
        assert "multiple of" in proc.stdout, proc.stdout


def test_box_parallel_load_matches_serial(clip_grays):
    """The pipelined multi-frame request path must match sequential requests."""
    par = eval_parallel(lambda c: _box(c, 320, 180), clip_grays, dtype=np.float32)
    ref = cpu_node(_box(clip_grays, 320, 180))
    for n in range(clip_grays.num_frames):
        seq = plane_to_ndarray(ref.get_frame(n), 0, np.float32)
        assert np.array_equal(par[n], seq), f"parallel/serial mismatch at frame {n}"
