"""Unit tests for core.vsfeel.Box.

The oracle is ``vskernels.Box`` (resize2's ``resize2.Custom`` with the box
kernel), which is also what ``vsaa.based_aa`` uses for an integer ``rfactor``.
Both depths and every color family are compared on the committed
bigbuckbunny_360p_grain.mp4 clip, whose grain exercises every tap window.

Run from the repository root:  uv run python -m pytest tests/test_box.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import assert_preserves_frame_props, cpu_node, plane as _plane

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
    ("yuv420_16", 2),
    ("yuv420_16", 4),
    ("yuv32", 2),
    ("yuv32", 4),
    ("yuv444_16", 2),
    ("yuv444_16", 4),
    ("yuv444_16", 1.5),
    ("rgb16", 2),
    ("rgb16", 4),
    ("rgb16", 1.5),
    ("rgb32", 2),
    ("rgb32", 4),
    ("rgb32", 1.5),
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
# Reference comparison: 16-bit integer and 32-bit float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,abs_bound,rel_bound", RATIOS)
def test_agrees_with_resize2_16bit(clip_16bit, ratio, abs_bound, rel_bound):
    """The integer path reproduces resize2's Box within the measured bound."""
    w, h = _target(clip_16bit, ratio)
    abs_d, _ = _worst(clip_16bit, w, h, [0])
    assert abs_d <= abs_bound, f"{ratio}: max|d|={abs_d} > {abs_bound}"


@pytest.mark.parametrize("ratio,abs_bound,rel_bound", RATIOS)
def test_agrees_with_resize2_float(clip_gray, ratio, abs_bound, rel_bound):
    """The float path reproduces resize2's Box to float32 rounding."""
    w, h = _target(clip_gray, ratio)
    _, rel_d = _worst(clip_gray, w, h, [0])
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


def test_chroma_planes_are_scaled(clip_yuv420_16):
    """A subsampled plane is averaged on its own lattice, not the luma's."""
    w, h = clip_yuv420_16.width // 2, clip_yuv420_16.height // 2
    out = cpu_node(_box(clip_yuv420_16, w, h))
    f = out.get_frame(0)
    assert (f.width, f.height) == (w, h)
    assert f.format.num_planes == 3
    # 640x360 -> 320x180 means chroma 320x180 -> 160x90
    assert _px(out, p=1).shape == (90, 160)
    assert _px(out, p=2).shape == (90, 160)


def test_gray_has_one_plane(clip_gray):
    w, h = _target(clip_gray, 2)
    assert _px(_box(clip_gray, w, h)).shape == (h, w)


# ---------------------------------------------------------------------------
# Residency
# ---------------------------------------------------------------------------


def test_is_resident(clip_16bit):
    resident = vs.core.std.GPUUpload(clip=clip_16bit)
    node, transfers = _transfers(lambda: _box(resident, resident.width // 2, resident.height // 2))
    assert node.gpu_resident
    assert transfers == []


def test_reference_is_not_resident(clip_16bit):
    """The CPU-only oracle this replaces is exactly why the filter exists."""
    resident = vs.core.std.GPUUpload(clip=clip_16bit)
    _, transfers = _transfers(
        lambda: Box().scale(resident, resident.width // 2, resident.height // 2, (0, 0))
    )
    assert transfers


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


def test_identity(clip_16bit):
    """Omitting the size, or asking for the input size, copies the plane."""
    nodes = [_box(clip_16bit, clip_16bit.width, clip_16bit.height), vs.core.vsfeel.Box(clip_16bit)]
    for node in nodes:
        assert np.array_equal(_px(node), _px(clip_16bit))


def test_deterministic(clip_16bit):
    w, h = _target(clip_16bit, 2)
    a, b = _box(clip_16bit, w, h), _box(clip_16bit, w, h)
    for n in (0, 7, 23):
        assert np.array_equal(_px(a, n=n), _px(b, n=n))


def test_preserves_frame_props(clip_16bit):
    assert_preserves_frame_props(
        lambda clip, **kw: _box(clip, clip.width // 2, clip.height // 2), clip_16bit
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_upscale(clip_gray):
    with pytest.raises(Exception, match="downscales"):
        _box(clip_gray, clip_gray.width * 2, clip_gray.height)


def test_rejects_8bit(clip_8bit):
    with pytest.raises(Exception, match="bitdepth"):
        _box(clip_8bit, clip_8bit.width // 2, clip_8bit.height)


def test_rejects_non_positive(clip_gray):
    with pytest.raises(Exception, match="positive"):
        _box(clip_gray, 0, clip_gray.height)
