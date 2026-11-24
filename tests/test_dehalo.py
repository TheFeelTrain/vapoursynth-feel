"""Unit tests for ``vsfeel.feel_dehalo`` (the device-resident ``fine_dehalo``).

The oracle is ``vsdehalo.fine_dehalo`` itself. For ``pre_ss`` both arms are
given ``core.vsfeel.NNEDI3`` for the doubling (the reference's own ``znedi3``
is not installed here), so the comparison grades the graph rather than the
kernel. The bounds are the measured mean and interior figures; the full-frame
maxima are the edge operator's border ring, which
``test_feel_dehalo_deviation_is_the_border_ring`` pins separately.

Run from the repository root:  tools/test.sh tests/test_dehalo.py -q
"""

import inspect

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_preserves_frame_props,
    eval_parallel,
    plane,
    transfers as _transfers,
)

from vstools import Range, get_peak_value

from vsfeel.dehalo import feel_dehalo
from vsdehalo import fine_dehalo

# The ring the edge operator's border policy reaches (its own 1 pixel, grown by
# rx and by the mask's own box blur).
RING = 8

# The interior bound shared by every case: measured 2.6e-4 at worst.
INSIDE_MAX = 1e-3


def _dtype(fmt):
    """The numpy dtype of one sample (conftest's helper stops at 16-bit ints)."""
    if fmt.sample_type == vs.FLOAT:
        return np.float16 if fmt.bits_per_sample == 16 else np.float32
    return np.uint8 if fmt.bits_per_sample == 8 else np.uint16


def _read(node, n=0, p=0):
    """One visible plane as float64, downloading a resident frame explicitly."""
    if node.gpu_resident:
        node = vs.core.std.GPUDownload(clip=node)
    return plane(node.get_frame(n), p, dtype=_dtype(node.format)).astype(np.float64)


def _diff(a_node, b_node, n=0, inset=0, p=0):
    """Absolute difference normalised to the format's full range."""
    a, b = _read(a_node, n, p), _read(b_node, n, p)
    peak = float(get_peak_value(a_node, range_in=Range.FULL))
    d = np.abs(a - b) / peak
    if inset:
        d = d[inset:-inset, inset:-inset]
    return d


@pytest.fixture
def host_nnedi3(monkeypatch):
    """Drive the reference's ``pre_ss`` doubling with ``core.vsfeel.NNEDI3``.

    ``fine_dehalo`` builds ``SuperSamplerProcess[NNEDI3]`` itself, so patching
    the class's function is the only way to give both arms the same doubling.
    """
    import vsaa.deinterlacers as deinterlacers

    def doubling(clip, field, dh, **kwargs):
        return vs.core.std.GPUDownload(clip=vs.core.vsfeel.NNEDI3(clip, field, dh, **kwargs))

    monkeypatch.setattr(
        deinterlacers.NNEDI3,
        "_deinterlacer_function",
        property(lambda self: doubling),
    )


# (case, kwargs, mean_max) -- every case also holds the shared INSIDE_MAX.
U16_CASES = [
    ("default", {}, 1e-5),
    ("ss=1", {"ss": 1}, 1e-5),
    ("no-mask", {"lowsens": -1, "highsens": -1}, 3e-5),
    ("strengths", {"darkstr": 0.4, "brightstr": 0.7}, 1.5e-5),
    ("radii", {"rx": 3, "ry": 1}, 1e-5),
    ("edgeproc", {"edgeproc": 0.5, "exclude": False}, 2e-5),
    ("two-iters", {"blur": (1.2, 1.8), "brightstr": (0.9, 0.5)}, 2e-5),
    ("thresholds", {"thmi": 60, "thma": 100, "thlimi": 30, "thlima": 80}, 1e-5),
    ("prewitt", {"edgemask": "prewitt"}, 1.5e-5),
    ("kirsch", {"edgemask": "kirsch"}, 1e-6),
    ("contra", {"contra": 0.5}, 1e-5),
]

PRE_SS_CASES = [
    ("pre_ss=2", {"pre_ss": 2}, 5e-6),
    ("pre_ss=2+strengths", {"pre_ss": 2, "darkstr": 0.3, "rx": 1}, 5e-6),
    ("pre_ss=4", {"pre_ss": 4}, 2e-6),
]

F32_CASES = [
    ("default", {}, 1.5e-5),
    ("pre_ss=2", {"pre_ss": 2}, 5e-6),
]


def test_feel_dehalo_is_resident(clip_gray16):
    """The whole chain stays on the device, with and without pre_ss.

    A single host stage (the edge mask, the box blur, one expression) would
    show up as a transfer here, and would cost a 4K round trip in a real
    script.
    """
    out, transfers = _transfers(lambda: feel_dehalo(clip_gray16, pre_ss=1))
    assert out.gpu_resident
    assert transfers == []

    out, transfers = _transfers(lambda: feel_dehalo(clip_gray16, pre_ss=2))
    assert out.gpu_resident
    assert transfers == []


@pytest.mark.parametrize("name,kwargs,mean_max", U16_CASES, ids=[c[0] for c in U16_CASES])
def test_feel_dehalo_matches_fine_dehalo_u16(clip_gray16, name, kwargs, mean_max):
    """Every parameter path matches the reference, away from the border ring."""
    ref = fine_dehalo(clip_gray16, **kwargs)
    out = feel_dehalo(clip_gray16, **kwargs)

    assert out.gpu_resident, name
    assert _diff(ref, out).mean() <= mean_max, name
    assert _diff(ref, out, inset=RING).max() <= INSIDE_MAX, name


@pytest.mark.parametrize("name,kwargs,mean_max", PRE_SS_CASES, ids=[c[0] for c in PRE_SS_CASES])
def test_feel_dehalo_matches_fine_dehalo_pre_ss(clip_gray16, host_nnedi3, name, kwargs, mean_max):
    """pre_ss reproduces the reference's supersample graph, not just its spirit.

    The doubling order (height first, then the transposed height), the field of
    each doubling and the final Point decimation all have to be the
    reference's, or the chain processes a different image.
    """
    ref = fine_dehalo(clip_gray16, **kwargs)
    out = feel_dehalo(clip_gray16, **kwargs)

    assert out.gpu_resident, name
    assert _diff(ref, out).mean() <= mean_max, name
    assert _diff(ref, out, inset=RING).max() <= INSIDE_MAX, name


@pytest.mark.parametrize("name,kwargs,mean_max", F32_CASES, ids=[c[0] for c in F32_CASES])
def test_feel_dehalo_matches_fine_dehalo_f32(clip_grays, host_nnedi3, name, kwargs, mean_max):
    """The float32 chain matches: the mask expressions are float there."""
    ref = fine_dehalo(clip_grays, **kwargs)
    out = feel_dehalo(clip_grays, **kwargs)

    assert _diff(ref, out).mean() <= mean_max, name
    assert _diff(ref, out, inset=RING).max() <= INSIDE_MAX, name


def test_feel_dehalo_matches_fine_dehalo_f16(clip_grayh, host_nnedi3):
    """Half is a first-class depth here, unlike the reference's default mask.

    ``fine_dehalo`` cannot run its default Robinson3 on f16 at all (vsmasktools
    probes ``core.std.Robinson3``, which does not exist, and the AttributeError
    escapes its suppression), so both arms name Prewitt.
    """
    ref = fine_dehalo(clip_grayh, edgemask="prewitt")
    out = feel_dehalo(clip_grayh, edgemask="prewitt")

    assert out.gpu_resident
    assert _diff(ref, out).mean() <= 2e-5
    assert _diff(ref, out, inset=RING).max() <= INSIDE_MAX

    # And the default detector, which the reference cannot run on f16.
    default = feel_dehalo(clip_grayh)
    assert default.gpu_resident
    assert np.array_equal(_read(default), _read(feel_dehalo(clip_grayh)))


def test_feel_dehalo_planes(clip_yuv420p16):
    """Luma-only leaves chroma alone, chroma-only is exact, all planes match.

    The chroma-only path is the one shape where every substituted stage agrees
    bit for bit, so it is also the sharpest check that the masks are not
    silently built on the wrong planes.
    """
    ref = fine_dehalo(clip_yuv420p16, planes=[1, 2], ss=1)
    out = feel_dehalo(clip_yuv420p16, planes=[1, 2], ss=1)

    assert _diff(ref, out, p=0).max() == 0.0, "luma is not selected, so it must not move"
    for p in (1, 2):
        assert _diff(ref, out, p=p).mean() <= 1e-5, p
        assert _diff(ref, out, p=p, inset=RING).max() <= INSIDE_MAX, p

    ref = fine_dehalo(clip_yuv420p16)
    out = feel_dehalo(clip_yuv420p16)
    assert out.gpu_resident
    assert _diff(ref, out).mean() <= 1e-5
    assert _diff(ref, out, inset=RING).max() <= INSIDE_MAX

    # Luma only: chroma is the source's, bit for bit.
    src = clip_yuv420p16
    out = vs.core.std.GPUDownload(clip=feel_dehalo(src))
    for p in (1, 2):
        assert np.array_equal(plane(src.get_frame(0), p), plane(out.get_frame(0), p))


def test_feel_dehalo_all_planes(clip_yuv420p16):
    """planes=[0,1,2] runs the mask on the whole clip, as the reference does."""
    ref = fine_dehalo(clip_yuv420p16, planes=[0, 1, 2], ss=1)
    out = feel_dehalo(clip_yuv420p16, planes=[0, 1, 2], ss=1)

    assert _diff(ref, out).mean() <= 1e-5
    assert _diff(ref, out, inset=RING).max() <= INSIDE_MAX


def test_feel_dehalo_deviation_is_the_border_ring(clip_gray16):
    """Every pixel beyond the ring is within the interior bound.

    This is the test that keeps the loose full-frame bound honest: a real
    regression anywhere in the picture cannot hide behind the border.
    """
    ref = fine_dehalo(clip_gray16)
    d = _diff(ref, feel_dehalo(clip_gray16))

    y, x = np.nonzero(d > INSIDE_MAX)
    h, w = d.shape

    assert y.size > 0, "the ring should be there: this is what makes the rest of it interesting"
    assert np.all((y < RING) | (y >= h - RING) | (x < RING) | (x >= w - RING))


def test_feel_dehalo_attach_masks(clip_gray16):
    """attach_masks republishes the very masks the function exposes.

    The property names are the reference's (scripts read them back), and the
    stored frames must be the same graph nodes, not a re-derivation.
    """
    out = feel_dehalo(clip_gray16, attach_masks=True, pre_ss=1)
    masks = feel_dehalo.masks
    frame = vs.core.std.GPUDownload(clip=out).get_frame(0)

    names = sorted(k for k in frame.props if k.startswith("FineDehaloMask"))
    assert names == [
        "FineDehaloMaskEdges",
        "FineDehaloMaskIgnoreDetails",
        "FineDehaloMaskLargeEdges",
        "FineDehaloMaskMain",
        "FineDehaloMaskSharpEdges",
        "FineDehaloMaskShrink",
    ]

    for key in masks:
        prop = "FineDehaloMask" + "".join(part.title() for part in key.split("_"))
        stored = vs.core.std.PropToClip(out, prop)
        assert np.array_equal(_read(stored), _read(masks[key])), prop

    assert_preserves_frame_props(feel_dehalo, clip_gray16, attach_masks=True)


def test_feel_dehalo_mask_mapping(clip_gray16):
    """``feel_dehalo.masks`` is the mapping the reference exposes, same keys."""
    out = feel_dehalo(clip_gray16, pre_ss=1)
    masks = feel_dehalo.masks

    assert list(masks) == [
        "EDGES",
        "SHARP_EDGES",
        "LARGE_EDGES",
        "IGNORE_DETAILS",
        "SHRINK",
        "MAIN",
    ]
    assert masks["main"] is masks.MAIN
    assert masks.MAIN is not None
    assert out.gpu_resident


def test_feel_dehalo_default_blur_is_the_reference_sigma(clip_gray16):
    """The default is ``Prefilter.GAUSS(sigma=1.4)``, which is 1.4.

    ``fine_dehalo``'s native default is the partial; accepting it too keeps a
    script that passes its own defaults identical to one that passes none.
    """
    from vsdenoise import Prefilter

    default = feel_dehalo(clip_gray16)
    named = feel_dehalo(clip_gray16, blur=Prefilter.GAUSS(sigma=1.4))

    assert np.array_equal(_read(default), _read(named))
    assert inspect.signature(feel_dehalo).parameters["blur"].default == 1.4


def test_feel_dehalo_8bit(clip_gray8):
    """8-bit is widened to 16 bits on the device, so it is close but not exact.

    The reference keeps 8-bit through the mask chain; widening gives the mask
    finer steps than the reference ever sees, so this path cannot be graded at
    the interior bound the other depths use.
    """
    ref = fine_dehalo(clip_gray8)
    out = feel_dehalo(clip_gray8)

    assert out.gpu_resident
    assert out.format == clip_gray8.format
    assert _diff(ref, out).mean() <= 3e-4
    assert _diff(ref, out).max() <= 6e-2


def test_feel_dehalo_validation(clip_gray16):
    """The reference's errors are kept, and the unsupported ones refuse early."""
    with pytest.raises(Exception, match="power of 2"):
        feel_dehalo(clip_gray16, pre_ss=3)

    with pytest.raises(Exception, match="power of 2"):
        feel_dehalo(clip_gray16, pre_ss=0)

    with pytest.raises(Exception, match="Unsupported keyword"):
        feel_dehalo(clip_gray16, nonsense=1)

    with pytest.raises(Exception):
        feel_dehalo(vs.core.std.BlankClip(width=64, height=64, format=vs.RGB24, length=1))

    with pytest.raises(Exception, match="lowsens and highsens"):
        feel_dehalo(clip_gray16, lowsens=-2)


def test_feel_dehalo_parallel_matches_serial(clip_gray16):
    """Concurrent frames agree with a sequential pull (the exec ring is shared)."""
    clip = clip_gray16.std.Trim(0, 3)

    node = feel_dehalo(clip, pre_ss=2)
    serial = np.stack([_read(node, n) for n in range(clip.num_frames)])
    parallel = np.asarray(eval_parallel(feel_dehalo, clip, pre_ss=2), dtype=np.float64)

    assert np.array_equal(serial, parallel)


def test_feel_dehalo_determinism(clip_gray16):
    """Two builds of the same graph produce the same frame."""
    a = feel_dehalo(clip_gray16)
    b = feel_dehalo(clip_gray16)

    assert np.array_equal(_read(a), _read(b))
