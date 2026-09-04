"""Unit tests for core.vsfeel.EEDI3H (horizontal EEDI3).

EEDI3H is a pure composition (Transpose -> EEDI3 -> Transpose, wired with
invokes in Eedi3HCreate): no kernels or host paths of its own. That gives a
very strong oracle — EEDI3H(x) must equal Transpose(EEDI3(Transpose(x)))
BIT-EXACTLY for every config (u16 and f32 alike: transposition only permutes
values, and the identical kernels then run identical operation sequences on
identical relative neighborhoods).

Agreement measured on the noise clip:
* transpose oracle: max diff 0 on every config tested (field 0/1/2/3, dh,
  mdis 3..40, nrad 0..3, vcheck 0..3, alpha/beta/gamma corners, mclip,
  sclip incl. 2N-frame field>1 form, YUV planes).
* vszipcl.EEDI3H (different family: full direction set, [0,1] normalized u16):
  ~1.7% pixels differ, max a few hundred LSB (measured 594, frac>1 0.02%).
  Loose sanity bounds only.

Run from the repository root:  python -m pytest tests/test_eedi3h.py
"""

import ctypes

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, frame_to_ndarray
from test_eedi3 import _plane, _dtype, _itemsize

pytestmark = pytest.mark.usefixtures("noise_gray")


def _runh(clip, field=1, num_streams=1, **kwargs):
    return vs.core.vsfeel.EEDI3H(
        clip,
        field=field,
        num_streams=num_streams,
        **kwargs,
    )


def _oracle(clip, field=1, num_streams=1, mclip=None, sclip=None, **kwargs):
    """Transpose(EEDI3(Transpose(clip))) with transposed aux clips."""
    t = vs.core.std.Transpose(clip)
    tm = vs.core.std.Transpose(mclip) if mclip is not None else None
    ts = vs.core.std.Transpose(sclip) if sclip is not None else None
    kw = dict(kwargs)
    if tm is not None:
        kw["mclip"] = tm
    if ts is not None:
        kw["sclip"] = ts
    inner = vs.core.vsfeel.EEDI3(t, field=field, num_streams=num_streams, **kw)
    return vs.core.std.Transpose(inner)


def _assert_oracle_exact(clip, dtype, out_w, out_h, frames, field, mclip=None,
                         sclip=None, **kwargs):
    """EEDI3H must equal the transpose oracle bit-exactly."""
    my = _runh(clip, field=field, mclip=mclip, sclip=sclip, **kwargs)
    ref = _oracle(clip, field=field, mclip=mclip, sclip=sclip, **kwargs)
    assert my.width == out_w and my.height == out_h
    assert ref.width == out_w and ref.height == out_h
    assert my.num_frames == ref.num_frames
    for n in frames:
        a = _plane(my.get_frame(n), 0, out_w, out_h, dtype).astype(np.float64)
        b = _plane(ref.get_frame(n), 0, out_w, out_h, dtype).astype(np.float64)
        d = np.abs(a - b)
        assert d.max() == 0, f"transpose-oracle mismatch at frame {n}: {d.max()}"


# ---------------------------------------------------------------------------
# transpose-oracle sweep (exact)
# ---------------------------------------------------------------------------

ORACLE_CASES_16 = [
    dict(field=1, mdis=5, nrad=1, vcheck=0),
    dict(field=0, mdis=5, nrad=1, vcheck=2),
    dict(field=1, mdis=20, nrad=2, vcheck=2),
    dict(field=1, mdis=20, nrad=3, vcheck=2),
    dict(field=1, mdis=40, nrad=3, vcheck=3),
    dict(field=1, mdis=3, nrad=0, vcheck=1),
    dict(field=1, mdis=20, nrad=2, vcheck=2,
         alpha=0.5, beta=0.25, gamma=5.0, vthresh0=8.0, vthresh1=64.0,
         vthresh2=9.0),
    dict(field=2, mdis=5, nrad=1, vcheck=2),
    dict(field=3, mdis=5, nrad=1, vcheck=2),
    dict(field=3, mdis=20, nrad=2, vcheck=2),
    dict(field=1, mdis=5, nrad=1, vcheck=2, dh=1),
]

ORACLE_CASES_32 = [
    dict(field=1, mdis=5, nrad=1, vcheck=0),
    dict(field=1, mdis=20, nrad=2, vcheck=2),
    dict(field=1, mdis=20, nrad=2, vcheck=2,
         alpha=0.5, beta=0.25, gamma=5.0, vthresh0=8.0, vthresh1=64.0,
         vthresh2=9.0),
    dict(field=3, mdis=5, nrad=1, vcheck=2),
    dict(field=1, mdis=5, nrad=1, vcheck=2, dh=1),
]


def _dh_out_w(kw):
    return 2 * WIDTH if kw.get("dh", 0) else WIDTH


@pytest.mark.parametrize("kw", ORACLE_CASES_16,
                         ids=[f"case{i}" for i in range(len(ORACLE_CASES_16))])
def test_eedi3h_transpose_oracle_16bit(noise_16bit, kw):
    """u16 EEDI3H == Transpose(EEDI3(Transpose)) bit-exactly."""
    field = kw.get("field", 1)
    out_w = _dh_out_w(kw)
    frames = [0, 11] if field <= 1 else [0, 5, 2 * noise_16bit.num_frames - 1]
    _assert_oracle_exact(noise_16bit, np.uint16, out_w, HEIGHT, frames, **kw)


@pytest.mark.parametrize("kw", ORACLE_CASES_32,
                         ids=[f"case{i}" for i in range(len(ORACLE_CASES_32))])
def test_eedi3h_transpose_oracle_32bit(noise_gray, kw):
    """f32 EEDI3H == Transpose(EEDI3(Transpose)) bit-exactly."""
    field = kw.get("field", 1)
    out_w = _dh_out_w(kw)
    frames = [0, 11] if field <= 1 else [0, 5, 2 * noise_gray.num_frames - 1]
    _assert_oracle_exact(noise_gray, np.float32, out_w, HEIGHT, frames, **kw)


def test_eedi3h_transpose_oracle_mclip(noise_16bit):
    """mclip flows through the composition transposed (exact)."""
    from test_eedi3 import _right_half_mask
    clip = noise_16bit
    kw = dict(field=1, mdis=5, nrad=1, vcheck=2)
    m = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 8)
    _assert_oracle_exact(clip, np.uint16, WIDTH, HEIGHT, [0, 11],
                         mclip=m, **kw)


def test_eedi3h_transpose_oracle_sclip(noise_16bit):
    """sclip (2N frames under field>1) flows through transposed (exact)."""
    clip = noise_16bit
    kw = dict(field=3, mdis=5, nrad=1, vcheck=2)
    s = vs.core.std.Interleave([clip, clip])
    n_out = 2 * clip.num_frames
    _assert_oracle_exact(clip, np.uint16, WIDTH, HEIGHT,
                         [0, 5, n_out - 1], sclip=s, **kw)


def test_eedi3h_transpose_oracle_mclip_sclip(noise_16bit):
    """mclip + sclip together (exact)."""
    from test_eedi3 import _right_half_mask
    clip = noise_16bit
    kw = dict(field=1, mdis=20, nrad=2, vcheck=2)
    m = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 8)
    _assert_oracle_exact(clip, np.uint16, WIDTH, HEIGHT, [0, 11],
                         mclip=m, sclip=clip, **kw)


# ---------------------------------------------------------------------------
# vszipcl reference (loose sanity: different algorithm family)
# ---------------------------------------------------------------------------

def test_eedi3h_vszipcl_loose(noise_16bit):
    """Rough agreement with vszipcl.EEDI3H (family gap, not an oracle).

    Measured on noise: ~1.7% pixels differ, max a few hundred LSB. Catches
    gross errors (wrong axis, broken composition) while allowing the family
    difference.
    """
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "EEDI3H"):
        pytest.skip("no vszipcl.EEDI3H reference")
    clip = noise_16bit
    kw = dict(field=1, mdis=5, nrad=1, vcheck=2)
    my = _runh(clip, **kw)
    ref = vs.core.vszipcl.EEDI3H(clip, num_streams=1, **kw)
    assert (my.width, my.height) == (ref.width, ref.height) == (WIDTH, HEIGHT)
    for n in (0, 11):
        a = _plane(my.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.int32)
        b = _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.int32)
        d = np.abs(a - b)
        assert (d > 0).mean() < 0.05, f"too many pixels differ at frame {n}"
        assert d.max() < 4096, f"max diff too large at frame {n}: {d.max()}"


# ---------------------------------------------------------------------------
# geometry / validation / determinism
# ---------------------------------------------------------------------------

def test_eedi3h_output_dims_and_props(noise_16bit):
    """Width/height/frames follow the horizontal convention."""
    clip = noise_16bit
    base = _runh(clip, field=1, mdis=5)
    assert (base.width, base.height) == (WIDTH, HEIGHT)
    assert base.num_frames == clip.num_frames
    dh = _runh(clip, field=1, mdis=5, dh=1)
    assert (dh.width, dh.height) == (2 * WIDTH, HEIGHT)
    dbl = _runh(clip, field=3, mdis=5)
    assert dbl.num_frames == 2 * clip.num_frames
    assert (dbl.width, dbl.height) == (WIDTH, HEIGHT)
    props = dbl.get_frame(0).props
    assert props["_FieldBased"] == 0  # progressive
    # duration halved under field>1 (same rule as EEDI3)
    dur_num = props.get("_DurationNum")
    src_props = clip.get_frame(0).props
    if dur_num is not None and src_props.get("_DurationNum") is not None:
        assert dur_num * src_props["_DurationDen"] * 2 == \
            src_props["_DurationNum"] * props["_DurationDen"]


def test_eedi3h_rejects_odd_width(noise_16bit):
    """The interpolated (horizontal) axis must be mod 2 when dh=False."""
    odd = noise_16bit.std.Crop(right=1)
    assert odd.width % 2 == 1
    with pytest.raises(vs.Error):
        _runh(odd, field=1, mdis=5).get_frame(0)
    # dh lifts the constraint (width doubles instead)
    ok = _runh(odd, field=1, mdis=5, dh=1)
    assert ok.width == 2 * odd.width
    ok.get_frame(0)


def test_eedi3h_masked_region_is_horizontal_cubic(noise_16bit):
    """Inside a masked (black) region the pixel is the HORIZONTAL cubic of
    the two kept columns (mirror of EEDI3's vertical-cubic guarantee)."""
    from test_eedi3 import _right_half_mask
    clip = noise_16bit
    kw = dict(field=0, mdis=5, nrad=1, vcheck=0)
    m = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 16)
    out = _runh(clip, mclip=m, **kw)
    src = noise_16bit
    f = out.get_frame(0)
    s = src.get_frame(0)
    d = _plane(f, 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
    a = _plane(s, 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
    # mask = white LEFT half: use an interp column there. field=0 interpolates
    # columns 0, 2, ...; column 10 is interp, taps at x-3, x-1, x+1, x+3 are
    # interior kept columns.
    col, row = 10, HEIGHT // 2
    taps = (9 * (int(a[row, col - 1]) + int(a[row, col + 1]))
            - (int(a[row, col - 3]) + int(a[row, col + 3])) + 8) // 16
    assert abs(int(d[row, col]) - int(taps)) <= 1, (
        f"masked px {row},{col}: {d[row, col]} vs hcubic {taps}")


def test_eedi3h_determinism_and_streams(noise_16bit):
    """Repeated runs and stream counts agree exactly."""
    clip = noise_16bit
    kw = dict(field=1, mdis=5, nrad=1, vcheck=2)
    a = _plane(_runh(clip, num_streams=1, **kw).get_frame(3),
               0, WIDTH, HEIGHT, np.uint16)
    b = _plane(_runh(clip, num_streams=1, **kw).get_frame(3),
               0, WIDTH, HEIGHT, np.uint16)
    assert np.array_equal(a, b)
    c = _plane(_runh(clip, num_streams=4, **kw).get_frame(3),
               0, WIDTH, HEIGHT, np.uint16)
    assert np.array_equal(a, c)


def test_eedi3h_yuv_planes(noise_16bit):
    """YUV input works; unprocessed chroma passes through (exact oracle)."""
    src = vs.core.std.ShufflePlanes(noise_16bit, [0, 0, 0], vs.YUV)
    src = vs.core.resize.Bicubic(src, format=vs.YUV420P16)
    kw = dict(field=1, mdis=5, nrad=1, vcheck=0, planes=[0])
    my = _runh(src, **kw)
    ref = _oracle(src, **kw)
    assert (my.width, my.height) == (src.width, src.height)
    for n in (0, 5):
        for plane in range(3):
            w = src.width >> (1 if plane else 0)
            h = src.height >> (1 if plane else 0)
            a = _plane(my.get_frame(n), plane, w, h, np.uint16)
            b = _plane(ref.get_frame(n), plane, w, h, np.uint16)
            assert np.array_equal(a, b), f"plane {plane} frame {n} differs"
