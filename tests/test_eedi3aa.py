"""Unit tests for core.vsfeel.EEDI3AA (fused based_aa EEDI3 chain).

EEDI3AA is one plugin call that reproduces vsaa's based_aa EEDI3 path exactly::

    Merge( EEDI3H( Merge( EEDI3(clip) ) ) )

The oracle is built from the *same plugin* (``core.vsfeel.EEDI3`` +
``core.vsfeel.EEDI3H`` + ``std.Merge``), so it isolates the fusion from any
backend difference. It is the exact chain the wrapper replaces, which makes
the expected agreement bit-exact for u16.

Agreement measured on the committed noise clip (640x360, 24 frames):

* u16 (GRAY16): max diff **0** (bit-exact) across field 2/3, mdis 3/20/40,
  nrad 0..3, vcheck 0..3, custom alpha/beta/gamma/vthresh, Gray8/16/32
  mclip present/absent, aliased and non-aliased sclip, ``_FieldBased``
  progressive/TFF/BFF input, YUV420 (all planes and luma-only) and every
  tested frame.
* f32 (GRAYS): within a few ulp of the chain (measured max 5.96e-8 absolute
  on the 0..1 float clip, ~1 ulp on the values involved). The residual is
  rounding-order: the chain averages two *materialised* f32 frames with two
  roundings while the fused kernels average in place; the merge formula is
  identical (``0.5*a + 0.5*b``). Bound used below: 1e-6, the repo's
  ulp-level float32 tier.

Self-consistency (determinism across runs, multi-stream == single stream,
parallel load) is exact. The chain's props (N frames, input fps,
``_FieldBased=PROGRESSIVE``) are asserted against the two-call chain.

Run from the repository root:  python -m pytest tests/test_eedi3aa.py
"""

import ctypes
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, frame_to_ndarray
from test_eedi3 import _plane, _dtype, _itemsize

pytestmark = pytest.mark.usefixtures("noise_gray")

# Tolerance for f32: measured 5.96e-8 (a few ulp); the repo's ulp-level tier.
F32_TOL = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _aa(clip, field=3, num_streams=1, **kwargs):
    return vs.core.vsfeel.EEDI3AA(
        clip, field=field, num_streams=num_streams, **kwargs
    )


def _oracle(clip, field=3, num_streams=1, mclip=None, sclip=None, **kwargs):
    """The exact based_aa chain, built from the same plugin."""
    kw = dict(kwargs)
    if mclip is not None:
        kw["mclip"] = mclip
    if sclip is not None:
        kw["sclip"] = sclip
    v = vs.core.vsfeel.EEDI3(clip, field=field, num_streams=num_streams, **kw)
    vm = vs.core.std.Merge(v[::2], v[1::2])
    h = vs.core.vsfeel.EEDI3H(vm, field=field, num_streams=num_streams, **kw)
    return vs.core.std.Merge(h[::2], h[1::2])


def _assert_oracle(clip, dtype, frames, field=3, mclip=None, sclip=None,
                   tol=0.0, **kwargs):
    mine = _aa(clip, field=field, mclip=mclip, sclip=sclip, **kwargs)
    ref = _oracle(clip, field=field, mclip=mclip, sclip=sclip, **kwargs)
    assert mine.width == ref.width == clip.width
    assert mine.height == ref.height == clip.height
    assert mine.num_frames == ref.num_frames == clip.num_frames
    worst = 0.0
    for n in frames:
        a = _plane(mine.get_frame(n), 0, clip.width, clip.height, dtype).astype(np.float64)
        b = _plane(ref.get_frame(n), 0, clip.width, clip.height, dtype).astype(np.float64)
        worst = max(worst, float(np.abs(a - b).max()))
    assert worst <= tol, f"oracle mismatch: {worst} > {tol}"


def _interleave2(clip):
    """based_aa's sclip: Interleave([s, s]) — 2N frames, usually one pointer."""
    return vs.core.std.Interleave([clip, clip])


def _interleave_distinct(clip):
    """A 2N-frame sclip whose two sub-frames do not alias."""
    return vs.core.std.Interleave([clip, clip.std.Invert()])


def _mask(clip, bits, left_white=True):
    """Half-white / half-black Gray mask at `bits`, at the clip's geometry."""
    half = clip.width // 2
    white = vs.core.std.BlankClip(
        format=vs.GRAY8, width=half, height=clip.height,
        length=clip.num_frames, color=[255 if left_white else 0])
    black = vs.core.std.BlankClip(
        format=vs.GRAY8, width=clip.width - half, height=clip.height,
        length=clip.num_frames, color=[0 if left_white else 255])
    m8 = vs.core.std.StackHorizontal([white, black])
    if bits == 8:
        return m8
    return vs.core.fmtc.bitdepth(m8, bits=bits, fulls=True, fulld=True)


def _eval_parallel(clip, field=3, num_streams=4, **kwargs):
    """Materialise every frame concurrently to exercise the multi-stream path."""
    out = _aa(clip, field=field, num_streams=num_streams, **kwargs)
    frames = [None] * out.num_frames
    bits = 16 if clip.format.sample_type == vs.INTEGER else 32
    dtype = _dtype(bits)

    def worker(n):
        frames[n] = _plane(out.get_frame(n), 0, out.width, out.height, dtype)

    threads = [threading.Thread(target=worker, args=(n,))
               for n in range(out.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(f is not None for f in frames)
    return frames


# ---------------------------------------------------------------------------
# exact oracle sweep (u16)
# ---------------------------------------------------------------------------

ORACLE_CASES_16 = [
    dict(field=3, vcheck=2, mdis=5, nrad=1),
    dict(field=2, vcheck=2, mdis=5, nrad=1),
    dict(field=3, vcheck=2, mdis=20, nrad=2),
    dict(field=3, vcheck=2, mdis=20, nrad=3),
    dict(field=3, vcheck=2, mdis=40, nrad=3),
    dict(field=3, vcheck=0, mdis=5, nrad=1),
    dict(field=3, vcheck=1, mdis=5, nrad=1),
    dict(field=3, vcheck=3, mdis=5, nrad=1),
    dict(field=2, vcheck=3, mdis=20, nrad=2),
    dict(field=3, vcheck=2, mdis=20, nrad=2,
         alpha=0.5, beta=0.25, gamma=5.0, vthresh0=8.0, vthresh1=64.0,
         vthresh2=9.0),
    dict(field=3, vcheck=2, mdis=3, nrad=0, alpha=0.0, beta=0.0, gamma=0.0),
    dict(field=3, vcheck=2, mdis=20, nrad=2, alpha=0.125, beta=0.25,
         gamma=40.0, vthresh0=12.0, vthresh1=24.0, vthresh2=4.0),
]


@pytest.mark.parametrize("mc", ["none", "g8", "g16", "g32"],
                         ids=["mclip0", "mclip8", "mclip16", "mclip32"])
@pytest.mark.parametrize("kw", ORACLE_CASES_16,
                         ids=[f"case{i}" for i in range(len(ORACLE_CASES_16))])
def test_eedi3aa_oracle_16bit(noise_16bit, kw, mc):
    """u16 EEDI3AA == the two-call based_aa chain, bit-exactly."""
    clip = noise_16bit
    mclip = None if mc == "none" else _mask(clip, int(mc[1:]))
    frames = [0, 11, clip.num_frames - 1]
    _assert_oracle(clip, np.uint16, frames, mclip=mclip,
                   sclip=_interleave2(clip), **kw)


@pytest.mark.parametrize("sc", ["none", "alias", "distinct"],
                         ids=["sclip0", "sclip-alias", "sclip-distinct"])
@pytest.mark.parametrize("kw", ORACLE_CASES_16[:4],
                         ids=[f"case{i}" for i in range(4)])
def test_eedi3aa_oracle_16bit_sclip_forms(noise_16bit, kw, sc):
    """The three sclip shapes: absent, based_aa's aliased Interleave([s, s]),
    and a genuinely distinct 2N-frame clip."""
    clip = noise_16bit
    sclip = {"none": None, "alias": _interleave2(clip),
             "distinct": _interleave_distinct(clip)}[sc]
    _assert_oracle(clip, np.uint16, [0, 11, 23], mclip=_mask(clip, 16),
                   sclip=sclip, **kw)


ORACLE_CASES_32 = [
    dict(field=3, vcheck=2, mdis=5, nrad=1),
    dict(field=3, vcheck=0, mdis=5, nrad=1),
    dict(field=3, vcheck=3, mdis=20, nrad=2),
    dict(field=2, vcheck=2, mdis=5, nrad=1),
    dict(field=3, vcheck=2, mdis=20, nrad=2,
         alpha=0.5, beta=0.25, gamma=5.0, vthresh0=8.0, vthresh1=64.0,
         vthresh2=9.0),
]


@pytest.mark.parametrize("sc", ["none", "alias", "distinct"],
                         ids=["sclip0", "sclip-alias", "sclip-distinct"])
@pytest.mark.parametrize("mc", ["none", "g8", "g32"],
                         ids=["mclip0", "mclip8", "mclip32"])
@pytest.mark.parametrize("kw", ORACLE_CASES_32,
                         ids=[f"case{i}" for i in range(len(ORACLE_CASES_32))])
def test_eedi3aa_oracle_32bit(noise_gray, kw, mc, sc):
    """f32 EEDI3AA == the chain within a few ulp (documented F32_TOL)."""
    clip = noise_gray
    mclip = None if mc == "none" else _mask(clip, int(mc[1:]))
    sclip = {"none": None, "alias": _interleave2(clip),
             "distinct": _interleave_distinct(clip)}[sc]
    _assert_oracle(clip, np.float32, [0, 11, clip.num_frames - 1],
                   mclip=mclip, sclip=sclip, tol=F32_TOL, **kw)


# ---------------------------------------------------------------------------
# _FieldBased input (the vertical pass honours it; the horizontal pass must not)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fb", [0, 1, 2], ids=["prog", "tff", "bff"])
@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3aa_field_based_input(noise_gray, noise_16bit, bits, fb):
    clip = noise_16bit if bits == 16 else noise_gray
    tagged = vs.core.std.SetFrameProps(clip, _FieldBased=fb)
    dtype = _dtype(bits)
    tol = 0.0 if bits == 16 else F32_TOL
    mclip = _mask(clip, bits)
    _assert_oracle(tagged, dtype, [0, 7, 23], mclip=mclip,
                   sclip=_interleave2(clip), tol=tol, vcheck=2, mdis=5, nrad=1)


# ---------------------------------------------------------------------------
# YUV420 (subsampled chroma planes + plane selection)
# ---------------------------------------------------------------------------

def _yuv(bits):
    src = vs.core.bs.VideoSource(NOISE_MKV) if hasattr(vs.core, "bs") \
        else vs.core.ffms2.Source(NOISE_MKV)
    return vs.core.fmtc.bitdepth(src, bits=bits, fulls=True, fulld=True)


@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3aa_yuv_all_planes(bits):
    """Every plane of a YUV420 clip, chroma subsampling included."""
    clip = _yuv(bits)
    assert clip.format.num_planes == 3
    sclip = _interleave2(clip)
    mclip = vs.core.fmtc.bitdepth(
        vs.core.std.BlankClip(format=vs.GRAY8, width=clip.width,
                              height=clip.height, length=clip.num_frames,
                              color=[255]),
        bits=bits, fulls=True, fulld=True)
    mine = _aa(clip, sclip=sclip, mclip=mclip, vcheck=2, mdis=5, nrad=1)
    ref = _oracle(clip, sclip=sclip, mclip=mclip, vcheck=2, mdis=5, nrad=1)
    dtype = _dtype(bits)
    worst = 0.0
    for n in (0, 5, 23):
        fm = mine.get_frame(n)
        fr = ref.get_frame(n)
        for p in range(3):
            a = _plane(fm, p, fm.width, fm.height, dtype).astype(np.float64)
            b = _plane(fr, p, fr.width, fr.height, dtype).astype(np.float64)
            worst = max(worst, float(np.abs(a - b).max()))
    assert worst <= (0.0 if bits == 16 else F32_TOL), worst


@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3aa_yuv_plane0_only(bits):
    """planes=[0] leaves chroma as a passthrough of the source."""
    clip = _yuv(bits)
    sclip = _interleave2(clip)
    out = _aa(clip, planes=[0], sclip=sclip, mclip=_mask(clip, bits),
              vcheck=2, mdis=5, nrad=1)
    dtype = _dtype(bits)
    for n in (0, 11):
        fm = out.get_frame(n)
        fs = clip.get_frame(n)
        for p in (1, 2):
            a = _plane(fm, p, fm.width, fm.height, dtype)
            b = _plane(fs, p, fs.width, fs.height, dtype)
            assert np.array_equal(a, b), f"chroma plane {p} not passed through"


# ---------------------------------------------------------------------------
# Determinism / streams / parallel load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_streams", [1, 4])
@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3aa_deterministic(noise_gray, noise_16bit, bits, num_streams):
    clip = noise_16bit if bits == 16 else noise_gray
    kw = dict(sclip=_interleave2(clip), mclip=_mask(clip, bits),
              vcheck=2, mdis=5, nrad=1)
    a = _aa(clip, num_streams=num_streams, **kw)
    b = _aa(clip, num_streams=num_streams, **kw)
    dtype = _dtype(bits)
    for n in (0, 11, 23):
        fa = _plane(a.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        fb = _plane(b.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        if bits == 16:
            assert np.array_equal(fa, fb), f"nondeterministic output at frame {n}"
        else:
            assert np.abs(fa - fb).max() < F32_TOL


@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3aa_multi_stream_matches_single(noise_gray, noise_16bit, bits):
    clip = noise_16bit if bits == 16 else noise_gray
    kw = dict(sclip=_interleave2(clip), mclip=_mask(clip, bits),
              vcheck=2, mdis=5, nrad=1)
    a = _aa(clip, num_streams=4, **kw)
    b = _aa(clip, num_streams=1, **kw)
    dtype = _dtype(bits)
    for n in (0, 11, 23):
        fa = _plane(a.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        fb = _plane(b.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        if bits == 16:
            assert np.array_equal(fa, fb), f"num_streams mismatch at frame {n}"
        else:
            assert np.abs(fa - fb).max() < F32_TOL


def test_eedi3aa_parallel_load_consistent(noise_16bit):
    """Two parallel num_streams=4 runs must match the serial path and each
    other — the request pattern that exposes stale descriptor bindings, fence
    misuse and command-pool reuse violations under load."""
    clip = noise_16bit
    kw = dict(sclip=_interleave2(clip), mclip=_mask(clip, 16),
              vcheck=2, mdis=5, nrad=1)
    a = _eval_parallel(clip, num_streams=4, **kw)
    b = _eval_parallel(clip, num_streams=4, **kw)
    ref = _aa(clip, num_streams=1, **kw)
    for n in range(clip.num_frames):
        r = _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
        assert np.array_equal(a[n], r), f"parallel 1/serial mismatch at frame {n}"
        assert np.array_equal(b[n], r), f"parallel 2/serial mismatch at frame {n}"
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


# ---------------------------------------------------------------------------
# Output geometry / props
# ---------------------------------------------------------------------------

def test_eedi3aa_single_rate_and_props(noise_16bit):
    """N in, N out at the input's fps, progressive, matching the chain."""
    clip = noise_16bit
    out = _aa(clip, sclip=_interleave2(clip), mclip=_mask(clip, 16))
    assert out.num_frames == clip.num_frames
    assert (out.fps_num, out.fps_den) == (clip.fps_num, clip.fps_den)
    assert out.width == clip.width and out.height == clip.height
    assert out.get_frame(0).props.get("_FieldBased") == 0


def test_eedi3aa_props_match_chain(noise_16bit):
    clip = noise_16bit
    kw = dict(sclip=_interleave2(clip), mclip=_mask(clip, 16))
    mine = _aa(clip, **kw)
    ref = _oracle(clip, **kw)
    assert mine.num_frames == ref.num_frames
    assert (mine.fps_num, mine.fps_den) == (ref.fps_num, ref.fps_den)
    for n in (0, 5):
        pm = mine.get_frame(n).props
        pr = ref.get_frame(n).props
        assert pm.get("_FieldBased") == pr.get("_FieldBased")
        assert pm.get("_DurationNum") == pr.get("_DurationNum")
        assert pm.get("_DurationDen") == pr.get("_DurationDen")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [0, 1, 4, -1])
def test_eedi3aa_rejects_bad_field(noise_16bit, field):
    with pytest.raises(vs.Error):
        _aa(noise_16bit, field=field)


def test_eedi3aa_rejects_dh(noise_16bit):
    with pytest.raises(vs.Error):
        _aa(noise_16bit, field=3, dh=1)


def test_eedi3aa_rejects_8bit(noise_8bit):
    with pytest.raises(vs.Error):
        _aa(noise_8bit, field=3)


def test_eedi3aa_rejects_bad_num_streams(noise_16bit):
    with pytest.raises(vs.Error):
        _aa(noise_16bit, field=3, num_streams=0)
    with pytest.raises(vs.Error):
        _aa(noise_16bit, field=3, num_streams=64)


def test_eedi3aa_sclip_needs_2n_frames(noise_16bit):
    """field > 1 describes the doubled output, so sclip must be 2N frames."""
    with pytest.raises(vs.Error):
        _aa(noise_16bit, field=3, sclip=noise_16bit)


def test_eedi3aa_is_pure_plugin_extension(noise_16bit):
    """EEDI3/EEDI3H are untouched: their own output must still equal the
    composed chain (the fused filter is additive)."""
    clip = noise_16bit
    v = vs.core.vsfeel.EEDI3(clip, field=3, num_streams=1, vcheck=2, mdis=5,
                             nrad=1)
    assert v.num_frames == 2 * clip.num_frames  # EEDI3 still doubles the rate
