"""Unit tests for core.vsfeel.NNEDI3.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The vsfeel implementation is a from-scratch Vulkan port of the
NNEDI3 predictor/prescreener math; the reference oracle is nnedi3vk (Vulkan),
which agrees BIT-EXACTLY with vszipcu's NNEDI3 (measured max diff 0 on the
noise clip, all planes, frames 0/5/11 — so matching nnedi3vk matches both).

Agreement measured on the noise clip (see notes/NNEDI3.md):

* GRAY16 / YUV420P16, every mode tested (field 0/1/2/3, dh, planes subsets,
  nsize 0..6, nns 0..4, qual 1/2, etype 0/1, pscrn 0..4): vsfeel == nnedi3vk
  within 1 LSB (max diff 0 on most content, 1 on rare rounding flips from
  the serial-vs-subgroup reduction order — mechanism documented in
  notes/NNEDI3.md).
* GRAYS: within ~3e-8 absolute (a few ulps); bound 1e-6.

Self-consistency (determinism across runs, multi-stream == single) is exact.

Run from the repository root:  python -m pytest tests/test_nnedi3.py
"""

import ctypes
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV

pytestmark = pytest.mark.usefixtures("noise_gray")


def _run(clip, field=1, num_streams=1, **kwargs):
    return vs.core.vsfeel.NNEDI3(
        clip,
        field=field,
        num_streams=num_streams,
        **kwargs,
    )


def _ref(clip, field=1, **kwargs):
    return vs.core.nnedi3vk.NNEDI3(clip, field=field, **kwargs)


def _plane_u16(frame, plane, width, height):
    return np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(height, width * 2),
    ).view(np.uint16).copy()


def _max_diff_u16(a, b, plane, width, height, frames):
    md = 0
    for n in frames:
        pa = _plane_u16(a.get_frame(n), plane, width, height)
        pb = _plane_u16(b.get_frame(n), plane, width, height)
        md = max(md, int(np.abs(pa.astype(np.int32) - pb.astype(np.int32)).max()))
    return md


# ---------------------------------------------------------------------------
# Reference agreement, GRAY16
# ---------------------------------------------------------------------------

REFERENCE_CASES_16 = [
    {},
    {"pscrn": 1},
    {"pscrn": 3},
    {"pscrn": 4},
    {"qual": 2},
    {"etype": 1},
    {"qual": 2, "etype": 1},
    {"nsize": 0},
    {"nsize": 3},
    {"nsize": 4},
    {"nns": 0},
    {"nns": 2},
    {"nns": 4},
    {"field": 0},
]


@pytest.mark.parametrize("kwargs", REFERENCE_CASES_16,
                         ids=[str(sorted(k.items())) for k in REFERENCE_CASES_16])
def test_nnedi3_matches_reference_16bit(noise_16bit, kwargs):
    a = _run(noise_16bit, **kwargs)
    b = _ref(noise_16bit, **kwargs)
    assert _max_diff_u16(a, b, 0, WIDTH, HEIGHT, (0, 11, 23)) <= 1


def test_nnedi3_pscrn0_within_1lsb(noise_16bit):
    """pscrn=0 forces every pixel through the predictor (serial accumulation
    vs the reference's subgroup reductions): allow 1 LSB."""
    a = _run(noise_16bit, pscrn=0)
    b = _ref(noise_16bit, pscrn=0)
    assert _max_diff_u16(a, b, 0, WIDTH, HEIGHT, (0, 11)) <= 1


def test_nnedi3_field_gt1_matches_reference(noise_16bit):
    for field in (2, 3):
        a = _run(noise_16bit, field=field)
        b = _ref(noise_16bit, field=field)
        assert a.num_frames == b.num_frames == 48
        assert _max_diff_u16(a, b, 0, WIDTH, HEIGHT, (0, 1, 24, 47)) <= 1


def test_nnedi3_dh_matches_reference(noise_16bit):
    a = _run(noise_16bit, dh=True)
    b = _ref(noise_16bit, dh=True)
    assert a.height == b.height == 2 * HEIGHT
    assert _max_diff_u16(a, b, 0, WIDTH, 2 * HEIGHT, (0, 11, 23)) <= 1


def _yuv420p16():
    core = vs.core
    if hasattr(core, "bs"):
        src = core.bs.VideoSource(NOISE_MKV)
    else:
        src = core.ffms2.Source(NOISE_MKV)
    return src.resize.Point(format=vs.YUV420P16)


def test_nnedi3_yuv_matches_reference():
    yuv = _yuv420p16()
    assert yuv.format.color_family == vs.YUV
    for kwargs in ({}, {"planes": [0]}, {"planes": [1, 2]}, {"nsize": 0}):
        a = _run(yuv, **kwargs)
        b = _ref(yuv, **kwargs)
        for p, (w, h) in enumerate(((WIDTH, HEIGHT), (WIDTH // 2, HEIGHT // 2),
                                    (WIDTH // 2, HEIGHT // 2))):
            assert _max_diff_u16(a, b, p, w, h, (0, 11)) <= 1, kwargs


# ---------------------------------------------------------------------------
# Reference agreement, float32 (bit-exact floats on configs tried)
# ---------------------------------------------------------------------------

REFERENCE_CASES_32 = [{}, {"pscrn": 0}, {"pscrn": 1}, {"qual": 2},
                      {"etype": 1}, {"nsize": 0}, {"nns": 4}, {"field": 0}]


@pytest.mark.parametrize("kwargs", REFERENCE_CASES_32,
                         ids=[str(sorted(k.items())) for k in REFERENCE_CASES_32])
def test_nnedi3_matches_reference_32bit(noise_gray, kwargs):
    from conftest import frame_to_ndarray
    a = _run(noise_gray, **kwargs)
    b = _ref(noise_gray, **kwargs)
    for n in (0, 11):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() <= 1e-6, f"frame {n} {kwargs}"


# ---------------------------------------------------------------------------
# Determinism / stream count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [16, 32])
def test_nnedi3_deterministic(noise_gray, noise_16bit, bits):
    clip = noise_16bit if bits == 16 else noise_gray
    a = _run(clip, nsize=0)
    b = _run(clip, nsize=0)
    for n in (0, 11, 23):
        fa, fb = a.get_frame(n), b.get_frame(n)
        if bits == 16:
            d = _plane_u16(fa, 0, WIDTH, HEIGHT).astype(np.int32) - \
                _plane_u16(fb, 0, WIDTH, HEIGHT).astype(np.int32)
        else:
            from conftest import frame_to_ndarray
            d = frame_to_ndarray(fa) - frame_to_ndarray(fb)
        assert np.abs(d).max() == 0, f"nondeterministic output at frame {n}"


@pytest.mark.parametrize("bits", [16, 32])
def test_nnedi3_multi_stream_matches_single(noise_gray, noise_16bit, bits):
    clip = noise_16bit if bits == 16 else noise_gray
    a = _run(clip, nsize=0, num_streams=4)
    b = _run(clip, nsize=0, num_streams=1)
    for n in (0, 11, 23):
        fa, fb = a.get_frame(n), b.get_frame(n)
        if bits == 16:
            d = _plane_u16(fa, 0, WIDTH, HEIGHT).astype(np.int32) - \
                _plane_u16(fb, 0, WIDTH, HEIGHT).astype(np.int32)
        else:
            from conftest import frame_to_ndarray
            d = frame_to_ndarray(fa) - frame_to_ndarray(fb)
        assert np.abs(d).max() == 0, f"stream-count divergence at frame {n}"


def test_nnedi3_parallel_load_consistent(noise_16bit):
    out = _run(noise_16bit, num_streams=4)
    expect = _plane_u16(_run(noise_16bit, num_streams=1).get_frame(5), 0, WIDTH, HEIGHT)
    got = [None] * out.num_frames

    def worker(n):
        got[n] = _plane_u16(out.get_frame(n), 0, WIDTH, HEIGHT)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(out.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(f is not None for f in got)
    assert np.abs(got[5].astype(np.int32) - expect.astype(np.int32)).max() == 0


# ---------------------------------------------------------------------------
# Frame geometry / properties
# ---------------------------------------------------------------------------

def test_nnedi3_field_gt1_doubles_frames(noise_16bit):
    out = _run(noise_16bit, field=3)
    assert out.num_frames == 2 * noise_16bit.num_frames
    assert out.fps_num * noise_16bit.fps_den == 2 * noise_16bit.fps_num * out.fps_den


def test_nnedi3_dh_doubles_height(noise_16bit):
    out = _run(noise_16bit, dh=True)
    assert out.height == 2 * noise_16bit.height
    assert out.width == noise_16bit.width


def test_nnedi3_output_props_progressive(noise_16bit):
    out = _run(noise_16bit, field=3)
    props = out.get_frame(1).props
    assert props["_FieldBased"] == 0
    assert "_Field" not in props


def test_nnedi3_kept_lines_copied(noise_16bit):
    """Lines the filter keeps must equal the source field rows exactly.
    Progressive content with field=1 keeps parity 0 (even output lines)."""
    out = _run(noise_16bit, field=1)
    for n in (0, 7):
        s = _plane_u16(noise_16bit.get_frame(n), 0, WIDTH, HEIGHT)
        o = _plane_u16(out.get_frame(n), 0, WIDTH, HEIGHT)
        assert np.abs(o[0::2].astype(np.int32) - s[0::2].astype(np.int32)).max() == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_nnedi3_rejects_8bit(noise_8bit):
    with pytest.raises(vs.Error):
        _run(noise_8bit).get_frame(0)


def test_nnedi3_rejects_bad_field(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, field=4)


def test_nnedi3_rejects_dh_with_field_gt1(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, field=3, dh=True)


def test_nnedi3_rejects_bad_nsize(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, nsize=7)


def test_nnedi3_rejects_bad_nns(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, nns=5)


def test_nnedi3_rejects_bad_qual(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, qual=3)


def test_nnedi3_rejects_bad_etype(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, etype=2)


def test_nnedi3_rejects_bad_pscrn(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, pscrn=5)


def test_nnedi3_rejects_bad_planes(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, planes=[0, 0])
    with pytest.raises(vs.Error):
        _run(noise_16bit, planes=[3])


def test_nnedi3_rejects_bad_num_streams(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, num_streams=0)


def test_nnedi3_rejects_odd_height(noise_16bit):
    odd = noise_16bit.std.Crop(bottom=1)
    with pytest.raises(vs.Error):
        _run(odd).get_frame(0)
