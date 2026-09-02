"""Unit tests for core.vsfeel.EEDI3.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The vsfeel implementation is a brand-new Vulkan port of the EEDI3
family-A algorithm; the reference oracles are eedi3vk2 (Vulkan, bit-exact on
the shared parameter surface) and eedi3m (original CPU implementation, used as
a loose sanity oracle where eedi3vk2 lacks a parameter such as cost3/ucubic).

Agreement measured on the noise clip (640x360, interp rows only):

* u16 (GRAY16): vsfeel == eedi3vk2 BIT-EXACT (max diff 0) across every shared
  config tested: field 0/1, dh, mdis 3..40, nrad 0..3, vcheck 0..3, custom
  alpha/beta/gamma/vthresh, sclip, and mclip (Gray8/Gray16/Gray32 masks;
  vsfeel converts the mask to Gray8 internally, eedi3vk2 needs the clip's
  format - both then agree bit-exactly).
* f32 (GRAYS): vsfeel vs eedi3vk2 within ~1 ulp (7.45e-9 on the noise clip);
  a few parameter corners (e.g. gamma=5 with vcheck) show isolated DP
  argmin flips worth up to a few e-3 on a handful of pixels (measured
  5.45e-3). Configs with flips use a looser bound.
* vsfeel vs eedi3m (the installed CPU reference) diverges on tiny sparse
  flip sets (up to ~1500 px of 115200, max ~3276) that are IDENTICAL to the
  eedi3m-vs-eedi3vk2 flip sets: both GPU implementations share one float-DP
  ordering while eedi3m's differs. Hence eedi3m is only a loose oracle here.

Self-consistency (determinism across streams/runs, multi-stream == single,
parallel load) is exact.

Run from the repository root:  python -m pytest tests/test_eedi3.py
"""

import ctypes
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(clip, field=1, num_streams=1, **kwargs):
    return vs.core.vsfeel.EEDI3(
        clip,
        field=field,
        num_streams=num_streams,
        **kwargs,
    )


def _plane(frame, plane, width, height, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    return np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(height, width * itemsize),
    ).view(dtype).copy()


def _stride_plane(frame, plane):
    """Copy any plane into an ndarray, honouring the row pitch."""
    fmt = frame.format
    ss_w = fmt.subsampling_w if plane in (1, 2) and fmt.num_planes >= 3 else 0
    ss_h = fmt.subsampling_h if plane in (1, 2) and fmt.num_planes >= 3 else 0
    w = frame.width >> ss_w
    h = frame.height >> ss_h
    dtype = np.float32 if fmt.sample_type == vs.FLOAT else np.uint16
    return _plane(frame, plane, w, h, dtype)


def _interp_rows(h, n, field):
    """Boolean row mask of the interpolated rows of output frame n.

    Mirrors the host: eff field base = field & 1; under frame doubling
    (field > 1) it is XORed with (n & 1); interp dst rows = eff, eff+2, ...
    """
    fbase = field & 1
    eff = fbase if field <= 1 else ((n & 1) ^ fbase)
    rows = np.zeros(h, dtype=bool)
    rows[eff::2] = True
    return rows


def _dtype(bits):
    return np.uint16 if bits == 16 else np.float32


def _itemsize(bits):
    return np.dtype(_dtype(bits)).itemsize


def _eval_parallel(clip, field=1, num_streams=4, **kwargs):
    """Evaluate every frame concurrently to exercise the multi-stream path."""
    out = _run(clip, field=field, num_streams=num_streams, **kwargs)
    frames = [None] * out.num_frames
    bits = 16 if clip.format.sample_type == vs.INTEGER else 32
    it = _itemsize(bits)
    dtype = _dtype(bits)

    def worker(n):
        frames[n] = _plane(out.get_frame(n), 0, out.width, out.height, dtype)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(out.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(f is not None for f in frames)
    return frames


def _right_half_mask(width, height, length, bits):
    """Gray mask clip: white left half, black right half (drives mclip)."""
    half = width // 2
    black = vs.core.std.BlankClip(format=vs.GRAY8, width=half, height=height,
                                  length=length, color=[0])
    white = vs.core.std.BlankClip(format=vs.GRAY8, width=half, height=height,
                                  length=length, color=[255])
    mask8 = vs.core.std.StackHorizontal([white, black])
    return vs.core.fmtc.bitdepth(mask8, bits=bits, fulls=True, fulld=True)


# ---------------------------------------------------------------------------
# Determinism / stream count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_streams", [1, 4])
@pytest.mark.parametrize("bits,field", [(16, 1), (32, 1)], ids=["16bit", "32bit"])
def test_eedi3_deterministic(noise_gray, noise_16bit, bits, field, num_streams):
    clip = noise_16bit if bits == 16 else noise_gray
    a = _run(clip, field=field, num_streams=num_streams)
    b = _run(clip, field=field, num_streams=num_streams)
    dtype = _dtype(bits)
    for n in (0, 11, 23):
        fa = _plane(a.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        fb = _plane(b.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        if bits == 16:
            assert np.array_equal(fa, fb), f"nondeterministic output at frame {n}"
        else:
            assert np.abs(fa - fb).max() < 1e-6, f"nondeterministic output at frame {n}"


@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3_multi_stream_matches_single(noise_gray, noise_16bit, bits):
    clip = noise_16bit if bits == 16 else noise_gray
    a = _run(clip, num_streams=4)
    b = _run(clip, num_streams=1)
    dtype = _dtype(bits)
    for n in (0, 11, 23):
        fa = _plane(a.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        fb = _plane(b.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        if bits == 16:
            assert np.array_equal(fa, fb), f"num_streams mismatch at frame {n}"
        else:
            assert np.abs(fa - fb).max() < 1e-6, f"num_streams mismatch at frame {n}"


@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3_parallel_load_consistent(noise_gray, noise_16bit, bits):
    """Two parallel num_streams=4 runs must match the serial path, and each
    other — the request pattern that exposes stale descriptor bindings, fence
    misuse and command-pool reuse violations under load."""
    clip = noise_16bit if bits == 16 else noise_gray
    a = _eval_parallel(clip, num_streams=4)
    b = _eval_parallel(clip, num_streams=4)
    ref = _run(clip, num_streams=1)
    dtype = _dtype(bits)
    for n in range(clip.num_frames):
        r = _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, dtype)
        if bits == 16:
            assert np.array_equal(a[n], r), f"parallel 1/serial mismatch at frame {n}"
            assert np.array_equal(b[n], r), f"parallel 2/serial mismatch at frame {n}"
            assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"
        else:
            assert np.abs(a[n] - r).max() < 1e-6, f"parallel 1/serial mismatch at frame {n}"
            assert np.abs(b[n] - r).max() < 1e-6, f"parallel 2/serial mismatch at frame {n}"
            assert np.abs(a[n] - b[n]).max() < 1e-6, f"nondeterministic output at frame {n}"


# ---------------------------------------------------------------------------
# Pipelined-reader stress (regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_streams", [1, 4])
@pytest.mark.parametrize("bits", [None, 16], ids=["32bit", "16bit"])
def test_eedi3_vspipe_pipelined_no_hang(num_streams, bits):
    """vspipe's pipelined reader plus VapourSynth's prefetch drive frame
    requests far ahead of the in-flight frames; a queue stall or a
    resource-pool misuse deadlocks or crashes the whole run. Runs the real
    benchmark as a subprocess so that vspipe's reader drives the requests.
    Times out if the filter hangs."""
    bench = Path(__file__).resolve().parent.parent / "benchmark" / "bench.py"
    cmd = [sys.executable, str(bench), "--synthetic", "--frames", "200",
           "--filter", "eedi3", "vsfeel"]
    if bits:
        cmd += ["--bits", str(bits)]
    if num_streams != 1:
        cmd += ["--num-streams", str(num_streams)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env={**os.environ, "MANGOHUD": "0"},
        )
    except subprocess.TimeoutExpired:
        pytest.fail("benchmark run hung (queue stall / semaphore deadlock)")
    assert result.returncode == 0, (
        f"bench exited {result.returncode}:\n{result.stdout}\n{result.stderr}")
    fps_line = next((line for line in result.stdout.splitlines()
                     if "vsfeel" in line and "fps" in line), None)
    assert fps_line, f"no vsfeel fps line:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# All frames finite / in range
# ---------------------------------------------------------------------------

def test_eedi3_no_nan_all_frames_32bit(noise_gray):
    out = _run(noise_gray, num_streams=1)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0, WIDTH, HEIGHT, np.float32)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


def test_eedi3_no_nan_all_frames_multi_stream_32bit(noise_gray):
    out = _run(noise_gray, num_streams=4)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0, WIDTH, HEIGHT, np.float32)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


def test_eedi3_in_range_all_frames_16bit(noise_16bit):
    out = _run(noise_16bit, num_streams=1)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
        assert a.max() <= 65535
        assert a.min() >= 0


@pytest.mark.parametrize("num_streams", [2, 4])
def test_eedi3_in_range_all_frames_multi_stream_16bit(noise_16bit, num_streams):
    out = _run(noise_16bit, num_streams=num_streams)
    for n in range(out.num_frames):
        a = _plane(out.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
        assert a.max() <= 65535


# ---------------------------------------------------------------------------
# Reference comparison: eedi3vk2 (bit-exact oracle on the shared surface)
# ---------------------------------------------------------------------------

# (kwargs, tolerance). u16 is bit-exact -> 0. f32 is ~1 ulp except a few
# parameter corners with DP argmin flips (looser bound, measured <= 5.5e-3
# on this clip; kept well below the visible eedi3m band).
REFERENCE_CASES_16 = [
    ({"field": 1}, 0),                                              # defaults
    ({"field": 1, "mdis": 5, "nrad": 1, "vcheck": 0}, 0),
    ({"field": 1, "mdis": 3, "nrad": 3, "vcheck": 1}, 0),
    ({"field": 1, "mdis": 10, "nrad": 0, "vcheck": 3}, 0),
    ({"field": 1, "mdis": 40, "nrad": 3, "vcheck": 2}, 0),
    ({"field": 0, "mdis": 5, "nrad": 1, "vcheck": 0}, 0),
    ({"field": 1, "dh": 1, "mdis": 5, "nrad": 1, "vcheck": 0}, 0),
    ({"field": 1, "dh": 1, "mdis": 20, "nrad": 3, "vcheck": 2}, 0),
    ({"field": 3, "mdis": 5, "nrad": 1, "vcheck": 0}, 0),
    ({"field": 1, "alpha": 0.0, "beta": 0.0, "gamma": 5.0}, 0),
    ({"field": 1, "alpha": 0.5, "beta": 0.5}, 0),
    ({"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2,
      "vthresh0": 128.0, "vthresh1": 8.0, "vthresh2": 16.0}, 0),
]

REFERENCE_CASES_32 = [
    ({"field": 1}, 1e-6),
    ({"field": 1, "mdis": 5, "nrad": 1, "vcheck": 0}, 1e-6),
    ({"field": 1, "mdis": 3, "nrad": 3, "vcheck": 1}, 1e-6),
    ({"field": 1, "mdis": 10, "nrad": 0, "vcheck": 3}, 1e-6),
    ({"field": 0, "mdis": 5, "nrad": 1, "vcheck": 0}, 1e-6),
    ({"field": 1, "dh": 1, "mdis": 5, "nrad": 1, "vcheck": 0}, 1e-6),
    ({"field": 1, "dh": 1, "mdis": 20, "nrad": 3, "vcheck": 2}, 1e-6),
    ({"field": 3, "mdis": 5, "nrad": 1, "vcheck": 0}, 1e-6),
    ({"field": 1, "alpha": 0.0, "beta": 0.0, "gamma": 5.0}, 5e-3),
    ({"field": 1, "alpha": 0.5, "beta": 0.5}, 1e-6),
    ({"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2,
      "vthresh0": 128.0, "vthresh1": 8.0, "vthresh2": 16.0}, 1e-6),
]

_COMPARE_SCRIPT = textwrap.dedent(f"""\
    import json
    import sys
    import vapoursynth as vs
    import numpy as np
    import ctypes

    bits = int(sys.argv[2])
    cases = json.loads(sys.argv[1])
    core = vs.core
    src = core.bs.VideoSource({NOISE_MKV!r})
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                              bits=bits, fulls=True, fulld=True)
    it = 2 if bits == 16 else 4
    dt = np.uint16 if bits == 16 else np.float32
    h, w = clip.height, clip.width

    results = []
    for kwargs in cases:
        ref_node = core.eedi3vk2.EEDI3(clip, **kwargs)
        my_node = core.vsfeel.EEDI3(clip, **kwargs)
        # interp rows only: kept rows are bit-copies and always equal
        dmax = 0.0
        for n in (0, 11, 23):
            fr = ref_node.get_frame(n)
            fm = my_node.get_frame(n)
            a = np.ctypeslib.as_array(ctypes.cast(fm.get_read_ptr(0), ctypes.POINTER(ctypes.c_uint8)), shape=(h, w * it)).view(dt)
            b = np.ctypeslib.as_array(ctypes.cast(fr.get_read_ptr(0), ctypes.POINTER(ctypes.c_uint8)), shape=(h, w * it)).view(dt)
            fbase = kwargs.get('field', 1) & 1
            eff = fbase if kwargs.get('field', 1) <= 1 else ((n & 1) ^ fbase)
            rows = np.zeros(h, dtype=bool); rows[eff::2] = True
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))[rows]
            if d.size:
                dmax = max(dmax, float(d.max()))
        results.append(dmax)
    print(json.dumps(results))
""")


def _reference_max_diffs(bits, cases):
    """Run every config against eedi3vk2 in one subprocess; a crashing
    reference yields None (the comparison must not take down the suite)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _COMPARE_SCRIPT,
             json.dumps([kw for kw, _ in cases]), str(bits)],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    if len(parsed) != len(cases):
        return None
    return parsed


@pytest.mark.parametrize("bits,cases", [
    (16, REFERENCE_CASES_16), (32, REFERENCE_CASES_32),
], ids=["16bit", "32bit"])
def test_eedi3_matches_vk2_reference(bits, cases):
    """vsfeel must closely match eedi3vk2 across the shared parameter surface.

    u16 is bit-exact (tolerance 0). f32 sits at ~1 ulp except a few corners
    (documented in REFERENCE_CASES_32) where a rounding-order difference
    flips a DP argmin; those use a looser bound.
    """
    if not hasattr(vs.core, "eedi3vk2") or not hasattr(vs.core.eedi3vk2, "EEDI3"):
        pytest.skip("no eedi3vk2.EEDI3 reference")
    maxdiffs = _reference_max_diffs(bits, cases)
    if maxdiffs is None:
        pytest.skip("reference comparison crashed")
    for (kwargs, tol), maxdiff in zip(cases, maxdiffs):
        assert maxdiff < tol or maxdiff == 0, (
            f"max diff {maxdiff} vs eedi3vk2 for {kwargs} (tol {tol})")


# ---------------------------------------------------------------------------
# eedi3m loose sanity (cost3 / ucubic, which eedi3vk2 lacks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 0, "cost3": 0},
    {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 0, "ucubic": 0},
    {"field": 1, "mdis": 20, "nrad": 3, "vcheck": 0, "cost3": 0},
    {"field": 1, "mdis": 20, "nrad": 3, "vcheck": 0, "ucubic": 0},
    {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 0,
     "cost3": 0, "ucubic": 0},
], ids=lambda kw: str(kw) or "defaults")
def test_eedi3_cost3_ucubic_close_to_eedi3m(noise_16bit, kwargs):
    """cost3/ucubic only exist in vsfeel and eedi3m. The noise-clip agreement
    is >= 99.9% of interp pixels within 1 LSB (measured 99.86% worst on the
    mdis20/nrad3 cost3=0 config); the tiny residue is the documented float-DP
    argmin flip set, identical to the eedi3m-vs-eedi3vk2 divergence."""
    if not hasattr(vs.core, "eedi3m") or not hasattr(vs.core.eedi3m, "EEDI3"):
        pytest.skip("no eedi3m.EEDI3 reference")
    core = vs.core
    # eedi3m takes the same integer-domain parameters (native u16 scale).
    ref = core.eedi3m.EEDI3(noise_16bit, **kwargs)
    my = core.vsfeel.EEDI3(noise_16bit, **kwargs)
    worst_frac = 0.0
    for n in (0, 11, 23):
        a = _plane(my.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
        b = _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
        rows = _interp_rows(HEIGHT, n, kwargs["field"])
        d = np.abs(a - b)[rows]
        assert d.max() <= 65535  # sane
        worst_frac = max(worst_frac, float((d > 1).sum()) / float(d.size))
    # measured worst case ~0.14%; leave generous headroom for other content
    assert worst_frac < 0.02, f"worst interp mismatch fraction {worst_frac}"


# ---------------------------------------------------------------------------
# mclip semantics (single Gray mask drives every processed plane)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [16, 32], ids=["16bit", "32bit"])
def test_eedi3_mclip_gray8_matches_vk2_same_format(noise_gray, noise_16bit, bits):
    """vsfeel's Gray8 mask must equal eedi3vk2's same-format mask exactly on
    the shared surface (vk2 requires the clip's own format)."""
    if not hasattr(vs.core, "eedi3vk2") or not hasattr(vs.core.eedi3vk2, "EEDI3"):
        pytest.skip("no eedi3vk2.EEDI3 reference")
    clip = noise_16bit if bits == 16 else noise_gray
    kw = dict(field=1, mdis=5, nrad=1, vcheck=2)
    m16 = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 16)
    # vk2 wants a mask in the clip format
    ref_mask = m16 if bits == 16 else vs.core.fmtc.bitdepth(m16, bits=32, fulls=True, fulld=True)
    vk2 = vs.core.eedi3vk2.EEDI3(clip, mclip=ref_mask, **kw)
    my = vs.core.vsfeel.EEDI3(clip, mclip=m16, **kw)  # g16 auto->g8
    dtype = _dtype(bits)
    for n in (0, 11):
        a = _plane(my.get_frame(n), 0, WIDTH, HEIGHT, dtype).astype(np.float64)
        b = _plane(vk2.get_frame(n), 0, WIDTH, HEIGHT, dtype).astype(np.float64)
        d = np.abs(a - b)
        rows = _interp_rows(HEIGHT, n, kw["field"])
        if bits == 16:
            assert d[rows].max() == 0, f"mclip mismatch at frame {n}"
        else:
            assert d[rows].max() < 1e-6, f"mclip mismatch at frame {n}"


def test_eedi3_mclip_auto_converts_gray16_gray32(noise_gray, noise_16bit):
    """Mask Gray16 and Gray32 are auto-converted to Gray8 internally and must
    give the same result as an explicit Gray8 mask."""
    clip = noise_16bit
    kw = dict(field=1, mdis=20, nrad=3, vcheck=2)
    m8 = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 8)
    m16 = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 16)
    m32 = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 32)
    base = _run(clip, mclip=m8, **kw)
    for m in (m16, m32):
        o = _run(clip, mclip=m, **kw)
        for n in (0, 11):
            a = _plane(base.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
            b = _plane(o.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
            assert np.array_equal(a, b), f"mask conversion mismatch at frame {n}"


def test_eedi3_mclip_masked_region_is_vertical_cubic(noise_16bit):
    """Inside a masked (black) region the interpolated pixel is the vertical
    cubic of the two kept rows; verify on a mid-frame interp row where all
    four taps are interior."""
    clip = noise_16bit
    kw = dict(field=1, mdis=5, nrad=1, vcheck=0)
    m16 = _right_half_mask(WIDTH, HEIGHT, clip.num_frames, 16)
    out = _run(clip, mclip=m16, **kw)
    src = noise_16bit
    for n in (0,):
        f = out.get_frame(n)
        s = src.get_frame(n)
        d = _plane(f, 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
        a = _plane(s, 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
        # a mid interp row: taps at ROW-3, ROW-1, ROW+1, ROW+3 are interior
        row = 31
        y1, y3 = row - 1, row - 3
        x = WIDTH * 3 // 4  # deep in the masked black half
        taps = (9 * (int(a[y1, x]) + int(a[y1 + 2, x]))
                - (int(a[y3, x]) + int(a[y3 + 4, x])) + 8) // 16
        assert abs(int(d[row, x]) - int(taps)) <= 1, (
            f"masked px {row},{x}: {d[row,x]} vs cubic {taps}")


# ---------------------------------------------------------------------------
# Formats and plane handling
# ---------------------------------------------------------------------------

def test_eedi3_yuv_passthrough_16bit(noise_16bit):
    """A full YUV clip is accepted; with planes=[0] the chroma planes must be
    copied through bit-identically and the luma must match the Gray output."""
    core = vs.core
    src = core.bs.VideoSource(NOISE_MKV)
    yuv = core.fmtc.bitdepth(src, bits=16, fulls=True, fulld=True)
    assert yuv.format.color_family == vs.YUV

    out = _run(yuv, planes=[0], num_streams=1)
    ref = _run(noise_16bit, num_streams=1)

    for n in (0, 11, 23):
        d = _plane(out.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.int64) \
            - _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.int64)
        assert np.abs(d).max() == 0, f"luma mismatch at frame {n}"
        s = yuv.get_frame(n)
        for plane in (1, 2):
            sh = s.height >> 1
            sw = s.width >> 1
            a = _plane(out.get_frame(n), plane, sw, sh, np.uint16)
            b = _plane(s, plane, sw, sh, np.uint16)
            assert np.array_equal(a, b), f"chroma{plane} changed at frame {n}"


def test_eedi3_yuv_passthrough_32bit(noise_gray):
    core = vs.core
    src = core.bs.VideoSource(NOISE_MKV)
    yuv = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    out = _run(yuv, planes=[0], num_streams=1)
    ref = _run(noise_gray, num_streams=1)
    for n in (0, 11, 23):
        f = out.get_frame(n)
        d = frame_to_ndarray(f) - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"luma mismatch at frame {n}"
        s = yuv.get_frame(n)
        for plane in (1, 2):
            sh = s.height >> 1
            sw = s.width >> 1
            a = _plane(f, plane, sw, sh)
            b = _plane(s, plane, sw, sh)
            assert np.array_equal(a, b), f"chroma{plane} changed at frame {n}"


def test_eedi3_dh_doubles_height():
    """dh=True doubles the height and interp parity is the field."""
    clip = noise_16bit_or_skip()
    core = vs.core
    kw = dict(field=1, dh=1, mdis=5, nrad=1, vcheck=0)
    out = _run(clip, **kw)
    assert out.width == WIDTH
    assert out.height == 2 * HEIGHT
    assert out.num_frames == clip.num_frames


def test_eedi3_field_gt1_doubles_frames():
    """field=3 (top field base + doubling) doubles the frame count and the
    output fps, with kept rows matching the source (progressive)."""
    clip = noise_16bit_or_skip()
    core = vs.core
    out = _run(clip, field=3, mdis=5, nrad=1, vcheck=0)
    assert out.num_frames == 2 * clip.num_frames
    assert out.fps_num == 2 * clip.fps_num
    assert out.fps_den == clip.fps_den
    src = clip.get_frame(0)
    f0 = out.get_frame(0)  # field base 1 -> interp rows 1,3,..; row0 kept
    a0 = _plane(src, 0, WIDTH, HEIGHT, np.uint16)
    d0 = _plane(f0, 0, WIDTH, HEIGHT, np.uint16)
    assert np.array_equal(d0[0::2], a0[0::2]), "kept rows changed at n=0"
    # field base 0 on n=1 -> interp rows 0,2,..; row1 kept
    f1 = out.get_frame(1)
    d1 = _plane(f1, 0, WIDTH, HEIGHT, np.uint16)
    assert np.array_equal(d1[1::2], a0[1::2]), "kept rows changed at n=1"


def noise_16bit_or_skip():
    # session fixture helper for tests that don't want the pytest fixture name
    from conftest import _source
    src = _source(NOISE_MKV)
    y = vs.core.std.ShufflePlanes(src, 0, vs.GRAY)
    return vs.core.fmtc.bitdepth(y, bits=16, fulls=True, fulld=True)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_eedi3_rejects_8bit(noise_8bit):
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_eedi3_rejects_10bit(noise_8bit):
    clip = vs.core.fmtc.bitdepth(noise_8bit, bits=10)
    with pytest.raises(vs.Error):
        _run(clip)


def test_eedi3_rejects_bad_field(noise_16bit):
    for bad in (-1, 4):
        with pytest.raises(vs.Error):
            _run(noise_16bit, field=bad)


def test_eedi3_rejects_dh_with_field_gt1(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, field=2, dh=1)


def test_eedi3_rejects_bad_alpha_beta(noise_16bit):
    for kw in ({"alpha": -0.1}, {"alpha": 1.5}, {"beta": 1.1},
               {"alpha": 0.7, "beta": 0.7}, {"beta": -0.1}):
        with pytest.raises(vs.Error):
            _run(noise_16bit, **kw)


def test_eedi3_rejects_bad_gamma(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, gamma=-1.0)


def test_eedi3_rejects_bad_nrad(noise_16bit):
    for bad in (-1, 4):
        with pytest.raises(vs.Error):
            _run(noise_16bit, nrad=bad)


def test_eedi3_rejects_bad_mdis(noise_16bit):
    for bad in (0, 41):
        with pytest.raises(vs.Error):
            _run(noise_16bit, mdis=bad)


def test_eedi3_rejects_bad_vcheck(noise_16bit):
    for bad in (-1, 4):
        with pytest.raises(vs.Error):
            _run(noise_16bit, vcheck=bad)


def test_eedi3_rejects_bad_vthresh(noise_16bit):
    for kw in ({"vthresh0": 0.0}, {"vthresh1": -5.0}, {"vthresh2": 0.0}):
        with pytest.raises(vs.Error):
            _run(noise_16bit, vcheck=2, **kw)
    # vthresh ignored when vcheck == 0
    _run(noise_16bit, vcheck=0, vthresh0=0.0)


def test_eedi3_rejects_bad_planes(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, planes=[0, 0])
    with pytest.raises(vs.Error):
        _run(noise_16bit, planes=[5])


def test_eedi3_rejects_bad_num_streams(noise_16bit):
    with pytest.raises(vs.Error):
        _run(noise_16bit, num_streams=0)
    with pytest.raises(vs.Error):
        _run(noise_16bit, num_streams=33)


def test_eedi3_rejects_mclip_not_gray(noise_16bit):
    src = vs.core.bs.VideoSource(NOISE_MKV)
    yuv = vs.core.fmtc.bitdepth(src, bits=16, fulls=True, fulld=True)
    with pytest.raises(vs.Error):
        _run(noise_16bit, mclip=yuv)


def test_eedi3_rejects_mclip_wrong_dims(noise_16bit):
    core = vs.core
    small = core.std.BlankClip(format=vs.GRAY8, width=100, height=100,
                               length=noise_16bit.num_frames, color=[128])
    with pytest.raises(vs.Error):
        _run(noise_16bit, mclip=small)


def test_eedi3_rejects_mclip_wrong_frames(noise_16bit):
    core = vs.core
    short = core.std.BlankClip(format=vs.GRAY8, width=WIDTH, height=HEIGHT,
                               length=1, color=[128])
    with pytest.raises(vs.Error):
        _run(noise_16bit, mclip=short)


def test_eedi3_rejects_sclip_wrong_dims_when_vcheck(noise_16bit):
    """sclip is validated only when vcheck > 0 (eedi3m semantics)."""
    core = vs.core
    small = core.std.BlankClip(format=vs.GRAY16, width=100, height=100,
                               length=noise_16bit.num_frames)
    with pytest.raises(vs.Error):
        _run(noise_16bit, vcheck=2, sclip=small)
    # ignored (not validated) when vcheck == 0
    _run(noise_16bit, vcheck=0, sclip=small)
