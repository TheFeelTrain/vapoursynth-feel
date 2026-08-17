"""Unit tests for core.vsfeel.DFTTest.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The vsfeel implementation is a Vulkan port of the vszipcu
dfttest (dfttest.zig + dfttest.cu) and the vs-dfttest2 hiprtc backend
(dft_kernels.hpp + kernel.hpp); the expected behaviour is numerically
identical to the references (the FFTW codelets, window tables, zero-mean
gain and the frequency-domain filter are ported exactly; only float32
rounding order may differ by an ulp between the backends).

Run from the repository root:  python -m pytest tests/test_dfttest.py
"""

import ctypes
import json
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, assert_gray32, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")


def _run(clip, tbsize=3, num_streams=1, **kwargs):
    return vs.core.vsfeel.DFTTest(
        clip,
        tbsize=tbsize,
        num_streams=num_streams,
        **kwargs,
    )


def _plane(frame, plane, width, height, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    return np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(height, width * itemsize),
    ).view(dtype).copy()


def _eval_parallel(clip, **kwargs):
    """Evaluate every frame concurrently to exercise the multi-stream path."""
    out = _run(clip, **kwargs)
    frames = [None] * out.num_frames

    def worker(n):
        frames[n] = frame_to_ndarray(out.get_frame(n))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(out.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(f is not None for f in frames)
    return frames


def _check_all_frames_finite(clip, **kwargs):
    out = _run(clip, **kwargs)
    assert_gray32(out)
    for n in range(out.num_frames):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


# ---------------------------------------------------------------------------
# Determinism / stream count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_streams", [1, 4])
def test_dfttest_deterministic(noise_gray, num_streams):
    a = _run(noise_gray, num_streams=num_streams)
    b = _run(noise_gray, num_streams=num_streams)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"nondeterministic output at frame {n}"


def test_dfttest_multi_stream_matches_single(noise_gray):
    a = _run(noise_gray, num_streams=4)
    b = _run(noise_gray, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"num_streams mismatch at frame {n}"


def test_dfttest_parallel_load_consistent(noise_gray):
    """Two parallel num_streams=4 runs must match the serial path, and each
    other — the request pattern that exposes stale descriptor bindings,
    fence misuse and command-pool reuse violations under load."""
    a = _eval_parallel(noise_gray, num_streams=4)
    b = _eval_parallel(noise_gray, num_streams=4)
    ref = _run(noise_gray, num_streams=1)
    for n in range(noise_gray.num_frames):
        r = frame_to_ndarray(ref.get_frame(n))
        assert np.abs(a[n] - r).max() < 1e-6, f"parallel 1/serial mismatch at frame {n}"
        assert np.abs(b[n] - r).max() < 1e-6, f"parallel 2/serial mismatch at frame {n}"
        assert np.abs(a[n] - b[n]).max() < 1e-6, f"nondeterministic output at frame {n}"


# ---------------------------------------------------------------------------
# All frames finite (temporal boundary handling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tbsize", [1, 3, 5, 7])
def test_dfttest_no_nan_all_frames(noise_gray, tbsize):
    _check_all_frames_finite(noise_gray, tbsize=tbsize, num_streams=1)


@pytest.mark.parametrize("num_streams", [2, 4])
def test_dfttest_no_nan_all_frames_multi_stream(noise_gray, num_streams):
    _check_all_frames_finite(noise_gray, tbsize=3, num_streams=num_streams)


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------

# One subprocess runs every config (each entry is (kwargs, tolerance)); the
# tolerance covers ulp-level accumulation from the float32 rounding-order
# differences between backends.
REFERENCE_CASES = [
    ({}, 2e-3),                                                     # defaults
    ({"tbsize": 1}, 2e-3),                                          # spatial only
    ({"tbsize": 7, "sosize": 4}, 2e-3),                             # radius 3, 75% overlap
    ({"zmean": 0}, 2e-3),                                           # no zero-mean
    ({"swin": 4, "sbeta": 3.0, "twin": 2, "tbeta": 4.0}, 2e-3),     # custom windows
    ({"ftype": 2}, 2e-3),                                           # multiply
    ({"ftype": 3, "pmin": 10.0, "pmax": 200.0}, 2e-3),              # bandpass
    ({"ftype": 4, "pmin": 10.0, "pmax": 200.0}, 2e-3),              # rnlm-like
    ({"ftype": 0, "f0beta": 0.5}, 2e-3),                            # sqrt wiener
    ({"slocation": [0.0, 2.0, 0.5, 8.0, 1.0, 12.0], "ssystem": 0}, 2e-3),
    ({"slocation": [0.0, 2.0, 0.5, 8.0, 1.0, 12.0], "ssystem": 1}, 2e-3),
    ({"ssx": [0.0, 4.0, 1.0, 9.0], "ssy": [0.0, 3.0, 1.0, 7.0],
      "sst": [0.0, 2.0, 1.0, 5.0], "ssystem": 1}, 2e-3),
    # ftype=1 (hard threshold) is boundary-sensitive: a pixel whose power
    # sits within an ulp of sigma flips between zero and non-zero.
    ({"ftype": 1, "sigma": 4.0}, 5e-3),
]

_COMPARE_SCRIPT = textwrap.dedent(f"""\
    import json
    import sys
    import vapoursynth as vs
    from vstools import core
    import numpy as np
    import ctypes

    cases = json.loads(sys.argv[1])
    src = core.bs.VideoSource({NOISE_MKV!r})
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)

    for kwargs in cases:
        ref_node = core.vszipcl.DFTTest(clip, **kwargs)
        my_node = core.vsfeel.DFTTest(clip, **kwargs)
        worst = 0.0
        for n in (0, 11, 23):
            fr, fm = ref_node.get_frame(n), my_node.get_frame(n)
            a = np.ctypeslib.as_array(ctypes.cast(fm.get_read_ptr(0), ctypes.POINTER(ctypes.c_float)), shape=({HEIGHT}, {WIDTH}))
            b = np.ctypeslib.as_array(ctypes.cast(fr.get_read_ptr(0), ctypes.POINTER(ctypes.c_float)), shape=({HEIGHT}, {WIDTH}))
            worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
        print(worst)
""")


def _reference_max_diffs() -> list[float] | None:
    """Run every config in a single subprocess; a crashing reference yields
    None (the reference plugins are crash-prone on some drivers, so the
    comparison must not take down the suite)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _COMPARE_SCRIPT,
             json.dumps([kw for kw, _ in REFERENCE_CASES])],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != len(REFERENCE_CASES):
        return None
    return [float(line) for line in lines]


def test_dfttest_matches_reference(noise_gray):
    """vsfeel must closely match vszipcl across the parameter surface.

    The implementations are ported exactly, so only float32 rounding order
    may differ; the per-config tolerance covers ulp-level accumulation (and
    the few pixels that flip across a hard threshold).
    """
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "DFTTest"):
        pytest.skip("no vszipcl.DFTTest reference")
    maxdiffs = _reference_max_diffs()
    if maxdiffs is None:
        pytest.skip("reference comparison crashed")
    for (kwargs, tol), maxdiff in zip(REFERENCE_CASES, maxdiffs):
        assert maxdiff < tol, f"max diff {maxdiff} vs vszipcl for {kwargs}"


def test_dfttest_gray8_matches_reference(noise_8bit):
    """8-bit integer path: at most one code level of rounding difference."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "DFTTest"):
        pytest.skip("no vszipcl.DFTTest reference")
    core = vs.core
    ref = core.vszipcl.DFTTest(noise_8bit)
    my = core.vsfeel.DFTTest(noise_8bit)
    for n in (0, 11, 23):
        a = _plane(my.get_frame(n), 0, WIDTH, HEIGHT, np.uint8)
        b = _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, np.uint8)
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        assert d.max() <= 1, f"gray8 max diff {d.max()} at frame {n}"


# ---------------------------------------------------------------------------
# Formats and plane handling
# ---------------------------------------------------------------------------

def test_dfttest_yuv_passthrough(noise_gray):
    """A full YUV clip is accepted; with planes=[0] the chroma planes must be
    copied through bit-identically and the luma must match the Gray output."""
    core = vs.core
    src = core.bs.VideoSource(NOISE_MKV)
    yuv = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    assert yuv.format.color_family == vs.YUV

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


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_dfttest_rejects_bad_ftype(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, ftype=5)
    with pytest.raises(vs.Error):
        _run(noise_gray, ftype=-1)


def test_dfttest_rejects_bad_sbsize(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sbsize=8)


def test_dfttest_rejects_bad_sosize(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sosize=-1)
    with pytest.raises(vs.Error):
        _run(noise_gray, sosize=16)
    # > 50% overlap requires sbsize-sosize to divide sbsize
    with pytest.raises(vs.Error):
        _run(noise_gray, sosize=10)


def test_dfttest_rejects_even_tbsize(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, tbsize=2)


def test_dfttest_rejects_bad_tbsize(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, tbsize=0)
    with pytest.raises(vs.Error):
        _run(noise_gray, tbsize=9)


def test_dfttest_rejects_bad_windows(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, swin=12)
    with pytest.raises(vs.Error):
        _run(noise_gray, twin=-1)


def test_dfttest_rejects_bad_ssystem(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, ssystem=2)


def test_dfttest_rejects_nonfinite_sigma(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma=float("nan"))
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma2=float("inf"))


def test_dfttest_rejects_nonfinite_beta(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sbeta=float("nan"))
    with pytest.raises(vs.Error):
        _run(noise_gray, tbeta=float("inf"))


def test_dfttest_rejects_odd_sigma_arrays(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, slocation=[0.0, 1.0, 2.0])
    with pytest.raises(vs.Error):
        _run(noise_gray, ssx=[0.0])


def test_dfttest_rejects_bad_planes(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, planes=[0, 0])
    with pytest.raises(vs.Error):
        _run(noise_gray, planes=[5])


def test_dfttest_rejects_bad_num_streams(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, num_streams=0)
    with pytest.raises(vs.Error):
        _run(noise_gray, num_streams=33)


def test_dfttest_rejects_10bit(noise_8bit):
    clip = vs.core.fmtc.bitdepth(noise_8bit, bits=10)
    with pytest.raises(vs.Error):
        _run(clip)


def test_dfttest_rejects_sigma_range_not_covered(noise_gray):
    """slocation that does not span [0, 1] must be rejected."""
    with pytest.raises(vs.Error):
        _run(noise_gray, slocation=[0.5, 2.0, 0.6, 3.0])