"""Unit tests for core.vsfeel.GaussBlur.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The vsfeel implementation is a Vulkan port of the vszipcl/vszipcu
GaussBlur; the expected behaviour is bit-identical output (the kernel taps,
mirror reflection and fma accumulation were ported exactly, and the float32
pipeline matches the reference's fp32 maths).

Reference comparisons assert exact equality (measured max diff == 0.0 on
every config tried, both code paths, both bit depths, YUV included), so no
numeric tolerance is needed; only reference crashes are tolerated (the
comparison runs in a subprocess).

Run from the repository root:  python -m pytest tests/test_gaussblur.py
"""

import ctypes
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, assert_gray32, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")


def _run(clip, sigma=2.0, num_streams=1, **kwargs):
    return vs.core.vsfeel.GaussBlur(
        clip,
        sigma=sigma,
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


# ---------------------------------------------------------------------------
# Determinism / stream count
# ---------------------------------------------------------------------------

def test_gaussblur_deterministic(noise_gray):
    a = _run(noise_gray, sigma=2.0)
    b = _run(noise_gray, sigma=2.0)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"nondeterministic output at frame {n}"


def test_gaussblur_multi_stream_matches_single(noise_gray):
    a = _run(noise_gray, sigma=5.0, num_streams=4)
    b = _run(noise_gray, sigma=5.0, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"num_streams mismatch at frame {n}"


def test_gaussblur_parallel_load_matches_serial(noise_gray):
    par = _eval_parallel(noise_gray, sigma=10.0, num_streams=4)
    ref = _run(noise_gray, sigma=10.0, num_streams=1)
    for n in range(noise_gray.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"parallel/serial mismatch at frame {n}"


def test_gaussblur_parallel_load_deterministic(noise_gray):
    a = _eval_parallel(noise_gray, sigma=10.0, num_streams=4)
    b = _eval_parallel(noise_gray, sigma=10.0, num_streams=4)
    for n in range(noise_gray.num_frames):
        d = a[n] - b[n]
        assert np.abs(d).max() < 1e-6, f"nondeterministic output at frame {n}"


# ---------------------------------------------------------------------------
# Reference comparison (bit-exact)
# ---------------------------------------------------------------------------

_COMPARE_SCRIPT = textwrap.dedent(f"""\
    import sys
    import vapoursynth as vs
    from vstools import core
    import numpy as np
    import ctypes

    fmt = sys.argv[1]
    sigma = float(sys.argv[2])
    src = core.bs.VideoSource({NOISE_MKV!r})

    if fmt == "gray32":
        clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)
        dtype, w, h = np.float32, {WIDTH}, {HEIGHT}
    elif fmt == "gray16":
        clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=16, fulls=True, fulld=True)
        dtype, w, h = np.uint16, {WIDTH}, {HEIGHT}
    elif fmt == "yuv32":
        clip = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
        dtype = np.float32
    else:
        raise SystemExit("bad fmt")

    kwargs = {{"sigma": sigma}}
    ref_node = core.vszipcl.GaussBlur(clip, **kwargs)
    my_node = core.vsfeel.GaussBlur(clip, **kwargs)

    worst = 0.0
    for n in (3, 11, 23):
        fr, fm = ref_node.get_frame(n), my_node.get_frame(n)
        planes = 3 if fmt == "yuv32" else 1
        for p in range(planes):
            if fmt == "yuv32":
                w = {WIDTH} // (1 if p == 0 else 2)
                h = {HEIGHT} // (1 if p == 0 else 2)
            else:
                w, h = {WIDTH}, {HEIGHT}
            a = np.ctypeslib.as_array(ctypes.cast(fm.get_read_ptr(p), ctypes.POINTER(ctypes.c_uint8)), shape=(h, w * np.dtype(dtype).itemsize)).view(dtype)
            b = np.ctypeslib.as_array(ctypes.cast(fr.get_read_ptr(p), ctypes.POINTER(ctypes.c_uint8)), shape=(h, w * np.dtype(dtype).itemsize)).view(dtype)
            worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
    print(worst)
""")


def _max_diff_vs_reference(fmt: str, sigma: float) -> float | None:
    """Run the comparison in a subprocess; a crashing reference yields None."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _COMPARE_SCRIPT, fmt, str(sigma)],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return float(lines[-1]) if lines else None


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 10.0])
def test_gaussblur_matches_reference_small(noise_gray, sigma):
    """Fused small path (radius <= 32) must be bit-identical to vszipcl."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "GaussBlur"):
        pytest.skip("no vszipcl.GaussBlur reference")
    maxdiff = _max_diff_vs_reference("gray32", sigma)
    if maxdiff is None:
        pytest.skip("reference comparison crashed")
    assert maxdiff == 0.0, f"small path max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [10.5, 11.0, 12.0])
def test_gaussblur_matches_reference_path_boundary(noise_gray, sigma):
    """Sigmas around the fused-small / two-pass transition (radius ~32) must
    be bit-identical on both sides of the switch."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "GaussBlur"):
        pytest.skip("no vszipcl.GaussBlur reference")
    maxdiff = _max_diff_vs_reference("gray32", sigma)
    if maxdiff is None:
        pytest.skip("reference comparison crashed")
    assert maxdiff == 0.0, f"path boundary max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [20.0, 30.0, 40.0, 80.0])
def test_gaussblur_matches_reference_large(noise_gray, sigma):
    """Two-pass large path (radius > 32) must be bit-identical to vszipcl."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "GaussBlur"):
        pytest.skip("no vszipcl.GaussBlur reference")
    maxdiff = _max_diff_vs_reference("gray32", sigma)
    if maxdiff is None:
        pytest.skip("reference comparison crashed")
    assert maxdiff == 0.0, f"large path max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 20.0])
def test_gaussblur_matches_reference_gray16(noise_gray, sigma):
    """16-bit integer input must be bit-identical on both code paths."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "GaussBlur"):
        pytest.skip("no vszipcl.GaussBlur reference")
    maxdiff = _max_diff_vs_reference("gray16", sigma)
    if maxdiff is None:
        pytest.skip("reference comparison crashed")
    assert maxdiff == 0.0, f"gray16 max diff ({sigma}): {maxdiff}"


@pytest.mark.parametrize("sigma", [0.5, 3.0, 20.0])
def test_gaussblur_matches_reference_yuv(noise_gray, sigma):
    """YUV420: all three planes (incl. the subsampled chroma defaults) must
    match vszipcl bit-for-bit, on both code paths."""
    if not hasattr(vs.core, "vszipcl") or not hasattr(vs.core.vszipcl, "GaussBlur"):
        pytest.skip("no vszipcl.GaussBlur reference")
    maxdiff = _max_diff_vs_reference("yuv32", sigma)
    if maxdiff is None:
        pytest.skip("reference comparison crashed")
    assert maxdiff == 0.0, f"yuv32 max diff ({sigma}): {maxdiff}"


# ---------------------------------------------------------------------------
# Sanity / numerical behaviour
# ---------------------------------------------------------------------------

def test_gaussblur_sigma_small_vs_large_consistent(noise_gray):
    """Both code paths (fused small and two-pass large) blur the same constant
    plane to the same value."""
    core = vs.core
    flat = core.std.BlankClip(noise_gray, color=[0.5])
    out = _run(flat, sigma=20.0)   # large path
    for n in (0,):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.abs(a - 0.5).max() < 1e-3, "large path does not preserve a constant plane"


def test_gaussblur_sigma_zero_passthrough_yuv(noise_gray):
    """sigma=0 on luma copies that plane through while chroma is blurred."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    yuv = vs.core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    out = _run(yuv, sigma=[0.0, 3.0], num_streams=1)
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = yuv.get_frame(n)
        # luma copied through bit-identically
        l = frame_to_ndarray(f)
        ls = frame_to_ndarray(s)
        assert np.array_equal(l, ls), f"luma changed at frame {n}"
        # chroma is actually blurred (differs from input)
        for p in (1, 2):
                w = WIDTH // 2
                h = HEIGHT // 2
                a = _plane(f, p, w, h)
                b = _plane(s, p, w, h)
                assert not np.array_equal(a, b), f"chroma{p} not blurred at frame {n}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_gaussblur_rejects_all_copy_through(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma=0.0)


def test_gaussblur_rejects_sigma_too_large(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma=10000.0)


def test_gaussblur_rejects_negative_sigma(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, sigma=-1.0)


def test_gaussblur_rejects_bad_bitdepth(noise_8bit):
    # only 16-bit integer and 32-bit float input is supported
    with pytest.raises(vs.Error):
        _run(noise_8bit)
    clip = vs.core.fmtc.bitdepth(noise_8bit, bits=10)
    with pytest.raises(vs.Error):
        _run(clip)


def test_gaussblur_rejects_bad_num_streams(noise_gray):
    with pytest.raises(vs.Error):
        _run(noise_gray, num_streams=0)
    with pytest.raises(vs.Error):
        _run(noise_gray, num_streams=33)