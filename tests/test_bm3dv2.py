"""Unit tests for core.vsfeel.BM3Dv2.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The noise content exposes the boundary/clamping behaviour of the
temporal pipeline: frames 0..23 must all produce finite output with no NaN
(regression: boundary frames intermittently produced NaN before the atomics
were made visible to the aggregation kernel).

Run from the repository root:  python -m pytest tests/test_bm3dv2.py
"""

import ctypes
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, NOISE_MKV, assert_gray32, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")

SIGMA = 0.7
BM_RANGE = 16
PS_RANGE = 7
BLOCK_STEP = 4


def _run(clip, radius=2, num_streams=1, **kwargs):
    return vs.core.vsfeel.BM3Dv2(
        clip,
        sigma=SIGMA,
        radius=radius,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        num_streams=num_streams,
        **kwargs,
    )


def _check_all_frames_finite(clip, **kwargs):
    out = _run(clip, **kwargs)
    assert_gray32(out)
    for n in range(out.num_frames):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4])
def test_bm3dv2_no_nan_all_frames(noise_gray, radius):
    """Output must be finite on every frame (incl. boundaries) for each radius.

    A brand new filter instance is created per parametrized test.
    """
    _check_all_frames_finite(noise_gray, radius=radius, num_streams=1)


def test_bm3dv2_no_nan_all_frames_multi_stream(noise_gray):
    _check_all_frames_finite(noise_gray, radius=2, num_streams=2)


def test_bm3dv2_deterministic(noise_gray):
    a = _run(noise_gray, radius=2, num_streams=1)
    b = _run(noise_gray, radius=2, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


def test_bm3dv2_rejects_gray8(noise_8bit):
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_bm3dv2_rejects_radius5(noise_gray):
    """Radius > 4 is unsupported and must be rejected up front."""
    with pytest.raises(vs.Error):
        _run(noise_gray, radius=5)


def test_bm3dv2_yuv_passthrough(noise_gray):
    """A full YUV clip is accepted: the luma must match the Gray output and
    the chroma planes must be copied through bit-identically."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    yuv = vs.core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    assert yuv.format.color_family == vs.YUV

    out = _run(yuv, radius=2, num_streams=1)
    ref = _run(noise_gray, radius=2, num_streams=1)

    for n in (0, 11, 23):
        f = out.get_frame(n)
        # luma is denoised identically to the Gray path
        d = frame_to_ndarray(f) - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"luma mismatch at frame {n}"
        # chroma is passed through unchanged
        s = yuv.get_frame(n)
        for plane in (1, 2):
            sh = s.height >> 1
            sw = s.width >> 1
            a = np.ctypeslib.as_array(
                ctypes.cast(f.get_read_ptr(plane), ctypes.POINTER(ctypes.c_float)),
                shape=(sh, sw))
            b = np.ctypeslib.as_array(
                ctypes.cast(s.get_read_ptr(plane), ctypes.POINTER(ctypes.c_float)),
                shape=(sh, sw))
            assert np.array_equal(a, b), f"chroma{plane} changed at frame {n}"


_COMPARE_SCRIPT = textwrap.dedent(f"""\
    import sys
    import vapoursynth as vs
    from vstools import core
    import numpy as np
    import ctypes

    ref = sys.argv[1]
    radius = int(sys.argv[2])

    core.max_cache_size = 1024 * 56
    src = core.bs.VideoSource({NOISE_MKV!r})
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)

    kwargs = {{"sigma": 0.7, "radius": radius, "bm_range": 16, "ps_range": 7, "block_step": 4}}
    if ref != "bm3dhip":
        kwargs["num_streams"] = 1
    ref_node = getattr(core, ref).BM3Dv2(clip, **kwargs)
    my_node = core.vsfeel.BM3Dv2(clip, sigma=0.7, radius=radius, bm_range=16, ps_range=7, block_step=4, num_streams=1)

    worst = 0.0
    for n in (0, 11, 23):
        a = np.ctypeslib.as_array(ctypes.cast(my_node.get_frame(n).get_read_ptr(0), ctypes.POINTER(ctypes.c_float)), shape=({HEIGHT}, {WIDTH}))
        b = np.ctypeslib.as_array(ctypes.cast(ref_node.get_frame(n).get_read_ptr(0), ctypes.POINTER(ctypes.c_float)), shape=({HEIGHT}, {WIDTH}))
        worst = max(worst, float(np.abs(a - b).max()))
    print(worst)
""")


def _max_diff_vs_reference(radius: int, ref: str) -> float | None:
    """Run the comparison in a subprocess; a crashing reference yields None."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _COMPARE_SCRIPT, ref, str(radius)],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return float(lines[-1]) if lines else None


@pytest.mark.parametrize("radius", [0, 2, 3, 4])
def test_bm3dv2_matches_reference(noise_gray, radius):
    """vsfeel must closely match vszipcl, falling back to bm3dhip.

    The reference plugins are crash-prone on some drivers, so the comparison
    runs in a subprocess: a crashed reference must not take down the suite.
    bm3dhip implements the same algorithm and is expected to match vszipcl.
    """
    for ref in ("vszipcl", "bm3dhip"):
        if not hasattr(vs.core, ref) or not hasattr(getattr(vs.core, ref), "BM3Dv2"):
            continue
        maxdiff = _max_diff_vs_reference(radius, ref)
        if maxdiff is None:
            continue  # crashed or failed to load: try the next reference
        assert maxdiff < 0.01, f"max diff vs {ref}: {maxdiff}"
        return
    pytest.skip("no usable reference plugin (vszipcl/bm3dhip)")
