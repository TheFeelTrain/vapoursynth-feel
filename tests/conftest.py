import os

import ctypes

import numpy as np
import pytest
import vapoursynth as vs

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
NOISE_MKV = os.path.join(TESTS_DIR, "noise_24f.mkv")
WIDTH = 640
HEIGHT = 360


def _source(path):
    core = vs.core
    if hasattr(core, "bs"):
        return core.bs.VideoSource(path)
    if hasattr(core, "ffms2"):
        return core.ffms2.Source(path)
    raise RuntimeError("no source plugin available (need bs or ffms2)")


def frame_to_ndarray(frame, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    arr = np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(0), ctypes.POINTER(ctypes.c_uint8)),
        shape=(HEIGHT, WIDTH * itemsize),
    ).view(dtype)
    return arr.copy()


@pytest.fixture(scope="session")
def noise_gray():
    """GrayS float32 clip of the committed 24-frame random-noise video."""
    src = _source(NOISE_MKV)
    y = vs.core.std.ShufflePlanes(src, 0, vs.GRAY)
    return vs.core.fmtc.bitdepth(y, bits=32, fulls=True, fulld=True)


@pytest.fixture(scope="session")
def noise_8bit():
    """GRAY8 clip of the same video (for input-validation tests)."""
    src = _source(NOISE_MKV)
    return vs.core.std.ShufflePlanes(src, 0, vs.GRAY)


@pytest.fixture(scope="session")
def noise_16bit():
    """GRAY16 clip of the same video (integer reference comparison)."""
    src = _source(NOISE_MKV)
    y = vs.core.std.ShufflePlanes(src, 0, vs.GRAY)
    return vs.core.fmtc.bitdepth(y, bits=16, fulls=True, fulld=True)


def assert_gray32(clip):
    fmt = clip.format
    assert fmt.color_family == vs.GRAY
    assert fmt.sample_type == vs.FLOAT
    assert fmt.bits_per_sample == 32
    assert clip.width == WIDTH
    assert clip.height == HEIGHT
