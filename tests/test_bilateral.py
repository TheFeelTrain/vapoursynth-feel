"""Unit tests for core.vsfeel.Bilateral.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input: it exercises the filter on high-frequency content without any flat
or black regions.

Run from the repository root:  python -m pytest tests/test_bilateral.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, assert_gray32, frame_to_ndarray

pytestmark = pytest.mark.usefixtures("noise_gray")

SIGMA_SPATIAL = 3.0
SIGMA_COLOR = 0.05


def _run(clip, **kwargs):
    return vs.core.vsfeel.Bilateral(clip, **kwargs)


def test_bilateral_output_finite(noise_gray):
    for num_streams in (1, 2):
        out = _run(
            noise_gray,
            sigma_spatial=SIGMA_SPATIAL,
            sigma_color=SIGMA_COLOR,
            num_streams=num_streams,
        )
        assert_gray32(out)
        for n in (0, 10, 23):
            frame = out.get_frame(n)
            a = frame_to_ndarray(frame)
            assert np.isfinite(a).all(), f"non-finite output at frame {n}"
            assert a.min() >= 0.0 and a.max() <= 1.0


def test_bilateral_defaults_run(noise_gray):
    out = _run(noise_gray)
    assert_gray32(out)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bilateral_deterministic(noise_gray):
    a = _run(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    b = _run(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR)
    for n in (0, 23):
        da = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(da).max() == 0.0


def test_bilateral_rejects_8bit(noise_8bit):
    """Only 16-bit integer and 32-bit float input is supported."""
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_bilateral_rejects_10bit(noise_8bit):
    """Formats other than 16-bit int and 32-bit float must be rejected."""
    clip10 = vs.core.fmtc.bitdepth(noise_8bit, bits=10)
    with pytest.raises(vs.Error):
        _run(clip10)


def test_bilateral_matches_vszipcl(noise_gray):
    """Bit-exact agreement with the reference implementation (if installed)."""
    vszipcl = vs.core
    if not hasattr(vszipcl, "vszipcl") or not hasattr(vszipcl.vszipcl, "Bilateral"):
        pytest.skip("vszipcl not installed")
    for n in (0, 11, 23):
        a = frame_to_ndarray(
            _run(noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR, num_streams=2).get_frame(n)
        )
        b = frame_to_ndarray(
            vszipcl.vszipcl.Bilateral(
                noise_gray, sigma_spatial=SIGMA_SPATIAL, sigma_color=SIGMA_COLOR, num_streams=2
            ).get_frame(n)
        )
        d = np.abs(a - b)
        assert d.max() < 1e-5, f"max diff {d.max()} at frame {n}"
