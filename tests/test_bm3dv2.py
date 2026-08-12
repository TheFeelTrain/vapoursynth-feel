"""Unit tests for core.vsfeel.BM3Dv2.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The noise content exposes the boundary/clamping behaviour of the
temporal pipeline: frames 0..23 must all produce finite output with no NaN
(regression: boundary frames intermittently produced NaN before the atomics
were made visible to the aggregation kernel).

Run from the repository root:  python -m pytest tests/test_bm3dv2.py
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import WIDTH, HEIGHT, assert_gray32, frame_to_ndarray

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


def test_bm3dv2_no_nan_all_frames(noise_gray):
    """Radius-2 output must be finite on every frame (incl. boundaries)."""
    _check_all_frames_finite(noise_gray, radius=2, num_streams=1)


def test_bm3dv2_no_nan_all_frames_multi_stream(noise_gray):
    _check_all_frames_finite(noise_gray, radius=2, num_streams=2)


def test_bm3dv2_radius0_finite(noise_gray):
    _check_all_frames_finite(noise_gray, radius=0, num_streams=1)


def test_bm3dv2_radius1_finite(noise_gray):
    _check_all_frames_finite(noise_gray, radius=1, num_streams=1)


def test_bm3dv2_deterministic(noise_gray):
    a = _run(noise_gray, radius=2, num_streams=1)
    b = _run(noise_gray, radius=2, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


def test_bm3dv2_rejects_gray8(noise_8bit):
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_bm3dv2_matches_vszipcl_radius0(noise_gray):
    vszipcl = vs.core
    if not hasattr(vszipcl, "vszipcl") or not hasattr(vszipcl.vszipcl, "BM3Dv2"):
        pytest.skip("vszipcl not installed")
    for n in (0, 11, 23):
        a = frame_to_ndarray(_run(noise_gray, radius=0, num_streams=1).get_frame(n))
        b = frame_to_ndarray(
            vszipcl.vszipcl.BM3Dv2(
                noise_gray,
                sigma=SIGMA,
                radius=0,
                bm_range=BM_RANGE,
                ps_range=PS_RANGE,
                block_step=BLOCK_STEP,
                num_streams=1,
            ).get_frame(n)
        )
        d = np.abs(a - b)
        assert d.max() < 0.01 and d.mean() < 1e-3, f"max diff {d.max()} at frame {n}"


def test_bm3dv2_matches_vszipcl_radius2(noise_gray):
    vszipcl = vs.core
    if not hasattr(vszipcl, "vszipcl") or not hasattr(vszipcl.vszipcl, "BM3Dv2"):
        pytest.skip("vszipcl not installed")
    for n in (0, 11, 22, 23):
        a = frame_to_ndarray(_run(noise_gray, radius=2, num_streams=1).get_frame(n))
        b = frame_to_ndarray(
            vszipcl.vszipcl.BM3Dv2(
                noise_gray,
                sigma=SIGMA,
                radius=2,
                bm_range=BM_RANGE,
                ps_range=PS_RANGE,
                block_step=BLOCK_STEP,
                num_streams=1,
            ).get_frame(n)
        )
        d = np.abs(a - b)
        assert d.max() < 0.01 and d.mean() < 1e-3, f"max diff {d.max()} at frame {n}"
