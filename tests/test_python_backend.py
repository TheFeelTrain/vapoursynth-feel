"""Smoke tests for the vsfeel Python package.

Filter behaviour is covered by the per-filter test modules; these only verify
that the package imports and its duck-typed backend runs through the
unmodified vs-jetpack wrappers.
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import frame_to_ndarray

pytest.importorskip("vstools")

from vsdenoise import bm3d, nl_means  # noqa: E402
from vsdenoise.fft import DFTTest  # noqa: E402
from vsrgtools import bilateral, gauss_blur  # noqa: E402

import vsfeel  # noqa: E402


def _backend():
    """Indirection keeps type-checkers quiet about the intentional duck typing."""
    return vsfeel.Backend


def test_backend_resolves():
    assert vsfeel.Backend.resolve() is vsfeel.Backend


def test_bilateral_runs_via_jetpack(noise_gray):
    out = bilateral(noise_gray, sigmaS=3.0, sigmaR=0.02, backend=_backend())
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_nl_means_runs_via_jetpack(noise_gray):
    out = nl_means(noise_gray, h=1.2, tr=1, a=2, s=4, backend=_backend())
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_gauss_blur_runs_via_jetpack(noise_gray):
    out = gauss_blur(noise_gray, 1.5, backend=_backend())
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bm3d_runs_via_jetpack(noise_gray):
    out = bm3d(noise_gray, 0.7, tr=2, profile=bm3d.Profile.FAST, backend=_backend())
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_dfttest_runs_via_jetpack(noise_gray):
    dft = DFTTest(noise_gray, backend=_backend())
    out = dft.denoise({0.0: 16.0, 0.5: 8.0, 1.0: 0.0}, tr=1)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()
