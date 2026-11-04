"""An all-zero sigma follows each filter's reference, not a shared policy.

GaussBlur rejects it (vszipcl does), BM3D and WNNM return the input clip
unchanged (bm3dvk2 and vszip do), and Bilateral returns a new frame that shares
the source planes. The README documents this split, so pin it per filter.
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import cpu_node, plane_to_ndarray


def _assert_same_pixels(node, clip, frames=(0, 11)):
    out = cpu_node(node)
    for n in frames:
        a = plane_to_ndarray(out.get_frame(n), 0, np.float32)
        b = plane_to_ndarray(clip.get_frame(n), 0, np.float32)
        assert np.array_equal(a, b), f"frame {n} differs from the source"


def test_gaussblur_rejects_zero_sigma(clip_grays):
    with pytest.raises(vs.Error, match="nothing to process"):
        vs.core.vsfeel.GaussBlur(clip_grays, sigma=[0.0])


def test_bm3d_zero_sigma_returns_the_source(clip_grays):
    _assert_same_pixels(vs.core.vsfeel.BM3D(clip_grays, sigma=[0.0]), clip_grays)


def test_wnnm_zero_sigma_returns_the_source(clip_grays):
    _assert_same_pixels(vs.core.vsfeel.WNNM(clip_grays, sigma=[0.0]), clip_grays)


def test_bilateral_zero_sigma_shares_the_source_planes(clip_grays):
    node = vs.core.vsfeel.Bilateral(clip_grays, sigma_spatial=[0.0], sigma_color=[0.0])
    _assert_same_pixels(node, clip_grays)
