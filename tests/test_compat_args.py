"""Legacy device_id/num_streams arguments are accepted and ignored.

Every filter registers both for script compatibility. They must not change the
pixels: the core owns the device and sizes the pipeline itself.

Bilateral also keeps vszipcl's positional argument order, so a script ported
from the reference binds the guide where it expects.
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import cpu_node, plane_to_ndarray

KWARGS = {
    "Bilateral": {"sigma_spatial": 3.0, "sigma_color": 0.05},
    "GaussBlur": {"sigma": 2.0},
    "DFTTest": {"tbsize": 1},
    "NLMeans": {"h": 1.5, "d": 0},
    "BM3D": {"sigma": 0.7, "radius": 0},
    "EEDI3": {"field": 1},
    "EEDI3H": {"field": 1},
    "EEDI3AA": {"field": 3},
    "NNEDI3": {"field": 1},
    "WNNM": {"sigma": [3.0], "radius": 0},
    "Box": {"width": 320, "height": 180},
}

# WNNM/BM3D aggregate with atomics; their last bits can differ between runs.
TOL = {"BM3D": 1e-5, "WNNM": 1e-5}


@pytest.mark.parametrize("name", sorted(KWARGS), ids=sorted(KWARGS))
def test_device_args_do_not_change_output(clip_grays, name):
    fn = getattr(vs.core.vsfeel, name)
    default = cpu_node(fn(clip_grays, **KWARGS[name]))
    ignored = cpu_node(fn(clip_grays, **KWARGS[name], device_id=-1, num_streams=0))
    tol = TOL.get(name, 0.0)
    for n in (0, 11):
        a = plane_to_ndarray(default.get_frame(n), 0, np.float32)
        b = plane_to_ndarray(ignored.get_frame(n), 0, np.float32)
        d = float(np.abs(a - b).max())
        assert d <= tol, f"{name}: device args changed the output by {d}"


def test_bilateral_positional_ref_matches_keyword_ref(clip_grays):
    """zipcl's 8th positional argument is the guide, not ``block_x``.

    vszipcl's order is clip, sigma_spatial, sigma_color, radius, device_id,
    num_streams, use_shared_memory, ref; a ported script passes the guide
    there. Binding it to ``block_x`` raised TypeError instead of running.
    """
    guide = vs.core.vsfeel.GaussBlur(clip_grays, sigma=1.5)
    positional = cpu_node(vs.core.vsfeel.Bilateral(clip_grays, 3.0, 0.05, 9, 0, 1, 1, guide))
    keyword = cpu_node(
        vs.core.vsfeel.Bilateral(
            clip_grays, sigma_spatial=3.0, sigma_color=0.05, radius=9, ref=guide
        )
    )
    for n in (0, 11):
        a = plane_to_ndarray(positional.get_frame(n), 0, np.float32)
        b = plane_to_ndarray(keyword.get_frame(n), 0, np.float32)
        assert np.array_equal(a, b), f"positional guide differs at frame {n}"
