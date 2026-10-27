"""RGB parity and plane-subset coverage for the cross-filter paths.

The per-filter files compare Gray/YUV; these cases pin that the same kernels
work on an RGB clip (the R80 API gives every plane the same geometry) and that
a filtered subset leaves the other planes bit-exact.

``reference_compare`` normalizes to sample units: float formats compare in
[0, 1], 16-bit integer formats in codes.
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    assert_preserves_frame_props,
    cpu_node,
    plane_to_ndarray,
    reference_compare,
    reference_spec,
)


CASES = [
    # (filter, reference plugin, reference filter, clip kind, kwargs, bound)
    (
        "Bilateral",
        "vszipcl",
        "Bilateral",
        "rgbs",
        {"sigma_spatial": 3.0, "sigma_color": 0.05},
        1e-6,
    ),
    (
        "Bilateral_h",
        "vszipcl",
        "Bilateral",
        "rgbh",
        {"sigma_spatial": 3.0, "sigma_color": 0.05},
        4.9e-4,
    ),
    (
        "Bilateral_48",
        "vszipcl",
        "Bilateral",
        "rgb48",
        {"sigma_spatial": 3.0, "sigma_color": 0.05},
        1,
    ),
    ("GaussBlur", "vszipcl", "GaussBlur", "rgbs", {"sigma": 2.0}, 0),
    ("GaussBlur_h", "vszipcl", "GaussBlur", "rgbh", {"sigma": 2.0}, 0),
    ("GaussBlur_48", "vszipcl", "GaussBlur", "rgb48", {"sigma": 2.0}, 0),
    ("DFTTest", "vszipcl", "DFTTest", "rgbs", {"tbsize": 1}, 1e-6),
    ("DFTTest_48", "vszipcl", "DFTTest", "rgb48", {"tbsize": 1}, 1),
    ("WNNM", "vszip", "WNNM", "rgbs", {"sigma": [3.0, 3.0, 3.0]}, 8e-3),
    ("EEDI3", "eedi3vk2", "EEDI3", "rgbs", {"field": 1}, 1e-6),
    ("EEDI3_48", "eedi3vk2", "EEDI3", "rgb48", {"field": 1}, 0),
    ("NNEDI3", "nnedi3vk", "NNEDI3", "rgbs", {"field": 1}, 1e-6),
    ("NNEDI3_h", "nnedi3vk", "NNEDI3", "rgbh", {"field": 1}, 0),
    ("NNEDI3_48", "nnedi3vk", "NNEDI3", "rgb48", {"field": 1}, 0),
]


@pytest.mark.parametrize(
    "name,plugin,ref_filter,kind,kwargs,bound",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_rgb_matches_reference(name, plugin, ref_filter, kind, kwargs, bound):
    spec = reference_spec(
        plugin,
        ref_filter,
        kind,
        frames=(0, 11),
        kwargs=kwargs,
        vsfeel_filter=name.split("_")[0],
    )
    payload = reference_compare(spec)
    assert payload["maxdiff"] <= bound, f"{name} max diff {payload}"


# Plane-subset cases: the filtered plane changes and the untouched planes are
# bit-identical to the source. `plane`/`pass` are the filter-specific controls.
_SUBSET = [
    ("DFTTest", {"planes": [0], "tbsize": 1}, 0, (1, 2)),
    ("EEDI3", {"planes": [0], "field": 1}, 0, (1, 2)),
    ("EEDI3H", {"planes": [0], "field": 1}, 0, (1, 2)),
    ("EEDI3AA", {"planes": [0], "field": 3}, 0, (1, 2)),
    ("NNEDI3", {"planes": [0], "field": 1}, 0, (1, 2)),
    ("WNNM", {"sigma": [3.0, 0.0, 0.0]}, 0, (1, 2)),
    ("Bilateral", {"sigma_spatial": [3.0, 0.0, 0.0], "sigma_color": [0.05, 0.0, 0.0]}, 0, (1, 2)),
]


@pytest.mark.parametrize(
    "name,kwargs,pass_plane,other_planes",
    _SUBSET,
    ids=[c[0] for c in _SUBSET],
)
def test_rgb_plane_subset_leaves_other_planes_untouched(
    clip_rgbs, name, kwargs, pass_plane, other_planes
):
    out = cpu_node(vs.core.vsfeel.__getattr__(name)(clip_rgbs, **kwargs))
    for n in (0, 11):
        src = [plane_to_ndarray(clip_rgbs.get_frame(n), p, np.float32) for p in range(3)]
        got = [plane_to_ndarray(out.get_frame(n), p, np.float32) for p in range(3)]
        assert np.abs(got[pass_plane] - src[pass_plane]).max() > 0, (
            f"{name}: plane 0 did not change"
        )
        for p in other_planes:
            assert np.array_equal(got[p], src[p]), f"{name}: plane {p} changed under planes=[0]"


@pytest.mark.parametrize(
    "name,builder",
    [
        ("Box", lambda c: vs.core.vsfeel.Box(c, width=320, height=180)),
        ("WNNM", lambda c: vs.core.vsfeel.WNNM(c, sigma=[3.0], radius=0)),
        ("BM3D", lambda c: vs.core.vsfeel.BM3D(c, sigma=0.7, radius=0)),
    ],
    ids=["Box", "WNNM", "BM3D"],
)
def test_rgb_preserves_frame_props(clip_rgbs, name, builder):
    assert_preserves_frame_props(lambda c: cpu_node(builder(c)), clip_rgbs, frames=(0, 11))
