"""CPU- vs GPU-resident guide clips, and invalid-guide rejection.

A guide (Bilateral/NLMeans `ref`/`rclip`, WNNM/BM3D `rclip`/`ref`) normally
comes from an earlier GPU filter, so it is GPU resident. The same guide
downloaded to the CPU must produce the same pixels; only the transport differs.
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import cpu_node, plane_to_ndarray


def _guide_pair(clip):
    gpu = vs.core.vsfeel.GaussBlur(clip, sigma=1.5)
    cpu = vs.core.std.GPUDownload(gpu)
    return cpu, gpu


CASES = [
    ("Bilateral", dict(sigma_spatial=3.0, sigma_color=0.05), "ref"),
    ("NLMeans", dict(h=1.5, d=0), "rclip"),
    ("WNNM", dict(sigma=[3.0], radius=0), "rclip"),
    ("BM3D", dict(sigma=0.7, radius=0), "ref"),
]


@pytest.mark.parametrize("name,kwargs,guide_kwarg", CASES, ids=[c[0] for c in CASES])
def test_gpu_resident_guide_matches_cpu_guide(clip_grays, name, kwargs, guide_kwarg):
    cpu_guide, gpu_guide = _guide_pair(clip_grays)
    fn = getattr(vs.core.vsfeel, name)
    a = cpu_node(fn(clip_grays, **kwargs, **{guide_kwarg: cpu_guide}))
    b = cpu_node(fn(clip_grays, **kwargs, **{guide_kwarg: gpu_guide}))
    for n in (0, 11):
        pa = plane_to_ndarray(a.get_frame(n), 0, np.float32)
        pb = plane_to_ndarray(b.get_frame(n), 0, np.float32)
        assert np.abs(pa - pb).max() <= 1e-5, f"{name}: guide residency changed the output"


@pytest.mark.parametrize("kind", ["format", "size", "length"])
@pytest.mark.parametrize("name,kwargs,guide_kwarg", CASES, ids=[c[0] for c in CASES])
def test_guide_mismatch_is_rejected(clip_grays, name, kwargs, guide_kwarg, kind):
    """Every guide-taking filter rejects every kind of mismatch.

    A guide must match the source's format, dimensions and frame count; each
    filter used to cover only some of the three.
    """
    guide = vs.core.vsfeel.GaussBlur(clip_grays, sigma=1.5)
    bad = {
        "format": vs.core.resize.Point(guide, format=vs.GRAY16),
        "size": vs.core.std.Crop(guide, right=8),
        "length": vs.core.std.Trim(guide, length=2),
    }[kind]
    fn = getattr(vs.core.vsfeel, name)
    with pytest.raises(vs.Error):
        fn(clip_grays, **kwargs, **{guide_kwarg: bad}).get_frame(11)
