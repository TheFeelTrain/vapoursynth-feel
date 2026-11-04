"""Default-parameter parity for the filters whose defaults are not pinned.

The per-filter files mostly test explicit parameters; a reference default that
drifts (or a vsfeel default that no longer matches it) would otherwise pass
unnoticed. BM3D's defaults are checked against the CPU implementation, whose
defaults are separate code.
"""

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    F16_STEP,
    cpu_node,
    format_dtype,
    plane_to_ndarray,
    reference_compare,
    reference_spec,
    source_clip,
)

# The same defaults at every depth the reference can read: u16 output is graded
# in whole codes, half in half steps (the references' f16 store rounding), and
# the f32 rows as before. The YUV420 rows exercise the per-plane chroma default
# (Bilateral's sigma and NLMeans' `channels=auto`) instead of a Gray clip.
CASES = [
    # (name, plugin, reference filter, clip kind, vsfeel kwargs, bound)
    ("Bilateral", "vszipcl", "Bilateral", "grays", {}, 1e-6),
    ("Bilateral", "vszipcl", "Bilateral", "gray16", {}, 1.0),
    ("Bilateral", "vszipcl", "Bilateral", "grayh", {}, F16_STEP),
    ("Bilateral", "vszipcl", "Bilateral", "yuv420p16", {}, 1.0),
    ("GaussBlur", "vszipcl", "GaussBlur", "grays", {}, 0),
    ("GaussBlur", "vszipcl", "GaussBlur", "gray16", {}, 1.0),
    ("GaussBlur", "vszipcl", "GaussBlur", "grayh", {}, F16_STEP),
    ("DFTTest", "vszipcl", "DFTTest", "grays", {}, 1e-6),
    ("DFTTest", "vszipcl", "DFTTest", "gray16", {}, 1.0),
    ("NLMeans", "vszipcl", "NLMeans", "grays", {}, 1e-6),
    ("NLMeans", "vszipcl", "NLMeans", "gray16", {}, 1.0),
    ("NLMeans", "vszipcl", "NLMeans", "grayh", {}, F16_STEP),
    ("NLMeans", "vszipcl", "NLMeans", "yuv420p16", {}, 1.0),
    ("WNNM", "vszip", "WNNM", "grays", {}, 8e-3),
]


@pytest.mark.parametrize(
    "name,plugin,ref_filter,kind,kwargs,bound",
    CASES,
    ids=[f"{c[0]}-{c[3]}" for c in CASES],
)
def test_defaults_match_reference(name, plugin, ref_filter, kind, kwargs, bound):
    spec = reference_spec(plugin, ref_filter, kind, frames=(0, 11), kwargs=kwargs)
    payload = reference_compare(spec)
    assert payload["maxdiff"] <= bound, f"{name} defaults: {payload}"


def test_bm3d_plugin_defaults_are_the_documented_values():
    """Omitting every argument must equal the documented defaults explicitly.

    The defaults are sigma 3, block_step 8, bm_range 9, radius 0, ps_num 2 and
    ps_range 4. A CPU comparison at sigma 3 is dominated by block-matching tie
    flips (measured up to 1e-2), so this pins the argument handling instead;
    the CPU parity of each default is covered by the explicit sweeps.
    """
    src = source_clip()
    gray = vs.core.fmtc.bitdepth(
        vs.core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True
    )
    clip = vs.core.std.Trim(vs.core.std.Crop(gray, right=576, bottom=296), length=3)
    default = cpu_node(vs.core.vsfeel.BM3D(clip))
    explicit = cpu_node(
        vs.core.vsfeel.BM3D(
            clip,
            sigma=[3.0, 3.0, 3.0],
            block_step=[8, 8, 8],
            bm_range=[9, 9, 9],
            radius=0,
            ps_num=[2, 2, 2],
            ps_range=[4, 4, 4],
        )
    )
    for n in (0, 1):
        a = plane_to_ndarray(default.get_frame(n), 0, np.float32)
        b = plane_to_ndarray(explicit.get_frame(n), 0, np.float32)
        assert np.abs(a - b).max() <= 1e-6, f"BM3D defaults differ at frame {n}"


def _assert_same_as_explicit(clip, default, explicit, tol=0.0):
    dt = format_dtype(clip.format)
    default = cpu_node(default)
    explicit = cpu_node(explicit)
    for n in (0, 1):
        a = plane_to_ndarray(default.get_frame(n), 0, dt).astype(np.float64)
        b = plane_to_ndarray(explicit.get_frame(n), 0, dt).astype(np.float64)
        d = float(np.abs(a - b).max())
        assert d <= tol, f"defaults differ from explicit values at frame {n} by {d}"


def test_dfttest_defaults_are_the_documented_values_f16(clip_grayh):
    """The half path (which the reference cannot read) parses the same
    documented defaults as an explicit call; f32/u16 parity is covered by the
    reference sweep above."""
    _assert_same_as_explicit(
        clip_grayh,
        vs.core.vsfeel.DFTTest(clip_grayh),
        vs.core.vsfeel.DFTTest(
            clip_grayh,
            tbsize=3,
            ftype=0,
            sigma=8.0,
            sigma2=8.0,
            pmin=0.0,
            pmax=500.0,
            sbsize=16,
            sosize=12,
            swin=0,
            twin=7,
            sbeta=2.5,
            tbeta=2.5,
            zmean=1,
            f0beta=1.0,
            ssystem=0,
        ),
    )


@pytest.mark.parametrize("fixture", ["clip_gray16", "clip_grayh"])
def test_wnnm_defaults_are_the_documented_values(request, fixture):
    """vszip's WNNM is f32-only, so the u16 and half paths are pinned by
    comparing the omitted defaults with the same values passed explicitly.

    Two instances accumulate with atomics, so the comparison is bounded by the
    same ordering floor as the reference comparison, not exact equality.
    """
    clip = request.getfixturevalue(fixture)
    tol = 1.0 if clip.format.sample_type == vs.INTEGER else 1e-3
    explicit = dict(
        sigma=[3.0],
        block_size=8,
        block_step=8,
        group_size=8,
        bm_range=7,
        radius=0,
        ps_num=2,
        ps_range=4,
        residual=0,
        adaptive_aggregation=1,
    )
    _assert_same_as_explicit(
        clip,
        vs.core.vsfeel.WNNM(clip),
        vs.core.vsfeel.WNNM(clip, **explicit),
        tol=tol,
    )
