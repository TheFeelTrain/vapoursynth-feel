"""Smoke tests for the vsfeel Python package.

Filter behaviour is covered by the per-filter test modules; these only verify
that the package imports and its duck-typed backend runs through the
unmodified vs-jetpack wrappers.
"""

import dataclasses

import numpy as np
import pytest
import vapoursynth as vs

from conftest import cpu_node, frame_to_ndarray

pytest.importorskip("vstools")
pytest.importorskip("vsaa")

from vsaa import EEDI3, NNEDI3  # noqa: E402
from vsaa import based_aa  # noqa: E402
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
    out = cpu_node(bilateral(noise_gray, sigmaS=3.0, sigmaR=0.02, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_nl_means_runs_via_jetpack(noise_gray):
    out = cpu_node(nl_means(noise_gray, h=1.2, tr=1, a=2, s=4, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_gauss_blur_runs_via_jetpack(noise_gray):
    out = cpu_node(gauss_blur(noise_gray, 1.5, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bm3d_runs_via_jetpack(noise_gray):
    out = cpu_node(bm3d(noise_gray, 0.7, tr=2, profile=bm3d.Profile.FAST, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_dfttest_runs_via_jetpack(noise_gray):
    dft = DFTTest(noise_gray, backend=_backend())
    out = dft.denoise({0.0: 16.0, 0.5: 8.0, 1.0: 0.0}, tr=1)
    assert np.isfinite(frame_to_ndarray(cpu_node(out).get_frame(0))).all()


def test_eedi3_runs_via_vsaa(noise_gray):
    out = cpu_node(EEDI3(backend=_backend()).antialias(noise_gray))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_eedi3h_fallback_matches_native(noise_gray):
    native = cpu_node(EEDI3(backend=_backend()).antialias(
        noise_gray, direction=EEDI3.AADirection.HORIZONTAL))

    class _NoH(vsfeel.FeelBackend):
        supports_h = False

    fallback = cpu_node(EEDI3(backend=_NoH()).antialias(
        noise_gray, direction=EEDI3.AADirection.HORIZONTAL))
    assert np.array_equal(frame_to_ndarray(native.get_frame(0)), frame_to_ndarray(fallback.get_frame(0)))


def test_eedi3aa_subclass_is_a_vsaa_eedi3():
    """vsfeel.EEDI3 is a lazy subclass, so based_aa's isinstance checks pass."""
    assert issubclass(vsfeel.EEDI3, EEDI3)
    # The vsfeel antialiaser defaults to the vsfeel backend, so based_aa needs
    # no backend= argument (an explicit one still wins).
    assert vsfeel.EEDI3().backend is vsfeel.Backend
    assert vsfeel.EEDI3(backend=EEDI3.Backend.CPU).backend is EEDI3.Backend.CPU
    # `import vsfeel` must not require vsaa: the subclass is built on access.
    assert vsfeel.__all__ == ["Backend", "FeelBackend", "EEDI3", "NNEDI3"]


def test_nnedi3_subclass_is_a_vsaa_nnedi3():
    """vsfeel.NNEDI3 keeps the reference's field surface exactly."""
    assert issubclass(vsfeel.NNEDI3, NNEDI3)
    assert [f.name for f in dataclasses.fields(vsfeel.NNEDI3)] == [
        f.name for f in dataclasses.fields(NNEDI3)]
    assert vsfeel.NNEDI3(nsize=3, nns=2, pscrn=1).copy(nsize=4).nsize == 4
    # vs.core.vsfeel.NNEDI3 is a fresh Function object per access, so compare
    # the plugin namespace rather than object identity.
    func = vsfeel.NNEDI3()._deinterlacer_function
    assert (func.plugin.namespace, func.name) == ("vsfeel", "NNEDI3")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"nsize": 1, "nns": 2, "qual": 1, "etype": 1, "pscrn": 0},
        {"nsize": 5, "nns": 0, "pscrn": 3},
    ],
)
def test_nnedi3_via_vsaa_matches_direct_call(noise_16bit, kwargs):
    """The wrapper adds only field/dh mapping over core.vsfeel.NNEDI3."""
    wrapped = cpu_node(vsfeel.NNEDI3(**kwargs).deinterlace(
        noise_16bit, tff=True, double_rate=False))
    direct = cpu_node(vs.core.vsfeel.NNEDI3(
        noise_16bit, field=1,
        nsize=kwargs.get("nsize", 0), nns=kwargs.get("nns", 4),
        qual=kwargs.get("qual", 2), etype=kwargs.get("etype", 0),
        pscrn=kwargs.get("pscrn", 4)))
    for n in (0, 7):
        assert np.array_equal(frame_to_ndarray(wrapped.get_frame(n), dtype=np.uint16),
                              frame_to_ndarray(direct.get_frame(n), dtype=np.uint16))


def test_nnedi3_supersample_doubles_dims(noise_16bit):
    out = cpu_node(vsfeel.NNEDI3().scale(noise_16bit, 2 * noise_16bit.width, 2 * noise_16bit.height))
    assert (out.width, out.height) == (2 * noise_16bit.width, 2 * noise_16bit.height)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0), dtype=np.uint16)).all()


def test_based_aa_with_vsfeel_supersampler_and_antialiaser(noise_gray):
    """The documented drop-in combo runs end to end."""
    out = cpu_node(based_aa(
        noise_gray, supersampler=vsfeel.NNEDI3(), antialiaser=vsfeel.EEDI3(),
        postfilter=False))
    assert (out.width, out.height) == (noise_gray.width, noise_gray.height)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_eedi3aa_matches_the_two_call_chain(noise_16bit):
    """The fused antialiaser reproduces the base class's chain bit-exactly."""
    fused = cpu_node(vsfeel.EEDI3(backend=_backend(), mdis=5, nrad=1).antialias(noise_16bit))
    chain = cpu_node(EEDI3(backend=_backend(), mdis=5, nrad=1).antialias(noise_16bit))
    assert fused.num_frames == chain.num_frames == noise_16bit.num_frames
    for n in (0, 5, 23):
        a = frame_to_ndarray(fused.get_frame(n), dtype=np.uint16)
        b = frame_to_ndarray(chain.get_frame(n), dtype=np.uint16)
        assert np.array_equal(a, b), f"fused chain mismatch at frame {n}"


def test_eedi3aa_falls_back_for_non_both(noise_16bit):
    """direction != BOTH keeps the base class's single-direction path."""
    fused = cpu_node(vsfeel.EEDI3(backend=_backend()).antialias(
        noise_16bit, direction=EEDI3.AADirection.HORIZONTAL))
    chain = cpu_node(EEDI3(backend=_backend()).antialias(
        noise_16bit, direction=EEDI3.AADirection.HORIZONTAL))
    for n in (0, 11):
        assert np.array_equal(frame_to_ndarray(fused.get_frame(n), dtype=np.uint16),
                              frame_to_ndarray(chain.get_frame(n), dtype=np.uint16))


def test_backend_context_routes_singletons(noise_gray):
    old_bilateral, old_gauss = bilateral.backend, gauss_blur.backend
    with _backend()():
        assert bilateral.backend is _backend()
        assert gauss_blur.backend is _backend()
        # implicit backend= (AUTO -> singleton) now routes through vsfeel
        assert np.isfinite(frame_to_ndarray(cpu_node(
            bilateral(noise_gray, sigmaS=3.0, sigmaR=0.02)).get_frame(0))).all()
        assert np.isfinite(frame_to_ndarray(cpu_node(
            gauss_blur(noise_gray, 1.5)).get_frame(0))).all()
    assert bilateral.backend == old_bilateral
    assert gauss_blur.backend == old_gauss
