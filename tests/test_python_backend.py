"""Smoke tests for the vsfeel Python package.

Filter behaviour is covered by the per-filter test modules; these only verify
that the package imports and its duck-typed backend runs through the
unmodified vs-jetpack wrappers.
"""

import dataclasses
import inspect
import os
import subprocess
import sys
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import CLIP_PATH, cpu_node, frame_to_ndarray, plane_to_ndarray

pytest.importorskip("vstools")
pytest.importorskip("vsaa")

from vsaa import EEDI3, NNEDI3
from vsaa import based_aa
from vsdenoise import bm3d, nl_means
from vsdenoise.fft import DFTTest
from vsrgtools import bilateral, gauss_blur

import vsfeel
from vsfeel.backend import _FeelBM3DPlugin


def _backend():
    """Indirection keeps type-checkers quiet about the intentional duck typing."""
    return vsfeel.Backend


# The wrapper's no-argument fields are based_aa's own antialiaser settings, not
# vsaa's bare EEDI3 ones; every fused-vs-chain comparison below must name them
# explicitly or it compares two different filters.
AA_PARAMS = dict(alpha=0.125, beta=0.25, gamma=40.0, vthresh=(12.0, 24.0, 4.0))


def test_backend_resolves():
    assert vsfeel.Backend.resolve() is vsfeel.Backend


def test_bilateral_runs_via_jetpack(clip_gray):
    out = cpu_node(bilateral(clip_gray, sigmaS=3.0, sigmaR=0.02, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_nl_means_runs_via_jetpack(clip_gray):
    out = cpu_node(nl_means(clip_gray, h=1.2, tr=1, a=2, s=4, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_gauss_blur_runs_via_jetpack(clip_gray):
    out = cpu_node(gauss_blur(clip_gray, 1.5, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bm3d_runs_via_jetpack(clip_gray):
    out = cpu_node(bm3d(clip_gray, 0.7, tr=2, profile=bm3d.Profile.FAST, backend=_backend()))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_dfttest_runs_via_jetpack(clip_gray):
    dft = DFTTest(clip_gray, backend=_backend())
    out = dft.denoise({0.0: 16.0, 0.5: 8.0, 1.0: 0.0}, tr=1)
    assert np.isfinite(frame_to_ndarray(cpu_node(out).get_frame(0))).all()


def test_eedi3_runs_via_vsaa(clip_gray):
    out = cpu_node(EEDI3(backend=_backend()).antialias(clip_gray))
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_eedi3h_fallback_matches_native(clip_gray):
    native = cpu_node(
        EEDI3(backend=_backend()).antialias(clip_gray, direction=EEDI3.AADirection.HORIZONTAL)
    )

    class _NoH(vsfeel.FeelBackend):
        supports_h = False

    fallback = cpu_node(
        EEDI3(backend=_NoH()).antialias(clip_gray, direction=EEDI3.AADirection.HORIZONTAL)
    )
    assert np.array_equal(
        frame_to_ndarray(native.get_frame(0)), frame_to_ndarray(fallback.get_frame(0))
    )


def test_eedi3h_fallback_transposes_aux_clips(clip_16bit):
    """The no-H fallback transposes sclip and mclip, not just the source.

    Compares the fallback against the native transpose-oracle chain with a
    distinct 2N-frame sclip and a non-None mclip, frame for frame.
    """
    from conftest import half_mask

    clip = clip_16bit
    # Single-direction interpolation keeps N frames, so the sclip is N-frame
    # here (the 2N form only exists for the fused double-rate path).
    sclip = clip.std.Invert()
    mclip = half_mask(clip, 16)
    kw = dict(
        mdis=5, nrad=1, vcheck=2, alpha=0.125, beta=0.25, gamma=40.0, vthresh=(12.0, 24.0, 4.0)
    )

    class _NoH(vsfeel.FeelBackend):
        supports_h = False

    fallback = cpu_node(
        EEDI3(backend=_NoH(), **kw).antialias(
            clip, direction=EEDI3.AADirection.HORIZONTAL, sclip=sclip, mclip=mclip
        )
    )
    # The fallback's own shape: transpose everything, run the vertical
    # interpolation, transpose back. Rebuilt here so that removing either
    # .std.Transpose() call in FeelBackend.transpose fails the comparison.
    t = vs.core.std.Transpose(clip)
    ts = sclip.std.Transpose()
    tm = mclip.std.Transpose()
    expect = cpu_node(
        EEDI3(backend=_backend(), **kw).antialias(
            t, direction=EEDI3.AADirection.VERTICAL, sclip=ts, mclip=tm
        )
    )
    expect = vs.core.std.Transpose(expect)
    assert fallback.num_frames == expect.num_frames == clip.num_frames
    for n in (0, 5, clip.num_frames - 1):
        a = frame_to_ndarray(fallback.get_frame(n), dtype=np.uint16)
        b = frame_to_ndarray(expect.get_frame(n), dtype=np.uint16)
        assert np.array_equal(a, b), f"aux-clip fallback mismatch at frame {n}"


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
        f.name for f in dataclasses.fields(NNEDI3)
    ]
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
def test_nnedi3_via_vsaa_matches_direct_call(clip_16bit, kwargs):
    """The wrapper adds only field/dh mapping over core.vsfeel.NNEDI3."""
    wrapped = cpu_node(vsfeel.NNEDI3(**kwargs).deinterlace(clip_16bit, tff=True, double_rate=False))
    direct = cpu_node(
        vs.core.vsfeel.NNEDI3(
            clip_16bit,
            field=1,
            nsize=kwargs.get("nsize", 0),
            nns=kwargs.get("nns", 4),
            qual=kwargs.get("qual", 2),
            etype=kwargs.get("etype", 0),
            pscrn=kwargs.get("pscrn", 4),
        )
    )
    for n in (0, 7):
        assert np.array_equal(
            frame_to_ndarray(wrapped.get_frame(n), dtype=np.uint16),
            frame_to_ndarray(direct.get_frame(n), dtype=np.uint16),
        )


def test_nnedi3_supersample_doubles_dims(clip_16bit):
    out = cpu_node(vsfeel.NNEDI3().scale(clip_16bit, 2 * clip_16bit.width, 2 * clip_16bit.height))
    assert (out.width, out.height) == (2 * clip_16bit.width, 2 * clip_16bit.height)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0), dtype=np.uint16)).all()


def test_based_aa_with_vsfeel_supersampler_and_antialiaser(clip_gray):
    """The documented drop-in combo runs end to end."""
    out = cpu_node(
        based_aa(
            clip_gray, supersampler=vsfeel.NNEDI3(), antialiaser=vsfeel.EEDI3(), postfilter=False
        )
    )
    assert (out.width, out.height) == (clip_gray.width, clip_gray.height)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_eedi3aa_matches_the_two_call_chain(clip_16bit):
    """The fused antialiaser reproduces the base class's chain bit-exactly."""
    fused = cpu_node(
        vsfeel.EEDI3(backend=_backend(), mdis=5, nrad=1, **AA_PARAMS).antialias(clip_16bit)
    )
    chain = cpu_node(EEDI3(backend=_backend(), mdis=5, nrad=1, **AA_PARAMS).antialias(clip_16bit))
    assert fused.num_frames == chain.num_frames == clip_16bit.num_frames
    for n in (0, 5, 23):
        a = frame_to_ndarray(fused.get_frame(n), dtype=np.uint16)
        b = frame_to_ndarray(chain.get_frame(n), dtype=np.uint16)
        assert np.array_equal(a, b), f"fused chain mismatch at frame {n}"


def test_eedi3aa_sclip_subframes_are_distinct(clip_16bit):
    """The fused 2N-frame sclip interleave keeps distinct sub-frames distinct.

    Feeds an N-frame sclip whose content differs from the source (inverted
    clip) through the wrapper: the wrapper's Interleave([s, s]) doubling must
    carry that content into both fused sub-frames, exactly like the base
    class's chain. A wrong doubling (dropped or reordered) fails bit-exactly.
    """
    clip = clip_16bit
    inv = clip.std.Invert()
    kw = dict(mdis=5, nrad=1, **AA_PARAMS)
    fused = cpu_node(vsfeel.EEDI3(backend=_backend(), sclip=inv, **kw).antialias(clip))
    chain = cpu_node(EEDI3(backend=_backend(), sclip=inv, **kw).antialias(clip))
    assert fused.num_frames == chain.num_frames == clip.num_frames
    for n in (0, 5, 23):
        a = frame_to_ndarray(fused.get_frame(n), dtype=np.uint16)
        b = frame_to_ndarray(chain.get_frame(n), dtype=np.uint16)
        assert np.array_equal(a, b), f"distinct-sclip mismatch at frame {n}"
    # The inverted sclip must actually change the output vs no sclip
    # (sanity that the content is meaningful, not a vacuous equality).
    plain = cpu_node(vsfeel.EEDI3(backend=_backend(), **kw).antialias(clip))
    changed = any(
        not np.array_equal(
            frame_to_ndarray(fused.get_frame(n), dtype=np.uint16),
            frame_to_ndarray(plain.get_frame(n), dtype=np.uint16),
        )
        for n in (0, 5, 23)
    )
    assert changed, "distinct sclip did not affect the output (test is vacuous)"


def test_wrapper_drops_params_the_plugin_rejects(clip_16bit, monkeypatch):
    """A parameter vsfeel's entry points never declare is filtered, not fatal.

    VapourSynth rejects an unknown keyword before dispatch, so the fused and
    the chain path must both drop it instead of failing to build. (``hp`` used
    to be that parameter; the plugin declares it now, so its forwarding is
    pinned by ``test_eedi3_wrapper_forwards_hp``.)
    """
    import vsaa.deinterlacers as _deinterlacers

    real = _deinterlacers.EEDI3.get_deint_args

    def with_bogus(self, **kwargs):
        return real(self, **kwargs) | {"vsfeel_no_such_param": False}

    monkeypatch.setattr(_deinterlacers.EEDI3, "get_deint_args", with_bogus)

    fused = cpu_node(vsfeel.EEDI3(**AA_PARAMS).antialias(clip_16bit))
    chain = cpu_node(EEDI3(backend=_backend(), **AA_PARAMS).antialias(clip_16bit))

    for out in (fused, chain):
        assert np.isfinite(frame_to_ndarray(out.get_frame(0), dtype=np.uint16)).all()


def test_eedi3_wrapper_forwards_hp(clip_16bit):
    """vs-jetpack sends EEDI3's ``hp``; every vsfeel entry point declares it.

    The wrapper filtered ``hp`` out while the plugin did not declare it, so
    each path is compared against the plugin call driven with ``hp``: the
    chain against the base class's vertical pass plus its double-rate merge
    (a bare vs-jetpack EEDI3 keeps its own ``vthresh`` default), and the fused
    path against ``EEDI3AA``. ``hp=False`` must differ from ``hp=True`` or the
    comparison would be vacuous.
    """
    clip = clip_16bit
    # (alpha, beta, gamma, nrad, mdis) and the two vthresh defaults in play:
    # a bare vsaa EEDI3 carries (32, 64, 4), vsfeel's fused wrapper (12, 24, 4).
    bare = dict(alpha=0.125, beta=0.25, gamma=40.0, nrad=1, mdis=5)
    bare_vth = dict(vthresh0=32.0, vthresh1=64.0, vthresh2=4.0)
    fused_vth = dict(vthresh0=12.0, vthresh1=24.0, vthresh2=4.0)

    fused = cpu_node(vsfeel.EEDI3(hp=True, mdis=5, nrad=1, **AA_PARAMS).antialias(clip))
    want_fused = cpu_node(vs.core.vsfeel.EEDI3AA(clip, 3, False, hp=1, **bare, **fused_vth))

    raw = vs.core.vsfeel.EEDI3(clip, 3, False, hp=1, **bare, **bare_vth)
    want_chain = cpu_node(vs.core.std.Merge(raw[::2], raw[1::2]))
    chain = cpu_node(
        EEDI3(backend=_backend(), hp=True, **bare).antialias(
            clip, direction=EEDI3.AADirection.VERTICAL
        )
    )
    plain = cpu_node(
        EEDI3(backend=_backend(), hp=False, **bare).antialias(
            clip, direction=EEDI3.AADirection.VERTICAL
        )
    )

    for n in (0, 5, 23):
        a = frame_to_ndarray(fused.get_frame(n), dtype=np.uint16)
        b = frame_to_ndarray(want_fused.get_frame(n), dtype=np.uint16)
        assert np.array_equal(a, b), f"fused path did not forward hp at frame {n}"
        c = frame_to_ndarray(chain.get_frame(n), dtype=np.uint16)
        d = frame_to_ndarray(want_chain.get_frame(n), dtype=np.uint16)
        assert np.array_equal(c, d), f"chain did not forward hp at frame {n}"
        e = frame_to_ndarray(plain.get_frame(n), dtype=np.uint16)
        assert not np.array_equal(c, e), f"hp did not change the output at frame {n}"


class _StubBM3D:
    """A ``core.vsfeel.BM3Dv2`` stand-in with an explicit signature.

    ``_drop_unsupported`` and the wrapper read ``__signature__``, which a plain
    Python callable does not set, so the stub provides one like the plugin does.
    """

    def __init__(self, declares_chroma: bool) -> None:
        self.calls: list[dict[str, object]] = []
        params = [
            inspect.Parameter("clip", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("sigma", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
        if declares_chroma:
            params.append(
                inspect.Parameter("chroma", inspect.Parameter.KEYWORD_ONLY, default=False)
            )
        self.__signature__ = inspect.Signature(params)

    def __call__(self, clip, **kwargs):
        self.calls.append(kwargs)
        return clip


def test_bm3d_wrapper_keeps_chroma_off_an_older_plugin():
    """A build whose BM3Dv2 has no ``chroma`` argument must not be sent one.

    The wrapper advertises ``chroma`` so vsdenoise still sees the reference's
    surface, but the *native* signature decides what is forwarded: the older
    build rejects the keyword as unsupported, so forwarding ``chroma=0`` broke
    every ordinary BM3D call.
    """
    clip = object()
    old = _StubBM3D(declares_chroma=False)
    wrapped = _FeelBM3DPlugin(old).BM3Dv2
    wrapped(clip, sigma=0.7)
    wrapped(clip, sigma=0.7, chroma=True)
    assert old.calls == [{"sigma": 0.7}, {"sigma": 0.7}]
    assert "chroma" in inspect.signature(wrapped).parameters

    new = _StubBM3D(declares_chroma=True)
    wrapped = _FeelBM3DPlugin(new).BM3Dv2
    wrapped(clip, sigma=0.7)
    wrapped(clip, sigma=0.7, chroma=True)
    assert new.calls == [{"sigma": 0.7, "chroma": 0}, {"sigma": 0.7, "chroma": 1}]


# Two environment generations in one process, the way vsview's reload drives
# them: the first build caches the module singletons, then its environment dies
# and a second one runs the same script.
_RELOAD_SCRIPT = """
import sys
sys.path.insert(0, {repo!r})

import vapoursynth as vs


class _Policy(vs.EnvironmentPolicy):
    def __init__(self):
        self._current = None

    def on_policy_registered(self, api):
        self._api = api
        self._current = None

    def on_policy_cleared(self):
        self._api = None

    def get_current_environment(self):
        return self._current

    def set_environment(self, environment):
        self._current = environment

    def is_alive(self, environment):
        return environment is self._current


pol = _Policy()
vs.register_policy(pol)
try:
    for generation in range(2):
        env = pol._api.create_environment()
        wrapped = pol._api.wrap_environment(env)
        try:
            with wrapped.use():
                import vsfeel
                from vsdenoise import bm3d
                from vsdenoise.blockmatch import BM3D

                clip = vs.core.std.BlankClip(format=vs.GRAYS, width=64, height=64, length=3)
                bm3d(clip, 0.7, 1, profile=BM3D.Profile.FAST, backend=vsfeel.Backend)
        finally:
            pol._api.destroy_environment(env)
finally:
    pol._api.unregister_policy()

print("RELOAD-OK")
"""


def test_bm3d_backend_survives_an_environment_reload():
    """The Backend singleton must not pin the environment it first ran in.

    vsview's reload destroys the VapourSynth environment while keeping imported
    modules cached. ``Backend.plugin`` used to cache the resolved
    ``core.vsfeel.BM3Dv2`` Function, which kept the destroyed core alive, so the
    next ``bm3d(...)`` through the wrapper raised "Use of invalidated Core (the
    environment has been destroyed)". ``vsdenoise`` reaches the function through
    ``backend.plugin.BM3Dv2`` on every call, so this has to resolve against the
    live core.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(vsfeel.__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", _RELOAD_SCRIPT.format(repo=repo)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"reload subprocess failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )
    assert "RELOAD-OK" in proc.stdout


def test_eedi3aa_falls_back_for_non_both(clip_16bit):
    """direction != BOTH keeps the base class's single-direction path."""
    fused = cpu_node(
        vsfeel.EEDI3(backend=_backend(), **AA_PARAMS).antialias(
            clip_16bit, direction=EEDI3.AADirection.HORIZONTAL
        )
    )
    chain = cpu_node(
        EEDI3(backend=_backend(), **AA_PARAMS).antialias(
            clip_16bit, direction=EEDI3.AADirection.HORIZONTAL
        )
    )
    for n in (0, 11):
        assert np.array_equal(
            frame_to_ndarray(fused.get_frame(n), dtype=np.uint16),
            frame_to_ndarray(chain.get_frame(n), dtype=np.uint16),
        )


def test_eedi3aa_no_arg_defaults_match_based_aa(clip_gray, monkeypatch):
    """`vsfeel.EEDI3()` must be the antialiaser ``based_aa`` builds for itself.

    ``based_aa`` only builds its own EEDI3 when the caller passes none, so
    capture exactly what it passes there and require the wrapper's no-argument
    fields to reproduce all of it. Without this, ``antialiaser=vsfeel.EEDI3()``
    silently ran different EEDI3 parameters than ``backend=vsfeel.Backend``.
    """
    import vsaa.funcs as _funcs

    captured = {}
    real = _funcs.EEDI3

    class _Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(_funcs, "EEDI3", _Spy)
    based_aa(clip_gray, supersampler=False, backend=_backend())
    assert captured, "based_aa did not build its default EEDI3"

    wrapper = vsfeel.EEDI3()
    for field, value in captured.items():
        assert getattr(wrapper, field) == value, field


def test_eedi3aa_falls_back_for_odd_geometry(clip_16bit, monkeypatch):
    """EEDI3AA needs both plane axes even, so odd geometry keeps the chain."""
    import vsaa.deinterlacers as _deinterlacers

    calls: list[int] = []

    def spy(self, clip, direction=None, **kwargs):
        calls.append(clip.width)
        return clip

    monkeypatch.setattr(_deinterlacers.EEDI3, "antialias", spy)
    odd = vs.core.std.Crop(clip_16bit, right=1)
    assert odd.width % 2 == 1
    out = vsfeel.EEDI3(backend=_backend()).antialias(odd)
    assert calls == [odd.width], "odd width must not take the fused path"
    assert out is odd


def test_eedi3aa_fused_path_requires_a_vsfeel_backend(clip_gray, monkeypatch):
    """An explicit CPU or reference backend keeps the base class's chain.

    EEDI3AA is vsfeel's fused implementation, so taking it when the caller
    selected another backend would silently run a different filter.
    """
    import vsaa.deinterlacers as _deinterlacers

    calls: list[object] = []

    def spy(self, clip, direction=None, **kwargs):
        calls.append(direction)
        return clip

    monkeypatch.setattr(_deinterlacers.EEDI3, "antialias", spy)

    out = vsfeel.EEDI3(backend=EEDI3.Backend.CPU).antialias(clip_gray)
    assert calls and out is clip_gray, "CPU backend must not take the fused path"

    calls.clear()
    vsfeel.EEDI3(backend=_backend()).antialias(clip_gray)
    assert not calls, "vsfeel backend must take the fused path"


def test_backend_context_routes_singletons(clip_gray):
    old_bilateral, old_gauss = bilateral.backend, gauss_blur.backend
    with _backend()():
        assert bilateral.backend is _backend()
        assert gauss_blur.backend is _backend()
        # implicit backend= (AUTO -> singleton) now routes through vsfeel
        assert np.isfinite(
            frame_to_ndarray(cpu_node(bilateral(clip_gray, sigmaS=3.0, sigmaR=0.02)).get_frame(0))
        ).all()
        assert np.isfinite(
            frame_to_ndarray(cpu_node(gauss_blur(clip_gray, 1.5)).get_frame(0))
        ).all()
    assert bilateral.backend == old_bilateral
    assert gauss_blur.backend == old_gauss


def test_bm3d_wrapper_forwards_chroma(clip_gray):
    """vsdenoise forces chroma=True on YUV444; the wrapper must forward it.

    The plugin's BM3Dv2 implements the reference's joint 4:4:4 entry, so a
    chroma=True request runs it instead of being rejected (the wrapper used to
    raise while the plugin denoised luma only). The clip is a real 4:4:4
    conversion of the noise source: a Gray clip resized to 4:4:4 has constant
    (neutral) chroma, which no filter can move. Gray keeps the default path.
    """
    from vsdenoise import bm3d as _bm3d

    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(CLIP_PATH), format=vs.YUV444PS)
    out = cpu_node(_bm3d(yuv444, 0.7, tr=2, profile=_bm3d.Profile.FAST, backend=_backend()))
    for plane in range(3):
        a = plane_to_ndarray(out.get_frame(0), plane)
        b = plane_to_ndarray(yuv444.get_frame(0), plane)
        assert np.isfinite(a).all(), f"non-finite plane {plane}"
        assert float(np.abs(a - b).max()) > 1e-4, f"plane {plane} was not denoised"
    # An explicit chroma=False is a vsdenoise-level duplicate (it forces its
    # own geometry-derived value too), so cover the pass-through at the
    # wrapper entry point instead: Gray keeps the default path running.
    gray_out = _bm3d(clip_gray, 0.7, tr=2, profile=_bm3d.Profile.FAST, backend=_backend())
    assert np.isfinite(frame_to_ndarray(cpu_node(gray_out).get_frame(0))).all()


def test_backend_context_is_thread_safe():
    """Threads serialize on the context instead of cross-restoring.

    The vsrgtools singletons are process-global; without the lock one thread
    can restore the CPU backend while the other is still inside, leaking
    vsfeel past the block. Each thread holds the block in turn and must see
    vsfeel throughout; the defaults must restore afterwards.
    """
    old_bilateral, old_gauss = bilateral.backend, gauss_blur.backend
    errors: list[str] = []

    def worker(i):
        try:
            with _backend()():
                assert bilateral.backend is _backend()
                assert gauss_blur.backend is _backend()
        except BaseException as exc:  # deliberate: collected and reported below
            errors.append(f"thread {i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not [t for t in threads if t.is_alive()], "context threads hung"
    assert not errors, "; ".join(errors)
    assert bilateral.backend == old_bilateral
    assert gauss_blur.backend == old_gauss


def test_backend_context_nesting_restores_in_order():
    """Nesting composes: each level restores its own previous value."""
    old_bilateral, old_gauss = bilateral.backend, gauss_blur.backend
    outer = vsfeel.FeelBackend()
    with outer():
        assert bilateral.backend is outer
        with _backend()():
            assert bilateral.backend is _backend()
        assert bilateral.backend is outer
    assert bilateral.backend == old_bilateral
    assert gauss_blur.backend == old_gauss
