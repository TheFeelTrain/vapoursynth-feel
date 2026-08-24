"""Duck-typed backend selectors for the vsfeel VapourSynth plugin.

These plug into the unmodified ``backend=`` arguments of vs-jetpack functions;
see the package docstring in ``__init__.py`` for usage examples.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["Backend", "FeelBackend"]


class FeelBackend:
    """Duck-typed backend accepted by vs-jetpack's ``backend=`` arguments."""

    value = "vsfeel"
    """Plugin namespace. Read by wrappers that pick the plugin themselves,
    e.g. ``vsrgtools.gauss_blur`` dispatches through ``getattr(core, backend.value)``."""

    num_streams = 2
    """Matches the stream count the vszipcl/vszipcu backends are given by
    vs-jetpack; override per call with the plugin's own ``num_streams``
    keyword."""

    def resolve(self) -> Self:
        """Resolve this backend to itself.

        vs-jetpack enums implement ``resolve()`` to map their AUTO member onto
        the function's default; an explicitly chosen backend always resolves to
        itself.
        """
        return self

    def _dispatch(self, func: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> vs.VideoNode:
        return getattr(args[0].vsfeel, func)(*args[1:], **{"num_streams": self.num_streams} | kwargs)


class _FeelBM3DPlugin:
    """Stand-in for the ``core.vsfeel`` plugin surface.

    Exposes ``BM3Dv2`` with a signature extended by the parameters other
    BM3Dv2 plugins accept (``chroma``, ...); arguments vsfeel does not support
    are silently dropped before calling the real function.
    """

    def __init__(self) -> None:
        import vapoursynth as vs

        func = vs.core.vsfeel.BM3Dv2
        accepted = frozenset(func.__signature__.parameters)
        sig = func.__signature__
        chroma = inspect.Parameter("chroma", inspect.Parameter.KEYWORD_ONLY, default=False)

        def bm3d_v2(*args: Any, **kwargs: Any) -> vs.VideoNode:
            return func(*args, **{k: v for k, v in kwargs.items() if k in accepted})

        bm3d_v2.__signature__ = sig.replace(parameters=[*sig.parameters.values(), chroma])  # type: ignore[attr-defined]
        self.BM3Dv2 = bm3d_v2


class FeelBilateral(FeelBackend):
    """Selects ``core.vsfeel.Bilateral``, e.g. for ``vsrgtools.bilateral``."""

    def Bilateral(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self._dispatch("Bilateral", (clip, *args), kwargs)


class FeelGaussBlur(FeelBackend):
    """Selects ``core.vsfeel.GaussBlur``, e.g. for ``vsrgtools.gauss_blur``.

    No dispatch method needed: ``gauss_blur`` looks the plugin up itself via
    ``getattr(core, backend.value).GaussBlur(...)``.
    """

    num_streams = 4
    """The stream count ``gauss_blur`` gives its GPU backends."""


class FeelNLMeans(FeelBackend):
    """Selects ``core.vsfeel.NLMeans``, e.g. for ``vsdenoise.nl_means``."""

    def NLMeans(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self._dispatch("NLMeans", (clip, *args), kwargs)


class FeelDFTTest(FeelBackend):
    """Selects ``core.vsfeel.DFTTest`` for ``vsdenoise.DFTTest``.

    Although the other DFTTest backends target the dfttest2 python package,
    ``denoise()`` emits classic dfttest-style keyword arguments
    (tbsize/ftype/swin/twin/slocation/ssx/ssy/sst/planes), which
    ``core.vsfeel.DFTTest`` accepts directly; dfttest2-only options are not
    available through this backend.
    """

    def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self._dispatch("DFTTest", (clip, *args), kwargs)


class FeelBM3D(FeelBackend):
    """Selects ``core.vsfeel.BM3Dv2``, e.g. for ``vsdenoise.bm3d``.

    ``bm3d`` obtains the plugin through ``backend.plugin`` and introspects its
    signature to filter profile arguments, so a shim is interposed that hides
    the parameters vsfeel does not implement (``chroma`` and friends).
    """

    def __init__(self) -> None:
        self._plugin: _FeelBM3DPlugin | None = None

    @property
    def plugin(self) -> _FeelBM3DPlugin:
        if self._plugin is None:
            self._plugin = _FeelBM3DPlugin()
        return self._plugin


class Backend:
    """Namespace of vsfeel backends for vs-jetpack functions.

    Pass these as the ``backend=`` argument of the matching vs-jetpack wrapper.
    """

    Bilateral = FeelBilateral()
    GaussBlur = FeelGaussBlur()
    NLMeans = FeelNLMeans()
    BM3D = FeelBM3D()
    DFTTest = FeelDFTTest()
