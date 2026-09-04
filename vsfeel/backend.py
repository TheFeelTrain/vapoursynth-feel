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


class FeelBackend:
    """Duck-typed backend accepted by vs-jetpack's ``backend=`` arguments.

    One instance serves every vs-jetpack wrapper: each wrapper only ever
    invokes the entry point matching its own filter, so the same object can be
    passed to ``vsrgtools.bilateral``, ``vsdenoise.bm3d``, ... alike.
    """

    value = "vsfeel"
    """Plugin namespace. Read by wrappers that pick the plugin themselves,
    e.g. ``vsrgtools.gauss_blur`` dispatches through ``getattr(core, backend.value)``."""

    supports_mclip = True
    """Whether the backend's EEDI3 supports mclip (read by vsaa.based_aa)."""

    supports_h = True
    """Whether the backend can interpolate columns (EEDI3H)."""

    num_streams: int | None = None
    """Override for the plugin's ``num_streams`` argument.

    ``None`` (the default) leaves the argument unset, so every filter uses its
    own plugin default (DFTTest/NLMeans 1, Bilateral/BM3Dv2 4). Set this to an
    int to force one value for all filters, or pass ``num_streams=`` to a
    wrapper for a single call — an explicit wrapper keyword always wins."""

    def resolve(self) -> Self:
        """Resolve this backend to itself.

        vs-jetpack enums implement ``resolve()`` to map their AUTO member onto
        the function's default; an explicitly chosen backend always resolves to
        itself.
        """
        return self

    def _dispatch(self, func: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> vs.VideoNode:
        if "num_streams" not in kwargs and self.num_streams is not None:
            kwargs = {**kwargs, "num_streams": self.num_streams}
        return getattr(args[0].vsfeel, func)(*args[1:], **kwargs)

    def Bilateral(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:  # noqa: N802
        return self._dispatch("Bilateral", (clip, *args), kwargs)

    def NLMeans(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:  # noqa: N802
        return self._dispatch("NLMeans", (clip, *args), kwargs)

    def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:  # noqa: N802
        return self._dispatch("DFTTest", (clip, *args), kwargs)

    def EEDI3(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:  # noqa: N802
        return self._dispatch("EEDI3", (clip, *args), kwargs)

    def EEDI3H(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:  # noqa: N802
        return self._dispatch("EEDI3H", (clip, *args), kwargs)

    @property
    def plugin(self) -> _FeelBM3DPlugin:
        """Plugin surface for wrappers that look the function up themselves,
        i.e. ``vsdenoise.bm3d`` through ``backend.plugin.BM3Dv2``."""
        if self._plugin is None:
            self._plugin = _FeelBM3DPlugin()
        return self._plugin

    def __init__(self) -> None:
        self._plugin: _FeelBM3DPlugin | None = None


Backend = FeelBackend()
