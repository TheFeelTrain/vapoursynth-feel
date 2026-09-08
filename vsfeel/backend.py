"""Duck-typed backend selectors for the vsfeel VapourSynth plugin.

These plug into the unmodified ``backend=`` arguments of vs-jetpack functions;
see the package docstring in ``__init__.py`` for usage examples.
"""

from __future__ import annotations

import inspect
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
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

    @property
    def should_h(self) -> bool:
        """Whether to interpolate columns natively (mirrors ``EEDI3.Backend.should_h``).

        vsfeel ships ``EEDI3H``, so this is just ``supports_h`` — the
        ``!= CPU`` half of the reference expression never applies here.
        """
        return self.supports_h

    @staticmethod
    def transpose(
        clip: vs.VideoNode,
        *,
        sclip: vs.VideoNode | None = None,
        mclip: vs.VideoNode | None = None,
        **kwargs: Any,
    ) -> tuple[vs.VideoNode, MutableMapping[str, vs.VideoNode | None]]:
        """Transpose the clip plus aux clips (mirrors ``EEDI3.Backend.transpose``).

        Fallback used when ``should_h`` is false; vsfeel always takes the
        native ``EEDI3H`` path, so this only runs for subclasses that flip
        ``supports_h`` off.
        """
        import vapoursynth as vs

        if isinstance(sclip, vs.VideoNode):
            sclip = sclip.std.Transpose()

        if isinstance(mclip, vs.VideoNode):
            mclip = mclip.std.Transpose()

        return clip.std.Transpose(), kwargs | {"sclip": sclip, "mclip": mclip}

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

    def EEDI3(  # noqa: N802
        self,
        clip: vs.VideoNode,
        field: int,
        *args: Any,
        sclip: vs.VideoNode | None = None,
        mclip: vs.VideoNode | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """Run ``core.vsfeel.EEDI3`` (mirrors ``EEDI3.Backend.EEDI3``).

        vsfeel supports ``mclip`` natively, so both aux clips always pass
        through; only ``num_streams`` is injected, like every other entry
        point here. ``field`` is required positionally (matching the
        reference), so unlike the other entry points this cannot be called
        with keywords alone.
        """
        if self.supports_mclip:
            aux = {"sclip": sclip, "mclip": mclip}
        else:
            aux = {"sclip": sclip}
        return self._dispatch("EEDI3", (clip, field, *args), aux | kwargs)

    def EEDI3H(  # noqa: N802
        self,
        clip: vs.VideoNode,
        field: int,
        *args: Any,
        sclip: vs.VideoNode | None = None,
        mclip: vs.VideoNode | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """Run ``core.vsfeel.EEDI3H`` (mirrors ``EEDI3.Backend.EEDI3H``).

        Takes the native path when ``should_h`` holds, else the
        transpose → EEDI3 → transpose fallback — same shape as the
        reference, which only differs in that its CPU backend (no
        ``EEDI3H``) always takes the fallback.
        """
        if self.should_h:
            if self.supports_mclip:
                tclips = {"sclip": sclip, "mclip": mclip}
            else:
                tclips = {"sclip": sclip}
            return self._dispatch("EEDI3H", (clip, field, *args), tclips | kwargs)
        clip, tclips = self.transpose(clip, sclip=sclip, mclip=mclip, **kwargs)
        return self.EEDI3(clip, field, *args, **tclips).std.Transpose()

    @property
    def plugin(self) -> _FeelBM3DPlugin:
        """Plugin surface for wrappers that look the function up themselves,
        i.e. ``vsdenoise.bm3d`` through ``backend.plugin.BM3Dv2``."""
        if self._plugin is None:
            self._plugin = _FeelBM3DPlugin()
        return self._plugin

    def __init__(self) -> None:
        self._plugin: _FeelBM3DPlugin | None = None

    @contextmanager
    def __call__(
        self, *, bilateral: bool = True, gauss_blur: bool = True
    ) -> Generator[Self, None, None]:
        """Route the ``backend=`` singletons of vs-jetpack through vsfeel.

        Usage (no vs-jetpack changes needed)::

            with vsfeel.Backend():
                based = based_aa(clip, backend=vsfeel.Backend)

        With no arguments this swaps the default backend of every supported
        wrapper singleton (currently ``vsrgtools.bilateral`` and
        ``vsrgtools.gauss_blur`` — the only wrappers that keep a settable
        global default) to this object for the duration of the block, then
        restores the previous defaults. Every other wrapper (``nl_means``,
        ``bm3d``, ``DFTTest``, ``EEDI3``) takes its backend per call, so
        there is nothing to swap for them — pass ``backend=`` explicitly.
        An explicit ``backend=`` keyword always wins over the swapped
        default, and nesting composes (each level restores its own
        previous values).

        Implementation note: the singletons validate assignments through a
        ``Backend(value)`` enum constructor that rejects foreign objects,
        so this writes their ``_backend`` slots directly instead of going
        through the property setters.
        """
        targets: list[Any] = []
        if bilateral:
            try:
                from vsrgtools import bilateral as _bilateral

                targets.append(_bilateral)
            except ImportError:
                pass
        if gauss_blur:
            try:
                from vsrgtools import gauss_blur as _gauss_blur

                targets.append(_gauss_blur)
            except ImportError:
                pass
        saved = [fn._backend for fn in targets]
        for fn in targets:
            fn._backend = self
        try:
            yield self
        finally:
            for fn, old in zip(targets, saved):
                fn._backend = old


Backend = FeelBackend()
