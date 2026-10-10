"""Duck-typed backend selectors for the vsfeel VapourSynth plugin.

These plug into the unmodified ``backend=`` arguments of vs-jetpack functions;
see the package docstring in ``__init__.py`` for usage examples.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["Backend", "FeelBackend"]

# Serializes the context manager below: the vsrgtools singletons are
# process-global, so two threads swapping them at once would cross-restore
# (A restores CPU while B is still inside, B then restores the value it
# captured and leaks vsfeel past the block). RLock keeps same-thread
# nesting working.
_backend_lock = threading.RLock()


def _drop_unsupported(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``kwargs`` without the keys ``func`` does not declare.

    vs-jetpack forwards its own parameters (recent releases send ``hp`` to
    EEDI3) and VapourSynth rejects unknown keywords before dispatch, so the
    plugin's signature is the only reliable filter.
    """
    signature = getattr(func, "__signature__", None)
    if signature is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


class _FeelBM3DPlugin:
    """Stand-in for the ``core.vsfeel`` plugin surface.

    Exposes the entry point under vs-jetpack's spelling, ``BM3Dv2``, while the
    filter itself is ``core.vsfeel.BM3D`` (the plugin keeps ``BM3Dv2``
    registered as a legacy alias). The signature is extended by the parameters
    other BM3D plugins accept (``chroma``, ...). vsfeel takes ``chroma`` itself
    now (the reference's joint 4:4:4 entry), so the wrapper forwards it and only
    drops what the plugin does not declare.
    """

    def __init__(self, func: Any = None) -> None:
        # Injectable so the forwarding can be tested against a build whose
        # BM3D does not declare `chroma` (see tests/test_python_backend.py).
        self._func = func

    def _target(self) -> Any:
        if self._func is not None:
            return self._func

        import vapoursynth as vs

        # Resolved against the core that is live *now*, never captured: see
        # the `BM3Dv2` docstring.
        return vs.core.vsfeel.BM3D

    @property
    def BM3Dv2(self) -> Any:
        """vs-jetpack's name for ``core.vsfeel.BM3D``, built against the live core.

        The property keeps vs-jetpack's lookup (``backend.plugin.BM3Dv2``);
        the plugin itself registers both ``BM3D`` and the legacy ``BM3Dv2``.

        A fresh wrapper per access, because the ``Backend`` singleton outlives
        the VapourSynth environment: vsview's reload destroys the environment
        and creates a new core, and a ``Function`` captured from the old one
        calls ``Core.ensure_valid()`` on the destroyed core ("Use of invalidated
        Core"). ``vsdenoise`` reaches this through ``Backend.plugin`` on every
        call, so looking the function up here is what keeps it reload-safe
        (``vsdenoise``'s own ``Backend.plugin`` is a ``core.lazy`` proxy for the
        same reason).
        """
        func = self._target()
        native = func.__signature__
        # The *native* signature, not the advertised one below, decides what is
        # safe to forward: an older build that does not declare ``chroma``
        # rejects it as an unsupported argument.
        declares_chroma = "chroma" in native.parameters
        sig = native
        if not declares_chroma:
            # Advertising it keeps the wrapper's surface identical to the other
            # BM3D plugins.
            chroma = inspect.Parameter("chroma", inspect.Parameter.KEYWORD_ONLY, default=False)
            sig = native.replace(parameters=[*native.parameters.values(), chroma])

        def bm3d_v2(*args: Any, **kwargs: Any) -> vs.VideoNode:
            # vsdenoise passes chroma twice (once in its own kwargs, once
            # forced from the clip geometry), so pop it before the duplicate
            # becomes a TypeError.
            chroma_value = kwargs.pop("chroma", False)
            forwarded = _drop_unsupported(func, kwargs)
            if declares_chroma:
                forwarded["chroma"] = int(bool(chroma_value))
            return func(*args, **forwarded)

        bm3d_v2.__signature__ = sig  # type: ignore[attr-defined]
        return bm3d_v2


class FeelBackend:
    """Duck-typed backend accepted by vs-jetpack's ``backend=`` arguments.

    One instance serves every vs-jetpack wrapper: each wrapper only ever
    invokes the entry point matching its own filter, so the same object can be
    passed to ``vsrgtools.bilateral``, ``vsdenoise.bm3d``, ... alike.

    The entry-point methods are deliberately PascalCase: vs-jetpack looks them
    up by the filter's own name, and ``_dispatch`` forwards the same spelling
    to ``core.vsfeel``.
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

    def resolve(self) -> Self:
        """Resolve this backend to itself.

        vs-jetpack enums implement ``resolve()`` to map their AUTO member onto
        the function's default; an explicitly chosen backend always resolves to
        itself.
        """
        return self

    def _dispatch(self, func: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> vs.VideoNode:
        plugin_func = getattr(args[0].vsfeel, func)
        return plugin_func(*args[1:], **_drop_unsupported(plugin_func, kwargs))

    def Bilateral(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self._dispatch("Bilateral", (clip, *args), kwargs)

    def NLMeans(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self._dispatch("NLMeans", (clip, *args), kwargs)

    def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self._dispatch("DFTTest", (clip, *args), kwargs)

    def EEDI3(
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
        through. ``field`` is required positionally (matching the reference),
        so unlike the other entry points this cannot be called with keywords
        alone.
        """
        if self.supports_mclip:
            aux = {"sclip": sclip, "mclip": mclip}
        else:
            aux = {"sclip": sclip}
        return self._dispatch("EEDI3", (clip, field, *args), aux | kwargs)

    def EEDI3H(
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
        through the property setters. The swap is serialized on a
        module-level RLock, so threads entering the block nest instead of
        interleaving snapshots.
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
        with _backend_lock:
            saved = [fn._backend for fn in targets]
            for fn in targets:
                fn._backend = self
            try:
                yield self
            finally:
                for fn, old in zip(targets, saved):
                    fn._backend = old


Backend = FeelBackend()
