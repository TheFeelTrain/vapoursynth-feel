"""vsaa integration: a fused EEDI3 antialiaser backed by ``vsfeel.EEDI3AA``.

``vsaa.based_aa`` drives its anti-aliasing through an ``EEDI3`` antialiaser,
whose ``antialias`` method runs the *whole* chain

    Merge( EEDI3H( Merge( EEDI3(clip) ) ) )

as two plugin calls plus two ``std.Merge`` nodes. ``vsfeel.EEDI3AA`` is that
exact chain fused into one plugin call (same kernels, same numerics — the
fused filter is bit-exact against the two-call chain for 16-bit integer input
and within a few ulp for float input), so this module subclasses the vsaa
antialiaser and overrides ``antialias`` to emit the single call.

Usage (no vs-jetpack changes needed)::

    from vsaa import based_aa
    import vsfeel

    aa = based_aa(clip, antialiaser=vsfeel.EEDI3(backend=vsfeel.Backend))

``vsfeel.EEDI3`` is a *subclass* of ``vsaa.deinterlacers.EEDI3``, so
``based_aa``'s ``isinstance`` checks, ``.sclip``/``.mclip``/``.backend``
attributes and every dataclass field come from the base class. ``import
vsfeel`` itself never imports vsaa; the subclass is built on first access to
``vsfeel.EEDI3`` (and this module can be imported directly as
``from vsfeel.vsaa import EEDI3``).

Anything the fused filter does not express — ``direction != BOTH``,
``double_rate=False``, ``transpose_first``, a ``Deinterlacer`` sclip, or an
input format EEDI3AA rejects — falls back to the base class's two-call chain,
so the subclass is a drop-in replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["EEDI3"]

from jetpytools import fallback
from vsaa.deinterlacers import Deinterlacer
from vsaa.deinterlacers import EEDI3 as _VsaaEEDI3
from vstools import VSFunctionNoArgs, core

_AADirection = _VsaaEEDI3.AADirection


def _fusable_format(clip: vs.VideoNode) -> bool:
    """Whether EEDI3AA accepts this clip (same surface as EEDI3 itself)."""
    fmt = clip.format
    import vapoursynth as vs

    if fmt is None:
        return False
    if fmt.sample_type == vs.INTEGER:
        return fmt.bits_per_sample == 16
    if fmt.sample_type == vs.FLOAT:
        return fmt.bits_per_sample == 32
    return False


class EEDI3(_VsaaEEDI3):
    """``vsaa`` EEDI3 antialiaser that fuses based_aa's chain into one call.

    Only ``antialias(direction=BOTH)`` in double-rate mode takes the fused
    path; every other call is delegated to the base implementation unchanged.
    """

    def antialias(  # type: ignore[override]
        self,
        clip: vs.VideoNode,
        direction: Any = _AADirection.BOTH,
        **kwargs: Any,
    ) -> vs.VideoNode:
        args = self.get_deint_args(**kwargs)
        sclip, mclip = args.pop("sclip", None), args.pop("mclip", None)

        if (
            direction == _AADirection.BOTH
            and self.double_rate
            and not self.transpose_first
            and not isinstance(sclip, Deinterlacer)
            and _fusable_format(clip)
        ):
            if sclip:
                if isinstance(sclip, VSFunctionNoArgs):
                    sclip = sclip(clip)
                # based_aa's double-rate sclip: one frame per OUTPUT frame.
                # EEDI3AA consumes both sub-frames internally, so this is the
                # same std.Interleave the base class builds.
                sclip = core.std.Interleave([sclip, sclip])
            if isinstance(mclip, VSFunctionNoArgs):
                mclip = mclip(clip)

            tff = fallback(args.pop("tff", self.tff), True)

            return core.vsfeel.EEDI3AA(
                clip,
                tff + 2,
                dh=False,
                sclip=sclip,
                mclip=mclip,
                **args,
            )

        return super().antialias(clip, direction=direction, **kwargs)
