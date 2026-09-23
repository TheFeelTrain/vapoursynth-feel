"""vsaa integration: vsfeel-backed EEDI3 and NNEDI3 wrappers.

Usage:

    from vsaa import based_aa
    import vsfeel

    aa = based_aa(clip, supersampler=vsfeel.NNEDI3(), antialiaser=vsfeel.EEDI3())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["EEDI3", "NNEDI3"]

from jetpytools import fallback
from vsaa.deinterlacers import Deinterlacer
from vsaa.deinterlacers import EEDI3 as _VsaaEEDI3
from vsaa.deinterlacers import NNEDI3 as _VsaaNNEDI3
from vstools import VSFunctionAllArgs, VSFunctionNoArgs, core

from .backend import Backend as _FeelBackend

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


@dataclass
class EEDI3(_VsaaEEDI3):
    """``vsaa`` EEDI3 antialiaser that fuses based_aa's chain into one call.

    Only ``antialias(direction=BOTH)`` in double-rate mode takes the fused
    path; every other call is delegated to the base implementation unchanged.

    ``backend`` defaults to ``vsfeel.Backend``: constructing the vsfeel
    antialiaser and then asking for another plugin's backend is contradictory,
    so that is what an omitted argument means. Pass ``backend=`` explicitly to
    override it.
    """

    backend: Any = _FeelBackend

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


@dataclass
class NNEDI3(_VsaaNNEDI3):
    """``vsaa`` NNEDI3 supersampler backed by ``core.vsfeel.NNEDI3``.

    Same fields and methods as the base class, so it drops into any
    ``supersampler=``/``scaler=`` slot, but every interpolation runs on the
    vsfeel GPU plugin. ``gpu`` selects no backend here and is ignored.
    """

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return core.vsfeel.NNEDI3
