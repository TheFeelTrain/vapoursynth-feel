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
from vsaa.deinterlacers import EEDI3 as _VsaaEEDI3
from vsaa.deinterlacers import NNEDI3 as _VsaaNNEDI3
from vsaa.deinterlacers import Deinterlacer
from vstools import VSFunctionAllArgs, VSFunctionNoArgs, core

from .backend import Backend as _FeelBackend
from .backend import _drop_unsupported

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


def _fusable_geometry(clip: vs.VideoNode, planes: Any = None) -> bool:
    """Whether every plane EEDI3AA would process has even dimensions.

    ``EEDI3AA`` runs the horizontal geometry internally, so unlike ``EEDI3`` it
    truncates an odd plane width (and leaves the last column unwritten); odd
    geometry therefore keeps the chain, which refuses the same clip. ``planes``
    is the plugin's own selection, so ``None`` means every plane.
    """
    fmt = clip.format
    if fmt is None:
        return False
    if planes is None:
        selected: Any = range(fmt.num_planes)
    elif isinstance(planes, int):
        selected = [planes]
    else:
        selected = planes
    for p in selected:
        if not 0 <= p < fmt.num_planes:
            return True  # out of range: let the plugin report it
        ss_w = fmt.subsampling_w if p > 0 and fmt.num_planes > 1 else 0
        ss_h = fmt.subsampling_h if p > 0 and fmt.num_planes > 1 else 0
        if ((clip.width >> ss_w) & 1) or ((clip.height >> ss_h) & 1):
            return False
    return True


def _feel_backend(backend: Any) -> bool:
    """Whether ``backend`` routes EEDI3 to the vsfeel plugin.

    The fused ``EEDI3AA`` call is only the vsfeel backend's chain: a CPU or
    reference backend must keep the base class's two-pass implementation.
    vsfeel's own backend objects carry the ``vsfeel`` namespace as their
    ``value``.
    """
    return getattr(backend, "value", None) == _FeelBackend.value


@dataclass
class EEDI3(_VsaaEEDI3):
    """``vsaa`` EEDI3 antialiaser that fuses based_aa's chain into one call.

    Only ``antialias(direction=BOTH)`` in double-rate mode takes the fused
    path; every other call is delegated to the base implementation unchanged.
    """

    alpha: float = 0.125
    gamma: float = 40.0
    vthresh: tuple[float | None, float | None, float | None] | None = (12.0, 24.0, 4.0)
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
            and _feel_backend(self.backend)
            and _fusable_format(clip)
            and _fusable_geometry(clip, args.get("planes"))
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

            func = core.vsfeel.EEDI3AA
            return func(
                clip,
                tff + 2,
                dh=False,
                sclip=sclip,
                mclip=mclip,
                **_drop_unsupported(func, args),
            )

        return super().antialias(clip, direction=direction, **kwargs)


@dataclass
class NNEDI3(_VsaaNNEDI3):
    """``vsaa`` NNEDI3 supersampler backed by ``core.vsfeel.NNEDI3``."""

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return core.vsfeel.NNEDI3
