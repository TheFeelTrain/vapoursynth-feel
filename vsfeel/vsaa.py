"""vsaa integration: vsfeel-backed EEDI3 and NNEDI3 wrappers.

Usage:

    from vsaa import based_aa
    import vsfeel

    aa = based_aa(clip, supersampler=vsfeel.NNEDI3(), antialiaser=vsfeel.EEDI3())

``feel_aa`` is the same chain with every step on the device (see its docstring).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["EEDI3", "NNEDI3", "feel_aa"]

from jetpytools import CustomTypeError, CustomValueError, fallback
from vsaa.deinterlacers import EEDI3 as _VsaaEEDI3
from vsaa.deinterlacers import NNEDI3 as _VsaaNNEDI3
from vsaa.deinterlacers import AntiAliaser, Deinterlacer
from vsaa.funcs import BasedAA
from vskernels import (
    Bilinear,
    Box,
    Catrom,
    ComplexScalerLike,
    NoScale,
    Scaler,
    ScalerLike,
    is_noscale_like,
)
from vsmasktools import EdgeDetect, EdgeDetectLike, Morpho, Prewitt
from vsrgtools import bilateral
from vsscale import ArtCNN
from vstools import (
    UnsupportedColorFamilyError,
    VSFunctionAllArgs,
    VSFunctionNoArgs,
    check_ref_clip,
    core,
    get_y,
    join,
    scale_mask,
    vs,
)

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
        return fmt.bits_per_sample in (16, 32)
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


class _ResidentScaler(Catrom):
    """``Catrom`` whose own ``scale`` runs on the device.

    Only needed where a scaler *object* is driven directly: ``SuperSampler.scale``
    closes with ``ComplexScaler.ensure_obj(self.scaler).scale(...)``, and vstools'
    function proxy downloads a GPU clip before every function it wraps, so a
    supersampled frame came back to the host for that final subpixel shift even
    though the transposes and the NNEDI3 doublings before it were resident. The
    shift is the same ``src_top``/``src_left`` the kernel already maps; anything
    else (a non-unit SAR, linear light, ...) falls back to the base
    implementation.

    Scalers that ``feel_aa`` drives itself do not need this: they go through
    ``_scale_on_device``, which calls ``core.resize`` directly, where every
    built-in kernel including this one is resident.
    """

    def scale(  # type: ignore[override]
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        if kwargs.pop("_sar_scale", 1) != 1 or kwargs:
            return super().scale(clip, width, height, shift, **kwargs)

        w, h = self._wh_norm(clip, width, height)
        return _scale_on_device(self, clip, w, h, shift)


def _scale_on_device(
    scaler: Any, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0)
) -> vs.VideoNode:
    """``scaler.scale``, on the device wherever the scaler allows it.

    vstools' function proxy downloads a GPU clip before every function it wraps,
    so vskernels' own ``scale`` cannot be called on a resident clip even for
    Point. The scaler still supplies the kernel name and its parameters, which
    go straight to ``core.resize`` for the built-in kernels -- Catrom, Lanczos
    and the rest are all resident through that entry point -- and to
    ``core.vsfeel.Box`` for ``Box``, whose ``resize2.Custom`` is CPU-only.

    A scaler with neither (a learned model such as ``ArtCNN``, or a custom
    kernel whose plugin is CPU-only) falls back to its own ``scale`` and may
    hand back a host frame. That is allowed rather than refused, because it is
    the only way to use those scalers at all; the caller makes the result
    resident once with ``_ensure_resident``.
    """
    if isinstance(scaler, Box):
        if shift != (0, 0):
            raise CustomValueError(
                "Box on the device takes no subpixel shift; pass a core.resize "
                "kernel (Catrom, Lanczos, Spline36) if you need one",
                feel_aa,
                f"shift={shift}",
            )
        return core.vsfeel.Box(clip, width=width, height=height)

    name = getattr(getattr(scaler, "scale_function", None), "name", None)
    fn = getattr(core.resize, name, None) if isinstance(name, str) else None

    if fn is not None:
        args = dict(scaler.get_scale_args(clip, shift, width, height))
        args.pop("width", None)
        args.pop("height", None)
        return fn(clip, width, height, **args)

    # A supersampler (NNEDI3) or a custom kernel: its own `scale` is the only
    # way in, so take whatever it produces. A learned model comes back on the
    # host; the caller uploads it once.
    return scaler.scale(clip, width, height)


def _ensure_resident(node: vs.VideoNode) -> vs.VideoNode:
    """Upload a frame that came back on the host, once, for all of its readers.

    One host frame in the middle of the chain would otherwise be uploaded again
    for every node that reads it: the supersampled frame is read by the
    antialiaser's ``clip`` and ``sclip`` and again by the pscale blend, and the
    core inserts a separate upload for each. Uploading here makes it one.
    """
    if node.gpu_resident:
        return node
    return core.std.GPUUpload(clip=node)


@dataclass
class EEDI3(_VsaaEEDI3):
    """``vsaa`` EEDI3 antialiaser that fuses based_aa's chain into one GPU call.

    Only ``antialias(direction=BOTH)`` in double-rate mode takes the fused
    path; every other call behaves like the unmodified ``vsaa`` filter.
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
    """``vsaa`` NNEDI3 supersampler backed by ``core.vsfeel.NNEDI3``.

    Its output stays on the device, so it can be used in a GPU-resident chain.
    """

    scaler: ComplexScalerLike = _ResidentScaler

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return core.vsfeel.NNEDI3


def _edge_mask(detector: Any, luma: vs.VideoNode) -> vs.VideoNode:
    """The detector's edge mask, on the device where ``core.std`` has it.

    vsmasktools routes every integer clip through the ``edgemasks`` plugin,
    whose Prewitt/Sobel border handling matches no valid policy (up to 41228 off
    on the 1-pixel ring, quantised to 8-bit steps). ``core.std`` implements the
    same operator, matches the reference formula exactly at the border, and is
    GPU-capable, so it is preferred; other detectors fall through to
    vsmasktools, which may transfer.
    """
    name = getattr(detector, "__name__", None) or type(detector).__name__
    std_fn = getattr(core.std, name, None)

    if std_fn is not None:
        return std_fn(luma, None, 1.0)

    return EdgeDetect.ensure_obj(detector, feel_aa).edgemask(luma)


def _has_params(func: Callable[..., Any], n: int) -> bool:
    empty = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.default is p.empty and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]
    ]
    return len(empty) == n


def feel_aa(
    clip: vs.VideoNode,
    rfactor: float = 2.0,
    mask: vs.VideoNode | EdgeDetectLike | Literal[False] = Prewitt,
    mask_thr: int = 60,
    pscale: float = 0.0,
    downscaler: ScalerLike | None = None,
    supersampler: ScalerLike | Literal[False] = ArtCNN,
    antialiaser: AntiAliaser | type[AntiAliaser] | None = EEDI3,
    prefilter: vs.VideoNode | VSFunctionNoArgs | Literal[False] = False,
    postfilter: Callable[..., vs.VideoNode] | dict[str, Any] | Literal[False] = BasedAA.postfilter,
    show_mask: bool = False,
    **aa_kwargs: Any,
) -> vs.VideoNode:
    """``vsaa.based_aa`` with the chain kept on the GPU.

    The edge mask, antialiasing, downscale and postfilter all run on the device,
    so only the input upload and the output download cross the bus.

    Its defaults match ``based_aa``: ``ArtCNN`` supersamples, the downscale uses
    ``Box`` at an exact integer ``rfactor`` and ``Catrom`` otherwise, and other
    arguments are ``based_aa``'s. The antialiaser and postfilter default to
    vsfeel's own.

    A scaler is used on the device when it has a ``core.resize`` equivalent
    (``Point``, ``Bilinear``, ``Bicubic``/``Catrom``/``Mitchell``/``Hermite``,
    ``Lanczos``, ``Spline16/36/64``) or is ``Box``. One that does not, such as
    the default ``ArtCNN``, runs wherever it can and its result is uploaded
    once, so everything after it stays resident. The input must be 16-bit
    integer or 16/32-bit float, with even dimensions.
    """
    UnsupportedColorFamilyError.check(clip, (vs.YUV, vs.GRAY), feel_aa)

    if rfactor <= 0.0:
        raise CustomValueError("rfactor must be greater than 0!", feel_aa, rfactor)

    # The one upload: the supersampled clip and every intermediate below exist
    # only in VRAM.
    src = core.std.GPUUpload(clip=clip)
    luma = get_y(src)

    if mask is not False and not isinstance(mask, vs.VideoNode):
        mask = _edge_mask(mask, luma)
        mask = Morpho.binarize_mask(mask, scale_mask(mask_thr, 8, 32))
        # std.BoxBlur, not vsrgtools' box_blur (vszip.BoxBlur): identical on
        # float, 1 LSB apart on integers, and this one stays on the device.
        mask = core.std.BoxBlur(core.std.Maximum(mask), hradius=1, hpasses=1, vradius=1, vpasses=1)

        if show_mask:
            return mask
    elif isinstance(mask, vs.VideoNode):
        mask = get_y(core.std.GPUUpload(clip=mask))

    if supersampler is False or is_noscale_like(supersampler):
        supersampler = downscaler = NoScale[Catrom]()
        rfactor = pscale = 1.0

    aaw, aah = [round(dimension * rfactor) for dimension in (clip.width, clip.height)]

    if downscaler is None:
        # based_aa's own choice: Box for an exact integer ratio, Catrom
        # otherwise. Box is a pure average and stays on the device, where
        # resize2's CPU-only Custom would download the frame.
        downscaler = (
            Box
            if (
                max(aaw, clip.width) % min(aaw, clip.width) == 0
                and max(aah, clip.height) % min(aah, clip.height) == 0
            )
            else Catrom
        )

    supersampler = Scaler.ensure_obj(supersampler, feel_aa)
    downscaler = Scaler.ensure_obj(downscaler, feel_aa)

    if rfactor < 1.0:
        downscaler, supersampler = supersampler, downscaler

    if callable(prefilter):
        ss_clip = prefilter(luma)
    elif isinstance(prefilter, vs.VideoNode):
        check_ref_clip(prefilter, clip, feel_aa)
        ss_clip = get_y(core.std.GPUUpload(clip=prefilter))
    else:
        ss_clip = luma

    # A supersampler with no device path (ArtCNN) lands the 4K frame on the
    # host; this is the one upload that puts it back for every reader below.
    ss = _ensure_resident(_scale_on_device(supersampler, ss_clip, aaw, aah))

    # A class here is instantiated per call, so the default stays one object
    # shared by every invocation without ever being mutated by one of them.
    if antialiaser is None:
        antialiaser = EEDI3
    if isinstance(antialiaser, type):
        antialiaser = antialiaser()

    if (
        isinstance(antialiaser, _VsaaEEDI3)
        and antialiaser.sclip is None
        and "sclip" not in aa_kwargs
    ):
        aa_kwargs["sclip"] = ss

    if (
        aa_kwargs.pop("use_mclip", True)
        and "mclip" not in aa_kwargs
        and isinstance(antialiaser, _VsaaEEDI3)
        and antialiaser.backend.supports_mclip
        and mask is not False
    ):
        aa_kwargs.update(
            mclip=mask if rfactor == 1 else _scale_on_device(Bilinear(), mask, ss.width, ss.height)
        )

    aa = antialiaser.antialias(ss, **aa_kwargs)
    aa = _ensure_resident(_scale_on_device(downscaler, aa, clip.width, clip.height))

    # pscale_blend's expression, on std.Expr: its own path goes through the
    # CPU-only akarin plugin (vsexprtools), which downloads every operand.
    if pscale != 1.0:
        aa = core.std.Expr(
            [
                ss_clip,
                aa,
                _ensure_resident(_scale_on_device(downscaler, ss, clip.width, clip.height)),
            ],
            f"x z - {1.0 - pscale} * y +",
        )

    if postfilter is BasedAA.postfilter or isinstance(postfilter, dict):
        # based_aa's default postfilter -- a three-way median of the antialiased
        # clip, the source and a bilateral smoother -- with vsfeel's Bilateral
        # pinned and std.Expr in place of MeanMode's akarin call.
        extra = {} if postfilter is BasedAA.postfilter else dict(postfilter)
        extra.setdefault("sigmaS", 2.0)
        extra.setdefault("sigmaR", 1 / 255)
        extra.setdefault("backend", _FeelBackend)
        aa = core.std.Expr([aa, ss_clip, bilateral(aa, luma, **extra)], "x y min x y max z min max")
    elif callable(postfilter):
        if _has_params(postfilter, 3):
            aa = postfilter(aa, ss_clip, luma)
        elif _has_params(postfilter, 1):
            aa = postfilter(aa)
        else:
            raise CustomTypeError("Unsupported number of parameters", feel_aa, repr(postfilter))
    elif postfilter is not False:
        aa = BasedAA.postfilter(aa, ss_clip, luma, **postfilter)

    if mask is not False and mask is not None:
        aa = luma.std.MaskedMerge(aa, mask)

    if clip.format.color_family is vs.YUV:
        aa = join(aa, src)

    return aa
