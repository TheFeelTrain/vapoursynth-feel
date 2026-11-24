"""``feel_dehalo``: ``vsdehalo.fine_dehalo`` kept on the GPU.

Usage:

    from vsfeel import feel_dehalo

    dehaloed = feel_dehalo(src)

Same parameters as ``fine_dehalo``. The input is uploaded once and every stage after it
stays in VRAM; ``pre_ss`` supersamples with ``core.vsfeel.NNEDI3`` and point-downscales.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["feel_dehalo"]

from jetpytools import (
    CustomIndexError,
    CustomTypeError,
    FuncExcept,
    mod_x,
    normalize_seq,
)
from vsdenoise import Prefilter
from vskernels import Lanczos, Mitchell, Point, Scaler, ScalerLike
from vsmasktools import (
    Coordinates,
    EdgeDetect,
    EuclideanDistance,
    MaskLike,
    MatrixEdgeDetect,
    Max,
    Robinson3,
    SingleMatrix,
    XxpandMode,
)
from vstools import (
    Planes,
    Range,
    UnsupportedColorFamilyError,
    check_progressive,
    core,
    get_peak_value,
    get_y,
    join,
    normalize_planes,
    plane,
    scale_mask,
    vs,
)

from .vsaa import ensure_resident, scale_on_device

_MASK_NAMES = ("EDGES", "SHARP_EDGES", "LARGE_EDGES", "IGNORE_DETAILS", "SHRINK", "MAIN")

_RECTANGLE = tuple(Coordinates.RECTANGLE)
_DIAMOND = tuple(Coordinates.DIAMOND)

# fine_dehalo's own default.
_GAUSS_SIGMA = 1.4


def _expr(
    clips: Sequence[vs.VideoNode],
    exprs: str | Sequence[str],
    planes: Planes = None,
    format: vs.VideoFormat | None = None,
) -> vs.VideoNode:
    """One expression per processed plane, through ``std.Expr``.

    ``norm_expr`` would route through the CPU-only akarin plugin and download every operand.
    """
    first = clips[0]
    selected = normalize_planes(first, planes)
    num = first.format.num_planes

    if isinstance(exprs, str):
        exprs = [exprs] * num

    return core.std.Expr(
        list(clips), [exprs[p] if p in selected else "" for p in range(num)], format
    )


def _token_values(clip: vs.VideoNode) -> tuple[float, float]:
    """Full-range peak, and peak plus one for integer clips."""
    peak = float(get_peak_value(clip, range_in=Range.FULL))

    return (peak, peak if clip.format.sample_type is vs.FLOAT else peak + 1.0)


def _gauss(clip: vs.VideoNode, sigma: Sequence[float], planes: Planes) -> vs.VideoNode:
    """``gauss_blur`` on the device; unprocessed planes get sigma 0."""
    selected = normalize_planes(clip, planes)
    per_plane = [float(sigma[p]) if p in selected else 0.0 for p in range(clip.format.num_planes)]

    return core.vsfeel.GaussBlur(clip, sigma=per_plane)


def _morpho(
    clip: vs.VideoNode,
    w: int,
    h: int,
    mode: XxpandMode,
    planes: Planes,
    *,
    expand: bool,
) -> vs.VideoNode:
    """Morpho-style expand (or inpand) from 3x3 maximum (minimum) steps."""
    op = core.std.Maximum if expand else core.std.Minimum

    for wi, hi in zip_longest(range(w, -1, -1), range(h, -1, -1), fillvalue=0):
        if wi > 0 and hi > 0:
            coords = _DIAMOND if (mode is XxpandMode.ELLIPSE and wi % 3 != 1) else _RECTANGLE
        elif wi > 0:
            coords = tuple(Coordinates.HORIZONTAL)
        elif hi > 0:
            coords = tuple(Coordinates.VERTICAL)
        else:
            break

        clip = op(clip, coordinates=list(coords), planes=planes)

    return clip


def _box_blur(clip: vs.VideoNode, passes: int, planes: Planes) -> vs.VideoNode:
    """``box_blur(radius=1)`` through the GPU BoxBlur."""
    return core.std.BoxBlur(
        clip, planes=planes, hradius=1, hpasses=passes, vradius=1, vpasses=passes
    )


def _gradient(dilated: str, eroded: str, *, is_float: bool) -> str:
    """``Morpho.gradient(radius=1)`` as max minus min; float sides clamp first."""
    if is_float:
        return f"{dilated} 0 max 1 min {eroded} 0 max 1 min -"

    return f"{dilated} {eroded} -"


def _halo_mask(
    work: vs.VideoNode,
    dehalo: vs.VideoNode,
    lowsens: Sequence[float],
    highsens: Sequence[float],
    planes: Planes,
) -> vs.VideoNode:
    """How much gradient the blur removed, per plane."""
    peak, range_size = _token_values(work)
    is_float = work.format.sample_type is vs.FLOAT

    # std.Expr names its clips x/y/z, then a, b, ...: the fourth clip is "a".
    work_gradient = _gradient("x", "y", is_float=is_float)
    dehalo_gradient = _gradient("z", "a", is_float=is_float)

    exprs = [
        (
            f"{work_gradient} 0 = 0 {work_gradient} {dehalo_gradient} - {work_gradient} / ? "
            f"{peak} * {scale_mask(lowsens[p], 8, work)} - "
            f"{work_gradient} {range_size} + {range_size} 2 * / {highsens[p] / 100} + *"
            + (" 0 max 1 min" if is_float else "")
        )
        for p in range(len(lowsens))
    ]

    return _expr(
        [
            core.std.Maximum(work, planes=planes),
            core.std.Minimum(work, planes=planes),
            core.std.Maximum(dehalo, planes=planes),
            core.std.Minimum(dehalo, planes=planes),
        ],
        exprs,
        planes,
    )


def _repair_minmax(clip: vs.VideoNode, ref: vs.VideoNode, planes: Planes) -> vs.VideoNode:
    """``repair.Mode.MINMAX_SQUARE1`` as a clamp to the reference's 3x3 range."""
    return _expr(
        [clip, core.std.Minimum(ref, planes=planes), core.std.Maximum(ref, planes=planes)],
        "x y max z min",
        planes,
    )


def _threshold(
    edges: vs.VideoNode,
    lo: Sequence[float],
    hi: Sequence[float],
    peak: float,
    planes: Planes,
) -> vs.VideoNode:
    """Linear ramp of the edge mask between per-plane thresholds."""
    return _expr(
        [edges],
        [f"x {lo[p]} - {hi[p]} {lo[p]} - / {peak} * 0 max {peak} min" for p in range(len(lo))],
        planes,
    )


def _convolution_mode(mode: str) -> str:
    """The ``std.Convolution`` mode for an edge matrix."""
    return mode[:1] if mode[:1] in ("s", "h", "v") else "s"


def _matrix_edge_mask(detector: MatrixEdgeDetect, clip: vs.VideoNode) -> vs.VideoNode:
    """A matrix edge detector as ``std.Convolution`` nodes plus its merge."""
    matrices = list(detector._get_matrices())
    divisors = list(detector._get_divisors())
    modes = list(detector._get_mode_types())

    convs = [
        core.std.Convolution(
            clip,
            matrix=list(matrix),
            divisor=float(divisor),
            saturate=False,
            mode=_convolution_mode(mode),
        )
        for matrix, divisor, mode in zip(matrices, divisors, modes)
    ]

    if isinstance(detector, SingleMatrix):
        return convs[0]

    if isinstance(detector, EuclideanDistance):
        return _expr(convs[:2], "x dup * y dup * + sqrt 0 max", None)

    if isinstance(detector, Max):
        merged = convs[0]
        for conv in convs[1:]:
            merged = _expr([merged, conv], "x y max", None)

        return merged

    raise CustomTypeError(
        "edge detector has no convolution form", feel_dehalo, type(detector).__name__
    )


def _edge_mask(edgemask: MaskLike, clip: vs.VideoNode) -> vs.VideoNode:
    """The edge mask on the device; detectors with their own kernel run on the host."""
    if isinstance(edgemask, vs.VideoNode):
        return ensure_resident(edgemask)

    if callable(edgemask) and not isinstance(edgemask, (EdgeDetect, type)):
        return ensure_resident(edgemask(clip, clip))

    detector = EdgeDetect.ensure_obj(edgemask, feel_dehalo)

    try:
        return _matrix_edge_mask(detector, clip)
    except (CustomTypeError, AttributeError):
        return ensure_resident(detector.edgemask(core.std.GPUDownload(clip=clip)))


class Masks(Mapping[str, vs.VideoNode]):
    """The masks of a ``feel_dehalo`` call, mirroring ``FineDehalo.Masks``."""

    def __init__(
        self,
        edges: vs.VideoNode,
        strong: vs.VideoNode,
        large: vs.VideoNode,
        light: vs.VideoNode,
        shrink: vs.VideoNode,
        main: vs.VideoNode,
    ) -> None:
        self._edges = edges
        self._strong = strong
        self._large = large
        self._light = light
        self._shrink = shrink
        self._main = main

    def __getitem__(self, index: str) -> vs.VideoNode:
        key = index.upper()

        if key in _MASK_NAMES:
            return getattr(self, key)

        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from _MASK_NAMES

    def __len__(self) -> int:
        return len(_MASK_NAMES)

    @property
    def EDGES(self) -> vs.VideoNode:
        return self._edges

    @property
    def SHARP_EDGES(self) -> vs.VideoNode:
        return self._strong

    @property
    def LARGE_EDGES(self) -> vs.VideoNode:
        return self._large

    @property
    def IGNORE_DETAILS(self) -> vs.VideoNode:
        return self._light

    @property
    def SHRINK(self) -> vs.VideoNode:
        return self._shrink

    @property
    def MAIN(self) -> vs.VideoNode:
        return self._main


def _build_masks(
    clip: vs.VideoNode,
    rx: int,
    ry: int | None,
    edgemask: MaskLike,
    thmi: Any,
    thma: Any,
    thlimi: Any,
    thlima: Any,
    exclude: Any,
    edgeproc: Any,
    planes: Planes,
) -> Masks:
    """``FineDehalo.Masks`` as the same node graph."""
    selected = normalize_planes(clip, planes)
    work = get_y(clip) if selected == [0] else clip
    num = work.format.num_planes
    ry = rx if ry is None else ry

    peak, _ = _token_values(work)

    thmif = [scale_mask(x, 8, work) for x in normalize_seq(thmi, num)]
    thmaf = [scale_mask(x, 8, work) for x in normalize_seq(thma, num)]
    thlimif = [scale_mask(x, 8, work) for x in normalize_seq(thlimi, num)]
    thlimaf = [scale_mask(x, 8, work) for x in normalize_seq(thlima, num)]
    excludes = normalize_seq(exclude, num)
    edgeprocs = normalize_seq(edgeproc, num)

    # Sharpest edges, grown to cover the halos.
    edges = _edge_mask(edgemask, work)
    strong = _threshold(edges, thmif, thmaf, peak, selected)
    large = _morpho(strong, rx, ry, XxpandMode.RECTANGLE, selected, expand=True)

    # Close edges merged into exclusion zones, then smoothed.
    light = _threshold(edges, thlimif, thlimaf, peak, selected)
    shrink = _morpho(light, rx, ry, XxpandMode.ELLIPSE, selected, expand=True)
    shrink = core.std.BinarizeMask(shrink, threshold=scale_mask(0.25, 32, work), planes=selected)
    shrink = _morpho(shrink, rx, ry, XxpandMode.ELLIPSE, selected, expand=False)
    shrink = _box_blur(shrink, 2, selected)

    mask = _expr(
        [strong, shrink, large],
        [
            (
                f"{edgeprocs[p]} 0 > z {int(excludes[p])} x y max x ? - 2 * dup x "
                f"{edgeprocs[p]} 2 3 / * * + ? 0 max {peak} min"
            )
            for p in range(num)
        ],
        selected,
    )

    mask = _expr([_box_blur(mask, 1, selected)], f"x 2 * 0 max {peak} min", selected)

    return Masks(edges, strong, large, light, shrink, mask)


def _iters(*values: Any) -> list[tuple[list[Any], ...]]:
    """Per-iteration, per-plane values, broadcast like the reference."""
    longest = max(len(v) if isinstance(v, tuple) else 1 for v in values)

    broadcast = [
        v + (v[-1],) * (longest - len(v)) if isinstance(v, tuple) else (v,) * longest
        for v in values
    ]

    return list(zip(*[[normalize_seq(item, 3) for item in group] for group in broadcast]))


def _blur(clip: vs.VideoNode, blur: Sequence[Any], planes: Planes) -> vs.VideoNode:
    """The iteration's blur: a Gaussian sigma, or a callable run as given."""
    head = blur[0]

    if callable(head):
        if getattr(head, "prefilter", None) is Prefilter.GAUSS:
            sigma = normalize_seq(getattr(head, "kwargs", {}).get("sigma", 1.5), 3)

            return _gauss(clip, sigma, planes)

        return ensure_resident(head(clip, planes=planes))

    return _gauss(clip, [float(x) for x in blur], planes)


def _dehalo_alpha(
    clip: vs.VideoNode,
    blur: Any,
    lowsens: Any,
    highsens: Any,
    ss: Any,
    darkstr: Any,
    brightstr: Any,
    planes: Planes,
    supersampler: ScalerLike,
    supersampler_ref: ScalerLike,
    func: FuncExcept,
) -> vs.VideoNode:
    """``dehalo_alpha`` with every node on the device."""
    supersampler = Scaler.ensure_obj(supersampler, func)
    supersampler_ref = Scaler.ensure_obj(supersampler_ref, func)

    work = dehalo = clip

    for (
        blur_i,
        lowsens_i,
        highsens_i,
        (ss_i, *_),
        darkstr_i,
        brightstr_i,
        (sser, *_),
        (sser_ref, *_),
    ) in _iters(blur, lowsens, highsens, ss, darkstr, brightstr, supersampler, supersampler_ref):
        dehalo = _blur(work, blur_i, planes)

        if all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
            mask = _halo_mask(work, dehalo, lowsens_i, highsens_i, planes)

            dehalo = core.std.MaskedMerge(dehalo, work, mask, planes)

        elif all(x == -1 for x in lowsens_i) and all(x == -1 for x in highsens_i):
            pass
        else:
            raise CustomIndexError("lowsens and highsens must be between 0 and 100!", func)

        if ss_i == 1:
            dehalo = _repair_minmax(work, dehalo, planes)
        else:
            # Clamp through the supersampled envelope, then downscale.
            ss_w = mod_x(work.width * ss_i, 2**work.format.subsampling_w)
            ss_h = mod_x(work.height * ss_i, 2**work.format.subsampling_h)

            clip_ss = scale_on_device(sser, work, ss_w, ss_h)
            inpand = scale_on_device(sser_ref, core.std.Minimum(dehalo, planes=planes), ss_w, ss_h)
            expand = scale_on_device(sser_ref, core.std.Maximum(dehalo, planes=planes), ss_w, ss_h)
            dehalo = scale_on_device(
                sser,
                _expr([clip_ss, inpand, expand], "x y max z min", planes),
                work.width,
                work.height,
            )

        work = dehalo = _expr(
            [work, dehalo],
            [
                f"x x y - dup {brightstr_i[p]} * dup1 {darkstr_i[p]} * ? -"
                for p in range(len(brightstr_i))
            ],
            planes,
        )

    return dehalo


def _nnedi3(clip: vs.VideoNode, field: int) -> vs.VideoNode:
    """NNEDI3 doubling with the wrapper's predictor settings."""
    return core.vsfeel.NNEDI3(clip, field, dh=True, nsize=0, nns=4, qual=2, etype=0, pscrn=4)


def _supersample(clip: vs.VideoNode, field: int, rfactor: int) -> vs.VideoNode:
    """NNEDI3 supersampling by ``rfactor``; the caller point-downscales after.

    Doubles the height first, then the width across a transpose. The first
    doubling takes ``field`` and the rest take the opposite, as the reference's
    sub-pixel shift accumulator settles.
    """
    steps = rfactor.bit_length() - 1
    fields = [field] + [1 - field] * (steps - 1)

    for doubling in fields:
        clip = _nnedi3(clip, doubling)

    clip = core.std.Transpose(clip)

    for doubling in fields:
        clip = _nnedi3(clip, doubling)

    return core.std.Transpose(clip)


def _return_clip(processed: vs.VideoNode, src: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    """Merge the processed planes back onto the source clip."""
    if planes == [0] and src.format.num_planes > 1:
        return join([processed, plane(src, 1), plane(src, 2)], src.format.color_family)

    if planes == [1, 2] and src.format.num_planes > 1:
        return join(
            [plane(src, 0), plane(processed, 1), plane(processed, 2)], src.format.color_family
        )

    return processed


def _integer_format(fmt: vs.VideoFormat, bits: int) -> vs.VideoFormat:
    """The same geometry and colour family at another integer depth."""
    return core.query_video_format(
        fmt.color_family,
        vs.INTEGER,
        bits,
        fmt.subsampling_w if fmt.num_planes > 1 else 0,
        fmt.subsampling_h if fmt.num_planes > 1 else 0,
    )


def _narrow(clip: vs.VideoNode, fmt: vs.VideoFormat) -> vs.VideoNode:
    """Back to an 8-bit input's depth; the widened chain's inverse."""
    if (
        clip.format.sample_type is fmt.sample_type
        and clip.format.bits_per_sample == fmt.bits_per_sample
    ):
        return clip

    return _expr([clip], f"x {1 << (16 - fmt.bits_per_sample)} /", None, fmt)


def _contrasharpen(
    flt: vs.VideoNode, src: vs.VideoNode, contra: float, planes: Planes
) -> vs.VideoNode:
    """Host-only contra-sharpening; imported lazily."""
    from vsrgtools import contrasharpening_dehalo

    return contrasharpening_dehalo(flt, src, contra, planes=planes)


def feel_dehalo(
    clip: vs.VideoNode,
    blur: Any = _GAUSS_SIGMA,
    lowsens: Any = 50.0,
    highsens: Any = 50.0,
    ss: Any = 1.5,
    darkstr: Any = 0.0,
    brightstr: Any = 1.0,
    rx: int = 2,
    ry: int | None = None,
    edgemask: MaskLike = Robinson3,
    thmi: Any = 80,
    thma: Any = 128,
    thlimi: Any = 50,
    thlima: Any = 100,
    exclude: Any = True,
    edgeproc: Any = 0.0,
    contra: float = 0.0,
    pre_ss: int = 1,
    planes: Planes = 0,
    attach_masks: bool = False,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """Halo removal with the ``fine_dehalo`` chain kept on the GPU.

    ``ss`` can be set per iteration, while ``blur``, ``lowsens``, ``highsens``,
    ``darkstr`` and ``brightstr`` can be set per plane and per iteration: a
    single value applies everywhere, a tuple is per-iteration, and a list
    inside the tuple is per-plane for that iteration. ``thmi``, ``thma``,
    ``thlimi``, ``thlima``, ``exclude`` and ``edgeproc`` can be set per plane.

    The output is close to ``fine_dehalo``'s, not bit-exact.

    Args:
        clip: Source clip.
        blur: Gaussian sigma, per-plane list, or per-iteration tuple of those. A
            ``Prefilter.GAUSS`` partial runs as its sigma, any other callable runs
            where it can and is uploaded once.
        lowsens: Lower sensitivity threshold; dehalo is fully applied below it.
            Setting both `lowsens` and `highsens` to `-1` disables the mask.
        highsens: Upper sensitivity threshold; dehalo is skipped above it.
            Setting both `lowsens` and `highsens` to `-1` disables the mask.
        ss: Supersampling factor for the aliasing clamp; 1 takes the other path.
        darkstr: Strength factor for suppressing dark halos.
        brightstr: Strength factor for suppressing bright halos.
        rx: Horizontal radius for halo removal.
        ry: Vertical radius. Defaults to ``rx``.
        edgemask: Edge detection object, class, name or mask clip.
        thmi: Minimum threshold for sharp edge selection.
        thma: Maximum threshold for sharp edge selection.
        thlimi: Minimum threshold for including previously ignored edges.
        thlima: Maximum threshold for including less distinct edges.
        exclude: Whether to exclude edges that are too close together.
        edgeproc: If greater than 0, adds the edge mask into the processing.
        contra: Contra-sharpening level; runs on the host (0 disables it).
        pre_ss: Power-of-two supersampling factor applied before processing.
        planes: Planes to process.
        attach_masks: Store the masks as frame properties on the output.
        func: Function used for error messages.
        **kwargs: ``supersampler``/``supersampler_ref`` for the ``ss`` clamp.

    Returns:
        Dehaloed clip, GPU resident.
    """
    func = func or feel_dehalo

    UnsupportedColorFamilyError.check(clip, (vs.YUV, vs.GRAY), func)

    assert check_progressive(clip, func)

    if pre_ss < 1 or pre_ss & (pre_ss - 1):
        raise CustomIndexError("`pre_ss` has to be a power of 2.", func, pre_ss)

    supersampler = kwargs.pop("supersampler", Lanczos())
    supersampler_ref = kwargs.pop("supersampler_ref", Mitchell())

    if kwargs:
        raise CustomTypeError("Unsupported keyword argument(s)", func, ", ".join(sorted(kwargs)))

    norm_planes = normalize_planes(clip, planes)
    out_format = clip.format

    # The one upload. No 8-bit io variant exists, so 8-bit is widened on the device.
    src = core.std.GPUUpload(clip=clip)
    if clip.format.sample_type is vs.INTEGER and clip.format.bits_per_sample < 16:
        src = _expr(
            [src],
            f"x {1 << (16 - clip.format.bits_per_sample)} *",
            None,
            _integer_format(out_format, 16),
        )

    work = get_y(src) if norm_planes == [0] else src

    if pre_ss > 1:
        # Supersample, run the chain, point-downscale. Any selected chroma plane means tff.
        field = int(any(p in norm_planes for p in (1, 2)))
        processed = feel_dehalo(
            _supersample(work, field, pre_ss),
            blur=blur,
            lowsens=lowsens,
            highsens=highsens,
            ss=ss,
            darkstr=darkstr,
            brightstr=brightstr,
            rx=rx,
            ry=ry,
            edgemask=edgemask,
            thmi=thmi,
            thma=thma,
            thlimi=thlimi,
            thlima=thlima,
            exclude=exclude,
            edgeproc=edgeproc,
            contra=contra,
            pre_ss=1,
            planes=planes,
            attach_masks=attach_masks,
            func=func,
            supersampler=supersampler,
            supersampler_ref=supersampler_ref,
        )
        down = core.std.CopyFrameProps(
            scale_on_device(Point(), processed, work.width, work.height), processed
        )

        return _narrow(_return_clip(down, src, norm_planes), out_format)

    masks = _build_masks(
        work,
        rx=rx,
        ry=ry,
        edgemask=edgemask,
        thmi=thmi,
        thma=thma,
        thlimi=thlimi,
        thlima=thlima,
        exclude=exclude,
        edgeproc=edgeproc,
        planes=planes,
    )
    feel_dehalo.masks = masks  # type: ignore[attr-defined]

    dehaloed = _dehalo_alpha(
        work,
        blur,
        lowsens,
        highsens,
        ss,
        darkstr,
        brightstr,
        planes,
        supersampler,
        supersampler_ref,
        func,
    )

    if contra:
        # Host-only (zsmooth's median); one round trip each way.
        dehaloed = ensure_resident(
            _contrasharpen(
                core.std.GPUDownload(clip=dehaloed), core.std.GPUDownload(clip=work), contra, planes
            )
        )

    out = _narrow(
        _return_clip(core.std.MaskedMerge(work, dehaloed, masks.MAIN, planes), src, norm_planes),
        out_format,
    )

    if attach_masks:
        out = out.std.CopyFrameProps(dehaloed)

        for key in masks:
            out = out.std.ClipToProp(
                masks[key], "FineDehaloMask" + "".join(part.title() for part in key.split("_"))
            )

    return out


feel_dehalo.masks = None  # type: ignore[attr-defined]
