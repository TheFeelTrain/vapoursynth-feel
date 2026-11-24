"""``feel_dehalo``: ``vsdehalo.fine_dehalo`` with the chain kept on the device.

``fine_dehalo`` is a host chain: an ``edgemasks`` compass mask, ``akarin``
expressions, ``vszip.BoxBlur``, ``resize2``'s Gaussian and zimg resamplers.
``feel_dehalo`` rebuilds the same graph out of the core's GPU filters and
``core.vsfeel``'s kernels -- the input is uploaded once and every stage after
it stays in VRAM, so the only transfer left is the download its consumer asks
for. ``pre_ss`` supersamples with ``core.vsfeel.NNEDI3`` and point-downscales on
the device, exactly as ``vsscale.pre_ss`` does.

The stages, in the reference's order, are ``FineDehalo.Masks``
(``vsdehalo/mask.py``) and ``dehalo_alpha`` (``vsdehalo/alpha.py``). The
substitutions are:

- ``core.edgemasks.<detector>`` -> the detector's own matrices as
  ``std.Convolution`` nodes plus its merge expression (``Max``,
  ``EuclideanDistance``, ``SingleMatrix``); a detector that cannot be written
  that way runs on the host and its mask is uploaded once.
- ``akarin.Expr`` -> ``std.Expr``, except that the core's GPU expression has no
  relative pixel access, so ``Morpho.gradient`` is a ``std.Maximum`` and a
  ``std.Minimum`` followed by the subtraction.
- ``vszip.BoxBlur`` -> ``std.BoxBlur``, ``resize2.Gaussian`` ->
  ``core.vsfeel.GaussBlur``, ``vszip.Limiter`` (float masks only) ->
  ``std.Expr``.
- ``resize2`` Lanczos/Mitchell -> ``core.resize`` Lanczos/Bicubic, which is
  what ``vsfeel.vsaa`` already does for ``feel_aa``'s scalers.
- ``znedi3`` -> ``core.vsfeel.NNEDI3`` for the ``pre_ss`` doubling.

The result is close to, not identical with, ``fine_dehalo``. Away from the
frame border every stage is the same arithmetic on a different device: measured
against the reference on the 640x360 fixture and on 1080p, the interior agrees
to 2.6e-4 of the full range at worst (mean ~1e-6), which is 0.07 codes of 255.
Four things are left, in order of size:

- The frame's outermost pixels, where ``edgemasks`` extends the edge operator
  with a policy no ``std.Convolution`` reproduces. ``Morpho.expand`` carries
  that pixel inward by ``rx``, so the ring is a few pixels wide and it is the
  only place the full-frame difference exceeds the interior bound.
- ``resize2``'s Gaussian (10 taps, with an asymmetric tail at the support edge)
  against ``GaussBlur``'s 9 symmetric ones. The impulse responses agree to
  1.4e-4, which is the ~26-code residual the blur leaves in the gradients.
- The core's GPU Lanczos at the ``ss=1.5`` clamp: zimg's kernel, but it samples
  26 of 345 600 pixels differently at 1.5x.
- ``std.BoxBlur`` against ``vszip.BoxBlur``: identical on float, one code apart
  on integer masks at worst.

Residency is what makes the win possible rather than a bonus. A port that left
each host node in place would insert about fifteen upload/download pairs per 4K
frame, which costs more than the chain's own work: the reference graph is
entirely CPU (``edgemasks``, ``akarin``, ``vszip``, ``resize2``, ``znedi3`` and
zimg's resizers), so what the device buys is the compute itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import vapoursynth as vs

__all__ = ["feel_dehalo"]

from jetpytools import CustomIndexError, CustomTypeError, FuncExcept, mod_x, normalize_seq
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

# The reference's own default, named so the signature can state it.
_GAUSS_SIGMA = 1.4


# ---------------------------------------------------------------------------
# core.std plumbing
# ---------------------------------------------------------------------------


def _expr(
    clips: Sequence[vs.VideoNode],
    exprs: str | Sequence[str],
    planes: Planes = None,
    format: vs.VideoFormat | None = None,
) -> vs.VideoNode:
    """``std.Expr`` with one expression per processed plane.

    ``norm_expr`` cannot be used here: it goes through ``akarin.Expr``, which
    downloads every operand. ``std.Expr`` compiles the same bytecode on the
    device as long as the expression has no relative pixel access.
    """
    first = clips[0]
    selected = normalize_planes(first, planes)
    num = first.format.num_planes

    if isinstance(exprs, str):
        per_plane = [exprs if p in selected else "" for p in range(num)]
    else:
        per_plane = [exprs[p] if p in selected else "" for p in range(num)]

    return core.std.Expr(list(clips), per_plane, format)


def _token_values(clip: vs.VideoNode) -> tuple[float, float]:
    """``mask_max`` and ``range_size`` for the mask expressions.

    ``ExprToken.MaskMax`` is the full-range peak of the *first* clip and
    ``ExprToken.RangeSize`` is that peak plus one for integer formats.
    """
    peak = float(get_peak_value(clip, range_in=Range.FULL))

    return (peak, peak if clip.format.sample_type is vs.FLOAT else peak + 1.0)


def _gauss(clip: vs.VideoNode, sigma: Sequence[float], planes: Planes) -> vs.VideoNode:
    """``gauss_blur`` on the device: unprocessed planes get sigma 0 (copied)."""
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
    """``Morpho.expand``/``inpand``: the descending 3x3 dilation/erosion ladder.

    ``_xxpand_transform`` walks both radii down together and picks the 3x3
    subset from the larger one; ``ELLIPSE`` is a diamond on every iteration
    that is not the first of a three-step group.
    """
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
    """``vsrgtools.box_blur(radius=1, mode=HV)`` through the GPU BoxBlur."""
    return core.std.BoxBlur(
        clip, planes=planes, hradius=1, hpasses=passes, vradius=1, vpasses=passes
    )


def _gradient(clip: vs.VideoNode, planes: Planes) -> vs.VideoNode:
    """``Morpho.gradient(radius=1)``: max 3x3 minus min 3x3.

    The reference builds this as one akarin expression with relative pixel
    access (clamping each side to [0, 1] on float clips); the core's GPU
    expression has no relative access, so the two neighbourhoods are real
    filters. Integer clips match the reference away from the border ring,
    where ``std.Maximum`` replicates and the expression mirrors.
    """
    dilated = core.std.Maximum(clip, planes=planes)
    eroded = core.std.Minimum(clip, planes=planes)

    if clip.format.sample_type is vs.FLOAT:
        return _expr([dilated, eroded], "x 0 max 1 min y 0 max 1 min -", planes)

    return _expr([dilated, eroded], "x y -", planes)


def _limit_mask(mask: vs.VideoNode, planes: Planes) -> vs.VideoNode:
    """``vstools.limiter(mask, mask=True)``: a no-op for integer clips."""
    if mask.format.sample_type is not vs.FLOAT:
        return mask

    return _expr([mask], "x 0 max 1 min", planes)


def _repair_minmax(clip: vs.VideoNode, ref: vs.VideoNode, planes: Planes) -> vs.VideoNode:
    """``repair.Mode.MINMAX_SQUARE1(clip, ref)``: clamp to the ref's 3x3 range.

    zsmooth mirrors at the borders where ``std.Minimum``/``std.Maximum``
    replicate, so the 1-pixel ring can differ; everything inside it matches.
    """
    return _expr(
        [clip, core.std.Minimum(ref, planes=planes), core.std.Maximum(ref, planes=planes)],
        "x y max z min",
        planes,
    )


# ---------------------------------------------------------------------------
# Edge masks
# ---------------------------------------------------------------------------


def _convolution_mode(mode: str) -> str:
    return mode[:1] if mode[:1] in ("s", "h", "v") else "s"


def _matrix_edge_mask(detector: MatrixEdgeDetect, clip: vs.VideoNode) -> vs.VideoNode:
    """A matrix edge detector as ``std.Convolution`` nodes plus its merge.

    This is the graph ``vsmasktools`` builds when it is asked to ``use_expr``:
    one convolution per matrix with ``saturate=False`` (which is ``abs``),
    merged by ``Max``, ``EuclideanDistance`` or ``SingleMatrix``.
    """
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
    """``normalize_mask(edgemask, clip, clip)``, on the device where possible."""
    if isinstance(edgemask, vs.VideoNode):
        return ensure_resident(edgemask)

    if callable(edgemask) and not isinstance(edgemask, (EdgeDetect, type)):
        return ensure_resident(edgemask(clip, clip))

    detector = EdgeDetect.ensure_obj(edgemask, feel_dehalo)

    try:
        return _matrix_edge_mask(detector, clip)
    except (CustomTypeError, AttributeError):
        # A detector with a kernel of its own (ASobel, FreyChen, a plugin's):
        # run it on the host and pay one round trip for its mask.
        return ensure_resident(detector.edgemask(core.std.GPUDownload(clip=clip)))


# ---------------------------------------------------------------------------
# The mask pipeline
# ---------------------------------------------------------------------------


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
    """``vsdehalo.mask.FineDehalo.Masks`` with the same node graph."""
    selected = normalize_planes(clip, planes)
    work = get_y(clip) if selected == [0] else clip
    num = work.format.num_planes
    ry = ry if ry is not None else rx

    peak, _ = _token_values(work)

    thmif, thmaf, thlimif, thlimaf = (
        [scale_mask(x, 8, work) for x in normalize_seq(values, num)]
        for values in (thmi, thma, thlimi, thlima)
    )
    exclude_i = normalize_seq(exclude, num)
    edgeproc_f = normalize_seq(edgeproc, num)

    edges = _edge_mask(edgemask, work)

    strong = _expr(
        [edges],
        [f"x {thmif[p]} - {thmaf[p]} {thmif[p]} - / {peak} * 0 max {peak} min" for p in range(num)],
        selected,
    )

    large = _morpho(strong, rx, ry, XxpandMode.RECTANGLE, selected, expand=True)

    light = _expr(
        [edges],
        [
            f"x {thlimif[p]} - {thlimaf[p]} {thlimif[p]} - / {peak} * 0 max {peak} min"
            for p in range(num)
        ],
        selected,
    )

    shrink = _morpho(light, rx, ry, XxpandMode.ELLIPSE, selected, expand=True)
    shrink = core.std.BinarizeMask(shrink, threshold=scale_mask(0.25, 32, work), planes=selected)
    shrink = _morpho(shrink, rx, ry, XxpandMode.ELLIPSE, selected, expand=False)
    shrink = _box_blur(shrink, 2, selected)

    mask = _expr(
        [strong, shrink, large],
        [
            (
                f"{edgeproc_f[p]} 0 > z {int(exclude_i[p])} x y max x ? - 2 * dup x "
                f"{edgeproc_f[p]} 2 3 / * * + ? 0 max {peak} min"
            )
            for p in range(num)
        ],
        selected,
    )

    mask = _expr([_box_blur(mask, 1, selected)], f"x 2 * 0 max {peak} min", selected)

    return Masks(edges, strong, large, light, shrink, mask)


# ---------------------------------------------------------------------------
# dehalo_alpha
# ---------------------------------------------------------------------------


def _iters(*values: Any) -> list[tuple[list[Any], ...]]:
    """``vsdehalo.alpha._normalize_iter_arr_t``: per-iteration per-plane values."""
    longest = max(len(v) if isinstance(v, tuple) else 1 for v in values)

    broadcast = [
        v + (v[-1],) * (longest - len(v)) if isinstance(v, tuple) else (v,) * longest
        for v in values
    ]

    return list(zip(*([normalize_seq(item, 3) for item in group] for group in broadcast)))


def _blur(clip: vs.VideoNode, blur: Sequence[Any], planes: Planes) -> vs.VideoNode:
    """The iteration's blur: a Gaussian sigma, or a callable run as given."""
    head = blur[0]

    if callable(head):
        if getattr(head, "prefilter", None) is Prefilter.GAUSS:
            sigma = normalize_seq(getattr(head, "kwargs", {}).get("sigma", 1.5), 3)

            return _gauss(clip, sigma, planes)

        # A blur of the caller's own: it runs wherever it can, and its result
        # is made resident once so the rest of the chain stays put.
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
    """``vsdehalo.alpha.dehalo_alpha`` with every node on the device."""
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
            peak, range_size = _token_values(work)

            mask = _expr(
                [_gradient(work, planes), _gradient(dehalo, planes)],
                [
                    (
                        f"x 0 = 0 x y - x / ? {peak} * {scale_mask(lowsens_i[p], 8, work)} - "
                        f"x {range_size} + {range_size} 2 * / {highsens_i[p] / 100} + *"
                    )
                    for p in range(len(lowsens_i))
                ],
                planes,
            )

            dehalo = core.std.MaskedMerge(dehalo, work, _limit_mask(mask, planes), planes)

        elif all(x == -1 for x in lowsens_i) and all(x == -1 for x in highsens_i):
            pass
        else:
            raise CustomIndexError("lowsens and highsens must be between 0 and 100!", func)

        if ss_i == 1:
            dehalo = _repair_minmax(work, dehalo, planes)
        else:
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


# ---------------------------------------------------------------------------
# pre_ss
# ---------------------------------------------------------------------------


def _nnedi3(clip: vs.VideoNode, field: int) -> vs.VideoNode:
    """``vsaa.NNEDI3``'s doubling, with the wrapper's own predictor settings."""
    return core.vsfeel.NNEDI3(clip, field, dh=True, nsize=0, nns=4, qual=2, etype=0, pscrn=4)


def _supersample(clip: vs.VideoNode, field: int, rfactor: int) -> vs.VideoNode:
    """``SuperSamplerProcess[NNEDI3]`` at ``rfactor`` with ``noshift``.

    ``SuperSampler.scale`` doubles the height until it reaches the target, then
    the width the same way (``transpose_first`` is False: the width is the
    transposed height). With no sub-pixel shift that is ``log2(rfactor)``
    NNEDI3 doublings per axis around two transposes.

    The field of each doubling comes from the accumulator the reference keeps
    for the sub-pixel shift it discards: it starts at 0, so the first doubling
    takes the caller's field and every later one takes the opposite (the
    accumulator settles at +/-0.5).
    """
    steps = rfactor.bit_length() - 1
    fields = [field] + [1 - field] * (steps - 1)

    for doubling in fields:
        clip = _nnedi3(clip, doubling)

    clip = core.std.Transpose(clip)

    for doubling in fields:
        clip = _nnedi3(clip, doubling)

    return core.std.Transpose(clip)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _return_clip(processed: vs.VideoNode, src: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    """``FunctionUtil.return_clip``: merge the chroma that was not processed."""
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
    """Return ``clip`` to an 8-bit input's depth (the widened chain's inverse)."""
    if (
        clip.format.sample_type is fmt.sample_type
        and clip.format.bits_per_sample == fmt.bits_per_sample
    ):
        return clip

    return _expr([clip], f"x {1 << (16 - fmt.bits_per_sample)} /", None, fmt)


def _contrasharpen(
    flt: vs.VideoNode, src: vs.VideoNode, contra: float, planes: Planes
) -> vs.VideoNode:
    """``vsrgtools.contrasharpening_dehalo`` on the host (its median is zsmooth's)."""
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
    """``vsdehalo.fine_dehalo`` with the chain kept on the GPU.

    The parameters are ``fine_dehalo``'s, with one difference: ``blur`` is a
    Gaussian sigma (or per-plane, or per-iteration sigmas) rather than the
    ``Prefilter.GAUSS`` partial, which is what the default runs. A
    ``Prefilter.GAUSS`` partial is accepted too, and so is any other blur
    callable, which runs where it can and is uploaded once.

    The input is uploaded once and everything after it runs on the device, so
    the output stays GPU resident; ``pre_ss`` uses ``core.vsfeel.NNEDI3`` for
    the doubling. ``u16``, ``f16`` and ``f32`` are the native depths; an 8-bit
    input is widened to 16 bits on the device, which makes it process at a
    finer mask depth than the reference (``fine_dehalo`` keeps the source
    depth through the whole mask chain).

    The output is close to ``fine_dehalo``'s, not bit-exact: the module
    docstring lists the substitutions and the measured deviations.

    Args:
        clip: Source clip.
        blur: Gaussian sigma, per-plane list, or per-iteration tuple of those.
        lowsens: Lower sensitivity threshold; -1 on both disables the mask.
        highsens: Upper sensitivity threshold.
        ss: Supersampling factor for the aliasing clamp; 1 disables it.
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

    # The one upload. An 8-bit input is widened on the device because no
    # vsfeel kernel has an 8-bit io variant; the chain then runs at 16 bits.
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
        # vsscale.pre_ss: supersample, run the chain, point-downscale. The
        # field follows fine_dehalo's `tff` (any chroma plane selected).
        field = int(any(p in norm_planes for p in (1, 2)))
        processed = feel_dehalo(
            _supersample(work, field, pre_ss),
            blur,
            lowsens,
            highsens,
            ss,
            darkstr,
            brightstr,
            rx,
            ry,
            edgemask,
            thmi,
            thma,
            thlimi,
            thlima,
            exclude,
            edgeproc,
            contra,
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
        work, rx, ry, edgemask, thmi, thma, thlimi, thlima, exclude, edgeproc, planes
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
        # contrasharpening_dehalo is a host chain (zsmooth's median, akarin's
        # expression); running it there is exact and costs one round trip.
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
