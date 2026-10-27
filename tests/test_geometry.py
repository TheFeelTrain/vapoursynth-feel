"""Regression tests for cropped / non-tightly-pitched plane geometry.

A 640-pixel fixture cannot see the width-dependent host-copy bugs covered
here: DFTTest's aligned row-load helper (a 630-px float32 row starts at a
24-mod-32 byte offset for row 1), the fixed ``width*height*itemsize`` uploads
in Bilateral/GaussBlur/BM3D (the VapourSynth pitch is rounded up), and
NNEDI3's prescreen grid, which under-covered the frame tail whenever its pixel
grouping did not divide the width.

The tests crop a 640-wide Gray clip to 630/638 px (the yuv420 variants also
give half-width chroma rows) and compare the affected filters against the
matching reference (vszipcl, eedi3vk2, nnedi3vk).  They also cover a
top/bottom crop.  The half-float GaussBlur case adds the second 2-byte sample
type to the odd-pitch geometry.  EEDI3H and
EEDI3AA have no exact external reference, so their oracle is the same plugin's
composition (Transpose -> EEDI3 -> Transpose, and Merge(EEDI3H(Merge(EEDI3)))).

Every comparison runs in a subprocess with a timeout: the original DFTTest bug
was a SIGSEGV in the CPU copy helper, which must fail a test rather than kill
pytest.  References are materialised before vsfeel is touched; tolerances are
measurement-bounded and recorded next to each case.
"""

import json
import textwrap

import pytest

from conftest import COMPARE_PRELUDE, HEIGHT, CLIP_FRAMES, CLIP_PATH, compare_or_skip

WIDTHS = (630, 638)
FRAMES = [0, 1, 2]
CROP_FROM = 640

# Measured max diffs on the committed noise clip (frames 0..2):
#   dfttest    f32 630/638 gray:   3.73e-9 -> ulp-level, 1e-6
#   dfttest    f32 630/638 yuv420: 5.96e-8 -> ulp-level, 1e-6
#   dfttest    u16 630/638 (both): 0 codes -> bit-exact, <= 1 LSB bound
#   gaussblur  f32/u16 630/638:    0.0      -> bit-exact
#   gaussblur  f16 630/638 (gray and yuv420): 0.0 -> bit-exact
#   bilateral  f32 630/638 gray:   4.7e-9..5.6e-9 (with and without guide)
#   bilateral  f32 630/638 yuv420: 2.38e-7  -> 1e-6
#   bilateral  u16 630/638 (both): 1 code   -> <= 1 LSB
#   bilateral  f16 630/638 (gray, with guide, yuv420): one half step (4.88e-4)
#                                     -> 1e-3, the same f16 bound as
#                                     tests/test_bilateral.py
#   bm3d       f32 630/638 (radius=1): 6.44e-3 -> block-match decision flips
#                                     on rounding order; 0.02 bound
#   eedi3      u16: bit-exact vs eedi3vk2 (shared surface); f32 ~1 ulp
#   eedi3h     u16/f32: 0.0 vs the same-plugin transpose oracle
#   eedi3aa    u16: 0.0; f32: a few ulp of the fused chain (~6e-8 measured)
#   nlmeans    u16: <= 1 code; f32: <= 7.2e-7 (1e-5 bound here, the same
#                     f32-accumulation-order band as test_nlmeans.py)
#   nlmeans    f16 630/638: one half step (4.88e-4) -> 1e-3, the same bound
#                     tests/test_nlmeans.py uses for the half sweeps
# The yuv420 cases exercise half-width chroma: at 630 luma px the chroma rows
# are 315 samples, i.e. 630 bytes for u16 (22 mod 32) — the alignment that made
# the old DFTTest download helper fault.
TOL_F32_ULP = 1e-6
TOL_U16_LSB = 1.0
TOL_F16 = 1e-3
TOL_BM3D = 0.02
TOL_EEDI3_U16 = 0.0
TOL_EEDI3_F32 = 1e-6
TOL_ORACLE_EXACT = 0.0
TOL_EEDI3AA_F32 = 1e-6
TOL_NLMEANS_U16 = 1.0
TOL_NLMEANS_F32 = 1e-5
TOL_NLMEANS_F16 = TOL_F16

_GEOM_SCRIPT = COMPARE_PRELUDE + textwrap.dedent(f"""\
    import json
    import sys
    import vapoursynth as vs
    from vstools import core

    spec = json.loads(sys.argv[1])
    bits = spec["bits"]
    width = spec["width"]
    top = spec.get("top", 0)
    bottom = spec.get("bottom", 0)
    guide_on = spec.get("guide", 0)
    color = spec.get("color", "gray")
    half = spec.get("half", 0)
    params = spec.get("params", {{}})
    frames = spec.get("frames", [0, 1, 2])
    filter_ = spec["filter"]

    if filter_ == "nnedi3":
        # nnedi3vk only accepts u16 input; vsfeel is the same
        # GRAY16 clip, so the case is registered as u16.
        assert bits == 16 and color == "gray", (bits, color)

    core.max_cache_size = 512
    src = core.bs.VideoSource({CLIP_PATH!r})
    # Same length as the fixtures (the file is 300 frames): a temporal filter
    # sees a different window at the last frame otherwise.
    if src.num_frames > {CLIP_FRAMES}:
        src = core.std.Trim(src, first=0, length={CLIP_FRAMES})
    if color == "yuv420":
        # subsampled chroma: the chroma planes are half width, so their rows
        # are a different (and typically less aligned) length than luma's
        main = core.fmtc.bitdepth(src, bits=bits, fulls=True, fulld=True)
        if half:
            main = core.resize.Point(main, format=vs.YUV420PH)
    else:
        gray = core.std.ShufflePlanes(src, 0, vs.GRAY)
        main = core.fmtc.bitdepth(gray, bits=bits, fulls=True, fulld=True)
        if half:
            main = core.resize.Point(main, format=vs.GRAYH)
    left = spec.get("left", 0)
    main = core.std.Crop(main, left=left,
                         right=({CROP_FROM} - width - left),
                         top=top, bottom=bottom)
    dtype = np.float16 if half else (np.float32 if bits == 32 else np.uint16)
    planes = list(range(main.format.num_planes))

    # Guide/reference clip from a different chain (distinct frame allocations).
    guide = None
    if guide_on:
        guide = core.resize.Bicubic(main, width=main.width, height=main.height,
                                    format=main.format)

    NAMES = {{"dfttest": "DFTTest", "gaussblur": "GaussBlur",
              "bilateral": "Bilateral", "bm3d": "BM3D",
              "nnedi3": "NNEDI3", "nlmeans": "NLMeans",
              "eedi3": "EEDI3", "eedi3h": "EEDI3H", "eedi3aa": "EEDI3AA"}}
    # EEDI3H / EEDI3AA have no exact external reference on the shared surface;
    # their oracle is the same plugin's own composition, so a crash while
    # materialising it is a vsfeel failure, never a missing reference.
    VSFEEL_ORACLE = filter_ in ("eedi3h", "eedi3aa")

    def build_ref():
        if filter_ == "dfttest":
            return core.vszipcl.DFTTest(main, **params)
        if filter_ == "gaussblur":
            return core.vszipcl.GaussBlur(main, **params)
        if filter_ == "bilateral":
            return core.vszipcl.Bilateral(main, ref=guide,
                                          **params)
        if filter_ == "bm3d":
            return core.vszipcl.BM3Dv2(main, **params)
        if filter_ == "nnedi3":
            return core.nnedi3vk.NNEDI3(main, field=1, **params)
        if filter_ == "nlmeans":
            return core.vszipcl.NLMeans(main, **params)
        if filter_ == "eedi3":
            return core.eedi3vk2.EEDI3(main, **params)
        if filter_ == "eedi3h":
            t = core.std.Transpose(main)
            return core.std.Transpose(
                core.vsfeel.EEDI3(t, **params))
        if filter_ == "eedi3aa":
            v = core.vsfeel.EEDI3(main, **params)
            vm = core.std.Merge(v[::2], v[1::2])
            h = core.vsfeel.EEDI3H(vm, **params)
            return core.std.Merge(h[::2], h[1::2])
        raise SystemExit("bad filter %r" % filter_)

    def build_my():
        if filter_ == "bilateral":
            return core.vsfeel.Bilateral(main, ref=guide,
                                         **params)
        if filter_ == "nnedi3":
            return core.vsfeel.NNEDI3(main, field=1,
                                      **params)
        return getattr(core.vsfeel, NAMES[filter_])(main,
                                                    **params)

    # --- reference/oracle phase: materialise and copy before touching vsfeel ---
    if VSFEEL_ORACLE:
        print("REF ok", flush=True)
    try:
        ref_node = cpu_node(build_ref())
        ref_frames = [[read_plane(ref_node.get_frame(n), p, dtype)
                       for p in planes] for n in frames]
    except SystemExit:
        raise
    except Exception as exc:
        if VSFEEL_ORACLE:
            print("VSFEEL fail: oracle %s: %s" % (type(exc).__name__, exc),
                  flush=True)
            raise SystemExit(3)
        print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(2)
    if not VSFEEL_ORACLE:
        # A non-finite reference sample is a broken oracle, not a vsfeel
        # failure. (EEDI3H/EEDI3AA's oracle *is* vsfeel's own composition, so
        # for those the check below stays a vsfeel failure.)
        for n, ref_planes in zip(frames, ref_frames):
            for p, b in zip(planes, ref_planes):
                if not np.isfinite(b.astype(np.float64)).all():
                    print("REF unavailable: non-finite reference at frame %d plane %d"
                          % (n, p), flush=True)
                    raise SystemExit(2)
        print("REF ok", flush=True)

    # --- vsfeel phase ---
    try:
        # Pixels are read directly, so a GPU-resident node is downloaded first.
        my_node = cpu_node(build_my())
        worst = 0.0
        for n, ref_planes in zip(frames, ref_frames):
            frame = my_node.get_frame(n)
            for p, b in zip(planes, ref_planes):
                a = read_plane(frame, p, dtype)
                if not np.isfinite(a.astype(np.float64)).all():
                    print("VSFEEL fail: non-finite output at frame %d plane %d"
                          % (n, p), flush=True)
                    raise SystemExit(3)
                worst = max(worst, float(np.abs(a.astype(np.float64)
                                                - b.astype(np.float64)).max()))
    except SystemExit:
        raise
    except Exception as exc:
        print("VSFEEL fail: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(3)
    print("RESULT " + json.dumps({{"maxdiff": worst, "width": main.width,
                                  "height": main.height, "color": color,
                                  "planes": planes, "frames": list(frames),
                                  "num_frames": my_node.num_frames}}), flush=True)
""")

_BILATERAL_PARAMS = {"sigma_spatial": 3.0, "sigma_color": 0.05}
_DFTTEST_PARAMS = {"tbsize": 3}
_GAUSS_PARAMS = {"sigma": 2.0}
_BM3D_PARAMS = {"sigma": 0.7, "radius": 1, "bm_range": 16, "ps_range": 7, "block_step": 4}
# Default pscrn (2) means P=4 pixels/thread, the grouping that under-covered.
_NNEDI3_PARAMS = {"pscrn": 2}
# EEDI3 family: the config eedi3vk2 agrees with bit-exactly on u16.
_EEDI3_PARAMS = {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2}
_EEDI3H_PARAMS = {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2}
# EEDI3AA only accepts field 2/3; 3 is the two-call based_aa form.
_EEDI3AA_PARAMS = {"field": 3, "mdis": 5, "nrad": 1, "vcheck": 2}
# hp (the reference's half-pel search): the doubled direction set and its
# even/odd interpolation meet a non-multiple-of-32 plane width here.
_EEDI3_HP_PARAMS = {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2, "hp": 1}
_EEDI3AA_HP_PARAMS = {"field": 3, "mdis": 5, "nrad": 1, "vcheck": 2, "hp": 1}
# A measured-vs-vszipcl NLMeans config (see test_nlmeans.REFERENCE_CASES).
_NLMEANS_PARAMS = {"d": 1, "a": 3, "s": 3, "h": 3.0, "wref": 0.4}

# Vertical crop used by the top/bottom cases: 3 off each side keeps the height
# even (360 - 6 = 354) for the EEDI3 family's row-parity arithmetic while
# moving the row origin to an odd row.
_CROP_TOP = 3
_CROP_BOTTOM = 3

CASES = []


def _add(filter_, bits, width, guide, params, tol, color="gray", top=0, bottom=0, half=0, left=0):
    spec = {
        "filter": filter_,
        "bits": bits,
        "width": width,
        "guide": guide,
        "color": color,
        "half": half,
        "params": params,
        "frames": FRAMES,
        "top": top,
        "bottom": bottom,
        "left": left,
    }
    cid = "%s-%s-w%d" % (filter_, "f16" if half else ("f32" if bits == 32 else "u16"), width)
    if guide:
        cid += "-guide"
    if color == "yuv420":
        cid += "-yuv420"
    if top or bottom:
        cid += "-v%d_%d" % (top, bottom)
    if left:
        cid += "-l%d" % left
    CASES.append(pytest.param(spec, tol, id=cid))


for _w in WIDTHS:
    _add("dfttest", 32, _w, 0, _DFTTEST_PARAMS, TOL_F32_ULP)
    _add("dfttest", 16, _w, 0, _DFTTEST_PARAMS, TOL_U16_LSB)
    _add("gaussblur", 32, _w, 0, _GAUSS_PARAMS, 0.0)
    _add("gaussblur", 16, _w, 0, _GAUSS_PARAMS, 0.0)
    # The half path adds a second 2-byte element type: 630 half samples are
    # 1260 bytes per row, so a cropped half plane has its own pitch alignment.
    _add("gaussblur", 16, _w, 0, _GAUSS_PARAMS, 0.0, half=1)
    _add("bilateral", 32, _w, 0, _BILATERAL_PARAMS, TOL_F32_ULP)
    _add("bilateral", 32, _w, 1, _BILATERAL_PARAMS, TOL_F32_ULP)
    _add("bilateral", 16, _w, 0, _BILATERAL_PARAMS, TOL_U16_LSB)
    _add("bilateral", 16, _w, 1, _BILATERAL_PARAMS, TOL_U16_LSB)
    # The half path adds a second 2-byte element type to Bilateral's odd-pitch
    # geometry: 630 half samples are 1260 bytes per row, and the guide clip is
    # a second half plane through the same bindings.
    _add("bilateral", 16, _w, 0, _BILATERAL_PARAMS, TOL_F16, half=1)
    _add("bilateral", 16, _w, 1, _BILATERAL_PARAMS, TOL_F16, half=1)
    _add("bm3d", 32, _w, 0, _BM3D_PARAMS, TOL_BM3D)
    # R3: NNEDI3's prescreen dispatch grid must cover every row's tail
    # (P=4 does not divide 630/638).  GRAY16 only: nnedi3vk wants an integer
    # clip, and vsfeel==nnedi3vk is bit-exact here (measured 0 codes).
    _add("nnedi3", 16, _w, 0, _NNEDI3_PARAMS, TOL_U16_LSB)
    # The EEDI3 family and NLMeans on the same cropped-width geometry.
    # EEDI3H is horizontal (the transpose composition), EEDI3AA the fused
    # based_aa chain, NLMeans the tiled spatial filter.
    _add("eedi3", 16, _w, 0, _EEDI3_PARAMS, TOL_EEDI3_U16)
    _add("eedi3", 32, _w, 0, _EEDI3_PARAMS, TOL_EEDI3_F32)
    _add("eedi3h", 16, _w, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT)
    _add("eedi3h", 32, _w, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT)
    _add("eedi3aa", 16, _w, 0, _EEDI3AA_PARAMS, TOL_ORACLE_EXACT)
    _add("eedi3aa", 32, _w, 0, _EEDI3AA_PARAMS, TOL_EEDI3AA_F32)
    _add("eedi3", 16, _w, 0, _EEDI3_HP_PARAMS, TOL_EEDI3_U16)
    _add("eedi3aa", 16, _w, 0, _EEDI3AA_HP_PARAMS, TOL_ORACLE_EXACT)
    _add("nlmeans", 16, _w, 0, _NLMEANS_PARAMS, TOL_NLMEANS_U16)
    _add("nlmeans", 32, _w, 0, _NLMEANS_PARAMS, TOL_NLMEANS_F32)
    # The half path on the odd cropped pitch: 613 half samples per row (of the
    # 640-wide source) is 1226 bytes, its own alignment for the 2-byte sample.
    _add("nlmeans", 16, _w, 0, _NLMEANS_PARAMS, TOL_NLMEANS_F16, half=1)

# Subsampled chroma (R1/R2): the chroma planes are half the luma width, so
# their visible rows are 315 * itemsize bytes at 630 px — a different pitch and
# alignment from luma's. This is the case that crashed the old aligned stream
# load in the DFTTest download helper (630-byte u16 rows are 22 mod 32).
for _w in WIDTHS:
    _add("dfttest", 32, _w, 0, _DFTTEST_PARAMS, TOL_F32_ULP, "yuv420")
    _add("dfttest", 16, _w, 0, _DFTTEST_PARAMS, TOL_U16_LSB, "yuv420")
    _add("gaussblur", 32, _w, 0, _GAUSS_PARAMS, 0.0, "yuv420")
    _add("gaussblur", 16, _w, 0, _GAUSS_PARAMS, 0.0, "yuv420")
    _add("gaussblur", 16, _w, 0, _GAUSS_PARAMS, 0.0, "yuv420", half=1)
    _add("bilateral", 32, _w, 0, _BILATERAL_PARAMS, TOL_F32_ULP, "yuv420")
    _add("bilateral", 16, _w, 0, _BILATERAL_PARAMS, TOL_U16_LSB, "yuv420")
    # Half chroma: the 315-sample chroma rows are 630 bytes, the second 2-byte
    # plane width in this matrix.
    _add("bilateral", 16, _w, 0, _BILATERAL_PARAMS, TOL_F16, "yuv420", half=1)
    # BM3D denoises every plane, so the half-width chroma rows are part of the
    # graded surface now (they were passed through before).
    _add("bm3d", 32, _w, 0, _BM3D_PARAMS, TOL_BM3D, "yuv420")

# Vertical crop: a non-zero row origin and a shorter plane exercise the row
# staging / pad-origin path that the right-only crop leaves at row 0.
_add("dfttest", 32, 630, 0, _DFTTEST_PARAMS, TOL_F32_ULP, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("bilateral", 32, 630, 0, _BILATERAL_PARAMS, TOL_F32_ULP, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("gaussblur", 16, 630, 0, _GAUSS_PARAMS, 0.0, half=1, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("nnedi3", 16, 630, 0, _NNEDI3_PARAMS, TOL_U16_LSB, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("eedi3", 16, 630, 0, _EEDI3_PARAMS, TOL_EEDI3_U16, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("eedi3h", 16, 630, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("eedi3aa", 16, 630, 0, _EEDI3AA_PARAMS, TOL_ORACLE_EXACT, top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("nlmeans", 32, 630, 0, _NLMEANS_PARAMS, TOL_NLMEANS_F32, top=_CROP_TOP, bottom=_CROP_BOTTOM)

# Left crops move the plane origin as well as its pitch. The visible width is
# still the requested one, so the output geometry check stays meaningful.
_add("dfttest", 32, 613, 0, _DFTTEST_PARAMS, TOL_F32_ULP, left=27)
_add("dfttest", 16, 613, 0, _DFTTEST_PARAMS, TOL_U16_LSB, left=27)
_add("gaussblur", 32, 613, 0, _GAUSS_PARAMS, 0.0, left=27)
_add("gaussblur", 16, 613, 0, _GAUSS_PARAMS, 0.0, left=27)
_add("gaussblur", 16, 613, 0, _GAUSS_PARAMS, 0.0, left=27, half=1)
_add("bilateral", 32, 613, 0, _BILATERAL_PARAMS, TOL_F32_ULP, left=27)
_add("bilateral", 16, 613, 0, _BILATERAL_PARAMS, TOL_U16_LSB, left=27)
_add("bilateral", 16, 613, 0, _BILATERAL_PARAMS, TOL_F16, left=27, half=1)
_add("bm3d", 32, 613, 0, _BM3D_PARAMS, TOL_BM3D, left=27)
_add("nlmeans", 32, 613, 0, _NLMEANS_PARAMS, TOL_NLMEANS_F32, left=27)
_add("nlmeans", 16, 613, 0, _NLMEANS_PARAMS, TOL_NLMEANS_U16, left=27)
_add("nnedi3", 16, 613, 0, _NNEDI3_PARAMS, TOL_U16_LSB, left=27)
_add("eedi3", 16, 613, 0, _EEDI3_PARAMS, TOL_EEDI3_U16, left=27)
_add("eedi3", 32, 613, 0, _EEDI3_PARAMS, TOL_EEDI3_F32, left=27)
_add("eedi3h", 16, 614, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT, left=26)
_add("eedi3aa", 16, 614, 0, _EEDI3AA_PARAMS, TOL_ORACLE_EXACT, left=26)
_add("eedi3aa", 32, 614, 0, _EEDI3AA_PARAMS, TOL_EEDI3AA_F32, left=26)


@pytest.mark.parametrize("spec,tol", CASES)
def test_cropped_width_matches_reference(spec, tol):
    """Each affected filter must run and track its reference at cropped size.

    ``compare_or_skip`` skips when the reference is unavailable and raises an
    AssertionError (with the captured subprocess tail) when vsfeel crashed,
    timed out or produced non-finite output.
    """
    payload = compare_or_skip(_GEOM_SCRIPT, [json.dumps(spec)], timeout=300)
    assert payload["width"] == spec["width"], (
        f"unexpected output width {payload['width']} for {spec}"
    )
    assert payload["height"] == HEIGHT - spec.get("top", 0) - spec.get("bottom", 0), (
        f"unexpected output height {payload['height']} for {spec}"
    )
    assert payload["frames"] == spec["frames"]
    maxdiff = float(payload["maxdiff"])
    assert maxdiff <= tol, (
        f"{spec['filter']} bits={spec['bits']} width={spec['width']} "
        f"crop=({spec.get('top', 0)},{spec.get('bottom', 0)}) "
        f"guide={spec.get('guide', 0)}: max diff {maxdiff} > tol {tol}"
    )
