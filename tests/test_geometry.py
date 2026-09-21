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
``num_streams=4`` variant and a top/bottom crop; the reference always stays
serial, so those cases double as multi-stream-vs-serial checks.  EEDI3H and
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

from conftest import COMPARE_PRELUDE, HEIGHT, NOISE_MKV, compare_or_skip

WIDTHS = (630, 638)
FRAMES = [0, 1, 2]
CROP_FROM = 640

# Measured max diffs on the committed noise clip (frames 0..2):
#   dfttest    f32 630/638 gray:   3.73e-9 -> ulp-level, 1e-6
#   dfttest    f32 630/638 yuv420: 5.96e-8 -> ulp-level, 1e-6
#   dfttest    u16 630/638 (both): 0 codes -> bit-exact, <= 1 LSB bound
#   gaussblur  f32/u16 630/638:    0.0      -> bit-exact
#   bilateral  f32 630/638 gray:   4.7e-9..5.6e-9 (with and without guide)
#   bilateral  f32 630/638 yuv420: 2.38e-7  -> 1e-6
#   bilateral  u16 630/638 (both): 1 code   -> <= 1 LSB
#   bm3d       f32 630/638 (radius=1): 6.44e-3 -> block-match decision flips
#                                     on rounding order; 0.02 bound
#   eedi3      u16: bit-exact vs eedi3vk2 (shared surface); f32 ~1 ulp
#   eedi3h     u16/f32: 0.0 vs the same-plugin transpose oracle
#   eedi3aa    u16: 0.0; f32: a few ulp of the fused chain (~6e-8 measured)
#   nlmeans    u16: <= 1 code; f32: <= 6e-5 (sweep band documented in
#                     test_nlmeans.py; 1e-4 bound here)
# The yuv420 cases exercise half-width chroma: at 630 luma px the chroma rows
# are 315 samples, i.e. 630 bytes for u16 (22 mod 32) — the alignment that made
# the old DFTTest download helper fault.
TOL_F32_ULP = 1e-6
TOL_U16_LSB = 1.0
TOL_BM3D = 0.02
TOL_EEDI3_16 = 0.0
TOL_EEDI3_32 = 1e-6
TOL_ORACLE_EXACT = 0.0
TOL_EEDI3AA_32 = 1e-6
TOL_NLMEANS_16 = 1.0
TOL_NLMEANS_32 = 1e-4

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
    params = spec.get("params", {{}})
    frames = spec.get("frames", [0, 1, 2])
    streams = spec.get("streams", 1)
    filter_ = spec["filter"]

    if filter_ == "nnedi3":
        # nnedi3vk only accepts 16-bit integer input; vsfeel is the same
        # GRAY16 clip, so the case is registered as 16-bit.
        assert bits == 16 and color == "gray", (bits, color)

    core.max_cache_size = 512
    src = core.bs.VideoSource({NOISE_MKV!r})
    if color == "yuv420":
        # subsampled chroma: the chroma planes are half width, so their rows
        # are a different (and typically less aligned) length than luma's
        main = core.fmtc.bitdepth(src, bits=bits, fulls=True, fulld=True)
    else:
        gray = core.std.ShufflePlanes(src, 0, vs.GRAY)
        main = core.fmtc.bitdepth(gray, bits=bits, fulls=True, fulld=True)
    main = core.std.Crop(main, right=({CROP_FROM} - width),
                         top=top, bottom=bottom)
    dtype = np.float32 if bits == 32 else np.uint16
    planes = list(range(main.format.num_planes))

    # Guide/reference clip from a different chain (distinct frame allocations).
    guide = None
    if guide_on:
        guide = core.resize.Bicubic(main, width=main.width, height=main.height,
                                    format=main.format)

    NAMES = {{"dfttest": "DFTTest", "gaussblur": "GaussBlur",
              "bilateral": "Bilateral", "bm3d": "BM3Dv2",
              "nnedi3": "NNEDI3", "nlmeans": "NLMeans",
              "eedi3": "EEDI3", "eedi3h": "EEDI3H", "eedi3aa": "EEDI3AA"}}
    # EEDI3H / EEDI3AA have no exact external reference on the shared surface;
    # their oracle is the same plugin's own composition, so a crash while
    # materialising it is a vsfeel failure, never a missing reference.
    VSFEEL_ORACLE = filter_ in ("eedi3h", "eedi3aa")

    def build_ref():
        if filter_ == "dfttest":
            return core.vszipcl.DFTTest(main, num_streams=1, **params)
        if filter_ == "gaussblur":
            return core.vszipcl.GaussBlur(main, num_streams=1, **params)
        if filter_ == "bilateral":
            return core.vszipcl.Bilateral(main, ref=guide, num_streams=1,
                                          **params)
        if filter_ == "bm3d":
            return core.vszipcl.BM3Dv2(main, num_streams=1, **params)
        if filter_ == "nnedi3":
            return core.nnedi3vk.NNEDI3(main, field=1, num_streams=1, **params)
        if filter_ == "nlmeans":
            return core.vszipcl.NLMeans(main, num_streams=1, **params)
        if filter_ == "eedi3":
            return core.eedi3vk2.EEDI3(main, num_streams=1, **params)
        if filter_ == "eedi3h":
            t = core.std.Transpose(main)
            return core.std.Transpose(
                core.vsfeel.EEDI3(t, num_streams=1, **params))
        if filter_ == "eedi3aa":
            v = core.vsfeel.EEDI3(main, num_streams=1, **params)
            vm = core.std.Merge(v[::2], v[1::2])
            h = core.vsfeel.EEDI3H(vm, num_streams=1, **params)
            return core.std.Merge(h[::2], h[1::2])
        raise SystemExit("bad filter %r" % filter_)

    def build_my():
        if filter_ == "bilateral":
            return core.vsfeel.Bilateral(main, ref=guide, num_streams=streams,
                                         **params)
        if filter_ == "nnedi3":
            return core.vsfeel.NNEDI3(main, field=1, num_streams=streams,
                                      **params)
        return getattr(core.vsfeel, NAMES[filter_])(main, num_streams=streams,
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
                if not (np.isfinite(a.astype(np.float64)).all()
                        and np.isfinite(b.astype(np.float64)).all()):
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
                                  "streams": streams,
                                  "num_frames": my_node.num_frames}}), flush=True)
""")

_BILATERAL_PARAMS = {"sigma_spatial": 3.0, "sigma_color": 0.05}
_DFTTEST_PARAMS = {"tbsize": 3}
_GAUSS_PARAMS = {"sigma": 2.0}
_BM3D_PARAMS = {"sigma": 0.7, "radius": 1, "bm_range": 16, "ps_range": 7,
                "block_step": 4}
# Default pscrn (2) means P=4 pixels/thread, the grouping that under-covered.
_NNEDI3_PARAMS = {"pscrn": 2}
# EEDI3 family: the config eedi3vk2 agrees with bit-exactly on u16.
_EEDI3_PARAMS = {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2}
_EEDI3H_PARAMS = {"field": 1, "mdis": 5, "nrad": 1, "vcheck": 2}
# EEDI3AA only accepts field 2/3; 3 is the two-call based_aa form.
_EEDI3AA_PARAMS = {"field": 3, "mdis": 5, "nrad": 1, "vcheck": 2}
# A measured-vs-vszipcl NLMeans config (see test_nlmeans.REFERENCE_CASES).
_NLMEANS_PARAMS = {"d": 1, "a": 3, "s": 3, "h": 3.0, "wref": 0.4}

# Vertical crop used by the top/bottom cases: 3 off each side keeps the height
# even (360 - 6 = 354) for the EEDI3 family's row-parity arithmetic while
# moving the row origin to an odd row.
_CROP_TOP = 3
_CROP_BOTTOM = 3

CASES = []


def _add(filter_, bits, width, guide, params, tol, color="gray", streams=1,
         top=0, bottom=0):
    spec = {"filter": filter_, "bits": bits, "width": width, "guide": guide,
            "color": color, "params": params, "frames": FRAMES,
            "streams": streams, "top": top, "bottom": bottom}
    cid = "%s-%s-w%d" % (filter_, "f32" if bits == 32 else "u16", width)
    if guide:
        cid += "-guide"
    if color == "yuv420":
        cid += "-yuv420"
    if streams != 1:
        cid += "-s%d" % streams
    if top or bottom:
        cid += "-v%d_%d" % (top, bottom)
    CASES.append(pytest.param(spec, tol, id=cid))


for _w in WIDTHS:
    _add("dfttest", 32, _w, 0, _DFTTEST_PARAMS, TOL_F32_ULP)
    _add("dfttest", 16, _w, 0, _DFTTEST_PARAMS, TOL_U16_LSB)
    _add("gaussblur", 32, _w, 0, _GAUSS_PARAMS, 0.0)
    _add("gaussblur", 16, _w, 0, _GAUSS_PARAMS, 0.0)
    _add("bilateral", 32, _w, 0, _BILATERAL_PARAMS, TOL_F32_ULP)
    _add("bilateral", 32, _w, 1, _BILATERAL_PARAMS, TOL_F32_ULP)
    _add("bilateral", 16, _w, 0, _BILATERAL_PARAMS, TOL_U16_LSB)
    _add("bilateral", 16, _w, 1, _BILATERAL_PARAMS, TOL_U16_LSB)
    _add("bm3d", 32, _w, 0, _BM3D_PARAMS, TOL_BM3D)
    # R3: NNEDI3's prescreen dispatch grid must cover every row's tail
    # (P=4 does not divide 630/638).  GRAY16 only: nnedi3vk wants an integer
    # clip, and vsfeel==nnedi3vk is bit-exact here (measured 0 codes).
    _add("nnedi3", 16, _w, 0, _NNEDI3_PARAMS, TOL_U16_LSB)
    # The EEDI3 family and NLMeans on the same cropped-width geometry.
    # EEDI3H is horizontal (the transpose composition), EEDI3AA the fused
    # based_aa chain, NLMeans the tiled spatial filter.
    _add("eedi3", 16, _w, 0, _EEDI3_PARAMS, TOL_EEDI3_16)
    _add("eedi3", 32, _w, 0, _EEDI3_PARAMS, TOL_EEDI3_32)
    _add("eedi3h", 16, _w, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT)
    _add("eedi3h", 32, _w, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT)
    _add("eedi3aa", 16, _w, 0, _EEDI3AA_PARAMS, TOL_ORACLE_EXACT)
    _add("eedi3aa", 32, _w, 0, _EEDI3AA_PARAMS, TOL_EEDI3AA_32)
    _add("nlmeans", 16, _w, 0, _NLMEANS_PARAMS, TOL_NLMEANS_16)
    _add("nlmeans", 32, _w, 0, _NLMEANS_PARAMS, TOL_NLMEANS_32)

# Subsampled chroma (R1/R2): the chroma planes are half the luma width, so
# their visible rows are 315 * itemsize bytes at 630 px — a different pitch and
# alignment from luma's. This is the case that crashed the old aligned stream
# load in the DFTTest download helper (630-byte u16 rows are 22 mod 32).
for _w in WIDTHS:
    _add("dfttest", 32, _w, 0, _DFTTEST_PARAMS, TOL_F32_ULP, "yuv420")
    _add("dfttest", 16, _w, 0, _DFTTEST_PARAMS, TOL_U16_LSB, "yuv420")
    _add("gaussblur", 32, _w, 0, _GAUSS_PARAMS, 0.0, "yuv420")
    _add("gaussblur", 16, _w, 0, _GAUSS_PARAMS, 0.0, "yuv420")
    _add("bilateral", 32, _w, 0, _BILATERAL_PARAMS, TOL_F32_ULP, "yuv420")
    _add("bilateral", 16, _w, 0, _BILATERAL_PARAMS, TOL_U16_LSB, "yuv420")

# num_streams=4 under cropped geometry.  The reference/oracle stays serial, so
# these are also multi-stream-vs-serial checks at a non-tight pitch (EEDI3
# ships with num_streams=8, so 4 exercises the lower sweep point).
_add("nnedi3", 16, 630, 0, _NNEDI3_PARAMS, TOL_U16_LSB, streams=4)
_add("eedi3", 16, 630, 0, _EEDI3_PARAMS, TOL_EEDI3_16, streams=4)
_add("eedi3h", 16, 630, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT, streams=4)
_add("eedi3aa", 16, 630, 0, _EEDI3AA_PARAMS, TOL_ORACLE_EXACT, streams=4)
_add("nlmeans", 32, 630, 0, _NLMEANS_PARAMS, TOL_NLMEANS_32, streams=4)

# Vertical crop: a non-zero row origin and a shorter plane exercise the row
# staging / pad-origin path that the right-only crop leaves at row 0.
_add("dfttest", 32, 630, 0, _DFTTEST_PARAMS, TOL_F32_ULP,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("bilateral", 32, 630, 0, _BILATERAL_PARAMS, TOL_F32_ULP,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("nnedi3", 16, 630, 0, _NNEDI3_PARAMS, TOL_U16_LSB,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("eedi3", 16, 630, 0, _EEDI3_PARAMS, TOL_EEDI3_16,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("eedi3h", 16, 630, 0, _EEDI3H_PARAMS, TOL_ORACLE_EXACT,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("eedi3aa", 16, 630, 0, _EEDI3AA_PARAMS, TOL_ORACLE_EXACT,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)
_add("nlmeans", 32, 630, 0, _NLMEANS_PARAMS, TOL_NLMEANS_32,
     top=_CROP_TOP, bottom=_CROP_BOTTOM)


@pytest.mark.parametrize("spec,tol", CASES)
def test_cropped_width_matches_reference(spec, tol):
    """Each affected filter must run and track its reference at cropped size.

    ``compare_or_skip`` skips when the reference is unavailable and raises an
    AssertionError (with the captured subprocess tail) when vsfeel crashed,
    timed out or produced non-finite output.
    """
    payload = compare_or_skip(_GEOM_SCRIPT, [json.dumps(spec)], timeout=300)
    assert payload["width"] == spec["width"], (
        f"unexpected output width {payload['width']} for {spec}")
    assert payload["height"] == \
        HEIGHT - spec.get("top", 0) - spec.get("bottom", 0), (
        f"unexpected output height {payload['height']} for {spec}")
    assert payload["streams"] == spec.get("streams", 1)
    assert payload["frames"] == spec["frames"]
    maxdiff = float(payload["maxdiff"])
    assert maxdiff <= tol, (
        f"{spec['filter']} bits={spec['bits']} width={spec['width']} "
        f"streams={spec.get('streams', 1)} "
        f"crop=({spec.get('top', 0)},{spec.get('bottom', 0)}) "
        f"guide={spec.get('guide', 0)}: max diff {maxdiff} > tol {tol}")
