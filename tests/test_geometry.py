"""Regression tests for cropped / non-tightly-pitched plane geometry (R1, R2).

Both findings are width-dependent host-copy bugs that a 640-pixel fixture
cannot see:

* R1: DFTTest's downloaded-row helper used an aligned 32-byte stream load on
  a source row that is not 32-byte aligned.  A 630-pixel float32 row starts at
  byte offset 2520 for row 1 (24 mod 32) and crashed the CPU copy helper.
* R2: Bilateral, GaussBlur and BM3D copied ``width * height * bytes_per_sample``
  in one operation while the VapourSynth pitch is rounded up (e.g. 2560 bytes
  for a 630-pixel float32 plane), so the upload read past/before the visible
  rows and dropped the tail of the last row.

The tests crop a 640-wide Gray clip to 630 and 638 pixels, run the affected
vsfeel filters, and compare against the matching vszipcl reference.  Because
the R1 crash is a SIGSEGV in the CPU copy helper, every comparison runs in a
subprocess with a timeout so a crash is reported as a test failure instead of
killing the pytest process.

Tolerances are measurement-bounded (measured values are recorded next to each
case); the reference comparisons use ``num_streams=1``.
"""

import json
import textwrap

import pytest

from conftest import COMPARE_PRELUDE, NOISE_MKV, compare_or_skip

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
# The yuv420 cases exercise half-width chroma: at 630 luma px the chroma rows
# are 315 samples, i.e. 630 bytes for u16 (22 mod 32) — the alignment that made
# the old DFTTest download helper fault.
TOL_F32_ULP = 1e-6
TOL_U16_LSB = 1.0
TOL_BM3D = 0.02

_GEOM_SCRIPT = COMPARE_PRELUDE + textwrap.dedent(f"""\
    import json
    import sys
    import vapoursynth as vs
    from vstools import core

    spec = json.loads(sys.argv[1])
    bits = spec["bits"]
    width = spec["width"]
    guide_on = spec.get("guide", 0)
    color = spec.get("color", "gray")
    params = spec.get("params", {{}})
    frames = spec.get("frames", [0, 1, 2])

    core.max_cache_size = 512
    src = core.bs.VideoSource({NOISE_MKV!r})
    if color == "yuv420":
        # subsampled chroma: the chroma planes are half width, so their rows
        # are a different (and typically less aligned) length than luma's
        main = core.fmtc.bitdepth(src, bits=bits, fulls=True, fulld=True)
    else:
        gray = core.std.ShufflePlanes(src, 0, vs.GRAY)
        main = core.fmtc.bitdepth(gray, bits=bits, fulls=True, fulld=True)
    main = core.std.Crop(main, right=({CROP_FROM} - width))
    dtype = np.float32 if bits == 32 else np.uint16
    planes = list(range(main.format.num_planes))

    # Guide/reference clip from a different chain (distinct frame allocations).
    guide = None
    if guide_on:
        guide = core.resize.Bicubic(main, width=main.width, height=main.height,
                                    format=main.format)

    filter_ = spec["filter"]

    # --- reference phase: materialise and copy before touching vsfeel ---
    try:
        if filter_ == "dfttest":
            ref_node = core.vszipcl.DFTTest(main, num_streams=1, **params)
        elif filter_ == "gaussblur":
            ref_node = core.vszipcl.GaussBlur(main, num_streams=1, **params)
        elif filter_ == "bilateral":
            ref_node = core.vszipcl.Bilateral(main, ref=guide, num_streams=1, **params)
        elif filter_ == "bm3d":
            ref_node = core.vszipcl.BM3Dv2(main, num_streams=1, **params)
        else:
            raise SystemExit("bad filter %r" % filter_)
        ref_frames = [[read_plane(ref_node.get_frame(n), p, dtype)
                       for p in planes] for n in frames]
    except Exception as exc:
        print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(2)
    print("REF ok", flush=True)

    # --- vsfeel phase ---
    try:
        if filter_ == "dfttest":
            my_node = core.vsfeel.DFTTest(main, num_streams=1, **params)
        elif filter_ == "gaussblur":
            my_node = core.vsfeel.GaussBlur(main, num_streams=1, **params)
        elif filter_ == "bilateral":
            my_node = core.vsfeel.Bilateral(main, ref=guide, num_streams=1, **params)
        else:
            my_node = core.vsfeel.BM3Dv2(main, num_streams=1, **params)
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
                                  "planes": planes,
                                  "frames": list(frames)}}), flush=True)
""")

_BILATERAL_PARAMS = {"sigma_spatial": 3.0, "sigma_color": 0.05}
_DFTTEST_PARAMS = {"tbsize": 3}
_GAUSS_PARAMS = {"sigma": 2.0}
_BM3D_PARAMS = {"sigma": 0.7, "radius": 1, "bm_range": 16, "ps_range": 7,
                "block_step": 4}

CASES = []


def _add(filter_, bits, width, guide, params, tol, color="gray"):
    spec = {"filter": filter_, "bits": bits, "width": width, "guide": guide,
            "color": color, "params": params, "frames": FRAMES}
    cid = "%s-%s-w%d%s%s" % (filter_, "f32" if bits == 32 else "u16", width,
                             "-guide" if guide else "",
                             "-yuv420" if color == "yuv420" else "")
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


@pytest.mark.parametrize("spec,tol", CASES)
def test_cropped_width_matches_reference(spec, tol):
    """Each affected filter must run and track vszipcl at 630/638 px.

    ``compare_or_skip`` skips when the reference is unavailable and raises an
    AssertionError (with the captured subprocess tail) when vsfeel crashed,
    timed out or produced non-finite output.
    """
    payload = compare_or_skip(_GEOM_SCRIPT, [json.dumps(spec)], timeout=300)
    assert payload["width"] == spec["width"], (
        f"unexpected output width {payload['width']} for {spec}")
    assert payload["frames"] == spec["frames"]
    maxdiff = float(payload["maxdiff"])
    assert maxdiff <= tol, (
        f"{spec['filter']} bits={spec['bits']} width={spec['width']} "
        f"guide={spec.get('guide', 0)}: max diff {maxdiff} > tol {tol}")
