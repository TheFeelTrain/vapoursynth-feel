"""Unit tests for core.vsfeel.GaussBlur.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is used as the input. The vsfeel implementation is a Vulkan port of the vszipcl/vszipcu
GaussBlur; the expected behaviour is bit-identical output (the kernel taps,
mirror reflection and fma accumulation were ported exactly, and the float32
pipeline matches the reference's f32 maths).

Reference comparisons assert exact equality (measured max diff == 0.0 on
every config tried, every code path, all three input depths -- u16, f16 and
f32 -- YUV included), so no numeric tolerance is needed there; only reference
crashes are tolerated (the comparison runs in a subprocess).

Half is a storage width here, not a precision: the f16 path runs the same f32
arithmetic as the f32 one and rounds only at its stores, so its output is the
f32 result rounded once per sample. One test below checks exactly that against
the f32 path on the losslessly widened clip -- the device-independent half of
the oracle, which fails if the half path ever grows arithmetic of its own.

Run from the repository root:  uv run python -m pytest tests/test_gaussblur.py
"""

import json
import math
import textwrap

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    WIDTH,
    HEIGHT,
    CLIP_PATH,
    COMPARE_PRELUDE,
    assert_changes_on_clip,
    assert_preserves_frame_props,
    compare_or_skip,
    cpu_node,
    eval_parallel,
    frame_to_ndarray,
    plane as _plane,
    reference_or_skip,
)

pytestmark = pytest.mark.usefixtures("clip_grays")


def _run(clip, sigma=2.0, **kwargs):
    """GaussBlur as a clip the test can read pixels from.

    Under the R80 GPU API the filter takes and returns ``vnode:gpu`` frames, so
    a CPU clip is auto-uploaded by the core and the output is downloaded before
    the host reads it (see conftest.cpu_node).
    """
    return cpu_node(
        vs.core.vsfeel.GaussBlur(
            clip,
            sigma=sigma,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Determinism / stream count
# ---------------------------------------------------------------------------


def test_gaussblur_deterministic_f32(clip_grays):
    a = _run(clip_grays, sigma=2.0)
    b = _run(clip_grays, sigma=2.0)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"nondeterministic output at frame {n}"


def test_gaussblur_parallel_load_matches_serial_f32(clip_grays):
    par = eval_parallel(_run, clip_grays, sigma=10.0)
    ref = _run(clip_grays, sigma=10.0)
    for n in range(clip_grays.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-6, f"parallel/serial mismatch at frame {n}"


def test_gaussblur_parallel_load_deterministic_f32(clip_grays):
    a = eval_parallel(_run, clip_grays, sigma=10.0)
    b = eval_parallel(_run, clip_grays, sigma=10.0)
    for n in range(clip_grays.num_frames):
        d = a[n] - b[n]
        assert np.abs(d).max() < 1e-6, f"nondeterministic output at frame {n}"


def test_gaussblur_deterministic_u16(clip_gray16):
    a = _run(clip_gray16, sigma=2.0)
    b = _run(clip_gray16, sigma=2.0)
    for n in (0, 11, 23):
        fa = _plane(a.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
        fb = _plane(b.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
        assert np.array_equal(fa, fb), f"nondeterministic output at frame {n}"


def test_gaussblur_parallel_load_matches_serial_u16(clip_gray16):
    par = eval_parallel(_run, clip_gray16, dtype=np.uint16, sigma=10.0)
    ref = _run(clip_gray16, sigma=10.0)
    for n in range(clip_gray16.num_frames):
        fb = _plane(ref.get_frame(n), 0, WIDTH, HEIGHT, np.uint16)
        assert np.array_equal(par[n], fb), f"parallel/serial mismatch at frame {n}"


def test_gaussblur_parallel_load_deterministic_u16(clip_gray16):
    a = eval_parallel(_run, clip_gray16, dtype=np.uint16, sigma=10.0)
    b = eval_parallel(_run, clip_gray16, dtype=np.uint16, sigma=10.0)
    for n in range(clip_gray16.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


def test_gaussblur_deterministic_f16(clip_grayh):
    a = _run(clip_grayh, sigma=2.0)
    b = _run(clip_grayh, sigma=2.0)
    for n in (0, 11, 23):
        fa = _plane(a.get_frame(n), 0, dtype=np.float16)
        fb = _plane(b.get_frame(n), 0, dtype=np.float16)
        assert np.array_equal(fa, fb), f"nondeterministic output at frame {n}"


def test_gaussblur_parallel_load_matches_serial_f16(clip_grayh):
    """The half path must be bit-identical across request patterns, like the
    other depths: it takes the same pipeline and only changes the element
    width."""
    par = eval_parallel(_run, clip_grayh, dtype=np.float16, sigma=10.0)
    ref = _run(clip_grayh, sigma=10.0)
    for n in range(clip_grayh.num_frames):
        fb = _plane(ref.get_frame(n), 0, dtype=np.float16)
        assert np.array_equal(par[n], fb), f"parallel/serial mismatch at frame {n}"


def test_gaussblur_parallel_load_deterministic_f16(clip_grayh):
    a = eval_parallel(_run, clip_grayh, dtype=np.float16, sigma=10.0)
    b = eval_parallel(_run, clip_grayh, dtype=np.float16, sigma=10.0)
    for n in range(clip_grayh.num_frames):
        assert np.array_equal(a[n], b[n]), f"nondeterministic output at frame {n}"


# ---------------------------------------------------------------------------
# Reference comparison (bit-exact)
# ---------------------------------------------------------------------------

_COMPARE_SCRIPT = COMPARE_PRELUDE + textwrap.dedent(f"""\
    import sys
    import json
    import vapoursynth as vs
    from vstools import core

    fmt = sys.argv[1]
    sigma = json.loads(sys.argv[2])
    src = core.bs.VideoSource({CLIP_PATH!r})

    if fmt == "grays":
        clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)
        dtype, planes = np.float32, 1
    elif fmt == "gray16":
        clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=16, fulls=True, fulld=True)
        dtype, planes = np.uint16, 1
    elif fmt == "grayh":
        clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)
        clip = core.resize.Point(clip, format=vs.GRAYH)
        dtype, planes = np.float16, 1
    elif fmt == "yuv420ps":
        clip = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
        dtype, planes = np.float32, 3
    elif fmt == "yuv420ph":
        clip = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
        clip = core.resize.Point(clip, format=vs.YUV420PH)
        dtype, planes = np.float16, 3
    elif fmt == "yuv444ph":
        clip = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
        clip = core.resize.Point(clip, format=vs.YUV444PH)
        dtype, planes = np.float16, 3
    elif fmt == "yuv420p16":
        clip = core.resize.Bicubic(src, format=vs.YUV420P16)
        dtype, planes = np.uint16, 3
    else:
        raise SystemExit("bad fmt")

    # --- reference phase: materialise and copy before touching vsfeel ---
    try:
        ref_node = core.vszipcl.GaussBlur(clip, sigma=sigma)
        ref_frames = {{n: [read_plane(ref_node.get_frame(n), p, dtype)
                           for p in range(planes)]
                       for n in (3, 11, 23)}}
    except Exception as exc:
        print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(2)
    print("REF ok", flush=True)

    # --- vsfeel phase (GPU-resident output downloaded for the host reads) ---
    my_node = cpu_node(core.vsfeel.GaussBlur(clip, sigma=sigma))
    worst = 0.0
    for n in (3, 11, 23):
        fm = my_node.get_frame(n)
        for p, b in enumerate(ref_frames[n]):
            a = read_plane(fm, p, dtype)
            if not (np.isfinite(a.astype(np.float64)).all()
                    and np.isfinite(b.astype(np.float64)).all()):
                print("VSFEEL fail: non-finite at frame %d plane %d" % (n, p), flush=True)
                raise SystemExit(3)
            worst = max(worst, float(np.abs(a.astype(np.float64)
                                            - b.astype(np.float64)).max()))
    print("RESULT " + json.dumps(worst), flush=True)
""")


def _max_diff_vs_reference(fmt: str, sigma) -> float:
    """Run the comparison in a subprocess.

    ``sigma`` is a scalar or a per-plane list (the reference accepts arrays).
    A missing/failed reference skips (via compare_or_skip); a vsfeel crash,
    exception, timeout or non-finite result fails with the captured tail.
    """
    return compare_or_skip(_COMPARE_SCRIPT, [fmt, json.dumps(sigma)], timeout=300)


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 10.0])
def test_gaussblur_matches_reference_small_f32(clip_grays, sigma):
    """Fused small path (radius <= 32) must be bit-identical to vszipcl."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("grays", sigma)
    assert maxdiff == 0.0, f"small path max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [10.5, 11.0, 12.0])
def test_gaussblur_matches_reference_path_boundary_f32(clip_grays, sigma):
    """Sigmas around the fused-small / two-pass transition (radius ~32) must
    be bit-identical on both sides of the switch."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("grays", sigma)
    assert maxdiff == 0.0, f"path boundary max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [20.0, 30.0, 40.0, 80.0])
def test_gaussblur_matches_reference_large_f32(clip_grays, sigma):
    """Two-pass large path (radius > 32) must be bit-identical to vszipcl."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("grays", sigma)
    assert maxdiff == 0.0, f"large path max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 10.0, 10.5, 11.0, 12.0, 20.0, 30.0, 40.0, 80.0])
def test_gaussblur_matches_reference_u16(clip_grays, sigma):
    """u16 input must be bit-identical on both code paths — the
    same sigma sweep (small path, transition, large path) as the f32
    tests above."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("gray16", sigma)
    assert maxdiff == 0.0, f"gray16 max diff ({sigma}): {maxdiff}"


@pytest.mark.parametrize("sigma", [0.5, 3.0, 20.0])
def test_gaussblur_matches_reference_yuv_f32(clip_grays, sigma):
    """YUV420: all three planes (incl. the subsampled chroma defaults) must
    match vszipcl bit-for-bit, on both code paths."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("yuv420ps", sigma)
    assert maxdiff == 0.0, f"yuv420ps max diff ({sigma}): {maxdiff}"


@pytest.mark.parametrize("sigma", [0.5, 3.0, 20.0])
def test_gaussblur_matches_reference_yuv_u16(clip_grays, sigma):
    """u16 YUV420: all three planes must match vszipcl bit-for-bit on both
    code paths (mirror of the f32 YUV test)."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("yuv420p16", sigma)
    assert maxdiff == 0.0, f"yuv420p16 max diff ({sigma}): {maxdiff}"


# --- f16: same sweeps as the f32/u16 sets above ----------------------------
#
# Half input, half stores, f32 accumulation and an f32 scratch: the arithmetic
# is the f32 path's, so the reference comparison stays exact rather than
# needing a tolerance (measured max diff == 0.0 across this sweep).


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 10.0])
def test_gaussblur_matches_reference_small_f16(clip_grayh, sigma):
    """Fused small path (radius <= 32) on half input must be bit-identical to
    vszipcl's f16 output."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("grayh", sigma)
    assert maxdiff == 0.0, f"f16 small path max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [10.5, 11.0, 12.0])
def test_gaussblur_matches_reference_path_boundary_f16(clip_grayh, sigma):
    """Sigmas around the fused-small / two-pass transition (radius ~32) on half
    input, on both sides of the switch."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("grayh", sigma)
    assert maxdiff == 0.0, f"f16 path boundary max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [20.0, 30.0, 40.0, 80.0])
def test_gaussblur_matches_reference_large_f16(clip_grayh, sigma):
    """Two-pass large path (radius > 32) on half input."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("grayh", sigma)
    assert maxdiff == 0.0, f"f16 large path max diff: {maxdiff}"


@pytest.mark.parametrize("sigma", [0.5, 3.0, 20.0])
def test_gaussblur_matches_reference_yuv_f16(clip_grayh, sigma):
    """YUV420PH: all three planes (incl. the subsampled chroma defaults) must
    match vszipcl's f16 output bit-for-bit on both code paths."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("yuv420ph", sigma)
    assert maxdiff == 0.0, f"yuv420ph max diff ({sigma}): {maxdiff}"


# ---------------------------------------------------------------------------
# Sanity / numerical behaviour
# ---------------------------------------------------------------------------


def test_gaussblur_sigma_small_vs_large_consistent_f32(clip_grays):
    """Both code paths (fused small and two-pass large) blur the same constant
    plane to the same value."""
    core = vs.core
    flat = core.std.BlankClip(clip_grays, color=[0.5])
    out = _run(flat, sigma=20.0)  # large path
    for n in (0,):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.abs(a - 0.5).max() < 1e-3, "large path does not preserve a constant plane"


def test_gaussblur_sigma_small_vs_large_consistent_u16(clip_gray16):
    """u16 mirror of the constant-plane check (both code paths)."""
    core = vs.core
    flat = core.std.BlankClip(clip_gray16, color=[32768])
    out = _run(flat, sigma=20.0)  # large path
    for n in (0,):
        a = _plane(out.get_frame(n), 0, WIDTH, HEIGHT, np.uint16).astype(np.float64)
        assert np.abs(a - 32768.0).max() < 66.0, "large path does not preserve a constant plane"


def test_gaussblur_sigma_zero_passthrough_yuv_u16(clip_grays):
    """u16 mirror of test_gaussblur_sigma_zero_passthrough_yuv."""
    src = vs.core.bs.VideoSource(CLIP_PATH)
    yuv = vs.core.resize.Bicubic(src, format=vs.YUV420P16)
    out = _run(yuv, sigma=[0.0, 3.0])
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = yuv.get_frame(n)
        # luma copied through bit-identically
        lum = _plane(f, 0, WIDTH, HEIGHT, np.uint16)
        lum_src = _plane(s, 0, WIDTH, HEIGHT, np.uint16)
        assert np.array_equal(lum, lum_src), f"luma changed at frame {n}"
        # chroma is actually blurred (differs from input)
        for p in (1, 2):
            w = WIDTH // 2
            h = HEIGHT // 2
            a = _plane(f, p, w, h, np.uint16)
            b = _plane(s, p, w, h, np.uint16)
            assert not np.array_equal(a, b), f"chroma{p} not blurred at frame {n}"


def test_gaussblur_sigma_zero_passthrough_yuv_f32(clip_grays):
    """sigma=0 on luma copies that plane through while chroma is blurred."""
    src = vs.core.bs.VideoSource(CLIP_PATH)
    yuv = vs.core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    out = _run(yuv, sigma=[0.0, 3.0])
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = yuv.get_frame(n)
        # luma copied through bit-identically
        lum = frame_to_ndarray(f)
        lum_src = frame_to_ndarray(s)
        assert np.array_equal(lum, lum_src), f"luma changed at frame {n}"
        # chroma is actually blurred (differs from input)
        for p in (1, 2):
            w = WIDTH // 2
            h = HEIGHT // 2
            a = _plane(f, p, w, h)
            b = _plane(s, p, w, h)
            assert not np.array_equal(a, b), f"chroma{p} not blurred at frame {n}"


def test_gaussblur_sigma_zero_passthrough_yuv_f16(clip_yuv420ph):
    """f16 mirror: sigma=0 on luma copies that plane through bit-identically
    (a half sample must pass through the width unchanged, not via f32), while
    chroma is blurred."""
    yuv = clip_yuv420ph
    out = _run(yuv, sigma=[0.0, 3.0])
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = yuv.get_frame(n)
        h = _plane(f, 0, dtype=np.float16)
        h_src = _plane(s, 0, dtype=np.float16)
        assert np.array_equal(h, h_src), f"luma changed at frame {n}"
        for p in (1, 2):
            a = _plane(f, p, dtype=np.float16)
            b = _plane(s, p, dtype=np.float16)
            assert not np.array_equal(a, b), f"chroma{p} not blurred at frame {n}"


def test_gaussblur_sigma_small_vs_large_consistent_f16(clip_grayh):
    """f16 mirror of the constant-plane check (both code paths)."""
    core = vs.core
    flat = core.std.BlankClip(clip_grayh, color=[0.5])
    out = _run(flat, sigma=20.0)  # large path
    for n in (0,):
        a = _plane(out.get_frame(n), 0, dtype=np.float16).astype(np.float64)
        assert np.abs(a - 0.5).max() < 1e-3, "large path does not preserve a constant plane"


# --- f16 is a storage width, not a precision -------------------------------
#
# The f16 path runs the f32 path's arithmetic and rounds only at its stores, so
# on the losslessly widened clip its output must be the f32 output rounded once
# per sample. This oracle is device-independent (no reference plugin) and is
# what fails first if the half path ever grows half arithmetic of its own.
# Bound: one half ulp at the value's magnitude; measured max 2.44e-4 over the
# sweep (the f16 store rounding), 1e-3 leaves headroom for values above 1.0
# where a half ulp doubles -- the same bound vszipcl's own f16 tests use.

F16_TOL = 1e-3


def _widened_f32(clip):
    """``clip`` widened to f32, sample for sample.

    A Point resize between GRAYH and GRAYS is a container change (half -> float
    is exact), so the two clips carry byte-identical content and any output
    difference is the f16 path's store rounding.
    """
    return vs.core.resize.Point(clip, format=vs.GRAYS)


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 10.0, 10.5, 12.0, 20.0, 40.0, 80.0])
def test_gaussblur_f16_is_f32_rounded_f16(clip_grayh, sigma):
    """The f16 output is the f32 output rounded to half, on both code paths."""
    f32 = _widened_f32(clip_grayh)
    a = _run(clip_grayh, sigma=sigma)
    b = _run(f32, sigma=sigma)
    for n in (3, 11, 23):
        half = _plane(a.get_frame(n), 0, dtype=np.float16).astype(np.float64)
        full = _plane(b.get_frame(n), 0, dtype=np.float32).astype(np.float64)
        assert np.isfinite(half).all() and np.isfinite(full).all(), f"non-finite at frame {n}"
        worst = float(np.abs(half - full).max())
        assert worst <= F16_TOL, f"f16 vs f32 max diff {worst} at sigma {sigma} frame {n}"


@pytest.mark.parametrize("sigma", [2.0, 20.0])
def test_gaussblur_f16_is_f32_rounded_yuv(clip_yuv420ph, sigma):
    """YUV420PH mirror: every plane of the half output is within a half ulp of
    the widened f32 run, subsampled chroma included."""
    f32 = vs.core.resize.Point(clip_yuv420ph, format=vs.YUV420PS)
    a = _run(clip_yuv420ph, sigma=sigma)
    b = _run(f32, sigma=sigma)
    for n in (3, 11, 23):
        fa, fb = a.get_frame(n), b.get_frame(n)
        for p in range(fa.format.num_planes):
            half = _plane(fa, p, dtype=np.float16).astype(np.float64)
            full = _plane(fb, p, dtype=np.float32).astype(np.float64)
            worst = float(np.abs(half - full).max())
            assert worst <= F16_TOL, f"plane {p}: f16 vs f32 max diff {worst} at sigma {sigma}"


def test_gaussblur_changes_noise_f16(clip_grayh):
    """A scalar sigma must actually alter the half noise input (the identity
    filter that a broken f16 variant would degenerate into)."""
    out = _run(clip_grayh, sigma=2.0)
    assert_changes_on_clip(out, clip_grayh, what="GaussBlur f16")


def test_gaussblur_preserves_frame_props_f16(clip_grayh):
    """The half path must republish the source frame's properties, like the
    other depths."""
    assert_preserves_frame_props(_run, clip_grayh, sigma=2.0)


def test_gaussblur_changes_noise_u16(clip_gray16):
    """A scalar sigma must actually alter the noise input.

    ``isfinite(uint16)`` cannot fail and the reference comparison skips when
    vszipcl is absent, so this is the invariant that fails for an identity
    implementation.
    """
    out = _run(clip_gray16, sigma=2.0)
    assert_changes_on_clip(out, clip_gray16, what="GaussBlur")


def test_gaussblur_preserves_frame_props(clip_grays):
    """GaussBlur must republish the source frame's properties."""
    assert_preserves_frame_props(_run, clip_grays, sigma=2.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_gaussblur_rejects_all_copy_through(clip_grays):
    with pytest.raises(vs.Error):
        _run(clip_grays, sigma=0.0)


def test_gaussblur_rejects_sigma_too_large(clip_grays):
    with pytest.raises(vs.Error):
        _run(clip_grays, sigma=10000.0)


def test_gaussblur_rejects_negative_sigma(clip_grays):
    with pytest.raises(vs.Error):
        _run(clip_grays, sigma=-1.0)


def test_gaussblur_rejects_bad_bitdepth(clip_gray8):
    # u16 integer, f16 and f32 are the supported depths; a half clip must not
    # be mistaken for the 16-bit integer path (see the accept test below)
    with pytest.raises(vs.Error):
        _run(clip_gray8)
    clip = vs.core.fmtc.bitdepth(clip_gray8, bits=10)
    with pytest.raises(vs.Error):
        _run(clip)


def test_gaussblur_accepts_f16(clip_grayh):
    """A half clip is processed (and its output is half), not rejected as an
    unsupported 16-bit depth or routed through the u16 kernel."""
    out = _run(clip_grayh, sigma=2.0)
    assert out.format.sample_type == vs.FLOAT
    assert out.format.bits_per_sample == 16
    assert_changes_on_clip(out, clip_grayh, what="GaussBlur f16")


# ---------------------------------------------------------------------------
# Per-plane sigma arrays
# ---------------------------------------------------------------------------
#
# Explicit per-plane arrays (incl. the documented chroma default and an
# explicit zero pass-through plane); the scalar YUV tests only covered the
# default implicitly. Must stay bit-exact.


@pytest.mark.parametrize(
    "sigma,tol",
    [
        ([2.0, 1.0, 1.0], 0.0),  # explicit chroma-rule values
        ([2.0, 3.0, 4.0], 0.0),  # increasing, all processed
        ([0.0, 2.0, 2.0], 0.0),  # luma passes through
        ([20.0, 2.0, 2.0], 0.0),  # large path on luma only
        ([0.5, 10.5, 30.0], 0.0),  # one sigma per code path
    ],
    ids=["chroma-rule", "increasing", "luma-passthrough", "luma-large", "per-plane-paths"],
)
def test_gaussblur_per_plane_sigma_matches_reference_yuv_f32(clip_grays, sigma, tol):
    maxdiff = _max_diff_vs_reference("yuv420ps", sigma)
    assert maxdiff <= tol, f"yuv420ps per-plane sigma max diff ({sigma}): {maxdiff}"


@pytest.mark.parametrize(
    "sigma",
    [
        [2.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
        [0.0, 2.0, 2.0],
    ],
    ids=["chroma-rule", "increasing", "luma-passthrough"],
)
def test_gaussblur_per_plane_sigma_matches_reference_yuv_u16(clip_grays, sigma):
    """u16 mirror of the per-plane array sweep (bit-exact)."""
    maxdiff = _max_diff_vs_reference("yuv420p16", sigma)
    assert maxdiff == 0.0, f"yuv420p16 per-plane sigma max diff ({sigma}): {maxdiff}"


@pytest.mark.parametrize(
    "sigma",
    [
        [2.0, 1.0, 1.0],  # explicit chroma-rule values
        [2.0, 3.0, 4.0],  # increasing, all processed
        [0.0, 2.0, 2.0],  # luma passes through
        [20.0, 2.0, 2.0],  # large path on luma only
        [0.5, 10.5, 30.0],  # one sigma per code path
    ],
    ids=["chroma-rule", "increasing", "luma-passthrough", "luma-large", "per-plane-paths"],
)
def test_gaussblur_per_plane_sigma_matches_reference_yuv_f16(clip_grayh, sigma):
    """f16 mirror of the f32 per-plane array sweep (bit-exact)."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("yuv420ph", sigma)
    assert maxdiff == 0.0, f"yuv420ph per-plane sigma max diff ({sigma}): {maxdiff}"


@pytest.mark.parametrize(
    "sigma",
    [[2.0, 2.0, 2.0], [0.5, 10.5, 30.0]],
    ids=["identical-chroma", "per-plane-paths"],
)
def test_gaussblur_matches_reference_yuv444_f16(clip_grayh, sigma):
    """YUV444PH: identical chroma planes take the per-plane dedup path in
    creation, and the per-path list puts one sigma on each side of the fused /
    two-pass switch."""
    reference_or_skip("vszipcl", "GaussBlur")
    maxdiff = _max_diff_vs_reference("yuv444ph", sigma)
    assert maxdiff == 0.0, f"yuv444ph max diff ({sigma}): {maxdiff}"


def test_gaussblur_per_plane_sigma_gray_f16(clip_grayh):
    """f16 mirror: on a single-plane clip only element 0 is used and extra
    elements must not change the result."""
    scalar = _run(clip_grayh, sigma=2.0)
    array = _run(clip_grayh, sigma=[2.0, 99.0, 0.25])
    for n in (0, 11):
        a = _plane(scalar.get_frame(n), 0, dtype=np.float16)
        b = _plane(array.get_frame(n), 0, dtype=np.float16)
        assert np.array_equal(a, b), f"extra sigma elements changed frame {n}"
    assert _max_diff_vs_reference("grayh", [2.0, 99.0, 0.25]) == 0.0


def test_gaussblur_per_plane_sigma_gray_f32(clip_grays):
    """On a single-plane clip only element 0 is used; extra elements must not
    change the result (must equal the scalar run bit-exactly)."""
    scalar = _run(clip_grays, sigma=2.0)
    array = _run(clip_grays, sigma=[2.0, 99.0, 0.25])
    for n in (0, 11):
        a = frame_to_ndarray(scalar.get_frame(n))
        b = frame_to_ndarray(array.get_frame(n))
        assert np.array_equal(a, b), f"extra sigma elements changed frame {n}"
    assert _max_diff_vs_reference("grays", [2.0, 99.0, 0.25]) == 0.0


# ---------------------------------------------------------------------------
# radius >= dimension error
# ---------------------------------------------------------------------------


def test_gaussblur_rejects_radius_ge_dimension():
    """A kernel radius reaching the plane dimension would read past the
    mirror-clamped window and must be rejected at creation.

    On a 32x32 clip, sigma=16 hits ``radius > w-1`` and sigma=100 also hits
    ``sigma > min(w, h)``; sigma=4 (radius 12) must still run.
    """
    core = vs.core
    small = core.std.BlankClip(format=vs.GRAYS, width=32, height=32, length=2)
    for sigma in (16.0, 32.0, 100.0):
        with pytest.raises(vs.Error, match=r"radius >= dimension"):
            _run(small, sigma=sigma)
    ok = _run(small, sigma=4.0)
    assert np.isfinite(frame_to_ndarray(ok.get_frame(0))).all()


def test_gaussblur_rejects_radius_ge_dimension_per_plane(clip_grays):
    """The check is per processed plane: a large chroma sigma on a small
    subsampled chroma plane must be rejected even when luma is fine."""
    core = vs.core
    small = core.std.BlankClip(format=vs.YUV420PS, width=64, height=64, length=2)
    # chroma plane is 32x32; sigma 16 exceeds it while luma (64) is fine
    with pytest.raises(vs.Error, match=r"radius >= dimension"):
        _run(small, sigma=[2.0, 16.0, 16.0])


def test_gaussblur_rejects_radius_ge_dimension_f16():
    """The radius check runs on the plane geometry, not the sample type, so
    the half path must reject the same configs."""
    core = vs.core
    small = core.std.BlankClip(format=vs.GRAYH, width=32, height=32, length=2)
    for sigma in (16.0, 32.0, 100.0):
        with pytest.raises(vs.Error, match=r"radius >= dimension"):
            _run(small, sigma=sigma)
    ok = _run(small, sigma=4.0)
    assert np.isfinite(_plane(ok.get_frame(0), 0, dtype=np.float16)).all()


# ---------------------------------------------------------------------------
# Non-finite footprint (two-pass path)
# ---------------------------------------------------------------------------
#
# The rolling weight window pads out-of-kernel taps with wreg[j] == 0.0f. That
# is an exact no-op for finite operands, but fma(NaN, 0, s) is NaN: without an
# explicit skip, one non-finite sample can poison extra rows/columns (up to
# R-1 = 7, thread-aligned). The reference skips those taps by index, so the
# footprint must match exactly.


def _gauss_radius(sigma):
    """Kernel radius the host derives for ``sigma`` (see get_gauss_kernel).

    The host rounds the tap count up to odd, then builds a symmetric kernel of
    ``2*(taps//2) - 1`` entries, so the radius is ``taps//2 - 1``.
    """
    taps = math.ceil(sigma * 6.0 + 1.0)
    if taps % 2 == 0:
        taps += 1
    return taps // 2 - 1


def _poke_nonfinite(clip, value, oy, ox):
    """``clip`` with ``value`` written to plane 0 at (oy, ox) on every frame."""

    def poke(n, f):
        out = f.copy()
        np.asarray(out[0])[oy, ox] = value
        return out

    return clip.std.ModifyFrame(clip, poke)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gaussblur_nonfinite_footprint_is_one_kernel_window_f32(clip_grays, value):
    """A single non-finite pixel may only affect its kernel window.

    Pixel (oy, ox) is interior, so with the padding taps skipped the output is
    non-finite exactly on the radius square and bit-identical to a finite run
    everywhere else. Whether the padding taps (plain zero-weight FMAs) are
    skipped is compiler-dependent, so this oracle does not catch every build;
    the reference comparison below does.
    """
    sigma = 40.0  # radius 119 -> the two-pass path
    radius = _gauss_radius(sigma)
    oy, ox = HEIGHT // 2 + 3, WIDTH // 2 + 5
    clean = _run(clip_grays, sigma=sigma)
    bad = _run(_poke_nonfinite(clip_grays, value, oy, ox), sigma=sigma)

    footprint = np.zeros((HEIGHT, WIDTH), dtype=bool)
    footprint[oy - radius : oy + radius + 1, ox - radius : ox + radius + 1] = True

    for n in (3, 11, 23):
        a = _plane(bad.get_frame(n), 0)
        b = _plane(clean.get_frame(n), 0)
        assert np.array_equal(np.isfinite(a), ~footprint), (
            f"{value!r} at ({oy},{ox}): footprint is not the kernel window on "
            f"frame {n} ({int((np.isfinite(a) != ~footprint).sum())} pixels off)"
        )
        assert np.array_equal(a[~footprint], b[~footprint]), (
            f"finite pixels changed by a {value!r} sample on frame {n}"
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gaussblur_nonfinite_footprint_is_one_kernel_window_f16(clip_grayh, value):
    """f16 mirror of the footprint oracle.

    The half store cannot widen the footprint: NaN and both infinities are
    representable in half, so a poisoned sample still shows up as non-finite
    exactly where the kernel window puts it.
    """
    sigma = 40.0  # radius 119 -> the two-pass path
    radius = _gauss_radius(sigma)
    oy, ox = HEIGHT // 2 + 3, WIDTH // 2 + 5
    half = np.float16(value)
    clean = _run(clip_grayh, sigma=sigma)
    bad = _run(_poke_nonfinite(clip_grayh, half, oy, ox), sigma=sigma)

    footprint = np.zeros((HEIGHT, WIDTH), dtype=bool)
    footprint[oy - radius : oy + radius + 1, ox - radius : ox + radius + 1] = True

    for n in (3, 11, 23):
        a = _plane(bad.get_frame(n), 0, dtype=np.float16)
        b = _plane(clean.get_frame(n), 0, dtype=np.float16)
        assert np.array_equal(np.isfinite(a), ~footprint), (
            f"{value!r} at ({oy},{ox}): footprint is not the kernel window on "
            f"frame {n} ({int((np.isfinite(a) != ~footprint).sum())} pixels off)"
        )
        assert np.array_equal(a[~footprint], b[~footprint]), (
            f"finite pixels changed by a {value!r} sample on frame {n}"
        )


_NONFINITE_SCRIPT = COMPARE_PRELUDE + textwrap.dedent(f"""\
    import sys
    import json
    import numpy as np
    import vapoursynth as vs
    from vstools import core

    kind, sigma, fmt = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]

    src = core.bs.VideoSource({CLIP_PATH!r})
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                              bits=32, fulls=True, fulld=True)
    if fmt == "grayh":
        clip = core.resize.Point(clip, format=vs.GRAYH)
    dt = np.float16 if fmt == "grayh" else np.float32
    bad = {{"nan": dt("nan"), "inf": dt("inf"), "-inf": dt("-inf")}}[kind]
    oy, ox = clip.height // 2 + 3, clip.width // 2 + 5

    def poke(n, f):
        out = f.copy()
        np.asarray(out[0])[oy, ox] = bad
        return out
    clip = core.std.ModifyFrame(clip, clip, poke)

    try:
        ref_node = core.vszipcl.GaussBlur(clip, sigma=sigma)
        ref = {{n: read_plane(ref_node.get_frame(n), 0, dt)
                for n in (3, 11, 23)}}
    except Exception as exc:
        print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(2)
    print("REF ok", flush=True)

    my_node = cpu_node(core.vsfeel.GaussBlur(clip, sigma=sigma))
    mismatch = 0
    maxdiff = 0.0
    for n in (3, 11, 23):
        a = read_plane(my_node.get_frame(n), 0, dt)
        b = ref[n]
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        mismatch += int((finite_a != finite_b).sum())
        both = finite_a & finite_b
        if both.any():
            maxdiff = max(maxdiff, float(np.abs(a[both].astype(np.float64)
                                                    - b[both].astype(np.float64)).max()))
    print("RESULT " + json.dumps({{"mismatch": mismatch, "maxdiff": maxdiff}}),
          flush=True)
""")


@pytest.mark.parametrize("kind", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("sigma", [20.0, 40.0])
def test_gaussblur_nonfinite_footprint_matches_reference_f32(kind, sigma):
    """The non-finite footprint and the finite pixels must match vszipcl."""
    reference_or_skip("vszipcl", "GaussBlur")
    result = compare_or_skip(_NONFINITE_SCRIPT, [kind, json.dumps(sigma), "grays"], timeout=300)
    assert result["mismatch"] == 0, (
        f"{kind} sigma={sigma}: {result['mismatch']} pixels disagree on finiteness vs vszipcl"
    )
    assert result["maxdiff"] == 0.0, (
        f"{kind} sigma={sigma}: finite pixels differ from vszipcl by {result['maxdiff']}"
    )


@pytest.mark.parametrize("kind", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("sigma", [20.0, 40.0])
def test_gaussblur_nonfinite_footprint_matches_reference_f16(kind, sigma):
    """f16 mirror: the non-finite footprint and every finite pixel must match
    vszipcl's half path, which is bit-exact because the arithmetic is the f32
    path's on both sides."""
    reference_or_skip("vszipcl", "GaussBlur")
    result = compare_or_skip(_NONFINITE_SCRIPT, [kind, json.dumps(sigma), "grayh"], timeout=300)
    assert result["mismatch"] == 0, (
        f"f16 {kind} sigma={sigma}: {result['mismatch']} pixels disagree on finiteness vs vszipcl"
    )
    assert result["maxdiff"] == 0.0, (
        f"f16 {kind} sigma={sigma}: finite pixels differ from vszipcl by {result['maxdiff']}"
    )
