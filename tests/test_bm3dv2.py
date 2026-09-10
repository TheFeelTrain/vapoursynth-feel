"""Unit tests for core.vsfeel.BM3Dv2.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The noise content exposes the boundary/clamping behaviour of the
temporal pipeline: frames 0..23 must all produce finite output with no NaN
(regression: boundary frames intermittently produced NaN before the atomics
were made visible to the aggregation kernel).

Run from the repository root:  python -m pytest tests/test_bm3dv2.py
"""

import json
import textwrap
import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    NOISE_MKV, COMPARE_PRELUDE, ReferenceUnavailable, assert_gray32,
    frame_to_ndarray, plane_to_ndarray, run_compare_subprocess,
)

pytestmark = pytest.mark.usefixtures("noise_gray")

SIGMA = 0.7
BM_RANGE = 16
PS_RANGE = 7
BLOCK_STEP = 4


def _run(clip, radius=2, num_streams=1, **kwargs):
    return vs.core.vsfeel.BM3Dv2(
        clip,
        sigma=SIGMA,
        radius=radius,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        num_streams=num_streams,
        **kwargs,
    )


def _check_all_frames_finite(clip, **kwargs):
    out = _run(clip, **kwargs)
    assert_gray32(out)
    for n in range(out.num_frames):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


def _eval_parallel(clip, **kwargs):
    """Evaluate every frame concurrently, the way vspipe does.

    Sequential get_frame() calls keep at most one frame in flight in the
    filter, so the pipelined multi-stream path (parallel arAllFramesReady
    invocations, reused command buffers and fences across streams) is never
    exercised. Requesting all frames from worker threads at once forces the
    deep pipeline and with it the cross-stream synchronization.
    """
    out = _run(clip, **kwargs)
    frames = [None] * out.num_frames

    def worker(n):
        frames[n] = frame_to_ndarray(out.get_frame(n))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(out.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(f is not None for f in frames)
    return frames


def test_bm3dv2_parallel_load_matches_serial(noise_gray):
    """Parallel request load with num_streams=4 must produce the same pixel
    values as the serial path.

    This is the request pattern that exposed stale descriptor bindings, fence
    misuse and command pool reuse violations under load on strict drivers
    (black or garbage output only when frames are processed concurrently).
    """
    par = _eval_parallel(noise_gray, radius=2, num_streams=4)
    ref = _run(noise_gray, radius=2, num_streams=1)
    for n in range(noise_gray.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"parallel/serial mismatch at frame {n}"


def test_bm3dv2_parallel_load_deterministic(noise_gray):
    """Two parallel num_streams=4 runs must produce identical output.

    A fence attached to two in-flight submissions makes frame results depend
    on the completion order of the streams, which only shows up when many
    frames are in flight at once.
    """
    a = _eval_parallel(noise_gray, radius=2, num_streams=4)
    b = _eval_parallel(noise_gray, radius=2, num_streams=4)
    for n in range(noise_gray.num_frames):
        d = a[n] - b[n]
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4])
def test_bm3dv2_no_nan_all_frames(noise_gray, radius):
    """Output must be finite on every frame (incl. boundaries) for each radius.

    A brand new filter instance is created per parametrized test.
    """
    _check_all_frames_finite(noise_gray, radius=radius, num_streams=1)


@pytest.mark.parametrize("num_streams", [2, 4])
def test_bm3dv2_no_nan_all_frames_multi_stream(noise_gray, num_streams):
    _check_all_frames_finite(noise_gray, radius=2, num_streams=num_streams)


def test_bm3dv2_deterministic(noise_gray):
    a = _run(noise_gray, radius=2, num_streams=1)
    b = _run(noise_gray, radius=2, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        # BM3D's aggregation uses float atomics, so two runs differ only by
        # atomic-order rounding (~2e-8 measured); the 1e-5 bound is the
        # established self-consistency bound.
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


def test_bm3dv2_deterministic_multi_stream(noise_gray):
    """Two separate num_streams=4 instances must produce the same output."""
    a = _run(noise_gray, radius=2, num_streams=4)
    b = _run(noise_gray, radius=2, num_streams=4)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


def test_bm3dv2_multi_stream_matches_single(noise_gray):
    """The pipelined num_streams=4 path must produce the same result as the
    serial num_streams=1 path."""
    a = _run(noise_gray, radius=2, num_streams=4)
    b = _run(noise_gray, radius=2, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"num_streams mismatch at frame {n}"


def test_bm3dv2_rejects_gray8(noise_8bit):
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_bm3dv2_rejects_radius5(noise_gray):
    """Radius > 4 is unsupported and must be rejected up front."""
    with pytest.raises(vs.Error):
        _run(noise_gray, radius=5)


def test_bm3dv2_rejects_bad_ref_format(noise_gray):
    """A \"ref\" with a different format/size must be rejected up front."""
    bad = noise_gray.std.AddBorders(right=1)
    with pytest.raises(vs.Error):
        _run(noise_gray, ref=bad)


def test_bm3dv2_ref_final_pass(noise_gray):
    """A basic estimate passed as \"ref\" drives the final (Wiener) pass.

    The final output must be finite on every frame, deterministic between two
    sequential num_streams=1 instances, and differ from the basic estimate
    (the Wiener refinement changes the pixels rather than copying them).
    """
    basic = _run(noise_gray, radius=2, num_streams=1)
    a = _run(noise_gray, radius=2, num_streams=1, ref=basic)
    b = _run(noise_gray, radius=2, num_streams=1, ref=basic)
    for n in (0, 11, 23):
        fa = frame_to_ndarray(a.get_frame(n))
        fb = frame_to_ndarray(b.get_frame(n))
        fb_ = frame_to_ndarray(basic.get_frame(n))
        assert np.isfinite(fa).all(), f"non-finite final output at frame {n}"
        assert np.abs(fa - fb).max() < 1e-5, f"nondeterministic final pass at frame {n}"
        assert np.abs(fa - fb_).max() > 1e-4, f"final pass did not refine frame {n}"


# ---------------------------------------------------------------------------
# Input validation (creation time, before GPU resources are allocated)
# ---------------------------------------------------------------------------

def _blank(w, h):
    return vs.core.std.BlankClip(width=w, height=h, format=vs.GRAYS, length=3)


@pytest.mark.parametrize("w,h", [(1, 1), (4, 4), (7, 8), (8, 7)])
def test_bm3dv2_rejects_dimensions_below_block(w, h):
    """R7: dimensions smaller than the 8x8 block must be rejected at creation.

    Before the fix a 4x4 clip was accepted and dispatched negative block
    coordinates into the patch loads.
    """
    with pytest.raises(vs.Error):
        vs.core.vsfeel.BM3Dv2(
            _blank(w, h), sigma=SIGMA, radius=2, bm_range=BM_RANGE,
            ps_range=PS_RANGE, block_step=BLOCK_STEP, num_streams=1,
        )


def test_bm3dv2_accepts_exactly_8x8():
    """An 8x8 clip is the smallest supported geometry and must run."""
    out = vs.core.vsfeel.BM3Dv2(
        _blank(8, 8), sigma=SIGMA, radius=2, bm_range=BM_RANGE,
        ps_range=PS_RANGE, block_step=BLOCK_STEP, num_streams=1,
    )
    a = frame_to_ndarray(out.get_frame(0))
    assert a.shape == (8, 8)
    assert np.isfinite(a).all()


def test_bm3dv2_device_id(noise_gray):
    """R11: device_id is validated and selects the requested device.

    device_id=-1 and an out-of-range id must fail at creation; device_id=0
    must behave like the default (measured max diff ~2e-8, the same
    atomic-order run-to-run floor as two default instances).
    """
    for bad in (-1, 99):
        with pytest.raises(vs.Error):
            _run(noise_gray, num_streams=1, device_id=bad)
    a = _run(noise_gray, num_streams=1, device_id=0)
    b = _run(noise_gray, num_streams=1)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"device_id=0 differs from default at frame {n}"


def test_bm3dv2_preserves_gray_frame_props(noise_gray):
    """R13: the grayscale path must keep the source frame's properties.

    ``_ColorRange`` is deprecated in VapourSynth R79 and is remapped by
    ``SetFrameProps``; the surviving equivalent key ``_Range`` is tagged
    instead. ``MyTag`` verifies arbitrary application metadata.
    """
    tagged = vs.core.std.SetFrameProps(
        noise_gray, _DurationNum=1001, _DurationDen=24000, _Range=0, MyTag=7,
    )
    out = _run(tagged, radius=2, num_streams=1)
    for n in (0, 11, 23):
        props = out.get_frame(n).props
        assert props["_DurationNum"] == 1001, f"_DurationNum lost at frame {n}"
        assert props["_DurationDen"] == 24000, f"_DurationDen lost at frame {n}"
        assert int(props["_Range"]) == 0, f"_Range lost at frame {n}"
        assert props["MyTag"] == 7, f"custom metadata lost at frame {n}"


def test_bm3dv2_yuv_passthrough(noise_gray):
    """A full YUV clip is accepted: the luma must match the Gray output and
    the chroma planes must be copied through bit-identically."""
    src = vs.core.bs.VideoSource(NOISE_MKV)
    yuv = vs.core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    assert yuv.format.color_family == vs.YUV

    out = _run(yuv, radius=2, num_streams=1)
    ref = _run(noise_gray, radius=2, num_streams=1)

    for n in (0, 11, 23):
        f = out.get_frame(n)
        # luma is denoised identically to the Gray path
        d = frame_to_ndarray(f) - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"luma mismatch at frame {n}"
        # chroma is passed through unchanged (stride-aware copy)
        s = yuv.get_frame(n)
        for plane in (1, 2):
            a = plane_to_ndarray(f, plane)
            b = plane_to_ndarray(s, plane)
            assert np.array_equal(a, b), f"chroma{plane} changed at frame {n}"


# ---------------------------------------------------------------------------
# Seek schedule (R5)
# ---------------------------------------------------------------------------

# The modulo-64 writer table used to let a seek schedule mix up which stream
# owned a cached source slot. 0/64/1/65/2 is the minimal collision: frame 64
# overwrites lookup entry 0 while frame 1 still shares frame 0's cache entry.
# The run happens in a subprocess with a timeout so a deadlock fails the test
# instead of hanging the whole suite.
_SEEK_SCRIPT = COMPARE_PRELUDE + textwrap.dedent(f"""\
    import json
    import sys
    import threading
    import vapoursynth as vs
    from vstools import core

    core.max_cache_size = 512
    src = core.bs.VideoSource({NOISE_MKV!r})
    gray = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                              bits=32, fulls=True, fulld=True)
    clip = core.std.Loop(gray, times=4)
    kwargs = dict(sigma={SIGMA}, radius=2, bm_range={BM_RANGE},
                  ps_range={PS_RANGE}, block_step={BLOCK_STEP})

    order = [0, 64, 1, 65, 2, 13, 79, 40, 95, 3]

    # Serial num_streams=1 run is the self-consistency oracle.
    try:
        base = core.vsfeel.BM3Dv2(clip, num_streams=1, **kwargs)
        ref = {{n: read_plane(base.get_frame(n), 0, np.float32) for n in order}}
    except Exception as exc:
        print("VSFEEL fail: serial run: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(3)
    print("REF ok", flush=True)

    out = core.vsfeel.BM3Dv2(clip, num_streams=4, **kwargs)
    got = {{}}
    errors = {{}}

    def worker(n):
        try:
            got[n] = read_plane(out.get_frame(n), 0, np.float32)
        except BaseException as exc:
            errors[n] = "%s: %s" % (type(exc).__name__, exc)

    threads = []
    for n in order:
        t = threading.Thread(target=worker, args=(n,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if errors:
        print("VSFEEL fail: " + json.dumps(errors), flush=True)
        raise SystemExit(3)
    worst = 0.0
    for n in order:
        a, b = got[n], ref[n]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print("VSFEEL fail: non-finite at frame %d" % n, flush=True)
            raise SystemExit(3)
        worst = max(worst, float(np.abs(a - b).max()))
    print("RESULT " + json.dumps(worst), flush=True)
""")


def test_bm3dv2_seek_collision_self_consistent():
    """R5: the 0/64/1/65/2 schedule on a 96-frame clip must not deadlock and
    must match a num_streams=1 run (measured max diff ~2e-8)."""
    try:
        worst = run_compare_subprocess(_SEEK_SCRIPT, [], timeout=300)
    except ReferenceUnavailable as exc:
        # There is no external reference here: anything that dies before the
        # serial run completes is a vsfeel failure, not a missing reference.
        raise AssertionError(f"seek-schedule subprocess failed early: {exc}") from exc
    assert worst < 1e-5, f"num_streams=4 seek schedule mismatch: {worst}"


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------

_COMPARE_SCRIPT = COMPARE_PRELUDE + textwrap.dedent(f"""\
    import json
    import sys
    import vapoursynth as vs
    from vstools import core

    ref = sys.argv[1]
    kwargs = json.loads(sys.argv[2])
    ref_pass = int(sys.argv[3])

    core.max_cache_size = 1024 * 56
    src = core.bs.VideoSource({NOISE_MKV!r})
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)

    if ref != "bm3dhip":
        kwargs["num_streams"] = 1

    frames = (0, 11, 23)

    basic = None
    if ref_pass:
        # The reference consumes vsfeel's basic estimate as its guide. Force
        # that clip here so a vsfeel failure is not mislabelled as a missing
        # reference; the frames stay cached for the reference below.
        try:
            basic = core.vsfeel.BM3Dv2(clip, **kwargs)
            _ = [read_plane(basic.get_frame(n), 0, np.float32) for n in frames]
        except Exception as exc:
            print("VSFEEL fail: basic estimate: %s: %s" % (type(exc).__name__, exc), flush=True)
            raise SystemExit(3)

    # --- reference phase: materialise and copy before touching vsfeel's
    #     graded node ---
    try:
        if ref_pass:
            ref_node = getattr(core, ref).BM3Dv2(clip, ref=basic, **kwargs)
        else:
            ref_node = getattr(core, ref).BM3Dv2(clip, **kwargs)
        ref_frames = [read_plane(ref_node.get_frame(n), 0, np.float32) for n in frames]
    except Exception as exc:
        print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(2)
    print("REF ok", flush=True)

    # --- vsfeel phase ---
    my_node = (core.vsfeel.BM3Dv2(clip, ref=basic, **kwargs) if ref_pass
               else core.vsfeel.BM3Dv2(clip, **kwargs))
    worst = 0.0
    for n, b in zip(frames, ref_frames):
        a = read_plane(my_node.get_frame(n), 0, np.float32)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print("VSFEEL fail: non-finite at frame %d" % n, flush=True)
            raise SystemExit(3)
        worst = max(worst, float(np.abs(a - b).max()))
    print("RESULT " + json.dumps(worst), flush=True)
""")

BASE_KWARGS = dict(sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4)


def _max_diff_vs_reference(kwargs: dict, ref: str, ref_pass: bool = False,
                           timeout: float = 600) -> float:
    """Run the comparison in a subprocess.

    Raises :class:`ReferenceUnavailable` when ``ref`` is missing/failed and
    :class:`AssertionError` (with the captured tail) when vsfeel failed.
    """
    return run_compare_subprocess(
        _COMPARE_SCRIPT, [ref, json.dumps(kwargs), str(int(ref_pass))],
        timeout=timeout,
    )


def _compare_against_any_reference(kwargs: dict, ref_pass: bool = False) -> tuple[str, float]:
    """Try vszipcl then bm3dhip; skip only if none is usable."""
    reasons = []
    for ref in ("vszipcl", "bm3dhip"):
        if not hasattr(vs.core, ref) or not hasattr(getattr(vs.core, ref), "BM3Dv2"):
            reasons.append(f"{ref}: not installed")
            continue
        try:
            return ref, _max_diff_vs_reference(kwargs, ref, ref_pass=ref_pass)
        except ReferenceUnavailable as exc:
            reasons.append(f"{ref}: {exc}")
    pytest.skip("no usable reference plugin (vszipcl/bm3dhip): " + "; ".join(reasons))


def test_bm3dv2_ref_matches_reference(noise_gray):
    """The final (Wiener) pass must closely match the reference implementations
    when given the same basic-estimate ref clip."""
    ref, maxdiff = _compare_against_any_reference(dict(BASE_KWARGS), ref_pass=True)
    assert maxdiff < 0.01, f"max diff vs {ref} (ref pass): {maxdiff}"


# Parameter sweep around the defaults: every entry is merged over
# BASE_KWARGS and compared against the reference (basic estimate).
#
# Tolerances are set from measurement. The basic estimate is sensitive to
# block-match decisions made near the bm_range threshold: a microscopic fp
# difference in distance accumulation can flip a block in or out of its
# group, which shows up as isolated speckle (<0.1% of pixels, spread over
# the whole frame). Larger sigma / more grouped blocks make this more
# likely, hence the looser bounds there. Remeasured with the repaired
# stride-aware/pinning oracle (vszipcl, basic estimate, frames 0/11/23):
# defaults 0.00786, sigma=0.3 0.00256, sigma=1.5 0.02025, block_step=2
# 0.00648, bm_range=9 0.00866, bm_range=22 0.00873, ps_range=5 0.00793,
# ps_range=9 0.00732, ps_num=5 0.01226, extractor_exp=6 0.00806.
SWEEP_CONFIGS = [
    ({}, 0.01),
    ({"sigma": 0.3}, 0.01),
    ({"sigma": 1.5}, 0.03),
    ({"block_step": 2}, 0.01),
    ({"bm_range": 9}, 0.01),
    ({"bm_range": 22}, 0.01),
    ({"ps_range": 5}, 0.01),
    ({"ps_range": 9}, 0.01),
    ({"ps_num": 5}, 0.02),
    ({"extractor_exp": 6}, 0.01),
]


def _sweep_id(cfg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in cfg.items()) or "defaults"


@pytest.mark.parametrize("cfg,tol", SWEEP_CONFIGS, ids=[_sweep_id(c) for c, _ in SWEEP_CONFIGS])
def test_bm3dv2_parameter_sweep_matches_reference(noise_gray, cfg, tol):
    """Parameter grid around the defaults must track the reference (basic
    estimate pass)."""
    ref, maxdiff = _compare_against_any_reference(dict(BASE_KWARGS, **cfg))
    assert maxdiff < tol, f"max diff vs {ref} ({_sweep_id(cfg)}): {maxdiff}"


# Remeasured with the repaired oracle (vszipcl, basic estimate, radius sweep):
# radius 0/2/3/4 = 0.00708/0.00786/0.00285/0.00300; the Wiener ref pass =
# 0.00273. The 0.01 bound covers all of them with margin.
@pytest.mark.parametrize("radius", [0, 2, 3, 4])
def test_bm3dv2_matches_reference(noise_gray, radius):
    """vsfeel must closely match vszipcl, falling back to bm3dhip.

    The reference plugins are crash-prone on some drivers, so the comparison
    runs in a subprocess: a crashed reference must not take down the suite.
    bm3dhip implements the same algorithm and is expected to match vszipcl.
    """
    ref, maxdiff = _compare_against_any_reference(dict(BASE_KWARGS, radius=radius))
    assert maxdiff < 0.01, f"max diff vs {ref}: {maxdiff}"
