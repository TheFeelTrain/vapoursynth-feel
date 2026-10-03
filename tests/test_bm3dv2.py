"""Unit tests for core.vsfeel.BM3Dv2.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is used as the input. The noise content exposes the boundary/clamping behaviour of the
temporal pipeline: frames 0..23 must all produce finite output with no NaN
(regression: boundary frames intermittently produced NaN before the atomics
were made visible to the aggregation kernel).

Run from the repository root:  uv run python -m pytest tests/test_bm3dv2.py
"""

import ctypes
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    CLIP_PATH,
    COMPARE_PRELUDE,
    ReferenceUnavailable,
    assert_all_frames_finite,
    assert_preserves_frame_props,
    assert_temporal_order_consistent,
    check_all_frames_finite,
    eval_parallel,
    frame_to_ndarray,
    plane_to_ndarray,
    run_compare_subprocess,
    skip_or_fail_reference,
    cpu_node,
    source_clip,
)


_CPU_AVAILABLE = hasattr(vs.core, "bm3d") and hasattr(vs.core.bm3d, "VBasic")


def BM3D(*args, **kwargs):
    """BM3Dv2 as a clip the test can read pixels from (see conftest.cpu_node)."""
    return cpu_node(vs.core.vsfeel.BM3Dv2(*args, **kwargs))


pytestmark = pytest.mark.usefixtures("clip_gray")

SIGMA = 0.7
BM_RANGE = 16
PS_RANGE = 7
BLOCK_STEP = 4


def _run(clip, radius=2, **kwargs):
    return BM3D(
        clip,
        sigma=SIGMA,
        radius=radius,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        **kwargs,
    )


def test_bm3dv2_parallel_load_matches_serial(clip_gray):
    """Parallel request load must produce the same pixel
    values as the serial path.

    This is the request pattern that exposed stale descriptor bindings, fence
    misuse and command pool reuse violations under load on strict drivers
    (black or garbage output only when frames are processed concurrently).
    """
    par = eval_parallel(_run, clip_gray, radius=2)
    ref = _run(clip_gray, radius=2)
    for n in range(clip_gray.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"parallel/serial mismatch at frame {n}"


def test_bm3dv2_wide_radius_parallel_load_matches_serial(clip_gray):
    """The widest window must survive the deep pipeline like the narrow ones.

    At radius 16 one output frame spans 33 estimate slots and 65 source
    frames, i.e. the instance's whole cache working set at once; a parallel
    request load is what exercises that ring under contention.
    """
    par = eval_parallel(_run, clip_gray, radius=16)
    ref = _run(clip_gray, radius=16)
    for n in range(clip_gray.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"parallel/serial mismatch at frame {n}"


def test_bm3dv2_parallel_load_deterministic(clip_gray):
    """Two parallel runs must produce identical output.

    A fence attached to two in-flight submissions makes frame results depend
    on the completion order of the streams, which only shows up when many
    frames are in flight at once.
    """
    a = eval_parallel(_run, clip_gray, radius=2)
    b = eval_parallel(_run, clip_gray, radius=2)
    for n in range(clip_gray.num_frames):
        d = a[n] - b[n]
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 16])
def test_bm3dv2_no_nan_all_frames(clip_gray, radius):
    """Output must be finite on every frame (incl. boundaries) for each radius.

    A brand new filter instance is created per parametrized test.
    """
    check_all_frames_finite(_run, clip_gray, radius=radius)


def test_bm3dv2_deterministic(clip_gray):
    a = _run(clip_gray, radius=2)
    b = _run(clip_gray, radius=2)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        # BM3D's aggregation uses float atomics, so two runs differ only by
        # atomic-order rounding (~2e-8 measured); the 1e-5 bound is the
        # established self-consistency bound.
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


def test_bm3dv2_cas_fallback_matches_hardware_atomics(clip_gray, monkeypatch):
    """The CAS arm a device without float32 add atomics gets must match.

    `VSFEEL_BM3D_CAS=1` forces the `-DNO_FLOAT_ATOMICS` kernel the host selects
    when the device has no `shaderBufferFloat32AtomicAdd`; the two arms add in
    a different order, so this is also what bounds that order's effect.
    """
    monkeypatch.delenv("VSFEEL_BM3D_CAS", raising=False)
    hardware = _run(clip_gray, radius=2)
    monkeypatch.setenv("VSFEEL_BM3D_CAS", "1")
    cas = _run(clip_gray, radius=2)
    for n in (0, 11, 23):
        a = frame_to_ndarray(hardware.get_frame(n))
        b = frame_to_ndarray(cas.get_frame(n))
        assert np.isfinite(b).all(), f"non-finite CAS output at frame {n}"
        # same accumulation-order floor as the nondeterminism test above:
        # the arm against itself measures 2.4e-7..3.6e-7 on real content, and
        # cross-arm is 3.0e-7 (frames 0/11/23, |max| ~ 1)
        assert np.abs(a - b).max() < 1e-5, f"CAS vs hardware atomics at frame {n}"


def test_bm3dv2_cas_fallback_holds_at_small_block_step(clip_gray, monkeypatch):
    """The CAS arm must stay exact where contention is highest.

    One res element receives up to `8 * (ceil((2*bm_range + 8)/block_step) +
    1)^2` adds -- every matched patch of every reference block whose search
    window can reach it -- so block_step=1 is the worst case (13448 at
    bm_range=16) and a retry budget below it can silently drop addends: the
    old 32-retry bound put 16 of 4096 pixels 1e-5..6.1e-5 out against the
    hardware arm, while the geometry-derived bound lands at float add order
    alone. The bound is what this pins.
    """

    def small_step(clip):
        return BM3D(clip, sigma=SIGMA, radius=2, bm_range=BM_RANGE, ps_range=PS_RANGE, block_step=1)

    monkeypatch.setenv("VSFEEL_BM3D_CAS", "1")
    cas = small_step(clip_gray)
    monkeypatch.delenv("VSFEEL_BM3D_CAS", raising=False)
    hardware = small_step(clip_gray)
    # Both arms add the same 13448 addends in a scheduler-dependent order, so
    # the comparison floor is the arms' own spread, not exactness: measured
    # 4.2e-7 (hardware vs itself) and 5.4e-7 (cross-arm) at |max| ~ 1, frames
    # 0/11. A dropped addend moves one element by 1/13448 of an addend, ~7e-5
    # of the result, so 1e-5 still separates the two (the two runs must be
    # scaled, since the same order noise was 7e-9 on the old near-black clip).
    for n in (0, 11):
        a = frame_to_ndarray(hardware.get_frame(n))
        b = frame_to_ndarray(cas.get_frame(n))
        assert np.isfinite(b).all(), f"non-finite CAS output at frame {n}"
        scale = max(1.0, float(np.abs(a).max()))
        worst = float(np.abs(a - b).max()) / scale
        assert worst < 1e-5, (
            f"CAS lost an addend at block_step=1, frame {n}: {worst:g} of full scale"
        )


def test_bm3dv2_disjoint_first_estimates_keep_slice_witnesses(clip_gray):
    """Concurrent opposite-end windows must not clear each other's tags."""
    import threading

    clip = clip_gray.std.CropAbs(width=64, height=64)
    kwargs = dict(sigma=SIGMA, radius=2, bm_range=2, ps_range=1, block_step=4)
    reference = BM3D(clip, **kwargs)
    expected = {n: frame_to_ndarray(reference.get_frame(n)) for n in (0, 23)}

    for _ in range(3):
        out = BM3D(clip, **kwargs)
        gate = threading.Barrier(2)
        got, errors = {}, []

        def worker(n):
            try:
                gate.wait()
                got[n] = frame_to_ndarray(out.get_frame(n))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in (0, 23)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, f"concurrent first estimates failed: {errors}"
        for n in (0, 23):
            assert np.abs(got[n] - expected[n]).max() < 1e-5, (
                f"first-use tag clear corrupted frame {n}"
            )


def test_bm3dv2_radius0_concurrent_first_use_and_fallback(clip_gray, monkeypatch):
    """Concurrent first requests must preserve private tags and fallback input."""
    monkeypatch.setenv("VSFEEL_BM3D_NOESTIMATE", "1")
    actual = eval_parallel(_run, clip_gray, radius=0)
    for n in range(clip_gray.num_frames):
        expected = frame_to_ndarray(clip_gray.get_frame(n))
        assert np.array_equal(actual[n], expected), (
            f"radius-zero fallback read the wrong source slot at frame {n}"
        )


# ---------------------------------------------------------------------------
# Error-path cache handoff and the radius-0 trace
# ---------------------------------------------------------------------------

# Both scripts run the plugin in a subprocess (the trace the filter prints goes
# to stderr, and the error-path test must not hang the suite) and reuse
# conftest's stride-aware readers.
_SUBPROCESS_PRELUDE = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
from conftest import CLIP_PATH, frame_to_ndarray
import numpy as np
import vapoursynth as vs
from vapoursynth import core


def source():
    if hasattr(core, "bs"):
        return core.bs.VideoSource(CLIP_PATH)
    return core.ffms2.Source(CLIP_PATH)


def gray32():
    return core.fmtc.bitdepth(core.std.ShufflePlanes(source(), 0, vs.GRAY),
                              bits=32, fulls=True, fulld=True)
"""

_TRACE_SCRIPT = (
    _SUBPROCESS_PRELUDE
    + r"""
node = core.vsfeel.BM3Dv2(gray32(), sigma=0.7, radius=int(sys.argv[2]),
                          bm_range=16, ps_range=7, block_step=4)
for n in range(4):
    node.get_frame(n)
print("TRACE OK")
"""
)

_FAULT_SCRIPT = (
    _SUBPROCESS_PRELUDE
    + r"""
import threading


def build():
    return core.std.GPUDownload(clip=core.vsfeel.BM3Dv2(
        gray32(), sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4))


oracle = build()
report = {}

# Sequential: frame 0's ring copies never land, then frame 1 needs the same
# source window its keys advertise. It must fail, not denoise against the stale
# slot contents.
node = build()
try:
    node.get_frame(0)
    report["seq0"] = "ok"
except Exception as exc:
    report["seq0"] = type(exc).__name__
try:
    frame = node.get_frame(1)
    report["seq1"] = "ok"
    report["seq1_maxdiff"] = float(np.abs(
        frame_to_ndarray(frame) - frame_to_ndarray(oracle.get_frame(1))).max())
except Exception as exc:
    report["seq1"] = type(exc).__name__

# Concurrent: one thread can reserve the other's slot as a reader before the
# writer's fault lands; the wait must be released, so the run cannot hang.
node = build()
result, got = {}, {}
gate = threading.Barrier(2)


def worker(n):
    try:
        gate.wait()
        got[n] = node.get_frame(n)
        result[n] = "ok"
    except Exception as exc:
        result[n] = type(exc).__name__


threads = [threading.Thread(target=worker, args=(n,)) for n in (0, 1)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
report["con0"], report["con1"] = result.get(0, "missing"), result.get(1, "missing")
if result.get(1) == "ok":
    report["con1_maxdiff"] = float(np.abs(
        frame_to_ndarray(got[1]) - frame_to_ndarray(oracle.get_frame(1))).max())
print("RESULT " + json.dumps(report))
"""
)


def _result_payload(stdout):
    payload = None
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT ") :])
    assert payload is not None, stdout[-2000:]
    return payload


def _run_subprocess_script(script, *argv, env=None):
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    run_env = {**os.environ, "MANGOHUD": "0", **(env or {})}
    return subprocess.run(
        [sys.executable, "-c", script, tests_dir, *map(str, argv)],
        capture_output=True,
        text=True,
        timeout=600,
        env=run_env,
    )


def test_bm3dv2_radius0_trace_prints_no_false_invariant():
    """A radius-0 trace must not claim the window tables hold frame -1.

    Radius 0 returns before the window-cache phase, so `res_frame`/`res_ready`
    keep their creation values; the invariant block used to print "holds frame
    -1" and "NOT READY" for every radius-0 frame, and read those tables without
    the cache lock the acquire/publish/wait path takes.
    """
    proc = _run_subprocess_script(_TRACE_SCRIPT, 0, env={"VSFEEL_BM3D_TRACE": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "TRACE OK" in proc.stdout, proc.stdout[-2000:]
    assert "NOT READY" not in proc.stderr, proc.stderr[-2000:]
    assert "holds frame -1" not in proc.stderr, proc.stderr[-2000:]
    # The trace really ran (otherwise the two absences above are vacuous).
    assert "submitted" in proc.stderr, proc.stderr[-2000:]


def test_bm3dv2_failed_estimation_never_publishes_uncopied_sources():
    """A frame whose ring copy never landed must not leave the source keys set.

    `VSFEEL_BM3D_FAULT=0` fails frame 0 after `acquire_cache` committed its
    source keys but before any chunk-0 command buffer, so the copies it
    promised never happened. A later frame keyed to the same source frame must
    not block-match against the slot's previous contents or uninitialised VRAM:
    it has to fail, and a concurrent reader that already reserved the slot must
    be released rather than block forever on a ready flag that never comes.
    """
    proc = _run_subprocess_script(_FAULT_SCRIPT, env={"VSFEEL_BM3D_FAULT": "0"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = _result_payload(proc.stdout)
    assert payload["seq0"] != "ok", payload
    assert payload["seq1"] != "ok", (
        f"a later frame denoised against an uncopied source slot: {payload}"
    )
    assert payload["con0"] != "ok", payload
    # A concurrent frame that got its own copies in first may finish correctly;
    # it must never finish on the failed frame's slot.
    assert payload["con1"] != "ok" or payload["con1_maxdiff"] < 1e-5, payload


def test_bm3dv2_nosearch_matches_search_on_constant_clip(monkeypatch):
    """The no-search arm must initialise the shared match tables, or the
    aggregation indexes stale LDS.

    Before the fix a constant clip came out 99.5% NaN; it now matches the
    searched run to one ulp.
    """
    clip = vs.core.std.BlankClip(width=64, height=64, format=vs.GRAYS, length=3, color=0.5)
    monkeypatch.delenv("VSFEEL_BM3D_NOSEARCH", raising=False)
    search = _run(clip, radius=2)
    monkeypatch.setenv("VSFEEL_BM3D_NOSEARCH", "1")
    nosearch = _run(clip, radius=2)
    for n in range(3):
        a = frame_to_ndarray(search.get_frame(n))
        b = frame_to_ndarray(nosearch.get_frame(n))
        assert np.isfinite(b).all(), f"non-finite no-search output at frame {n}"
        # measured 2.98e-8 (one ulp) between the two arms on this input
        assert np.abs(a - b).max() < 1e-6, f"no-search vs search at frame {n}"
        assert np.abs(b - 0.5).max() < 1e-6, f"no-search left the constant at frame {n}"


def test_bm3dv2_rejects_gray8(clip_8bit):
    with pytest.raises(vs.Error):
        _run(clip_8bit)


def test_bm3dv2_rejects_radius17(clip_gray):
    """Radius > 16 is unsupported and must be rejected up front."""
    with pytest.raises(vs.Error):
        _run(clip_gray, radius=17)


@pytest.mark.parametrize("radius", [15, 16])
def test_bm3dv2_accepts_reference_radius_cap(clip_gray, radius):
    """The radius cap must match the references' (vszipcl 16, bm3dvk 15).

    Windows this wide are the end-to-end case for the aggregation's derived
    per-slice table, which replaced the push-constant one.
    """
    out = _run(clip_gray, radius=radius)
    for n in (0, 23):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


def test_bm3dv2_rejects_bad_ref_format(clip_gray):
    """A \"ref\" with a different format/size must be rejected up front."""
    bad = clip_gray.std.AddBorders(right=1)
    with pytest.raises(vs.Error):
        _run(clip_gray, ref=bad)


def test_bm3dv2_ref_final_pass(clip_gray):
    """A basic estimate passed as \"ref\" drives the final (Wiener) pass.

    The final output must be finite on every frame, deterministic between two
    sequential instances, and differ from the basic estimate
    (the Wiener refinement changes the pixels rather than copying them).
    """
    basic = _run(clip_gray, radius=2)
    a = _run(clip_gray, radius=2, ref=basic)
    b = _run(clip_gray, radius=2, ref=basic)
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
        BM3D(
            _blank(w, h),
            sigma=SIGMA,
            radius=2,
            bm_range=BM_RANGE,
            ps_range=PS_RANGE,
            block_step=BLOCK_STEP,
        )


def test_bm3dv2_accepts_exactly_8x8():
    """An 8x8 clip is the smallest supported geometry and must run."""
    out = BM3D(
        _blank(8, 8),
        sigma=SIGMA,
        radius=2,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
    )
    a = frame_to_ndarray(out.get_frame(0))
    assert a.shape == (8, 8)
    assert np.isfinite(a).all()


def test_bm3dv2_rejects_int32_res_overflow():
    """A stack above 2^31 floats must be rejected at creation.

    The kernel addresses `res` through signed 32-bit offsets, so a 4K radius-4
    stack wrapped negatively. 8K keeps the overflow true under both estimate
    cache sizings (7.2e9 floats at the minimum working set, 1.25e10 with the
    default slack), unlike the 4K/ns=4 config whose overflow depended on the
    slack being allocated.
    """
    with pytest.raises(vs.Error, match="32-bit"):
        BM3D(
            _blank(7680, 4320),
            sigma=SIGMA,
            radius=4,
            bm_range=BM_RANGE,
            ps_range=PS_RANGE,
            block_step=BLOCK_STEP,
        )


def test_bm3dv2_accepts_radius4_within_addressing_limit():
    """The guard must not reject radius 4 when the stack stays addressable."""
    out = BM3D(
        _blank(8, 8),
        sigma=SIGMA,
        radius=4,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
    )
    assert out.num_frames == 3


def test_bm3dv2_device_id(clip_gray):
    """device_id and num_streams are accepted no-ops.

    Under the R80 GPU API the core owns the one Vulkan device per process and
    sizes in-flight depth itself, so both arguments are registered only so
    existing scripts keep loading. Any value, including a negative one, must be
    ignored and produce the default result (measured max diff ~2e-8, the
    atomic-order run-to-run floor).
    """
    b = _run(clip_gray)
    for kwargs in (
        dict(device_id=0),
        dict(device_id=-1),
        dict(device_id=99),
        dict(num_streams=0),
        dict(num_streams=64),
    ):
        a = _run(clip_gray, **kwargs)
        for n in (0, 11, 23):
            d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
            assert np.abs(d).max() < 1e-5, f"{kwargs} differs from default at frame {n}"


def test_bm3dv2_preserves_gray_frame_props(clip_gray):
    """R13: the grayscale path must keep the source frame's properties.

    ``_ColorRange`` is deprecated in VapourSynth R79 and is remapped by
    ``SetFrameProps``; the surviving equivalent key ``_Range`` is tagged
    instead. ``MyTag`` verifies arbitrary application metadata.
    """
    assert_preserves_frame_props(_run, clip_gray, radius=2)


def test_bm3dv2_chroma_planes_are_denoised(clip_gray):
    """A YUV clip is denoised on every plane, luma exactly as the Gray path.

    The references denoise each plane independently by default (per-plane
    geometry, parameters and sigma), so a plane left equal to the source is the
    old luma-only passthrough, not a denoised result.
    """
    yuv = vs.core.fmtc.bitdepth(source_clip(), bits=32, fulls=True, fulld=True)
    assert yuv.format.color_family == vs.YUV

    out = _run(yuv, radius=2)
    ref = _run(clip_gray, radius=2)

    for n in (0, 11, 23):
        f = out.get_frame(n)
        # luma is denoised identically to the Gray path (atomic-order floor)
        d = frame_to_ndarray(f) - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"luma mismatch at frame {n}"
        # every plane moved away from the source
        s = yuv.get_frame(n)
        for plane in range(yuv.format.num_planes):
            a = plane_to_ndarray(f, plane)
            b = plane_to_ndarray(s, plane)
            assert np.isfinite(a).all(), f"non-finite plane {plane} at frame {n}"
            assert float(np.abs(a - b).max()) > 1e-4, (
                f"plane {plane} was left at the source value at frame {n}"
            )


@pytest.mark.parametrize("sigma", [[0.7, 0.0, 0.0], [0.0, 0.7, 0.0]])
def test_bm3dv2_unprocessed_plane_is_source_copy(sigma):
    """A plane whose sigma is below FLT_EPSILON is a bit-exact source copy,
    while the others are still denoised (the reference's PROC_MASK)."""
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(CLIP_PATH), bits=32, fulls=True, fulld=True)
    out = BM3D(
        yuv, sigma=sigma, radius=2, bm_range=BM_RANGE, ps_range=PS_RANGE, block_step=BLOCK_STEP
    )
    for n in (0, 11, 23):
        f = out.get_frame(n)
        s = yuv.get_frame(n)
        for plane in range(yuv.format.num_planes):
            a = plane_to_ndarray(f, plane)
            b = plane_to_ndarray(s, plane)
            assert np.isfinite(a).all(), f"non-finite plane {plane} at frame {n}"
            if sigma[plane] == 0.0:
                assert np.array_equal(a, b), f"sigma=0 plane {plane} changed at frame {n}"
            else:
                assert float(np.abs(a - b).max()) > 1e-4, (
                    f"sigma={sigma[plane]} plane {plane} was not denoised at frame {n}"
                )


def test_bm3dv2_all_planes_below_epsilon_returns_source():
    """Every plane below FLT_EPSILON hands the source clip back unchanged.

    Both references' BM3Dv2 take this shortcut instead of building a filter, so
    the result is bit-exact. The core auto-uploads a CPU clip for the plugin's
    ``vnode:gpu`` argument, so the node it hands back is GPU resident like any
    other BM3Dv2 output.
    """
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(CLIP_PATH), bits=32, fulls=True, fulld=True)
    out = vs.core.vsfeel.BM3Dv2(yuv, sigma=[0.0], radius=2)
    assert out.gpu_resident, "BM3Dv2 output must stay GPU resident"
    node = cpu_node(out)
    for n in (0, 11, 23):
        a, b = node.get_frame(n), yuv.get_frame(n)
        for plane in range(yuv.format.num_planes):
            assert np.array_equal(plane_to_ndarray(a, plane), plane_to_ndarray(b, plane)), (
                f"plane {plane} changed at frame {n}"
            )


def test_bm3dv2_per_plane_parameters_are_honoured():
    """Each plane runs with its own sigma/block_step/bm_range/ps_num/ps_range.

    Two runs that differ only in plane 1's parameters must differ on plane 1
    and agree elsewhere: a filter that ignored the per-plane arrays would
    produce identical frames.
    """
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(CLIP_PATH), bits=32, fulls=True, fulld=True)
    base = dict(
        sigma=[0.7, 0.7, 0.7],
        radius=2,
        bm_range=[16, 16, 16],
        ps_range=[7, 7, 7],
        block_step=[4, 4, 4],
        ps_num=[2, 2, 2],
    )
    other = dict(
        base, sigma=[0.7, 0.3, 0.7], block_step=[4, 8, 4], bm_range=[16, 4, 16], ps_range=[7, 2, 7]
    )
    a = BM3D(yuv, **base)
    b = BM3D(yuv, **other)
    for n in (0, 11, 23):
        for plane in range(3):
            d = float(
                np.abs(
                    plane_to_ndarray(a.get_frame(n), plane)
                    - plane_to_ndarray(b.get_frame(n), plane)
                ).max()
            )
            if plane == 1:
                assert d > 1e-4, f"plane 1 parameters were ignored at frame {n}"
            else:
                assert d < 1e-5, f"plane {plane} changed with plane 1's parameters"


def test_bm3dv2_chroma_requires_yuv444():
    """chroma=True is the reference's joint entry and needs 4:4:4 input."""
    yuv420 = vs.core.fmtc.bitdepth(
        vs.core.bs.VideoSource(CLIP_PATH), bits=32, fulls=True, fulld=True
    )
    with pytest.raises(vs.Error, match="YUV444"):
        vs.core.vsfeel.BM3Dv2(yuv420, sigma=0.7, chroma=1)
    rgb = vs.core.resize.Bicubic(
        vs.core.bs.VideoSource(CLIP_PATH), format=vs.RGBS, matrix_in_s="709"
    )
    with pytest.raises(vs.Error, match="YUV444"):
        vs.core.vsfeel.BM3Dv2(rgb, sigma=0.7, chroma=1)


def test_bm3dv2_joint_chroma_denoises_every_plane():
    """chroma=True packs the three 4:4:4 planes into one entry; a sigma-zero
    plane of that entry is skipped and comes from the source."""
    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(CLIP_PATH), format=vs.YUV444PS)
    out = BM3D(
        yuv444,
        sigma=[0.7, 0.7, 0.7],
        radius=2,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        chroma=1,
    )
    for n in (0, 11, 23):
        f, s = out.get_frame(n), yuv444.get_frame(n)
        for plane in range(3):
            a = plane_to_ndarray(f, plane)
            assert np.isfinite(a).all(), f"non-finite plane {plane} at frame {n}"
            assert float(np.abs(a - plane_to_ndarray(s, plane)).max()) > 1e-4, (
                f"joint plane {plane} was not denoised at frame {n}"
            )

    mixed = BM3D(
        yuv444,
        sigma=[0.7, 0.0, 0.7],
        radius=2,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        chroma=1,
    )
    for n in (0, 11):
        f, s = mixed.get_frame(n), yuv444.get_frame(n)
        assert np.array_equal(plane_to_ndarray(f, 1), plane_to_ndarray(s, 1)), (
            f"joint sigma=0 plane 1 changed at frame {n}"
        )
        assert float(np.abs(plane_to_ndarray(f, 0) - plane_to_ndarray(s, 0)).max()) > 1e-4


def test_bm3dv2_joint_chroma_shares_luma_groups():
    """Joint mode is not the per-plane mode: its chroma planes are filtered
    with the groups block matching found on luma, so the two differ."""
    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(CLIP_PATH), format=vs.YUV444PS)
    kw = dict(
        sigma=[0.7, 0.7, 0.7], radius=2, bm_range=BM_RANGE, ps_range=PS_RANGE, block_step=BLOCK_STEP
    )
    separate = BM3D(yuv444, **kw)
    joint = BM3D(yuv444, chroma=1, **kw)
    for n in (0, 11, 23):
        for plane in (1, 2):
            d = float(
                np.abs(
                    plane_to_ndarray(joint.get_frame(n), plane)
                    - plane_to_ndarray(separate.get_frame(n), plane)
                ).max()
            )
            assert d > 1e-4, f"plane {plane} is identical in joint and per-plane mode"
        d0 = float(
            np.abs(
                plane_to_ndarray(joint.get_frame(n), 0) - plane_to_ndarray(separate.get_frame(n), 0)
            ).max()
        )
        assert d0 < 1e-5, f"joint mode changed luma at frame {n}"


def test_bm3dv2_rgb_is_denoised():
    """RGB input runs all three planes like the references do."""
    rgb = vs.core.resize.Bicubic(
        vs.core.bs.VideoSource(CLIP_PATH), format=vs.RGBS, matrix_in_s="709"
    )
    out = BM3D(
        rgb, sigma=0.7, radius=2, bm_range=BM_RANGE, ps_range=PS_RANGE, block_step=BLOCK_STEP
    )
    for n in (0, 11, 23):
        f, s = out.get_frame(n), rgb.get_frame(n)
        for plane in range(3):
            a = plane_to_ndarray(f, plane)
            assert np.isfinite(a).all(), f"non-finite RGB plane {plane} at frame {n}"
            assert float(np.abs(a - plane_to_ndarray(s, plane)).max()) > 1e-4, (
                f"RGB plane {plane} was not denoised at frame {n}"
            )


def test_bm3dv2_rejects_subsampled_plane_below_block():
    """A processed chroma plane smaller than the 8x8 block must be refused.

    The kernel clamps block origins to (dimension - 8), so an 8-row YUV420 clip
    (4-row chroma) would address before its plane's buffer.
    """
    yuv420 = vs.core.fmtc.bitdepth(
        vs.core.bs.VideoSource(CLIP_PATH), bits=32, fulls=True, fulld=True
    )
    small = vs.core.std.CropAbs(yuv420, width=640, height=8)
    with pytest.raises(vs.Error, match="8x8"):
        vs.core.vsfeel.BM3Dv2(small, sigma=0.7, radius=1)
    # the same clip is fine when only luma is denoised
    ok = vs.core.vsfeel.BM3Dv2(small, sigma=[0.7, 0.0], radius=1)
    assert ok.num_frames == small.num_frames


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": [0.7, 0.7, 0.7]},
        {"sigma": [0.7], "chroma": 1},
    ],
)
def test_bm3dv2_accepts_yuv444(kwargs):
    """The per-plane and joint entry points both accept 4:4:4 input."""
    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(CLIP_PATH), format=vs.YUV444PS)
    out = BM3D(yuv444, radius=1, bm_range=2, ps_range=1, block_step=4, **kwargs)
    for n in (0, 5):
        for plane in range(3):
            assert np.isfinite(plane_to_ndarray(out.get_frame(n), plane)).all()


@pytest.mark.parametrize("sigma", [0.0, 1e-9])
@pytest.mark.parametrize("use_ref", [False, True])
def test_bm3dv2_sigma_below_epsilon_passes_through(clip_gray, sigma, use_ref):
    """A plane whose sigma is below FLT_EPSILON is a bit-exact source copy,
    as in the reference's PROC_MASK. Covers the old sigma=0 + ref 0/0 NaN."""
    basic = None
    if use_ref:
        basic = BM3D(
            clip_gray,
            sigma=SIGMA,
            radius=2,
            bm_range=BM_RANGE,
            ps_range=PS_RANGE,
            block_step=BLOCK_STEP,
        )
        _ = frame_to_ndarray(basic.get_frame(0))
    out = BM3D(
        clip_gray,
        sigma=sigma,
        radius=2,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        **({"ref": basic} if use_ref else {}),
    )
    for n in (0, 11, 23):
        a = frame_to_ndarray(out.get_frame(n))
        b = frame_to_ndarray(clip_gray.get_frame(n))
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"
        assert np.array_equal(a, b), f"sigma={sigma} plane not passed through at {n}"


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
    src = core.bs.VideoSource({CLIP_PATH!r})
    gray = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                              bits=32, fulls=True, fulld=True)
    clip = core.std.Loop(gray, times=4)
    kwargs = dict(sigma={SIGMA}, radius=2, bm_range={BM_RANGE},
                  ps_range={PS_RANGE}, block_step={BLOCK_STEP})

    def vsfeel(clip, **kw):
        # Pixels are read directly here, so the GPU-resident node is downloaded.
        node = core.vsfeel.BM3Dv2(clip, **kw)
        return core.std.GPUDownload(clip=node) if node.gpu_resident else node

    order = [0, 64, 1, 65, 2, 13, 79, 40, 95, 3]

    # A serial run is the self-consistency oracle.
    try:
        base = vsfeel(clip, **kwargs)
        ref = {{n: read_plane(base.get_frame(n), 0, np.float32) for n in order}}
    except Exception as exc:
        print("VSFEEL fail: serial run: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(3)
    print("REF ok", flush=True)

    out = vsfeel(clip, **kwargs)
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
    must match a serial run (measured max diff ~2e-8)."""
    try:
        worst = run_compare_subprocess(_SEEK_SCRIPT, [], timeout=300)
    except ReferenceUnavailable as exc:
        # There is no external reference here: anything that dies before the
        # serial run completes is a vsfeel failure, not a missing reference.
        raise AssertionError(f"seek-schedule subprocess failed early: {exc}") from exc
    assert worst < 1e-5, f"seek schedule mismatch: {worst}"


def test_bm3dv2_frame_request_order_matches_serial():
    """repeat/reverse/far/random request orders must match the serial run.

    The aggregation accumulates with atomicAdd, so the self-consistency floor
    is the ordering rounding of the sums (~3e-8 measured), not exact equality.
    """
    assert_temporal_order_consistent(
        "BM3Dv2",
        {
            "sigma": SIGMA,
            "radius": 2,
            "bm_range": BM_RANGE,
            "ps_range": PS_RANGE,
            "block_step": BLOCK_STEP,
        },
        tol=1e-5,
        timeout=300,
    )


@pytest.mark.parametrize("nframes", [1, 2])
def test_bm3dv2_short_clip_temporal_window(nframes):
    """A clip shorter than 2*radius+1 must still be order-independent."""
    assert_temporal_order_consistent(
        "BM3Dv2",
        {
            "sigma": SIGMA,
            "radius": 2,
            "bm_range": BM_RANGE,
            "ps_range": PS_RANGE,
            "block_step": BLOCK_STEP,
        },
        tol=1e-5,
        nframes=nframes,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Matching threshold (th_mse), group sizes and temporal real-frame bounds
# ---------------------------------------------------------------------------


def _structured_clip(nframes, seed=5, size=64):
    """A drifting pattern plus noise, so thresholds actually reject candidates.

    The committed noise clip is far too low-amplitude: its sigma-scaled default
    threshold accepts every candidate, which makes every group a full eight and
    hides the retention and length-N contracts entirely.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    arr = np.zeros((nframes, size, size), dtype=np.float32)
    for f in range(nframes):
        base = 0.5 + 0.3 * np.sin((xs + 3.0 * f) * 0.3) * np.cos(ys * 0.2)
        arr[f] = (base + rng.normal(0, 0.05, (size, size))).astype(np.float32)

    src = vs.core.std.BlankClip(width=size, height=size, format=vs.GRAYS, length=nframes)

    def setter(n, f):
        out = f.copy()
        plane = np.asarray(out[0])
        plane[:] = arr[n][: plane.shape[0], : plane.shape[1]]
        return out

    return vs.core.std.ModifyFrame(src, src, setter), arr


@pytest.mark.parametrize("bad", [-1.0, -1e-9, float("nan"), float("inf"), float("-inf")])
def test_bm3dv2_rejects_bad_th_mse(clip_gray, bad):
    """A threshold that is negative or not finite has no meaning and must be
    rejected at creation, before any GPU resource exists."""
    with pytest.raises(vs.Error):
        _run(clip_gray, th_mse=bad)


def test_bm3dv2_accepts_enormous_finite_th_mse(clip_gray):
    """A finite but unrepresentable threshold saturates to the largest float
    instead of narrowing to infinity."""
    out = _run(clip_gray, th_mse=1e300)
    assert np.isfinite(frame_to_ndarray(out.get_frame(0))).all()


def test_bm3dv2_validates_th_mse_before_the_all_zero_shortcut():
    """BM3Dv2 returns the source clip when every sigma is zero; an invalid
    th_mse must still be an error rather than being conditionally accepted."""
    with pytest.raises(vs.Error):
        vs.core.vsfeel.BM3Dv2(_blank(64, 64), sigma=[0.0], th_mse=-1.0)


def test_bm3dv2_th_mse_selects_the_group():
    """A rejecting threshold must produce a different result from a permissive
    one, and zero must behave as the reference-only group."""
    clip, _ = _structured_clip(5)
    kwargs = dict(sigma=[8.0], radius=2, bm_range=9, ps_range=4, block_step=8)
    loose = frame_to_ndarray(BM3D(clip, th_mse=1e6, **kwargs).get_frame(2))
    tight = frame_to_ndarray(BM3D(clip, th_mse=300.0, **kwargs).get_frame(2))
    zero = frame_to_ndarray(BM3D(clip, th_mse=0.0, **kwargs).get_frame(2))
    assert np.isfinite(loose).all() and np.isfinite(tight).all()
    assert np.abs(loose - tight).max() > 1e-3, "the threshold changed nothing"
    assert np.abs(loose - zero).max() > 1e-3
    # The reference-only group cannot grow with the radius: no temporal
    # neighbour enters a group the threshold never lets it into.
    r0 = frame_to_ndarray(
        BM3D(
            clip, sigma=[8.0], th_mse=0.0, radius=0, bm_range=9, ps_range=4, block_step=8
        ).get_frame(2)
    )
    assert np.abs(r0 - zero).max() < 1e-5


@pytest.mark.parametrize(
    "th", [0.0, 50.0, 100.0, 150.0, 200.0, 240.0, 280.0, 300.0, 350.0, 500.0, 1e6]
)
def test_bm3dv2_length_n_transform_round_trips(th):
    """With a near-zero threshold every coefficient survives, so the forward
    and inverse group transforms of length N must return the input exactly,
    for whatever N each reference block's group happens to have. A wrong
    length-N normalization shows up here as a scaled (8/N) output."""
    clip, arr = _structured_clip(5)
    out = frame_to_ndarray(
        BM3D(
            clip, sigma=[1e-4], radius=2, bm_range=9, ps_range=4, block_step=8, th_mse=th
        ).get_frame(2)
    )
    assert np.isfinite(out).all()
    assert np.abs(out - arr[2]).max() < 1e-5


@pytest.mark.parametrize("nframes,radii", [(1, (0, 16)), (2, (1, 16))])
def test_bm3dv2_searches_only_real_frames(nframes, radii):
    """A clip with no temporal neighbours must ignore the radius entirely: the
    window must not search, or aggregate, the same endpoint frame once per
    out-of-range window position."""
    clip, _ = _structured_clip(nframes)
    kwargs = dict(sigma=[8.0], bm_range=9, ps_range=4, block_step=8)
    for n in range(nframes):
        a = frame_to_ndarray(BM3D(clip, radius=radii[0], **kwargs).get_frame(n))
        b = frame_to_ndarray(BM3D(clip, radius=radii[1], **kwargs).get_frame(n))
        assert np.abs(a - b).max() < 1e-5, f"frame {n}: radius {radii} disagree"


def test_bm3dv2_aggregates_only_real_neighbours():
    """An output frame may only depend on the real estimate centres inside the
    clip, so extending the clip beyond n + 2*radius must leave output n alone
    (each of its centres searches no further than m + radius)."""
    short, _ = _structured_clip(9)
    long, _ = _structured_clip(13)
    kwargs = dict(sigma=[8.0], radius=2, bm_range=9, ps_range=4, block_step=8)
    for n in range(5):
        a = frame_to_ndarray(BM3D(short, **kwargs).get_frame(n))
        b = frame_to_ndarray(BM3D(long, **kwargs).get_frame(n))
        assert np.abs(a - b).max() < 1e-5, f"frame {n} depends on the clip's tail"


def test_bm3dv2_mixed_group_sizes_have_no_nan():
    """Groups of different sizes coexist in one workgroup; every frame of a
    clip whose thresholds span full, partial and reference-only groups must be
    finite and still denoise."""
    clip, arr = _structured_clip(7)
    kwargs = dict(sigma=[8.0], radius=3, bm_range=9, ps_range=4, block_step=4)
    for th in (0.0, 250.0, 320.0, 400.0, 1e6):
        out = BM3D(clip, th_mse=th, **kwargs)
        for n in (0, 3, 6):
            a = frame_to_ndarray(out.get_frame(n))
            assert np.isfinite(a).all(), f"non-finite at frame {n}, th_mse={th}"
        assert np.abs(frame_to_ndarray(out.get_frame(3)) - arr[3]).max() > 1e-4


_DCT_CASE = re.compile(r"case (\d+):(.*?)break;", re.S)
_DCT_SUM = re.compile(r"o(\d) = (.*?);")


def _dct_expr(expr, identity):
    """Evaluate one generated forward row into a length-8 coefficient vector."""
    acc = np.zeros(8, dtype=np.float64)
    while True:
        fma = re.match(r"^fma\((-?[\d.e+-]+)f, v\[(\d)\], (.*)\)$", expr)
        mul = re.match(r"^(-?[\d.e+-]+)f \* v\[(\d)\]$", expr)
        if fma:
            acc += np.float64(np.float32(fma.group(1))) * identity[int(fma.group(2))]
            expr = fma.group(3)
            continue
        if mul:
            acc += np.float64(np.float32(mul.group(1))) * identity[int(mul.group(2))]
            break
        if expr == "0.0f":
            break
        neg = re.match(r"^-v\[(\d)\]$", expr)
        pos = re.match(r"^v\[(\d)\]$", expr)
        assert neg or pos, expr
        acc += (-1.0 if neg else 1.0) * identity[int((neg or pos).group(1))]
        break
    return np.float32(acc)


def _parse_dct(source, name):
    """Read one length-N transform out of the shader as a list of matrices."""
    body = re.search(
        rf"void {name}\(inout float v\[8\], int n\) \{{(.*?)\n\}}", source, re.S
    ).group(1)
    identity = np.eye(8, dtype=np.float32)
    out = {}
    for case in _DCT_CASE.finditer(body):
        mat = np.zeros((8, 8), dtype=np.float32)
        for row in _DCT_SUM.finditer(case.group(2)):
            mat[int(row.group(1))] = _dct_expr(row.group(2).strip(), identity)
        out[int(case.group(1))] = mat
    return out


def test_bm3dv2_group_dct_tables_are_a_scaled_dct_ii_pair():
    """The generated length-N tables must be a scaled DCT-II with
    A^T A = 2N I and an inverse that is exactly A's transpose: the filtering
    sigma and the 1/(512N) normalization are derived from that gain."""
    source = (Path(__file__).resolve().parents[1] / "src" / "bm3d.comp").read_text()
    fwd = _parse_dct(source, "group_dct_fwd_n")
    inv = _parse_dct(source, "group_dct_inv_n")
    assert sorted(fwd) == list(range(1, 8))
    assert sorted(inv) == list(range(1, 8))
    for n in range(1, 8):
        a = fwd[n][:n, :n]
        assert np.allclose(fwd[n][n:], 0.0) and np.allclose(inv[n][n:], 0.0)
        assert np.allclose(a.T @ a / (2 * n), np.eye(n), atol=1e-6), n
        assert np.allclose(inv[n][:n, :n], a.T), n
        # A_1[0][0] = sqrt(2): a one-member group is not the identity, so the
        # normalization really is 1/(512N) at every N rather than 1/512.
        assert fwd[n][0, 0] == pytest.approx(np.sqrt(2.0), rel=1e-7)


# ---------------------------------------------------------------------------
# Reference comparison: mawen's CPU V-BM3D, the implementation we ported
# ---------------------------------------------------------------------------
#
# This is the only oracle. vszipcl and bm3dvk implement the *CUDA* matcher and
# the CUDA filtering conventions, so they answer a different question and are
# not compared against at all (see notes/BM3D.md).
#
# The CPU plugin is driven as `VBasic` + `VAggregate` for the temporal stages
# and as `Basic` for the radius-zero spatial one, with every parameter the two
# sides share set explicitly (`block_size=8`, `group_size=8`, `bm_step=1`,
# `ps_step=1`, and `matrix=1` to pin the CPU's color-matrix norm to the bt709
# value vsfeel's sigma and threshold factors carry). `th_mse` is passed
# explicitly whenever vsfeel is given one, and otherwise reproduced from the
# documented stage default.
#
# The residual is the CPU's unspecified tie order and its SSE accumulation
# order: equal-distance blocks are ordered arbitrarily by `std::partial_sort`,
# and one flipped seed moves the whole predictive chain. It therefore grows
# with how ambiguous the content's block distances are; the bounds below are
# measurements on the committed clip, not theory.

_CPU_SCRIPT = (
    COMPARE_PRELUDE
    + r"""
import json
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.max_cache_size = 2048
spec = json.loads(sys.argv[1])
kind = spec["clip"]
feel = dict(spec["kwargs"])
frames = spec["frames"]
stage = spec["stage"]
# "int16": both sides produce 16 bit integer output -- vsfeel's u16 path and the
# CPU's VAggregate(sample=INTEGER) / Basic -- so every plane is read back as
# samples scaled into [0, 1] and one bound covers both input depths.
int16 = spec.get("sample") == "int16"
planes = None

source = core.bs.VideoSource(spec["source"])
if kind == "gray32":
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(source, 0, vs.GRAY), bits=32,
                              fulls=True, fulld=True)
elif kind == "yuv420_32":
    clip = core.fmtc.bitdepth(source, bits=32, fulls=True, fulld=True)
elif kind == "yuv444_32":
    clip = core.resize.Bicubic(source, format=vs.YUV444PS)
elif kind == "rgb32":
    clip = core.resize.Bicubic(source, format=vs.RGBS, matrix_in_s="709")
elif kind == "gray16":
    clip = core.fmtc.bitdepth(core.std.ShufflePlanes(source, 0, vs.GRAY), bits=16,
                              fulls=True, fulld=True)
elif kind == "yuv444_16":
    clip = core.resize.Bicubic(source, format=vs.YUV444P16)
else:
    raise SystemExit("bad clip kind %r" % kind)
if int16:
    # The CPU scales an *integer* clip by the frame's _Range: limited (or
    # absent) maps [16 << (b-8), 235 << (b-8)] to [0, 1] and clips the output
    # back. vsfeel's u16 path is full range, like its fp32 one and like every
    # other vsfeel filter, so the two sides only meet in the same domain on a
    # full-range clip.
    clip = core.std.SetFrameProp(clip, prop="_Range", intval=1)
planes = list(range(clip.format.num_planes))


def read_all(node, n):
    frame = node.get_frame(n)
    if frame.format.sample_type == vs.INTEGER:
        return [read_plane(frame, p, np.uint16).astype(np.float64) / 65535.0
                for p in planes]
    return [read_plane(frame, p, np.float32) for p in planes]


def vsfeel(clip, **kw):
    node = core.vsfeel.BM3Dv2(clip, **kw)
    return core.std.GPUDownload(clip=node) if node.gpu_resident else node


feel.setdefault("sigma", [0.7])
feel.setdefault("radius", 2)
feel.setdefault("bm_range", 9)
feel.setdefault("ps_range", 4)
feel.setdefault("ps_num", 2)
feel.setdefault("block_step", 8)

sigma = feel["sigma"]
if not isinstance(sigma, (list, tuple)):
    sigma = [sigma]
radius = feel["radius"]

# bm3d.Basic is the spatial-only filter: it takes no temporal parameters.
cpu = dict(
    sigma=list(sigma),
    block_size=8,
    block_step=feel["block_step"],
    group_size=8,
    bm_range=feel["bm_range"],
    bm_step=1,
    matrix=1,
)
if stage != "image":
    cpu.update(radius=radius, ps_num=feel["ps_num"], ps_range=feel["ps_range"],
               ps_step=1)
if "th_mse" in feel:
    cpu["th_mse"] = feel["th_mse"]
else:
    # The CPU's own stage defaults, in its user domain.
    cpu["th_mse"] = (200.0 + sigma[0] * 10.0) if stage == "final" else (400.0 + sigma[0] * 80.0)

# The final stage runs both sides on the *same* guide (vsfeel's basic estimate),
# so the comparison isolates the Wiener pass.
guide = None
if stage == "final":
    try:
        guide = vsfeel(clip, sigma=list(sigma), radius=radius,
                       block_step=feel["block_step"], bm_range=feel["bm_range"],
                       ps_num=feel["ps_num"], ps_range=feel["ps_range"])
        _ = [read_all(guide, n) for n in frames]
    except Exception as exc:
        print("VSFEEL fail: basic estimate: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(3)

try:
    sample = vs.INTEGER if int16 else vs.FLOAT
    if stage == "image":
        ref_node = core.bm3d.Basic(clip, **cpu)
    else:
        basic = core.bm3d.VBasic(clip, **cpu)
        if stage == "final":
            final = core.bm3d.VFinal(clip, ref=guide, **cpu)
            ref_node = core.bm3d.VAggregate(final, radius=radius, sample=sample)
        else:
            ref_node = core.bm3d.VAggregate(basic, radius=radius, sample=sample)
    ref_frames = [read_all(ref_node, n) for n in frames]
except Exception as exc:
    print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
    raise SystemExit(2)
print("REF ok", flush=True)

my_node = vsfeel(clip, **dict(feel, **({"ref": guide} if stage == "final" else {})))
worst = 0.0
per_plane = [0.0] * len(planes)
for n, ref_planes in zip(frames, ref_frames):
    my_planes = read_all(my_node, n)
    for i, (a, b) in enumerate(zip(my_planes, ref_planes)):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print("VSFEEL fail: non-finite at frame %d plane %d" % (n, planes[i]), flush=True)
            raise SystemExit(3)
        d = float(np.abs(a - b).max())
        per_plane[i] = max(per_plane[i], d)
        worst = max(worst, d)
print("RESULT " + json.dumps({"maxdiff": worst, "per_plane": per_plane}), flush=True)
"""
)


def _cpu_compare(kwargs, stage="basic", clip="gray32", frames=(0, 11, 23), sample="float"):
    """Worst per-plane diff against the CPU implementation, in a subprocess."""
    if not _CPU_AVAILABLE:
        skip_or_fail_reference("the CPU bm3d plugin (Basic/VBasic/VAggregate) is not installed")
    spec = {
        "source": CLIP_PATH,
        "clip": clip,
        "frames": list(frames),
        "kwargs": dict(kwargs),
        "stage": stage,
        "sample": sample,
    }
    return run_compare_subprocess(_CPU_SCRIPT, [json.dumps(spec)], timeout=1800)


BASE_KWARGS = dict(sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4)

# Measured on the committed clip (640x360 Big Buck Bunny + grain), frames
# 0/11/23, worst plane, against the reference build the dev group pins. The CPU
# is `VBasic`+`VAggregate` (or `Basic` at radius 0) with the parameters above
# and its own stage default for th_mse. The residuals are the CPU's own tie
# order and one SSD ulp, not porting error: see notes/BM3D.md.
RADIUS_CASES = [
    # (radius, stage, bound, measured)
    (0, "image", 4e-3, 0.00144),
    (1, "basic", 4e-3, 0.00129),
    (2, "basic", 4e-3, 0.00126),
    (3, "basic", 4e-3, 0.00113),
    (4, "basic", 4e-3, 0.00117),
    (5, "basic", 4e-3, 0.00190),
    (6, "basic", 4e-3, 0.00161),
]


@pytest.mark.parametrize(
    "radius,stage,bound,measured", RADIUS_CASES, ids=[str(c[0]) for c in RADIUS_CASES]
)
def test_bm3dv2_matches_cpu_radius(clip_gray, radius, stage, bound, measured):
    """Every supported radius must track the CPU implementation."""
    payload = _cpu_compare(dict(BASE_KWARGS, radius=radius), stage=stage)
    assert payload["maxdiff"] < bound, f"radius {radius} vs CPU: {payload} (was {measured})"


# The parameter sweep around the defaults, every entry merged over
# BASE_KWARGS and graded against the CPU implementation. extractor_exp is
# absent on purpose: it is a vsfeel/vszipcl accumulation knob the CPU plugin
# does not have, so it is covered by the self-consistency tests instead.
SWEEP_CONFIGS = [
    # (kwargs, bound, measured)
    ({}, 3e-3, 0.00126),
    ({"sigma": 0.3}, 1.5e-3, 0.00044),
    ({"sigma": 1.5}, 5e-3, 0.00201),
    ({"block_step": 1}, 1.5e-3, 0.00037),
    ({"block_step": 2}, 1.5e-3, 0.00050),
    ({"block_step": 3}, 2e-3, 0.00071),
    ({"block_step": 5}, 6e-3, 0.00231),
    ({"block_step": 6}, 6e-3, 0.00222),
    ({"block_step": 7}, 7e-3, 0.00272),
    ({"block_step": 8}, 8e-3, 0.00309),
    ({"bm_range": 1}, 3e-3, 0.00117),
    ({"bm_range": 4}, 3e-3, 0.00116),
    ({"bm_range": 9}, 3e-3, 0.00125),
    ({"bm_range": 22}, 3e-3, 0.00129),
    ({"bm_range": 32}, 3e-3, 0.00126),
    ({"ps_range": 5}, 3e-3, 0.00105),
    ({"ps_range": 9}, 3e-3, 0.00126),
    ({"ps_num": 1}, 3e-3, 0.00125),
    ({"ps_num": 2}, 3e-3, 0.00126),
    ({"ps_num": 3}, 3e-3, 0.00113),
    ({"ps_num": 4}, 3e-3, 0.00113),
    ({"ps_num": 5}, 3e-3, 0.00094),
    ({"ps_num": 6}, 3e-3, 0.00097),
    ({"ps_num": 7}, 3e-3, 0.00097),
    ({"ps_num": 8}, 3e-3, 0.00097),
]


def _sweep_id(cfg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in cfg.items()) or "empty+BASE_KWARGS"


@pytest.mark.parametrize(
    "cfg,bound,measured", SWEEP_CONFIGS, ids=[_sweep_id(c) for c, _, _ in SWEEP_CONFIGS]
)
def test_bm3dv2_parameter_sweep_matches_cpu(clip_gray, cfg, bound, measured):
    """Parameter grid around the defaults, graded against the CPU plugin."""
    payload = _cpu_compare(dict(BASE_KWARGS, **cfg))
    assert payload["maxdiff"] < bound, f"max diff vs CPU ({cfg}): {payload} (was {measured})"


@pytest.mark.parametrize(
    "kwargs,bound,measured",
    [
        ({}, 3e-3, 0.00126),
        ({"sigma": [0.5, 1.1, 0.3]}, 2e-3, 0.00074),
        ({"th_mse": 1200.0}, 3e-3, 0.00126),
        ({"ps_num": 5}, 3e-3, 0.00094),
    ],
    ids=["plugin-defaults", "sigma-array-inherit", "explicit-th-mse", "ps-num-5"],
)
def test_bm3dv2_plugin_defaults_match_cpu(clip_gray, kwargs, bound, measured):
    """The documented defaults must produce the CPU's result, not just any
    finite frame (test_device_limits/test_lifecycle only check those)."""
    payload = _cpu_compare(dict(BASE_KWARGS, **kwargs))
    assert payload["maxdiff"] < bound, f"max diff vs CPU ({kwargs}): {payload}"


def test_bm3dv2_th_mse_matches_cpu(clip_gray):
    """An explicit th_mse must reproduce the CPU's run at the same value.

    The two domains line up exactly (`th_mse` is multiplied by the same
    color-matrix norm on both sides), so a systematic unit error shows up as a
    different group and a large diff."""
    for th in (600.0, 1040.0, 2000.0):
        payload = _cpu_compare(dict(BASE_KWARGS, th_mse=th))
        assert payload["maxdiff"] < 3e-3, f"th_mse={th} vs CPU: {payload}"


def test_bm3dv2_final_stage_matches_cpu(clip_gray):
    """The Wiener pass, with both sides driven by the same guide.

    The guide is vsfeel's basic estimate, so this isolates the final stage's
    matching and Wiener shrinkage from any basic-stage difference. Its
    accumulation order jitters the result by ~2e-5 between processes."""
    payload = _cpu_compare(BASE_KWARGS, stage="final")
    assert payload["maxdiff"] < 5e-4, f"final pass vs CPU: {payload} (was 0.00052)"


def test_bm3dv2_luma_only_color_matches_cpu():
    """A colour clip with chroma disabled must match the CPU on luma.

    Only the luma plane is comparable: the CPU scales a plane's sigma by that
    plane's colour-matrix norm (normY/normU/normV) while vsfeel's factors carry
    normY for every plane, and its RGB input goes through no opponent-colour
    transform at all. With `sigma=[x, 0, 0]` both sides process luma alone, so
    the domains are identical. 4:4:4 because the CPU's plugin refuses
    subsampled input when chroma is processed.
    """
    payload = _cpu_compare(dict(BASE_KWARGS, sigma=[0.7, 0.0, 0.0]), clip="yuv444_32")
    assert payload["per_plane"][0] < 3e-3, f"luma vs CPU: {payload}"


def test_bm3dv2_per_plane_color_matches_cpu():
    """All three planes of a 4:4:4 clip, chroma included.

    The chroma thresholds differ by the normU/normV ratio (0.64/0.68 against
    normY 0.75), so this is a close but not exact domain; the measured bound
    carries that, and it still catches a gross matcher or indexing error on
    chroma. The joint `chroma=True` entry has no CPU equivalent at all and is
    covered by the self-consistency tests.
    """
    payload = _cpu_compare(BASE_KWARGS, clip="yuv444_32")
    assert payload["maxdiff"] < 1.5e-2, f"color vs CPU: {payload} (was 0.00676)"


def test_bm3dv2_color_ref_pass_matches_cpu():
    """The Wiener pass on luma of a colour clip, chroma disabled."""
    payload = _cpu_compare(
        dict(BASE_KWARGS, sigma=[0.7, 0.0, 0.0]), stage="final", clip="yuv444_32"
    )
    assert payload["per_plane"][0] < 1e-3, f"colour final pass vs CPU: {payload}"


# ---------------------------------------------------------------------------
# 16 bit integer input
# ---------------------------------------------------------------------------
#
# The u16 path is the fp32 path with a different transport: the source ring and
# the estimate stacks hold float at both depths, an integer clip's samples are
# widened once when the ring is filled (gain 1/65535 with no offset, the
# reference's Int2Float), and the aggregation rounds its result back to native
# samples (its Float2Int). Nothing about the algorithm changes, which is why it
# is graded against the fp32 comparisons' own bounds.
#
# The oracle is the CPU plugin's 16 bit output: VBasic + VAggregate at
# sample=INTEGER, or plain Basic at radius 0. The CPU scales an *integer* clip
# by the frame's _Range property (limited range maps [4096, 60160] to [0, 1]
# and clips the output back), so _CPU_SCRIPT pins _Range=1: vsfeel's u16 path
# is full range, like its fp32 path and every other vsfeel filter, and the two
# sides only meet in the same domain on a full-range clip.

U16_BASE = BASE_KWARGS


def _std_args(**kwargs):
    """The module's standard test arguments, with `kwargs` overriding them."""
    args = dict(
        sigma=SIGMA,
        radius=2,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
    )
    args.update(kwargs)
    return args


def _run16(clip, **kwargs):
    """BM3Dv2 at the standard test arguments (depth-agnostic: the fp32 arm of
    the equivalence test below uses it too, on a widened clip)."""
    return BM3D(clip, **_std_args(**kwargs))


def _u16_compare(kwargs, stage="basic", clip="gray16", frames=(0, 11, 23)):
    return _cpu_compare(
        dict(U16_BASE, **kwargs), stage=stage, clip=clip, frames=frames, sample="int16"
    )


# Measured like RADIUS_CASES above, on the same clip and frames, with the
# u16 arm in place of the fp32 one: the residual is the CPU's tie order, the
# same one the fp32 comparisons carry, plus at most one output code.
U16_CASES = [
    # (radius, stage, bound, measured)
    (0, "image", 4e-3, 0.00113),
    (1, "basic", 4e-3, 0.00116),
    (2, "basic", 4e-3, 0.00116),
    (4, "basic", 4e-3, 0.00119),
]


@pytest.mark.parametrize(
    "radius,stage,bound,measured", U16_CASES, ids=[f"r{c[0]}" for c in U16_CASES]
)
def test_bm3dv2_u16_matches_cpu(clip_16bit, radius, stage, bound, measured):
    """Every supported radius must track the CPU's 16 bit output."""
    payload = _u16_compare(dict(radius=radius), stage=stage)
    assert payload["maxdiff"] < bound, f"u16 radius {radius} vs CPU: {payload} (was {measured})"


U16_SWEEP = [
    # (kwargs, bound, measured)
    ({}, 3e-3, 0.00116),
    ({"sigma": 1.5}, 6e-3, 0.00272),
    ({"sigma": 0.3}, 2e-3, 0.00055),
    ({"block_step": 1}, 1.5e-3, 0.00041),
    ({"block_step": 8}, 8e-3, 0.00327),
    ({"bm_range": 4}, 3e-3, 0.00117),
    ({"ps_range": 9}, 3e-3, 0.00131),
    ({"ps_num": 5}, 3e-3, 0.00102),
    ({"th_mse": 1200.0}, 3e-3, 0.00116),
]


@pytest.mark.parametrize(
    "cfg,bound,measured", U16_SWEEP, ids=[_sweep_id(c) for c, _, _ in U16_SWEEP]
)
def test_bm3dv2_u16_parameter_sweep_matches_cpu(clip_16bit, cfg, bound, measured):
    """The parameter grid around the defaults, on an integer clip."""
    payload = _u16_compare(cfg)
    assert payload["maxdiff"] < bound, f"u16 max diff vs CPU ({cfg}): {payload} (was {measured})"


def test_bm3dv2_u16_final_stage_matches_cpu(clip_16bit):
    """The Wiener pass on an integer clip, both sides driven by the same guide.

    The guide is vsfeel's own u16 basic estimate, so this isolates the final
    stage's matching and shrinkage from any basic-stage difference -- and it
    also exercises the second ring section (the `ref` clip is widened by the
    same copy kernel as the source)."""
    payload = _u16_compare({}, stage="final")
    assert payload["maxdiff"] < 1e-3, f"u16 final pass vs CPU: {payload} (was 0.00062)"


def test_bm3dv2_u16_joint_chroma_matches_cpu():
    """The joint 4:4:4 entry on an integer clip.

    Same close-but-not-exact chroma domain as the fp32 colour test: the CPU
    scales each plane's sigma by that plane's matrix norm, vsfeel carries
    normY for all three."""
    payload = _u16_compare({"sigma": [0.7, 0.7, 0.7]}, clip="yuv444_16")
    assert payload["maxdiff"] < 1.5e-2, f"u16 joint chroma vs CPU: {payload}"


def _widen_to_floats(clip16):
    """The exact floats the ring copy produces for each sample.

    bm3d_copy16 writes `float(int(sample)) * (1.0f / 65535.0f)`; reproducing it
    in numpy (one float32 multiply of the same two operands) is what makes the
    fp32 arm below see bit-identical data, and therefore what makes the
    equality assertion meaningful rather than a tolerance.
    """
    scale = np.float32(1.0 / 65535.0)
    base = clip16.resize.Point(format=vs.GRAYS)

    def widen(n, f):
        src, tmpl = f
        out = tmpl.copy()
        raw = np.ctypeslib.as_array(
            ctypes.cast(out.get_write_ptr(0), ctypes.POINTER(ctypes.c_uint8)),
            shape=(out.height, out.get_stride(0)),
        )
        s = np.ctypeslib.as_array(
            ctypes.cast(src.get_read_ptr(0), ctypes.POINTER(ctypes.c_uint8)),
            shape=(src.height, src.get_stride(0)),
        )
        u = s[:, : src.width * 2].copy().view(np.uint16)
        raw[:, : out.width * 4].view(np.float32)[:] = u.astype(np.float32) * scale
        return out

    return vs.core.std.ModifyFrame(base, [clip16, base], widen)


def _u16_path_is_the_fp32_path(clip16, **kwargs):
    """u16 output vs the fp32 output on widened input, in whole codes."""
    u16_out = _run16(clip16, **kwargs)
    f32_out = BM3D(_widen_to_floats(clip16), **_std_args(**kwargs))
    worst = 0
    for n in (0, 11, 23):
        a = plane_to_ndarray(u16_out.get_frame(n), 0, np.uint16).astype(np.int32)
        v = plane_to_ndarray(f32_out.get_frame(n), 0, np.float32)
        b = np.floor(
            np.clip(
                v * np.float32(65535.0) + np.float32(0.5),
                np.float32(0.0),
                np.float32(65535.0),
            )
        ).astype(np.int32)
        worst = max(worst, int(np.abs(a - b).max()))
    return worst


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"sigma": 1.5}, {"radius": 0}, {"block_step": 1}, {"extractor_exp": 1}],
    ids=["default", "sigma1.5", "radius0", "block_step1", "extractor"],
)
def test_bm3dv2_u16_is_the_fp32_path_with_a_rounded_store(clip_16bit, kwargs):
    """The u16 output must be the fp32 result, rounded to native samples.

    Both arms run the same estimation kernel over bit-identical floats (the
    fp32 arm is fed exactly what the copy kernel widens the integer clip to),
    so the only admissible difference is the store's rounding plus the
    aggregation's accumulation order -- and the latter is already known to
    jitter by one code between runs (the fp32 self-consistency tests carry
    it). Measured: 1 code at every configuration here, which this pins: a
    wrong gain, offset, clamp or rounding would show up as thousands.
    """
    worst = _u16_path_is_the_fp32_path(clip_16bit, **kwargs)
    assert worst <= 1, f"u16 output is not the fp32 result rounded ({kwargs}): {worst} codes"


def test_bm3dv2_u16_output_is_16_bit_and_keeps_props(clip_16bit):
    """An integer clip stays integer, keeps its frame properties, and changes."""
    out = _run16(clip_16bit)
    fmt = out.format
    assert fmt.sample_type == vs.INTEGER and fmt.bits_per_sample == 16
    assert_all_frames_finite(out)
    src = plane_to_ndarray(clip_16bit.get_frame(11), 0, np.uint16).astype(np.int32)
    got = plane_to_ndarray(out.get_frame(11), 0, np.uint16).astype(np.int32)
    assert np.abs(got - src).max() > 8, "the u16 path did not denoise"
    assert_preserves_frame_props(_run16, clip_16bit, radius=2)


def test_bm3dv2_u16_parallel_load_matches_serial(clip_16bit):
    """The parallel request load must agree with the serial one, in codes."""
    par = eval_parallel(_run16, clip_16bit, radius=2, dtype=np.uint16)
    ref = _run16(clip_16bit, radius=2)
    for n in range(clip_16bit.num_frames):
        d = par[n].astype(np.int32) - plane_to_ndarray(ref.get_frame(n), 0, np.uint16).astype(
            np.int32
        )
        assert np.abs(d).max() <= 1, f"u16 parallel/serial mismatch at frame {n}"


def test_bm3dv2_u16_deterministic(clip_16bit):
    """Two runs must agree within the aggregation's one-code jitter."""
    a = eval_parallel(_run16, clip_16bit, radius=2, dtype=np.uint16)
    b = eval_parallel(_run16, clip_16bit, radius=2, dtype=np.uint16)
    for n in range(clip_16bit.num_frames):
        d = np.abs(a[n].astype(np.int32) - b[n].astype(np.int32))
        assert d.max() <= 1, f"u16 nondeterministic output at frame {n}"


def test_bm3dv2_u16_cas_fallback_matches_hardware_atomics(clip_16bit, monkeypatch):
    """The no-float-atomics accumulation must agree on an integer clip too."""
    monkeypatch.setenv("VSFEEL_BM3D_CAS", "1")
    a = _run16(clip_16bit, radius=2)
    monkeypatch.delenv("VSFEEL_BM3D_CAS")
    b = _run16(clip_16bit, radius=2)
    for n in (0, 11, 23):
        d = np.abs(
            plane_to_ndarray(a.get_frame(n), 0, np.uint16).astype(np.int32)
            - plane_to_ndarray(b.get_frame(n), 0, np.uint16).astype(np.int32)
        )
        assert d.max() <= 1, f"u16 CAS vs hardware atomics at frame {n}"


def test_bm3dv2_u16_rejects_a_float_ref(clip_16bit, clip_gray):
    """A \"ref\" of the other depth must be rejected up front."""
    with pytest.raises(vs.Error):
        _run16(clip_16bit, ref=_run(clip_gray, radius=2))


def test_bm3dv2_u16_zero_sigma_returns_the_source(clip_16bit):
    """All planes below epsilon hand the integer clip straight back."""
    out = cpu_node(vs.core.vsfeel.BM3Dv2(clip_16bit, sigma=0.0))
    assert out.format.id == clip_16bit.format.id
    a = plane_to_ndarray(out.get_frame(3), 0, np.uint16)
    b = plane_to_ndarray(cpu_node(clip_16bit).get_frame(3), 0, np.uint16)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Multi-plane self-consistency
# ---------------------------------------------------------------------------
#
# The per-entry caches (source ring, estimate stacks, witnesses) are separate
# tables now, so the request-order and parallel-load oracles are re-run on a
# color clip where three entries are live at once.


def _yuv420_clip():
    return vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(CLIP_PATH), bits=32, fulls=True, fulld=True)


def test_bm3dv2_color_parallel_load_matches_serial():
    """Three live entries must survive the deep parallel pipeline."""
    clip = _yuv420_clip()
    kwargs = dict(sigma=0.7, radius=2, bm_range=2, ps_range=1, block_step=4)
    par = eval_parallel(BM3D, clip, plane=1, **kwargs)
    ref = BM3D(clip, **kwargs)
    for n in range(clip.num_frames):
        d = par[n] - plane_to_ndarray(ref.get_frame(n), 1)
        assert np.abs(d).max() < 1e-5, f"parallel/serial chroma mismatch at frame {n}"


def test_bm3dv2_color_parallel_load_repeated():
    """The per-entry reservation is atomic, so concurrent neighbours must drain.

    Reserving one entry at a time deadlocked roughly 40% of concurrent
    three-entry runs (see notes/BM3D.md): a frame could hold entry 0's slots
    while waiting for entry 1's, against the mirror image. Each round is a fresh
    node under eval_parallel's worker timeout, so a regression fails with a
    stuck-worker report instead of hanging the suite.
    """
    clip = _yuv420_clip().std.CropAbs(width=320, height=180)
    kwargs = dict(sigma=0.7, radius=2, bm_range=2, ps_range=1, block_step=4)
    for round_ in range(3):
        frames = eval_parallel(BM3D, clip, plane=2, timeout=120.0, **kwargs)
        assert all(np.isfinite(f).all() for f in frames), f"non-finite round {round_}"


def test_bm3dv2_color_deterministic():
    """Two runs must agree per plane (the atomic-order floor, ~3e-8)."""
    clip = _yuv420_clip()
    kwargs = dict(sigma=0.7, radius=2, bm_range=2, ps_range=1, block_step=4)
    a, b = BM3D(clip, **kwargs), BM3D(clip, **kwargs)
    for n in (0, 11, 23):
        for plane in range(3):
            d = plane_to_ndarray(a.get_frame(n), plane) - plane_to_ndarray(b.get_frame(n), plane)
            assert np.abs(d).max() < 1e-5, f"nondeterministic plane {plane} at frame {n}"


@pytest.mark.parametrize("clip", ["yuv420_32", "yuv444_32", "rgb32"])
def test_bm3dv2_color_frame_request_order_matches_serial(clip):
    """Every request order must reproduce the serial run on a color clip."""
    assert_temporal_order_consistent(
        "BM3Dv2",
        {"sigma": 0.7, "radius": 2, "bm_range": 2, "ps_range": 1, "block_step": 4},
        tol=1e-5,
        nframes=12,
        clip=clip,
        timeout=300,
    )


def test_bm3dv2_joint_frame_request_order_matches_serial():
    """The joint entry's three planes share one slot and one witness table."""
    assert_temporal_order_consistent(
        "BM3Dv2",
        {
            "sigma": [0.7, 0.7, 0.7],
            "radius": 2,
            "bm_range": 2,
            "ps_range": 1,
            "block_step": 4,
            "chroma": 1,
        },
        tol=1e-5,
        nframes=12,
        clip="yuv444_32",
        timeout=300,
    )
