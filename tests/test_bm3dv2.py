"""Unit tests for core.vsfeel.BM3Dv2.

The committed tests/noise_24f.mkv clip (24 frames of random noise) is used as
the input. The noise content exposes the boundary/clamping behaviour of the
temporal pipeline: frames 0..23 must all produce finite output with no NaN
(regression: boundary frames intermittently produced NaN before the atomics
were made visible to the aggregation kernel).

Run from the repository root:  uv run python -m pytest tests/test_bm3dv2.py
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    NOISE_MKV,
    COMPARE_PRELUDE,
    ReferenceUnavailable,
    assert_preserves_frame_props,
    assert_temporal_order_consistent,
    check_all_frames_finite,
    eval_parallel,
    frame_to_ndarray,
    plane_to_ndarray,
    run_compare_subprocess,
    skip_or_fail_reference,
    cpu_node,
)


def BM3D(*args, **kwargs):
    """BM3Dv2 as a clip the test can read pixels from (see conftest.cpu_node)."""
    return cpu_node(vs.core.vsfeel.BM3Dv2(*args, **kwargs))


pytestmark = pytest.mark.usefixtures("noise_gray")

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


def test_bm3dv2_parallel_load_matches_serial(noise_gray):
    """Parallel request load must produce the same pixel
    values as the serial path.

    This is the request pattern that exposed stale descriptor bindings, fence
    misuse and command pool reuse violations under load on strict drivers
    (black or garbage output only when frames are processed concurrently).
    """
    par = eval_parallel(_run, noise_gray, radius=2)
    ref = _run(noise_gray, radius=2)
    for n in range(noise_gray.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"parallel/serial mismatch at frame {n}"


def test_bm3dv2_wide_radius_parallel_load_matches_serial(noise_gray):
    """The widest window must survive the deep pipeline like the narrow ones.

    At radius 16 one output frame spans 33 estimate slots and 65 source
    frames, i.e. the instance's whole cache working set at once; a parallel
    request load is what exercises that ring under contention.
    """
    par = eval_parallel(_run, noise_gray, radius=16)
    ref = _run(noise_gray, radius=16)
    for n in range(noise_gray.num_frames):
        d = par[n] - frame_to_ndarray(ref.get_frame(n))
        assert np.abs(d).max() < 1e-5, f"parallel/serial mismatch at frame {n}"


def test_bm3dv2_parallel_load_deterministic(noise_gray):
    """Two parallel runs must produce identical output.

    A fence attached to two in-flight submissions makes frame results depend
    on the completion order of the streams, which only shows up when many
    frames are in flight at once.
    """
    a = eval_parallel(_run, noise_gray, radius=2)
    b = eval_parallel(_run, noise_gray, radius=2)
    for n in range(noise_gray.num_frames):
        d = a[n] - b[n]
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 16])
def test_bm3dv2_no_nan_all_frames(noise_gray, radius):
    """Output must be finite on every frame (incl. boundaries) for each radius.

    A brand new filter instance is created per parametrized test.
    """
    check_all_frames_finite(_run, noise_gray, radius=radius)


def test_bm3dv2_deterministic(noise_gray):
    a = _run(noise_gray, radius=2)
    b = _run(noise_gray, radius=2)
    for n in (0, 11, 23):
        d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
        # BM3D's aggregation uses float atomics, so two runs differ only by
        # atomic-order rounding (~2e-8 measured); the 1e-5 bound is the
        # established self-consistency bound.
        assert np.abs(d).max() < 1e-5, f"nondeterministic output at frame {n}"


def test_bm3dv2_cas_fallback_matches_hardware_atomics(noise_gray, monkeypatch):
    """The CAS arm a device without float32 add atomics gets must match.

    `VSFEEL_BM3D_CAS=1` forces the `-DNO_FLOAT_ATOMICS` kernel the host selects
    when the device has no `shaderBufferFloat32AtomicAdd`; the two arms add in
    a different order, so this is also what bounds that order's effect.
    """
    monkeypatch.delenv("VSFEEL_BM3D_CAS", raising=False)
    hardware = _run(noise_gray, radius=2)
    monkeypatch.setenv("VSFEEL_BM3D_CAS", "1")
    cas = _run(noise_gray, radius=2)
    for n in (0, 11, 23):
        a = frame_to_ndarray(hardware.get_frame(n))
        b = frame_to_ndarray(cas.get_frame(n))
        assert np.isfinite(b).all(), f"non-finite CAS output at frame {n}"
        # measured 2.98e-8 (add-order rounding) on the noise clip
        assert np.abs(a - b).max() < 1e-5, f"CAS vs hardware atomics at frame {n}"


def test_bm3dv2_cas_fallback_holds_at_small_block_step(noise_gray, monkeypatch):
    """The CAS arm must stay exact where contention is highest.

    One res element receives up to `8 * (ceil((2*bm_range + 8)/block_step) +
    1)^2` adds -- every matched patch of every reference block whose search
    window can reach it -- so block_step=1 is the worst case (13448 at
    bm_range=16) and a retry budget below it can silently drop addends: the
    old 32-retry bound put 16 of 4096 pixels 1e-5..6.1e-5 out against the
    hardware arm, while the geometry-derived bound lands at 7e-9 (float add
    order alone). The bound is what this pins.
    """

    def small_step(clip):
        return BM3D(clip, sigma=SIGMA, radius=2, bm_range=BM_RANGE, ps_range=PS_RANGE, block_step=1)

    monkeypatch.setenv("VSFEEL_BM3D_CAS", "1")
    cas = small_step(noise_gray)
    monkeypatch.delenv("VSFEEL_BM3D_CAS", raising=False)
    hardware = small_step(noise_gray)
    for n in (0, 11):
        a = frame_to_ndarray(hardware.get_frame(n))
        b = frame_to_ndarray(cas.get_frame(n))
        assert np.isfinite(b).all(), f"non-finite CAS output at frame {n}"
        assert np.abs(a - b).max() < 1e-7, (
            f"CAS lost an addend at block_step=1, frame {n}: {np.abs(a - b).max():g}"
        )


def test_bm3dv2_disjoint_first_estimates_keep_slice_witnesses(noise_gray):
    """Concurrent opposite-end windows must not clear each other's tags."""
    import threading

    clip = noise_gray.std.CropAbs(width=64, height=64)
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


def test_bm3dv2_radius0_concurrent_first_use_and_fallback(noise_gray, monkeypatch):
    """Concurrent first requests must preserve private tags and fallback input."""
    monkeypatch.setenv("VSFEEL_BM3D_NOESTIMATE", "1")
    actual = eval_parallel(_run, noise_gray, radius=0)
    for n in range(noise_gray.num_frames):
        expected = frame_to_ndarray(noise_gray.get_frame(n))
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
from conftest import NOISE_MKV, frame_to_ndarray
import numpy as np
import vapoursynth as vs
from vapoursynth import core


def source():
    if hasattr(core, "bs"):
        return core.bs.VideoSource(NOISE_MKV)
    return core.ffms2.Source(NOISE_MKV)


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


def test_bm3dv2_rejects_gray8(noise_8bit):
    with pytest.raises(vs.Error):
        _run(noise_8bit)


def test_bm3dv2_rejects_radius17(noise_gray):
    """Radius > 16 is unsupported and must be rejected up front."""
    with pytest.raises(vs.Error):
        _run(noise_gray, radius=17)


@pytest.mark.parametrize("radius", [15, 16])
def test_bm3dv2_accepts_reference_radius_cap(noise_gray, radius):
    """The radius cap must match the references' (vszipcl 16, bm3dvk 15).

    Windows this wide are the end-to-end case for the aggregation's derived
    per-slice table, which replaced the push-constant one.
    """
    out = _run(noise_gray, radius=radius)
    for n in (0, 23):
        a = frame_to_ndarray(out.get_frame(n))
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


def test_bm3dv2_rejects_bad_ref_format(noise_gray):
    """A \"ref\" with a different format/size must be rejected up front."""
    bad = noise_gray.std.AddBorders(right=1)
    with pytest.raises(vs.Error):
        _run(noise_gray, ref=bad)


def test_bm3dv2_ref_final_pass(noise_gray):
    """A basic estimate passed as \"ref\" drives the final (Wiener) pass.

    The final output must be finite on every frame, deterministic between two
    sequential instances, and differ from the basic estimate
    (the Wiener refinement changes the pixels rather than copying them).
    """
    basic = _run(noise_gray, radius=2)
    a = _run(noise_gray, radius=2, ref=basic)
    b = _run(noise_gray, radius=2, ref=basic)
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


def test_bm3dv2_device_id(noise_gray):
    """device_id and num_streams are accepted no-ops.

    Under the R80 GPU API the core owns the one Vulkan device per process and
    sizes in-flight depth itself, so both arguments are registered only so
    existing scripts keep loading. Any value, including a negative one, must be
    ignored and produce the default result (measured max diff ~2e-8, the
    atomic-order run-to-run floor).
    """
    b = _run(noise_gray)
    for kwargs in (
        dict(device_id=0),
        dict(device_id=-1),
        dict(device_id=99),
        dict(num_streams=0),
        dict(num_streams=64),
    ):
        a = _run(noise_gray, **kwargs)
        for n in (0, 11, 23):
            d = frame_to_ndarray(a.get_frame(n)) - frame_to_ndarray(b.get_frame(n))
            assert np.abs(d).max() < 1e-5, f"{kwargs} differs from default at frame {n}"


def test_bm3dv2_preserves_gray_frame_props(noise_gray):
    """R13: the grayscale path must keep the source frame's properties.

    ``_ColorRange`` is deprecated in VapourSynth R79 and is remapped by
    ``SetFrameProps``; the surviving equivalent key ``_Range`` is tagged
    instead. ``MyTag`` verifies arbitrary application metadata.
    """
    assert_preserves_frame_props(_run, noise_gray, radius=2)


def test_bm3dv2_chroma_planes_are_denoised(noise_gray):
    """A YUV clip is denoised on every plane, luma exactly as the Gray path.

    The references denoise each plane independently by default (per-plane
    geometry, parameters and sigma), so a plane left equal to the source is the
    old luma-only passthrough, not a denoised result.
    """
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True)
    assert yuv.format.color_family == vs.YUV

    out = _run(yuv, radius=2)
    ref = _run(noise_gray, radius=2)

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
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True)
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
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True)
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
    yuv = vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True)
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
        vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True
    )
    with pytest.raises(vs.Error, match="YUV444"):
        vs.core.vsfeel.BM3Dv2(yuv420, sigma=0.7, chroma=1)
    rgb = vs.core.resize.Bicubic(
        vs.core.bs.VideoSource(NOISE_MKV), format=vs.RGBS, matrix_in_s="709"
    )
    with pytest.raises(vs.Error, match="YUV444"):
        vs.core.vsfeel.BM3Dv2(rgb, sigma=0.7, chroma=1)


def test_bm3dv2_joint_chroma_denoises_every_plane():
    """chroma=True packs the three 4:4:4 planes into one entry; a sigma-zero
    plane of that entry is skipped and comes from the source."""
    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(NOISE_MKV), format=vs.YUV444PS)
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
    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(NOISE_MKV), format=vs.YUV444PS)
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
        vs.core.bs.VideoSource(NOISE_MKV), format=vs.RGBS, matrix_in_s="709"
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
        vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True
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
    yuv444 = vs.core.resize.Bicubic(vs.core.bs.VideoSource(NOISE_MKV), format=vs.YUV444PS)
    out = BM3D(yuv444, radius=1, bm_range=2, ps_range=1, block_step=4, **kwargs)
    for n in (0, 5):
        for plane in range(3):
            assert np.isfinite(plane_to_ndarray(out.get_frame(n), plane)).all()


@pytest.mark.parametrize("sigma", [0.0, 1e-9])
@pytest.mark.parametrize("use_ref", [False, True])
def test_bm3dv2_sigma_below_epsilon_passes_through(noise_gray, sigma, use_ref):
    """A plane whose sigma is below FLT_EPSILON is a bit-exact source copy,
    as in the reference's PROC_MASK. Covers the old sigma=0 + ref 0/0 NaN."""
    basic = None
    if use_ref:
        basic = BM3D(
            noise_gray,
            sigma=SIGMA,
            radius=2,
            bm_range=BM_RANGE,
            ps_range=PS_RANGE,
            block_step=BLOCK_STEP,
        )
        _ = frame_to_ndarray(basic.get_frame(0))
    out = BM3D(
        noise_gray,
        sigma=sigma,
        radius=2,
        bm_range=BM_RANGE,
        ps_range=PS_RANGE,
        block_step=BLOCK_STEP,
        **({"ref": basic} if use_ref else {}),
    )
    for n in (0, 11, 23):
        a = frame_to_ndarray(out.get_frame(n))
        b = frame_to_ndarray(noise_gray.get_frame(n))
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
    src = core.bs.VideoSource({NOISE_MKV!r})
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
    spec = json.loads(sys.argv[4])
    kind = spec.get("clip", "gray32")
    frames = spec.get("frames", [0, 11, 23])

    def vsfeel(clip, **kw):
        # The host cannot read a GPU-resident frame's pixels, and this script
        # reads them directly, so the node is downloaded here.
        node = core.vsfeel.BM3Dv2(clip, **kw)
        return core.std.GPUDownload(clip=node) if node.gpu_resident else node

    core.max_cache_size = 1024 * 56
    src = core.bs.VideoSource({NOISE_MKV!r})
    if kind == "gray32":
        clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY), bits=32, fulls=True, fulld=True)
    elif kind == "yuv420_32":
        clip = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    elif kind == "yuv444_32":
        clip = core.resize.Bicubic(src, format=vs.YUV444PS)
    elif kind == "rgb32":
        clip = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")
    else:
        raise SystemExit("bad clip kind %r" % kind)

    planes = list(range(clip.format.num_planes))

    def read_all(node, n):
        frame = node.get_frame(n)
        return [read_plane(frame, p, np.float32) for p in planes]

    basic = None
    if ref_pass:
        # The reference consumes vsfeel's basic estimate as its guide. Force
        # that clip here so a vsfeel failure is not mislabelled as a missing
        # reference; the frames stay cached for the reference below.
        try:
            basic = vsfeel(clip, **kwargs)
            _ = [read_all(basic, n) for n in frames]
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
        ref_frames = [read_all(ref_node, n) for n in frames]
    except Exception as exc:
        print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
        raise SystemExit(2)
    print("REF ok", flush=True)

    # --- vsfeel phase ---
    my_node = (vsfeel(clip, ref=basic, **kwargs) if ref_pass
               else vsfeel(clip, **kwargs))
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
    print("RESULT " + json.dumps({{"maxdiff": worst, "per_plane": per_plane}}), flush=True)
""")

BASE_KWARGS = dict(sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4)


def _max_diff_vs_reference(
    kwargs: dict,
    ref: str,
    ref_pass: bool = False,
    clip: str = "gray32",
    frames=(0, 11, 23),
    timeout: float = 600,
) -> dict:
    """Run the comparison in a subprocess; returns ``maxdiff``/``per_plane``.

    Raises :class:`ReferenceUnavailable` when ``ref`` is missing/failed and
    :class:`AssertionError` (with the captured tail) when vsfeel failed.
    """
    spec = {"clip": clip, "frames": list(frames)}
    return run_compare_subprocess(
        _COMPARE_SCRIPT,
        [ref, json.dumps(kwargs), str(int(ref_pass)), json.dumps(spec)],
        timeout=timeout,
    )


def _compare_against_any_reference(
    kwargs: dict, ref_pass: bool = False, clip: str = "gray32"
) -> tuple[str, dict]:
    """Try vszipcl, then bm3dhip; skip only if none is usable.

    ``clip`` names the input the comparison runs on (see the compare script);
    the color kinds grade the chroma planes as well as luma.
    """
    reasons = []
    for ref in ("vszipcl", "bm3dhip"):
        if not hasattr(vs.core, ref) or not hasattr(getattr(vs.core, ref), "BM3Dv2"):
            reasons.append(f"{ref}: not installed")
            continue
        try:
            return ref, _max_diff_vs_reference(kwargs, ref, ref_pass=ref_pass, clip=clip)
        except ReferenceUnavailable as exc:
            reasons.append(f"{ref}: {exc}")
    skip_or_fail_reference("no usable reference plugin (vszipcl/bm3dhip): " + "; ".join(reasons))


def _compare_against_bm3dvk(kwargs: dict, ref_pass: bool = False, clip: str = "yuv444_32"):
    """bm3dvk first (the preferred reference where the two disagree), then the
    others through the shared helper."""
    if hasattr(vs.core, "bm3dvk") and hasattr(vs.core.bm3dvk, "BM3Dv2"):
        try:
            return "bm3dvk", _max_diff_vs_reference(kwargs, "bm3dvk", ref_pass=ref_pass, clip=clip)
        except ReferenceUnavailable:
            pass
    return _compare_against_any_reference(kwargs, ref_pass=ref_pass, clip=clip)


def test_bm3dv2_ref_matches_reference(noise_gray):
    """The final (Wiener) pass must closely match the reference implementations
    when given the same basic-estimate ref clip."""
    ref, payload = _compare_against_any_reference(dict(BASE_KWARGS), ref_pass=True)
    assert payload["maxdiff"] < 0.01, f"max diff vs {ref} (ref pass): {payload}"


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
#
# Filled-in ranges (same oracle/session): block_step 1..8 worst 0.0132,
# ps_num 1..8 0.0124, bm_range 1/4/32 0.0087, extractor_exp 3/8 0.0111,
# radius=1 0.0074. The 0.02 bound covers the loosest with ~50% headroom.
SWEEP_CONFIGS = [
    ({}, 0.01),
    ({"sigma": 0.3}, 0.01),
    ({"sigma": 1.5}, 0.03),
    ({"block_step": 2}, 0.01),
    ({"block_step": 1}, 0.01),
    ({"block_step": 3}, 0.01),
    ({"block_step": 5}, 0.02),
    ({"block_step": 6}, 0.02),
    ({"block_step": 7}, 0.02),
    ({"block_step": 8}, 0.02),
    ({"bm_range": 9}, 0.01),
    ({"bm_range": 22}, 0.01),
    ({"bm_range": 1}, 0.01),
    ({"bm_range": 4}, 0.01),
    ({"bm_range": 32}, 0.01),
    ({"ps_range": 5}, 0.01),
    ({"ps_range": 9}, 0.01),
    ({"ps_num": 5}, 0.02),
    ({"ps_num": 1}, 0.02),
    ({"ps_num": 2}, 0.02),
    ({"ps_num": 3}, 0.02),
    ({"ps_num": 4}, 0.02),
    ({"ps_num": 6}, 0.02),
    ({"ps_num": 7}, 0.02),
    ({"ps_num": 8}, 0.02),
    ({"extractor_exp": 6}, 0.01),
    ({"extractor_exp": 3}, 0.02),
    ({"extractor_exp": 8}, 0.02),
    ({"radius": 1}, 0.01),
]


def _sweep_id(cfg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in cfg.items()) or "empty+BASE_KWARGS"


@pytest.mark.parametrize("cfg,tol", SWEEP_CONFIGS, ids=[_sweep_id(c) for c, _ in SWEEP_CONFIGS])
def test_bm3dv2_parameter_sweep_matches_reference(noise_gray, cfg, tol):
    """Parameter grid around the defaults must track the reference (basic
    estimate pass)."""
    ref, payload = _compare_against_any_reference(dict(BASE_KWARGS, **cfg))
    assert payload["maxdiff"] < tol, f"max diff vs {ref} ({_sweep_id(cfg)}): {payload}"


# The plugin's own defaults (sigma=3.0, radius=0, bm_range=9, block_step=8,
# ps_range=4, ps_num=2, extractor_exp=0) are not what BASE_KWARGS selects, so
# the sweep's empty entry grades a different configuration. Grade the true
# defaults directly -- never merged with BASE_KWARGS -- plus the short-array
# inheritance case (a one-element sigma makes the other planes inherit
# element 0). Measured vszipcl basic estimate, frames 0/11/23: defaults 0.0188,
# sigma=[0.7] 0.0071.
@pytest.mark.parametrize(
    "kwargs,tol",
    [
        ({}, 0.03),
        ({"sigma": [0.7]}, 0.02),
    ],
    ids=["plugin-defaults", "sigma-array-inherit"],
)
def test_bm3dv2_plugin_defaults_match_reference(noise_gray, kwargs, tol):
    """The documented defaults must produce the reference result, not just
    any finite frame (test_device_limits/test_lifecycle only check those)."""
    ref, payload = _compare_against_any_reference(kwargs)
    assert payload["maxdiff"] < tol, f"max diff vs {ref} ({kwargs}): {payload}"


# Remeasured with the repaired oracle (vszipcl, basic estimate, radius sweep):
# radius 0/1/2/3/4 = 0.00708/0.00742/0.00786/0.00285/0.00300; the Wiener ref
# pass = 0.00273. The 0.01 bound covers all of them with margin. Radius 5 and 6
# are the first windows wider than the aggregation's push-constant table, i.e.
# the derived arm's reference comparison; the references' own cap (16) is
# covered by the self-consistency tests instead, because at that window the
# reference's fused accumulator cache alone wants gigabytes.
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 5, 6])
def test_bm3dv2_matches_reference(noise_gray, radius):
    """vsfeel must closely match vszipcl, falling back to bm3dhip.

    The reference plugins are crash-prone on some drivers, so the comparison
    runs in a subprocess: a crashed reference must not take down the suite.
    bm3dhip implements the same algorithm and is expected to match vszipcl.
    """
    ref, payload = _compare_against_any_reference(dict(BASE_KWARGS, radius=radius))
    assert payload["maxdiff"] < 0.01, f"max diff vs {ref}: {payload}"


# ---------------------------------------------------------------------------
# extractor_exp: the documented ">= 3 = bitwise reproducible" claim
# ---------------------------------------------------------------------------
#
# The reference pre-rounds the atomic addends with `(x + E) - E`
# (kernel.cu:741-742), which makes the sums order-independent. EXTRACTOR is a
# spec constant, so RADV used to constant-fold the pair back to x and the
# parameter was a silent no-op (identical ISA for 0 and 20). bm3d.comp now
# computes it under `precise` inside `if (EXTRACTOR != 0.0)`; the default
# path's ISA is unchanged. After the fix, two fresh ns=1 runs are
# bit-identical at 3/6/8 (3.7e-8 at 0) and the output tracks the reference.


@pytest.mark.parametrize("extractor_exp", [3, 6, 8])
def test_bm3dv2_extractor_exp_is_bit_reproducible(noise_gray, extractor_exp):
    """Two runs at extractor_exp >= 3 must be identical (reference: exact)."""
    a = _run(noise_gray, radius=2, extractor_exp=extractor_exp)
    b = _run(noise_gray, radius=2, extractor_exp=extractor_exp)
    for n in (0, 11, 23):
        fa = frame_to_ndarray(a.get_frame(n))
        fb = frame_to_ndarray(b.get_frame(n))
        assert np.array_equal(fa, fb), (
            f"extractor_exp={extractor_exp} not reproducible at frame {n}"
        )


def test_bm3dv2_extractor_exp_changes_aggregation(noise_gray):
    """A 2^20 extractor quantises the atomic addends to a 0.125 grid, which
    must move the output (the reference goes non-finite at 2^20). An output
    identical to extractor_exp=0 means the constant is folded away again.

    Finiteness is asserted before the movement: an all-NaN output also has a
    NaN max diff, so `not isfinite or moved` would accept the very regression
    this test exists to catch.
    """
    base = _run(noise_gray, radius=2, extractor_exp=0)
    coarse = _run(noise_gray, radius=2, extractor_exp=20)
    for n in (0, 11, 23):
        fb = frame_to_ndarray(base.get_frame(n))
        fc = frame_to_ndarray(coarse.get_frame(n))
        assert np.isfinite(fb).all() and np.isfinite(fc).all(), (
            f"non-finite extractor_exp output at frame {n}"
        )
        assert float(np.abs(fb - fc).max()) > 1e-4, f"extractor_exp=20 left frame {n} unchanged"


# ---------------------------------------------------------------------------
# Multi-plane reference comparison
# ---------------------------------------------------------------------------
#
# The default (per-plane) mode block-matches every plane on its own content at
# its own geometry; chroma=True is the references' joint entry, where one
# kernel filters all three 4:4:4 planes with the groups found on luma. Both are
# graded against bm3dvk (the preferred reference where the two disagree) with
# vszipcl/bm3dhip as the fallback. Tolerances are the same measurement-bounded
# kind as the luma sweep: the residual is a handful of block-match decisions
# that flip on rounding order, not a systematic difference.
#
# Measured per-plane maxima, bm3dvk, frames 0/11/23 (640x360 noise clip):
# yuv420 r2 0.0066/0.0068/0.0079, yuv444 r2 0.0070/0.0052/0.0056,
# rgb r2 0.0077/0.0071/0.0075, joint yuv444 r2 0.0070/0.0054/0.0056,
# joint r0 0.0061/2.4e-7/2.4e-7, per-plane params 0.0066/0.0164/0.0099,
# Wiener ref pass (joint) 0.0027/0.0016/0.0014. The 0.02 bound covers the
# loosest with headroom; the joint-r0 chroma planes agree at ulp level.
MULTIPLANE_CLIPS = [
    ("yuv420_32", 0.02),
    ("yuv444_32", 0.02),
    ("rgb32", 0.02),
]


@pytest.mark.parametrize("clip,tol", MULTIPLANE_CLIPS, ids=[c for c, _ in MULTIPLANE_CLIPS])
def test_bm3dv2_color_clip_matches_reference(clip, tol):
    """Every plane of a color clip must track the reference's per-plane mode."""
    ref, payload = _compare_against_bm3dvk(dict(BASE_KWARGS), clip=clip)
    assert payload["maxdiff"] < tol, f"max diff vs {ref} ({clip}): {payload}"


@pytest.mark.parametrize(
    "kwargs,tol",
    [
        ({"chroma": 1}, 0.02),
        ({"chroma": 1, "radius": 0}, 0.02),
        ({"chroma": 1, "radius": 1}, 0.02),
        ({"chroma": 1, "radius": 3}, 0.03),
        ({"chroma": 1, "sigma": [0.5, 1.1, 0.3]}, 0.03),
        ({"chroma": 1, "sigma": [0.7, 0.0, 0.9]}, 0.03),
        ({"chroma": 1, "sigma": [0.0, 0.7, 0.7]}, 0.02),
        ({"chroma": 1, "block_step": 1}, 0.03),
        ({"chroma": 1, "ps_num": 5}, 0.03),
        ({"chroma": 1, "extractor_exp": 6}, 0.03),
    ],
    ids=[
        "r2",
        "r0",
        "r1",
        "r3",
        "per-plane-sigma",
        "mixed-sigma",
        "luma-skipped",
        "block-step-1",
        "ps-num-5",
        "extractor-6",
    ],
)
def test_bm3dv2_joint_chroma_matches_reference(kwargs, tol):
    """chroma=True must reproduce the reference's joint (luma-grouped) entry."""
    ref, payload = _compare_against_bm3dvk(dict(BASE_KWARGS, **kwargs), clip="yuv444_32")
    assert payload["maxdiff"] < tol, f"max diff vs {ref} (chroma=1, {kwargs}): {payload}"


@pytest.mark.parametrize(
    "clip,kwargs",
    [
        ("yuv420_32", {}),
        ("yuv444_32", {}),
        ("rgb32", {}),
        (
            "yuv420_32",
            {
                "sigma": [0.7, 0.4, 1.2],
                "bm_range": [16, 4, 12],
                "ps_range": [7, 2, 6],
                "block_step": [4, 8, 6],
                "ps_num": [2, 3, 1],
            },
        ),
        ("yuv420_32", {"sigma": [0.7, 0.0, 0.0]}),
        ("yuv444_32", {"chroma": 1}),
    ],
    ids=["yuv420", "yuv444", "rgb", "per-plane-params", "mixed-sigma", "joint"],
)
def test_bm3dv2_color_ref_pass_matches_reference(clip, kwargs):
    """The Wiener pass must match on color clips too, both entry modes."""
    ref, payload = _compare_against_bm3dvk(dict(BASE_KWARGS, **kwargs), ref_pass=True, clip=clip)
    assert payload["maxdiff"] < 0.02, f"max diff vs {ref} ({clip}, {kwargs}, ref pass): {payload}"


def test_bm3dv2_per_plane_params_match_reference():
    """Per-plane parameter arrays must land on the reference's per-plane run."""
    kwargs = dict(
        sigma=[0.7, 0.4, 1.2],
        radius=2,
        bm_range=[16, 4, 12],
        ps_range=[7, 2, 6],
        block_step=[4, 8, 6],
        ps_num=[2, 3, 1],
    )
    ref, payload = _compare_against_bm3dvk(kwargs, clip="yuv420_32")
    assert payload["maxdiff"] < 0.03, f"max diff vs {ref} (per-plane params): {payload}"


# ---------------------------------------------------------------------------
# Multi-plane self-consistency
# ---------------------------------------------------------------------------
#
# The per-entry caches (source ring, estimate stacks, witnesses) are separate
# tables now, so the request-order and parallel-load oracles are re-run on a
# color clip where three entries are live at once.


def _yuv420_clip():
    return vs.core.fmtc.bitdepth(vs.core.bs.VideoSource(NOISE_MKV), bits=32, fulls=True, fulld=True)


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
