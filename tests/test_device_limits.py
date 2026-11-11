"""Device capability checks: workgroup memory and the VRAM budget.

The plugin requires Vulkan 1.4, which guarantees 256 invocations per compute
workgroup (Roapmap 2022: `maxComputeWorkGroupInvocations` 256 and
`maxComputeWorkGroupSize` 256x256x64, where core Vulkan only requires 128), but
only 16 KiB of workgroup memory, while several kernels ask for up to 18 KiB.
Every pipeline is therefore measured against the device before it is created,
and a device that cannot host one gets an error naming the kernel rather than a
failure out of vkCreateComputePipelines.

The allocation targets that *are* tuning (EEDI3's batch, NLMeans' u4a ring) are
the values measured on the RX 7900 XTX, each capped by the core's VRAM eviction
limit, so a device with a smaller allowance gets smaller allocations instead of
the tuned ones.

No GPU in the test box is that small, so the checks are exercised by forcing the
device limits down through the knobs the plugin reads (VSFEEL_LIMIT_SHARED_MEMORY,
VSFEEL_LIMIT_STORAGE_RANGE, VSFEEL_LIMIT_VRAM_BUDGET). Those are read once per
process when the core's device is brought up, so each case runs in its own
subprocess. `VSFEEL_LIMIT_INVOCATIONS` is not one of them: the 256-invocation
floor is a Vulkan 1.4 guarantee, so a smaller value would test a device no
supported driver reports.

Run from the repository root:  tools/test.sh tests/test_device_limits.py -q
"""

import json
import os
import re
import subprocess
import sys

import pytest

from conftest import COMPARE_PRELUDE, CLIP_PATH, run_compare_subprocess

# Every filter the limit cases build, including the ones a 16 KiB or a
# 128-invocation device can still host.
_LIMITS_SCRIPT = (
    COMPARE_PRELUDE
    + r"""
import hashlib
import json
import sys

import vapoursynth as vs

core = vs.core
spec = json.loads(sys.argv[1])
src = core.bs.VideoSource(spec["source"])
clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                          bits=32, fulls=True, fulld=True)
clip_large = core.resize.Bicubic(clip, width=1920, height=1080,
                                 format=vs.GRAYS)
clip_4k = core.resize.Bicubic(clip, width=3840, height=2160,
                              format=vs.GRAYS)
clip16 = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                            bits=16, fulls=True, fulld=True)


def build(name, params):
    if name == "nnedi3":
        return core.vsfeel.NNEDI3(clip, field=1, **params)
    if name == "nnedi3_large":
        return core.vsfeel.NNEDI3(clip_large, field=1, **params)
    if name == "eedi3":
        return core.vsfeel.EEDI3(clip, field=1, **params)
    if name == "eedi3h":
        return core.vsfeel.EEDI3H(clip, field=1, **params)
    if name == "eedi3aa":
        # AA runs a vertical and a transposed horizontal pass; both have their own
        # plane geometry and therefore their own folded grids.
        return core.vsfeel.EEDI3AA(clip, field=2, **params)
    if name == "bm3d":
        return core.vsfeel.BM3D(clip, **params)
    if name == "dfttest":
        return core.vsfeel.DFTTest(clip, **params)
    if name == "nlmeans":
        return core.vsfeel.NLMeans(clip, **params)
    if name == "nlmeans_4k":
        return core.vsfeel.NLMeans(clip_4k, **params)
    if name == "bilateral":
        kwargs = {"sigma_spatial": 6.0, "sigma_color": 0.1}
        kwargs.update(params)
        return core.vsfeel.Bilateral(clip, **kwargs)
    if name == "gaussblur":
        return core.vsfeel.GaussBlur(clip, sigma=2.0)
    if name == "wnnm_u16":
        return core.vsfeel.WNNM(clip16, sigma=3.0, radius=0)
    if name == "wnnm":
        return core.vsfeel.WNNM(clip, sigma=3.0, radius=0, **params)
    if name == "box":
        return core.vsfeel.Box(clip)
    if name == "box_down":
        # A real two-pass downscale (the shared clip is 640x360): the shape test
        # needs both kernels.
        return core.vsfeel.Box(clip, width=320, height=180)
    raise SystemExit("unknown filter %r" % name)


out = {}
for case in spec["cases"]:
    label = case["label"]
    try:
        node = cpu_node(build(case["filter"], case.get("params") or {}))
        frame = node.get_frame(spec.get("frame", 2))
        dtype = np.uint16 if frame.format.sample_type == vs.INTEGER else np.float32
        plane = read_plane(frame, 0, dtype)
        out[label] = {"sha1": hashlib.sha1(plane.tobytes()).hexdigest(),
                      "finite": bool(np.isfinite(plane.astype(np.float64)).all())}
    except Exception as exc:
        out[label] = {"error": "%s: %s" % (type(exc).__name__, exc)}
print("REF ok", flush=True)
print("RESULT " + json.dumps(out), flush=True)
"""
)

# Every filter the default-shape smoke test builds: one per entry point, with
# the formats each supports.
_DEFAULT_CASES = [
    "dfttest",
    "nlmeans",
    "eedi3",
    "eedi3h",
    "eedi3aa",
    "nnedi3",
    "bm3d",
    "wnnm_u16",
    "gaussblur",
    "bilateral",
]


def _run_limits(cases, env=None):
    """Run every case in a fresh process, optionally under a limit override."""
    spec = {"source": CLIP_PATH, "cases": cases}
    return run_compare_subprocess(_LIMITS_SCRIPT, [json.dumps(spec)], env=env or {})


def test_default_shapes_are_used_when_the_device_allows_them():
    """Every filter must create and produce a frame with no limit forced.

    `VSFEEL_LIMIT_INVOCATIONS` no longer has a case here: the plugin needs
    Vulkan 1.4, which guarantees the 256-invocation workgroups the remaining
    kernels use, so faking a 128-invocation device would only test a
    configuration no supported driver reports.
    """
    cases = [{"label": name, "filter": name} for name in _DEFAULT_CASES]
    res = _run_limits(cases)
    for case in cases:
        assert res[case["label"]].get("finite") is True, res[case["label"]]


def test_shared_memory_limit_selects_the_small_predict_tile():
    """A 16 KiB device runs the PXP=4 predictor from its 12 KiB tile.

    The predict module is chosen by window size: `n4` (288 vec4 rows, 18 KiB)
    covers every window and `n4m` (192 rows, 12 KiB) all but the 48x6 network.
    A 12288-byte limit is the sharp one: `n4` cannot be created under it at all,
    so the FS=96 case only survives if the host picked `n4m` -- and its pixels
    have to stay bit-identical to the unconstrained run. FS=288 has no smaller
    tile, so the same device must report 18432 against 16384.
    """
    wide = {"label": "fs96", "filter": "nnedi3", "params": {"nsize": 1, "nns": 3}}
    huge = {"label": "fs288", "filter": "nnedi3", "params": {"nsize": 3, "nns": 3}}
    cases = [wide, huge]
    default = _run_limits(cases)
    small = _run_limits(cases, {"VSFEEL_LIMIT_SHARED_MEMORY": "12288"})

    assert default["fs96"].get("finite") is True, default["fs96"]
    assert default["fs288"].get("finite") is True, default["fs288"]
    assert small["fs96"].get("finite") is True, small["fs96"]
    assert small["fs96"]["sha1"] == default["fs96"]["sha1"], "tile variant changed pixels"

    err = small["fs288"].get("error", "")
    assert "18432" in err and "12288" in err, small["fs288"]


# ---------------------------------------------------------------------------
# Tuned allocation targets vs the VRAM budget
# ---------------------------------------------------------------------------
#
# EEDI3's batch and NLMeans' u4a ring are sized from targets measured on the
# RX 7900 XTX, and each is capped by the core's VRAM eviction limit. Both print
# what they chose under their own trace flag, so the cap is checked by running
# the same creation twice: once on the device's real budget, once with the knob
# forcing a smaller one. Creation only -- no frame is evaluated.

_BUDGET_SCRIPT = r"""
import sys

import vapoursynth as vs

core = vs.core
if sys.argv[1] == "eedi3":
    # 1080p with the graded mdis: one frame's scratch is tens of MiB, so a
    # forced budget smaller than that has to change the batch.
    clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAYS, length=3)
    core.vsfeel.EEDI3(clip, field=1, mdis=20)
elif sys.argv[1] == "eedi3_4k":
    # The geometry the batch knees were swept on: 2x2160p, luma only.
    clip = core.std.BlankClip(width=3840, height=2160, format=vs.GRAY16, length=3)
    core.vsfeel.EEDI3(clip, field=1, mdis=20)
elif sys.argv[1] == "eedi3aa":
    clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAYS, length=3)
    core.vsfeel.EEDI3AA(clip, field=3, mdis=20)
elif sys.argv[1] == "eedi3aa_4k":
    clip = core.std.BlankClip(width=3840, height=2160, format=vs.GRAY16, length=3)
    core.vsfeel.EEDI3AA(clip, field=3, mdis=20)
elif sys.argv[1] == "dfttest":
    # 1080p: the ring is capped by the frame's block rows at the default budget,
    # so the plan can only change if the budget does.
    clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAYS, length=3)
    core.vsfeel.DFTTest(clip)
else:
    clip = core.std.BlankClip(width=640, height=360, format=vs.GRAYS, length=3)
    core.vsfeel.NLMeans(clip, d=3)
"""


def _trace(which, env=None):
    """stderr of a creation run, with that filter's trace flag turned on."""
    if which in ("eedi3", "eedi3_4k", "eedi3aa", "eedi3aa_4k"):
        trace_flag = "VSFEEL_EEDI3_TRACE"
    elif which == "dfttest":
        trace_flag = "VSFEEL_DFTTEST_TRACE"
    else:
        trace_flag = "VSFEEL_NLMEANS_VRAM"
    run_env = {**os.environ, "MANGOHUD": "0", trace_flag: "1", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _BUDGET_SCRIPT, which],
        capture_output=True,
        text=True,
        timeout=600,
        env=run_env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stderr


def _planned_batch(cap, scratch):
    """The plan from the trace's own numbers: what the cap holds, up to eight."""
    return max(1, min(8, int(cap // scratch)))


def _eedi3_batch(stderr):
    """(cap MiB, scratch MiB, batch, cap source) from the batch plan line."""
    plan = re.search(
        r"batch cap=([\d.]+) MiB \(([^)]+)\) scratch=([\d.]+) MiB batch=(\d+)",
        stderr,
    )
    assert plan, stderr[-2000:]
    return (
        float(plan.group(1)),
        float(plan.group(3)),
        int(plan.group(4)),
        plan.group(2),
    )


def _nlmeans_ring(stderr):
    """(ring MiB, ring target MiB, batch) from the VRAM banner."""
    m = re.search(r"ring ([\d.]+) MiB of ([\d.]+) MiB target, \d+ slots, batch (\d+)", stderr)
    assert m, stderr[-2000:]
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def _nlmeans_llc(stderr):
    """(ring MiB, last-level cache MiB) from the VRAM banner."""
    ring = re.search(r"ring ([\d.]+) MiB", stderr)
    llc = re.search(r"cache ([\d.]+) MiB", stderr)
    assert ring and llc, stderr[-2000:]
    return float(ring.group(1)), float(llc.group(1))


def _dft_plan(stderr):
    """(target MiB, ring block rows, stripes) from the creation plan lines."""
    b = re.search(
        r"stripe budget ([\d.]+) MiB \(tuned ([\d.]+), llc ([\d.]+), "
        r"vram/16 ([\d.]+), max buffer ([\d.]+)",
        stderr,
    )
    p = re.search(
        r"target ([\d.]+) MiB, ring (\d+) of \d+ block rows "
        r"\([\d.]+ MiB\), (\d+) stripes",
        stderr,
    )
    assert b and p, stderr[-2000:]
    return float(p.group(1)), int(p.group(2)), int(p.group(3)), float(b.group(3))


def test_dfttest_stripe_budget_is_device_sized():
    """The stripe ring is sized from the part's cache, bounded by its VRAM.

    Nothing in Vulkan reports a cache size, so the target comes from a table of
    published last-level cache sizes keyed by PCI IDs, capped at the tuned 64
    MiB and floored at 32: the measured stripe ladder on the 96 MiB part
    (64/32/24/16/8 MiB -> 2068/1832/1484/1481/1412 fps at 1080p) says the stripe
    count dominates below 32, whatever the cache does. An unknown part gets the
    floor. The plan only partitions the dispatch, so the frame must not move.
    """
    target, ring, stripes, llc = _dft_plan(_trace("dfttest"))
    if llc == 0.0:
        assert target == 32.0, target
    elif llc >= 64.0:
        assert target == 64.0, (llc, target)
    else:
        assert target == llc, (llc, target)
    assert 32.0 <= target <= 64.0, target

    # The knob overrides the table: a 32 MiB budget halves the ring.
    small_target, small_ring, small_stripes, _ = _dft_plan(
        _trace("dfttest", {"VSFEEL_DFTTEST_L2_TARGET": str(32 << 20)})
    )
    assert small_target == 32.0, small_target
    assert small_ring * 2 == ring, (ring, small_ring)
    assert small_stripes > stripes, (stripes, small_stripes)

    # The VRAM allowance bounds it too: a 256 MiB allowance is 16 MiB of ring.
    vram_target, vram_ring, vram_stripes, _ = _dft_plan(
        _trace("dfttest", {"VSFEEL_LIMIT_VRAM_BUDGET": str(256 << 20)})
    )
    assert vram_target == 16.0, vram_target
    assert vram_ring < small_ring, (small_ring, vram_ring)
    assert vram_stripes > small_stripes, (small_stripes, vram_stripes)

    # A 640x360 plane is one ring under any of those, so use an 8 MiB target
    # there to force a second partition, and compare the frame.
    case = {"label": "dfttest", "filter": "dfttest"}
    default = _run_limits([case])
    small = _run_limits([case], {"VSFEEL_DFTTEST_L2_TARGET": str(8 << 20)})
    assert default["dfttest"].get("finite") is True, default["dfttest"]
    assert small["dfttest"].get("finite") is True, small["dfttest"]
    assert small["dfttest"]["sha1"] == default["dfttest"]["sha1"], (
        "the stripe budget changed pixels"
    )


def test_dfttest_ring_fits_a_small_storage_range():
    """A device whose buffer range is below the tuned plan runs a smaller ring.

    The ring is one buffer, so the budget is clamped by maxStorageBufferRange:
    before that clamp a 32 MiB-range device rejected the whole filter with
    "reduce the frame size" even though a 5-stripe plan fits.
    """
    case = {"label": "dfttest", "filter": "dfttest"}
    default = _run_limits([case])
    tight = _run_limits([case], {"VSFEEL_LIMIT_STORAGE_RANGE": str(32 << 20)})
    assert tight["dfttest"].get("finite") is True, tight["dfttest"]
    assert tight["dfttest"]["sha1"] == default["dfttest"]["sha1"], (
        "the range-clamped plan changed pixels"
    )


def test_eedi3_batch_is_capped_by_the_vram_budget():
    """The batch is what the VRAM cap holds, never more, and one when nothing fits.

    The plan used to be a byte target (256 MiB) floored at two frames, which
    planned two frames at 2160p and six at 1080p f32; both are slower than eight
    (46% and 6%, measured on this build). A cap that cannot host two frames must
    still run one frame per submission rather than overshoot.
    """
    cap, scratch, batch, source = _eedi3_batch(_trace("eedi3"))
    assert source == "vram/12", source
    assert batch == _planned_batch(cap, scratch), (batch, cap, scratch)
    assert batch == 8, (batch, cap, scratch)

    # Half a frame of cap: one frame per submission, not two.
    tiny = _eedi3_batch(_trace("eedi3", {"VSFEEL_EEDI3_SCRATCH_MB": str(int(scratch * 1.5))}))
    assert tiny[2] == 1, tiny
    assert tiny[0] == int(scratch * 1.5), tiny


@pytest.mark.parametrize("which", ["eedi3_4k", "eedi3aa_4k"], ids=["eedi3", "eedi3aa"])
def test_batch_plan_follows_a_ten_gib_allowance(which):
    """A small allowance plans the frames its cap holds, not the eight clamp.

    The cap is at most a twelfth of the allowance, so a 10 GiB device holds seven
    2160p EEDI3 frames (107 MiB each) but only five 2160p EEDI3AA ones (162.4
    MiB): the plan has to follow the scratch, which is what a fixed target could
    not do. The core's own limit also moves with memory pressure, so the cap is
    an upper bound rather than an equality.
    """
    cap, scratch, batch, source = _eedi3_batch(
        _trace(which, {"VSFEEL_LIMIT_VRAM_BUDGET": str(10 << 30)})
    )
    assert source == "vram/12", source
    assert 0 < cap <= 10240 / 12 + 0.1, cap
    assert batch == _planned_batch(cap, scratch), (batch, cap, scratch)
    assert batch < 8, (batch, cap, scratch)


def test_eedi3aa_batch_scales_with_its_own_scratch():
    """EEDI3AA plans from its own frame size, not the single-stage one.

    AA's four sub-passes make its 2160p scratch 1.5x the vertical filter's, so at
    the same geometry and cap it must plan fewer frames. A plan that reused
    EEDI3's numbers would land on the same count here, which is the regression
    the old untested AA branch allowed.
    """
    e_cap, e_scratch, e_batch, _ = _eedi3_batch(_trace("eedi3_4k"))
    a_cap, a_scratch, a_batch, _ = _eedi3_batch(_trace("eedi3aa_4k"))
    assert a_scratch > e_scratch, (a_scratch, e_scratch)
    assert a_batch == _planned_batch(a_cap, a_scratch), (a_batch, a_cap, a_scratch)
    assert a_batch < e_batch, (a_batch, e_batch, a_scratch, e_scratch)


def test_eedi3_scratch_budget_knob_moves_the_plan():
    """VSFEEL_EEDI3_SCRATCH_MB overrides the batch's scratch cap.

    The plan's only input is the cap, so the knob is the cap: a 100 MiB budget
    plans two of the 1080p f32 frames (36.8 MiB each) and 50 MiB plans one, which
    is how a sweep asks what a smaller device would choose.
    """
    cap, scratch, batch, _ = _eedi3_batch(_trace("eedi3"))
    assert batch == 8, (batch, cap, scratch)

    cap, scratch, batch, source = _eedi3_batch(_trace("eedi3", {"VSFEEL_EEDI3_SCRATCH_MB": "100"}))
    assert source == "knob", source
    assert cap == 100.0, cap
    assert batch == _planned_batch(cap, scratch), (batch, cap, scratch)
    assert batch == 2, (batch, cap, scratch)

    _, _, lowered, _ = _eedi3_batch(_trace("eedi3", {"VSFEEL_EEDI3_SCRATCH_MB": "50"}))
    assert lowered == 1, lowered


def test_nlmeans_round_target_follows_the_cache():
    """NLMeans' per-round live set is sized from the part's cache.

    The tuned 96 MiB is the target card's last-level cache; a part with a
    smaller one plans a smaller round instead of streaming it, floored at 32 MiB
    like DFTTest's stripe ring. The knob overrides the tuned value in MiB and the
    VRAM allowance still caps the result.
    """
    trace = _trace("nlmeans")
    ring, target, batch = _nlmeans_ring(trace)
    _, llc = _nlmeans_llc(trace)
    want = 96.0 if llc == 0.0 or llc >= 96.0 else max(32.0, llc)
    assert target == want, (llc, target)
    assert batch >= 1, batch
    # The banner names every per-frame buffer and the cache it used.
    for field in ("u2", "u5", "addr", "tables", "cache"):
        assert field in trace, trace[-2000:]
    # The knob is the override, and the VRAM cap still applies after it.
    _, forced, _ = _nlmeans_ring(_trace("nlmeans", {"VSFEEL_NLMEANS_RING_TARGET": "48"}))
    assert forced == 48.0, forced
    _, capped, _ = _nlmeans_ring(_trace("nlmeans", {"VSFEEL_LIMIT_VRAM_BUDGET": str(256 << 20)}))
    assert capped == 16.0, capped


def test_nlmeans_live_set_is_bounded_at_creation():
    """One round must fit the per-frame allowance, and a supported case still runs.

    4K f32 `d=2` is the case the check must not reject: its window alone is
    159 MiB of a 317 MiB live set, far past the 96 MiB cache target, so the
    comparison is against an eighth of the VRAM allowance. A 64 MiB allowance
    cannot host even one round of the 360p case and has to say so at creation.
    """
    large = {"label": "nlmeans4k", "filter": "nlmeans_4k", "params": {"d": 2}}
    supported = _run_limits([large])
    assert supported["nlmeans4k"].get("finite") is True, supported["nlmeans4k"]

    case = {"label": "live", "filter": "nlmeans", "params": {"d": 3}}
    res = _run_limits([case], {"VSFEEL_LIMIT_VRAM_BUDGET": str(64 << 20)})
    err = res["live"].get("error", "")
    assert "one weight+acc round needs" in err and "eighth" in err, res["live"]


def test_nlmeans_sweep_tables_are_budgeted():
    """The m=0..d sweep tables are budgeted, not allocated at any size.

    At `a=64, d=16` they are 215.8 MiB (142.4 wq + 73.4 aq) while one variant is
    read per frame; the budget is a thirty-second of the VRAM allowance capped
    at 64 MiB. `a=64` with a small `d` still creates (the suite renders it).
    """
    created = _run_limits([{"label": "small", "filter": "nlmeans", "params": {"a": 64}}])
    assert created["small"].get("finite") is True, created["small"]

    case = {"label": "wide", "filter": "nlmeans", "params": {"a": 64, "d": 16}}
    res = _run_limits([case], {"VSFEEL_LIMIT_VRAM_BUDGET": str(1 << 30)})
    err = res["wide"].get("error", "")
    assert "sweep tables need" in err and "215.8" in err and "32.0 MiB" in err, res["wide"]


def test_nlmeans_ring_is_capped_by_the_vram_budget():
    """The 96 MiB live-set target is a target, not a claim on the device."""
    ring, target_mib, batch = _nlmeans_ring(_trace("nlmeans"))
    if batch == 1:
        pytest.skip("device budget is already at the one-entry ring floor")
    assert ring <= target_mib + 1e-6

    # A quarter of the target must plan a smaller round.
    ring2, target2, batch2 = _nlmeans_ring(
        _trace("nlmeans", {"VSFEEL_LIMIT_VRAM_BUDGET": str(int(target_mib / 4 * 16) << 20)})
    )
    assert batch2 < batch, (batch, batch2, ring, target_mib)
    # One entry per round is the floor, so a target smaller than one entry's
    # ring is not expressible; above it the ring never exceeds the target.
    assert ring2 <= max(target2, ring2 / batch2) + 1e-6


def test_nlmeans_pack_is_capped_by_the_grid_z_limit():
    """The weight dispatch's Z is a round's row count: one or two per entry.

    A round of ``batch`` sweep entries dispatches one workgroup per table row
    and every entry with a non-zero temporal offset emits two, so ``batch`` has
    to yield to ``maxComputeWorkGroupCount[2]`` rather than dispatch an invalid
    grid. The default already fits the guaranteed 65535; the forced limit here
    (640x360, d=3) is well below it, so the banner must show the clamp and the
    frame must still evaluate.
    """
    rows_per_entry = 2  # d=3: every kk != 0 entry emits a +q and a -q row
    z_limit = 16

    _, _, batch = _nlmeans_ring(_trace("nlmeans"))
    assert batch * rows_per_entry <= 65535, batch

    force = {"VSFEEL_LIMIT_GRID_Z": str(z_limit), "VSFEEL_NLMEANS_PACK": "16384"}
    _, _, capped = _nlmeans_ring(_trace("nlmeans", force))
    assert capped == z_limit // rows_per_entry, capped

    # Creation alone cannot see an over-limit dispatch; the frame run must.
    res = _run_limits([{"label": "nlmeans", "filter": "nlmeans"}], force)
    assert res["nlmeans"].get("finite") is True, res["nlmeans"]


def test_bilateral_auto_lds_gate_falls_back_to_the_plain_kernel():
    """The automatic LDS gate must drop to the plain kernel when even the
    smallest shared tile is over budget, exactly as an explicit
    use_shared_memory=False does.

    sigma_spatial=8 derives radius 24. The auto 32x16 block gives a
    (2*24+2*32)*(2*24+16)*4 = 28672 B two-column tile and a 20480 B one-column
    one; the shape search's floor is 4x8, whose one-column tile is
    (48+4)*(48+8)*4 = 11648 B. An 8192-byte limit admits none of them, so both
    runs below use the plain kernel and must agree bit-for-bit.
    """
    params = {"sigma_spatial": 8.0, "sigma_color": 0.15}
    auto = _run_limits(
        [{"label": "auto", "filter": "bilateral", "params": params}],
        {"VSFEEL_LIMIT_SHARED_MEMORY": "8192"},
    )
    plain = _run_limits(
        [
            {
                "label": "plain",
                "filter": "bilateral",
                "params": {**params, "use_shared_memory": False},
            }
        ],
        {"VSFEEL_LIMIT_SHARED_MEMORY": "8192"},
    )
    assert auto["auto"].get("finite") is True, auto["auto"]
    assert plain["plain"].get("finite") is True, plain["plain"]
    assert auto["auto"]["sha1"] == plain["plain"]["sha1"], (
        "the auto LDS gate did not select the plain kernel"
    )

    # Anti-vacuity: on the unconstrained device the same config takes the
    # shared kernel, whose wide-radius output differs from the plain one, so a
    # forced limit that changed nothing would leave these hashes equal.
    shared = _run_limits([{"label": "shared", "filter": "bilateral", "params": params}])
    assert shared["shared"].get("finite") is True, shared["shared"]
    assert shared["shared"]["sha1"] != plain["plain"]["sha1"], (
        "the forced LDS limit did not change the selected kernel"
    )


_BILATERAL_LDS_SCRIPT = r"""
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=640, height=360, format=vs.GRAYS, length=3)
core.vsfeel.Bilateral(clip, sigma_spatial=8.0, sigma_color=0.15).get_frame(0)
"""


def _bilateral_lds(env=None):
    """The `lds=` of every bilateral pipeline that creation created.

    One pipeline per unique plane config, so the single number here names the
    kernel the gate selected: the two-column shared tile, the one-column one,
    or 0 for the plain gather.
    """
    run_env = {**os.environ, "MANGOHUD": "0", "VSFEEL_DEBUG": "1", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _BILATERAL_LDS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        env=run_env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return [int(m) for m in re.findall(r"pipeline bilateral .* lds=(\d+)", proc.stderr)]


def test_bilateral_lds_gate_has_four_ordered_tiers():
    """R=24 with the auto 32x16 block pins every tier the gate can select.

    The two-column shared tile is 28672 B, the one-column one 20480 B, and the
    shape search's 4x8 floor gives 11648 B, so limits between them walk the
    tiers down instead of jumping to the ~4x slower plain kernel. Only a limit
    under the floor reaches plain: the tier no longer depends on vszipcl's own
    budget, since the clipped-border build serves the planes it filters plain.
    The pipeline's `lds=` is where the choice is observable, and the two-column
    kernel is worth ~16% (measured), so a silent regression to the fallback must
    fail here rather than only show up in the benchmark.
    """
    assert _bilateral_lds() == [28672], "the two-column tile is not the default"
    assert _bilateral_lds({"VSFEEL_LIMIT_SHARED_MEMORY": "24576"}) == [20480], (
        "a limit over the one-column tile but under the two-column one must "
        "select the one-column shared kernel"
    )
    assert _bilateral_lds({"VSFEEL_LIMIT_SHARED_MEMORY": "12288"}) == [11648], (
        "a limit under the auto shapes must halve the block to 4x8, not go plain"
    )
    assert _bilateral_lds({"VSFEEL_LIMIT_SHARED_MEMORY": "8192"}) == [0], (
        "a budget under every shared tile must select the plain kernel"
    )


# ---------------------------------------------------------------------------
# GaussBlur: the fused path's tile vs the guaranteed workgroup memory
# ---------------------------------------------------------------------------

_GAUSS_SCRIPT = r"""
import sys

import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=640, height=360, format=vs.GRAYS, length=3)
core.vsfeel.GaussBlur(clip, sigma=float(sys.argv[1]))
"""


def _gauss_lds(sigma, env=None):
    """The `lds=` of every gaussblur pipeline that creation created.

    The fused small path makes one pipeline carrying the tile; the two-pass path
    makes two (vert + horiz) with no shared memory at all, so the numbers say
    which path each config took.
    """
    run_env = {**os.environ, "MANGOHUD": "0", "VSFEEL_DEBUG": "1", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _GAUSS_SCRIPT, str(sigma)],
        capture_output=True,
        text=True,
        timeout=600,
        env=run_env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return [int(m) for m in re.findall(r"pipeline gaussblur .* lds=(\d+)", proc.stderr)]


def test_gaussblur_small_path_fits_the_guaranteed_workgroup_memory():
    """The fused path's largest tile is 1 728 B, far under the 16 KiB floor.

    The tile is `VRT*BLK_Y*(BLK_X + 2*RAD)` floats and `RAD` is capped by
    LARGE_THRESHOLD, so the bound follows the measured threshold: at radius 3
    with the shipped 64x2 block the largest is 3*2*(64+6)*4 = 1 680 B. Pinned so
    a change to either constant fails loudly (`gaussblur.cpp`'s static_assert
    catches it at build time as well), and so the LDS number stays a way to tell
    which path a config took.
    """
    floor = {"VSFEEL_LIMIT_SHARED_MEMORY": "16384"}  # the Vulkan minimum
    lds = [v for sigma in (0.5, 1.0, 1.2) for v in _gauss_lds(sigma, floor)]
    assert 1680 in lds, lds  # radius 3, the largest the small path asks for
    assert max(lds) <= 16384, lds  # and it fits the floor
    # Sigma 1.4 is radius 4, already two-pass: it creates two pipelines
    # (vert + horiz) and uses no shared memory, so the numbers say which path a
    # config took.
    assert _gauss_lds(1.4, floor) == [0, 0]


# ---------------------------------------------------------------------------
# Dispatch grids and resource limits
# ---------------------------------------------------------------------------


def test_dispatch_grids_fold_into_y_when_x_is_small():
    """A grid wider than the device's X count must still cover every element.

    Vulkan guarantees only 65535 workgroups per dimension, and this box reports
    exactly that in Y while X is 2^32-1; the kernels fold a 1D grid into
    `ID.y * NumWorkGroups.x + ID.x` and the host (or the compaction, for the
    indirect predictor) sizes X and Y to match. Forcing X to 4 exercises both
    paths; anything the fold drops is a pixel the kernel never writes, so the
    digests have to match the unconstrained run exactly.
    """
    cases = [
        {"label": "nnedi3-direct", "filter": "nnedi3", "params": {"pscrn": 0}},
        {"label": "nnedi3-indirect", "filter": "nnedi3", "params": {"pscrn": 2}},
        {"label": "eedi3", "filter": "eedi3"},
        {"label": "eedi3h", "filter": "eedi3h"},
        {"label": "eedi3aa", "filter": "eedi3aa"},
    ]
    default = _run_limits(cases)
    folded = _run_limits(cases, {"VSFEEL_LIMIT_GRID_X": "4"})
    for case in cases:
        label = case["label"]
        assert default[label].get("finite") is True, default[label]
        assert folded[label].get("finite") is True, folded[label]
        assert folded[label]["sha1"] == default[label]["sha1"], label


def test_dispatch_grid_overflow_is_reported():
    """An element count that cannot fit X and Y must fail naming the kernel."""
    cases = [
        {"label": "nnedi3", "filter": "nnedi3", "params": {"pscrn": 0}},
        {"label": "eedi3", "filter": "eedi3"},
    ]
    res = _run_limits(cases, {"VSFEEL_LIMIT_GRID_X": "1", "VSFEEL_LIMIT_GRID_Y": "1"})
    for case in cases:
        err = res[case["label"]].get("error", "")
        assert "workgroups" in err and "1x1" in err, res[case["label"]]


def test_storage_buffer_range_limit_is_reported():
    """Every binding is VK_WHOLE_SIZE, so maxStorageBufferRange caps the buffer.

    BM3D's estimate cache is the filter that reaches it first (190 MiB at 1080p,
    over the 128 MiB core minimum), so a small forced range must come back as the
    buffer's size against the limit rather than a driver failure.
    """
    res = _run_limits(
        [{"label": "bm3d", "filter": "bm3d"}], {"VSFEEL_LIMIT_STORAGE_RANGE": str(1 << 20)}
    )
    err = res["bm3d"].get("error", "")
    assert "maxStorageBufferRange" in err and str(1 << 20) in err, res["bm3d"]


def test_dfttest_scratch_range_is_a_creation_error():
    """The padded window and block ring are checked before the first frame.

    gpu_make_buffer already rejects an oversized per-frame buffer, but that
    used to happen on every frame; creation must name the plane and the limit.
    """
    res = _run_limits(
        [{"label": "dfttest", "filter": "dfttest"}],
        {"VSFEEL_LIMIT_STORAGE_RANGE": str(1 << 20)},
    )
    err = res["dfttest"].get("error", "")
    assert "padded window" in err and "maxStorageBufferRange" in err, res["dfttest"]
    assert str(1 << 20) in err, res["dfttest"]


_BOX_SHAPE_SCRIPT = r"""
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAY16, length=2)
core.vsfeel.Box(clip, width=960, height=540)
"""


def _box_shape(env=None):
    """(returncode, stderr) of a Box creation with the banner on."""
    run_env = {**os.environ, "MANGOHUD": "0", "VSFEEL_DEBUG": "1", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _BOX_SHAPE_SCRIPT],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=300,
    )
    return proc.returncode, proc.stderr


def test_box_shape_follows_the_device():
    """Box runs 64x4 where the device allows it and 16x8 where it cannot.

    The shape is a measured choice: 64x4 is +1-2% over 16x8 at 1080p->540p and
    4K->1080p, and 32x8 was measured flat. 16x8 is the shape the
    128-invocation floor allows, so the fallback keeps a small device running
    instead of failing creation. Invisible in the pixels, so the banner names
    the choice and the sha1 below pins that it changes nothing.
    """
    code, stderr = _box_shape()
    assert code == 0 and "shape=64x4" in stderr, stderr[-2000:]

    low = {"VSFEEL_LIMIT_INVOCATIONS": "128"}
    code, stderr = _box_shape(low)
    assert code == 0 and "shape=16x8" in stderr, stderr[-2000:]

    # Forcing the wide shape on a 128-invocation device is refused by name.
    code, stderr = _box_shape({"VSFEEL_BOX_BLK_X": "64", **low})
    assert code != 0 and "cannot launch a 64x4 workgroup" in stderr, stderr[-2000:]

    case = [{"label": "box", "filter": "box_down"}]
    wide = _run_limits(case)
    narrow = _run_limits(case, {"VSFEEL_BOX_BLK_X": "16"})
    assert wide["box"].get("finite") is True, wide["box"]
    assert wide["box"]["sha1"] == narrow["box"]["sha1"], "the shape changed pixels"


def test_nnedi3_scratch_range_is_a_creation_error():
    """The list/indirect scratch is checked before the first frame.

    A 1080p field needs ~3.95 MiB; 2 MiB leaves the weight buffers (about
    1.2 MiB) room and makes the per-frame scratch the first thing over the
    limit.
    """
    res = _run_limits(
        [{"label": "nnedi3", "filter": "nnedi3_large"}],
        {"VSFEEL_LIMIT_STORAGE_RANGE": str(2 << 20)},
    )
    err = res["nnedi3"].get("error", "")
    assert "per-frame scratch" in err and "maxStorageBufferRange" in err, res["nnedi3"]
    assert str(2 << 20) in err, res["nnedi3"]


def test_nnedi3_scratch_is_budgeted_against_the_vram_allowance():
    """The list scratch must also fit an eighth of the VRAM allowance.

    1080p f32 at field=1 is a 540-row field: 1920 * 540 * 4 = 4.05 MiB of list
    per in-flight frame, which the eighth of a 24 MiB allowance (3 MiB) cannot
    host. The range check passes there, so this is the allowance's own error and
    the same clip still renders at the device's real allowance.
    """
    case = [{"label": "nnedi3", "filter": "nnedi3_large"}]
    res = _run_limits(case, {"VSFEEL_LIMIT_VRAM_BUDGET": str(24 << 20)})
    err = res["nnedi3"].get("error", "")
    assert "per-frame scratch" in err and "VRAM allowance" in err, res["nnedi3"]
    assert "eighth" in err, res["nnedi3"]

    ok = _run_limits(case)
    assert ok["nnedi3"].get("finite") is True, ok["nnedi3"]


def test_nnedi3_output_plane_range_is_a_creation_error():
    """A doubled output plane no buffer can hold fails on the constructor.

    1080p f32 at field=1 writes a 1920x1080 output (8.29 MB) from a 4.05 MB
    scratch, so a 6 MiB range accepts the weights and the scratch and rejects
    the plane the predictor would write: the alternative is an allocation error
    on every get_frame.
    """
    res = _run_limits(
        [{"label": "nnedi3", "filter": "nnedi3_large"}],
        {"VSFEEL_LIMIT_STORAGE_RANGE": str(6 << 20)},
    )
    err = res["nnedi3"].get("error", "")
    assert "output" in err and "maxStorageBufferRange" in err, res["nnedi3"]
    assert str(6 << 20) in err, res["nnedi3"]


# ---------------------------------------------------------------------------
# Per-frame scratch vs maxStorageBufferRange at creation
# ---------------------------------------------------------------------------
#
# These filters size one scratch buffer per frame. A device whose storage range
# cannot bind it used to create the filter and then fail every get_frame; the
# cases below only build, so the error has to come from the constructor.

_SCRATCH_RANGE_SCRIPT = r"""
import json
import sys

import vapoursynth as vs

core = vs.core


def build(case):
    clip = core.std.BlankClip(
        width=case["w"], height=case["h"], format=getattr(vs, case["format"]),
        length=1,
    )
    kwargs = dict(case.get("kwargs") or {})
    name = case["filter"]
    if name == "eedi3":
        core.vsfeel.EEDI3(clip, **kwargs)
    elif name == "eedi3aa":
        core.vsfeel.EEDI3AA(clip, **kwargs)
    elif name == "gaussblur":
        core.vsfeel.GaussBlur(clip, **kwargs)
    elif name == "nlmeans":
        core.vsfeel.NLMeans(clip, **kwargs)
    elif name == "box":
        core.vsfeel.Box(clip, **kwargs)
    else:
        raise SystemExit("unknown filter %r" % name)


out = {}
for case in json.loads(sys.argv[1])["cases"]:
    try:
        build(case)
        out[case["label"]] = {"result": "created"}
    except Exception as exc:
        out[case["label"]] = {"error": "%s: %s" % (type(exc).__name__, exc)}
print("RESULT " + json.dumps(out))
"""


def _scratch_range(cases, env=None):
    """Create every case in a fresh process; return {label: result/error}."""
    run_env = {**os.environ, "MANGOHUD": "0", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _SCRATCH_RANGE_SCRIPT, json.dumps({"cases": cases})],
        capture_output=True,
        text=True,
        timeout=600,
        env=run_env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(proc.stdout[-2000:])


def test_eedi3_family_scratch_range_is_a_creation_error():
    """A 4K scratch over the storage range is refused by the constructor.

    EEDI3's 4K f32 scratch (146.7 MiB) and EEDI3AA's 4K u16 scratch (162.4 MiB)
    exceed the 128 MiB conformant range; EEDI3's 4K u16 scratch (107.0 MiB) must
    still create, or the check would reject what the device can run.
    """
    cases = [
        {
            "label": "eedi3-f32",
            "filter": "eedi3",
            "w": 3840,
            "h": 2160,
            "format": "GRAYS",
            "kwargs": {"field": 1, "mdis": 20},
        },
        {
            "label": "eedi3-u16",
            "filter": "eedi3",
            "w": 3840,
            "h": 2160,
            "format": "GRAY16",
            "kwargs": {"field": 1, "mdis": 20},
        },
        {
            "label": "eedi3aa-u16",
            "filter": "eedi3aa",
            "w": 3840,
            "h": 2160,
            "format": "GRAY16",
            "kwargs": {"field": 3, "mdis": 20},
        },
    ]
    res = _scratch_range(cases, {"VSFEEL_LIMIT_STORAGE_RANGE": str(128 << 20)})
    assert res["eedi3-u16"].get("result") == "created", res["eedi3-u16"]
    for label, size in (("eedi3-f32", 153847872), ("eedi3aa-u16", 170317952)):
        err = res[label].get("error", "")
        assert "per-frame scratch" in err, res[label]
        assert str(128 << 20) in err and str(size) in err, res[label]


def test_gaussblur_scratch_range_is_a_creation_error():
    """sigma=20 takes the two-pass path: the scratch is a creation error.

    The 640x360 u16 plane (460 800 B) fits a 512 KiB range and the 921 600-byte
    two-pass scratch does not.
    """
    res = _scratch_range(
        [
            {
                "label": "gaussblur",
                "filter": "gaussblur",
                "w": 640,
                "h": 360,
                "format": "GRAY16",
                "kwargs": {"sigma": 20.0},
            }
        ],
        {"VSFEEL_LIMIT_STORAGE_RANGE": str(512 << 10)},
    )
    err = res["gaussblur"].get("error", "")
    assert "two-pass scratch" in err and "921600" in err, res["gaussblur"]
    assert str(512 << 10) in err, res["gaussblur"]


def test_box_scratch_range_is_a_creation_error():
    """The packed RGB scratch is checked at creation, not on every frame.

    A 640x360 RGBS -> 320x180 box packs 1 382 400 B; the 921 600-byte source
    plane fits a 1 MiB range and the scratch does not.
    """
    res = _scratch_range(
        [
            {
                "label": "box",
                "filter": "box",
                "w": 640,
                "h": 360,
                "format": "RGBS",
                "kwargs": {"width": 320, "height": 180},
            }
        ],
        {"VSFEEL_LIMIT_STORAGE_RANGE": str(1 << 20)},
    )
    err = res["box"].get("error", "")
    assert "two-pass scratch" in err and "1382400" in err, res["box"]
    assert str(1 << 20) in err, res["box"]


def test_nlmeans_scratch_range_is_a_creation_error():
    """The weight ring is refused at creation, with the ring size named.

    A 60 MB range passes the 921 600-byte plane and fails the 92 160 000-byte
    ring; the pack knob pins the ring so the case does not depend on the
    device's VRAM allowance.
    """
    res = _scratch_range(
        [
            {
                "label": "nlmeans",
                "filter": "nlmeans",
                "w": 640,
                "h": 360,
                "format": "GRAYS",
                "kwargs": {"d": 2},
            }
        ],
        {"VSFEEL_LIMIT_STORAGE_RANGE": "60000000", "VSFEEL_NLMEANS_PACK": "50"},
    )
    err = res["nlmeans"].get("error", "")
    assert "weight ring" in err and "92160000" in err, res["nlmeans"]
    assert "60000000" in err, res["nlmeans"]


def test_nlmeans_plane_range_error_names_the_uv_plane():
    """channels="UV" checks plane 1, so the error must say so."""
    res = _scratch_range(
        [
            {
                "label": "uv",
                "filter": "nlmeans",
                "w": 640,
                "h": 360,
                "format": "YUV420PS",
                "kwargs": {"d": 2, "channels": "UV"},
            }
        ],
        {"VSFEEL_LIMIT_STORAGE_RANGE": "200000"},
    )
    err = res["uv"].get("error", "")
    assert "plane 1" in err and "200000" in err, res["uv"]


def test_wnnm_select_and_estimate_grid_overflow_is_reported():
    """The 1D WNNM phases cannot fold into Y, so creation must reject them.

    With X forced to 4 the group grid still folds, but the select/Gram/apply
    grids do not; leaving them to dispatch would silently leave work unwritten.
    """
    res = _run_limits(
        [{"label": "wnnm", "filter": "wnnm"}],
        {"VSFEEL_LIMIT_GRID_X": "4"},
    )
    err = res["wnnm"].get("error", "")
    assert "WNNM select" in err and "workgroups" in err, res["wnnm"]


def test_push_descriptor_limit_is_reported():
    """The binding count is bounded by maxPushDescriptors, checked explicitly."""
    res = _run_limits(
        [{"label": "nlmeans", "filter": "nlmeans"}], {"VSFEEL_LIMIT_PUSH_DESCRIPTORS": "4"}
    )
    err = res["nlmeans"].get("error", "")
    assert "push descriptors" in err and "4" in err, res["nlmeans"]


def test_bilateral_clamps_an_oversized_explicit_block():
    """A block over the device's invocation budget must clamp, not error.

    The clamp keeps every dimension the device accepts and halves the larger
    axis until the product fits, so a request no device can launch still
    creates and renders (it lands on 32x32 on the target). 64x64 is 4096
    invocations, over the target's 1024 budget and over the 256 Vulkan 1.4
    guarantees.
    The block shape is a tiling choice, so its pixels must equal the auto
    shape's; the old fallback replaced both dimensions with 16x16 and lost the
    requested one.
    """
    cases = [
        {
            "label": "explicit",
            "filter": "bilateral",
            "params": {"block_x": 64, "block_y": 64},
        },
        {"label": "auto", "filter": "bilateral"},
    ]
    res = _run_limits(cases)
    assert res["auto"].get("finite") is True, res["auto"]
    out = res["explicit"]
    assert out.get("finite") is True, out
    assert out["sha1"] == res["auto"]["sha1"], "block shape changed pixels"


def test_two_dimensional_grid_overflow_is_reported():
    """A 2D tiling kernel cannot fold an over-wide plane into Y.

    Bilateral, GaussBlur, Box and DFTTest address their output as (ID.x,
    ID.y), so a width past the device's X group limit has nowhere to go: it
    must fail with the grid and the limits rather than dispatch a clamped grid
    that leaves the tail unwritten.
    """
    cases = [
        {"label": "bilateral", "filter": "bilateral"},
        {"label": "gaussblur", "filter": "gaussblur"},
        {"label": "box", "filter": "box"},
        {"label": "dfttest", "filter": "dfttest"},
    ]
    res = _run_limits(cases, {"VSFEEL_LIMIT_GRID_X": "4"})
    for case in cases:
        err = res[case["label"]].get("error", "")
        assert "workgroup grid" in err and "4x" in err, res[case["label"]]

    # The same config must still evaluate when the device advertises enough X.
    default = _run_limits([{"label": "dfttest", "filter": "dfttest"}])
    assert default["dfttest"].get("finite") is True, default["dfttest"]


def test_subgroup_size_requests_require_compute_stage_support():
    """A required-size mask without COMPUTE must degrade, not break.

    Requesting requiredSubgroupSize for a stage outside
    requiredSubgroupSizeStages is invalid usage, so with the mask cleared the
    filters whose default width already fits (DFTTest, NLMeans, BM3D) must run
    unchanged, and the two that need exactly 32 lanes (NNEDI3, EEDI3) must say
    so rather than hand the driver an invalid required size.
    """
    labels = ["dfttest", "nlmeans", "bm3d"]
    cases = [{"label": name, "filter": name} for name in labels]
    default = _run_limits(cases)
    cleared = _run_limits(cases, {"VSFEEL_LIMIT_SUBGROUP_STAGES": "0"})
    for label in labels:
        assert default[label].get("finite") is True, default[label]
        assert cleared[label].get("finite") is True, cleared[label]
    # BM3D aggregates with atomicAdd, so its last bits differ run to run and a
    # digest cannot be compared across processes; the other two are exact.
    for label in ("dfttest", "nlmeans"):
        assert cleared[label]["sha1"] == default[label]["sha1"], label

    strict = [{"label": "nnedi3", "filter": "nnedi3"}, {"label": "eedi3", "filter": "eedi3"}]
    res = _run_limits(strict, {"VSFEEL_LIMIT_SUBGROUP_STAGES": "0"})
    for case in strict:
        err = res[case["label"]].get("error", "")
        assert "32-lane subgroups" in err, res[case["label"]]
