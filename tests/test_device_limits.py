"""Device capability checks: workgroup size, workgroup memory and VRAM budget.

Vulkan only guarantees 128 invocations per compute workgroup, 128 per dimension
and 16 KiB of workgroup memory, while several kernels here ask for 256
invocations or 18 KiB. Every pipeline is therefore measured against the device
before it is created, and a device that cannot host one gets an error naming the
kernel rather than a failure out of vkCreateComputePipelines.

The allocation targets that *are* tuning (EEDI3's batch, NLMeans' u4a ring) are
the values measured on the RX 7900 XTX, each capped by the core's VRAM eviction
limit, so a device with a smaller allowance gets smaller allocations instead of
the tuned ones.

No GPU in the test box is that small, so the checks are exercised by forcing the
device limits down through the knobs the plugin reads (VSFEEL_LIMIT_INVOCATIONS,
VSFEEL_LIMIT_SHARED_MEMORY, VSFEEL_LIMIT_VRAM_BUDGET). Those are read once per
process when the core's device is brought up, so each case runs in its own
subprocess.

Run from the repository root:  uv run python -m pytest tests/test_device_limits.py
"""

import json
import os
import re
import subprocess
import sys

import pytest

from conftest import COMPARE_PRELUDE, CLIP_PATH, run_compare_subprocess

# Every filter that needs a 256-invocation workgroup, plus one that fits 128.
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


def build(name, params):
    if name == "nnedi3":
        return core.vsfeel.NNEDI3(clip, field=1, **params)
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
    if name == "bilateral":
        kwargs = {"sigma_spatial": 6.0, "sigma_color": 0.1}
        kwargs.update(params)
        return core.vsfeel.Bilateral(clip, **kwargs)
    if name == "gaussblur":
        return core.vsfeel.GaussBlur(clip, sigma=2.0)
    raise SystemExit("unknown filter %r" % name)


out = {}
for case in spec["cases"]:
    label = case["label"]
    try:
        node = cpu_node(build(case["filter"], case.get("params") or {}))
        plane = read_plane(node.get_frame(spec.get("frame", 2)), 0, np.float32)
        out[label] = {"sha1": hashlib.sha1(plane.tobytes()).hexdigest(),
                      "finite": bool(np.isfinite(plane.astype(np.float64)).all())}
    except Exception as exc:
        out[label] = {"error": "%s: %s" % (type(exc).__name__, exc)}
print("REF ok", flush=True)
print("RESULT " + json.dumps(out), flush=True)
"""
)

# name -> kwargs; the empty ones just take the filter's defaults.
_256_KERNELS = ["dfttest", "nlmeans", "eedi3", "nnedi3", "bm3d"]


def _run_limits(cases, env=None):
    """Run every case in a fresh process, optionally under a limit override."""
    spec = {"source": CLIP_PATH, "cases": cases}
    return run_compare_subprocess(_LIMITS_SCRIPT, [json.dumps(spec)], env=env or {})


def test_workgroup_limit_is_reported_per_kernel():
    """A device that allows 128 invocations must name the kernel, not fail raw.

    Every message has to carry both numbers: what the kernel needs and what the
    device allows, since the whole point of the check is that the driver's own
    failure names neither.
    """
    cases = [{"label": name, "filter": name} for name in _256_KERNELS]
    cases.append({"label": "gaussblur", "filter": "gaussblur"})
    cases.append({"label": "bilateral", "filter": "bilateral"})
    res = _run_limits(cases, {"VSFEEL_LIMIT_INVOCATIONS": "128"})

    for name in _256_KERNELS:
        err = res[name].get("error", "")
        assert "256 invocations" in err and "128 invocations" in err, res[name]
    # These two fit 128: 16x8 grid-stride, and a block the host shrinks to 16x8.
    assert res["gaussblur"].get("finite") is True, res["gaussblur"]
    assert res["bilateral"].get("finite") is True, res["bilateral"]


def test_default_shapes_are_used_when_the_device_allows_them():
    """The same cases must run normally with no limit forced (anti-vacuity)."""
    cases = [{"label": name, "filter": name} for name in _256_KERNELS + ["gaussblur"]]
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
else:
    clip = core.std.BlankClip(width=640, height=360, format=vs.GRAYS, length=3)
    core.vsfeel.NLMeans(clip, d=3)
"""


def _trace(which, env=None):
    """stderr of a creation run, with that filter's trace flag turned on."""
    trace_flag = "VSFEEL_EEDI3_TRACE" if which == "eedi3" else "VSFEEL_NLMEANS_VRAM"
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


def _eedi3_batch(stderr):
    """(scratch MiB, batch, reported limit MiB) from the trace."""
    scratch = re.search(r"scratch=([\d.]+) MiB", stderr)
    batch = re.search(r"batch=(\d+)", stderr)
    limit = re.search(r"limit=(\d+) MiB", stderr)
    assert scratch and batch and limit, stderr[-2000:]
    return float(scratch.group(1)), int(batch.group(1)), float(limit.group(1))


def _nlmeans_ring(stderr):
    """(ring MiB, ring budget MiB, pack) from the VRAM banner."""
    m = re.search(r"ring ([\d.]+) MiB of ([\d.]+) MiB, \d+ slots, pack (\d+)", stderr)
    assert m, stderr[-2000:]
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def test_eedi3_batch_is_capped_by_the_vram_budget():
    """The tuned batch must yield to the budget it was capped by.

    The old code floored the target back up to 256 MiB *after* capping it at
    limit/CAP_DIVISOR, so a small allowance was ignored: the batch could plan
    several times the cap as scratch. Two frames are the minimum a submission
    needs to overlap, so the batch only drops below two when even those do not
    fit.
    """
    # The divisor the batch planner caps its tuned scratch target with. It is
    # 12 rather than 16 because EEDI3AA's measured knee (batch 4 at 2x2160p,
    # 670 MiB of scratch) sits just above limit/16 on a 10 GiB allowance.
    cap_divisor = 12
    scratch, batch, limit_mib = _eedi3_batch(_trace("eedi3"))
    cap = limit_mib / cap_divisor
    if cap >= 2 * scratch:
        assert batch >= 2, (batch, scratch, limit_mib)
    assert batch * scratch <= max(cap, scratch) + 1e-6

    # A budget whose cap is half a frame cannot host two frames: one frame, not
    # the five the floor used to pick here.
    tiny_mib = scratch * 8
    scratch2, batch2, limit2 = _eedi3_batch(
        _trace("eedi3", {"VSFEEL_LIMIT_VRAM_BUDGET": str(int(tiny_mib) << 20)})
    )
    assert batch2 == 1, (batch2, scratch2, tiny_mib)
    cap2 = min(limit2, tiny_mib) / cap_divisor
    assert batch2 * scratch2 <= max(cap2, scratch2) + 1e-6


def test_nlmeans_ring_is_capped_by_the_vram_budget():
    """The 64 MiB u4a ring optimum is a target, not a claim on the device."""
    ring, budget_mib, pack = _nlmeans_ring(_trace("nlmeans"))
    if pack == 1:
        pytest.skip("device budget is already at the one-pack ring floor")
    assert ring <= budget_mib + 1e-6

    # A quarter of the tuned ring budget must pack fewer entries per round.
    ring2, budget2, pack2 = _nlmeans_ring(
        _trace("nlmeans", {"VSFEEL_LIMIT_VRAM_BUDGET": str(int(budget_mib / 4 * 16) << 20)})
    )
    assert pack2 < pack, (pack, pack2, ring, budget_mib)
    # One pack is the floor (a run group has to fit), so a ring smaller than
    # that is not expressible; above it the ring never exceeds the budget.
    assert ring2 <= max(budget2, ring2 / pack2) + 1e-6


def test_nlmeans_pack_is_capped_by_the_grid_z_limit():
    """The weight dispatch's Z is a batch's row count: one or two per entry.

    A batch of ``qb*pack`` sweep entries dispatches one workgroup per table row
    and every entry with a non-zero temporal offset emits two, so ``pack`` has
    to yield to ``maxComputeWorkGroupCount[2]`` rather than dispatch an invalid
    grid. The default already fits the guaranteed 65535; the forced limit here
    (640x360, d=3 -> qb=8) is well below it, so the banner must show the clamp
    and the frame must still evaluate.
    """
    rows_per_entry = 2  # d=3: every kk != 0 entry emits a +q and a -q row
    qb = 8  # 640x360 is under the qb=8 threshold
    z_limit = 16

    _, _, pack = _nlmeans_ring(_trace("nlmeans"))
    assert pack * rows_per_entry * qb <= 65535, pack

    force = {"VSFEEL_LIMIT_GRID_Z": str(z_limit), "VSFEEL_NLMEANS_PACK": "16384"}
    _, _, capped = _nlmeans_ring(_trace("nlmeans", force))
    assert capped == z_limit // (rows_per_entry * qb), capped

    # Creation alone cannot see an over-limit dispatch; the frame run must.
    res = _run_limits([{"label": "nlmeans", "filter": "nlmeans"}], force)
    assert res["nlmeans"].get("finite") is True, res["nlmeans"]


def test_bilateral_auto_lds_gate_falls_back_to_the_plain_kernel():
    """The automatic LDS gate must pick the plain kernel when the tile is over
    budget, exactly as an explicit use_shared_memory=False does.

    Every bilateral test elsewhere passes use_shared_memory explicitly, so the
    automatic branch never ran. sigma_spatial=8 derives radius 24 with the
    auto-tuned 32x16 block, whose shared tile is (2*24+32)*(2*24+16)*4 =
    20480 B: it fits the device's 64 KiB but not a forced 12288-byte limit.
    Both runs below therefore use the plain kernel, and must agree bit-for-bit.
    """
    params = {"sigma_spatial": 8.0, "sigma_color": 0.15}
    auto = _run_limits(
        [{"label": "auto", "filter": "bilateral", "params": params}],
        {"VSFEEL_LIMIT_SHARED_MEMORY": "12288"},
    )
    plain = _run_limits(
        [
            {
                "label": "plain",
                "filter": "bilateral",
                "params": {**params, "use_shared_memory": False},
            }
        ],
        {"VSFEEL_LIMIT_SHARED_MEMORY": "12288"},
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
    """The fused path's largest tile is 7 680 B, under the 16 KiB floor.

    That is why the 48 KiB cap that used to sit beside the device check could
    never bind: the small path's tile is `VRT*BLK_Y*(BLK_X + 2*RAD)` floats and
    `RAD` is capped by LARGE_THRESHOLD well below it. Pinned so a change to
    either constant fails loudly (`gaussblur.cpp`'s static_assert catches it at
    build time as well), and so the device's own limit stays the check that
    decides which path a config takes.
    """
    floor = {"VSFEEL_LIMIT_SHARED_MEMORY": "16384"}  # the Vulkan minimum
    lds = [v for sigma in (2.0, 5.0, 8.0, 11.0) for v in _gauss_lds(sigma, floor)]
    assert 7680 in lds, lds  # the largest tile the small path asks for
    assert max(lds) <= 16384, lds  # and it fits the floor


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


def test_push_descriptor_limit_is_reported():
    """The binding count is bounded by maxPushDescriptors, checked explicitly."""
    res = _run_limits(
        [{"label": "nlmeans", "filter": "nlmeans"}], {"VSFEEL_LIMIT_PUSH_DESCRIPTORS": "4"}
    )
    err = res["nlmeans"].get("error", "")
    assert "push descriptors" in err and "4" in err, res["nlmeans"]


def test_two_dimensional_grid_overflow_is_reported():
    """Bilateral and GaussBlur cannot fold an over-wide plane into Y.

    Their kernels address a tile as (ID.x, ID.y), so a width past the device's
    X group limit has nowhere to go: it must fail with the grid and the limits
    rather than dispatch a clamped grid that leaves the tail unwritten.
    """
    cases = [
        {"label": "bilateral", "filter": "bilateral"},
        {"label": "gaussblur", "filter": "gaussblur"},
    ]
    res = _run_limits(cases, {"VSFEEL_LIMIT_GRID_X": "4"})
    for case in cases:
        err = res[case["label"]].get("error", "")
        assert "workgroup grid" in err and "4x" in err, res[case["label"]]


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
