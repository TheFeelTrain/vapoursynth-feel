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

Run from the repository root:  python -m pytest tests/test_device_limits.py
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import vapoursynth as vs

from conftest import COMPARE_PRELUDE, NOISE_MKV, run_compare_subprocess

# Every filter that needs a 256-invocation workgroup, plus one that fits 128.
_LIMITS_SCRIPT = COMPARE_PRELUDE + r'''
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
    if name == "bm3d":
        return core.vsfeel.BM3Dv2(clip, **params)
    if name == "dfttest":
        return core.vsfeel.DFTTest(clip, **params)
    if name == "nlmeans":
        return core.vsfeel.NLMeans(clip, **params)
    if name == "bilateral":
        return core.vsfeel.Bilateral(clip, sigma_spatial=6.0, sigma_color=0.1)
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
'''

# name -> kwargs; the empty ones just take the filter's defaults.
_256_KERNELS = ["dfttest", "nlmeans", "eedi3", "nnedi3", "bm3d"]


def _run_limits(cases, env=None):
    """Run every case in a fresh process, optionally under a limit override."""
    spec = {"source": NOISE_MKV, "cases": cases}
    return run_compare_subprocess(_LIMITS_SCRIPT, [json.dumps(spec)],
                                  env=env or {})


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

_BUDGET_SCRIPT = r'''
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
'''


def _trace(which, env=None):
    """stderr of a creation run, with that filter's trace flag turned on."""
    trace_flag = "VSFEEL_EEDI3_TRACE" if which == "eedi3" else "VSFEEL_NLMEANS_VRAM"
    run_env = {**os.environ, "MANGOHUD": "0", trace_flag: "1", **(env or {})}
    proc = subprocess.run([sys.executable, "-c", _BUDGET_SCRIPT, which],
                          capture_output=True, text=True, timeout=600, env=run_env)
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
    limit/16, so a small allowance was ignored: the batch could plan several
    times the cap as scratch. Two frames are the minimum a submission needs to
    overlap, so the batch only drops below two when even those do not fit.
    """
    scratch, batch, limit_mib = _eedi3_batch(_trace("eedi3"))
    if limit_mib / 16 >= 2 * scratch:
        assert batch >= 2, (batch, scratch, limit_mib)
    assert batch * scratch <= max(limit_mib / 16, scratch) + 1e-6

    # A budget whose cap is half a frame cannot host two frames: one frame, not
    # the five the floor used to pick here.
    tiny_mib = scratch * 8
    scratch2, batch2, limit2 = _eedi3_batch(
        _trace("eedi3", {"VSFEEL_LIMIT_VRAM_BUDGET": str(int(tiny_mib) << 20)}))
    assert batch2 == 1, (batch2, scratch2, tiny_mib)
    cap2 = min(limit2, tiny_mib) / 16
    assert batch2 * scratch2 <= max(cap2, scratch2) + 1e-6


def test_nlmeans_ring_is_capped_by_the_vram_budget():
    """The 64 MiB u4a ring optimum is a target, not a claim on the device."""
    ring, budget_mib, pack = _nlmeans_ring(_trace("nlmeans"))
    if pack == 1:
        pytest.skip("device budget is already at the one-pack ring floor")
    assert ring <= budget_mib + 1e-6

    # A quarter of the tuned ring budget must pack fewer entries per round.
    ring2, budget2, pack2 = _nlmeans_ring(_trace("nlmeans", {
        "VSFEEL_LIMIT_VRAM_BUDGET": str(int(budget_mib / 4 * 16) << 20)}))
    assert pack2 < pack, (pack, pack2, ring, budget_mib)
    # One pack is the floor (a run group has to fit), so a ring smaller than
    # that is not expressible; above it the ring never exceeds the budget.
    assert ring2 <= max(budget2, ring2 / pack2) + 1e-6
