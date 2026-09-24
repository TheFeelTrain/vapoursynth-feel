"""Device capability checks: workgroup size and workgroup memory.

Vulkan only guarantees 128 invocations per compute workgroup, 128 per dimension
and 16 KiB of workgroup memory, while several kernels here ask for 256
invocations or 18 KiB. Every pipeline is therefore measured against the device
before it is created, and a device that cannot host one gets an error naming the
kernel rather than a failure out of vkCreateComputePipelines.

No GPU in the test box is that small, so the checks are exercised by forcing the
limits down through the knobs ``get_gpu_device`` reads (VSFEEL_LIMIT_INVOCATIONS,
VSFEEL_LIMIT_SHARED_MEMORY). Those are read once per process when the core's
device is brought up, so each case runs in its own subprocess.

Run from the repository root:  python -m pytest tests/test_device_limits.py
"""

import json

import numpy as np
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
