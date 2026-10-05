"""BM3Dv2's block matcher against an independent scalar model of mawen's CPU
matcher (``tests/bm3d_oracle.py``).

The GPU kernel's selected group is observed through ``VSFEEL_BM3D_MATCHTRACE``,
which records, for one reference block of one centre frame, the final group, the
current-frame list and every temporal frame's retained/seed counts. These are
coordinate-level comparisons: an image difference cannot separate a matcher bug
from a transform or aggregation difference.

The clips are synthesised so the thresholds are meaningful -- the committed
noise clip is far too low-amplitude for its sigma-scaled default threshold to
reject anything, which makes every group a full eight and hides the whole
retention contract.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from conftest import COMPARE_PRELUDE
from bm3d_oracle import match_group

W = H = 32
NFRAMES = 7

# One process traces every configuration: the trace target and the threshold are
# read when the filter instance is created, so the environment can be changed
# between cases.
_BATCH = (
    COMPARE_PRELUDE
    + """
import json
import os
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.max_cache_size = 512
arr = np.load(sys.argv[1])
jobs = json.loads(sys.argv[2])
src = core.std.BlankClip(width=arr.shape[2], height=arr.shape[1],
                         format=vs.GRAYS, length=arr.shape[0], color=0.0)


def setter(n, f):
    out = f.copy()
    plane = np.asarray(out[0])
    plane[:] = arr[n][:plane.shape[0], :plane.shape[1]]
    return out


clip = core.std.ModifyFrame(src, src, setter)
for job in jobs:
    os.environ["VSFEEL_BM3D_MATCHTRACE"] = "%d,%d,%d" % (
        job["frame"], job["x"], job["y"])
    node = core.vsfeel.BM3Dv2(clip, **job["kwargs"])
    node = core.std.GPUDownload(clip=node) if node.gpu_resident else node
    if job.get("sequence"):
        for f in range(job["frame"] + 1):
            node.get_frame(f)
    else:
        node.get_frame(job["frame"])
    print("DONE " + job["id"], flush=True)
"""
)


def make_clip(kind, seed=11, nframes=NFRAMES):
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return rng.normal(0.5, 0.08, (nframes, H, W)).astype(np.float32)
    if kind == "grad":
        # Stronger noise and a faster drift than "motion": the block distances
        # spread over the threshold, so group sizes of one, two, three and
        # seven all occur instead of every frame filling up.
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        out = np.zeros((nframes, H, W), dtype=np.float32)
        for f in range(nframes):
            base = 0.5 + 0.3 * np.sin((xs + 3.0 * f) * 0.3) * np.cos(ys * 0.2)
            out[f] = (base + rng.normal(0, 0.05, (H, W))).astype(np.float32)
        return out
    # A smooth pattern drifting two pixels per frame plus a little noise: a
    # neighbouring frame matches well at a shifted origin, so the per-frame
    # retained counts vary instead of all filling up.
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((nframes, H, W), dtype=np.float32)
    for f in range(nframes):
        base = 0.5 + 0.25 * np.sin((xs + 2.0 * f + 40.0) * 0.35) * np.cos(ys * 0.21)
        out[f] = (base + rng.normal(0, 0.01, (H, W))).astype(np.float32)
    return out


_HEADER = re.compile(
    r"\[bm3d-trace\] x=(\d+) y=(\d+) th_mse=(\S+) th_sse=(\S+) n=(\d+) "
    r"retained=(\d+) overflow=(\d+) current=(\d+) seeds=(\d+)"
)
_MEMBER = re.compile(r"\[bm3d-trace\] member (\d+) x=(\d+) y=(\d+) z=(\d+) err=(\S+)")
_FRAME = re.compile(
    r"\[bm3d-trace\] (bwd|fwd) step=(\d+) frame=(\d+) retained=(\d+) "
    r"seeds=(\d+)"
)
_MISSING = "no group matched"


def _run_batch(npy, jobs):
    env = {**os.environ, "MANGOHUD": "0"}
    env.pop("VSFEEL_BM3D_MATCHTRACE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _BATCH, str(npy), json.dumps(jobs)],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"trace subprocess failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
        )

    out = []
    current = None
    for line in proc.stderr.splitlines():
        if _MISSING in line:
            out.append(None)
            current = None
            continue
        m = _HEADER.search(line)
        if m:
            current = {
                "x": int(m.group(1)),
                "y": int(m.group(2)),
                "th_mse": float(m.group(3)),
                "th_sse": float(m.group(4)),
                "n": int(m.group(5)),
                "retained": int(m.group(6)),
                "overflow": int(m.group(7)),
                "current": int(m.group(8)),
                "seeds": int(m.group(9)),
                "members": [],
                "frames": [],
            }
            out.append(current)
            continue
        if current is None:
            continue
        m = _MEMBER.search(line)
        if m:
            current["members"].append(
                (int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5)))
            )
            continue
        m = _FRAME.search(line)
        if m:
            current["frames"].append(
                (m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(5)))
            )
    done = len([line for line in proc.stdout.splitlines() if line.startswith("DONE ")])
    assert done == len(jobs), f"{done}/{len(jobs)} jobs completed"
    assert len(out) == len(jobs), f"{len(out)} traces for {len(jobs)} jobs"
    return out


def base_kwargs(sigma=8.0, radius=2, bm_range=9, ps_num=2, ps_range=4, **extra):
    kwargs = dict(
        sigma=[sigma],
        radius=radius,
        block_step=8,
        bm_range=bm_range,
        ps_num=ps_num,
        ps_range=ps_range,
    )
    kwargs.update(extra)
    return kwargs


def default_th(sigma):
    return 400.0 + sigma * 80.0


# (id, clip kind, frame, x, y, extra kwargs)
CASES = [
    ("default", "noise", 3, 8, 8, {}),
    ("ps1", "noise", 3, 8, 8, {"ps_num": 1}),
    ("ps4", "noise", 3, 8, 8, {"ps_num": 4}),
    ("ps8", "noise", 3, 8, 8, {"ps_num": 8}),
    ("psr2", "noise", 3, 8, 8, {"ps_range": 2}),
    ("psr1", "noise", 3, 8, 8, {"ps_range": 1}),
    ("bm4", "noise", 3, 8, 8, {"bm_range": 4}),
    ("r0", "noise", 3, 8, 8, {"radius": 0}),
    ("r1", "noise", 3, 8, 8, {"radius": 1}),
    ("r3", "noise", 3, 8, 8, {"radius": 3}),
    ("th300", "noise", 3, 8, 8, {"th_mse": 300.0}),
    ("th700", "noise", 3, 8, 8, {"th_mse": 700.0}),
    ("th0", "noise", 3, 8, 8, {"th_mse": 0.0}),
    ("th250ps1", "noise", 3, 8, 8, {"th_mse": 250.0, "ps_num": 1}),
    # clip ends: fewer real temporal neighbours than the radius asks for
    ("first", "noise", 0, 0, 0, {}),
    ("firstr3", "noise", 0, 0, 0, {"radius": 3}),
    ("last", "noise", 6, 24, 24, {}),
    ("secondr3", "noise", 1, 8, 8, {"radius": 3}),
    ("mth200", "motion", 3, 8, 8, {"th_mse": 200.0}),
    ("mth400", "motion", 3, 8, 8, {"th_mse": 400.0}),
    ("mth600", "motion", 3, 8, 8, {"th_mse": 600.0}),
    ("mth900", "motion", 3, 8, 8, {"th_mse": 900.0}),
    ("mth600ps1", "motion", 3, 8, 8, {"th_mse": 600.0, "ps_num": 1}),
    ("mth600ps3r3", "motion", 3, 8, 8, {"th_mse": 600.0, "ps_num": 3, "radius": 3}),
    ("mfirst", "motion", 1, 0, 0, {"th_mse": 900.0}),
    ("mlast", "motion", 6, 24, 24, {"th_mse": 400.0}),
    # Group sizes below eight, measured on the structured clip with the
    # normY-scaled threshold the host now applies.
    ("grad1", "grad", 2, 8, 8, {"th_mse": 300.0}),
    ("grad3", "grad", 2, 8, 8, {"th_mse": 350.0}),
    ("grad5", "grad", 2, 8, 8, {"th_mse": 400.0}),
    ("grad8", "grad", 2, 8, 8, {"th_mse": 450.0}),
    ("gradover", "grad", 2, 8, 8, {"th_mse": 550.0}),
    ("grad3r3", "grad", 2, 8, 8, {"th_mse": 350.0, "radius": 3}),
    ("grad3ps1", "grad", 2, 8, 8, {"th_mse": 350.0, "ps_num": 1}),
]


@pytest.fixture(scope="session")
def traces(tmp_path_factory):
    """Every case's trace, from one subprocess per clip."""
    out = {}
    for kind in ("noise", "motion", "grad"):
        jobs = [
            dict(id=cid, frame=frame, x=x, y=y, kwargs=base_kwargs(**extra))
            for (cid, ckind, frame, x, y, extra) in CASES
            if ckind == kind
        ]
        path = tmp_path_factory.mktemp(kind) / "clip.npy"
        np.save(path, make_clip(kind))
        out.update(zip([j["id"] for j in jobs], _run_batch(path, jobs)))
    return out


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_bm3dv2_matcher_matches_oracle(traces, case):
    """The selected group, its order, every frame's retained/seed counts, the
    reserved reference and the reference-only threshold case must all agree with
    the scalar model of the CPU matcher."""
    cid, kind, frame, x, y, extra = case
    kwargs = base_kwargs(**extra)
    sigma = kwargs["sigma"][0]
    th_user = kwargs.get("th_mse", default_th(sigma))
    expect, info = match_group(
        frame,
        x,
        y,
        make_clip(kind),
        kwargs["radius"],
        kwargs["bm_range"],
        kwargs["ps_num"],
        kwargs["ps_range"],
        th_user,
    )

    got = traces[cid]
    assert got is not None, "the kernel never recorded the traced reference block"
    assert got["n"] == len(expect)
    # "retained" is the combined list's length, which stops growing at eight;
    # the overflow flag is what records that more candidates were appended.
    total = len(info["current"]) + sum(len(m) for _, m in info["frames"])
    assert got["retained"] == min(total, 8)
    assert got["overflow"] == int(total > 8)
    assert got["current"] == len(info["current"])
    assert got["seeds"] == min(kwargs["ps_num"], len(info["current"]))

    seen = [tuple(m[:3]) for m in got["members"][: got["n"]]]
    want = [(x_, y_, kwargs["radius"] + (f - frame)) for (f, x_, y_, _) in expect]
    assert seen == want
    for m, (_, _, _, err) in zip(got["members"][: got["n"]], expect):
        assert m[3] == pytest.approx(float(err), rel=1e-4, abs=1e-6)

    # Per-frame retention and prediction seeding for both directions.
    want_frames = [(f, len(m), min(kwargs["ps_num"], len(m))) for f, m in info["frames"]]
    got_frames = [(f, r, s) for (_, f, r, s) in got["frames"]]
    assert sorted(got_frames) == sorted(want_frames)


def test_bm3dv2_group_shrinks_below_eight(traces):
    """A rejecting threshold must produce real groups smaller than eight, and
    the exercised configurations must include several of them."""
    shrink = ("grad1", "grad3", "grad5", "grad8", "gradover")
    sizes = [traces[cid]["n"] for cid in shrink]
    assert len(set(sizes)) >= 4, sizes
    assert min(sizes) == 1 and max(sizes) == 8, sizes
    # An eight-member group that never overflowed keeps concatenation order.
    assert traces["grad8"]["overflow"] == 0
    assert traces["gradover"]["overflow"] == 1
    for cid in shrink:
        t = traces[cid]
        members = [tuple(m[:3]) for m in t["members"][: t["n"]]]
        assert members[0] == (8, 8, 2), "the reference block must lead the group"
        assert len(set(members)) == len(members), "no origin may repeat in a group"


def test_bm3dv2_threshold_zero_is_reference_only(traces):
    """th_mse = 0 keeps the CPU's reference-only group, with no search at all."""
    got = traces["th0"]
    assert got["n"] == 1
    assert got["retained"] == 1
    assert got["members"][0][:3] == (8, 8, 2)
    assert got["frames"] == []


def test_bm3dv2_matcher_trace_survives_the_estimate_cache(tmp_path):
    """An earlier request may compute the traced centre; the trace must follow.

    A sequential load (0..frame, what vspipe produces) has frame - radius
    estimate the traced centre first, so by the time the traced frame is
    requested its centre comes from the estimate cache and no dispatch records
    the trace. Each sequential trace must equal the single-request one, which
    is what the other matcher cases exercise.
    """
    path = tmp_path / "clip.npy"
    np.save(path, make_clip("noise"))
    jobs = [
        dict(id=kind, frame=3, x=8, y=8, kwargs=base_kwargs(**extra), sequence=seq)
        for kind, extra, seq in (
            ("single_r2", {}, False),
            ("seq_r2", {}, True),
            ("single_r3", {"radius": 3}, False),
            ("seq_r3", {"radius": 3}, True),
        )
    ]
    got = dict(zip([j["id"] for j in jobs], _run_batch(path, jobs)))
    for single, seq in (("single_r2", "seq_r2"), ("single_r3", "seq_r3")):
        assert got[seq] is not None, "the sequential load lost the trace"
        assert got[seq] == got[single], f"{seq} differs from {single}"


@pytest.mark.parametrize("sequence", [False, True], ids=["single", "sequential"])
def test_bm3dv2_radius16_trace_reaches_its_last_temporal_word(tmp_path, sequence):
    """radius 16 makes the record 102 words (the last counter is index 101).

    The buffer used to be one word short, so the kernel's last temporal write
    and the dumper's read of it landed past both the buffer and its mapping. A
    33-frame clip gives 16 steps in each direction, and the oracle's per-frame
    counts must survive for every one of them.
    """
    frame, radius = 16, 16
    clip = make_clip("noise", nframes=2 * radius + 1)
    path = tmp_path / "clip.npy"
    np.save(path, clip)
    kwargs = base_kwargs(radius=radius)
    job = dict(id="r16", frame=frame, x=8, y=8, kwargs=kwargs, sequence=sequence)
    got = _run_batch(path, [job])[0]
    assert got is not None, "the radius-16 trace was never recorded"

    _, info = match_group(
        frame,
        8,
        8,
        clip,
        radius,
        kwargs["bm_range"],
        kwargs["ps_num"],
        kwargs["ps_range"],
        default_th(kwargs["sigma"][0]),
    )
    want = [(f, len(m), min(kwargs["ps_num"], len(m))) for f, m in info["frames"]]
    assert len(want) == 2 * radius
    assert sorted((f, r, s) for (_, f, r, s) in got["frames"]) == sorted(want)


def test_bm3dv2_trace_rejects_a_non_origin(tmp_path):
    """A reference position the matcher never visits is an error: silently
    tracing the nearest grid origin would report a different block's group."""
    path = tmp_path / "clip.npy"
    np.save(path, make_clip("noise"))
    env = {**os.environ, "MANGOHUD": "0"}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _BATCH,
            str(path),
            json.dumps([dict(id="c", frame=3, x=3, y=5, kwargs=base_kwargs())]),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode != 0
    assert "not a reference-block position" in proc.stderr
