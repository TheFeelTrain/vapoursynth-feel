"""Resource boundedness: repeated create/evaluate/destroy and mixed-filter concurrency.

``test_create_destroy_cycles_keep_memory_bounded`` runs 200 cycles per filter
with an anchor node pinning the shared device.  Without it each cycle's device
dies with the cycle and the driver reclaims whatever the cycle failed to free,
so nothing accumulates; with it, a per-cycle leak does.  Process-local GPU
memory comes from the render node's ``/proc/self/fdinfo`` (``drm-memory-vram``/
``-gtt``/``-cpu``) — unlike the device-wide sysfs figure the loop also samples,
it is not perturbed by the desktop.  Both are reported; only the per-process
series is graded, and RSS covers host-side owners.  A subprocess per filter
keeps a hang or crash out of the suite.

``test_mixed_filter_concurrency_matches_solo`` creates every filter at once on
the shared device, evaluates them together and compares each against its solo
run; BM3D uses ``extractor_exp=8`` so its atomic aggregation is
order-independent, as the stream matrix does.

Measured on the RX 7900XTX, 200 cycles at the shipped state: per-process GPU
memory is byte-flat (0.0 MB rise at 4 KiB quantisation) for every filter; RSS
drift stays within a few MB (worst BM3Dv2 ~5 MB, and that decelerates — 1000
cycles add ~5 MB more, allocator warmup rather than a per-cycle leak).  The
limits below sit above that floor and still trip on a per-process leak of
roughly 30 KiB per cycle.

Run from the repository root:  python -m pytest tests/test_resources.py
"""

import json
import statistics
import subprocess
import sys

import pytest

from conftest import COMPARE_PRELUDE, NOISE_MKV

# Filters and the arguments the cycle test runs them with.  The concurrency test
# reuses them on a 64x64 crop with num_streams=2.
FILTERS = ["Bilateral", "GaussBlur", "DFTTest", "NLMeans", "BM3Dv2",
           "EEDI3", "EEDI3H", "EEDI3AA", "NNEDI3"]

CYCLES = 200
WARMUP = 20  # device / pipeline-cache / allocator settling, discarded

# see the module docstring for the measured values these sit above.  The
# device-wide sysfs figure is reported but not graded: it is shared, so under a
# parallel test run another worker's allocations move it.
GPU_RISE_LIMIT = 4 * 1024 * 1024
RSS_RISE_LIMIT = 32 * 1024 * 1024


_COMMON = r'''
import ctypes
import glob
import json
import os
import sys
import threading
import time

import numpy as np
import vapoursynth as vs

NOISE_MKV = __NOISE_MKV__
CORE = vs.core
CORE.max_cache_size = 64

NAMES = ["Bilateral", "GaussBlur", "DFTTest", "NLMeans", "BM3Dv2",
         "EEDI3", "EEDI3H", "EEDI3AA", "NNEDI3"]

# g16-input predictors; the rest take g32
_WIDE = ("EEDI3", "EEDI3H", "EEDI3AA", "NNEDI3")
_DTYPE = {n: (np.uint16 if n in _WIDE else np.float32) for n in NAMES}

_KWARGS = {
    "Bilateral": {"sigma_spatial": 3.0, "sigma_color": 0.05},
    "GaussBlur": {"sigma": 2.0},
    "DFTTest": {"tbsize": 3},
    "NLMeans": {"d": 2},
    "BM3Dv2": {"sigma": 0.7, "radius": 2, "bm_range": 16, "ps_range": 7,
               "block_step": 4, "extractor_exp": 8},
    "EEDI3": {"field": 1},
    "EEDI3H": {"field": 1},
    "EEDI3AA": {"field": 3},
    "NNEDI3": {"field": 1},
}


def make_clips():
    src = CORE.bs.VideoSource(NOISE_MKV)
    g = CORE.std.ShufflePlanes(src, 0, vs.GRAY)
    g32 = CORE.fmtc.bitdepth(g, bits=32, fulls=True, fulld=True)
    g16 = CORE.fmtc.bitdepth(g, bits=16, fulls=True, fulld=True)
    return g32, g16


def build(name, g32, g16, num_streams=None):
    clip = g16 if name in _WIDE else g32
    kwargs = dict(_KWARGS[name])
    if num_streams is not None:
        kwargs["num_streams"] = num_streams
    return getattr(CORE.vsfeel, name)(clip, **kwargs)


def drm_memory():
    """Per-process (vram, gtt, cpu) bytes from the DRM render-node fds; None
    when the process holds no DRM fd at all."""
    vram = gtt = cpu = 0
    found = False
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink("/proc/self/fd/" + fd)
        except OSError:
            continue
        if "/dev/dri/" not in target:
            continue
        try:
            with open("/proc/self/fdinfo/" + fd) as fh:
                text = fh.read()
        except OSError:
            continue
        if "drm-driver:" not in text:
            continue
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields = value.split()
            if not fields:
                continue
            try:
                n = int(fields[0]) * 1024
            except ValueError:
                continue
            if key == "drm-memory-vram":
                vram += n
                found = True
            elif key == "drm-memory-gtt":
                gtt += n
            elif key == "drm-memory-cpu":
                cpu += n
    return (vram, gtt, cpu) if found else None


def device_vram():
    """Device-wide VRAM in use, summed over every DRM card (sysfs)."""
    total = 0
    found = False
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            with open(path) as fh:
                total += int(fh.read().strip())
            found = True
        except OSError:
            pass
    return total if found else None


def read_rss():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def sample():
    m = drm_memory()
    return {
        "vram": m[0] if m else None,
        "gtt": m[1] if m else None,
        "cpu": m[2] if m else None,
        "dev": device_vram(),
        "rss": read_rss(),
    }
'''

_CYCLE_SCRIPT = _COMMON + r'''
def main():
    name = sys.argv[1]
    g32, g16 = make_clips()
    tiny = CORE.std.Crop(g32, right=g32.width - 64, bottom=g32.height - 64)
    # Pin the shared device: a live instance keeps it (and the DRM fd) alive, so
    # anything a cycle fails to free stays accounted to this process instead of
    # being reclaimed when the cycle's own device is destroyed.
    anchor = CORE.vsfeel.GaussBlur(tiny, sigma=1.5)
    anchor.get_frame(0)

    samples = []
    start = time.perf_counter()
    for _ in range(__CYCLES__):
        node = build(name, g32, g16)
        node.get_frame(0)
        del node
        CORE.clear_cache()
        samples.append(sample())
    secs = time.perf_counter() - start

    del anchor
    CORE.clear_cache()
    after = drm_memory()
    print("RESULT " + json.dumps({
        "filter": name,
        "cycles": len(samples),
        "samples": samples,
        "secs": secs,
        "teardown": list(after) if after else None,
    }), flush=True)


main()
'''

_CONCURRENCY_SCRIPT = COMPARE_PRELUDE + _COMMON + r'''
SIZE = 64
FRAMES = 4


def main():
    g32, g16 = make_clips()
    c32 = CORE.std.Crop(g32, right=g32.width - SIZE, bottom=g32.height - SIZE)
    c16 = CORE.std.Crop(g16, right=g16.width - SIZE, bottom=g16.height - SIZE)

    # solo oracle first: one node at a time, nothing else resident
    solo = {}
    for name in NAMES:
        node = cpu_node(build(name, c32, c16, num_streams=2))
        solo[name] = [read_plane(node.get_frame(n), 0, _DTYPE[name])
                      for n in range(FRAMES)]
        del node
        CORE.clear_cache()

    # every filter resident at once, each evaluated from its own thread and
    # released together by a barrier so the GPU work actually overlaps
    nodes = {name: cpu_node(build(name, c32, c16, num_streams=2)) for name in NAMES}
    results = {}
    errors = []
    barrier = threading.Barrier(len(NAMES))

    def worker(name):
        try:
            barrier.wait()
            node = nodes[name]
            results[name] = [read_plane(node.get_frame(n), 0, _DTYPE[name])
                             for n in range(FRAMES)]
        except BaseException as exc:  # noqa: BLE001 - reported through RESULT
            errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))

    threads = [threading.Thread(target=worker, args=(name,)) for name in NAMES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    diffs = {}
    for name in NAMES:
        if name not in results:
            diffs[name] = None
            continue
        worst = 0.0
        for a, b in zip(solo[name], results[name]):
            worst = max(worst, float(np.abs(
                a.astype(np.float64) - b.astype(np.float64)).max()))
        diffs[name] = worst

    print("RESULT " + json.dumps({"diffs": diffs, "errors": errors}), flush=True)


main()
'''

_CYCLE_SCRIPT = (_CYCLE_SCRIPT.replace("__NOISE_MKV__", repr(NOISE_MKV))
                 .replace("__CYCLES__", str(CYCLES)))
_CONCURRENCY_SCRIPT = _CONCURRENCY_SCRIPT.replace("__NOISE_MKV__",
                                                  repr(NOISE_MKV))


def _run(script, argv=(), timeout=600.0):
    """Run one subprocess script and return the JSON payload of its RESULT line."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, *[str(a) for a in argv]],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "resource subprocess hung after %.0fs\n--- stdout tail ---\n%s\n"
            "--- stderr tail ---\n%s"
            % (timeout, (exc.stdout or "")[-2000:], (exc.stderr or "")[-2000:])
        )
    tail = ("--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % ((proc.stdout or "")[-2000:], (proc.stderr or "")[-2000:]))
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT "):
            payload = line[len("RESULT "):].strip()
    assert proc.returncode == 0 and payload is not None, (
        "resource subprocess exited %s without a RESULT line\n%s"
        % (proc.returncode, tail))
    return json.loads(payload)


def _trend(samples):
    """Median/min of the first and last quarter of the post-warmup series.

    The median tracks sustained drift and ignores single-sample noise; the
    running minimum additionally catches a leak whose samples sit on a rising
    floor rather than a clean ramp.
    """
    series = [float(s) for s in samples if s is not None]
    if len(series) <= 2 * WARMUP:  # metric unavailable on this box
        return None
    body = series[WARMUP:]
    quarter = max(1, len(body) // 4)
    first, last = body[:quarter], body[-quarter:]
    return {
        "first": statistics.median(first),
        "last": statistics.median(last),
        "rise": statistics.median(last) - statistics.median(first),
        "min_rise": min(last) - min(first),
        "min": min(body),
        "max": max(body),
    }


def _format(key, trend, limit=None):
    text = ("  %-4s first %8.2f MB  last %8.2f MB  rise %+7.2f MB  "
            "min-rise %+7.2f MB  min %8.2f  max %8.2f"
            % (key, trend["first"] / 1048576, trend["last"] / 1048576,
               trend["rise"] / 1048576, trend["min_rise"] / 1048576,
               trend["min"] / 1048576, trend["max"] / 1048576))
    return (text + "  (limit %.0f MB)" % (limit / 1048576) if limit is not None
            else text + "  (reported only)")


@pytest.mark.parametrize("filter_name", FILTERS)
def test_create_destroy_cycles_keep_memory_bounded(filter_name):
    """200 create/evaluate/destroy cycles must not raise RSS or GPU memory.

    A monotonically rising per-process GPU figure (VRAM, GTT or host-visible
    memory) means a cycle abandons a buffer, device memory, command pool, fence
    or mapped window; a rising RSS means the host-side owner leaks.  The
    device-wide sysfs figure is reported for context but graded loosely because
    other GPU clients share it.
    """
    payload = _run(_CYCLE_SCRIPT, [filter_name], timeout=600.0)
    samples = payload["samples"]
    assert payload["cycles"] == CYCLES and len(samples) == CYCLES, (
        "%s: expected %d cycles, got %d" % (filter_name, CYCLES,
                                            len(samples)))

    keys = ("vram", "gtt", "cpu", "rss", "dev")
    graded = ("vram", "gtt", "cpu", "rss")
    trends = {k: _trend([s[k] for s in samples]) for k in keys}
    limits = {"vram": GPU_RISE_LIMIT, "gtt": GPU_RISE_LIMIT,
              "cpu": GPU_RISE_LIMIT, "rss": RSS_RISE_LIMIT}

    report = ["%s: %d cycles, %.1f ms/cycle"
              % (filter_name, payload["cycles"],
                 1000.0 * payload["secs"] / payload["cycles"])]
    for key in keys:
        report.append("  %-4s unavailable" % key if trends[key] is None
                      else _format(key, trends[key], limits.get(key)))
    report.append("  device released after teardown: %s (teardown %r)"
                  % (payload["teardown"] in (None, [0, 0, 0]),
                     payload["teardown"]))
    print("\n".join(report), flush=True)

    assert trends["rss"] is not None, "no RSS samples were taken"
    assert trends["vram"] is not None or trends["dev"] is not None, (
        "no GPU memory accounting available (both per-process DRM fdinfo and "
        "sysfs mem_info_vram_used are missing)\n" + "\n".join(report))

    bad = []
    for key in graded:
        trend = trends[key]
        if trend is None:
            continue
        grew = max(trend["rise"], trend["min_rise"])
        if grew > limits[key]:
            bad.append("%s grew %.2f MB over %d cycles (limit %.0f MB)"
                       % (key, grew / 1048576, CYCLES, limits[key] / 1048576))
    assert not bad, (
        "%s leaks over repeated create/destroy cycles:\n  %s\n%s"
        % (filter_name, "\n  ".join(bad), "\n".join(report)))

    # The per-cycle trends above are the leak check. The teardown figure is
    # reported only: under the R80 GPU API the core owns the one device, so it
    # (and its frame cache) is still resident after the last instance is freed
    # and a nonzero figure says nothing about the filter's own resources.
    _ = payload["teardown"]


def test_mixed_filter_concurrency_matches_solo():
    """Every filter's output under mixed concurrent load must equal its solo run.

    All nine instances share one device; each is driven from its own thread.
    Cross-instance interference — a queue/stream mix-up, a shared cache slot
    handed out twice, a global flag flipped per resource — shows up as a
    pixel difference against the solo oracle, which nothing else in the suite
    runs under this load.
    """
    payload = _run(_CONCURRENCY_SCRIPT, timeout=900.0)
    assert not payload["errors"], (
        "mixed-filter concurrency raised: %s" % "; ".join(payload["errors"]))

    missing = [n for n in FILTERS if payload["diffs"].get(n) is None]
    assert not missing, "no concurrent output for: %s" % ", ".join(missing)

    bad = {n: d for n, d in payload["diffs"].items()
           if d is not None and d != 0.0}
    assert not bad, (
        "concurrent output differs from the solo run (self-consistency must be "
        "exact): %s" % bad)
