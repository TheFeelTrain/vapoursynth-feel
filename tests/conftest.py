import ctypes
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import vapoursynth as vs

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
NOISE_MKV = os.path.join(TESTS_DIR, "noise_24f.mkv")
WIDTH = 640
HEIGHT = 360


def _source(path):
    core = vs.core
    if hasattr(core, "bs"):
        return core.bs.VideoSource(path)
    if hasattr(core, "ffms2"):
        return core.ffms2.Source(path)
    raise RuntimeError("no source plugin available (need bs or ffms2)")


def _plane_size(frame, plane):
    """Visible width/height of ``plane`` in this frame's format."""
    fmt = frame.format
    ss_w = ss_h = 0
    if plane > 0 and fmt.num_planes > 1:
        # VapourSynth stores the log2 subsampling factor for the chroma planes.
        ss_w = fmt.subsampling_w or 0
        ss_h = fmt.subsampling_h or 0
    w = (frame.width + (1 << ss_w) - 1) >> ss_w
    h = (frame.height + (1 << ss_h) - 1) >> ss_h
    return w, h


def format_dtype(fmt):
    """Numpy dtype for one sample of a VapourSynth format."""
    return np.float32 if fmt.sample_type == vs.FLOAT else np.uint16


def plane_to_ndarray(frame, plane, dtype=np.float32):
    """Copy one *visible* plane of a VapourSynth frame into a fresh ndarray.

    This is stride-aware: each visible row is sliced out of the frame's actual
    row pitch, so cropped / padded planes (where ``stride != width *
    itemsize``) read correctly.  The result is a copy and therefore never
    aliases frame memory, which VapourSynth recycles as soon as the Python
    wrapper of a temporary frame is dropped.
    """
    dt = np.dtype(dtype)
    w, h = _plane_size(frame, plane)
    stride = frame.get_stride(plane)
    raw = np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(h, stride),
    )
    return raw[:, : w * dt.itemsize].copy().view(dt).reshape(h, w)


def frame_to_ndarray(frame, dtype=np.float32, plane=0):
    """Plane-0 convenience wrapper around :func:`plane_to_ndarray`."""
    return plane_to_ndarray(frame, plane, dtype)


def plane_to_float64(frame, plane):
    """Visible plane widened to float64 (NaN-safe comparisons)."""
    return plane_to_ndarray(frame, plane, format_dtype(frame.format)).astype(np.float64)


def max_diff(a_node, b_node, planes=None, frames=(0, 11, 23)):
    """Worst absolute difference between two clips' visible planes.

    Finiteness is asserted on both sides first: Python's ``max(0.0, nan) ==
    0.0`` makes a bare ``max()`` over the difference report a perfect 0.0 match
    for any non-finite image, so a NaN would silently pass every comparison.
    ``planes=None`` means every plane of the frame.
    """
    worst = 0.0
    for n in frames:
        fa, fb = a_node.get_frame(n), b_node.get_frame(n)
        sel = range(fa.format.num_planes) if planes is None else planes
        for p in sel:
            x, y = plane_to_float64(fa, p), plane_to_float64(fb, p)
            assert np.isfinite(x).all(), f"non-finite first clip at frame {n} plane {p}"
            assert np.isfinite(y).all(), f"non-finite second clip at frame {n} plane {p}"
            worst = max(worst, float(np.abs(x - y).max()))
    return worst


def assert_all_frames_finite(node, frames=None, plane=0):
    """Every frame's visible plane must be finite (float64 widening)."""
    frames = range(node.num_frames) if frames is None else frames
    for n in frames:
        a = plane_to_float64(node.get_frame(n), plane)
        assert np.isfinite(a).all(), f"non-finite output at frame {n}"


def assert_changes_on_noise(out, src, frames=None, plane=0, what="filter"):
    """Anti-vacuity: ``out`` must alter ``src`` somewhere.

    A finiteness (or uint16 range) assertion alone passes for an identity
    filter, so every such test also has to show the filter did something.
    """
    frames = range(out.num_frames) if frames is None else frames
    worst = 0.0
    for n in frames:
        d = plane_to_float64(out.get_frame(n), plane) - plane_to_float64(src.get_frame(n), plane)
        assert np.isfinite(d).all(), f"non-finite difference at frame {n}"
        worst = max(worst, float(np.abs(d).max()))
    assert worst > 0.0, f"{what} left the noise input unchanged (identity filter)"


# ---------------------------------------------------------------------------
# Reference-comparison subprocess protocol
# ---------------------------------------------------------------------------
#
# Reference plugins are crash-prone and a vsfeel bug can segfault too, so a
# comparison runs in a subprocess.  To distinguish "the reference is missing /
# failed" (skip) from "vsfeel failed" (fail), the compare script prints:
#
#     REF ok                       reference frames materialised successfully
#     REF unavailable: <reason>    reference plugin missing or its eval failed
#     VSFEEL fail: <reason>        vsfeel-side failure the script detected
#     RESULT <json>                final comparison payload
#
# The reference frames must be materialised (and copied into numpy) *before*
# the vsfeel node is constructed or read, and `REF ok` printed at that point.


def pytest_addoption(parser):
    parser.addoption(
        "--require-references", action="store_true", default=False,
        help="fail instead of skipping when a reference plugin is unavailable",
    )


# Set by the session fixture below.  A missing reference silently deletes the
# correctness net, so the default is to skip but this switch makes it loud.
_REQUIRE_REFERENCES = False


@pytest.fixture(scope="session", autouse=True)
def _configure_references(request):
    global _REQUIRE_REFERENCES
    _REQUIRE_REFERENCES = bool(request.config.getoption("--require-references"))


class ReferenceUnavailable(RuntimeError):
    """The reference plugin or the reference-side evaluation failed."""


def skip_or_fail_reference(reason):
    """Skip because a reference is unavailable, or fail under
    ``--require-references`` (which exists to keep a missing reference from
    silently deleting the correctness net)."""
    if _REQUIRE_REFERENCES:
        pytest.fail(f"reference required but unavailable "
                    f"(--require-references): {reason}")
    pytest.skip(reason)


def reference_or_skip(plugin, func=None):
    """Return ``core.<plugin>`` (or ``core.<plugin>.<func>``).

    Skips when the reference plugin is not installed; with
    ``--require-references`` the absence is a failure instead.
    """
    ref = getattr(vs.core, plugin, None)
    name = plugin if func is None else f"{plugin}.{func}"
    if ref is not None and (func is None or hasattr(ref, func)):
        return ref
    skip_or_fail_reference(f"no {name} reference plugin")


_REF_OK = "REF ok"
_REF_UNAVAILABLE = "REF unavailable:"
_VSFEEL_FAIL = "VSFEEL fail:"
_RESULT = "RESULT "


def _as_text(data):
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return data


def _tail(text, n=25):
    lines = _as_text(text).splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"


def run_compare_subprocess(code, argv=(), timeout=600.0, env=None):
    """Run a comparison script in a subprocess and parse its report.

    Returns the JSON payload printed on the ``RESULT`` line.  ``env`` adds
    overrides to the inherited environment (e.g. a queue-cap knob).

    Raises :class:`ReferenceUnavailable` when the reference could not be
    materialised (missing plugin, its own evaluation failed, or the subprocess
    died/hung before printing ``REF ok``).  Raises :class:`AssertionError`,
    with the captured stdout/stderr tail, when the reference ran but the
    vsfeel side failed (exception, crash, timeout, malformed/missing result).
    """
    argv = [str(a) for a in argv]
    run_env = {**os.environ, **env} if env else None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, *argv],
            capture_output=True, text=True, timeout=timeout, env=run_env,
        )
        stdout = _as_text(proc.stdout)
        stderr = _as_text(proc.stderr)
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        returncode = None
        timed_out = True

    ref_ok = False
    ref_reason = None
    vsfeel_reason = None
    payload = None
    for line in stdout.splitlines():
        s = line.strip()
        if s == _REF_OK:
            ref_ok = True
        elif s.startswith(_REF_UNAVAILABLE):
            ref_reason = s[len(_REF_UNAVAILABLE):].strip()
        elif s.startswith(_VSFEEL_FAIL):
            vsfeel_reason = s[len(_VSFEEL_FAIL):].strip()
        elif s.startswith(_RESULT):
            payload = s[len(_RESULT):].strip()

    status = ("timed out after %.0fs" % timeout if timed_out
              else "exited %s" % returncode)
    diag = (
        "compare subprocess %s\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
        % (status, _tail(stdout), _tail(stderr))
    )

    if vsfeel_reason is not None:
        raise AssertionError(
            "vsfeel failed in comparison subprocess: %s\n%s"
            % (vsfeel_reason or "unspecified", diag))
    if ref_reason is not None:
        raise ReferenceUnavailable(ref_reason or "unspecified reference failure")
    if timed_out:
        if ref_ok:
            raise AssertionError(
                "comparison subprocess timed out after %.0fs; the reference "
                "materialised, so vsfeel or the comparison hung\n%s"
                % (timeout, diag))
        raise ReferenceUnavailable(
            "reference subprocess timed out after %.0fs before materialising\n%s"
            % (timeout, diag))
    if not ref_ok:
        raise ReferenceUnavailable(
            "reference subprocess failed before materialising (exit %s)\n%s"
            % (returncode, diag))
    if returncode != 0:
        raise AssertionError(
            "comparison subprocess exited %s after the reference materialised\n%s"
            % (returncode, diag))
    if payload is None:
        raise AssertionError(
            "comparison subprocess produced no RESULT line\n%s" % diag)
    try:
        return json.loads(payload)
    except ValueError as exc:
        raise AssertionError(
            "comparison RESULT was not valid JSON: %r\n%s" % (payload, diag)
        ) from exc


def compare_or_skip(code, argv=(), timeout=600.0):
    """Like :func:`run_compare_subprocess`, but skip on a bad reference."""
    try:
        return run_compare_subprocess(code, argv, timeout)
    except ReferenceUnavailable as exc:
        skip_or_fail_reference(f"reference unavailable: {exc}")


# Inline prelude for the compare scripts (they run under ``python -c`` and
# cannot import this module).  ``read_plane`` is the stride-aware, copying
# reader the subprocess uses for every plane.
COMPARE_PRELUDE = '''\
import ctypes
import numpy as np


def _plane_dims(frame, plane):
    fmt = frame.format
    ss_w = fmt.subsampling_w if plane > 0 and fmt.num_planes > 1 else 0
    ss_h = fmt.subsampling_h if plane > 0 and fmt.num_planes > 1 else 0
    return (((frame.width + (1 << ss_w) - 1) >> ss_w),
            ((frame.height + (1 << ss_h) - 1) >> ss_h))


def read_plane(frame, plane, dtype):
    """Copy one visible plane out of the frame, honouring the row pitch."""
    dt = np.dtype(dtype)
    w, h = _plane_dims(frame, plane)
    raw = np.ctypeslib.as_array(
        ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8)),
        shape=(h, frame.get_stride(plane)))
    return raw[:, :w * dt.itemsize].copy().view(dt).reshape(h, w)
'''


# ---------------------------------------------------------------------------
# Generic one-case reference comparison
# ---------------------------------------------------------------------------
#
# Reference plugins are crash-prone, so every reference comparison runs in a
# subprocess (see the protocol above).  A test describes its case with
# :func:`reference_spec` and calls :func:`reference_compare`; the script below
# builds the clip, materialises the reference before touching vsfeel, then
# compares the requested frames/planes.

REFERENCE_SCRIPT = COMPARE_PRELUDE + r'''
import json
import sys
import vapoursynth as vs

core = vs.core


def build_clip(spec):
    src = core.bs.VideoSource(spec["source"])
    gray = lambda: core.std.ShufflePlanes(src, 0, vs.GRAY)
    kind = spec["clip"]
    if kind == "gray8":
        clip = gray()
    elif kind == "gray16":
        clip = core.fmtc.bitdepth(gray(), bits=16, fulls=True, fulld=True)
    elif kind == "gray32":
        clip = core.fmtc.bitdepth(gray(), bits=32, fulls=True, fulld=True)
    elif kind == "yuv16":
        clip = core.fmtc.bitdepth(src, bits=16, fulls=True, fulld=True)
    elif kind == "yuv32":
        clip = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
    elif kind == "yuv420_16":
        clip = core.resize.Bicubic(src, format=vs.YUV420P16)
    elif kind == "yuv444_16":
        clip = core.resize.Bicubic(src, format=vs.YUV444P16)
    elif kind == "rgb32":
        clip = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")
    elif kind == "rgb16":
        clip = core.resize.Bicubic(src, format=vs.RGB48, matrix_in_s="709")
    else:
        raise SystemExit("bad clip kind %r" % kind)
    if spec.get("crop"):
        clip = core.std.Crop(clip, **spec["crop"])
    return clip


def build_guide(clip, spec):
    kind = (spec or {}).get("kind", "none")
    if kind == "none":
        return None
    if kind == "same":
        return clip
    if kind == "flipvertical":
        return core.std.FlipVertical(clip)
    if kind == "boxblur":
        return core.std.BoxBlur(clip, hradius=spec.get("hradius", 5),
                                vradius=spec.get("vradius", 5))
    if kind == "resize":
        return core.resize.Bicubic(clip, width=clip.width, height=clip.height,
                                   format=clip.format)
    raise SystemExit("bad guide kind %r" % kind)


spec = json.loads(sys.argv[1])
clip = build_clip(spec)
guide = build_guide(clip, spec.get("guide"))
frames = spec.get("frames", [0, 11, 23])
dt = np.float32 if clip.format.sample_type == vs.FLOAT else np.uint16
planes = spec.get("planes") or list(range(clip.format.num_planes))
kwargs = dict(spec.get("kwargs") or {})
if guide is not None:
    kwargs[spec.get("guide_kwarg", "ref")] = guide

ref_func = getattr(getattr(core, spec["plugin"], None), spec["filter"], None)

# --- reference phase: materialise and copy before touching vsfeel ---
try:
    if ref_func is None:
        raise RuntimeError("no %s.%s reference" % (spec["plugin"], spec["filter"]))
    ref_node = ref_func(clip, **kwargs)
    ref_frames = [[read_plane(ref_node.get_frame(n), p, dt) for p in planes]
                  for n in frames]
except Exception as exc:
    print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
    raise SystemExit(2)
print("REF ok", flush=True)

# --- vsfeel phase ---
try:
    my_node = getattr(core.vsfeel, spec.get("vsfeel_filter", spec["filter"]))(clip, **kwargs)
    maxdiff = 0.0
    ndiff = 0
    total = 0
    for n, ref_planes in zip(frames, ref_frames):
        frame = my_node.get_frame(n)
        for p, b in zip(planes, ref_planes):
            a = read_plane(frame, p, dt)
            if not (np.isfinite(a.astype(np.float64)).all()
                    and np.isfinite(b.astype(np.float64)).all()):
                print("VSFEEL fail: non-finite at frame %d plane %d" % (n, p), flush=True)
                raise SystemExit(3)
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            maxdiff = max(maxdiff, float(d.max()))
            ndiff += int((d > 0).sum())
            total += int(d.size)
except SystemExit:
    raise
except Exception as exc:
    print("VSFEEL fail: %s: %s" % (type(exc).__name__, exc), flush=True)
    raise SystemExit(3)
print("RESULT " + json.dumps({
    "maxdiff": maxdiff,
    "ndiff_frac": (ndiff / total) if total else 0.0,
    "width": my_node.width,
    "height": my_node.height,
    "num_frames": my_node.num_frames,
}), flush=True)
'''


def reference_spec(plugin, filter, clip, frames=(0, 11, 23), planes=None,
                   kwargs=None, guide=None, guide_kwarg="ref", crop=None,
                   vsfeel_filter=None):
    """Build the JSON spec consumed by :data:`REFERENCE_SCRIPT`.

    ``clip`` names the input format ("gray32", "gray16", "yuv420_16", ...);
    ``guide`` is a JSONable description of the joint-filter clip, if any.
    """
    spec = {
        "source": NOISE_MKV,
        "clip": clip,
        "plugin": plugin,
        "filter": filter,
        "frames": list(frames),
        "kwargs": dict(kwargs or {}),
    }
    if planes is not None:
        spec["planes"] = list(planes)
    if crop is not None:
        spec["crop"] = dict(crop)
    if guide is not None:
        spec["guide"] = dict(guide)
        spec["guide_kwarg"] = guide_kwarg
    if vsfeel_filter is not None:
        spec["vsfeel_filter"] = vsfeel_filter
    return spec


def reference_compare(spec, timeout=600.0):
    """Run one reference-vs-vsfeel comparison in a subprocess.

    Returns the RESULT payload: ``maxdiff``, ``ndiff_frac`` and the vsfeel
    output geometry.  A missing/crashing reference skips (or fails under
    ``--require-references``); a vsfeel crash, timeout or non-finite result
    fails with the captured subprocess tail.
    """
    return compare_or_skip(REFERENCE_SCRIPT, [json.dumps(spec)], timeout=timeout)


# ---------------------------------------------------------------------------
# Temporal frame-request order
# ---------------------------------------------------------------------------
#
# A filter that caches temporal state must return the same pixels whatever order
# the graph asks for its frames in: vspipe requests frames concurrently and the
# scheduler re-requests in-flight frames (DFTTest's notes prove it), which is the
# only pattern that exposes a cache holder/lifetime bug.  Each ordering runs on
# a *fresh* node under the subprocess timeout, while the num_streams=1 serial
# run is the oracle, so a hang fails the test instead of the suite.
#
# The orders are fixed (no RNG): forward proves the oracle path, reverse and
# "far" request temporally distant neighbours, interleave and scramble thrash
# the frame cache, and "revisit" re-requests frames the node already served.

TEMPORAL_ORDER_SCRIPT = COMPARE_PRELUDE + r'''
import json
import sys
from math import gcd

import vapoursynth as vs

spec = json.loads(sys.argv[1])
core = vs.core
core.max_cache_size = spec.get("max_cache_size", 256)

src = core.bs.VideoSource(spec["source"])
clip = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                          bits=32, fulls=True, fulld=True)
if spec.get("nframes"):
    clip = core.std.Loop(clip, times=spec["nframes"])

nf = clip.num_frames
plane = spec.get("plane", 0)
kwargs = spec["params"]

# A stride coprime with nf visits every frame exactly once while jumping around;
# keep it deterministic so a failure is reproducible.
key = next(k for k in range(3, 2 * nf + 3, 2) if gcd(k, nf) == 1)
scramble = [(i * key) % nf for i in range(nf)]
far = sorted(range(nf), key=lambda n: (abs(n - nf // 2), n))
interleave = list(range(0, nf, 2)) + [n for n in range(nf) if n % 2][::-1]
revisit = [n for n in [0, 3, 3, 1, 5, 1, nf - 1, 0, nf - 2, 2, nf - 1]]
orders = {
    "forward": list(range(nf)),
    "reverse": list(range(nf))[::-1],
    "far": far,
    "interleave": interleave,
    "scramble": scramble,
    "revisit": revisit,
}


def run(order):
    node = getattr(core.vsfeel, spec["filter"])(clip, **kwargs)
    return {n: read_plane(node.get_frame(n), plane, np.float32).copy() for n in order}


try:
    ref = run(orders["forward"])
except Exception as exc:
    print("VSFEEL fail: serial run: %s: %s" % (type(exc).__name__, exc), flush=True)
    raise SystemExit(3)
print("REF ok", flush=True)

worst = {}
for name, order in orders.items():
    try:
        got = run(order)
    except Exception as exc:
        print("VSFEEL fail: %s: %s: %s" % (name, type(exc).__name__, exc), flush=True)
        raise SystemExit(3)
    w = 0.0
    for n in order:
        a, b = got[n].astype(np.float64), ref[n].astype(np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print("VSFEEL fail: non-finite, order %s frame %d" % (name, n), flush=True)
            raise SystemExit(3)
        w = max(w, float(np.abs(a - b).max()))
    worst[name] = w

print("RESULT " + json.dumps({"orders": worst, "num_frames": nf}), flush=True)
'''


def temporal_order_diff(filter_name, params, nframes=None, plane=0,
                        timeout=600.0):
    """Worst pixel diff per frame-request ordering, against the serial run.

    ``filter_name`` is the ``core.vsfeel`` function; ``params`` its keyword
    arguments (temporal radius/window set by the caller); ``nframes`` loops the
    clip to that length, which is how the 1- and 2-frame cases are built.  Skips
    from the reference protocol are turned into failures: there is no external
    reference here, so anything that dies before the oracle completes is a
    vsfeel failure, not a missing plugin.
    """
    spec = {"source": NOISE_MKV, "filter": filter_name, "params": params,
            "plane": plane, "max_cache_size": 256}
    if nframes is not None:
        spec["nframes"] = int(nframes)
    try:
        payload = run_compare_subprocess(TEMPORAL_ORDER_SCRIPT,
                                         [json.dumps(spec)], timeout=timeout)
    except ReferenceUnavailable as exc:
        raise AssertionError(
            f"temporal-order subprocess failed before the oracle completed: {exc}"
        ) from exc
    return payload["orders"]


def assert_temporal_order_consistent(filter_name, params, tol=1e-5,
                                     nframes=None, plane=0, timeout=600.0):
    """Every request order must reproduce the serial result within ``tol``."""
    worst = temporal_order_diff(filter_name, params, nframes=nframes,
                                plane=plane, timeout=timeout)
    bad = {name: w for name, w in worst.items() if w > tol}
    assert not bad, (
        f"{filter_name} output depends on frame-request order "
        f"(tol {tol:g}, params {params}): {bad}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def noise_gray():
    """GrayS float32 clip of the committed 24-frame random-noise video."""
    src = _source(NOISE_MKV)
    y = vs.core.std.ShufflePlanes(src, 0, vs.GRAY)
    return vs.core.fmtc.bitdepth(y, bits=32, fulls=True, fulld=True)


@pytest.fixture(scope="session")
def noise_8bit():
    """GRAY8 clip of the same video (for input-validation tests)."""
    src = _source(NOISE_MKV)
    return vs.core.std.ShufflePlanes(src, 0, vs.GRAY)


@pytest.fixture(scope="session")
def noise_16bit():
    """GRAY16 clip of the same video (integer reference comparison)."""
    src = _source(NOISE_MKV)
    y = vs.core.std.ShufflePlanes(src, 0, vs.GRAY)
    return vs.core.fmtc.bitdepth(y, bits=16, fulls=True, fulld=True)


def assert_gray32(clip):
    fmt = clip.format
    assert fmt.color_family == vs.GRAY
    assert fmt.sample_type == vs.FLOAT
    assert fmt.bits_per_sample == 32
    assert clip.width == WIDTH
    assert clip.height == HEIGHT
