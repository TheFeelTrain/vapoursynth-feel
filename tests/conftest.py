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


class ReferenceUnavailable(RuntimeError):
    """The reference plugin or the reference-side evaluation failed."""


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


def run_compare_subprocess(code, argv=(), timeout=600.0):
    """Run a comparison script in a subprocess and parse its report.

    Returns the JSON payload printed on the ``RESULT`` line.

    Raises :class:`ReferenceUnavailable` when the reference could not be
    materialised (missing plugin, its own evaluation failed, or the subprocess
    died/hung before printing ``REF ok``).  Raises :class:`AssertionError`,
    with the captured stdout/stderr tail, when the reference ran but the
    vsfeel side failed (exception, crash, timeout, malformed/missing result).
    """
    argv = [str(a) for a in argv]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, *argv],
            capture_output=True, text=True, timeout=timeout,
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
        pytest.skip("reference unavailable: %s" % exc)


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
