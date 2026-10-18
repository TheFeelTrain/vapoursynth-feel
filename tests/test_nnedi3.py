"""Unit tests for core.vsfeel.NNEDI3.

The committed tests/bigbuckbunny_360p_grain.mp4 clip (real 360p content with
baked-in grain) is used as the input. The vsfeel implementation is a from-scratch Vulkan port of the
NNEDI3 predictor/prescreener math; the reference oracle is nnedi3vk (Vulkan),
which agrees BIT-EXACTLY with vszipcu's NNEDI3 (measured max diff 0 on the
noise clip, all planes, frames 0/5/11 — so matching nnedi3vk matches both).

Agreement measured on the noise clip (see notes/NNEDI3.md):

* GRAY16 / YUV420P16, every mode tested (field 0/1/2/3, dh, planes subsets,
  nsize 0..6, nns 0..4, qual 1/2, etype 0/1, pscrn 0..4): vsfeel == nnedi3vk
  within 1 LSB (max diff 0 on most content, 1 on rare rounding flips from
  the serial-vs-subgroup reduction order — mechanism documented in
  notes/NNEDI3.md).
* GRAYS: within ~3e-8 absolute (a few ulps); bound 1e-6.
* GRAYH: BIT-EXACT with nnedi3vk's f16 output when the packed half predictor
  runs (the default on a device with shaderFloat16), because both run the same
  half GEMV. The storage-only fallback (VSFEEL_NNEDI3_PRED16=0, or a device
  without the feature) instead runs the f32 arithmetic and lands within the
  reference's own f16 error, ~2.3e-3.

Self-consistency (determinism across runs, multi-stream == single) is exact.

Run from the repository root:  uv run python -m pytest tests/test_nnedi3.py
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import vapoursynth as vs

from conftest import (
    WIDTH,
    HEIGHT,
    CLIP_FRAMES,
    CLIP_PATH,
    COMPARE_PRELUDE,
    assert_changes_on_clip,
    assert_preserves_frame_props,
    compare_or_skip,
    cpu_node,
    eval_parallel,
    frame_to_ndarray,
    plane as _plane,
    reference_compare,
    reference_or_skip,
    reference_spec,
)

pytestmark = pytest.mark.usefixtures("clip_grays")


def _run(clip, field=1, **kwargs):
    """NNEDI3 as a clip the test can read pixels from (see conftest.cpu_node).

    Under the R80 GPU API the filter takes and returns ``vnode:gpu`` frames, so
    a CPU clip is auto-uploaded by the core and the output is downloaded before
    the host reads it.
    """
    return cpu_node(
        vs.core.vsfeel.NNEDI3(
            clip,
            field=field,
            **kwargs,
        )
    )


def _ref_compare(fmt, frames, params, planes=None, env=None):
    """Compare against nnedi3vk in a subprocess; skip if it is not installed.

    Returns the RESULT payload (``maxdiff`` plus the vsfeel output geometry).
    ``field`` defaults to 1 (both plugins require it). ``env`` pins a
    creation-time variant (the plugin reads its knobs when the filter is built,
    so an in-process node cannot be steered per test).
    """
    reference_or_skip("nnedi3vk", "NNEDI3")
    params = dict(params)
    params.setdefault("field", 1)
    spec = reference_spec("nnedi3vk", "NNEDI3", fmt, frames=frames, planes=planes, kwargs=params)
    return reference_compare(spec, env=env)


# ---------------------------------------------------------------------------
# Reference agreement, GRAY16
# ---------------------------------------------------------------------------

REFERENCE_CASES_U16 = [
    {},
    {"pscrn": 1},
    {"pscrn": 3},
    {"pscrn": 4},
    {"qual": 2},
    {"etype": 1},
    {"qual": 2, "etype": 1},
    {"nsize": 0},
    {"nsize": 3},
    {"nsize": 4},
    {"nns": 0},
    {"nns": 2},
    {"nns": 4},
    {"field": 0},
    # gaps in the nsize / nns sweeps; measured 0 LSB on the noise clip
    {"nsize": 1},
    {"nsize": 2},
    {"nsize": 5},
    {"nns": 1},
    {"nns": 3},
]


@pytest.mark.parametrize(
    "kwargs", REFERENCE_CASES_U16, ids=[str(sorted(k.items())) for k in REFERENCE_CASES_U16]
)
def test_nnedi3_matches_reference_u16(clip_gray16, kwargs):
    payload = _ref_compare("gray16", (0, 11, 23), kwargs)
    assert payload["maxdiff"] <= 1, kwargs


def test_nnedi3_pscrn0_within_1lsb(clip_gray16):
    """pscrn=0 forces every pixel through the predictor (serial accumulation
    vs the reference's subgroup reductions): allow 1 LSB."""
    payload = _ref_compare("gray16", (0, 11), {"pscrn": 0})
    assert payload["maxdiff"] <= 1


def test_nnedi3_field_gt1_matches_reference(clip_gray16):
    for field in (2, 3):
        payload = _ref_compare("gray16", (0, 1, 24, 47), {"field": field})
        assert payload["num_frames"] == 48
        assert payload["maxdiff"] <= 1


def test_nnedi3_dh_matches_reference(clip_gray16):
    payload = _ref_compare("gray16", (0, 11, 23), {"dh": True})
    assert payload["height"] == 2 * HEIGHT
    assert payload["maxdiff"] <= 1


def _yuv420p16():
    core = vs.core
    if hasattr(core, "bs"):
        src = core.bs.VideoSource(CLIP_PATH)
    else:
        src = core.ffms2.Source(CLIP_PATH)
    return src.resize.Point(format=vs.YUV420P16)


def test_nnedi3_yuv_matches_reference():
    yuv = _yuv420p16()
    assert yuv.format.color_family == vs.YUV
    # dh doubles the luma height; the chroma writers have to cover the whole
    # (2*h) >> subSamplingH plane, last row included.
    for kwargs in ({}, {"planes": [0]}, {"planes": [1, 2]}, {"nsize": 0}, {"dh": True}):
        payload = _ref_compare("yuv420p16", (0, 11), kwargs)
        assert payload["maxdiff"] <= 1, kwargs


# ---------------------------------------------------------------------------
# Reference agreement, float32 (bit-exact floats on configs tried)
# ---------------------------------------------------------------------------

REFERENCE_CASES_F32 = [
    {},
    {"pscrn": 0},
    {"pscrn": 1},
    {"qual": 2},
    {"etype": 1},
    {"nsize": 0},
    {"nns": 4},
    {"field": 0},
    # float-path gaps: field 2/3 and dh (u16 only before)
    {"field": 2},
    {"field": 3},
    {"dh": True},
]


@pytest.mark.parametrize(
    "kwargs", REFERENCE_CASES_F32, ids=[str(sorted(k.items())) for k in REFERENCE_CASES_F32]
)
def test_nnedi3_matches_reference_f32(clip_grays, kwargs):
    payload = _ref_compare("grays", (0, 11), kwargs)
    assert payload["maxdiff"] <= 1e-6, kwargs


# ---------------------------------------------------------------------------
# Reference agreement, float16
# ---------------------------------------------------------------------------
#
# Half is an io width for most filters, but NNEDI3 computes in it: with the
# optional shaderFloat16 feature the predictor GEMV runs packed half (the
# arithmetic nnedi3vk's own f16 kernel uses), which makes vsfeel's f16 output
# BIT-EXACT with nnedi3vk's f16 output while sitting ~2.3e-3 off this file's
# f32 path — that gap is the reference's own f16 error, not an extra one of
# ours. Without the feature, or with VSFEEL_NNEDI3_PRED16=0, the f16 kernels
# keep the f32 arithmetic and land within one half step of the f32 path
# (measured 0 on the committed clip).
#
# The default-mode bound therefore has to cover the reference's f16 error; the
# two pinned modes are graded exactly, and the storage-only path is separately
# held against the f32 path.

F16_PACKED_MAX = 2.6e-3  # measured 2.51e-3 (nsize=3) on identical inputs
F16_STORAGE_ULP = 4.8828125e-4  # one half step at |v| in [0.5, 1)

REFERENCE_CASES_F16 = [
    {},
    {"pscrn": 0},
    {"pscrn": 1},
    {"pscrn": 3},
    {"pscrn": 4},
    {"qual": 2},
    {"etype": 1},
    {"qual": 2, "etype": 1},
    {"nsize": 0},
    {"nsize": 3},
    {"nsize": 6},
    {"nns": 0},
    {"nns": 2},
    {"nns": 4},
    {"field": 0},
    {"field": 2},
    {"field": 3},
    {"dh": True},
]


@pytest.mark.parametrize(
    "kwargs", REFERENCE_CASES_F16, ids=[str(sorted(k.items())) for k in REFERENCE_CASES_F16]
)
def test_nnedi3_matches_reference_f16(clip_grayh, kwargs):
    """Default f16 mode, whatever the device selects (packed or storage-only)."""
    payload = _ref_compare("grayh", (0, 11, 23), kwargs)
    assert payload["maxdiff"] <= F16_PACKED_MAX, kwargs


F16_EXACT_CASES = [
    {},
    {"pscrn": 0},
    {"pscrn": 1},
    {"nsize": 3},
    {"nns": 4},
    {"field": 0},
    {"field": 3},
    {"dh": True},
    {"qual": 2, "etype": 1},
]


@pytest.mark.parametrize(
    "kwargs", F16_EXACT_CASES, ids=[str(sorted(k.items())) for k in F16_EXACT_CASES]
)
def test_nnedi3_f16_packed_bit_exact(clip_grayh, kwargs):
    """The packed half predictor must reproduce nnedi3vk's f16 output exactly.

    Both run the same half GEMV over the same rounded weights with the same
    window centering and f32 row-sum correction, so any difference here is a
    real indexing or layout bug (this is the bit-exact oracle for the path).
    """
    payload = _ref_compare("grayh", (0, 11), kwargs, env={"VSFEEL_NNEDI3_PRED16": "1"})
    assert payload["maxdiff"] == 0.0, kwargs


@pytest.mark.parametrize(
    "kwargs", F16_EXACT_CASES, ids=[str(sorted(k.items())) for k in F16_EXACT_CASES]
)
def test_nnedi3_f16_storage_only_matches_reference(clip_grayh, kwargs):
    """The storage-only f16 path is graded against the reference's own f16
    arithmetic, so the bound is the reference's error, not ours."""
    payload = _ref_compare("grayh", (0, 11), kwargs, env={"VSFEEL_NNEDI3_PRED16": "0"})
    assert payload["maxdiff"] <= F16_PACKED_MAX, kwargs


# The device-independent half of the oracle: f16 in/out against the f32 path
# on *identical* input values. The half clip is built first and the f32 clip is
# widened from it (exact), because a half clip rounded from an f32 one carries
# different samples and the window normalization amplifies that by 1/std — the
# comparison then measures the input rounding, not the arithmetic.
_F16_VS_F32_SCRIPT = (
    COMPARE_PRELUDE
    + r"""
import json
import sys

import vapoursynth as vs

core = vs.core
src = core.bs.VideoSource(sys.argv[1])
src = core.std.Trim(src, first=0, length=int(sys.argv[3]))
gray = core.std.ShufflePlanes(src, 0, vs.GRAY)
g16 = core.fmtc.bitdepth(gray, bits=16, fulls=True, fulld=True)
half = core.resize.Point(core.fmtc.bitdepth(g16, bits=32, fulls=True, fulld=True),
                         format=vs.GRAYH)
f32 = core.resize.Point(half, format=vs.GRAYS)
params = json.loads(sys.argv[2])
frames = (0, 11, 23)


def run(clip):
    node = cpu_node(core.vsfeel.NNEDI3(clip, **params))
    dt = np.float16 if clip.format.bits_per_sample == 16 else np.float32
    return {n: read_plane(node.get_frame(n), 0, dt).astype(np.float64) for n in frames}


try:
    ref = run(f32)
except Exception as exc:
    print("REF unavailable: %s: %s" % (type(exc).__name__, exc), flush=True)
    raise SystemExit(2)
print("REF ok", flush=True)

try:
    got = run(half)
except Exception as exc:
    print("VSFEEL fail: %s: %s" % (type(exc).__name__, exc), flush=True)
    raise SystemExit(3)

worst = max(float(np.abs(got[n] - ref[n]).max()) for n in frames)
print("RESULT " + json.dumps({"maxdiff": worst}), flush=True)
"""
)


@pytest.mark.parametrize(
    "kwargs", [{}, {"pscrn": 0}, {"nsize": 3}, {"dh": True}], ids=lambda k: str(sorted(k.items()))
)
def test_nnedi3_f16_storage_only_matches_f32_path(kwargs):
    """f16 io with f32 arithmetic is the f32 path within one half step."""
    payload = compare_or_skip(
        _F16_VS_F32_SCRIPT,
        [CLIP_PATH, json.dumps(dict(kwargs, field=1)), str(CLIP_FRAMES)],
        env={"VSFEEL_NNEDI3_PRED16": "0"},
    )
    assert payload["maxdiff"] <= F16_STORAGE_ULP, kwargs


@pytest.mark.parametrize(
    "kwargs", [{}, {"pscrn": 0}, {"nsize": 3}, {"dh": True}], ids=lambda k: str(sorted(k.items()))
)
def test_nnedi3_f16_matches_f32_path_within_reference(kwargs):
    """The packed half arithmetic costs at most nnedi3vk's own f16 error.

    Measured identical to it (2.03e-3 at the default config) because the packed
    kernel *is* the reference's f16 arithmetic; a device without shaderFloat16
    falls back to the storage-only path, which is tighter still.
    """
    payload = compare_or_skip(
        _F16_VS_F32_SCRIPT,
        [CLIP_PATH, json.dumps(dict(kwargs, field=1)), str(CLIP_FRAMES)],
    )
    assert payload["maxdiff"] <= F16_PACKED_MAX, kwargs


def test_nnedi3_f16_field_gt1_matches_reference(clip_grayh):
    for field in (2, 3):
        payload = _ref_compare("grayh", (0, 1, 24, 47), {"field": field})
        assert payload["num_frames"] == 48
        assert payload["maxdiff"] <= F16_PACKED_MAX


def test_nnedi3_f16_yuv_matches_reference(clip_yuv420ps):
    yuv = vs.core.resize.Point(clip_yuv420ps, format=vs.YUV420PH)
    assert yuv.format.id == vs.YUV420PH
    # dh doubles the luma height; the chroma writers have to cover the whole
    # (2*h) >> subSamplingH plane, last row included.
    for kwargs in ({}, {"planes": [0]}, {"planes": [1, 2]}, {"nsize": 0}, {"dh": True}):
        payload = _ref_compare("yuv420ph", (0, 11), kwargs)
        assert payload["maxdiff"] <= F16_PACKED_MAX, kwargs


def _creation_stderr(env=None):
    """stderr of a GRAYH creation run, with the device banner on.

    The predictor variant is chosen from a device feature and an env knob, so
    it is invisible in the pixels; the creation banner is the only place a test
    can pin it (notes/METHOD.md: a creation-time variant needs a test).
    """
    code = (
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        f"src = core.bs.VideoSource({CLIP_PATH!r})\n"
        "gray = core.std.ShufflePlanes(src, 0, vs.GRAY)\n"
        "f32 = core.fmtc.bitdepth(gray, bits=32, fulls=True, fulld=True)\n"
        "half = core.resize.Point(f32, format=vs.GRAYH)\n"
        "node = core.vsfeel.NNEDI3(half, field=1)\n"
        "core.std.GPUDownload(clip=node).get_frame(0)\n"
    )
    run_env = {**os.environ, "MANGOHUD": "0", "VSFEEL_DEBUG": "1", **(env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=run_env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stderr


def test_nnedi3_f16_predictor_variant_pinned():
    """The packed predictor runs by default where the device supports it, and
    the opt-out selects the storage-only one."""
    default = _creation_stderr()
    feature = re.search(r"shaderFloat16=(\d)", default)
    assert feature, default[-2000:]
    if feature.group(1) == "1":
        assert "predictor=f16-packed" in default, default[-2000:]
    storage = _creation_stderr({"VSFEEL_NNEDI3_PRED16": "0"})
    assert "predictor=f16-storage" in storage, storage[-2000:]


# ---------------------------------------------------------------------------
# Determinism / stream count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [16, 32])
def test_nnedi3_deterministic(clip_grays, clip_gray16, bits):
    clip = clip_gray16 if bits == 16 else clip_grays
    a = _run(clip, nsize=0)
    b = _run(clip, nsize=0)
    for n in (0, 11, 23):
        fa, fb = a.get_frame(n), b.get_frame(n)
        if bits == 16:
            d = _plane(fa, 0, WIDTH, HEIGHT).astype(np.int32) - _plane(fb, 0, WIDTH, HEIGHT).astype(
                np.int32
            )
        else:
            d = frame_to_ndarray(fa) - frame_to_ndarray(fb)
        assert np.abs(d).max() == 0, f"nondeterministic output at frame {n}"


def test_nnedi3_parallel_load_consistent(clip_gray16):
    expect = _plane(_run(clip_gray16).get_frame(5), 0)
    got = eval_parallel(_run, clip_gray16)
    assert np.abs(got[5].astype(np.int32) - expect.astype(np.int32)).max() == 0


def test_nnedi3_f16_deterministic(clip_grayh):
    a = _run(clip_grayh, nsize=0)
    b = _run(clip_grayh, nsize=0)
    for n in (0, 11, 23):
        d = _plane(a.get_frame(n), 0).astype(np.float64) - _plane(b.get_frame(n), 0).astype(
            np.float64
        )
        assert np.abs(d).max() == 0, f"nondeterministic f16 output at frame {n}"


def test_nnedi3_f16_parallel_load_consistent(clip_grayh):
    """The packed half kernel's shared tile and subgroup reductions must not
    depend on how many frames are in flight."""
    expect = _plane(_run(clip_grayh).get_frame(5), 0)
    got = eval_parallel(_run, clip_grayh)
    assert np.abs(got[5].astype(np.float64) - expect.astype(np.float64)).max() == 0


# ---------------------------------------------------------------------------
# Frame geometry / properties
# ---------------------------------------------------------------------------


def test_nnedi3_field_gt1_doubles_frames(clip_gray16):
    out = _run(clip_gray16, field=3)
    assert out.num_frames == 2 * clip_gray16.num_frames
    assert out.fps_num * clip_gray16.fps_den == 2 * clip_gray16.fps_num * out.fps_den


def test_nnedi3_dh_doubles_height(clip_gray16):
    out = _run(clip_gray16, dh=True)
    assert out.height == 2 * clip_gray16.height
    assert out.width == clip_gray16.width


def test_nnedi3_output_props_progressive(clip_gray16):
    out = _run(clip_gray16, field=3)
    props = out.get_frame(1).props
    assert props["_FieldBased"] == 0
    assert "_Field" not in props


def test_nnedi3_kept_lines_copied(clip_gray16):
    """Lines the filter keeps must equal the source field rows exactly.
    Progressive content with field=1 keeps parity 0 (even output lines)."""
    out = _run(clip_gray16, field=1)
    for n in (0, 7):
        s = _plane(clip_gray16.get_frame(n), 0, WIDTH, HEIGHT)
        o = _plane(out.get_frame(n), 0, WIDTH, HEIGHT)
        assert np.abs(o[0::2].astype(np.int32) - s[0::2].astype(np.int32)).max() == 0


def test_nnedi3_changes_noise_u16(clip_gray16):
    """The interpolated rows must actually alter the noise input.

    ``isfinite(uint16)`` cannot fail and the reference comparison skips when
    nnedi3vk is absent, so this is the invariant that fails for an identity
    implementation.
    """
    out = _run(clip_gray16, field=1)
    assert_changes_on_clip(out, clip_gray16, what="NNEDI3")


def test_nnedi3_preserves_frame_props(clip_gray16):
    """field=1 keeps the duration while tagging the output progressive."""
    assert_preserves_frame_props(_run, clip_gray16, field=1)


# --- f16 geometry / properties, mirroring the u16 and f32 cases above -------


def test_nnedi3_f16_field_gt1_doubles_frames(clip_grayh):
    out = _run(clip_grayh, field=3)
    assert out.num_frames == 2 * clip_grayh.num_frames
    assert out.fps_num * clip_grayh.fps_den == 2 * clip_grayh.fps_num * out.fps_den


def test_nnedi3_f16_dh_doubles_height(clip_grayh):
    out = _run(clip_grayh, dh=True)
    assert out.height == 2 * clip_grayh.height
    assert out.width == clip_grayh.width


def test_nnedi3_f16_output_props_progressive(clip_grayh):
    out = _run(clip_grayh, field=3)
    props = out.get_frame(1).props
    assert props["_FieldBased"] == 0
    assert "_Field" not in props


def test_nnedi3_f16_kept_lines_copied(clip_grayh):
    """Kept lines are raw plane copies in every io variant, so a half clip's
    kept rows must come back bit-identical (the packed predictor never touches
    them)."""
    out = _run(clip_grayh, field=1)
    for n in (0, 7):
        s = _plane(clip_grayh.get_frame(n), 0)
        o = _plane(out.get_frame(n), 0)
        assert np.abs(o[0::2].astype(np.float64) - s[0::2].astype(np.float64)).max() == 0


def test_nnedi3_f16_changes_noise(clip_grayh):
    out = _run(clip_grayh, field=1)
    assert_changes_on_clip(out, clip_grayh, what="NNEDI3 f16")


def test_nnedi3_f16_preserves_frame_props(clip_grayh):
    assert_preserves_frame_props(_run, clip_grayh, field=1)


def test_nnedi3_f16_accepts_half(clip_grayh):
    """A half clip keeps its own format: a u16 kernel fed half samples would
    reinterpret the bit patterns (or an f32 kernel would read two samples per
    element), so the format id is part of the contract."""
    out = _run(clip_grayh, field=1)
    assert out.format.id == clip_grayh.format.id
    assert out.format.sample_type == vs.FLOAT
    assert out.format.bits_per_sample == 16


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_nnedi3_requires_field(clip_gray16):
    """`field` is a required argument (EEDI3 does the same).

    It used to be registered as optional but read with a null error pointer,
    so omitting it took VapourSynth's VS_FATAL_ERROR path (`fprintf` +
    `std::terminate`): the process died (SIGABRT) and Python could not catch
    it. It must now raise a normal, catchable vs.Error.
    """
    with pytest.raises(vs.Error):
        vs.core.vsfeel.NNEDI3(clip_gray16)


def test_nnedi3_rejects_8bit(clip_gray8):
    with pytest.raises(vs.Error):
        _run(clip_gray8).get_frame(0)


def test_nnedi3_rejects_bad_field(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, field=4)


def test_nnedi3_rejects_dh_with_field_gt1(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, field=3, dh=True)


def test_nnedi3_rejects_bad_nsize(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, nsize=7)


def test_nnedi3_rejects_bad_nns(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, nns=5)


def test_nnedi3_rejects_bad_qual(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, qual=3)


def test_nnedi3_rejects_bad_etype(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, etype=2)


def test_nnedi3_rejects_bad_pscrn(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, pscrn=5)


def test_nnedi3_rejects_bad_planes(clip_gray16):
    with pytest.raises(vs.Error):
        _run(clip_gray16, planes=[0, 0])
    with pytest.raises(vs.Error):
        _run(clip_gray16, planes=[3])


def test_nnedi3_rejects_odd_height(clip_gray16):
    odd = clip_gray16.std.Crop(bottom=1)
    with pytest.raises(vs.Error):
        _run(odd).get_frame(0)


def test_nnedi3_rejects_dh_over_int32_height():
    """dh doubles vi_out.height, an int, so it has to refuse heights past
    INT32_MAX/2 *before* the multiply (the geometry check only fires after the
    signed overflow). BlankClip builds the node without allocating the frame,
    so the guard is exercised at creation. The message is asserted because a
    wrapped height would otherwise still be rejected, with a different error.
    """
    clip = vs.core.std.BlankClip(width=1, height=1 << 30, format=vs.GRAY16, length=1)
    with pytest.raises(vs.Error, match="too tall"):
        vs.core.vsfeel.NNEDI3(clip, field=1, dh=True)
