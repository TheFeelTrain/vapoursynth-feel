"""``num_streams`` matrix: every filter must match its serial run exactly.

The plugin ships a different default stream count per filter (EEDI3 8;
NNEDI3 / BM3D / Bilateral / GaussBlur 4; NLMeans 2; DFTTest 1) and only a
couple of the small counts were exercised anywhere before.  This module sweeps
the full ``{1, 3, 4, 6, 8, 16, 32}`` matrix for every filter on two clips:

* ``noise24`` — the committed 640x360 24-frame random-noise clip.
* ``tiny``    — a 64x64 4-frame crop of it: small geometry, and fewer frames
  than the large stream counts, so 16 and 32 streams must still match serial.

The multi-stream node is evaluated the way vspipe requests it — every frame
requested concurrently from worker threads — because sequential ``get_frame``
calls keep at most one frame in flight and never enter the pipelined path.
Every frame is compared bit-exactly (``np.array_equal``) against a fresh
``num_streams=1`` run; for a float32 pipeline that is the strictest possible
check and catches any stream-dependent reordering of the maths.

BM3D is the one exception, and only because its *default* aggregation is not
deterministic to begin with: the kernel sums per-group contributions with float
``atomicAdd``, so the result depends on scheduling order and even two
``num_streams=1`` runs differ by ~3e-8.  ``extractor_exp >= 3`` pre-rounds the
addends (the reference's ``(x + E) - E``) and makes the sums order-independent,
so the matrix uses ``extractor_exp=8`` where the reproducibility guarantee
applies — num_streams is still what varies.  The default is covered, with the
same measured band, by ``test_bm3dv2_extractor_exp_is_bit_reproducible`` and
the multi-stream comparison in ``test_bm3dv2.py``.

Run from the repository root:  python -m pytest tests/test_streams.py
"""

import threading

import numpy as np
import pytest
import vapoursynth as vs

from conftest import format_dtype, plane_to_ndarray

STREAM_COUNTS = (1, 3, 4, 6, 8, 16, 32)
TINY_SIZE = 64
TINY_FRAMES = 4

# Per-filter builders.  Parameters are representative configs that are valid
# on both clips and keep the run cheap enough for the whole matrix.
def _dfttest(clip, num_streams):
    return vs.core.vsfeel.DFTTest(clip, tbsize=3, num_streams=num_streams)


def _gaussblur(clip, num_streams):
    return vs.core.vsfeel.GaussBlur(clip, sigma=2.0, num_streams=num_streams)


def _bilateral(clip, num_streams):
    return vs.core.vsfeel.Bilateral(
        clip, sigma_spatial=3.0, sigma_color=0.05, num_streams=num_streams)


def _bm3d(clip, num_streams):
    # extractor_exp makes the float atomic aggregation order-independent; see
    # the module docstring.
    return vs.core.vsfeel.BM3Dv2(
        clip, sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4,
        extractor_exp=8, num_streams=num_streams)


def _eedi3(clip, num_streams):
    return vs.core.vsfeel.EEDI3(
        clip, field=1, mdis=5, nrad=1, vcheck=2, num_streams=num_streams)


def _eedi3h(clip, num_streams):
    return vs.core.vsfeel.EEDI3H(
        clip, field=1, mdis=5, nrad=1, vcheck=2, num_streams=num_streams)


def _eedi3aa(clip, num_streams):
    # field 3 is the two-call based_aa form EEDI3AA reproduces.
    return vs.core.vsfeel.EEDI3AA(
        clip, field=3, mdis=5, nrad=1, vcheck=2, num_streams=num_streams)


def _nnedi3(clip, num_streams):
    return vs.core.vsfeel.NNEDI3(clip, field=1, num_streams=num_streams)


def _nlmeans(clip, num_streams):
    return vs.core.vsfeel.NLMeans(
        clip, d=1, a=3, s=3, h=3.0, wref=0.4, num_streams=num_streams)


CASES = [
    pytest.param("dfttest", _dfttest, id="dfttest"),
    pytest.param("gaussblur", _gaussblur, id="gaussblur"),
    pytest.param("bilateral", _bilateral, id="bilateral"),
    pytest.param("bm3d", _bm3d, id="bm3d"),
    pytest.param("eedi3", _eedi3, id="eedi3"),
    pytest.param("eedi3h", _eedi3h, id="eedi3h"),
    pytest.param("eedi3aa", _eedi3aa, id="eedi3aa"),
    pytest.param("nnedi3", _nnedi3, id="nnedi3"),
    pytest.param("nlmeans", _nlmeans, id="nlmeans"),
]


@pytest.fixture(scope="module")
def stream_clips(noise_gray):
    """The two matrix clips: the full 24-frame input and a 64x64 tiny crop."""
    tiny = vs.core.std.Crop(noise_gray, right=noise_gray.width - TINY_SIZE,
                            bottom=noise_gray.height - TINY_SIZE)
    tiny = vs.core.std.Trim(tiny, first=0, last=TINY_FRAMES - 1)
    return {"noise24": noise_gray, "tiny": tiny}


def _request_all(node, dtype):
    """Request every frame concurrently, the way vspipe drives the filter.

    Worker exceptions are collected and re-raised on the main thread: a
    failure (or a filter-side crash) must fail the test rather than leave a
    ``None`` slot behind.
    """
    frames = [None] * node.num_frames
    errors = []

    def work(n):
        try:
            frames[n] = plane_to_ndarray(node.get_frame(n), 0, dtype)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append((n, exc))

    threads = [threading.Thread(target=work, args=(n,))
               for n in range(node.num_frames)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise AssertionError(
            "frame request failed at frame %d: %r" % (errors[0][0], errors[0][1]))
    return frames


@pytest.mark.parametrize("clip_name", ("noise24", "tiny"))
@pytest.mark.parametrize("name,build", CASES)
def test_num_streams_matches_serial(stream_clips, name, build, clip_name):
    """Every stream count must reproduce the serial output bit-exactly.

    The serial run is computed once and every count in the matrix — including
    a second fresh ``num_streams=1`` node, which doubles as a determinism
    check — is compared against it frame by frame.  All disagreements are
    collected so the report names every offending count, not just the first.
    """
    clip = stream_clips[clip_name]
    dtype = format_dtype(clip.format)
    ref = [plane_to_ndarray(build(clip, 1).get_frame(n), 0, dtype)
           for n in range(clip.num_frames)]

    bad = []
    for num_streams in STREAM_COUNTS:
        out = _request_all(build(clip, num_streams), dtype)
        differing = [n for n in range(clip.num_frames)
                     if not np.array_equal(ref[n], out[n])]
        if differing:
            worst = max(float(np.abs(ref[n].astype(np.float64)
                                   - out[n].astype(np.float64)).max())
                        for n in differing)
            bad.append("num_streams=%d: %d/%d frames differ (max |diff| %.3g)"
                       % (num_streams, len(differing), clip.num_frames, worst))
        del out

    assert not bad, (
        "%s on %s is not bit-identical to its serial run:\n  %s"
        % (name, clip_name, "\n  ".join(bad)))
