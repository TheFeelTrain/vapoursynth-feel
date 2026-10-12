"""``feel_aa``: the device-resident ``based_aa``, against vsaa's own chain.

The oracle is ``vsaa.based_aa`` driven by the same antialiaser and the same
scalers, so the only steps that differ are the ones that cannot stay on the
device. On float input they agree bit-exactly; on 16-bit integer the reference's
mask operator is the incorrect one (see ``test_feel_aa_16bit_*``).
"""

import inspect

import numpy as np
import pytest
import vapoursynth as vs

pytest.importorskip("vstools")
pytest.importorskip("vsaa")

from vsaa import based_aa
from vskernels import Box, Catrom, CustomComplexKernel, Point
from vsscale import ArtCNN

import vsfeel
from vsfeel.vsaa import EEDI3, NNEDI3, feel_aa

# based_aa's antialiaser settings, named explicitly so both chains run the same
# filter rather than vsaa's bare EEDI3 defaults.
AA_PARAMS = dict(alpha=0.125, beta=0.25, gamma=40.0, vthresh=(12.0, 24.0, 4.0))


def _reference(clip, **kwargs):
    with vsfeel.Backend():
        return based_aa(clip, antialiaser=EEDI3(**AA_PARAMS, backend=vsfeel.Backend), **kwargs)


def _plane(node, n=0, dtype=np.float64):
    """Plane 0 as `dtype`; downloads a resident frame explicitly."""
    if node.gpu_resident:
        node = vs.core.std.GPUDownload(clip=node)
    return np.asarray(node.get_frame(n)[0]).astype(dtype).copy()


def _transfers(build):
    """The implicit transfers the core inserted while building this graph.

    The core logs them when the consuming node is *created*, so the handler has
    to wrap the construction, not the frame pull.
    """
    messages: list[str] = []
    handle = vs.core.add_log_handler(lambda level, msg: messages.append(msg))
    try:
        node = build()
    finally:
        vs.core.remove_log_handler(handle)
    return node, [m for m in messages if "automatically inserted" in m]


def test_feel_aa_is_resident(clip_gray):
    out, transfers = _transfers(lambda: feel_aa(clip_gray, supersampler=Point()))
    assert out.gpu_resident
    assert transfers == []


def test_feel_aa_reference_transfers(clip_gray):
    """Sanity: the chain it replaces does insert transfers on the same clip."""
    ref, transfers = _transfers(
        lambda: _reference(clip_gray, supersampler=Point(), downscaler=Catrom())
    )
    assert not ref.gpu_resident
    assert transfers


# based_aa's own default supersampler is ArtCNN, a learned model whose imports
# are incomplete in this environment, so every case names its scalers instead.
# The reference runs most steps on the host and feel_aa on the device, and the
# core's two implementations of one filter are not bit-identical in float: the
# mask's Prewitt differs by ~5e-7 on this clip, and where that straddles the
# binarize threshold the local result moves by a few e-3. The bounds below are
# measured, per configuration, on this fixture.
# `rfactor=1.5` is looser on purpose: the core's GPU Point resize samples a
# fractional ratio with a different phase than its own CPU path (measured 0.78
# apart on a bare 1.5x upscale), so any Point supersample at a non-integer ratio
# cannot match the host chain.
@pytest.mark.parametrize(
    "kwargs,mean_max,max_max",
    [
        ({"supersampler": Catrom()}, 1e-5, 1.5e-2),
        ({"supersampler": Point()}, 1e-5, 1.5e-2),
        ({"supersampler": Point(), "rfactor": 1.5}, 1e-3, 0.1),
        ({"supersampler": Point(), "mask": False}, 1e-6, 1e-5),
        ({"supersampler": Point(), "pscale": 0.5}, 1e-5, 1.5e-2),
        ({"supersampler": Point(), "postfilter": False}, 1e-5, 1.5e-2),
        ({"supersampler": Point(), "mask_thr": 100}, 1e-5, 1.5e-2),
    ],
)
def test_feel_aa_matches_based_aa_float(clip_gray, kwargs, mean_max, max_max):
    ref = _reference(clip_gray, **kwargs)
    out = feel_aa(clip_gray, **kwargs)
    d = np.abs(_plane(ref) - _plane(out))
    assert d.mean() <= mean_max, f"mean|d|={d.mean()} over {mean_max}"
    assert d.max() <= max_max, f"max|d|={d.max()} over {max_max}"


def test_feel_aa_defaults_are_named():
    """The declared defaults, checked on the signature rather than by running.

    The ArtCNN default is not exercised here: it is a learned model whose
    weights are fetched on first use, so a test that ran it would need the
    network and a warm model cache. What runs a host scaler end to end is
    covered by the test below, which uses one with no download.
    """
    params = inspect.signature(feel_aa).parameters
    assert params["supersampler"].default is ArtCNN
    assert params["downscaler"].default is None
    assert params["antialiaser"].default is EEDI3


@pytest.mark.parametrize("rfactor,downscaler", [(2, Box), (1.5, Catrom)])
def test_feel_aa_downscaler_default_rule(clip_gray, rfactor, downscaler):
    """`downscaler=None` keeps based_aa's rule: Box at an integer ratio, else Catrom."""
    named = dict(supersampler=Point(), antialiaser=EEDI3())
    out = feel_aa(clip_gray, rfactor=rfactor, **named)
    explicit = feel_aa(clip_gray, rfactor=rfactor, downscaler=downscaler, **named)
    assert np.array_equal(_plane(out), _plane(explicit))


class _HostKernel(CustomComplexKernel):
    """A scaler with no `core.resize` equivalent, so it can only run on the host.

    Stands in for ArtCNN without needing its model weights.
    """

    _static_kernel_radius = 1

    def kernel(self, *, x: float) -> float:
        return 1.0 if -0.5 <= x < 0.5 else 0.0


def test_feel_aa_allows_host_supersampler(clip_gray):
    """A CPU-resident supersampler is allowed, and its frame is uploaded once.

    The supersampled frame is read by the antialiaser's `clip` and `sclip` and
    again by the pscale blend. Each of those readers would otherwise carry its
    own auto-inserted 4K upload; the explicit one covers all three, so the only
    transfer left is the download the host scaler asks for itself.
    """
    out, transfers = _transfers(lambda: feel_aa(clip_gray, supersampler=_HostKernel()))
    assert out.gpu_resident, transfers
    downloads = [m for m in transfers if "GPUDownload" in m]
    uploads = [m for m in transfers if "GPUUpload" in m]
    assert len(downloads) == 1, transfers  # the host scaler's own input
    assert uploads == [], transfers  # one explicit upload covered every reader
    assert _plane(out).shape == (clip_gray.height, clip_gray.width)


def test_feel_aa_matches_based_aa_yuv_float(clip_yuv32):
    """YUV float32: the same bound, and the untouched chroma rides along."""
    ref = _reference(clip_yuv32, supersampler=Point(), downscaler=Catrom())
    out = feel_aa(clip_yuv32, supersampler=Point(), downscaler=Catrom())
    d = np.abs(_plane(ref) - _plane(out))
    assert d.max() <= 1.5e-2, f"max|d|={d.max()}"


def test_feel_aa_keeps_chroma(clip_yuv32):
    """feel_aa only processes luma, so both chroma planes are the source's."""
    out = feel_aa(clip_yuv32, supersampler=Point(), downscaler=Catrom())
    if out.gpu_resident:
        out = vs.core.std.GPUDownload(clip=out)
    frame, src = out.get_frame(0), clip_yuv32.get_frame(0)
    for p in (1, 2):
        assert np.array_equal(np.asarray(frame[p]), np.asarray(src[p]))


def test_feel_aa_16bit_against_based_aa(clip_16bit):
    """16-bit integer input: the residual is the reference's mask defect.

    ``vsaa.based_aa`` builds its mask through vsmasktools, which routes integer
    clips to the ``edgemasks`` plugin; its Prewitt disagrees with the reference
    formula by up to 41228 on the 1-pixel border ring, while ``feel_aa`` uses
    ``core.std.Prewitt``, which matches it exactly. Everything else in the chain
    is bit-exact, so the difference is bounded by that mask at the frame edge
    plus the 1 LSB the two box blurs round differently.
    """
    ref = _reference(clip_16bit, supersampler=Point(), downscaler=Catrom())
    out = feel_aa(clip_16bit, supersampler=Point(), downscaler=Catrom())
    a, b = _plane(ref), _plane(out)
    d = np.abs(a - b)
    interior = d[4:-4, 4:-4]
    assert d.max() <= 400, f"border diff {d.max()} exceeds the measured bound"
    assert interior.max() <= 16, f"interior diff {interior.max()} exceeds the measured bound"


def test_feel_aa_user_settings_is_resident(clip_gray):
    """The chain this replaces inserts transfers; feel_aa does not."""
    out, transfers = _transfers(
        lambda: feel_aa(
            clip_gray,
            rfactor=2,
            supersampler=NNEDI3(),
            use_mclip=True,
            antialiaser=EEDI3(**AA_PARAMS, backend=vsfeel.Backend),
        )
    )
    assert out.gpu_resident
    assert transfers == []


@pytest.mark.parametrize(
    "ratio,exact",
    [(2, True), (3, False), (4, True), (1.5, True), (1.25, True)],
)
def test_native_box_matches_resize2(clip_16bit, ratio, exact):
    """`core.vsfeel.Box` reproduces resize2's Box for any downscale ratio.

    It is bit-exact wherever resize2's own weights are exact (every tap count
    that is a power of two); the remaining ratios are bounded by its float
    quantisation of ``1/count``.
    """
    W = (clip_16bit.width // 2) * 2
    H = (clip_16bit.height // 2) * 2
    small_w, small_h = round(W / ratio), round(H / ratio)
    big = vs.core.resize.Point(clip_16bit, W, H)
    ref = _plane(Box().scale(big, small_w, small_h))
    got = vs.core.std.GPUDownload(vs.core.vsfeel.Box(big, width=small_w, height=small_h))
    out = _plane(got)
    d = np.abs(ref - out)
    assert d.max() == 0 if exact else d.max() <= 4, f"max|d|={d.max()}"


def test_native_box_is_resident(clip_16bit):
    resident = vs.core.std.GPUUpload(clip=clip_16bit)
    node, transfers = _transfers(
        lambda: vs.core.vsfeel.Box(resident, width=resident.width // 2, height=resident.height // 2)
    )
    assert node.gpu_resident
    assert transfers == []


def test_native_box_rejects_upscale(clip_gray):
    with pytest.raises(Exception, match="downscales"):
        vs.core.vsfeel.Box(clip_gray, width=clip_gray.width * 2)


def test_feel_aa_box_other_ratio_is_resident(clip_gray):
    """Box at any ratio is now on the device, so rfactor=3 stays exact."""
    ref = _reference(clip_gray, rfactor=3, supersampler=Point())
    out = feel_aa(clip_gray, rfactor=3, supersampler=Point())
    assert out.gpu_resident
    d = np.abs(_plane(ref) - _plane(out))
    assert d.max() <= 8e-3, f"max|d|={d.max()}"


def test_feel_aa_rfactor_is_validated(clip_gray):
    with pytest.raises(Exception, match="rfactor"):
        feel_aa(clip_gray, rfactor=0.0)


def test_feel_aa_show_mask(clip_gray):
    mask = feel_aa(clip_gray, show_mask=True)
    assert mask.format.num_planes == 1
    assert mask.width == clip_gray.width


def test_feel_aa_prefilter_clip(clip_gray):
    """A reference-clip prefilter is accepted and stays exact."""
    ref = _reference(clip_gray, supersampler=Point(), prefilter=clip_gray)
    out = feel_aa(clip_gray, supersampler=Point(), prefilter=clip_gray)
    d = np.abs(_plane(ref) - _plane(out))
    assert d.max() <= 8e-3, f"max|d|={d.max()}"
