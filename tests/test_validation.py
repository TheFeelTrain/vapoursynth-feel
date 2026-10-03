"""Smoke test: one frame per vsfeel filter under the Vulkan validation layers.

The validation layers are the cheapest net for the descriptor / binding /
barrier / device-extension class of defects. ``VK_INSTANCE_LAYERS`` is read by
the loader when the instance is created, so each filter runs in a subprocess
and the process output is scanned for ``Validation Error`` / ``VUID``.

The layer reports to stdout on this loader; both streams are scanned.  A
``VK_LOADER_DEBUG=layer`` marker confirms the layer really was inserted, so the
test cannot pass vacuously on a box without it (the loader silently ignores a
missing ``VK_INSTANCE_LAYERS`` entry).

Run from the repository root:  uv run python -m pytest tests/test_validation.py
"""

import os
import subprocess
import sys
import textwrap

import pytest

from conftest import CLIP_PATH

_LAYER = "VK_LAYER_KHRONOS_validation"

# One frame per filter, exercising the creation path and the frame path
# (temporal / multi-pass filters get three frames).
FILTERS = [
    "Bilateral",
    "Bilateral16",
    "GaussBlur",
    "GaussBlurLarge",
    "DFTTest",
    "NLMeans",
    "BM3Dv2",
    "BM3Dv2_16",
    "BM3Dv2Color",
    "BM3Dv2Joint",
    "EEDI3",
    "EEDI3H",
    "EEDI3AA",
    "NNEDI3",
]

_SCRIPT = textwrap.dedent(f"""\
    import sys
    import vapoursynth as vs

    name = sys.argv[1]
    core = vs.core
    if hasattr(core, "bs"):
        src = core.bs.VideoSource({CLIP_PATH!r})
    else:
        src = core.ffms2.Source({CLIP_PATH!r})
    g32 = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                             bits=32, fulls=True, fulld=True)
    g16 = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                             bits=16, fulls=True, fulld=True)

    if name == "Bilateral":
        node = core.vsfeel.Bilateral(g32, sigma_spatial=3.0, sigma_color=0.05)
    elif name == "Bilateral16":
        # The 16-bit SSBOs take the StorageBuffer-class codegen that needs
        # storageBuffer16BitAccess; the SPIR-V 1.0 Uniform+BufferBlock form
        # would need the feature the core does not enable.
        node = core.vsfeel.Bilateral(g16, sigma_spatial=3.0, sigma_color=0.05)
    elif name == "GaussBlur":
        node = core.vsfeel.GaussBlur(g32, sigma=2.0)
    elif name == "GaussBlurLarge":
        # sigma=20 derives radius 60 > LARGE_THRESHOLD, so this is the two-pass
        # path: separate horizontal/vertical dispatches, the push-descriptor
        # set, the barrier, and the 16-bit packed store through the aliased
        # binding 4 that only ENTRY_HORIZ && BITS == 16 declares.
        node = core.vsfeel.GaussBlur(g16, sigma=20.0)
    elif name == "DFTTest":
        node = core.vsfeel.DFTTest(g32, tbsize=3)
    elif name == "NLMeans":
        node = core.vsfeel.NLMeans(g32, d=2)
    elif name == "BM3Dv2":
        node = core.vsfeel.BM3Dv2(g32, sigma=0.7, radius=2, bm_range=16,
                                  ps_range=7, block_step=4)
    elif name == "BM3Dv2_16":
        # The integer input path: the copy-and-widen kernel rides the
        # estimation kernel's descriptor set and push-constant block, and the
        # aggregation writes native samples into the output plane.
        node = core.vsfeel.BM3Dv2(g16, sigma=0.7, radius=2, bm_range=16,
                                  ps_range=7, block_step=4)
    elif name == "BM3Dv2Color":
        # Three per-plane entries live at once: separate source rings, estimate
        # stacks and witness regions, one aggregation dispatch per plane.
        yuv = core.fmtc.bitdepth(src, bits=32, fulls=True, fulld=True)
        node = core.vsfeel.BM3Dv2(yuv, sigma=[0.7, 0.5, 0.5], radius=2,
                                  bm_range=2, ps_range=1, block_step=4)
    elif name == "BM3Dv2Joint":
        # The joint 4:4:4 entry: one packed stack holding all three planes.
        yuv444 = core.resize.Bicubic(src, format=vs.YUV444PS)
        node = core.vsfeel.BM3Dv2(yuv444, sigma=[0.7, 0.7, 0.7], radius=2,
                                  bm_range=2, ps_range=1, block_step=4, chroma=1)
    elif name == "EEDI3":
        node = core.vsfeel.EEDI3(g16, field=1)
    elif name == "EEDI3H":
        node = core.vsfeel.EEDI3H(g16, field=1)
    elif name == "EEDI3AA":
        node = core.vsfeel.EEDI3AA(g16, field=3)
    elif name == "NNEDI3":
        node = core.vsfeel.NNEDI3(g16, field=1)
    else:
        raise SystemExit("unknown filter %r" % name)

    for n in (0, 3, 7):
        node.get_frame(n)
    print("VALIDATION OK", flush=True)
""")


@pytest.mark.parametrize("filter_name", FILTERS)
def test_validation_layer_smoke(filter_name):
    env = {**os.environ, "VK_INSTANCE_LAYERS": _LAYER, "VK_LOADER_DEBUG": "layer", "MANGOHUD": "0"}
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, filter_name],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    stdout, stderr = proc.stdout, proc.stderr
    tail = "--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (stdout[-2000:], stderr[-2000:])

    if not any("Insert instance layer" in line and _LAYER in line for line in stderr.splitlines()):
        pytest.skip(f"{_LAYER} is not installed")

    assert proc.returncode == 0 and "VALIDATION OK" in stdout, (
        f"{filter_name} failed to evaluate under the validation layer\n{tail}"
    )

    hits = [
        line
        for line in stdout.splitlines() + stderr.splitlines()
        if "Validation Error" in line or "VUID" in line
    ]
    assert not hits, (
        f"{filter_name}: {len(hits)} validation-layer message(s):\n"
        + "\n".join(hits[:10])
        + "\n"
        + tail
    )


_LEAK_SCRIPT = textwrap.dedent("""\
    import sys

    import vapoursynth as vs

    core = vs.core
    clip = core.std.BlankClip(width=640, height=360, format=vs.GRAY16, length=2)
    name = sys.argv[1]
    try:
        if name == "EEDI3":
            core.vsfeel.EEDI3(clip, field=1, mdis=5, nrad=1)
        elif name == "NNEDI3":
            core.vsfeel.NNEDI3(clip, field=1)
        else:
            raise SystemExit("unknown filter %r" % name)
    except vs.Error:
        pass
    else:
        raise SystemExit("creation should have failed under the 128 limit")
    print("LEAK OK", flush=True)
""")


@pytest.mark.parametrize("filter_name", ["EEDI3", "NNEDI3"])
def test_failed_creation_does_not_leak_pipelines(filter_name):
    """A creation that fails partway must destroy the pipelines it already made.

    Under ``VSFEEL_LIMIT_INVOCATIONS=128`` a 256-invocation pipeline fails
    after the earlier 128-invocation ones were created, so an abandoned
    partial set is exactly the pipelines already built -- which the validation
    layer names at ``vkDestroyDevice``
    (``VUID-vkDestroyDevice-device-05137``). EEDI3 loses two (row, vcheck)
    before pad; NNEDI3 loses two (prescreen, predict) before the kept-row
    writer.
    """
    env = {
        **os.environ,
        "VK_INSTANCE_LAYERS": _LAYER,
        "VK_LOADER_DEBUG": "layer",
        "VSFEEL_LIMIT_INVOCATIONS": "128",
        "MANGOHUD": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _LEAK_SCRIPT, filter_name],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    stdout, stderr = proc.stdout, proc.stderr
    tail = "--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (stdout[-2000:], stderr[-2000:])

    if not any("Insert instance layer" in line and _LAYER in line for line in stderr.splitlines()):
        pytest.skip(f"{_LAYER} is not installed")

    assert proc.returncode == 0 and "LEAK OK" in stdout, (
        f"the failing {filter_name} creation did not reach its own error path\n{tail}"
    )

    hits = [
        line
        for line in stdout.splitlines() + stderr.splitlines()
        if "not been destroyed" in line or "05137" in line
    ]
    assert not hits, (
        f"a failed {filter_name} creation leaked Vulkan objects:\n"
        + "\n".join(hits[:10])
        + "\n"
        + tail
    )
