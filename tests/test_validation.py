"""Smoke test: one frame per vsfeel filter under the Vulkan validation layers.

The validation layers are the cheapest net for the descriptor / binding /
barrier / device-extension class of defects. ``VK_INSTANCE_LAYERS`` is read by
the loader when the instance is created, so each filter runs in a subprocess
and the process output is scanned for ``Validation Error`` / ``VUID``.

The layer reports to stdout on this loader; both streams are scanned.  A
``VK_LOADER_DEBUG=layer`` marker confirms the layer really was inserted, so the
test cannot pass vacuously on a box without it (the loader silently ignores a
missing ``VK_INSTANCE_LAYERS`` entry).

Run from the repository root:  python -m pytest tests/test_validation.py
"""

import os
import subprocess
import sys
import textwrap

import pytest

from conftest import NOISE_MKV

_LAYER = "VK_LAYER_KHRONOS_validation"

# One frame per filter, exercising the creation path and the frame path
# (temporal / multi-pass filters get three frames).
FILTERS = ["Bilateral", "GaussBlur", "DFTTest", "NLMeans", "BM3Dv2",
           "EEDI3", "EEDI3H", "EEDI3AA", "NNEDI3"]

_SCRIPT = textwrap.dedent(f"""\
    import sys
    import vapoursynth as vs

    name = sys.argv[1]
    core = vs.core
    if hasattr(core, "bs"):
        src = core.bs.VideoSource({NOISE_MKV!r})
    else:
        src = core.ffms2.Source({NOISE_MKV!r})
    g32 = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                             bits=32, fulls=True, fulld=True)
    g16 = core.fmtc.bitdepth(core.std.ShufflePlanes(src, 0, vs.GRAY),
                             bits=16, fulls=True, fulld=True)

    if name == "Bilateral":
        node = core.vsfeel.Bilateral(g32, sigma_spatial=3.0, sigma_color=0.05)
    elif name == "GaussBlur":
        node = core.vsfeel.GaussBlur(g32, sigma=2.0)
    elif name == "DFTTest":
        node = core.vsfeel.DFTTest(g32, tbsize=3)
    elif name == "NLMeans":
        node = core.vsfeel.NLMeans(g32, d=2)
    elif name == "BM3Dv2":
        node = core.vsfeel.BM3Dv2(g32, sigma=0.7, radius=2, bm_range=16,
                                  ps_range=7, block_step=4)
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
    env = {**os.environ, "VK_INSTANCE_LAYERS": _LAYER,
           "VK_LOADER_DEBUG": "layer", "MANGOHUD": "0"}
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, filter_name],
                          capture_output=True, text=True, timeout=300, env=env)
    stdout, stderr = proc.stdout, proc.stderr
    tail = ("--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (stdout[-2000:], stderr[-2000:]))

    if not any("Insert instance layer" in line and _LAYER in line
               for line in stderr.splitlines()):
        pytest.skip(f"{_LAYER} is not installed")

    assert proc.returncode == 0 and "VALIDATION OK" in stdout, (
        f"{filter_name} failed to evaluate under the validation layer\n{tail}")

    hits = [line for line in stdout.splitlines() + stderr.splitlines()
            if "Validation Error" in line or "VUID" in line]
    assert not hits, (
        f"{filter_name}: {len(hits)} validation-layer message(s):\n"
        + "\n".join(hits[:10]) + "\n" + tail)
