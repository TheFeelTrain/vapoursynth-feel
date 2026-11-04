"""Every registered argument appears somewhere in README.md.

The README is where users see a filter's argument list, so a registered
argument that is missing there is effectively invisible (the missing `hp` and
`th_mse` entries were exactly that). The compatibility pair is exempt because
the README documents it once for every filter.
"""

import inspect
import re
from pathlib import Path

import pytest
import vapoursynth as vs

README = Path(__file__).resolve().parent.parent / "README.md"
TEXT = README.read_text()
COMPAT = {"clip", "device_id", "num_streams"}
FILTERS = [
    "Bilateral",
    "BM3D",
    "BM3Dv2",
    "Box",
    "DFTTest",
    "EEDI3",
    "EEDI3H",
    "EEDI3AA",
    "GaussBlur",
    "NLMeans",
    "NNEDI3",
    "WNNM",
]


@pytest.mark.parametrize("name", FILTERS)
def test_every_registered_argument_is_documented(name):
    fn = getattr(vs.core.vsfeel, name)
    missing = [
        param
        for param in inspect.signature(fn).parameters
        if param not in COMPAT and not re.search(rf"\b{re.escape(param)}\b", TEXT)
    ]
    assert not missing, f"{name}: README does not document {missing}"
