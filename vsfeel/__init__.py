"""Use vsfeel's GPU filters through vs-jetpack wrappers.

vsfeel ships with a drop-in backend for the ``backend=`` argument of
vs-jetpack filters (vsrgtools, vsdenoise, ...). Just pass it in:

    import vsfeel

    from vsrgtools import bilateral, gauss_blur
    from vsdenoise import bm3d, DFTTest, nl_means

    blurred = bilateral(clip, ref, 3.0, 0.02, backend=vsfeel.Backend)
    smooth = gauss_blur(clip, 1.5, backend=vsfeel.Backend)
    denoised = nl_means(clip, h=1.2, tr=1, a=2, s=4, backend=vsfeel.Backend)
    clean = bm3d(clip, 0.7, tr=2, profile=bm3d.Profile.FAST, backend=vsfeel.Backend)
    dft = DFTTest(clip, backend=vsfeel.Backend).denoise({0: 16, 0.5: 8, 1.0: 0}, tr=1)

To also route the *internal* filters vs-jetpack calls on its own (e.g. the
bilateral postfilter inside ``vsaa.based_aa``), use the backend as a context
manager — no vs-jetpack changes needed:

    from vsaa import based_aa

    with vsfeel.Backend():
        based = based_aa(clip, backend=vsfeel.Backend)

``based_aa``'s EEDI3 antialiaser runs two EEDI3 calls plus two ``std.Merge``
nodes; ``vsfeel.EEDI3`` is a subclass of ``vsaa``'s antialiaser that replaces
that whole chain with the fused ``core.vsfeel.EEDI3AA`` call:

    aa = based_aa(clip, antialiaser=vsfeel.EEDI3(backend=vsfeel.Backend))

It is resolved lazily (PEP 562), so ``import vsfeel`` never requires vsaa.
"""

from .backend import Backend, FeelBackend

__all__ = ["Backend", "FeelBackend", "EEDI3"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "EEDI3":
        from .vsaa import EEDI3

        return EEDI3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
