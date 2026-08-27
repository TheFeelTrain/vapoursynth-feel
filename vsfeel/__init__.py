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

"""

from .backend import Backend, FeelBackend

__all__ = ["Backend", "FeelBackend"]
