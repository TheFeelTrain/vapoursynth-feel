"""Python companion package for the vsfeel VapourSynth plugin.

Exposes duck-typed backend selectors that plug straight into the ``backend=``
arguments of vs-jetpack functions (vsrgtools, vsdenoise, ...), selecting
``core.vsfeel`` without modifying those packages:

    from vsrgtools import bilateral, gauss_blur
    from vsdenoise import bm3d, nl_means
    from vsdenoise.fft import DFTTest
    import vsfeel

    blurred = bilateral(clip, ref, 3.0, 0.02, backend=vsfeel.Backend)
    smooth = gauss_blur(clip, 1.5, backend=vsfeel.Backend)
    denoised = nl_means(clip, h=1.2, tr=1, a=2, s=4, backend=vsfeel.Backend)
    clean = bm3d(clip, 0.7, tr=2, profile=bm3d.Profile.FAST, backend=vsfeel.Backend)
    dft = DFTTest(clip, backend=vsfeel.Backend).denoise({0: 16, 0.5: 8, 1.0: 0}, tr=1)

This works because vs-jetpack never type-checks its backends: it only calls
``resolve()`` on them and then either invokes the method named after the
filter (``backend.Bilateral(...)``, ``backend.NLMeans(...)``, ...) or resolves
the plugin itself through ``getattr(core, backend.value)`` /
``backend.plugin``.
"""

from .backend import Backend, FeelBackend

__all__ = ["Backend", "FeelBackend"]
