#!/usr/bin/env python3
"""Load an installed vapoursynth-feel wheel without a GPU.

Run from outside the source tree after installing the wheel: VapourSynth must
load the plugin library from the wheel's own plugin directory, register every
filter, and report the version baked in at build time. A wheel with the wrong
platform library name, a missing manifest or a stale plugin fails here instead
of on a user's machine.
"""

import sys

import vapoursynth as vs

import vsfeel

EXPECTED = {
    "Bilateral",
    "GaussBlur",
    "DFTTest",
    "NLMeans",
    "BM3D",
    "EEDI3",
    "EEDI3H",
    "EEDI3AA",
    "NNEDI3",
    "WNNM",
    "Box",
}
LIBRARY = {"win32": "vsfeel.dll", "darwin": "libvsfeel.dylib"}.get(sys.platform, "libvsfeel.so")


def main() -> None:
    if not callable(getattr(vsfeel, "Backend", None)):
        raise SystemExit("vsfeel.Backend is missing from the installed package")

    core = vs.core
    names = {function.name for function in core.vsfeel.functions()}
    missing = EXPECTED - names
    if missing:
        raise SystemExit(f"plugin does not register: {sorted(missing)}")

    loaded = [p.plugin_path for p in core.plugins() if p.namespace == "vsfeel"]
    if not loaded:
        raise SystemExit("VapourSynth did not load the vsfeel plugin")
    path = loaded[0]
    if not path.endswith(LIBRARY):
        raise SystemExit(f"loaded {path}, expected {LIBRARY}")
    if "site-packages" not in path and "dist-packages" not in path:
        raise SystemExit(f"plugin loaded from outside the installed wheel: {path}")

    version = core.vsfeel.Version()
    if not isinstance(version, str) or not version:
        raise SystemExit(f"plugin reports no version: {version!r}")

    print(f"wheel smoke ok: {len(names)} functions, {path}, version {version}")


if __name__ == "__main__":
    main()
