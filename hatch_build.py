import importlib.util
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

# The plugin is x86-64 on Linux and Windows. Every TU gets AVX2, the host code
# needs a POSIX or Win32 process/cache API, and the NNEDI3 weights are embedded
# with objcopy or an RC resource.
_SUPPORTED_PLATFORMS = {
    ("linux", "x86_64"), ("linux", "amd64"),
    ("win32", "x86_64"), ("win32", "amd64"),
}


def find_tool(name: str) -> str:
    """Locate a build tool, preferring PATH.

    cmake is a declared build requirement, but pip does not guarantee the build
    environment's scripts directory is on PATH, so fall back to the interpreter's
    own scripts directory (which is that environment when isolated).
    """
    found = shutil.which(name)
    if found is not None:
        return found
    suffix = ".exe" if sys.platform == "win32" else ""
    candidate = Path(sysconfig.get_path("scripts")) / f"{name}{suffix}"
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(f"{name} not found on PATH or in {candidate.parent}")


def vapoursynth_include_dir() -> str | None:
    """VapourSynth headers as seen by the interpreter running this hook.

    vapoursynth is a build requirement, so this environment has the headers even
    though whichever interpreter CMake's FindPython3 picks may not.
    """
    spec = importlib.util.find_spec("vapoursynth")
    if spec is None or spec.origin is None:
        return None
    include = Path(spec.origin).parent / "include"
    return str(include) if (include / "VapourSynth4.h").is_file() else None


# Deliberately not parameterized: hatchling's BuildHookInterface gained a second
# generic parameter in 1.32, so any explicit subscript breaks one version or the
# other.
class CustomHook(BuildHookInterface):
    """Compile the plugin with CMake and stage it into the wheel's plugin dir."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if (sys.platform, platform.machine().lower()) not in _SUPPORTED_PLATFORMS:
            raise RuntimeError(
                "vapoursynth-feel builds only on Linux and Windows x86-64 "
                f"(got {sys.platform}/{platform.machine()})."
            )

        # Root-relative: the wheel's include list is root-relative, so staging
        # must not depend on the process CWD.
        root = Path(self.root).resolve()
        self.target_dir = root / "vapoursynth/plugins/vsfeel"
        self.build_dir = root / "build/pack"   # scratch configure/build tree (gitignored)
        self.install_dir = root / "install"    # cmake --install staging prefix

        build_data["pure_python"] = False
        # A `py3-none-<platform>` wheel: the payload is a VapourSynth plugin, not
        # a CPython extension, so no python/abi tag is involved. The Linux tag is
        # rewritten to manylinux by auditwheel in CI.
        build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"

        cmake = find_tool("cmake")
        cmake_args = [
            cmake, "-S", str(root), "-B", str(self.build_dir),
            "-D", "CMAKE_BUILD_TYPE=Release",
            # hatch-vcs is the single version source; embed the version it
            # resolved for this build instead of letting CMake guess one
            "-D", f"VSFEEL_VERSION={version}",
            # stage into install/<target_dir> instead of the live VapourSynth
            # plugin directory (which is an absolute path when vapoursynth is
            # importable, and cmake --install ignores --prefix for those).
            # Relative on purpose: an absolute DESTINATION would bypass the
            # --prefix under which finalize cleans up.
            "-D", f"VSFEEL_INSTALL_DIR={self.target_dir.relative_to(root)}",
            # The build environment's interpreter, so CMake does not have to
            # find one on PATH (it generates the SPIR-V header at build time).
            "-D", f"Python3_EXECUTABLE={sys.executable}",
        ]
        include_dir = vapoursynth_include_dir()
        if include_dir is not None:
            cmake_args += ["-D", f"VS_INCLUDE_DIR={include_dir}"]
        subprocess.run(cmake_args, check=True, cwd=root)
        subprocess.run(
            [cmake, "--build", str(self.build_dir), "--config", "Release",
             "--parallel"],
            check=True, cwd=root,
        )
        subprocess.run(
            [cmake, "--install", str(self.build_dir),
             "--config", "Release",   # multi-config generators (Visual Studio)
             "--prefix", str(self.install_dir)],
            check=True, cwd=root,
        )

        # Copy exactly this platform's plugin library into the wheel. A missing
        # file fails the build loudly instead of silently shipping an empty,
        # unloadable wheel.
        libs = sorted(p for p in self.install_dir.rglob("*")
                      if p.is_file() and p.suffix in (".so", ".dll"))
        if not libs:
            raise RuntimeError(
                f"CMake install produced no vsfeel library under {self.install_dir}/ "
                "— the plugin failed to compile/link."
            )
        manifests = sorted(self.install_dir.rglob("manifest.vs"))
        if not manifests:
            raise RuntimeError(
                f"CMake install produced no manifest.vs under {self.install_dir}/."
            )

        self.target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(libs[0], self.target_dir)
        shutil.copy2(manifests[0], self.target_dir)

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        # The wheel is already assembled here; drop the staged tree (vapoursynth/…)
        # so the source checkout stays clean. parents[1] is "vapoursynth/"
        # (parents[0] is ".../plugins").
        shutil.rmtree(self.target_dir.parents[1], ignore_errors=True)
