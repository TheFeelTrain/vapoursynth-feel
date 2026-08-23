import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags


class CustomHook(BuildHookInterface[Any]):
    """Compile the plugin with CMake and stage it into the wheel's plugin dir."""

    target_dir = Path("vapoursynth/plugins/vsfeel")
    build_dir = Path("build/pack")   # scratch configure/build tree (gitignored)
    install_dir = Path("install")    # cmake --install staging prefix

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"

        root = Path(self.root)
        subprocess.run(
            ["cmake", "-S", str(root), "-B", str(root / self.build_dir),
             "-D", "CMAKE_BUILD_TYPE=Release",
             # stage into install/<target_dir> instead of the live VapourSynth
             # plugin directory (which is an absolute path when vapoursynth is
             # importable, and cmake --install ignores --prefix for those)
             "-D", f"VSFEEL_INSTALL_DIR={self.target_dir}"],
            check=True, cwd=root,
        )
        subprocess.run(
            ["cmake", "--build", str(root / self.build_dir), "--config", "Release",
             "--parallel"],
            check=True, cwd=root,
        )
        subprocess.run(
            ["cmake", "--install", str(root / self.build_dir),
             "--prefix", str(root / self.install_dir)],
            check=True, cwd=root,
        )

        # Copy exactly this platform's plugin library into the wheel. A missing
        # file fails the build loudly instead of silently shipping an empty,
        # unloadable wheel.
        libs = sorted(p for p in (root / self.install_dir).rglob("*")
                      if p.is_file() and p.suffix in (".so", ".dll", ".dylib"))
        if not libs:
            raise RuntimeError(
                f"CMake install produced no vsfeel library under {self.install_dir}/ "
                "— the plugin failed to compile/link."
            )
        manifests = sorted((root / self.install_dir).rglob("manifest.vs"))
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
