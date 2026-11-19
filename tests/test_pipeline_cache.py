"""The persistent pipeline cache: one file per filter family, one save per family.

A filter instance holds the only strong reference to its GPUDevice, so writing
the cache from the device destructor rewrote the whole blob every time a node
died. The test creates two EEDI3 variants and one GaussBlur in one process and
asserts the cache directory does not change until the process exits; a second,
warm run must leave it untouched. The family files are checked after exit,
where the flush runs.
"""

import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """\
    import gc
    import os

    import vapoursynth as vs

    core = vs.core
    cache = os.environ["VSFEEL_PIPELINE_CACHE"]


    def snapshot():
        out = {}
        for name in os.listdir(cache):
            if not name.endswith(".bin"):
                continue
            st = os.stat(os.path.join(cache, name))
            out[name] = (st.st_ino, st.st_mtime_ns, st.st_size)
        return out


    before = snapshot()
    clip = core.std.BlankClip(width=320, height=180, format=vs.GRAY16, length=2)
    for mdis in (5, 7):
        node = core.vsfeel.EEDI3(clip, field=1, mdis=mdis, nrad=1)
        node.get_frame(0)
        del node
        gc.collect()
        core.clear_cache()
    node = core.vsfeel.GaussBlur(clip, sigma=[2.0])
    node.get_frame(0)
    del node
    gc.collect()
    core.clear_cache()

    after = snapshot()
    if after != before:
        print("CACHE CHANGED during the process:", before, after, flush=True)
        raise SystemExit(3)
    print("RESULT", flush=True)
    """
)


def _run(cache_dir):
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "MANGOHUD": "0",
            "VSFEEL_PIPELINE_CACHE": str(cache_dir),
        },
    )


def test_pipeline_cache_is_per_family_and_batched(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    first = _run(cache)
    assert first.returncode == 0, (
        f"cold process exited {first.returncode}\n"
        f"--- stdout ---\n{first.stdout[-2000:]}\n"
        f"--- stderr ---\n{first.stderr[-2000:]}"
    )
    files = sorted(path.name for path in cache.iterdir() if path.suffix == ".bin")
    assert len(files) == 2, files
    assert any(name.endswith("_eedi3.bin") for name in files), files
    assert any(name.endswith("_gaussblur.bin") for name in files), files

    before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in cache.iterdir()
        if path.suffix == ".bin"
    }
    second = _run(cache)
    assert second.returncode == 0, (
        f"warm process exited {second.returncode}\n"
        f"--- stdout ---\n{second.stdout[-2000:]}\n"
        f"--- stderr ---\n{second.stderr[-2000:]}"
    )
    after = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in cache.iterdir()
        if path.suffix == ".bin"
    }
    assert after == before, "a warm run rewrote the pipeline cache"
