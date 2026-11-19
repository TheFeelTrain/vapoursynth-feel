"""The persistent pipeline cache writes once per process, one file per family.

The child creates two EEDI3 variants and one GaussBlur and asserts the cache
directory does not change while it runs. After exit there is one file per
family; a further warm run must then leave those files untouched.
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


    before = snapshot() if os.environ.get("VSFEEL_TEST_SNAPSHOT", "1") != "0" else {}
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

    if os.environ.get("VSFEEL_TEST_SNAPSHOT", "1") != "0":
        after = snapshot()
        if after != before:
            print("CACHE CHANGED during the process:", before, after, flush=True)
            raise SystemExit(3)
    print("RESULT", flush=True)
    """
)


def _run(cache_dir, *, snapshot=True):
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "MANGOHUD": "0",
            "VSFEEL_PIPELINE_CACHE": str(cache_dir),
            "VSFEEL_TEST_SNAPSHOT": "1" if snapshot else "0",
        },
    )


def _snapshot(cache_dir):
    return {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in cache_dir.iterdir()
        if path.suffix == ".bin"
    }


def _run_ok(cache_dir, label, *, snapshot=True):
    proc = _run(cache_dir, snapshot=snapshot)
    assert proc.returncode == 0, (
        f"{label} process exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_pipeline_cache_is_per_family_and_batched(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    _run_ok(cache, "cold")
    files = sorted(path.name for path in cache.iterdir() if path.suffix == ".bin")
    assert len(files) == 2, files
    assert any(name.endswith("_eedi3.bin") for name in files), files
    assert any(name.endswith("_gaussblur.bin") for name in files), files

    _run_ok(cache, "warm")
    warm = _snapshot(cache)
    # A driver may re-encode the blob the first time it is seeded, so compare
    # two warm runs: whatever the first one wrote must now be stable.
    _run_ok(cache, "second warm")
    assert _snapshot(cache) == warm, "a warm run rewrote the pipeline cache"

    # A stale hash for a known family is dead weight; another build's family is
    # not ours to delete.
    uuid = files[0].split("_")[2]
    stale = cache / f"pipeline_cache_{uuid}_deadbeefdeadbeef_gaussblur.bin"
    foreign = cache / f"pipeline_cache_{uuid}_deadbeefdeadbeef_notafilter.bin"
    stale.write_bytes(b"stale")
    foreign.write_bytes(b"foreign")
    _run_ok(cache, "prune", snapshot=False)
    remaining = {path.name for path in cache.iterdir()}
    assert stale.name not in remaining
    assert foreign.name in remaining
