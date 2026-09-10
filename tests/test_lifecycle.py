"""Filter lifecycle under concurrency (R9).

Each vsfeel filter used to call ``vkDeviceWaitIdle`` in its destructor without
holding the shared queues' locks.  Destroying one node while another node was
submitting work therefore both violated Vulkan's host synchronization rules and
stalled every other stream on the device.  Teardown now drains only the queues
this instance submitted on, under their locks.

The test runs in a subprocess so a regression shows up as a failed assertion
with captured output instead of hanging the suite: it evaluates one graph from
several threads while repeatedly creating and destroying other filter nodes on
the same device.
"""

import subprocess
import sys
import textwrap

import pytest

from conftest import NOISE_MKV

_LIFECYCLE_SCRIPT = textwrap.dedent(f"""\
    import sys
    import threading
    import vapoursynth as vs

    core = vs.core
    core.max_cache_size = 512

    src = core.bs.VideoSource({NOISE_MKV!r})
    gray = core.std.ShufflePlanes(src, 0, vs.GRAY)
    f32 = core.fmtc.bitdepth(gray, bits=32, fulls=True, fulld=True)
    # a longer clip so the worker threads keep submitting frames throughout
    clip = core.std.Loop(f32, 3)

    steady = core.vsfeel.GaussBlur(clip, sigma=[2.0], num_streams=4)
    n_frames = steady.num_frames
    stop = threading.Event()
    errors = []
    produced = []

    def worker(offset):
        try:
            n = offset
            while n < n_frames and not stop.is_set():
                f = steady.get_frame(n)
                if not (f.width and f.height):
                    raise AssertionError("empty frame %d" % n)
                produced.append(n)
                n += 2
        except Exception as exc:                  # noqa: BLE001 - reported below
            errors.append("%s: %s" % (type(exc).__name__, exc))
            stop.set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()

    # create, evaluate and destroy other nodes on the same shared device while
    # the workers above keep the GPU busy
    try:
        for _ in range(3):
            tmp = core.vsfeel.DFTTest(clip, sigma=1.0, num_streams=2)
            tmp.get_frame(1)
            del tmp
            tmp = core.vsfeel.Bilateral(f32, sigma_spatial=[3.0],
                                        sigma_color=[0.05], num_streams=2)
            tmp.get_frame(1)
            del tmp
            tmp = core.vsfeel.BM3Dv2(clip, sigma=[0.7], radius=1, num_streams=2)
            tmp.get_frame(1)
            del tmp
            core.clear_cache()
    finally:
        stop.set()
        for t in threads:
            t.join()

    if errors:
        print("LIFECYCLE fail: " + "; ".join(errors), flush=True)
        raise SystemExit(3)
    if not produced:
        print("LIFECYCLE fail: no frames were produced", flush=True)
        raise SystemExit(3)
    print("RESULT " + str(len(produced)), flush=True)
""")


def test_concurrent_create_destroy_with_active_graph(noise_gray):
    """Creating/destroying nodes must not disturb a concurrently running graph.

    ``noise_gray`` is requested only so this test depends on the usual source
    setup; the subprocess builds its own graph.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LIFECYCLE_SCRIPT],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "lifecycle subprocess hung (deadlock during concurrent teardown?)\n"
            "--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % ((exc.stdout or "")[-2000:], (exc.stderr or "")[-2000:])
        )
    out = proc.stdout or ""
    assert "LIFECYCLE fail" not in out, (
        "concurrent lifecycle failure:\n%s\n--- stderr ---\n%s"
        % (out[-2000:], (proc.stderr or "")[-2000:]))
    assert proc.returncode == 0 and "RESULT " in out, (
        "lifecycle subprocess exited %s\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (proc.returncode, out[-2000:], (proc.stderr or "")[-2000:]))
