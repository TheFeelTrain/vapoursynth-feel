#!/usr/bin/env python3
"""Single-file benchmark for the vsfeel plugin.

Runs the same clip through every installed BM3Dv2 (or Bilateral)
implementation side by side and reports the throughput. The timing is done
with vspipe so the numbers stay comparable with the previous per-plugin
scripts (benchmark/feel_test.py, cl_test.py, cu_test.py).

Usage:
    python3 benchmark/bench.py                    # all BM3Dv2 plugins
    python3 benchmark/bench.py vsfeel vszipcu     # a subset
    python3 benchmark/bench.py --bilateral        # benchmark Bilateral instead
    python3 benchmark/bench.py --frames 500       # timed frame count
    python3 benchmark/bench.py --clip /path/to/input.mkv
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CLIP = "/home/encode/test/jpbd.mkv"

BM3D_ARGS = "sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4"
BILATERAL_ARGS = "sigma_spatial=3.0, sigma_color=0.02, num_streams=4"

# extra vpy lines needed to make a plugin available
LOADERS = {
    "vszipcu": 'core.std.LoadPlugin("/home/encode/test/vapoursynth-ziphip/zig-out/lib/libvszipcu.so")',
}

# per-plugin filter calls (bm3dhip has no num_streams)
FILTERS = {
    "bm3dv2": {
        "vsfeel": f"core.vsfeel.BM3Dv2(get_y(clip), {BM3D_ARGS}, num_streams=2)",
        "vszipcl": f"core.vszipcl.BM3Dv2(get_y(clip), {BM3D_ARGS}, num_streams=2)",
        "vszipcu": f"core.vszipcu.BM3Dv2(get_y(clip), {BM3D_ARGS}, num_streams=2)",
        "bm3dhip": f"core.bm3dhip.BM3Dv2(get_y(clip), {BM3D_ARGS})",
    },
    "bilateral": {
        "vsfeel": f"core.vsfeel.Bilateral(get_y(clip), {BILATERAL_ARGS})",
        "vszipcl": f"core.vszipcl.Bilateral(get_y(clip), {BILATERAL_ARGS})",
    },
}

VSPIPE_TEMPLATE = """\
from vssource import BestSource
from vstools import core, get_y

core.max_cache_size = 1024 * 56

clip = BestSource(cachepath=None).source({clip!r}, 32)

{extra}

{chain}.set_output()
"""


def run_vspipe(vpy_path: Path, frames: int) -> float | None:
    cmd = ["vspipe", "--start", "0", "--end", str(frames - 1), str(vpy_path), "/dev/null"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    for line in (result.stderr or "").splitlines():
        if "Output" in line and "fps" in line:
            # vspipe: "Output 1000 frames in 12.83 seconds (77.94 fps)"
            return float(line.rsplit("(", 1)[1].split("fps")[0].strip())
    return None


def bench(filter_name: str, plugin: str, clip: str, frames: int) -> float | None:
    chain = FILTERS[filter_name][plugin]
    vpy = VSPIPE_TEMPLATE.format(
        clip=clip,
        extra=LOADERS.get(plugin, ""),
        chain=chain,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"bench_{plugin}.vpy"
        path.write_text(vpy)
        return run_vspipe(path, frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vsfeel against other BM3D implementations")
    parser.add_argument("plugins", nargs="*", help="plugins to run (default: all)")
    parser.add_argument("--filter", choices=sorted(FILTERS), default="bm3dv2",
                        help="filter to benchmark (default: bm3dv2)")
    parser.add_argument("--bilateral", action="store_const", const="bilateral", dest="filter",
                        help="shorthand for --filter bilateral")
    parser.add_argument("--frames", type=int, default=None,
                        help="frames to time (default: 1000 for bm3dv2, 10000 for bilateral)")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="input clip path")
    args = parser.parse_args()
    if args.frames is None:
        args.frames = 10000 if args.filter == "bilateral" else 1000

    calls = FILTERS[args.filter]
    plugins = args.plugins or list(calls)
    plugins = [p for p in plugins if p in calls]
    if not plugins:
        sys.exit("no valid plugins requested")

    print(f"benchmarking core.{args.filter.upper()} on {args.frames} frames of:")
    print(f"  {args.clip}\n")

    results = []
    for plugin in plugins:
        fps = bench(args.filter, plugin, args.clip, args.frames)
        results.append((plugin, fps))
        if fps is None:
            print(f"  {plugin:10s}  unavailable / failed")
        else:
            print(f"  {plugin:10s}  {fps:9.2f} fps")

    valid = [(p, f) for p, f in results if f is not None]
    if len(valid) > 1:
        valid.sort(key=lambda x: x[1], reverse=True)
        print()
        for rank, (plugin, fps) in enumerate(valid, 1):
            note = "  <- fastest" if rank == 1 else ""
            print(f"  {rank}. {plugin:10s} {fps:9.2f} fps{note}")


if __name__ == "__main__":
    main()
