#!/usr/bin/env python3
"""Single-file benchmark for the vsfeel plugin.

Runs the same clip through every installed BM3Dv2 (or Bilateral)
implementation side by side and reports the throughput. The timing is done
with vspipe so the numbers stay comparable with the previous per-plugin
scripts (benchmark/feel_test.py, cl_test.py, cu_test.py).

Usage:
    python3 benchmark/bench.py                           # all BM3Dv2 plugins
    python3 benchmark/bench.py vsfeel vszipcu            # a subset
    python3 benchmark/bench.py --bilateral               # benchmark Bilateral instead
    python3 benchmark/bench.py --bm3d-args "sigma=1.5, radius=4, num_streams=1"
    python3 benchmark/bench.py --frames 500 --clip /path/to/input.mkv
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CLIP = "/home/encode/test/jpbd.mkv"

# extra vpy lines needed to make a plugin available
LOADERS = {
    "vszipcu": 'core.std.LoadPlugin("/home/encode/test/vapoursynth-ziphip/zig-out/lib/libvszipcu.so")',
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


def bench(plugin: str, chain: str, clip: str, frames: int) -> float | None:
    vpy = VSPIPE_TEMPLATE.format(
        clip=clip,
        extra=LOADERS.get(plugin, ""),
        chain=chain,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"bench_{plugin}.vpy"
        path.write_text(vpy)
        return run_vspipe(path, frames)


def _parse_bm3d_args(raw: str) -> tuple[list[str], str]:
    """Split a "key=value, key=value" string into call args and a canonical
    description. num_streams is handled per-plugin (bm3dhip has none)."""
    pairs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            sys.exit(f"invalid BM3D arg {part!r} (expected key=value)")
        pairs.append((key, value))
    if not pairs:
        sys.exit("no BM3D args given")
    args_all = ", ".join(f"{k}={v}" for k, v in pairs)
    args_hip = ", ".join(f"{k}={v}" for k, v in pairs if k != "num_streams")
    desc = args_all
    return [args_all, args_hip], desc


def build_calls(args) -> tuple[dict[str, str], str]:
    """Return (plugin -> filter call, human-readable args line)."""
    if args.filter == "bm3dv2":
        pair, desc = _parse_bm3d_args(args.bm3d_args)
        args_all, args_hip = pair
        calls = {
            "vsfeel": f"core.vsfeel.BM3Dv2(get_y(clip), {args_all})",
            "vszipcl": f"core.vszipcl.BM3Dv2(get_y(clip), {args_all})",
            "vszipcu": f"core.vszipcu.BM3Dv2(get_y(clip), {args_all})",
            "bm3dhip": f"core.bm3dhip.BM3Dv2(get_y(clip), {args_hip})",  # no num_streams
        }
        desc = args_all
    else:
        common = f"sigma_spatial={args.sigma_spatial}, sigma_color={args.sigma_color}"
        calls = {
            "vsfeel": f"core.vsfeel.Bilateral(get_y(clip), {common})",
            "vszipcl": f"core.vszipcl.Bilateral(get_y(clip), {common})",
        }
        desc = f"sigma_spatial={args.sigma_spatial}, sigma_color={args.sigma_color}, num_streams=4"
    return calls, desc


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vsfeel against other BM3D implementations")
    parser.add_argument("plugins", nargs="*", help="plugins to run (default: all)")
    parser.add_argument("--filter", choices=("bm3dv2", "bilateral"), default="bm3dv2",
                        help="filter to benchmark (default: bm3dv2)")
    parser.add_argument("--bilateral", action="store_const", const="bilateral", dest="filter",
                        help="shorthand for --filter bilateral")
    parser.add_argument("--frames", type=int, default=None,
                        help="frames to time (default: 1000 for bm3dv2, 10000 for bilateral)")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="input clip path")
    # BM3Dv2 args as a single comma-separated "key=value" string
    parser.add_argument("--bm3d-args", dest="bm3d_args",
                        default="sigma=0.7, radius=2, bm_range=16, ps_range=7, block_step=4, num_streams=2",
                        help='BM3Dv2 args, e.g. "sigma=1.5, radius=4, num_streams=1"')
    # Bilateral args
    parser.add_argument("--sigma-spatial", type=float, default=3.0, dest="sigma_spatial")
    parser.add_argument("--sigma-color", type=float, default=0.02, dest="sigma_color")
    args = parser.parse_args()

    if args.frames is None:
        args.frames = 10000 if args.filter == "bilateral" else 1000

    calls, args_desc = build_calls(args)
    plugins = args.plugins or list(calls)
    plugins = [p for p in plugins if p in calls]
    if not plugins:
        sys.exit("no valid plugins requested")

    print(f"{args.filter.upper()} benchmark | {args.frames} frames | clip: {args.clip}")
    print(f"args: {args_desc}\n")

    results = []
    for plugin in plugins:
        fps = bench(plugin, calls[plugin], args.clip, args.frames)
        results.append((plugin, fps))
        if fps is None:
            print(f"  {plugin:10s}  unavailable / failed")
        else:
            print(f"  {plugin:10s}  {fps:9.2f} fps")
    print()

    valid = [(p, f) for p, f in results if f is not None]
    if len(valid) > 1:
        valid.sort(key=lambda x: x[1], reverse=True)
        for rank, (plugin, fps) in enumerate(valid, 1):
            print(f"  {rank}. {plugin:10s} {fps:9.2f} fps")


if __name__ == "__main__":
    main()
