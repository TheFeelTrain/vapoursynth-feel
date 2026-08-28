#!/usr/bin/env python3
"""Benchmark vsfeel filters against the reference implementations.

The benchmark is data-driven: every filter is described by one entry in the
FILTERS registry below (its CLI args, the input clip expression, and a builder
that maps each supported plugin to the vpy call that runs it). Plugins are
described separately in PLUGINS. Adding a new filter = one new entry; adding a
new reference plugin = one new entry.

Timing is done with vspipe so results stay comparable across plugins.

Usage:
    python3 benchmark/bench.py                                   # all filters
    python3 benchmark/bench.py --filter gaussblur                # one filter
    python3 benchmark/bench.py --filter gaussblur vsfeel vszipcl # subset of plugins
    python3 benchmark/bench.py --filter gaussblur --gauss-sigma 5.0
    python3 benchmark/bench.py --frames 500 --clip /path/to/input.mkv
    python3 benchmark/bench.py --no-cache          # live decode: full chain incl. BestSource

By default the first --cache-frames frames of the real clip are decoded and
held in RAM while vspipe is still evaluating the script (its fps figure only
covers the output loop), so timing reflects real-content filter throughput
without the BestSource decode bottleneck. --synthetic swaps real content for a
BlankClip; --no-cache restores live decoding.
"""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_CLIP = "/home/encode/test/jpbd.mkv"

def make_vpy(
    clip: str,
    extra: str,
    chain: str,
    frames: int,
    synth_format: str | None,
    cache_frames: int | None = None,
    cache_conv: str | None = None,
) -> str:
    if synth_format:
        # synthetic clip: measure pure filter throughput, no decode bottleneck
        clip_expr = (
            "core.std.BlankClip(width=1920, height=1080, "
            f"format={synth_format}, length={frames})"
        )
        cache_setup = ""
    elif cache_frames:
        # real clip, but decode + convert + hold the first N frames in Python
        # while the script is being evaluated (before vspipe starts timing),
        # then serve them through a ModifyFrame shim and loop to reach the
        # requested frame count. std.Cache() is an explicit no-op on current
        # VapourSynth. `cache_conv` is the filter's input expression (e.g.
        # "depth(clip, 16)") applied BEFORE caching, so the timed region
        # measures only filter throughput on the correct input format — the
        # chain's own input expression then reduces to an identity.
        clip_expr = f"BestSource(cachepath=None).source({clip!r}, 32)"
        conv = f"clip = {cache_conv}\n" if cache_conv else ""
        cache_setup = (
            f"m = min({cache_frames}, clip.num_frames)\n"
            f"{conv}"
            "_src_frames = [clip.get_frame(n) for n in range(m)]\n"
            "def _serve_cached(n, f):\n"
            "    return _src_frames[n % m]\n"
            "_blank = clip.std.BlankClip()\n"
            "_served = _blank.std.ModifyFrame(_blank, _serve_cached)\n"
            f"clip = (_served * -(-{frames} // m)).std.Trim(0, {frames - 1})\n"
        )
    else:
        clip_expr = f"BestSource(cachepath=None).source({clip!r}, 32)"
        cache_setup = ""
    return f"""\
from vssource import BestSource
from vstools import core, depth, get_y
import vapoursynth as vs

core.max_cache_size = 1024 * 48

clip = {clip_expr}

{cache_setup}
{extra}

{chain}.set_output()
"""


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

@dataclass
class Plugin:
    name: str
    loader: str | None = None  # extra vpy line required to make it available


PLUGINS = {
    "vsfeel": Plugin("vsfeel"),
    "vszipcl": Plugin("vszipcl"),
    "vszipcu": Plugin(
        "vszipcu",
        loader='core.std.LoadPlugin("/home/encode/test/vapoursynth-ziphip/zig-out/lib/libvszipcu.so")',
    ),
    "bilateralhip": Plugin("bilateralhip"),
    "bm3dhip": Plugin("bm3dhip"),
    "nlm_hip": Plugin("nlm_hip"),
}


# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

@dataclass
class Arg:
    """One filter parameter exposed as a CLI flag.

    ``key`` is the parameter name used inside the vpy call (and in the printed
    description); ``flag``/``dest`` are the CLI spelling and its namespace
    attribute (prefixed so different filters never collide when run together).
    """

    key: str
    flag: str
    dest: str
    type: type
    default: Any
    help: str = ""


@dataclass
class FilterSpec:
    title: str
    default_frames: int
    args: list[Arg]
    build: Callable[[argparse.Namespace, str], dict[str, str]]
    input: str = "depth(get_y(clip), 16)"  # clip expression the filter is applied to
    synth_format: str | None = "vs.GRAY16"  # BlankClip format for --synthetic (None disables)
    default_streams: int = 4  # num_streams when --num-streams is not given


def _bm3d_build(ns: argparse.Namespace, clip: str) -> dict[str, str]:
    ns_num = ns.num_streams if ns.num_streams is not None else 4
    common = (
        f"sigma={ns.bm3d_sigma}, radius={ns.bm3d_radius}, "
        f"bm_range={ns.bm3d_bm_range}, ps_range={ns.bm3d_ps_range}, "
        f"block_step={ns.bm3d_block_step}"
    )
    with_streams = f"{common}, num_streams={ns_num}"
    return {
        "vsfeel": f"core.vsfeel.BM3Dv2({clip}, {with_streams})",
        "vszipcl": f"core.vszipcl.BM3Dv2({clip}, {with_streams})",
        "vszipcu": f"core.vszipcu.BM3Dv2({clip}, {with_streams})",
        "bm3dhip": f"core.bm3dhip.BM3Dv2({clip}, {common})",  # no num_streams
    }


def _bilateral_build(ns: argparse.Namespace, clip: str) -> dict[str, str]:
    ns_num = ns.num_streams if ns.num_streams is not None else 4
    args = (
        f"sigma_spatial={ns.bilateral_sigma_spatial}, "
        f"sigma_color={ns.bilateral_sigma_color}, num_streams={ns_num}"
    )
    return {
        "vsfeel": f"core.vsfeel.Bilateral({clip}, {args})",
        "vszipcl": f"core.vszipcl.Bilateral({clip}, {args})",
        "vszipcu": f"core.vszipcu.Bilateral({clip}, {args})",
        "bilateralhip": f"core.bilateralhip.Bilateral({clip}, {args})",
    }


def _gauss_build(ns: argparse.Namespace, clip: str) -> dict[str, str]:
    ns_num = ns.num_streams if ns.num_streams is not None else 4
    args = f"sigma={ns.gauss_sigma}, num_streams={ns_num}"
    return {
        "vsfeel": f"core.vsfeel.GaussBlur({clip}, {args})",
        "vszipcl": f"core.vszipcl.GaussBlur({clip}, {args})",
        "vszipcu": f"core.vszipcu.GaussBlur({clip}, {args})",
    }


def _dfttest_build(ns: argparse.Namespace, clip: str) -> dict[str, str]:
    # all three plugins share the vszipcu parameter surface; the references
    # default num_streams to 1, so this filter honours the global --num-streams
    # only when explicitly given (defaulting to 1 otherwise)
    num_streams = ns.num_streams if ns.num_streams is not None else 1
    args = (
        f"ftype={ns.dfttest_ftype}, sigma={ns.dfttest_sigma}, sigma2={ns.dfttest_sigma2}, "
        f"pmin={ns.dfttest_pmin}, pmax={ns.dfttest_pmax}, sbsize=16, sosize={ns.dfttest_sosize}, "
        f"tbsize={ns.dfttest_tbsize}, swin={ns.dfttest_swin}, twin={ns.dfttest_twin}, "
        f"sbeta={ns.dfttest_sbeta}, tbeta={ns.dfttest_tbeta}, zmean={ns.dfttest_zmean}, "
        f"f0beta={ns.dfttest_f0beta}, num_streams={num_streams}"
    )
    return {
        "vsfeel": f"core.vsfeel.DFTTest({clip}, {args})",
        "vszipcl": f"core.vszipcl.DFTTest({clip}, {args})",
        "vszipcu": f"core.vszipcu.DFTTest({clip}, {args})",
    }


def _nlmeans_build(ns: argparse.Namespace, clip: str) -> dict[str, str]:
    ns_num = ns.num_streams if ns.num_streams is not None else 2
    args = (
        f"d={ns.nlmeans_d}, a={ns.nlmeans_a}, s={ns.nlmeans_s}, h={ns.nlmeans_h}, "
        f"wmode={ns.nlmeans_wmode}, wref={ns.nlmeans_wref}, channels='UV', "
        f"num_streams={ns_num}"
    )
    return {
        "vsfeel": f"core.vsfeel.NLMeans({clip}, {args})",
        "vszipcl": f"core.vszipcl.NLMeans({clip}, {args})",
        "vszipcu": f"core.vszipcu.NLMeans({clip}, {args})",
        "nlm_hip": f"core.nlm_hip.NLMeans({clip}, {args})",
    }


FILTERS: dict[str, FilterSpec] = {
    "bm3dv2": FilterSpec(
        title="BM3Dv2",
        default_frames=1000,
        args=[
            Arg("sigma", "--bm3d-sigma", "bm3d_sigma", float, 0.7),
            Arg("radius", "--bm3d-radius", "bm3d_radius", int, 2),
            Arg("bm_range", "--bm3d-bm-range", "bm3d_bm_range", int, 16),
            Arg("ps_range", "--bm3d-ps-range", "bm3d_ps_range", int, 7),
            Arg("block_step", "--bm3d-block-step", "bm3d_block_step", int, 4),
        ],
        build=_bm3d_build,
        input="depth(get_y(clip), 32)",
        synth_format="vs.GRAYS",
    ),
    "bilateral": FilterSpec(
        title="Bilateral",
        default_frames=5000,
        args=[
            Arg("sigma_spatial", "--bilateral-sigma-spatial", "bilateral_sigma_spatial", float, 3.0),
            Arg("sigma_color", "--bilateral-sigma-color", "bilateral_sigma_color", float, 0.02),
        ],
        build=_bilateral_build,
    ),
    "gaussblur": FilterSpec(
        title="GaussBlur",
        default_frames=5000,
        args=[
            Arg("sigma", "--gauss-sigma", "gauss_sigma", float, 16.0),
        ],
        build=_gauss_build,
    ),
    "dfttest": FilterSpec(
        title="DFTTest",
        default_frames=3000,
        args=[
            Arg("ftype", "--dfttest-ftype", "dfttest_ftype", int, 0),
            Arg("sigma", "--dfttest-sigma", "dfttest_sigma", float, 8.0),
            Arg("sigma2", "--dfttest-sigma2", "dfttest_sigma2", float, 8.0),
            Arg("pmin", "--dfttest-pmin", "dfttest_pmin", float, 0.0),
            Arg("pmax", "--dfttest-pmax", "dfttest_pmax", float, 500.0),
            Arg("sosize", "--dfttest-sosize", "dfttest_sosize", int, 12),
            Arg("tbsize", "--dfttest-tbsize", "dfttest_tbsize", int, 3),
            Arg("swin", "--dfttest-swin", "dfttest_swin", int, 0),
            Arg("twin", "--dfttest-twin", "dfttest_twin", int, 7),
            Arg("sbeta", "--dfttest-sbeta", "dfttest_sbeta", float, 2.5),
            Arg("tbeta", "--dfttest-tbeta", "dfttest_tbeta", float, 2.5),
            Arg("zmean", "--dfttest-zmean", "dfttest_zmean", int, 1),
            Arg("f0beta", "--dfttest-f0beta", "dfttest_f0beta", float, 1.0),
        ],
        build=_dfttest_build,
        input="depth(get_y(clip), 16)",
        default_streams=1,
    ),
    "nlmeans": FilterSpec(
        title="NLMeans",
        default_frames=3000,
        args=[
            Arg("d", "--nlmeans-d", "nlmeans_d", int, 2),
            Arg("a", "--nlmeans-a", "nlmeans_a", int, 2),
            Arg("s", "--nlmeans-s", "nlmeans_s", int, 4),
            Arg("h", "--nlmeans-h", "nlmeans_h", float, 0.2),
            Arg("wmode", "--nlmeans-wmode", "nlmeans_wmode", int, 0),
            Arg("wref", "--nlmeans-wref", "nlmeans_wref", float, 1.0),
        ],
        build=_nlmeans_build,
        # chroma denoising on the subsampled planes is NLMeans' main use case
        input="depth(clip, 16)",
        synth_format="vs.YUV420P16",
        default_streams=2,
    ),
}


# ---------------------------------------------------------------------------
# vspipe timing
# ---------------------------------------------------------------------------

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


def bench(plugin: str, chain: str, clip: str, frames: int, synth_format: str | None,
          cache_frames: int | None = None, cache_conv: str | None = None) -> float | None:
    vpy = make_vpy(
        clip=clip,
        extra=PLUGINS[plugin].loader or "",
        chain=chain,
        frames=frames,
        synth_format=synth_format,
        cache_frames=cache_frames,
        cache_conv=cache_conv,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"bench_{plugin}.vpy"
        path.write_text(vpy)
        return run_vspipe(path, frames)


def args_desc(spec: FilterSpec, ns: argparse.Namespace) -> str:
    pairs = [f"{a.key}={getattr(ns, a.dest)}" for a in spec.args]
    num = ns.num_streams if ns.num_streams is not None else spec.default_streams
    pairs.append(f"num_streams={num}")
    return ", ".join(pairs)


def bench_filter(spec: FilterSpec, ns: argparse.Namespace) -> None:
    calls = spec.build(ns, spec.input)
    plugins = ns.plugins or list(calls)
    plugins = [p for p in plugins if p in calls]
    if not plugins:
        sys.exit(f"no valid plugins requested for --filter {ns.filter}")

    frames = ns.frames or spec.default_frames
    synth = spec.synth_format if ns.synthetic else None
    cache_frames = ns.cache_frames if (ns.cached and synth is None) else None
    # the cache holds frames in the filter's input format (e.g. depth(clip,16))
    # so the timed region measures only filter throughput, like --synthetic
    cache_conv = spec.input if cache_frames else None
    if synth:
        clip_desc = f"BlankClip 1920x1080 {synth.removeprefix('vs.')}"
    else:
        clip_desc = str(ns.clip)
    print(f"{spec.title} benchmark | {frames} frames | clip: {clip_desc}")
    print(f"args: {args_desc(spec, ns)}\n")

    results = []
    for plugin in plugins:
        fps = bench(plugin, calls[plugin], ns.clip, frames, synth, cache_frames, cache_conv)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vsfeel filters against the reference implementations."
    )
    parser.add_argument("plugins", nargs="*",
                        help="plugins to run (default: every plugin providing the filter)")
    parser.add_argument("-f", "--filter", choices=[*FILTERS, "all"], default="all",
                        help="filter to benchmark (default: all)")
    parser.add_argument("--frames", type=int, default=None,
                        help="frames to time (default: per-filter, see FILTERS)")
    parser.add_argument("--num-streams", type=int, default=None,
                        help="num_streams passed to the filters (default: 1, or the reference default)")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="input clip path")
    parser.add_argument("--synthetic", action="store_true",
                        help="use a synthetic 1920x1080 YUV420PS BlankClip instead of --clip "
                             "(decoder-independent, measures pure filter throughput)")
    parser.add_argument("--cache", dest="cached", action="store_true", default=True,
                        help="default mode: decode the first --cache-frames frames of the real clip "
                             "into RAM before vspipe starts timing (removes the BestSource bottleneck "
                             "while keeping real content; frames loop to reach --frames)")
    parser.add_argument("--no-cache", dest="cached", action="store_false",
                        help="decode live during timing instead (measures the full chain, "
                             "bottlenecked by BestSource at ~630 fps)")
    parser.add_argument("--cache-frames", type=int, default=1000,
                        help="number of leading frames to preload with --cache (default: 1000)")

    for spec in FILTERS.values():
        for arg in spec.args:
            parser.add_argument(arg.flag, dest=arg.dest, type=arg.type,
                                default=arg.default, help=arg.help)

    return parser.parse_args()


def main() -> None:
    ns = parse_args()
    ns.clip = str(Path(ns.clip).expanduser().resolve())
    filters = list(FILTERS) if ns.filter == "all" else [ns.filter]
    for fname in filters:
        bench_filter(FILTERS[fname], ns)


if __name__ == "__main__":
    main()