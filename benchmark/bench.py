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
import re
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
        # VapourSynth. `cache_conv` is the cache expression (e.g.
        # "depth(get_y(clip),16)" or a 2x Point upscale), so the timed region
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
# AA-style vpy (EEDI3 anti-aliasing benchmark, mirrors vsaa.based_aa)
# ---------------------------------------------------------------------------

AA_MASK_BITS = 16  # mask/depth chain runs at 16-bit like the ss clip

def make_aa_vpy(
    clip: str,
    extra: str,
    chain: str,
    frames: int,
    cache_frames: int | None = 400,
    eedi3_field: int = 3,
) -> str:
    """Real-clip vpy that mirrors vsaa.based_aa's EEDI3 usage:

    - luma of the source at AA_MASK_BITS (16)
    - edge mask: Prewitt -> binarize(mask_thr=60 scaled to depth) -> box_blur(Maximum)
    - both luma and mask are Point-upscaled 2x (the "supersampling"; the user
      chose a plain Point upscale over ArtCNN)
    - the first `cache_frames` 2x luma AND 2x mask frames are decoded into RAM
      while vspipe evaluates the script, so the timed region covers only the
      EEDI3 call (with its mclip/sclip), not the CPU mask/upscale work.
      cache_frames=None keeps a live decode (full chain incl. mask work).
    The chain string receives `clip` (the 2x luma), `mclip` (the 2x mask) and
    `sclip`. based_aa runs its default EEDI3 in double-rate mode (field = 3:
    tff + double_rate*2), so the filter outputs 2 frames per input frame and
    sclip must describe the output: one frame per OUTPUT frame, built exactly
    like based_aa does with Interleave([s, s]). When eedi3_field <= 1 the
    sclip is the plain clip.
    """
    if cache_frames is not None:
        cache_lines = f"""\
m = min({cache_frames}, ss.num_frames)
_ss_frames = [ss.get_frame(n) for n in range(m)]
_msk_frames = [mclip.get_frame(n) for n in range(m)]
def _serve_ss(n, f):
    return _ss_frames[n % m]
def _serve_msk(n, f):
    return _msk_frames[n % m]
_blank = ss.std.BlankClip()
clip = (_blank.std.ModifyFrame(_blank, _serve_ss) * -(-{frames} // m)).std.Trim(0, {frames} - 1)
_blankm = mclip.std.BlankClip()
mclip = (_blankm.std.ModifyFrame(_blankm, _serve_msk) * -(-{frames} // m)).std.Trim(0, {frames} - 1)
sclip = {f"core.std.Interleave([clip, clip])" if eedi3_field > 1 else "clip"}
"""
    else:
        sclip_expr = "core.std.Interleave([ss, ss])" if eedi3_field > 1 else "clip"
        cache_lines = f"clip = ss\nsclip = {sclip_expr}\n"
    return f"""\
from vssource import BestSource
from vstools import core, depth, get_y
from vsmasktools import EdgeDetect, Morpho, Prewitt
from vsrgtools import box_blur
from vstools import scale_mask
import vapoursynth as vs

core.max_cache_size = 1024 * 48

src = BestSource(cachepath=None).source({clip!r}, 32)
luma = depth(get_y(src), {AA_MASK_BITS})

# vsaa.based_aa mask chain (defaults: Prewitt, mask_thr=60)
mask = EdgeDetect.ensure_obj(Prewitt).edgemask(luma)
mask = Morpho.binarize_mask(mask, scale_mask(60, 8, {AA_MASK_BITS}))
mask = box_blur(mask.std.Maximum())

# Point 2x upscale of input + mask (user: "double the size of the input frames
# in the cache ... a simple Point upscale is good enough")
ss = core.resize.Point(luma, luma.width * 2, luma.height * 2)
mclip = core.resize.Point(mask, mask.width * 2, mask.height * 2)

{cache_lines}
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
    "eedi3vk2": Plugin("eedi3vk2"),
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
    ``type`` is the argparse value parser; bool Args use _str_to_bool.
    """

    key: str
    flag: str
    dest: str
    type: type
    default: Any
    help: str = ""


def _str_to_bool(v: str) -> bool:
    """argparse parser for bool flags (accepts 0/1/true/false/yes/no/on/off)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


@dataclass
class FilterSpec:
    title: str
    default_frames: int
    args: list[Arg]
    build: Callable[[argparse.Namespace, str], dict[str, str]]
    input: str = "depth(get_y(clip), 16)"  # clip expression the filter is applied to
    synth_format: str | None = "vs.GRAY16"  # BlankClip format for --synthetic (None disables)
    default_streams: int = 4  # num_streams when --num-streams is not given
    # When set, the real-clip benchmark uses a custom vpy (see make_aa_vpy) that
    # doubles the input with a Point upscale and feeds the filter auxiliary clips
    # derived from it. Only sensible for EEDI3-style AA benchmarks.
    aa: bool = False


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


def _eedi3_build(ns: argparse.Namespace, clip: str) -> dict[str, str]:
    """EEDI3 anti-aliasing chain (mirrors vsaa.based_aa defaults).

    Real-clip runs (aa FilterSpec) use make_aa_vpy, which defines `clip`
    (2x Point-upscaled luma), `sclip` (= clip) and `mclip` (2x upscaled vsaa
    edge mask). vsfeel and eedi3vk2 take both sclip and mclip (based_aa passes
    mclip when the backend supports it); vszipcl/vszipcu support sclip only.
    --eedi3-mclip 0 drops the mclip from the vsfeel/eedi3vk2 calls.

    """
    ns_num = ns.num_streams if ns.num_streams is not None else 4
    use_mclip = getattr(ns, "eedi3_mclip", True)
    common = (
        f"field={ns.eedi3_field}, mdis={ns.eedi3_mdis}, nrad={ns.eedi3_nrad}, "
        f"alpha={ns.eedi3_alpha}, beta={ns.eedi3_beta}, gamma={ns.eedi3_gamma}, "
        f"vcheck={ns.eedi3_vcheck}, vthresh0={ns.eedi3_vthresh0}, "
        f"vthresh1={ns.eedi3_vthresh1}, vthresh2={ns.eedi3_vthresh2}"
    )
    if getattr(ns, "synthetic", False):
        return {
            "vsfeel": f"core.vsfeel.EEDI3({clip}, {common}, num_streams={ns_num})",
            "eedi3vk2": f"core.eedi3vk2.EEDI3({clip}, {common}, num_streams={ns_num})",
            "vszipcl": f"core.vszipcl.EEDI3({clip}, {common}, num_streams={ns_num})",
            "vszipcu": f"core.vszipcu.EEDI3({clip}, {common}, num_streams={ns_num})",
        }
    if use_mclip:
        mcap = "sclip=sclip, mclip=mclip"
    else:
        mcap = "sclip=sclip"
    with_mclip = f"{common}, {mcap}"
    with_sclip = f"{common}, sclip=sclip"
    return {
        "vsfeel": f"core.vsfeel.EEDI3({clip}, {with_mclip}, num_streams={ns_num})",
        "eedi3vk2": f"core.eedi3vk2.EEDI3({clip}, {with_mclip}, num_streams={ns_num})",
        "vszipcl": f"core.vszipcl.EEDI3({clip}, {with_sclip}, num_streams={ns_num})",
        "vszipcu": f"core.vszipcu.EEDI3({clip}, {with_sclip}, num_streams={ns_num})",
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
    "eedi3": FilterSpec(
        title="EEDI3",
        default_frames=2000,
        args=[
            # based_aa runs its default EEDI3 antialiaser in double-rate mode:
            # field = tff + double_rate*2 = 3 (progressive input, tff=True), i.e.
            # 2 output frames per input frame interpolating each field parity,
            # which based_aa folds back with std.Merge(clip[::2], clip[1::2]).
            # The merge is not EEDI3 work, so the benchmark times field=3 alone
            # (the exact double-rate call based_aa makes) and skips the merge.
            Arg("field", "--eedi3-field", "eedi3_field", int, 3),
            Arg("mdis", "--eedi3-mdis", "eedi3_mdis", int, 20),
            Arg("nrad", "--eedi3-nrad", "eedi3_nrad", int, 2),
            Arg("alpha", "--eedi3-alpha", "eedi3_alpha", float, 0.125),
            Arg("beta", "--eedi3-beta", "eedi3_beta", float, 0.25),
            Arg("gamma", "--eedi3-gamma", "eedi3_gamma", float, 40.0),
            Arg("vcheck", "--eedi3-vcheck", "eedi3_vcheck", int, 2),
            Arg("vthresh0", "--eedi3-vthresh0", "eedi3_vthresh0", float, 12.0),
            Arg("vthresh1", "--eedi3-vthresh1", "eedi3_vthresh1", float, 24.0),
            Arg("vthresh2", "--eedi3-vthresh2", "eedi3_vthresh2", float, 4.0),
            Arg("mclip", "--eedi3-mclip", "eedi3_mclip", _str_to_bool, True,
                "pass the vsaa edge mask as mclip to vsfeel/eedi3vk2 (default: true)"),
        ],
        build=_eedi3_build,
        # Real runs mirror vsaa.based_aa: source at 16-bit, vsaa edge mask,
        # both Point-upscaled 2x, then EEDI3 with sclip/mclip. --synthetic
        # (used by the hang tests) falls back to a plain 1x GRAY16 call.
        input="depth(get_y(clip), 16)",
        aa=True,
        default_streams=4,
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


def bench_aa(plugin: str, chain: str, clip: str, frames: int,
             cache_frames: int | None = None,
             eedi3_field: int = 3) -> float | None:
    """Run an EEDI3 anti-aliasing style benchmark (see make_aa_vpy)."""
    vpy = make_aa_vpy(
        clip=clip,
        extra=PLUGINS[plugin].loader or "",
        chain=chain,
        frames=frames,
        cache_frames=cache_frames,
        eedi3_field=eedi3_field,
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


def _format_for_bits(fmt: str, bits: int) -> str:
    """Return the sibling of a ``vs.`` synthetic format name at the requested
    bit depth (GRAYS <-> GRAY16, YUV420PS <-> YUV420P16, ...)."""
    m = re.match(r"^(vs\.[A-Za-z0-9]+?)(S|16|PS|P16)$", fmt)
    if not m:
        raise SystemExit(
            f"--bits {bits}: cannot map synthetic format {fmt!r} to {bits}-bit")
    base = m.group(1)
    suffix = "16" if bits == 16 else "S"
    if not base.endswith("GRAY"):
        suffix = "P16" if bits == 16 else "PS"
    return f"{base}{suffix}"


def _input_for_bits(expr: str, bits: int) -> str:
    """Override the depth argument of the cache-conversion expression
    (``depth(X, N)`` -> ``depth(X, bits)``)."""
    return re.sub(r"depth\(([^,]+), \d+\)",
                  lambda m: f"depth({m.group(1)}, {bits})", expr)


def bench_filter(spec: FilterSpec, ns: argparse.Namespace) -> None:
    # --bits overrides the filter's input expression (depth(X,16) -> depth(X,32))
    # for BOTH the chain and the cached frames: the chain must consume the same
    # format the cache holds, or the timed region re-converts behind the filter
    input_expr = _input_for_bits(spec.input, ns.bits) if ns.bits else spec.input
    calls = spec.build(ns, input_expr)
    plugins = ns.plugins or list(calls)
    plugins = [p for p in plugins if p in calls]
    if not plugins:
        sys.exit(f"no valid plugins requested for --filter {ns.filter}")

    frames = ns.frames or spec.default_frames
    synth = spec.synth_format if ns.synthetic else None
    if synth and ns.bits:
        synth = _format_for_bits(synth, ns.bits)
    cache_frames = ns.cache_frames if (ns.cached and synth is None) else None
    # the cache holds frames in the filter's input format (e.g. depth(clip,16))
    # so the timed region measures only filter throughput, like --synthetic
    cache_conv = input_expr if cache_frames else None
    if synth:
        clip_desc = f"BlankClip 1920x1080 {synth.removeprefix('vs.')}"
    elif spec.aa:
        clip_desc = f"{str(ns.clip)} -> 2x Point"
    else:
        clip_desc = str(ns.clip)
    bits_desc = f" | bits: {ns.bits}" if ns.bits else ""
    print(f"{spec.title} benchmark | {frames} frames | clip: {clip_desc}{bits_desc}")
    print(f"args: {args_desc(spec, ns)}\n")

    results = []
    for plugin in plugins:
        if spec.aa and synth is None:
            # AA benchmark on real content: cache holds the 2x Point-upscaled
            # luma + the 2x edge mask (both ~16.6 MB/frame at 1080p->2160p
            # GRAY16), so the timed region measures only the EEDI3 call.
            fps = bench_aa(plugin, calls[plugin], ns.clip, frames,
                           min(cache_frames or 400, 500),
                           getattr(ns, "eedi3_field", 3))
        else:
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
                        help="use a synthetic 1920x1080 BlankClip instead of --clip "
                             "(decoder-independent, measures pure filter throughput)")
    parser.add_argument("--bits", type=int, choices=(16, 32), default=None,
                        help="benchmark the 16-bit or 32-bit input path: with "
                             "--synthetic this swaps the BlankClip format, otherwise "
                             "it overrides the cached-frame conversion depth "
                             "(default: each filter's configured depth)")
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