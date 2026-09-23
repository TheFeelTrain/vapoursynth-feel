#!/usr/bin/env python3
"""Benchmark vsfeel filters against the reference implementations.

The benchmark is data-driven: every filter is described by one entry in the
FILTERS registry below (its CLI args, the input clip expression, and a builder
that maps each supported plugin to the vpy call that runs it). Plugins are
described separately in PLUGINS. Adding a new filter = one new entry; adding a
new reference plugin = one new entry.

Timing is done with vspipe so results stay comparable across plugins.

Usage:
    python3 tools/benchmark.py                                   # all filters
    python3 tools/benchmark.py --filter gaussblur                # one filter
    python3 tools/benchmark.py --filter gaussblur vsfeel vszipcl # subset of plugins
    python3 tools/benchmark.py --filter gaussblur --gauss-sigma 5.0
    python3 tools/benchmark.py --filter gaussblur --repeat 5      # median of 5, alternating order
    python3 tools/benchmark.py --filter dfttest --pair vszipcl    # same-session pair + ratio
    python3 tools/benchmark.py --frames 500 --clip /path/to/input.mkv
    python3 tools/benchmark.py --no-cache          # live decode: full chain incl. BestSource
    python3 tools/benchmark.py --check-fresh       # refuse to run against a stale .so

Every plugin is timed --repeat times (default 3) and the median is reported with
min/max/spread; the plugin order alternates between repeats so clock/thermal
drift hits both arms of a comparison equally. Each vspipe run gets the
environment from ``vspipe_env()`` -- MANGOHUD off, and RADV's transfer-only
SDMA queue opted into -- and is killed after --timeout seconds.

By default the first --cache-frames frames of the real clip are decoded and
held in RAM while vspipe is still evaluating the script (its fps figure only
covers the output loop), so timing reflects real-content filter throughput
without the BestSource decode bottleneck. --synthetic swaps real content for a
BlankClip; --no-cache restores live decoding.
"""

import argparse
import hashlib
import importlib.util
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_CLIP = "/home/encode/test/jpbd.mkv"
DEFAULT_TIMEOUT = 900  # seconds per vspipe run (H1: a hang must not hang the harness)
# AA real-clip runs cache the 2x luma AND the 2x mask; cap them by bytes, not
# frames (H8). The budget is exclusive of VapourSynth's own 48 GB frame cache,
# which is additive: 6 GiB keeps the pair well under the incident point.
AA_CACHE_MB = 6144
REPO_ROOT = Path(__file__).resolve().parent.parent

def make_vpy(
    clip: str,
    extra: str,
    chain: str,
    frames: int,
    synth_format: str | None,
    cache_frames: int | None = None,
    cache_conv: str | None = None,
    gpu_cache: bool = False,
    gpu_cache_mb: int | None = None,
) -> str:
    # With --gpu-cache the script also defines `clip_gpu`: the same frames,
    # already resident on the device. The upload happens while the script is
    # evaluated (before vspipe starts timing), exactly like the CPU preload, so a
    # GPU-capable filter can be timed without the GPUUpload stage in the graph.
    # Every arm is given it: plugins that declare vnode:gpu/vnode:all consume it
    # directly, and the rest are called on std.GPUDownload(clip_gpu) so they pay
    # the transfer a real chain would charge them. Handing a legacy plugin the
    # CPU cache instead is not a fair comparison -- its neighbour's input sits in
    # VRAM and it would never see that cost.
    #
    # Two traps when measuring transfers against this:
    #  - `std.GPUDownload(<CPU clip>)` is a silent no-op. gpuTransferCreate
    #    returns its input unchanged whenever the residency already matches, so
    #    a "download" arm built on a CPU clip does no GPU work at all (measured
    #    3746 fps) and makes an upload arm look expensive next to nothing. Only
    #    `GPUDownload(GPUUpload(clip))` exercises a real transfer.
    #  - Serving the cached GPU frames through a Python ModifyFrame costs more
    #    than the upload it removes: that node is fmParallelRequests, so its
    #    callback serializes the whole front of the chain (509 vs 533 fps at
    #    radius 0). clip_gpu is therefore a plain node and the GPU frames are
    #    warmed into the core's own frame cache instead.
    gpu_lines = ""
    if synth_format:
        # synthetic clip: measure pure filter throughput, no decode bottleneck
        clip_expr = (
            "core.std.BlankClip(width=1920, height=1080, "
            f"format={synth_format}, length={frames})"
        )
        cache_setup = ""
        gpu_lines = "clip_gpu = core.std.GPUUpload(clip=clip)\n" if gpu_cache else ""
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
        # The GPU copy of the cache is whole frames in VRAM, so it gets a byte
        # budget of its own; both caches then serve the same span.
        size_lines = (
            "_fbytes = (clip.width * clip.height * clip.format.bytes_per_sample\n"
            "           * clip.format.num_planes)\n"
            f"_cap = max(1, {gpu_cache_mb} * 1024 * 1024 // _fbytes)\n"
            f"m = min({cache_frames}, _cap, clip.num_frames)\n"
        ) if (gpu_cache and gpu_cache_mb) else f"m = min({cache_frames}, clip.num_frames)\n"
        # The cap has to be measured on the cached format, so with --gpu-cache
        # the conversion comes first; without it the order is unchanged.
        if gpu_cache:
            head = conv + size_lines
        else:
            head = f"m = min({cache_frames}, clip.num_frames)\n" + conv
        cache_setup = (
            f"{head}"
            "_src_frames = [clip.get_frame(n) for n in range(m)]\n"
            "def _serve_cached(n, f):\n"
            "    return _src_frames[n % m]\n"
            "_blank = clip.std.BlankClip()\n"
            "_served = _blank.std.ModifyFrame(_blank, _serve_cached)\n"
            f"clip = (_served * -(-{frames} // m)).std.Trim(0, {frames - 1})\n"
        )
        if gpu_cache:
            # clip_gpu stays a plain native node: the uploads are warmed into the
            # core's own frame cache here (outside the timed region) and the
            # cached frames are then served straight from it. Serving them
            # through a Python ModifyFrame instead costs more than the upload it
            # removes -- that node is fmParallelRequests, so its callback
            # serializes the whole front of the chain (measured 509 vs 533 fps at
            # radius 0, i.e. the "no upload" arm was the slower one).
            gpu_lines = (
                "_gpu_src = core.std.GPUUpload(clip=_served)\n"
                "_gpu_warm = [_gpu_src.get_frame(n) for n in range(m)]\n"
                f"clip_gpu = (_gpu_src * -(-{frames} // m)).std.Trim(0, {frames - 1})\n"
            )
    else:
        clip_expr = f"BestSource(cachepath=None).source({clip!r}, 32)"
        cache_setup = ""
        gpu_lines = "clip_gpu = core.std.GPUUpload(clip=clip)\n" if gpu_cache else ""
    return f"""\
from vssource import BestSource
from vstools import core, depth, get_y
import vapoursynth as vs

# Big enough that the timed region never re-runs the decode/mask chain (a
# small value here collapses every plugin: measured 61 fps vs 220 for vsfeel
# and 39 vs 200 for vszipcl at 8 GB).
core.max_cache_size = 1024 * 48

clip = {clip_expr}

{cache_setup}
{gpu_lines}
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
    bits: int = AA_MASK_BITS,
    cache_bytes: int | None = None,
    gpu_cache: bool = False,
    download_inputs: bool = False,
) -> str:
    """Real-clip vpy that mirrors vsaa.based_aa's EEDI3 usage:

    - luma of the source at `bits` (16 or 32, like the ss clip)
    - edge mask: Prewitt -> binarize(mask_thr=60 scaled to depth) -> box_blur(Maximum)
    - both luma and mask are Point-upscaled 2x (the "supersampling"; the user
      chose a plain Point upscale over ArtCNN)
    - up to `cache_frames` 2x luma AND 2x mask frames are decoded into RAM
      while vspipe evaluates the script, so the timed region covers only the
      EEDI3 call (with its mclip/sclip), not the CPU mask/upscale work; the
      script additionally caps the hold at `cache_bytes` computed from the
      actual frame size, so the budget scales with bit depth.
      cache_frames=None keeps a live decode (full chain incl. mask work).
    The chain string receives `clip` (the 2x luma), `mclip` (the 2x mask) and
    `sclip`. based_aa runs its default EEDI3 in double-rate mode (field = 3:
    tff + double_rate*2), so the filter outputs 2 frames per input frame and
    sclip must describe the output: one frame per OUTPUT frame, built exactly
    like based_aa does with Interleave([s, s]). When eedi3_field <= 1 the
    sclip is the plain clip. Only the fused `core.vsfeel.EEDI3AA` arm consumes
    `sclip` directly; the reference arms are the vsaa antialiaser, which takes
    the single-rate `clip` as its sclip and interleaves it itself, just as
    based_aa does.
    """
    if cache_frames is not None:
        if cache_bytes:
            cap_lines = (
                "_ss_b = (ss.width * ss.height * ss.format.bytes_per_sample\n"
                "         * ss.format.num_planes)\n"
                "_msk_b = (mclip.width * mclip.height * mclip.format.bytes_per_sample\n"
                "          * mclip.format.num_planes)\n"
                f"_cap = max(1, {cache_bytes} // max(1, _ss_b + _msk_b))\n"
                f"m = min({cache_frames}, _cap, ss.num_frames)\n"
            )
        else:
            cap_lines = f"m = min({cache_frames}, ss.num_frames)\n"
        if gpu_cache:
            # --gpu-cache: put the cached frames on the device, so the timed
            # region has no GPUUpload stage in the graph. Both device caches are
            # warmed into the core's own frame cache while the script is
            # evaluated (outside the timed region), exactly like make_vpy's
            # clip_gpu; serving them through a Python ModifyFrame instead costs
            # more than the upload it removes (see make_vpy).
            #
            # download_inputs is the other half of the same comparison: an arm
            # whose filter does NOT take vnode:gpu still has to consume the
            # device-resident frames the graph produced, i.e. it pays
            # std.GPUDownload -- which is exactly what it would pay mid-chain.
            if download_inputs:
                names = (
                    "clip = core.std.GPUDownload(clip=clip_gpu)\n"
                    "mclip = core.std.GPUDownload(clip=mclip_gpu)\n"
                    "sclip = core.std.GPUDownload(clip=sclip_gpu)\n"
                )
            else:
                names = "clip = clip_gpu\nmclip = mclip_gpu\nsclip = sclip_gpu\n"
            cache_build = (
                "_gpu_ss = core.std.GPUUpload(clip=_ss_served)\n"
                "_gpu_msk = core.std.GPUUpload(clip=_msk_served)\n"
                "_gpu_warm = [_gpu_ss.get_frame(n) for n in range(m)]\n"
                "_gpu_mwarm = [_gpu_msk.get_frame(n) for n in range(m)]\n"
                f"clip_gpu = (_gpu_ss * -(-{frames} // m)).std.Trim(0, {frames} - 1)\n"
                f"mclip_gpu = (_gpu_msk * -(-{frames} // m)).std.Trim(0, {frames} - 1)\n"
                f"sclip_gpu = {{sclip_expr}}\n"
            ) + names
            gpu_sclip = ("core.std.Interleave([clip_gpu, clip_gpu])"
                         if eedi3_field > 1 else "clip_gpu")
            cache_build = cache_build.replace("{sclip_expr}", gpu_sclip)
        else:
            cache_build = (
                f"clip = (_ss_served * -(-{frames} // m)).std.Trim(0, {frames} - 1)\n"
                f"mclip = (_msk_served * -(-{frames} // m)).std.Trim(0, {frames} - 1)\n"
            )
        cache_lines = f"""\
{cap_lines}_ss_frames = [ss.get_frame(n) for n in range(m)]
_msk_frames = [mclip.get_frame(n) for n in range(m)]
def _serve_ss(n, f):
    return _ss_frames[n % m]
def _serve_msk(n, f):
    return _msk_frames[n % m]
_blank = ss.std.BlankClip()
_ss_served = _blank.std.ModifyFrame(_blank, _serve_ss)
_blankm = mclip.std.BlankClip()
_msk_served = _blankm.std.ModifyFrame(_blankm, _serve_msk)
{cache_build}
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

# Big enough that the timed region never re-runs the decode/mask chain (a
# small value here collapses every plugin: measured 61 fps vs 220 for vsfeel
# and 39 vs 200 for vszipcl at 8 GB).
core.max_cache_size = 1024 * 48

src = BestSource(cachepath=None).source({clip!r}, 32)
luma = depth(get_y(src), {bits})

# vsaa.based_aa mask chain (defaults: Prewitt, mask_thr=60). The threshold
# must be scaled to 32 (not to `bits`): Morpho.binarize_mask re-scales its
# midthr from the 32-bit float range to the clip's format, so passing an
# already-16-bit-scaled 15420 becomes 65535 and the mask comes out all zero
# (which silently disables mclip and turns EEDI3 into a row copier).
# vsaa/funcs.py:171 does exactly this with scale_mask(mask_thr, 8, 32).
mask = EdgeDetect.ensure_obj(Prewitt).edgemask(luma)
mask = Morpho.binarize_mask(mask, scale_mask(60, 8, 32))
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
    "vszipcu": Plugin("vszipcu"),
    "eedi3vk2": Plugin("eedi3vk2"),
    "nnedi3vk": Plugin("nnedi3vk"),
    "bilateralhip": Plugin("bilateralhip"),
    "bm3dhip": Plugin("bm3dhip"),
    "nlm_hip": Plugin("nlm_hip"),
    "bm3dvk": Plugin("bm3dvk"),
    "knlmvk": Plugin("knlmvk"),
}


def _plugin_loader(plugin: str) -> str:
    """Extra vpy line a plugin needs, with a clear error for an unknown name.

    ``PLUGINS[plugin].loader`` used to raise a bare ``KeyError`` from deep
    inside the run loop; a typo in ``--plugins`` is now reported at resolve
    time instead of being silently dropped.
    """
    if plugin not in PLUGINS:
        raise SystemExit(
            f"unknown plugin {plugin!r}: not in PLUGINS "
            f"({', '.join(sorted(PLUGINS))})")
    return PLUGINS[plugin].loader or ""


def resolve_plugins(requested: list[str], calls: dict[str, str],
                    title: str) -> list[str]:
    """Filter ``requested`` down to the plugins that provide this filter.

    Unknown names and plugins that do not implement the filter are reported on
    stderr rather than dropped silently, so a typo cannot produce a bogus pass.
    """
    out: list[str] = []
    for p in requested:
        if p not in PLUGINS:
            print(f"  [ignoring unknown plugin {p!r}: not in PLUGINS]", file=sys.stderr)
        elif p not in calls:
            print(f"  [ignoring plugin {p!r}: does not provide {title}]", file=sys.stderr)
        elif p not in out:
            out.append(p)
    return out


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
    build: Callable[[argparse.Namespace, str, "FilterSpec"], dict[str, str]]
    input: str = "depth(get_y(clip), 16)"  # clip expression the filter is applied to
    synth_format: str | None = "vs.GRAY16"  # BlankClip format for --synthetic (None disables)
    # When set, the real-clip benchmark uses a custom vpy (see make_aa_vpy) that
    # doubles the input with a Point upscale and feeds the filter auxiliary clips
    # derived from it. Only sensible for EEDI3-style AA benchmarks.
    aa: bool = False
    # Plugins whose call for this filter takes a GPU-resident input, i.e. the
    # filter is declared vnode:gpu or vnode:all. Under --gpu-cache only these get
    # `clip_gpu` (no core GPUUpload stage); every other arm keeps the CPU cache,
    # so one run still compares like with like.
    gpu_plugins: frozenset[str] = frozenset()


def _bm3d_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    common = (
        f"sigma={ns.bm3d_sigma}, radius={ns.bm3d_radius}, "
        f"bm_range={ns.bm3d_bm_range}, ps_range={ns.bm3d_ps_range}, "
        f"block_step={ns.bm3d_block_step}"
    )
    return {
        "vsfeel": f"core.vsfeel.BM3Dv2({clip}, {common})",
        "vszipcl": f"core.vszipcl.BM3Dv2({clip}, {common})",
        "bm3dvk": f"core.bm3dvk.BM3Dv2({clip}, {common})"
    }


def _bilateral_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    args = (
        f"sigma_spatial={ns.bilateral_sigma_spatial}, "
        f"sigma_color={ns.bilateral_sigma_color}"
    )
    return {
        "vsfeel": f"core.vsfeel.Bilateral({clip}, {args})",
        "vszipcl": f"core.vszipcl.Bilateral({clip}, {args})",
        "vszipcu": f"core.vszipcu.Bilateral({clip}, {args})"
    }


def _gauss_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    args = f"sigma={ns.gauss_sigma}"
    return {
        "vsfeel": f"core.vsfeel.GaussBlur({clip}, {args})",
        "vszipcl": f"core.vszipcl.GaussBlur({clip}, {args})",
        "vszipcu": f"core.vszipcu.GaussBlur({clip}, {args})",
    }


def _dfttest_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    # all three plugins share the vszipcu parameter surface
    args = (
        f"ftype={ns.dfttest_ftype}, sigma={ns.dfttest_sigma}, sigma2={ns.dfttest_sigma2}, "
        f"pmin={ns.dfttest_pmin}, pmax={ns.dfttest_pmax}, sbsize=16, sosize={ns.dfttest_sosize}, "
        f"tbsize={ns.dfttest_tbsize}, swin={ns.dfttest_swin}, twin={ns.dfttest_twin}, "
        f"sbeta={ns.dfttest_sbeta}, tbeta={ns.dfttest_tbeta}, zmean={ns.dfttest_zmean}, "
        f"f0beta={ns.dfttest_f0beta}"
    )
    return {
        "vsfeel": f"core.vsfeel.DFTTest({clip}, {args})",
        "vszipcl": f"core.vszipcl.DFTTest({clip}, {args})",
        "vszipcu": f"core.vszipcu.DFTTest({clip}, {args})",
    }


def _nlmeans_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    args = (
        f"d={ns.nlmeans_d}, a={ns.nlmeans_a}, s={ns.nlmeans_s}, h={ns.nlmeans_h}, "
        f"wmode={ns.nlmeans_wmode}, wref={ns.nlmeans_wref}, channels='UV'"
    )
    return {
        "vsfeel": f"core.vsfeel.NLMeans({clip}, {args})",
        "vszipcl": f"core.vszipcl.NLMeans({clip}, {args})",
        "vszipcu": f"core.vszipcu.NLMeans({clip}, {args})",
        "nlm_hip": f"core.nlm_hip.NLMeans({clip}, {args})",
        "knlmvk": (
            f"core.knlmvk.KNLMeans({clip}, d={ns.nlmeans_d}, a={ns.nlmeans_a}, "
            f"s={ns.nlmeans_s}, h={ns.nlmeans_h}, wmode={ns.nlmeans_wmode}, "
            f"wref={ns.nlmeans_wref}, channels='UV')"
        ),
    }


def _eedi3_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    """EEDI3 anti-aliasing chain (mirrors vsaa.based_aa defaults).

    Real-clip runs (aa FilterSpec) use make_aa_vpy, which defines `clip`
    (2x Point-upscaled luma), `sclip` (= clip) and `mclip` (2x upscaled vsaa
    edge mask). vsfeel and eedi3vk2 take both sclip and mclip (based_aa passes
    mclip when the backend supports it); vszipcl/vszipcu support sclip only.
    --eedi3-mclip 0 drops the mclip from the vsfeel/eedi3vk2 calls.

    """
    use_mclip = getattr(ns, "eedi3_mclip", True)
    common = (
        f"field={ns.eedi3_field}, mdis={ns.eedi3_mdis}, nrad={ns.eedi3_nrad}, "
        f"alpha={ns.eedi3_alpha}, beta={ns.eedi3_beta}, gamma={ns.eedi3_gamma}, "
        f"vcheck={ns.eedi3_vcheck}, vthresh0={ns.eedi3_vthresh0}, "
        f"vthresh1={ns.eedi3_vthresh1}, vthresh2={ns.eedi3_vthresh2}"
    )
    if getattr(ns, "synthetic", False):
        return {
            "vsfeel": f"core.vsfeel.EEDI3({clip}, {common})",
            "eedi3vk2": f"core.eedi3vk2.EEDI3({clip}, {common})",
            "vszipcl": f"core.vszipcl.EEDI3({clip}, {common})"
        }
    if use_mclip:
        mcap = "sclip=sclip, mclip=mclip"
    else:
        mcap = "sclip=sclip"
    with_mclip = f"{common}, {mcap}"
    with_sclip = f"{common}, sclip=sclip"
    return {
        "vsfeel": f"core.vsfeel.EEDI3({clip}, {with_mclip})",
        "eedi3vk2": f"core.eedi3vk2.EEDI3({clip}, {with_mclip})",
        "vszipcl": f"core.vszipcl.EEDI3({clip}, {with_sclip})"
    }


def _eedi3h_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    """The horizontal EEDI3: the same call surface as _eedi3_build, one name up.

    eedi3vk2 does not register an EEDI3H, so the references are vszipcl and
    vszipcu (which do).
    """
    use_mclip = getattr(ns, "eedi3_mclip", True)
    common = (
        f"field={ns.eedi3_field}, mdis={ns.eedi3_mdis}, nrad={ns.eedi3_nrad}, "
        f"alpha={ns.eedi3_alpha}, beta={ns.eedi3_beta}, gamma={ns.eedi3_gamma}, "
        f"vcheck={ns.eedi3_vcheck}, vthresh0={ns.eedi3_vthresh0}, "
        f"vthresh1={ns.eedi3_vthresh1}, vthresh2={ns.eedi3_vthresh2}"
    )
    with_mclip = common + (", sclip=sclip, mclip=mclip" if use_mclip else ", sclip=sclip")
    with_sclip = f"{common}, sclip=sclip"
    return {
        "vsfeel": f"core.vsfeel.EEDI3H({clip}, {with_mclip})",
        "vszipcl": f"core.vszipcl.EEDI3H({clip}, {with_sclip})",
        "vszipcu": f"core.vszipcu.EEDI3H({clip}, {with_sclip})",
    }


def _vsaa_eedi3_backend(plugin: str) -> str | None:
    """Name of the ``vsaa`` ``EEDI3.Backend`` member that drives ``plugin``.

    ``based_aa`` reaches a reference plugin through this enum, over
    ``EEDI3.Backend.EEDI3`` / ``EEDI3H``. Deriving the member from the plugin
    name keeps the benchmark honest about which plugin actually ships EEDI3H:
    ``EEDI3.Backend.should_h`` is True only for vszipcl/vszipcu, while
    eedi3vk2 does *not* register EEDI3H and takes the transpose -> EEDI3 ->
    transpose fallback.
    """
    try:
        from vsaa import EEDI3 as _VsaaEEDI3
    except ImportError:
        return None
    return next((m.name for m in _VsaaEEDI3.Backend if m.value == plugin), None)


def _eedi3aa_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    """The fused based_aa EEDI3 chain: vsfeel.EEDI3AA vs based_aa's own chain.

    ``vsfeel`` runs the fused single call. Every reference plugin runs the
    *actual* ``vsaa`` antialiaser ``based_aa`` drives, so the chain cannot
    drift from the wrapper: vsaa owns ``should_h`` (native EEDI3H only where
    the plugin ships it — vszipcl and vszipcu; eedi3vk2 falls back to
    transpose -> EEDI3 -> transpose), ``supports_mclip`` (forwarded to both
    the native and the fallback path) and the double-rate
    ``Interleave([s, s])`` sclip, and derives the plugin's ``field`` from
    ``(tff, double_rate)``. ``--eedi3-field`` is mapped back onto that pair
    (based_aa's field is ``tff + double_rate*2``).

    ``clip`` is the 2x supersampled luma (based_aa's ``ss``), ``mclip`` the
    2x mask and ``sclip`` the already-interleaved 2N clip: the fused arm takes
    it directly (``core.vsfeel.EEDI3AA`` consumes the 2N clip, exactly as
    ``vsfeel.vsaa.EEDI3`` hands it over) while the reference arms pass the
    single-rate ``clip`` as sclip, which is what based_aa does (its
    antialiaser interleaves it itself).
    """
    use_mclip = getattr(ns, "eedi3_mclip", True)
    field = ns.eedi3_field
    if field <= 1:
        # EEDI3AA is the fused *double-rate* chain (four sub-passes + two
        # merges); a single-rate field has no fused form to grade.
        raise SystemExit(
            "the eedi3aa benchmark grades the double-rate chain only "
            f"(field 2 or 3), got --eedi3-field {field}")
    # vsfeel's plugin takes the three thresholds separately; the vsaa
    # antialiaser carries based_aa's `vthresh=(v0, v1, v2)` object field and
    # expands it itself.
    common_plugin = (
        f"alpha={ns.eedi3_alpha}, beta={ns.eedi3_beta}, gamma={ns.eedi3_gamma}, "
        f"nrad={ns.eedi3_nrad}, mdis={ns.eedi3_mdis}, vcheck={ns.eedi3_vcheck}, "
        f"vthresh0={ns.eedi3_vthresh0}, vthresh1={ns.eedi3_vthresh1}, "
        f"vthresh2={ns.eedi3_vthresh2}"
    )
    common_vsaa = (
        f"alpha={ns.eedi3_alpha}, beta={ns.eedi3_beta}, gamma={ns.eedi3_gamma}, "
        f"nrad={ns.eedi3_nrad}, mdis={ns.eedi3_mdis}, vcheck={ns.eedi3_vcheck}, "
        f"vthresh=({ns.eedi3_vthresh0}, {ns.eedi3_vthresh1}, {ns.eedi3_vthresh2})"
    )
    # based_aa's default antialiaser runs double-rate, so field = tff + 2.
    tff = field - 2

    if getattr(ns, "synthetic", False):
        # The synthetic vpy defines only `clip` (no ss/mask): mirror
        # based_aa's sclip=ss with the clip itself and run maskless.
        out = {
            "vsfeel": (
                f"core.vsfeel.EEDI3AA({clip}, field={field}, {common_plugin}, "
                f"sclip=core.std.Interleave([{clip}, {clip}])"
            ),
        }
        for p in ("eedi3vk2", "vszipcl", "vszipcu"):
            member = _vsaa_eedi3_backend(p)
            if member is None:
                continue
            out[p] = (
                "from vsaa import EEDI3 as _VsaaEEDI3\n"
                f"_VsaaEEDI3(backend=_VsaaEEDI3.Backend.{member}, {common_vsaa})"
                f".antialias({clip}, tff={tff}, sclip={clip})"
            )
        return out

    aux = "sclip=sclip, mclip=mclip" if use_mclip else "sclip=sclip"
    aa_aux = "sclip=clip" + (", mclip=mclip" if use_mclip else "")
    out: dict[str, str] = {
        "vsfeel": (
            f"core.vsfeel.EEDI3AA({clip}, field={field}, {common_plugin}, {aux})"
        ),
    }
    for p in ("eedi3vk2", "vszipcl", "vszipcu"):
        member = _vsaa_eedi3_backend(p)
        if member is None:
            continue
        out[p] = (
            "from vsaa import EEDI3 as _VsaaEEDI3\n"
            f"_VsaaEEDI3(backend=_VsaaEEDI3.Backend.{member}, {common_vsaa})"
            f".antialias({clip}, tff={tff}, {aa_aux})"
        )
    return out


def _nnedi3_build(ns: argparse.Namespace, clip: str, spec: FilterSpec) -> dict[str, str]:
    common = (
        f"field={ns.nnedi3_field}, dh={ns.nnedi3_dh}, "
        f"nsize={ns.nnedi3_nsize}, nns={ns.nnedi3_nns}, qual={ns.nnedi3_qual}, "
        f"etype={ns.nnedi3_etype}, pscrn={ns.nnedi3_pscrn}"
    )
    if ns.nnedi3_planes:
        common += f", planes=[{ns.nnedi3_planes}]"
    return {
        "vsfeel": f"core.vsfeel.NNEDI3({clip}, {common})",
        "nnedi3vk": f"core.nnedi3vk.NNEDI3({clip}, {common})",
        "vszipcu": f"core.vszipcu.NNEDI3({clip}, {common})",
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
        # vsfeel's BM3Dv2 runs on the R80 GPU API (vnode:gpu in/out); so does
        # bm3dvk's. vszipcl stays on the CPU cache.
        gpu_plugins=frozenset({"vsfeel", "bm3dvk"}),
    ),
    "bilateral": FilterSpec(
        title="Bilateral",
        default_frames=5000,
        args=[
            Arg("sigma_spatial", "--bilateral-sigma-spatial", "bilateral_sigma_spatial", float, 3.0),
            Arg("sigma_color", "--bilateral-sigma-color", "bilateral_sigma_color", float, 0.02),
        ],
        build=_bilateral_build,
        # vsfeel's Bilateral runs on the R80 GPU API (vnode:gpu in/out); the
        # references stay on the CPU cache.
        gpu_plugins=frozenset({"vsfeel"}),
    ),
    "gaussblur": FilterSpec(
        title="GaussBlur",
        default_frames=5000,
        args=[
            Arg("sigma", "--gauss-sigma", "gauss_sigma", float, 16.0),
        ],
        build=_gauss_build,
        # vsfeel's GaussBlur is vnode:gpu under the R80 GPU API; the
        # references stay on the CPU cache.
        gpu_plugins=frozenset({"vsfeel"}),
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
        # vsfeel's DFTTest runs on the R80 GPU API (vnode:gpu in/out); the
        # references are CPU filters.
        gpu_plugins=frozenset({"vsfeel"}),
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
        # vsfeel's NLMeans is vnode:gpu under the R80 GPU API; knlmvk always was.
        gpu_plugins=frozenset({"vsfeel", "knlmvk"}),
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
        # vsfeel's EEDI3 is vnode:gpu; the reference arms are not.
        gpu_plugins=frozenset({"vsfeel"}),
    ),
    "eedi3h": FilterSpec(
        title="EEDI3H (horizontal)",
        default_frames=2000,
        args=[
            # Same surface as the eedi3 entry; EEDI3H interpolates columns.
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
                "pass the vsaa edge mask as mclip to vsfeel (default: true)"),
        ],
        build=_eedi3h_build,
        input="depth(get_y(clip), 16)",
        aa=True,
        # vsfeel's EEDI3H is vnode:gpu; the reference arms are not.
        gpu_plugins=frozenset({"vsfeel"}),
    ),
    "eedi3aa": FilterSpec(
        title="EEDI3AA (based_aa)",
        default_frames=1000,
        args=[
            # Same surface as the eedi3 entry: based_aa's default antialiaser
            # in double-rate mode (field = tff + double_rate*2 = 3).
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
                "pass the vsaa edge mask as mclip (default: true)"),
        ],
        build=_eedi3aa_build,
        # Grades the whole based_aa EEDI3 chain: vsfeel's fused call against
        # the chain vsaa.based_aa itself runs on each reference plugin
        # (see _eedi3aa_build — the reference arms ARE the vsaa antialiaser).
        input="depth(get_y(clip), 16)",
        aa=True,
        # vsfeel's EEDI3AA is vnode:gpu; the reference arms are not.
        gpu_plugins=frozenset({"vsfeel"}),
    ),
    "nnedi3": FilterSpec(
        title="NNEDI3",
        default_frames=5000,
        args=[
            Arg("field", "--nnedi3-field", "nnedi3_field", int, 3),
            Arg("dh", "--nnedi3-dh", "nnedi3_dh", int, 0),
            Arg("planes", "--nnedi3-planes", "nnedi3_planes", str, None),
            Arg("nsize", "--nnedi3-nsize", "nnedi3_nsize", int, 0),
            Arg("nns", "--nnedi3-nns", "nnedi3_nns", int, 4),
            Arg("qual", "--nnedi3-qual", "nnedi3_qual", int, 2),
            Arg("etype", "--nnedi3-etype", "nnedi3_etype", int, 0),
            Arg("pscrn", "--nnedi3-pscrn", "nnedi3_pscrn", int, 4),
        ],
        build=_nnedi3_build,
        # vsfeel's NNEDI3 is vnode:gpu under the R80 GPU API; the references
        # stay on the CPU cache.
        gpu_plugins=frozenset({"vsfeel"}),
    ),
}


# ---------------------------------------------------------------------------
# vspipe timing
# ---------------------------------------------------------------------------

_FPS_RE = re.compile(
    r"Output\s+(\d+)\s+frames?\s+in\s+[\d.]+\s+seconds?\s+\(([\d.]+)\s*fps\)")


def _tail(text: str | bytes | None, lines: int = 15) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def vspipe_env() -> dict[str, str]:
    """MANGOHUD off (it is only a display overlay), and RADV's transfer-only SDMA
    queue opted into: without it every GPU filter's download is a graphics-engine
    copy that competes with the kernel (+10-31% with it; `notes/BILATERAL.md`).
    Any other experimental flags the caller set are preserved.
    """
    env = {**os.environ, "MANGOHUD": "0"}
    flags = [f for f in env.get("RADV_EXPERIMENTAL", "").split(",") if f]
    if "transfer_queue" not in flags:
        flags.append("transfer_queue")
    env["RADV_EXPERIMENTAL"] = ",".join(flags)
    return env


def run_vspipe(vpy_path: Path, frames: int, timeout: float = DEFAULT_TIMEOUT) -> float | None:
    """Time one vspipe run; return its fps, or None if the run failed.

    A run is failed (never silently accepted) when vspipe exits non-zero, is
    killed by ``--timeout``, prints no timing line, or reports a frame count
    other than the one requested — timing a different amount of work than the
    header advertises is not a measurement.
    """
    cmd = ["vspipe", "--start", "0", "--end", str(frames - 1), str(vpy_path), "/dev/null"]
    env = vspipe_env()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        print(f"  [vspipe timed out after {timeout:g}s] {vpy_path.name}", file=sys.stderr)
        tail = _tail(exc.stderr)
        if tail:
            print(textwrap.indent(tail, "    "), file=sys.stderr)
        return None
    if result.returncode != 0:
        # A failed run must not read as an unavailable plugin: surface the tail
        # of vspipe's stderr so the cause (bad argument, crash, missing plugin)
        # is visible instead of being silently swallowed.
        print(f"  [vspipe failed: exit {result.returncode}] {vpy_path.name}", file=sys.stderr)
        tail = _tail(result.stderr)
        if tail:
            print(textwrap.indent(tail, "    "), file=sys.stderr)
        return None
    match = _FPS_RE.search(result.stderr or "")
    if match is None:
        # exited 0 but printed no timing line: report that too
        print(f"  [vspipe produced no fps line: {vpy_path.name}]", file=sys.stderr)
        tail = _tail(result.stderr, 10)
        if tail:
            print(textwrap.indent(tail, "    "), file=sys.stderr)
        return None
    got, fps = int(match.group(1)), float(match.group(2))
    if got != frames:
        print(f"  [vspipe reported {got} frames, expected {frames}] {vpy_path.name}",
              file=sys.stderr)
        return None
    return fps


def bench(plugin: str, chain: str, clip: str, frames: int, synth_format: str | None,
          cache_frames: int | None = None, cache_conv: str | None = None,
          timeout: float = DEFAULT_TIMEOUT, gpu_cache: bool = False,
          gpu_cache_mb: int | None = None) -> float | None:
    vpy = make_vpy(
        clip=clip,
        extra=_plugin_loader(plugin),
        chain=chain,
        frames=frames,
        synth_format=synth_format,
        cache_frames=cache_frames,
        cache_conv=cache_conv,
        gpu_cache=gpu_cache,
        gpu_cache_mb=gpu_cache_mb,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"bench_{plugin}.vpy"
        path.write_text(vpy)
        return run_vspipe(path, frames, timeout)


def bench_aa(plugin: str, chain: str, clip: str, frames: int,
             cache_frames: int | None = None,
             eedi3_field: int = 3,
             bits: int = AA_MASK_BITS,
             cache_bytes: int | None = None,
             timeout: float = DEFAULT_TIMEOUT,
             gpu_cache: bool = False,
             download_inputs: bool = False) -> float | None:
    """Run an EEDI3 anti-aliasing style benchmark (see make_aa_vpy)."""
    vpy = make_aa_vpy(
        clip=clip,
        extra=_plugin_loader(plugin),
        chain=chain,
        frames=frames,
        cache_frames=cache_frames,
        eedi3_field=eedi3_field,
        bits=bits,
        cache_bytes=cache_bytes,
        gpu_cache=gpu_cache,
        download_inputs=download_inputs,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"bench_{plugin}.vpy"
        path.write_text(vpy)
        return run_vspipe(path, frames, timeout)


def _stats(values: list[float]) -> tuple[float, float, float]:
    """(median, min, max) of the successful repeats."""
    return statistics.median(values), min(values), max(values)


def _fmt_stats(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:9.2f} fps  [n=1 (--repeat 1 has no spread)]"
    median, lo, hi = _stats(values)
    spread = 100.0 * (hi - lo) / median if median else 0.0
    return (f"{median:9.2f} fps  [min {lo:8.2f} max {hi:8.2f} "
            f"spread {spread:4.1f}%] n={len(values)}")


def args_desc(spec: FilterSpec, ns: argparse.Namespace) -> str:
    pairs = [f"{a.key}={getattr(ns, a.dest)}" for a in spec.args]
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


def _cache_desc(spec: FilterSpec, ns: argparse.Namespace, synth: str | None,
                cache_frames: int | None) -> str:
    if synth is not None:
        return "cache: N/A (BlankClip)"
    if cache_frames is None:
        return "cache: N/A"
    if spec.aa:
        return (f"cache: 2x luma+mclip"
                f"{ns.aa_cache_mb} MiB byte budget")
    return f"cache: first {cache_frames} frames"


def _run_once(spec: FilterSpec, ns: argparse.Namespace, plugin: str, chain: str,
              frames: int, synth: str | None, cache_frames: int | None,
              cache_conv: str | None, gpu_cache: bool = False,
              download_inputs: bool = False) -> float | None:
    """One timed vspipe run of one plugin: the repeat/interleave unit.

    `gpu_cache` says this arm's chain consumes device-resident frames, so the
    script has to build the GPU cache. `download_inputs` says the arm's filter
    does not accept them and so has to pay std.GPUDownload for them, which is
    what a real chain would make it pay.
    """
    if spec.aa and synth is None:
        budget = ns.aa_cache_mb * 1024 * 1024 if ns.aa_cache_mb else None
        return bench_aa(plugin, chain, ns.clip, frames, cache_frames,
                        getattr(ns, "eedi3_field", 3), ns.bits or AA_MASK_BITS,
                        budget, ns.timeout, gpu_cache=gpu_cache,
                        download_inputs=download_inputs)
    return bench(plugin, chain, ns.clip, frames, synth, cache_frames, cache_conv,
                 ns.timeout, gpu_cache=gpu_cache, gpu_cache_mb=ns.gpu_cache_mb)


def _resolve_pair(ns: argparse.Namespace, calls: dict[str, str],
                  plugins: list[str], title: str) -> list[str]:
    """``--pair``: keep only the vsfeel arm and one reference."""
    if ns.pair is None:
        return plugins
    if "vsfeel" not in calls:
        sys.exit(f"--pair: {title} has no vsfeel arm")
    if ns.pair == "auto":
        ref = next((p for p in plugins if p != "vsfeel"), None)
        if ref is None:
            sys.exit(f"--pair: no reference plugin available for {title}")
    else:
        ref = ns.pair
    if ref not in calls:
        sys.exit(f"--pair: {ref!r} does not provide {title}")
    return ["vsfeel", ref]


def bench_filter(spec: FilterSpec, ns: argparse.Namespace) -> None:
    # --bits overrides the filter's input expression (depth(X,16) -> depth(X,32))
    # for BOTH the chain and the cached frames: the chain must consume the same
    # format the cache holds, or the timed region re-converts behind the filter
    input_expr = _input_for_bits(spec.input, ns.bits) if ns.bits else spec.input
    frames = ns.frames or spec.default_frames
    synth = spec.synth_format if ns.synthetic else None
    if synth and ns.bits:
        synth = _format_for_bits(synth, ns.bits)
    # --cache-frames 0 and --no-cache both mean "no preload" — including for the
    # AA filters, which used to fall back to a hardcoded 400-frame preload.
    cache_frames = ns.cache_frames if (ns.cached and synth is None and ns.cache_frames) else None
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

    calls = spec.build(ns, input_expr, spec)
    # --gpu-cache: every arm consumes the device-resident frames, so the
    # run matches a chain whose upstream node is a GPU filter. Arms whose
    # filter declares vnode:gpu take them directly; a legacy CPU filter gets
    # them through std.GPUDownload and pays that transfer, which is what the
    # real chain would charge it. (An arm that would have been handed a CPU
    # cache instead is not a fair comparison -- its neighbour's input is in
    # VRAM and it would never see that.)
    #
    # The AA vpy is rebuilt per arm rather than substituting names: with
    # download_inputs it defines clip/mclip/sclip as downloads of the warmed
    # device frames, so the chain string stays as it is.
    gpu_arms: list[str] = []
    download_arms: list[str] = []
    if getattr(ns, "gpu_cache", False):
        gpu_arms = list(calls)
        download_arms = [p for p in calls if p not in spec.gpu_plugins]
        if not spec.aa:
            gpu_calls = spec.build(ns, "clip_gpu", spec)
            dl_calls = spec.build(
                ns, "core.std.GPUDownload(clip=clip_gpu)", spec)
            for p in calls:
                calls[p] = gpu_calls[p] if p in spec.gpu_plugins else dl_calls[p]
    plugins = resolve_plugins(ns.plugins or list(calls), calls, spec.title)
    plugins = _resolve_pair(ns, calls, plugins, spec.title)
    if not plugins:
        sys.exit(f"no valid plugins requested for --filter {ns.filter}")
    print(f"{spec.title} benchmark | {frames} frames | clip: {clip_desc}{bits_desc}")
    print(f"args: {args_desc(spec, ns)}")
    gpu_desc = ""
    if getattr(ns, "gpu_cache", False):
        if not gpu_arms:
            gpu_desc = " | gpu cache: no GPU-input arm in this filter"
        elif cache_frames:
            gpu_desc = (f" | gpu cache: {ns.gpu_cache_mb} MiB; "
                        f"download arms: {', '.join(download_arms) or 'none'}")
        else:
            # No preload to mirror (BlankClip or --no-cache): the arms get a
            # GPUUpload-fed clip, which is what the graph would insert anyway.
            gpu_desc = (f" | gpu cache: live upload; download arms: "
                        f"{', '.join(download_arms) or 'none'}")
    print(f"{_cache_desc(spec, ns, synth, cache_frames)}{gpu_desc} | "
          f"repeat: {ns.repeat} | timeout: {ns.timeout:g}s\n")

    runs: dict[str, list[float]] = {p: [] for p in plugins}
    for r in range(ns.repeat):
        # Alternate the plugin order between repeats so clock/thermal drift
        # is shared between the arms instead of favouring the first one.
        reverse = ns.interleave and r % 2 == 1
        for plugin in (list(reversed(plugins)) if reverse else plugins):
            fps = _run_once(spec, ns, plugin, calls[plugin], frames, synth,
                            cache_frames, cache_conv,
                            gpu_cache=plugin in gpu_arms,
                            download_inputs=plugin in download_arms)
            if fps is not None:
                runs[plugin].append(fps)
    for plugin in plugins:
        if runs[plugin]:
            print(f"  {plugin:10s}  {_fmt_stats(runs[plugin])}")
        else:
            print(f"  {plugin:10s}  unavailable / failed")
    print()

    if ns.pair and len(plugins) == 2 and runs[plugins[0]] and runs[plugins[1]]:
        a, b = plugins
        ma, mb = statistics.median(runs[a]), statistics.median(runs[b])
        print(f"  pair: {a} {ma:9.2f} fps vs {b} {mb:9.2f} fps -> {ma / mb:.3f}x\n")

    valid = sorted(
        ((p, statistics.median(v)) for p, v in runs.items() if v),
        key=lambda x: x[1], reverse=True)
    if len(valid) > 1:
        for rank, (plugin, fps) in enumerate(valid, 1):
            print(f"  {rank}. {plugin:10s} {fps:9.2f} fps")
        print()


# ---------------------------------------------------------------------------
# Freshness: never benchmark a stale plugin binary
# ---------------------------------------------------------------------------

def _default_plugin_so() -> Path | None:
    """Installed libvsfeel.so path, discovered without importing the core."""
    spec = importlib.util.find_spec("vapoursynth")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg = Path(next(iter(spec.submodule_search_locations)))
    return pkg / "plugins" / "vsfeel" / "libvsfeel.so"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_fresh(install_so: Path | None, build_so: Path | None) -> None:
    """Assert the installed .so is the current build, newer than every source.

    Encodes rule 01: a source edit after the build, or a build that was never
    copied into the plugin directory, must stop the run rather than silently
    benchmark the previous binary.
    """
    if install_so is None or not install_so.exists():
        sys.exit(f"--check-fresh: installed plugin not found: {install_so}")
    build = build_so or (REPO_ROOT / "build" / install_so.name)
    if not build.exists():
        sys.exit(f"--check-fresh: built plugin not found: {build} (build first)")
    if _sha256(build) != _sha256(install_so):
        sys.exit(f"--check-fresh: {install_so} differs from {build}; "
                 "copy the built .so into the plugin directory")
    stale = [p for p in sorted(REPO_ROOT.glob("src/*"))
             if p.suffix in (".cpp", ".h", ".comp")
             and p.stat().st_mtime > build.stat().st_mtime]
    if (REPO_ROOT / "CMakeLists.txt").stat().st_mtime > build.stat().st_mtime:
        stale.append(REPO_ROOT / "CMakeLists.txt")
    if stale:
        names = ", ".join(str(p.relative_to(REPO_ROOT)) for p in stale[:5])
        sys.exit(f"--check-fresh: sources newer than {build}: {names} (rebuild)")
    print(f"freshness: {install_so} matches {build}, newer than all sources")


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
    parser.add_argument("--gpu-cache", action="store_true",
                        help="preload the cached frames on the device and hand them to every "
                             "arm, so the run matches a chain whose upstream node is a GPU "
                             "filter. Filters taking vnode:gpu input consume them directly; "
                             "every other plugin gets them through std.GPUDownload and pays "
                             "that transfer, as it would mid-chain")
    parser.add_argument("--gpu-cache-mb", type=int, default=6144,
                        help="VRAM budget for --gpu-cache frames; the cached span is capped "
                             "to what fits, for both caches (default: 6144)")
    parser.add_argument("--repeat", "--repeats", dest="repeat", type=int, default=3,
                        help="timed runs per plugin; the median is reported with "
                             "min/max/spread and the plugin order alternates "
                             "(default: 3)")
    parser.add_argument("--no-interleave", dest="interleave", action="store_false",
                        help="do not alternate the plugin order between repeats "
                             "(interleaving is on by default)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"kill a vspipe run after this many seconds "
                             f"(default: {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--pair", nargs="?", const="auto", default=None, metavar="PLUGIN",
                        help="same-session pairing: run only vsfeel and one "
                             "reference (default: the first requested reference) "
                             "back-to-back and print the fps ratio")
    parser.add_argument("--aa-cache-mb", type=int, default=AA_CACHE_MB,
                        help="byte budget for the EEDI3 AA Python cache, in MiB "
                             f"(default: {AA_CACHE_MB}); this budget is in addition "
                             "to VapourSynth's own frame cache")
    parser.add_argument("--check-fresh", action="store_true",
                        help="refuse to run unless the installed .so matches the "
                             "build and is newer than every source")
    parser.add_argument("--vsfeel-so", type=Path, default=None,
                        help="path to the installed libvsfeel.so (default: discovered)")
    parser.add_argument("--build-so", type=Path, default=None,
                        help="path to build/libvsfeel.so (default: build/<installed name>)")

    # Filters may share CLI flags (eedi3 and eedi3aa expose the same EEDI3
    # surface); register each flag once.
    seen_flags: set[str] = set()
    for spec in FILTERS.values():
        for arg in spec.args:
            if arg.flag in seen_flags:
                continue
            seen_flags.add(arg.flag)
            parser.add_argument(arg.flag, dest=arg.dest, type=arg.type,
                                default=arg.default, help=arg.help)

    return parser.parse_args()


def main() -> None:
    ns = parse_args()
    ns.clip = str(Path(ns.clip).expanduser().resolve())
    if ns.repeat < 1:
        sys.exit(f"--repeat must be >= 1, got {ns.repeat}")
    if ns.aa_cache_mb < 0:
        sys.exit(f"--aa-cache-mb must be >= 0, got {ns.aa_cache_mb}")
    if ns.check_fresh:
        check_fresh(ns.vsfeel_so or _default_plugin_so(), ns.build_so)
    filters = list(FILTERS) if ns.filter == "all" else [ns.filter]
    for fname in filters:
        bench_filter(FILTERS[fname], ns)


if __name__ == "__main__":
    main()