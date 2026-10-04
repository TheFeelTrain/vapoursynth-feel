"""Build plumbing: plugin version reporting, the SPIR-V header generator, and
the shipped build tools (benchmark script generation, wheel staging)."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import vapoursynth as vs

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "src" / "gen_spirv_header.py"
BENCHMARK = ROOT / "tools" / "benchmark.py"
HATCH_BUILD = ROOT / "hatch_build.py"

# A minimal header-only SPIR-V module: magic, version, generator, bound,
# schema. The generator only inspects the header, so this exercises it fully.
SPV_HEADER = bytes.fromhex("0302230700010000" + "00000000" * 3)


def run_generator(tmp_path, data):
    spv = tmp_path / "shader.spv"
    spv.write_bytes(data)
    out = tmp_path / "shader.h"
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(out), str(spv)],
        capture_output=True,
        text=True,
    )
    return proc, out


def test_plugin_reports_a_real_version():
    """The installed plugin exposes the version baked in at build time."""
    version = vs.core.vsfeel.Version()
    assert isinstance(version, str) and version
    # Guards the old failure mode, where the version silently degraded to a
    # bare git hash because no r* tag was reachable.
    assert re.match(r"^\d+\.\d+", version), version


def test_generator_emits_the_standard_includes(tmp_path):
    proc, out = run_generator(tmp_path, SPV_HEADER)
    assert proc.returncode == 0, proc.stderr
    header = out.read_text()
    assert "#include <cstddef>" in header
    assert "#include <cstdint>" in header
    assert "static const uint32_t shader_spv[]" in header


def test_generator_rejects_a_truncated_module(tmp_path):
    """A length that is not a multiple of 4 must be a hard error."""
    proc, out = run_generator(tmp_path, SPV_HEADER[:-2])
    assert proc.returncode != 0
    assert not out.exists()
    assert "multiple of 4" in proc.stderr


def test_generator_rejects_a_non_spirv_module(tmp_path):
    proc, out = run_generator(tmp_path, b"\x00" * 20)
    assert proc.returncode != 0
    assert not out.exists()
    assert "not SPIR-V" in proc.stderr


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_synthetic_scripts_compile_and_eval(monkeypatch):
    """Every ``--synthetic`` vpy the benchmark can emit must compile and run.

    The eedi3aa arm used to leave its outermost call unclosed and the eedi3h
    arm referenced undefined ``sclip``/``mclip``; both made the run die before
    any plugin call, which the harness reports only as "unavailable / failed".
    Compiling the generated vpy catches the SyntaxError, and evaluating each
    call against a stubbed ``core`` catches the undefined names.
    """
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--synthetic"])
    ns = bench.parse_args()
    assert ns.synthetic
    for name, spec in bench.FILTERS.items():
        calls = spec.build(ns, spec.input, spec)
        assert "vsfeel" in calls, name
        for plugin, chain in calls.items():
            vpy = bench.make_vpy(
                clip=spec.input,
                extra=bench._plugin_loader(plugin),
                chain=chain,
                frames=1,
                synth_format=spec.synth_format,
            )
            compile(vpy, f"<{name}/{plugin}>", "exec")
            if chain.startswith("from "):
                continue  # reference arms import vsaa, which may be absent
            # The names the generated vpy prelude defines; the chain only ever
            # reads the clip and the vstools helpers applied to it.
            env = {
                "core": MagicMock(),
                "clip": object(),
                "depth": MagicMock(),
                "get_y": MagicMock(),
            }
            exec(chain, env)


def test_benchmark_no_download_is_a_chain_suffix(monkeypatch):
    """``--no-download`` must stay a suffix of every chain it is applied to.

    The eedi3aa reference arms begin with a ``from vsaa import`` line, so a
    crop that enclosed the chain instead of following it would be a SyntaxError
    for exactly those arms.
    """
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--synthetic", "--no-download"])
    ns = bench.parse_args()
    assert ns.no_download
    assert "8x8" in bench._mode_desc(ns)
    for name, spec in bench.FILTERS.items():
        for plugin, chain in spec.build(ns, spec.input, spec).items():
            cropped = bench.no_download_chain(chain)
            assert cropped.endswith(
                f".std.CropAbs(width={bench.NO_DOWNLOAD_WINDOW}, height={bench.NO_DOWNLOAD_WINDOW})"
            ), name
            vpy = bench.make_vpy(
                clip=spec.input,
                extra=bench._plugin_loader(plugin),
                chain=cropped,
                frames=1,
                synth_format=spec.synth_format,
            )
            compile(vpy, f"<{name}/{plugin}>", "exec")
            if chain.startswith("from "):
                continue  # reference arms import vsaa, which may be absent
            env = {
                "core": MagicMock(),
                "clip": object(),
                "depth": MagicMock(),
                "get_y": MagicMock(),
            }
            exec(cropped, env)


def test_benchmark_default_mode_does_not_crop(monkeypatch):
    """The crop is opt-in: a default run's header must not claim the mode."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--synthetic"])
    ns = bench.parse_args()
    assert not ns.no_download
    assert bench._mode_desc(ns) == ""


def test_hatch_stages_exactly_this_platforms_library(tmp_path):
    """A stale foreign-platform library must fail the wheel build, not ship."""
    pytest.importorskip("hatchling")
    hb = _load_module("hatch_build", HATCH_BUILD)
    staged = tmp_path / "vapoursynth/plugins/vsfeel"
    staged.mkdir(parents=True)
    (staged / "libvsfeel.so").write_bytes(b"so")

    assert hb.find_staged_library(staged, "linux") == staged / "libvsfeel.so"
    assert hb.plugin_library_name("win32") == "vsfeel.dll"

    # A Windows DLL left behind by an earlier build is ignored on Linux, and a
    # Linux-only staging cannot satisfy a Windows build.
    (staged / "vsfeel.dll").write_bytes(b"dll")
    assert hb.find_staged_library(staged, "linux") == staged / "libvsfeel.so"
    dll_only = tmp_path / "dll_only"
    dll_only.mkdir()
    (dll_only / "vsfeel.dll").write_bytes(b"dll")
    with pytest.raises(RuntimeError):
        hb.find_staged_library(dll_only, "linux")
    assert hb.find_staged_library(dll_only, "win32") == dll_only / "vsfeel.dll"

    # The old "first .so sorted() finds" pick would silently stage this one.
    (staged / "libother.so").write_bytes(b"other")
    with pytest.raises(RuntimeError):
        hb.find_staged_library(staged, "linux")

    # Nothing staged at all is an error too.
    with pytest.raises(RuntimeError):
        hb.find_staged_library(tmp_path / "empty", "linux")
