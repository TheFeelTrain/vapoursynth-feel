"""Build plumbing: plugin version reporting, the SPIR-V header generator, and
the shipped build tools (the shader-limits checker, benchmark script
generation, wheel staging)."""

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
SHADER_LIMITS = ROOT / "tools" / "shader_limits.py"
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


def test_shader_limits_stem_survives_windows_globs():
    """A glob result must reduce to a variant stem on either platform.

    glob echoes the pattern's separators and joins the matched tail with the
    platform's own, so a forward-slash pattern on Windows yields
    ``build/vk_spv\\eedi3_32_row.spv``. Splitting on ``/`` alone then leaves the
    directory in the stem, which matches no host declaration: every module is
    reported unchecked and ``--check`` passes without comparing anything.
    """
    limits = _load_module("shader_limits", SHADER_LIMITS)
    for path in (
        "build/vk_spv/eedi3_32_row.spv",
        "build/vk_spv\\eedi3_32_row.spv",
        "build\\vk_spv\\eedi3_32_row.spv",
        "eedi3_32_row.spv",
    ):
        assert limits.path_basename(path) == "eedi3_32_row.spv", path
        assert limits.variant_stem(path) == "eedi3_32_row", path
    # A non-.spv match keeps its whole name (the pattern is user-supplied).
    assert limits.variant_stem("build\\vk_spv\\notes.txt") == "notes.txt"


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
        bench._apply_filter_args(ns, name, spec)
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
        bench._apply_filter_args(ns, name, spec)
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


def test_benchmark_ranking_annotates_vsfeel_ratio(monkeypatch, capsys):
    """vsfeel's ranking line carries its lead over the fastest reference."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--filter",
            "gaussblur",
            "--synthetic",
            "--frames",
            "10",
            "--repeat",
            "1",
            "vsfeel",
            "vszipcl",
        ],
    )
    ns = bench.parse_args()
    fps = iter([200.0, 100.0])
    monkeypatch.setattr(bench, "run_vspipe", lambda *a, **k: next(fps))
    bench.bench_filter("gaussblur", ns)
    out = capsys.readouterr().out
    assert "1. vsfeel" in out
    assert "(2.000x vs vszipcl)" in out
    # The reference line stays bare: the ratio is vsfeel's, not a full matrix.
    assert "(0.500x" not in out


def test_benchmark_ranking_without_vsfeel_falls_back(monkeypatch, capsys):
    """Without vsfeel there is nothing to grade: ratios go against the fastest."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--filter",
            "gaussblur",
            "--synthetic",
            "--frames",
            "10",
            "--repeat",
            "1",
            "vszipcl",
            "vszipcu",
        ],
    )
    ns = bench.parse_args()
    fps = iter([200.0, 100.0])
    monkeypatch.setattr(bench, "run_vspipe", lambda *a, **k: next(fps))
    bench.bench_filter("gaussblur", ns)
    out = capsys.readouterr().out
    assert "1. vszipcl" in out
    assert "(0.500x vs vszipcl)" in out
    assert "(1.000x" not in out


def test_benchmark_args_string_overrides_defaults(monkeypatch):
    """A --<filter>-args string lands on the namespace; the rest keep defaults."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(
        sys, "argv", ["benchmark.py", "--bm3d-args", "sigma=0.7, radius=2, th_mse=None"]
    )
    ns = bench.parse_args()
    bench._apply_filter_args(ns, "bm3d", bench.FILTERS["bm3d"])
    assert ns.bm3d_sigma == 0.7
    assert ns.bm3d_radius == 2
    assert ns.bm3d_th_mse is None
    # Untouched params keep their defaults.
    assert ns.bm3d_bm_range == 9
    assert ns.bm3d_ps_range == 4
    assert ns.bm3d_block_step == 8


def test_benchmark_args_string_converts_types(monkeypatch):
    """Ints, bools and the planes list parse; omission means defaults."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--eedi3-args",
            "field=3, mclip=0, hp=true",
            "--nnedi3-args",
            "planes=[0,1]",
        ],
    )
    ns = bench.parse_args()
    bench._apply_filter_args(ns, "eedi3", bench.FILTERS["eedi3"])
    assert ns.eedi3_field == 3
    assert ns.eedi3_mclip is False
    assert ns.eedi3_hp is True
    bench._apply_filter_args(ns, "nnedi3", bench.FILTERS["nnedi3"])
    assert ns.nnedi3_planes == "0,1"
    assert ns.nnedi3_field == 3  # default, not in the string
    # Shared dests never leak across filters: re-applying resets to defaults.
    bench._apply_filter_args(ns, "eedi3h", bench.FILTERS["eedi3h"])
    assert ns.eedi3_mclip is True


def test_benchmark_args_string_rejects_misuse(monkeypatch):
    """Unknown keys, missing '=' and bad values abort instead of benchmarking defaults."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    ns = bench.parse_args()
    spec = bench.FILTERS["bm3d"]
    for bad in ("sig ma=1", "sigma", "radius=two", "radius=", "sigma=1, sigma"):
        with pytest.raises(SystemExit):
            bench._parse_filter_args(ns, "bm3d", spec, bad)
    with pytest.raises(SystemExit):
        bench._parse_filter_args(ns, "bm3d", spec, "planes=[0,1")
    # The --<filter>-args dests default to None (main() reads them to reject a
    # string given for a filter that is not run).
    assert ns.bm3d_args is None


def test_ab_env_spec_sets_and_unsets():
    """Arm env specs distinguish set, unset and inherit (presence alone can gate)."""
    bench = _load_module("benchmark", BENCHMARK)
    assert bench._parse_ab_env("VSFEEL_X=1,VSFEEL_Y=0", "--ab-a-env") == [
        ("VSFEEL_X", "1"),
        ("VSFEEL_Y", "0"),
    ]
    assert bench._parse_ab_env("-VSFEEL_X", "--ab-b-env") == [("VSFEEL_X", None)]
    assert bench._parse_ab_env("", "--ab-a-env") == []
    assert bench._parse_ab_env("A=1,,B=2", "--ab-a-env") == [("A", "1"), ("B", "2")]
    # A value may itself contain '=' (split on the first one only).
    assert bench._parse_ab_env("A=b=c", "--ab-a-env") == [("A", "b=c")]
    for bad in ("VSFEEL_X", "1A=1", "-"):
        with pytest.raises(SystemExit):
            bench._parse_ab_env(bad, "--ab-a-env")


def test_ab_env_applies_and_restores(monkeypatch):
    """An arm's overrides are visible inside the cell and gone after it."""
    import os

    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setenv("VSFEEL_KEEP", "orig")
    monkeypatch.delenv("VSFEEL_NEW", raising=False)
    with bench._ab_env([("VSFEEL_KEEP", "arm"), ("VSFEEL_NEW", "1"), ("VSFEEL_ABSENT", None)]):
        assert os.environ["VSFEEL_KEEP"] == "arm"
        assert os.environ["VSFEEL_NEW"] == "1"
        assert "VSFEEL_ABSENT" not in os.environ
    assert os.environ["VSFEEL_KEEP"] == "orig"
    assert "VSFEEL_NEW" not in os.environ


def test_ab_round_order():
    """Alternate flips the first arm per round; ABBA is A B B A every round."""
    bench = _load_module("benchmark", BENCHMARK)
    assert bench._ab_round_order(1, "alternate") == [0, 1]
    assert bench._ab_round_order(2, "alternate") == [1, 0]
    assert bench._ab_round_order(3, "alternate") == [0, 1]
    assert bench._ab_round_order(1, "abba") == [0, 1, 1, 0]
    assert bench._ab_round_order(2, "abba") == [0, 1, 1, 0]


def test_ab_names(monkeypatch):
    """Arm names default to a/b and must be two distinct non-empty names."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--synthetic"])
    assert bench._ab_names(bench.parse_args()) == ("a", "b")
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-names", "new,old"])
    assert bench._ab_names(bench.parse_args()) == ("new", "old")
    for bad in ("only", "a,a", "a,", ",b", "a,b,c"):
        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-names", bad])
        with pytest.raises(SystemExit):
            bench._ab_names(bench.parse_args())


def test_ab_flags_activate_ab_mode(monkeypatch):
    """Any arm flag (--ab-so or either env) selects bench_filter_ab in main."""
    bench = _load_module("benchmark", BENCHMARK)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--synthetic"])
    assert not bench._ab_active(bench.parse_args())
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-so", "a.so", "b.so"])
    ns = bench.parse_args()
    assert bench._ab_active(ns)
    assert [str(p) for p in ns.ab_so] == ["a.so", "b.so"]
    assert bench._ab_rounds(ns) == 2
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-b-env", "VSFEEL_X=1"])
    assert bench._ab_active(bench.parse_args())
    # A bare modifier still selects A/B mode (two identical arms), so a
    # misspelled arm setup cannot silently run a normal benchmark instead.
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-names", "new,old"])
    assert bench._ab_active(bench.parse_args())
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-rounds", "3"])
    ns = bench.parse_args()
    assert bench._ab_active(ns)
    assert bench._ab_rounds(ns) == 3
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--ab-rounds", "0"])
    with pytest.raises(SystemExit):
        bench._ab_rounds(bench.parse_args())


def test_ab_install_copies_and_verifies(tmp_path):
    """The swap installs byte-identical copies (the sha guard ab_dfttest.sh had)."""
    import hashlib

    bench = _load_module("benchmark", BENCHMARK)
    src = tmp_path / "new.so"
    src.write_bytes(b"fake-binary-\x00" * 100)
    dst = tmp_path / "installed.so"
    dst.write_bytes(b"old")
    bench._install_ab_so(src, dst)
    assert dst.read_bytes() == src.read_bytes()
    assert bench._sha256(dst) == hashlib.sha256(src.read_bytes()).hexdigest()


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
