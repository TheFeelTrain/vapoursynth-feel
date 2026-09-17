"""Build plumbing: plugin version reporting and the SPIR-V header generator."""

import re
import subprocess
import sys
from pathlib import Path

import vapoursynth as vs

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "src" / "gen_spirv_header.py"

# A minimal header-only SPIR-V module: magic, version, generator, bound,
# schema. The generator only inspects the header, so this exercises it fully.
SPV_HEADER = bytes.fromhex("0302230700010000" + "00000000" * 3)


def run_generator(tmp_path, data):
    spv = tmp_path / "shader.spv"
    spv.write_bytes(data)
    out = tmp_path / "shader.h"
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(out), str(spv)],
        capture_output=True, text=True,
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
