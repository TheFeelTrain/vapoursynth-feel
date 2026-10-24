#!/usr/bin/env python3
"""Derive the C++ analysers' view of the configured build tree.

build/spirv_binaries.h is tens of megabytes of generated uint32_t data that
every filter includes. Parsing it once per translation unit dominated
cppcheck (most of a minute per file, tens of GB of RAM for the run) and was a
large part of clang-tidy's per-file time. The data holds no logic and both
gates keep their findings in src/, so the analysers get a stub with the same
symbols plus a compile database that resolves the stub first.

Usage: lint_stub.py --build-dir DIR [--out-dir DIR]

Writes <out>/spirv_binaries.h and <out>/compile_commands.json; a no-op when
both are newer than what they derive from.
"""

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

_ARRAY_RE = re.compile(r"static const uint32_t (\w+)\[\] = \{.*?\n\};", re.S)
_SIZE_RE = re.compile(r"static const size_t (\w+)_spv_size = sizeof\(\1_spv\);")
_DECL = "static const uint32_t "


def fail(message: str) -> None:
    print(f"lint_stub.py: {message}", file=sys.stderr)
    sys.exit(1)


def stub_header(text: str) -> str:
    """Replace every SPIR-V array body with one element, keeping every symbol."""
    # The array symbol is `<stem>_spv`; the companion size symbol strips that.
    symbols = _ARRAY_RE.findall(text)
    names = [sym.removesuffix("_spv") for sym in symbols]
    if not names:
        fail("no SPIR-V arrays in build/spirv_binaries.h (regenerate it)")
    if text.count(_DECL) != len(names):
        fail("unexpected declaration in build/spirv_binaries.h; refusing to stub it")
    # Each array has a matching `sizeof` companion; a mismatch means the stub
    # would not define what the host references, so the analysis would fail.
    if set(_SIZE_RE.findall(text)) != set(names):
        fail("SPIR-V array and size declarations do not pair up")
    return _ARRAY_RE.sub(lambda m: f"static const uint32_t {m.group(1)}[] = {{0}};", text)


def stub_database(database: list[dict], include_dir: Path) -> list[dict]:
    """Point each entry's first include path at the stub header."""
    flag = f"-I{include_dir}"
    stubbed = []
    for entry in database:
        entry = dict(entry)
        if "arguments" in entry:
            argv = list(entry["arguments"])
            entry["arguments"] = [argv[0], flag, *argv[1:]]
        else:
            argv = shlex.split(entry["command"])
            argv.insert(1, flag)
            entry["command"] = shlex.join(argv)
        stubbed.append(entry)
    return stubbed


def write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def up_to_date(out: Path, inputs: list[Path]) -> bool:
    if not out.exists():
        return False
    newest = max(p.stat().st_mtime for p in inputs)
    return out.stat().st_mtime >= newest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--out-dir", help="default: <build-dir>/lint")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    header = build_dir / "spirv_binaries.h"
    database = build_dir / "compile_commands.json"
    for path in (header, database):
        if not path.is_file():
            fail(f"{path} is missing (run tools/install.sh first)")
    out_dir = (Path(args.out_dir) if args.out_dir else build_dir / "lint").resolve()
    stub = out_dir / "spirv_binaries.h"
    stub_db = out_dir / "compile_commands.json"

    script = Path(__file__).resolve()
    if up_to_date(stub, [header, script]) and up_to_date(stub_db, [database, script]):
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(stub, stub_header(header.read_text()))
    stubbed = stub_database(json.loads(database.read_text()), out_dir)
    write_atomic(stub_db, json.dumps(stubbed, indent=1) + "\n")


if __name__ == "__main__":
    main()
