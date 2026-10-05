#!/usr/bin/env bash
#
# Run the pytest suite in parallel. Use this instead of a bare `pytest tests`:
# the suite is mostly one-shot subprocesses (median ~1.4 s each, the bulk of
# the serial wall time), so it parallelises well, but the worker count must
# stay capped and distribution must stay file-level.
#
# The project venv is the environment under test (`uv sync` installs the suite's
# reference plugins into it, and tools/install.sh installs the plugin into its
# plugin directory). VSFEEL_PYTHON overrides the interpreter.
#
# More than 8 workers exhausts the GPU (`vkQueueSubmit failed`), and a plain
# `-n 8` is flaky because NLMeans a=64,d=16 hard-recovers the GPU whenever
# another client shares it; `--dist loadfile` keeps that config from
# overlapping another heavy file. See notes/NLMEANS.md.
#
# Usage: tools/test.sh [pytest args...]        default target: tests
#        tools/test.sh tests/test_nlmeans.py -q
#        VSFEEL_TEST_WORKERS=1 tools/test.sh   serial, no xdist
#
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)

py=${VSFEEL_PYTHON:-}
if [[ -z $py ]]; then
    if [[ -x $root_dir/.venv/bin/python ]]; then
        py=$root_dir/.venv/bin/python
    else
        py=python3
        printf 'test.sh: no .venv; falling back to %s (run `uv sync` first)\n' \
            "$(command -v python3)" >&2
    fi
fi

# The venv's bin first: tests shell out to `vspipe` (test_dfttest/test_eedi3),
# and a PATH lookup that found the system one would drive the system plugin
# directory instead of the library this checkout just built.
if [[ -d $root_dir/.venv/bin ]]; then
    PATH=$root_dir/.venv/bin:$PATH
    export PATH
fi

# Every filter's comparison subprocess loads the Vulkan implicit layers, and
# MangoHud's overlay is one of them when MANGOHUD is set in the shell. Under
# heavy parallel load its own teardown thread aborts ("free(): invalid
# pointer") in a process whose comparison already succeeded, which the harness
# reports as a filter failure. Results do not depend on the overlay (AGENTS.md
# asks for MANGOHUD=0 on every timed run), so the suite forces it off; run
# pytest directly to keep the shell's setting.
export MANGOHUD=0

# One command: cap at 8 workers, never trust `-n auto` (32 on the dev box).
workers=${VSFEEL_TEST_WORKERS:-$(nproc 2>/dev/null || echo 8)}
case $workers in
    ''|*[!0-9]*) workers=8 ;;
esac
if [ "$workers" -gt 8 ]; then
    workers=8
fi

if [ "$#" -eq 0 ]; then
    set -- tests
fi

# A flags-only invocation (`tools/test.sh -q`) must still target `tests`: with
# no path at all pytest collects from the rootdir, sweeps in the reference/
# trees, and their test modules share basenames with ours, which makes pytest's
# rootdir-based module naming fail every tests/test_*.py with "import file
# mismatch". Any non-flag argument naming an existing path is an explicit
# target; anything else (`-k eedi3`, `loadfile`) is not. A pytest node id
# (`tests/test_x.py::test_y`) is not a path either, so only the part before the
# first `::` is tested: treating the whole id as missing appends `tests` and
# silently expands an isolated case into the whole suite.
target_given=
for arg in "$@"; do
    case $arg in
        -*) continue ;;
    esac
    if [ -e "${arg%%::*}" ]; then
        target_given=1
    fi
done
if [ -z "$target_given" ]; then
    set -- "$@" tests
fi

cd -- "$root_dir"

# reference/ is read-only porting material, never part of this suite: it holds
# the references' own tests plus their dependency trees (lvsfunc, vs-jetpack),
# which are not ours to run and whose collection is what breaks ours.
ignore_reference=(--ignore=reference)

if [ "$workers" -le 1 ]; then
    exec "$py" -m pytest -q "${ignore_reference[@]}" "$@"
fi

if ! "$py" -c 'import xdist' >/dev/null 2>&1; then
    printf 'test.sh: pytest-xdist is not installed (dev dependency group); running serially\n' >&2
    exec "$py" -m pytest -q "${ignore_reference[@]}" "$@"
fi

exec "$py" -m pytest -q -n "$workers" --dist loadfile \
    "${ignore_reference[@]}" "$@"
