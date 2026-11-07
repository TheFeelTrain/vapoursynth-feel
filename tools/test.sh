#!/usr/bin/env bash
#
# Run the pytest suite in parallel. Use this instead of a bare `pytest tests`:
# the suite is mostly one-shot subprocesses, so it parallelises well, but the
# worker count must stay capped and distribution must stay file-level.
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
# Usage: tools/test.sh [--all] [pytest args...]   default target: tests
#        tools/test.sh --all                     full gate, BM3D included
#        tools/test.sh tests/test_nlmeans.py -q
#        VSFEEL_TEST_WORKERS=1 tools/test.sh     serial, no xdist
#
# Without --all this skips tests/test_bm3d.py, the single biggest block of
# suite time. Naming BM3D (or putting "bm3d" in an option) includes it, and
# --all is the full gate. Use tools/test_changed.sh while iterating; run --all
# once before hand-off, not after every edit.
#
# A missing reference plugin fails the run by default (`--require-references`):
# the reference comparisons skip when one is absent, which would silently delete
# the correctness net. Set VSFEEL_ALLOW_MISSING_REFS=1 for a box that
# intentionally lacks them.
#
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)
# shellcheck source=tools/venv.sh
. "$script_dir/venv.sh"

py=${VSFEEL_PYTHON:-}
if [[ -z $py ]]; then
    if ! py=$(vsfeel_venv_python "$root_dir"); then
        py=$(vsfeel_system_python) || {
            printf 'test.sh: no Python interpreter found (run `uv sync` first)\n' >&2
            exit 1
        }
        printf 'test.sh: no .venv; falling back to %s (run `uv sync` first)\n' \
            "$py" >&2
    fi
fi

# The venv's bin first: tests shell out to `vspipe` (test_dfttest/test_eedi3),
# and a PATH lookup that found the system one would drive the system plugin
# directory instead of the library this checkout just built.
if venv_bin=$(vsfeel_venv_bin "$root_dir"); then
    PATH=$venv_bin:$PATH
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

# Missing references are failures unless explicitly allowed (see header).
refs_args=()
if [[ ${VSFEEL_ALLOW_MISSING_REFS:-0} != 1 ]]; then
    refs_args+=(--require-references)
fi

# One command: cap at 8 workers, never trust `-n auto` (32 on the dev box).
workers=${VSFEEL_TEST_WORKERS:-$(nproc 2>/dev/null || echo 8)}
case $workers in
    ''|*[!0-9]*) workers=8 ;;
esac
if [ "$workers" -gt 8 ]; then
    workers=8
fi

# --all is the explicit full gate, consumed here so pytest never sees it.
# VSFEEL_TEST_ALL=1 is the same switch for wrappers that cannot add an argument.
include_all=0
if [[ ${VSFEEL_TEST_ALL:-0} == 1 ]]; then
    include_all=1
fi
args=()
for arg in "$@"; do
    case $arg in
        --all) include_all=1 ;;
        *) args+=("$arg") ;;
    esac
done
if [ "${#args[@]}" -gt 0 ]; then
    set -- "${args[@]}"
else
    set --
fi

if [ "$#" -eq 0 ]; then
    set -- tests
fi

# The worker cap and `--dist loadfile` are not advisory: more than 8 workers
# exhausts the GPU (see AGENTS.md), and a user-supplied -n would appear after
# the one this script passes, so pytest would win. Reject the override instead
# of silently running 16 workers; VSFEEL_TEST_WORKERS is the supported knob.
for arg in "$@"; do
    case $arg in
        -n|--numprocesses|--dist|-n[0-9]*|-n=*|--numprocesses=*|--dist=*)
            printf 'test.sh: -n and --dist are controlled by this script; use VSFEEL_TEST_WORKERS (max 8) to change workers\n' >&2
            exit 1
            ;;
    esac
done

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

# BM3D is the one filter whose numerical suite is long enough to dominate a
# run. Keep it out of the default and broad smoke runs; any argument naming it,
# and --lf (which can replay a BM3D failure), opts back in.
default_target=0
include_bm3d=0
for arg in "$@"; do
    case $arg in
        tests|tests/) default_target=1 ;;
        *[Bb][Mm]3[Dd]*) include_bm3d=1 ;;
        --lf|--last-failed) include_bm3d=1 ;;
    esac
done
bm3d_args=()
if [ "$include_all" -eq 0 ] && [ "$default_target" -eq 1 ] && [ "$include_bm3d" -eq 0 ]; then
    bm3d_args=(--ignore=tests/test_bm3d.py)
    printf 'test.sh: BM3D excluded (it is most of the suite). Run tools/test.sh --all for the full gate.\n' >&2
fi

cd -- "$root_dir"

# reference/ is read-only porting material, never part of this suite: it holds
# the references' own tests plus their dependency trees (lvsfunc, vs-jetpack),
# which are not ours to run and whose collection is what breaks ours.
ignore_reference=(--ignore=reference)

if [ "$workers" -le 1 ]; then
    exec "$py" -m pytest -q "${ignore_reference[@]}" "${bm3d_args[@]}" "${refs_args[@]}" "$@"
fi

if ! "$py" -c 'import xdist' >/dev/null 2>&1; then
    printf 'test.sh: pytest-xdist is not installed (dev dependency group); running serially\n' >&2
    exec "$py" -m pytest -q "${ignore_reference[@]}" "${bm3d_args[@]}" "${refs_args[@]}" "$@"
fi

exec "$py" -m pytest -q -n "$workers" --dist loadfile \
    "${ignore_reference[@]}" "${bm3d_args[@]}" "${refs_args[@]}" "$@"
