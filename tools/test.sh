#!/usr/bin/env bash
#
# Run the pytest suite in parallel. Use this instead of a bare `pytest tests`:
# the suite is 357 one-shot subprocesses (median 1.4 s each, ~95% of the serial
# wall time), so it parallelises ~4x, but the worker count must stay capped and
# distribution must stay file-level.
#
#   -pytest-xdist -n 8 --dist loadfile : 789 tests in ~133 s (serial: 522 s)
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

py=${VSFEEL_PYTHON:-python3}

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

cd -- "$root_dir"

if [ "$workers" -le 1 ]; then
    exec "$py" -m pytest -q "$@"
fi

if ! "$py" -c 'import xdist' >/dev/null 2>&1; then
    printf 'test.sh: pytest-xdist is not installed (dev dependency group); running serially\n' >&2
    exec "$py" -m pytest -q "$@"
fi

exec "$py" -m pytest -q -n "$workers" --dist loadfile "$@"
