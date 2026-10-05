#!/usr/bin/env bash
#
# Venv discovery shared by tools/install.sh, tools/test.sh and tools/lint.sh.
#
# uv creates `.venv/bin` on POSIX and `.venv/Scripts` on a native Windows
# Python (where the interpreter is `python.exe`), so a script that only looks at
# `bin/` silently falls back to the system interpreter there: the wrong
# VapourSynth for install.sh/test.sh, and no `python3` at all for lint.sh.
# Sourced, not executed; every helper takes the checkout root as its argument
# and prints its answer only on success, so `x=$(...) || fallback` reads the way
# it looks under `set -e`.

vsfeel_venv_bin() {
    local root=$1 candidate
    for candidate in "$root/.venv/bin" "$root/.venv/Scripts"; do
        if [[ -d $candidate ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

vsfeel_venv_python() {
    local bin
    bin=$(vsfeel_venv_bin "$1") || return 1
    if [[ -x $bin/python ]]; then
        printf '%s' "$bin/python"
    elif [[ -x $bin/python.exe ]]; then
        printf '%s' "$bin/python.exe"
    else
        return 1
    fi
}

vsfeel_system_python() {
    local candidate
    # `python3` first (the POSIX name), then `python` for a Windows box that
    # only ships the launcher-less name.
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}
