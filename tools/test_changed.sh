#!/usr/bin/env bash
#
# Run the test files that cover the current working-tree changes.
#
# This is the iteration loop, not the completion gate: it maps changed sources
# and tests to their filter's unit tests, so a one-line edit does not pay for
# the whole suite. Run tools/test.sh --all once before hand-off.
#
# The mapping is deliberately conservative. A change to shared plumbing
# (src/vsfeel.h, src/vsfeel.cpp, tests/conftest.py, CMakeLists.txt, the
# environment or build files) has no narrow test set, so it runs the full gate.
# An unrecognised path does the same rather than silently testing nothing.
#
# Usage: tools/test_changed.sh [--list] [--path PATH ...] [--all]
#        tools/test_changed.sh --list             show the selection
#        tools/test_changed.sh --path src/bm3d.cpp --list
#        tools/test_changed.sh --all              bypass the map, run the gate
#
# By default the changes are `git diff` plus staged and untracked files.
#
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

list_only=0
force_all=0
paths=()
while [ "$#" -gt 0 ]; do
    case $1 in
        --list) list_only=1 ;;
        --all) force_all=1 ;;
        --path)
            shift
            if [ "$#" -eq 0 ]; then
                printf 'test_changed.sh: --path needs a path\n' >&2
                exit 2
            fi
            paths+=("$1")
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'test_changed.sh: unknown argument %s (see --help)\n' "$1" >&2
            exit 2
            ;;
    esac
    shift
done

if [ "$force_all" -eq 1 ]; then
    if [ "$list_only" -eq 1 ]; then
        printf 'tools/test.sh --all\n'
        exit 0
    fi
    exec "$script_dir/test.sh" --all
fi

cd -- "$root_dir"

if [ "${#paths[@]}" -eq 0 ]; then
    while IFS= read -r path; do
        paths+=("$path")
    done < <(
        {
            git diff --name-only
            git diff --cached --name-only
            git ls-files --others --exclude-standard
        } | sort -u
    )
fi

selected_file=$(mktemp)
trap 'rm -f "$selected_file"' EXIT

add_test() {
    if [ -f "$1" ]; then
        printf '%s\n' "$1" >> "$selected_file"
    fi
}

shared_reason=
unknown_reason=
add_shared() {
    shared_reason=${shared_reason:-$1}
}

for path in "${paths[@]}"; do
    case $path in
        .scratch/*|reports/*|notes/*|*.html)
            continue ;;
        tests/test_*.py)
            add_test "$path" ;;
        tests/conftest.py|tests/bigbuckbunny_360p_grain.mp4)
            add_shared "$path" ;;
        tests/bm3d_oracle.py)
            add_test tests/test_bm3d_matcher.py ;;
        src/*)
            base=${path#src/}
            case $base in
                vsfeel.cpp|vsfeel.h)
                    add_shared "$path" ;;
                gen_spirv_header.py|config.h.in)
                    add_test tests/test_build.py ;;
                bm3d*)
                    add_test tests/test_bm3d.py
                    add_test tests/test_bm3d_matcher.py ;;
                bilateral*)
                    add_test tests/test_bilateral.py ;;
                box*)
                    add_test tests/test_box.py ;;
                dfttest*)
                    add_test tests/test_dfttest.py ;;
                eedi3*)
                    add_test tests/test_eedi3.py
                    add_test tests/test_eedi3aa.py
                    add_test tests/test_eedi3h.py ;;
                gaussblur*)
                    add_test tests/test_gaussblur.py ;;
                nlmeans*)
                    add_test tests/test_nlmeans.py ;;
                nnedi3*)
                    add_test tests/test_nnedi3.py ;;
                wnnm*)
                    add_test tests/test_wnnm.py ;;
                *)
                    unknown_reason=${unknown_reason:-$path} ;;
            esac ;;
        tools/*|hatch_build.py|.github/*)
            add_test tests/test_build.py ;;
        pyproject.toml|uv.lock|.python-version|CMakeLists.txt)
            add_shared "$path" ;;
        vsfeel/vsaa.py)
            add_test tests/test_vsaa.py ;;
        vsfeel/backend.py|vsfeel/__init__.py)
            add_test tests/test_python_backend.py
            add_test tests/test_vsaa.py ;;
        vsfeel/*)
            add_test tests/test_python_backend.py ;;
        AGENTS.md|README.md|CHANGELOG.md|LICENSE)
            continue ;;
        .clang-format|.clang-tidy|.editorconfig|.gitignore|.gitattributes)
            continue ;;
        *)
            unknown_reason=${unknown_reason:-$path} ;;
    esac
done

if [ -n "$shared_reason" ] || [ -n "$unknown_reason" ]; then
    reason=${shared_reason:-$unknown_reason}
    if [ -n "$shared_reason" ]; then
        printf 'test_changed.sh: %s is shared plumbing; running the full gate\n' "$reason" >&2
    else
        printf 'test_changed.sh: no test mapping for %s; running the full gate\n' "$reason" >&2
    fi
    if [ "$list_only" -eq 1 ]; then
        printf 'tools/test.sh --all\n'
        exit 0
    fi
    exec "$script_dir/test.sh" --all
fi

sort -u "$selected_file" -o "$selected_file"
tests=()
while IFS= read -r test; do
    tests+=("$test")
done < "$selected_file"

if [ "${#tests[@]}" -eq 0 ]; then
    printf 'test_changed.sh: no test-relevant changes\n' >&2
    exit 0
fi

if [ "$list_only" -eq 1 ]; then
    printf '%s\n' "${tests[@]}"
    exit 0
fi

printf 'test_changed.sh: %d changed path(s) -> %d test file(s)\n' \
    "${#paths[@]}" "${#tests[@]}" >&2
printf '  %s\n' "${tests[@]}" >&2
exec "$script_dir/test.sh" "${tests[@]}"
