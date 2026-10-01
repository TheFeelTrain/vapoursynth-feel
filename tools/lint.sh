#!/usr/bin/env bash
#
# Every code-quality gate over the C++ host code, the GLSL kernels and the
# Python tooling. CI runs exactly this (see .github/workflows/lint.yml).
#
#   tools/lint.sh                  all gates
#   tools/lint.sh --fix            apply the mechanical fixes, then check
#   tools/lint.sh format tidy      only the named gates
#
# Gates:
#   format    clang-format over src/*.cpp src/*.h, per .clang-format
#   shaders   glslc -Werror + spirv-val, through the build's shader-validate target
#   tidy      clang-tidy over the compile database, per .clang-tidy
#   cppcheck  cppcheck over the compile database
#   ruff      ruff check over vsfeel/ tools/ hatch_build.py
#             src/gen_spirv_header.py tests/
#
# shaders/tidy/cppcheck need a configured build directory (they read
# build/compile_commands.json and build/vk_spv/); tools/install.sh writes
# both. VSFEEL_BUILD_DIR moves it. The project venv's bin comes first on PATH,
# so the pinned clang-format/clang-tidy/ruff from the `lint` dependency group
# (`uv sync --group lint`) are what runs; CLANG_FORMAT / CLANG_TIDY / CPPCHECK /
# RUFF name a binary explicitly to bypass that.
#
# A gate whose tool is missing reports SKIP rather than passing quietly; a gate
# whose build directory is missing is an error, because then it did not run.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=${VSFEEL_BUILD_DIR:-$root_dir/build}

# The lint wheels are a dependency group, not system packages; putting the venv
# first is what makes a local run use the same versions CI does.
if [[ -d $root_dir/.venv/bin ]]; then
    PATH=$root_dir/.venv/bin:$PATH
    export PATH
fi

clang_format=${CLANG_FORMAT:-clang-format}
clang_tidy=${CLANG_TIDY:-clang-tidy}
cppcheck=${CPPCHECK:-cppcheck}
ruff=${RUFF:-ruff}

fix=0
gates=()

usage() {
    cat <<EOF
usage: tools/lint.sh [--fix] [gate ...]
  gates: format shaders tidy cppcheck ruff   (default: all of them)
  --fix  apply clang-format -i and clang-tidy --fix first
EOF
}

for arg in "$@"; do
    case $arg in
        --fix) fix=1 ;;
        -h|--help) usage; exit 0 ;;
        format|shaders|tidy|cppcheck|ruff) gates+=("$arg") ;;
        *) printf 'lint.sh: unknown argument %s (try --help)\n' "$arg" >&2; exit 2 ;;
    esac
done
if [ "${#gates[@]}" -eq 0 ]; then
    gates=(format shaders tidy cppcheck ruff)
fi

have() { command -v "$1" >/dev/null 2>&1; }

skip() { printf '    skipped: %s not found\n' "$1"; return 2; }

need_build() {
    if [ ! -f "$build_dir/compile_commands.json" ]; then
        printf '    %s/compile_commands.json is missing\n' "$build_dir" >&2
        printf '    run tools/install.sh (or configure with -D CMAKE_EXPORT_COMPILE_COMMANDS=ON)\n' >&2
        return 1
    fi
}

gate_format() {
    have "$clang_format" || { skip "$clang_format"; return 2; }
    local files=("$root_dir"/src/*.cpp "$root_dir"/src/*.h)
    if [ "$fix" -eq 1 ]; then
        "$clang_format" -i "${files[@]}"
    else
        "$clang_format" --dry-run --Werror "${files[@]}"
    fi
}

gate_shaders() {
    have cmake || { skip cmake; return 2; }
    if [ ! -f "$build_dir/CMakeCache.txt" ]; then
        printf '    %s is not configured; run tools/install.sh first\n' "$build_dir" >&2
        return 1
    fi
    # Recompiles stale .spv (glslc -Werror) and validates every module with
    # spirv-val. The target exists only when spirv-val was found at configure
    # time, so a build without SPIRV-Tools fails here instead of passing.
    cmake --build "$build_dir" --target shader-validate
}

gate_tidy() {
    have "$clang_tidy" || { skip "$clang_tidy"; return 2; }
    need_build || return 1
    local files=("$root_dir"/src/*.cpp)
    if [ "$fix" -eq 1 ]; then
        # Fix what is fixable, then check: the gate must report the tree's real
        # state, not the pre-fix one.
        "$clang_tidy" -p "$build_dir" --fix --quiet "${files[@]}" || true
    fi
    "$clang_tidy" -p "$build_dir" --quiet "${files[@]}"
}

gate_cppcheck() {
    have "$cppcheck" || { skip "$cppcheck"; return 2; }
    need_build || return 1
    # *:*/vapoursynth/include/* drops findings in VSVulkan4.h. The plugin's own
    # path contains "vapoursynth-feel", which that glob deliberately misses.
    "$cppcheck" --project="$build_dir/compile_commands.json" \
        --file-filter='*/src/*' \
        --enable=warning,performance,portability \
        --inline-suppr --quiet --error-exitcode=1 \
        --suppress=missingIncludeSystem \
        --suppress='*:*/vapoursynth/include/*' \
        --template='{file}:{line}: {severity}: {message} [{id}]'
}

gate_ruff() {
    have "$ruff" || { skip "$ruff"; return 2; }
    # The check set is curated in pyproject.toml [tool.ruff lint]: low-noise
    # correctness rules (pyflakes, pycodestyle-error, ambiguous names) that
    # the whole tree passes, rather than the full default set whose style
    # opinions (line length, %-format, blind-except in test harnesses) the
    # tree deliberately does not follow. Keep the gate and the config in
    # step: a new finding means a new defect, not new style to debate.
    local files=("$root_dir"/vsfeel "$root_dir"/tools "$root_dir"/hatch_build.py
        "$root_dir"/src/gen_spirv_header.py "$root_dir"/tests)
    if [ "$fix" -eq 1 ]; then
        "$ruff" check --fix --quiet "${files[@]}" || true
        "$ruff" format --quiet "${files[@]}" || true
    fi
    "$ruff" check "${files[@]}"
}

status=()
failed=0
for gate in "${gates[@]}"; do
    printf '==> %s\n' "$gate"
    rc=0
    "gate_$gate" || rc=$?
    case $rc in
        0) status+=("$gate: ok") ;;
        2) status+=("$gate: skipped") ;;
        *) status+=("$gate: FAILED"); failed=1 ;;
    esac
done

printf '\n'
for line in "${status[@]}"; do
    printf '  %s\n' "$line"
done
exit "$failed"
