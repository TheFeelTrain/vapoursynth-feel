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
#   shaders   glslc -Werror + spirv-val, through the build's shader-validate
#             target, then tools/shader_limits.py --check against the host's
#             GpuWorkgroup literals
#   tidy      clang-tidy over the compile database, per .clang-tidy
#   cppcheck  cppcheck over the compile database
#   ruff      ruff check + ruff format --check over vsfeel/ tools/
#             hatch_build.py src/gen_spirv_header.py tests/
#   notes     tools/notes_check.py over notes/*.md, per notes/AGENTS.md
#
# shaders/tidy/cppcheck need a configured build directory (they read the compile
# database, build/spirv_binaries.h and build/vk_spv/); tools/install.sh writes
# them, and tools/lint_stub.py derives build/lint/ from the database and header.
# VSFEEL_BUILD_DIR moves the tree. The project venv's bin comes first on PATH,
# so the pinned clang-format/clang-tidy/ruff from the `lint` dependency group
# (`uv sync --group lint`) are what runs; CLANG_FORMAT / CLANG_TIDY / CPPCHECK /
# RUFF name a binary explicitly to bypass that.
#
# A gate whose tool is missing reports SKIP rather than passing quietly; a gate
# whose build directory is missing is an error, because then it did not run.
#
# tidy and cppcheck are the two slow gates, and both are per-TU work: clang-tidy
# has no -j of its own, so run-clang-tidy.py -- the console script the pinned
# clang-tidy wheel installs -- drives it; cppcheck takes -j itself and uses its
# thread executor (the process one pays for copying results back). The TUs are
# independent, so the fan-out changes nothing about the findings.
# VSFEEL_LINT_JOBS sets the worker count; the default is the CPU count capped at
# 16, because an analyser takes a few hundred MB and an unbounded fan-out on a
# big box is how a lint run gets OOM-killed.
#
# The gates are independent, so they run concurrently: a full run costs about
# the slowest gate, not the sum. --fix runs them one at a time, since the fix-ups
# rewrite sources that the other gates read.
#
# Both analysers used to parse build/spirv_binaries.h -- tens of MB of generated
# SPIR-V data -- once per TU, which dominated cppcheck and was a large part of
# clang-tidy. The header holds no logic and neither gate reports on it (both keep
# findings in src/), so tools/lint_stub.py hands them a stub with the same
# symbols plus a compile database that resolves it first. That is what brought a
# full run from minutes down to about the clang-tidy pass.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)
# shellcheck source=tools/venv.sh
. "$script_dir/venv.sh"
build_dir=${VSFEEL_BUILD_DIR:-$root_dir/build}
# The analysers' derived view of the build tree; tools/lint_stub.py writes it.
lint_dir=$build_dir/lint

# The lint wheels are a dependency group, not system packages; putting the venv
# first is what makes a local run use the same versions CI does.
if venv_bin=$(vsfeel_venv_bin "$root_dir"); then
    PATH=$venv_bin:$PATH
    export PATH
fi

clang_format=${CLANG_FORMAT:-clang-format}
clang_tidy=${CLANG_TIDY:-clang-tidy}
cppcheck=${CPPCHECK:-cppcheck}
ruff=${RUFF:-ruff}
# tools/shader_limits.py and tools/notes_check.py are stdlib-only, so any
# interpreter runs them: the venv's when there is one (it is also what the PATH
# block above resolves `python3` to), else the system's.
python=${VSFEEL_PYTHON:-}
if [[ -z $python ]]; then
    if ! python=$(vsfeel_venv_python "$root_dir"); then
        python=$(vsfeel_system_python) || {
            printf 'lint.sh: no Python interpreter found (run `uv sync` first)\n' >&2
            exit 1
        }
    fi
fi

fix=0
gates=()

usage() {
    cat <<EOF
usage: tools/lint.sh [--fix] [gate ...]
  gates: format shaders tidy cppcheck ruff notes   (default: all of them)
  --fix  apply clang-format -i, clang-tidy --fix and ruff's fixes first
  VSFEEL_LINT_JOBS sets the tidy/cppcheck worker count (default: CPUs, max 16)
EOF
}

for arg in "$@"; do
    case $arg in
        --fix) fix=1 ;;
        -h|--help) usage; exit 0 ;;
        format|shaders|tidy|cppcheck|ruff|notes) gates+=("$arg") ;;
        *) printf 'lint.sh: unknown argument %s (try --help)\n' "$arg" >&2; exit 2 ;;
    esac
done
if [ "${#gates[@]}" -eq 0 ]; then
    gates=(format shaders tidy cppcheck ruff notes)
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

# Worker count for the two analysis gates (see the header comment).
lint_jobs() {
    local n=${VSFEEL_LINT_JOBS:-}
    if [ -z "$n" ]; then
        # nproc on Linux, sysctl on macOS; 1 keeps the gate working anywhere else.
        n=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || printf 1)
    fi
    if [ "$n" -gt 16 ]; then
        n=16
    fi
    printf '%s' "$n"
}

# The stub header and compile database the analysers use. Derived from the
# build's generated header, so this runs after the shaders gate.
lint_view() {
    "$python" "$root_dir/tools/lint_stub.py" --build-dir "$build_dir" \
        --out-dir "$lint_dir" || return 1
}

needs_view() {
    local gate
    for gate in "${gates[@]}"; do
        case $gate in tidy|cppcheck) return 0 ;; esac
    done
    return 1
}

gate_format() {
    have "$clang_format" || { skip "$clang_format"; return 2; }
    local files=("$root_dir"/src/*.cpp "$root_dir"/src/*.h)
    if [ "$fix" -eq 1 ]; then
        "$clang_format" -i "${files[@]}" || return 1
    else
        "$clang_format" --dry-run --Werror "${files[@]}" || return 1
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
    # Each step guards its own status: the gates run under `gate_x || rc=$?`,
    # which turns safe-mode off inside them, so an unguarded failure would let
    # the next step's success (a limit check against existing artifacts) mask
    # it. `return 1`, never `return $?`: exit status 2 is reserved for SKIP.
    # --parallel because a shader change can make this recompile many variants.
    cmake --build "$build_dir" --target shader-validate --parallel "$(lint_jobs)" || return 1
    # The host declares each pipeline's workgroup and LDS before creating it (the
    # device-limit check reads those numbers), so the SPIR-V and the
    # `GpuWorkgroup` literals have to agree; --check does that comparison instead
    # of leaving it to whoever remembers to read the table.
    "$python" "$root_dir/tools/shader_limits.py" --check \
        --src "$root_dir/src" --header "$build_dir/spirv_binaries.h" \
        "$build_dir/vk_spv/*.spv" || return 1
}

gate_tidy() {
    have "$clang_tidy" || { skip "$clang_tidy"; return 2; }
    need_build || return 1
    if [ "${view_ok:-0}" -ne 1 ]; then
        printf "    the analysers' stub header could not be built\n" >&2
        return 1
    fi
    local files=("$root_dir"/src/*.cpp)
    local -a tidy
    # run-clang-tidy.py ships with the pinned clang-tidy wheel; -clang-tidy-binary
    # pins it to the same binary the version banner named, instead of letting the
    # driver resolve its own from PATH.
    if have run-clang-tidy.py; then
        tidy=(run-clang-tidy.py -clang-tidy-binary "$(command -v "$clang_tidy")" \
            -p "$lint_dir" -quiet -j "$(lint_jobs)")
    else
        tidy=("$clang_tidy" -p "$lint_dir" --quiet)
    fi
    if [ "$fix" -eq 1 ]; then
        # Fix what is fixable, then check: the gate must report the tree's real
        # state, not the pre-fix one. run-clang-tidy.py only applies the fixes
        # here; the plain pass below is what reports whatever is left.
        "${tidy[@]}" -fix "${files[@]}" || true
    fi
    "${tidy[@]}" "${files[@]}" || return 1
}

gate_cppcheck() {
    have "$cppcheck" || { skip "$cppcheck"; return 2; }
    need_build || return 1
    if [ "${view_ok:-0}" -ne 1 ]; then
        printf "    the analysers' stub header could not be built\n" >&2
        return 1
    fi
    # *:*/vapoursynth/include/* drops findings in VSVulkan4.h. The plugin's own
    # path contains "vapoursynth-feel", which that glob deliberately misses.
    # -j is per-file here, the same set --file-filter keeps. The thread executor
    # avoids the process executor's result copying; a crash is still a failure.
    "$cppcheck" --executor=thread -j "$(lint_jobs)" \
        --project="$lint_dir/compile_commands.json" \
        --file-filter='*/src/*' \
        --enable=warning,performance,portability \
        --inline-suppr --quiet --error-exitcode=1 \
        --suppress=missingIncludeSystem \
        --suppress='*:*/vapoursynth/include/*' \
        --template='{file}:{line}: {severity}: {message} [{id}]' || return 1
}

gate_ruff() {
    have "$ruff" || { skip "$ruff"; return 2; }
    # The check set is curated in pyproject.toml [tool.ruff lint]: low-noise
    # correctness rules (pyflakes, pycodestyle-error, the whitespace
    # .editorconfig requires, dead noqa directives) that the whole tree passes,
    # rather than the full default set whose style opinions (%-format,
    # blind-except in test harnesses) the tree deliberately does not follow.
    # Keep the gate and the config in step: a new finding means a new defect,
    # not new style to debate. Line breaking is not in that set: it is
    # ruff format's job, checked below against [tool.ruff] line-length, the
    # same way clang-format gates the C++ side.
    local files=("$root_dir"/vsfeel "$root_dir"/tools "$root_dir"/hatch_build.py
        "$root_dir"/src/gen_spirv_header.py "$root_dir"/tests)
    if [ "$fix" -eq 1 ]; then
        "$ruff" check --fix --quiet "${files[@]}" || true
        "$ruff" format --quiet "${files[@]}" || true
    fi
    # Both halves report, and both guard: a `ruff check` finding must not be
    # masked by a clean `ruff format --check` (see gate_shaders).
    "$ruff" check "${files[@]}" || return 1
    "$ruff" format --check "${files[@]}" || return 1
}

gate_notes() {
    have "$python" || { skip "$python"; return 2; }
    # notes/AGENTS.md is the authority on the notes; this checks the half of it a
    # machine can read: the fixed parts and their order, both line budgets, the
    # file index, report IDs. It reads no build directory, so unlike the three
    # gates above it also runs on a bare checkout.
    "$python" "$root_dir/tools/notes_check.py" "$root_dir/notes" || return 1
}

status=()
failed=0
# The tool's own version next to the gate. clang-format/clang-tidy/ruff come from
# the pinned lint group, cppcheck from Homebrew on CI and the distro here, so a
# finding that reproduces on neither box is a version difference, and the log has
# to show which one ran.
gate_version() {
    # clang-tidy prints a banner first, so take the first line naming a version.
    # A tool the gate will skip must not abort the run here: this runs outside
    # the gate's own availability check, and `set -e` would exit on the failed
    # substitution, so the missing-tool SKIP never printed.
    case $1 in
        format) "$clang_format" --version 2>/dev/null | grep -m1 -i version || true ;;
        tidy) "$clang_tidy" --version 2>/dev/null | grep -m1 -i version || true ;;
        cppcheck) "$cppcheck" --version 2>/dev/null | head -1 || true ;;
        ruff) "$ruff" --version 2>/dev/null | head -1 || true ;;
        *) : ;;
    esac
}
# The shaders gate goes first and alone: it can regenerate
# build/spirv_binaries.h, which the analysers' stub is derived from. The rest
# read the same tree and write only their own log, so they run concurrently --
# unless --fix is rewriting sources for the other gates to read.
ordered=()
for gate in "${gates[@]}"; do
    [ "$gate" = shaders ] && ordered+=(shaders)
done
for gate in "${gates[@]}"; do
    [ "$gate" = shaders ] || ordered+=("$gate")
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/vsfeel-lint.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

# Run one gate in this (sub)shell and record its status, which the caller reads
# back once every gate has finished. 2 is the gates' reserved "tool missing"
# status; every real failure is a 1, so a tool that happens to exit 2 cannot
# masquerade as skipped.
run_gate() {
    local gate=$1 rc=0
    "gate_$gate" || rc=$?
    printf '%s\n' "$rc" >"$work_dir/$gate.rc"
}

rest=()
if [ "${ordered[0]:-}" = shaders ]; then
    run_gate shaders >"$work_dir/shaders.log" 2>&1 || true
    rest=("${ordered[@]:1}")
else
    rest=("${ordered[@]}")
fi

# The stub header is derived from the build's header, so it is prepared only
# after the shaders gate may have rewritten that header.
view_ok=1
if needs_view; then
    if [ ! -f "$build_dir/compile_commands.json" ]; then
        view_ok=0    # each gate's need_build check explains this
    elif ! lint_view >"$work_dir/stub.log" 2>&1; then
        view_ok=0
        cat "$work_dir/stub.log" >&2
    fi
fi

if [ "${#rest[@]}" -gt 0 ]; then
    if [ "$fix" -eq 1 ]; then
        for gate in "${rest[@]}"; do
            run_gate "$gate" >"$work_dir/$gate.log" 2>&1 || true
        done
    else
        pids=()
        for gate in "${rest[@]}"; do
            (run_gate "$gate") >"$work_dir/$gate.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
    fi
fi

# Print the gates in the order they were asked for, whatever order they
# finished in, so the log reads the same as a serial run.
for gate in "${ordered[@]}"; do
    version=$(gate_version "$gate")
    if [ -n "$version" ]; then
        printf '==> %s (%s)\n' "$gate" "$version"
    else
        printf '==> %s\n' "$gate"
    fi
    cat "$work_dir/$gate.log"
    # A gate killed before it recorded a status (OOM, for one) is a failure.
    rc=$(cat "$work_dir/$gate.rc" 2>/dev/null) || rc=1
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
