#!/usr/bin/env bash
#
# Build libvsfeel.so, install it into VapourSynth's plugin directory, and prove
# the installed copy is byte-identical to the build. Use this instead of copying
# by hand: a failed compile used to be indistinguishable from a successful one,
# and the notes record a phantom bug traced to a stale .spv shipped that way.
#
# Usage: tools/install.sh [-b BUILD_DIR] [-p PLUGIN_DIR] [-c BUILD_TYPE]
#
set -euo pipefail

die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)

build_dir=${VSFEEL_BUILD_DIR:-$root_dir/build}
plugin_dir=${VSFEEL_PLUGIN_DIR:-}
build_type=Release

usage() {
    cat <<EOF
usage: tools/install.sh [-b DIR] [-p DIR] [-c TYPE]
  -b DIR   build directory (default: $build_dir; env VSFEEL_BUILD_DIR)
  -p DIR   VapourSynth plugin root (default: autodetected; env VSFEEL_PLUGIN_DIR)
  -c TYPE  CMake build type (default: $build_type)
EOF
}

while getopts ':b:p:c:h' opt; do
    case $opt in
        b) build_dir=$OPTARG ;;
        p) plugin_dir=$OPTARG ;;
        c) build_type=$OPTARG ;;
        h) usage; exit 0 ;;
        :) die "option -$OPTARG needs an argument" ;;
        \?) die "unknown option -$OPTARG (try -h)" ;;
    esac
done

# Ask the module that will actually load the plugin where its plugins live, so a
# venv or a differently-versioned Python cannot install into the wrong tree.
if [[ -z $plugin_dir ]]; then
    plugin_dir=$(python3 -c 'import vapoursynth as vs; print(vs.get_plugin_dir())') \
        || die "cannot import vapoursynth; pass -p DIR or set VSFEEL_PLUGIN_DIR"
fi
[[ -n $plugin_dir ]] || die "autodetected an empty plugin directory"

build_lib=$build_dir/libvsfeel.so
install_subdir=$plugin_dir/vsfeel
install_lib=$install_subdir/libvsfeel.so

hash_file() {
    [[ -f $1 ]] || die "expected a file at $1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$1" | cut -d' ' -f1
    else
        shasum -a 256 -- "$1" | cut -d' ' -f1
    fi
}

echo "==> configure $root_dir in $build_dir ($build_type)"
cmake -S "$root_dir" -B "$build_dir" -D "CMAKE_BUILD_TYPE=$build_type"

echo "==> build"
cmake --build "$build_dir" --config "$build_type"

[[ -f $build_lib ]] || die "build produced no $build_lib"

build_hash=$(hash_file "$build_lib")

mkdir -p "$install_subdir"

# Skip the copy when the plugin directory already holds this exact build, so a
# second run is a true no-op rather than a needless mtime bump.
if [[ -f $install_lib ]] && [[ $(hash_file "$install_lib") == "$build_hash" ]]; then
    echo "==> install: already up to date, copy skipped"
else
    cp -f -- "$build_lib" "$install_lib"
    echo "==> install: copied $build_lib -> $install_lib"
fi

manifest_src=$build_dir/vk/manifest.vs
manifest_dst=$install_subdir/manifest.vs
if [[ -f $manifest_src ]] && ! cmp -s -- "$manifest_src" "$manifest_dst"; then
    cp -f -- "$manifest_src" "$manifest_dst"
    echo "==> install: wrote $manifest_dst"
fi

install_hash=$(hash_file "$install_lib")

printf 'build     sha256 %s  %s\n' "$build_hash" "$build_lib"
printf 'installed sha256 %s  %s\n' "$install_hash" "$install_lib"

if [[ $build_hash != "$install_hash" ]]; then
    die "installed copy does not match the build; the plugin directory is stale"
fi
echo "==> ok: installed plugin matches the build"
