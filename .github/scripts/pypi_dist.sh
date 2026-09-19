#!/usr/bin/env bash
#
# Look up distributions of the current release on PyPI so a re-run can reuse
# what is already published instead of rebuilding it. A distribution's filename
# contains its version, and a version is only ever built once, so "the filename
# is on PyPI" means "this exact wheel is already built and good".
#
#   pypi_dist.sh published <filename>      exit 0 if <filename> is published
#   pypi_dist.sh fetch <filename> <dir>    download it into <dir>
#
# FORCE_BUILD=true makes `published` always report "not published".
set -euo pipefail

INDEX="https://pypi.org/simple/vapoursynth-feel/"

url_of() {
    # Anchored on the whole name, so e.g. foo.whl cannot match foo.whl.metadata.
    curl -fsS "${INDEX}" 2>/dev/null |
        grep -oE "https://[^\"#]*/${1}(#|\$)" |
        head -1 | cut -d'#' -f1 || true
}

case "${1:-}" in
published)
    file="${2:?filename required}"
    if [ "${FORCE_BUILD:-false}" = "true" ]; then
        echo "FORCE_BUILD=true, not reusing published files"
        exit 1
    fi
    if [ -n "$(url_of "${file}")" ]; then
        echo "${file} is already on PyPI"
        exit 0
    fi
    exit 1
    ;;
fetch)
    file="${2:?filename required}"
    dir="${3:?destination required}"
    url="$(url_of "${file}")"
    if [ -z "${url}" ]; then
        echo "no published URL for ${file}" >&2
        exit 1
    fi
    mkdir -p "${dir}"
    curl -fsSL -o "${dir}/${file}" "${url}"
    echo "reused ${file} from PyPI"
    ;;
*)
    echo "usage: $0 published|fetch <filename> [dir]" >&2
    exit 2
    ;;
esac
