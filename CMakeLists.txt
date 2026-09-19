cmake_minimum_required(VERSION 3.21)

project(vapoursynth-feel LANGUAGES C CXX)

# The NNEDI3 weight blob is linked in as an RCDATA resource on Windows, where
# the default toolchains ship no objcopy.
if(WIN32)
    enable_language(RC)
endif()

# Vulkan is reached through volk, which dlopens the loader at runtime: nothing
# links libvulkan, and that is what lets the Linux wheel be manylinux (no
# manylinux policy whitelists libvulkan.so.1). The headers are pinned by hash
# rather than taken from a Vulkan SDK so a bare container can build the plugin;
# release tarballs are used instead of git clones because they are an order of
# magnitude smaller and need no git.
include(FetchContent)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # volk is a static lib linked into a .so
FetchContent_Declare(vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz
    URL_HASH SHA256=e87dce08116151f6b6d7de6b6faf41498e87e6cf848ff16fa3bd5402190ad4a3)
FetchContent_MakeAvailable(vulkan_headers)
# Pin volk to the fetched headers: left to itself it prefers a system Vulkan
# SDK, so the same commit would compile against different headers per machine.
set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH
    "Vulkan headers used to build volk" FORCE)
FetchContent_Declare(volk
    URL https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz
    URL_HASH SHA256=6400c7b23e24d17e4f04bac49b55b06c4e87677d33398e90344743ec73560ca6)
FetchContent_MakeAvailable(volk)

find_program(VK_GLSLC glslc)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Git QUIET)

# ---------------------------------------------------------------------------
# Version: hatch-vcs (pyproject.toml) is the single source of truth. A
# packaging build passes the version in; a bare CMake build resolves it via
# the same setuptools_scm entry point and only then falls back to git describe.
# The result lands in config.h and reaches the runtime as VSFEEL_VERSION.
# ---------------------------------------------------------------------------
if(NOT VSFEEL_VERSION)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "from setuptools_scm import get_version; print(get_version())"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _scm_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _scm_status
        ERROR_VARIABLE _scm_error
    )
    if(_scm_status EQUAL 0 AND _scm_version)
        set(VSFEEL_VERSION "${_scm_version}")
    endif()
endif()

if(NOT VSFEEL_VERSION AND GIT_FOUND)
    # `--always` is deliberately absent: it makes describe succeed with a bare
    # hash even when no version tag is reachable, which is what previously hid
    # the failure. Check the status and fall back explicitly instead.
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long
                --match "[0-9]*" --match "v[0-9]*" --dirty
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _git_describe
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _git_status
        ERROR_QUIET
    )
    if(_git_status EQUAL 0)
        string(REGEX REPLACE "^v" "" VSFEEL_VERSION "${_git_describe}")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE _git_hash
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _hash_status
        )
        if(_hash_status EQUAL 0 AND _git_hash)
            set(VSFEEL_VERSION "0.0.dev0+g${_git_hash}")
        endif()
    endif()
endif()

if(NOT VSFEEL_VERSION)
    set(VSFEEL_VERSION "unknown")
    message(WARNING
        "vapoursynth-feel: no version could be determined "
        "(setuptools_scm: ${_scm_error}; git describe status: ${_git_status})")
endif()
message(STATUS "vapoursynth-feel ${VSFEEL_VERSION}")

# Helpers
function(configure_common target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
    )
endfunction()

# Manifest
if(WIN32)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" "[VapourSynth Manifest V1]\nvsfeel\n")
else()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" "[VapourSynth Manifest V1]\nlibvsfeel\n")
endif()

# SPIR-V: compile the GLSL kernels with glslc when available.
# Each kernel is compiled for 16/32-bit staging formats.

# --- shader variant table -----------------------------------------------------
# One "|"-separated "<out>|<source>|<defs>" entry per .spv (pipes, not
# semicolons: a ";" would split a quoted entry into three separate list
# elements). The table is the only place a variant is declared: entry point, bit
# depth, radius and every other -D live in the <defs> field, so adding a variant
# is one line and there is no entry-name -> -D ladder to extend in several
# places. add_spv_variant() below turns an entry into the glslc rule.
#
# The <defs> fields are quoted so that entries stay one CMake list element even
# when they carry spaces; separate_arguments() splits them back into arguments.
#
# --target-env is deliberately left as it is here (bilateral/bm3d/gaussblur/
# nlmeans compile as SPIR-V 1.0, dfttest/eedi3/nnedi3 as 1.6). Unifying it is a
# codegen change and belongs in its own measured commit, not in this refactor.

set(VK_BITS "16" "32")

# Bilateral: one entry point per file, per io variant.
set(VK_BILATERAL_VARIANTS)
foreach(component IN ITEMS shared plain)
    foreach(bits IN LISTS VK_BITS)
        list(APPEND VK_BILATERAL_VARIANTS
            "bilateral_${component}_${bits}|bilateral_${component}.comp|-DBITS=${bits}")
    endforeach()
endforeach()

# BM3D: two single-entry files, no io variants (float only).
set(VK_BM3D_VARIANTS
    "bm3d|bm3d.comp|"
    "bm3d_agg|bm3d_agg.comp|")

# GaussBlur: one kernel with three entry points (fused small path + the two
# large-path passes), per io variant (16 int, 32 float).
set(VK_GAUSS_ENTRIES gauss vert horiz)
set(VK_GAUSS_ENTRY_DEFS -DENTRY_GAUSS -DENTRY_VERT -DENTRY_HORIZ)
set(VK_GAUSS_VARIANTS)
foreach(bits IN LISTS VK_BITS)
    foreach(entry_i RANGE 0 2)
        list(GET VK_GAUSS_ENTRIES ${entry_i} entry)
        list(GET VK_GAUSS_ENTRY_DEFS ${entry_i} entry_def)
        list(APPEND VK_GAUSS_VARIANTS
            "gaussblur_${bits}_${entry}|gaussblur.comp|-DBITS=${bits} ${entry_def}")
    endforeach()
endforeach()

# DFTTest: pad + fused + col2im kernels per io variant (16 int, 32
# float); the fused kernel is additionally compiled per temporal radius
# (0..3, i.e. tbsize 1/3/5/7) so its loops unroll and the working array
# stays in registers (the reference does the same via `#define RADIUS`).
# The fused kernel uses subgroup barriers instead of workgroup barriers
# (each 16-lane block lives entirely in one wavefront), so SPIR-V 1.3+ is
# required.
set(VK_DFT_ENTRIES pad_slot pad_direct col2im)
set(VK_DFT_ENTRY_DEFS -DENTRY_PAD_SLOT -DENTRY_PAD_DIRECT -DENTRY_COL2IM)
set(VK_DFT_VARIANTS)
foreach(bits IN LISTS VK_BITS)
    foreach(entry_i RANGE 0 2)
        list(GET VK_DFT_ENTRIES ${entry_i} entry)
        list(GET VK_DFT_ENTRY_DEFS ${entry_i} entry_def)
        list(APPEND VK_DFT_VARIANTS
            "dfttest_${bits}_${entry}|dfttest.comp|--target-env=vulkan1.3 -DBITS=${bits} ${entry_def}")
    endforeach()
    foreach(radius RANGE 0 3)
        list(APPEND VK_DFT_VARIANTS
            "dfttest_${bits}_fused_r${radius}|dfttest.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_FUSED -DRADIUS=${radius}"
            # slot-direct fused variant (no padded-fallback branch in im2col);
            # used when every temporal slice of the frame lives in a slot
            "dfttest_${bits}_fused_direct_r${radius}|dfttest.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_FUSED -DFUSED_DIRECT -DRADIUS=${radius}")
    endforeach()
endforeach()

# NLMeans: weight / accumulation / finish / pad entry points per io
# variant (16-bit int, 32-bit float); everything else is a specialization
# constant.
set(VK_NLM_ENTRIES weight acc finish pad)
set(VK_NLM_ENTRY_DEFS -DENTRY_WEIGHT -DENTRY_ACC -DENTRY_FINISH -DENTRY_PAD)
set(VK_NLM_VARIANTS)
foreach(bits IN LISTS VK_BITS)
    foreach(entry_i RANGE 0 3)
        list(GET VK_NLM_ENTRIES ${entry_i} entry)
        list(GET VK_NLM_ENTRY_DEFS ${entry_i} entry_def)
        list(APPEND VK_NLM_VARIANTS
            "nlmeans_${bits}_${entry}|nlmeans.comp|-DBITS=${bits} ${entry_def}")
    endforeach()
endforeach()

# EEDI3: row kernel (one 32-lane WG per interp row, subgroup-register DP
# with shuffles) + vcheck kernel (single WG serial row walk), per io
# variant (16 int, 32 float). The row kernel needs subgroup ops + the
# Vulkan memory model, so SPIR-V 1.3+ (vulkan1.3 target) is required.
# PROBE and MAXW are cache variables: PROBE is a diagnostics-only ablation
# level, MAXW is the max plane width the vcheck LDS ping-pong can hold, kept in
# step with the host's fit check (EEDI3_MAXW_LDS) so the two cannot drift into a
# pipeline whose static LDS exceeds the device.
set(EEDI3_PROBE "0" CACHE STRING "EEDI3 diagnostics-only shader ablation level")
set(EEDI3_MAXW "4096" CACHE STRING "EEDI3 vcheck LDS max plane width")
set(VK_EEDI3_VARIANTS)
foreach(bits IN LISTS VK_BITS)
    list(APPEND VK_EEDI3_VARIANTS
        "eedi3_${bits}_row|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_ROW"
        "eedi3_${bits}_vcheck|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK"
        "eedi3_${bits}_vcheck_lds|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_LDS=1"
        "eedi3_${bits}_vcheck_para|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_PARA=1"
        "eedi3_${bits}_vcheck_para_j1|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_PARA=2"
        "eedi3_${bits}_vcheck_para_j2|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_PARA=3"
        "eedi3_${bits}_vcheck_para_j3|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_PARA=4"
        "eedi3_${bits}_vcheck_para_j4|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_PARA=5"
        "eedi3_${bits}_vcheck_para_j5|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DPROBE=${EEDI3_PROBE} -DMAXW=${EEDI3_MAXW} -DENTRY_VCHECK -DVCHECK_PARA=6"
        # upload kernels: mirror-pad builder + empty-row vcheck copy and the
        # assemble/transpose passes (plain 1D kernels, fixed local size, no
        # subgroup ops); vulkan1.3 target keeps one rule for the file.
        "eedi3_${bits}_pad|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_PAD"
        "eedi3_${bits}_vcopy|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_VCOPY"
        "eedi3_${bits}_blit|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_BLIT"
        "eedi3_${bits}_xpose|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_XPOSE"
        "eedi3_${bits}_compose|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_COMPOSE"
        "eedi3_${bits}_assemblev|eedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_ASSEMBLEV")
endforeach()

# NNEDI3: prescreener + cooperative predictor (+PXP=4 narrow variant for
# wide networks, +PXP=4 small-tile variant for FS<=64: 4KB LDS instead
# of 18KB) + count + full-frame assembler per io variant. The predictor
# needs subgroup ops + the Vulkan memory model, so SPIR-V 1.3+.
set(VK_NNEDI3_VARIANTS)
foreach(bits IN LISTS VK_BITS)
    list(APPEND VK_NNEDI3_VARIANTS
        "nnedi3_${bits}_pad|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_PAD"
        "nnedi3_${bits}_prescreen|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_PRESCREEN"
        "nnedi3_${bits}_predict|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_PREDICT"
        "nnedi3_${bits}_predict_n4|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_PREDICT -DPXP=4"
        "nnedi3_${bits}_predict_n4s|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_PREDICT -DPXP=4 -DSHSTRIDE=64u"
        "nnedi3_${bits}_count|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_COUNT"
        "nnedi3_${bits}_assemble|nnedi3.comp|--target-env=vulkan1.3 -DBITS=${bits} -DENTRY_ASSEMBLE")
endforeach()

set(VK_VARIANTS
    ${VK_BILATERAL_VARIANTS}
    ${VK_BM3D_VARIANTS}
    ${VK_GAUSS_VARIANTS}
    ${VK_DFT_VARIANTS}
    ${VK_NLM_VARIANTS}
    ${VK_EEDI3_VARIANTS}
    ${VK_NNEDI3_VARIANTS})

# --- the glslc rule -----------------------------------------------------------

# One .spv per table entry. <defs> is a single string so the rule stays one
# command; it is split back into arguments here.
#
# Each rule also DEPENDS on a per-variant flags file holding that variant's
# glslc arguments. Ninja tracks the command line itself, but the default Unix
# Makefiles generator does not: without the stamp a -D, a --target-env or a
# PROBE/MAXW cache change leaves a stale .spv behind and silently ships the old
# kernel. The file is written with file(GENERATE) (content-hash: only a real
# flag change touches it, so an unrelated CMakeLists edit does not recompile
# the shaders).
function(add_spv_variant out source defs)
    separate_arguments(flags UNIX_COMMAND "${defs}")

    set(spv "${VK_SPV_DIR}/${out}.spv")
    set(stamp "${VK_SPV_DIR}/${out}.spv.flags")
    file(GENERATE OUTPUT "${stamp}" CONTENT "${flags}\n")

    add_custom_command(
        OUTPUT "${spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${VK_SPV_DIR}"
        COMMAND "${VK_GLSLC}" -O -fshader-stage=compute ${flags} -o "${spv}" "${source}"
        DEPENDS "${source}" "${stamp}"
        VERBATIM)
    set(VK_SPV_OUTPUTS ${VK_SPV_OUTPUTS} "${spv}" PARENT_SCOPE)
endfunction()

if(VK_GLSLC)
    set(VK_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/vk_spv")
    set(VK_SPV_OUTPUTS)
    foreach(variant IN LISTS VK_VARIANTS)
        string(REPLACE "|" ";" variant_fields "${variant}")
        list(LENGTH variant_fields field_count)
        if(field_count LESS 2)
            message(FATAL_ERROR "malformed shader variant entry: ${variant}")
        endif()
        list(GET variant_fields 0 out)
        list(GET variant_fields 1 source)
        list(SUBLIST variant_fields 2 -1 defs)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${source}")
            message(FATAL_ERROR "shader variant ${out}: missing src/${source}")
        endif()
        add_spv_variant("${out}" "${CMAKE_CURRENT_SOURCE_DIR}/src/${source}" "${defs}")
    endforeach()
else()
    message(FATAL_ERROR "glslc not found")
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/gen_spirv_header.py"
            --out "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
            ${VK_SPV_OUTPUTS}
    DEPENDS ${VK_SPV_OUTPUTS}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/gen_spirv_header.py"
    VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h" PROPERTIES GENERATED TRUE)

# NNEDI3 weights: link the raw blob straight into the binary so the 13.5 MB
# never passes through the compiler, and the wheel stays self-contained
# (reference/ is not shipped). objcopy turns it into an object exposing
# _binary_nnedi3_weights_bin_{start,end}; MSVC has no objcopy, so Windows
# embeds the same bytes as an RCDATA resource that nnedi3.cpp reads with
# FindResource (see weights_blob()).
if(WIN32)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/nnedi3_weights.bin"
                   "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.bin" COPYONLY)
    # Bare filename: the resource compiler runs with the build dir as its CWD.
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.rc"
         "NNEDI3_WEIGHTS RCDATA \"nnedi3_weights.bin\"\n")
    set(VSFEEL_WEIGHTS_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.rc")
else()
    find_program(VK_OBJCOPY objcopy REQUIRED)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.o"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nnedi3_weights.bin"
                "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.bin"
        # run under the output dir so the embedded symbols derive from the bare
        # filename (_binary_nnedi3_weights_bin_{start,end}), not the build path
        COMMAND ${VK_OBJCOPY} --input binary --output elf64-x86-64
                --binary-architecture i386:x86-64
                nnedi3_weights.bin nnedi3_weights.o
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/nnedi3_weights.bin"
        VERBATIM)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.o"
        PROPERTIES GENERATED TRUE)
    set(VSFEEL_WEIGHTS_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.o")
endif()

# Plugin
add_library(vsfeel SHARED
    src/vsfeel.cpp
    src/bilateral.cpp
    src/bm3d.cpp
    src/gaussblur.cpp
    src/dfttest.cpp
    src/nlmeans.cpp
    src/eedi3.cpp
    src/nnedi3.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    ${VSFEEL_WEIGHTS_SOURCE})
configure_common(vsfeel)
if(WIN32)
    # The manifest above names the plugin "vsfeel", so the file must be
    # vsfeel.dll: MinGW would otherwise emit libvsfeel.dll and the host would
    # never find it.
    set_target_properties(vsfeel PROPERTIES PREFIX "")
endif()
if(MSVC)
    # The sources are UTF-8 with non-ASCII punctuation in comments and messages.
    target_compile_options(vsfeel PRIVATE /arch:AVX2 /utf-8 /EHsc)
else()
    target_compile_options(vsfeel PRIVATE -mavx2)
    # Keep volk's vk* function pointers and everything else internal; the plugin
    # only needs to export VapourSynthPluginInit2 (VS_EXTERNAL_API marks it).
    target_compile_options(vsfeel PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()
# Build the plugin with a real warning baseline so defects do not accumulate.
# -Wmissing-field-initializers is deliberately excluded: vsfeel.h silences it
# for the Vulkan/VapourSynth designated-initializer convention.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vsfeel PRIVATE
        -Wall -Wextra -Wshadow -Wno-missing-field-initializers)
endif()
# Keep the host's EEDI3 vcheck LDS fit check in step with the -DMAXW the shader
# was compiled with (defined above, in the EEDI3 shader block).
target_compile_definitions(vsfeel PRIVATE EEDI3_MAXW_LDS=${EEDI3_MAXW})
if(WIN32 AND NOT MSVC)
    # Link the MinGW runtime statically: a plugin that needs libstdc++-6.dll,
    # libgcc_s_seh-1.dll and libwinpthread-1.dll is not loadable by a stock
    # VapourSynth install. -s drops the symbol table from the release DLL.
    target_link_options(vsfeel PRIVATE -static-libgcc -static-libstdc++ -static -s)
endif()
target_link_libraries(vsfeel PRIVATE volk::volk)
target_include_directories(vsfeel PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# Config header
configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# VapourSynth discovery & installation
find_package(PkgConfig QUIET MODULE)

# Packaging override: when set (e.g. by hatch_build.py), artifacts are
# installed to this relative directory instead of the live VapourSynth
# plugin directory, regardless of whether vapoursynth is importable.
# STRING, not PATH: PATH cache entries are canonicalized to absolute,
# which would bypass CMAKE_INSTALL_PREFIX at install time.
set(VSFEEL_INSTALL_DIR "" CACHE STRING "Relative plugin staging directory")

set(VS_PLUGIN_DIR "")

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)
endif()

if(VS_FOUND)
    message(STATUS "Found VapourSynth R${VS_VERSION}")
    target_include_directories(vsfeel PRIVATE ${VS_INCLUDE_DIRS})
else()
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers (containing VapourSynth4.h)")
    if(NOT VS_INCLUDE_DIR)
        # No pkg-config, which is the norm on Windows and in manylinux build
        # containers. The vapoursynth package ships its own CLI that knows where
        # its headers are, and it is on PATH in an isolated build environment.
        execute_process(
            COMMAND vapoursynth get-include
            OUTPUT_VARIABLE _vs_cli_include
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_vs_cli_include AND EXISTS "${_vs_cli_include}/VapourSynth4.h")
            set(VS_INCLUDE_DIR "${_vs_cli_include}" CACHE PATH
                "Path to VapourSynth headers (containing VapourSynth4.h)" FORCE)
        endif()
    endif()
    if(NOT VS_INCLUDE_DIR)
        # Fallback for when that CLI is not on PATH: ask the interpreter CMake
        # found, though it may not be the one the build environment installed
        # the headers into.
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import os, vapoursynth; print(os.path.join(os.path.dirname(vapoursynth.__file__), 'include'))"
            OUTPUT_VARIABLE _vs_py_include
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_vs_py_include AND EXISTS "${_vs_py_include}/VapourSynth4.h")
            set(VS_INCLUDE_DIR "${_vs_py_include}" CACHE PATH
                "Path to VapourSynth headers (containing VapourSynth4.h)" FORCE)
        endif()
    endif()
    if(NOT VS_INCLUDE_DIR)
        message(FATAL_ERROR
            "VapourSynth headers not found. Install the vapoursynth Python "
            "package or configure with -DVS_INCLUDE_DIR=<dir containing VapourSynth4.h>")
    endif()
    message(STATUS "Using VapourSynth headers from ${VS_INCLUDE_DIR}")
    target_include_directories(vsfeel PRIVATE "${VS_INCLUDE_DIR}")
endif()

# Where the running VapourSynth loads plugins from. Skipped when the packaging
# override is set: that staging directory is relative on purpose, so it has to
# go through CMAKE_INSTALL_PREFIX.
if(NOT VSFEEL_INSTALL_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import vapoursynth; print(vapoursynth.get_plugin_dir())"
        OUTPUT_VARIABLE VS_PLUGIN_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(STRIP "${VS_PLUGIN_DIR}" VS_PLUGIN_DIR)
endif()

# The packaging override wins over the live plugin directory, and RUNTIME is
# what installs the DLL on Windows (LIBRARY alone installs nothing there).
if(VSFEEL_INSTALL_DIR)
    set(_vsfeel_install_dir "${VSFEEL_INSTALL_DIR}")
elseif(VS_PLUGIN_DIR)
    set(_vsfeel_install_dir "${VS_PLUGIN_DIR}/vsfeel")
else()
    set(_vsfeel_install_dir "")
endif()

if(_vsfeel_install_dir)
    install(TARGETS vsfeel
            LIBRARY DESTINATION "${_vsfeel_install_dir}"
            RUNTIME DESTINATION "${_vsfeel_install_dir}")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" DESTINATION "${_vsfeel_install_dir}")
else()
    # Isolated build environments (pip/pdm-build) can import no vapoursynth
    # module; stage into <prefix>/lib|bin + <prefix>/ so packaging hooks can
    # collect the artifacts.
    install(TARGETS vsfeel LIBRARY RUNTIME)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" DESTINATION .)
endif()

# wo39 rebuild probe comment
