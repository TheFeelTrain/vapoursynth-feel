cmake_minimum_required(VERSION 3.21)

project(vapoursynth-feel LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_program(VK_GLSLC glslc)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    if(VCS_TAG)
        string(STRIP "${VCS_TAG}" VCS_TAG)
    endif()
endif()

if(VCS_TAG)
    message(STATUS "vapoursynth-feel ${VCS_TAG}")
else()
    message(WARNING "unknown plugin version")
    set(VCS_TAG "unknown")
endif()

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
set(VK_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/vk_spv")

set(VK_KERNEL_COMPONENTS shared plain)
set(VK_BITS 16 32)

if(VK_GLSLC)
    set(VK_SPV_OUTPUTS)
    foreach(component IN LISTS VK_KERNEL_COMPONENTS)
        foreach(bits IN LISTS VK_BITS)
            list(APPEND VK_SPV_OUTPUTS "${VK_SPV_DIR}/bilateral_${component}_${bits}.spv")
            add_custom_command(
                OUTPUT "${VK_SPV_DIR}/bilateral_${component}_${bits}.spv"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} -O -fshader-stage=compute -DBITS=${bits}
                        -o "${VK_SPV_DIR}/bilateral_${component}_${bits}.spv"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/bilateral_${component}.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/bilateral_${component}.comp"
                VERBATIM)
        endforeach()
    endforeach()
    foreach(kernel IN ITEMS bm3d bm3d_agg)
        list(APPEND VK_SPV_OUTPUTS "${VK_SPV_DIR}/${kernel}.spv")
        add_custom_command(
            OUTPUT "${VK_SPV_DIR}/${kernel}.spv"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
            COMMAND ${VK_GLSLC} -O -fshader-stage=compute
                    -o "${VK_SPV_DIR}/${kernel}.spv"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/${kernel}.comp"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${kernel}.comp"
            VERBATIM)
    endforeach()
    # GaussBlur: one kernel with three entry points (fused small path + the two
    # large-path passes), per io variant (16 int, 32 float)
    set(VK_GAUSS_ENTRIES gauss vert horiz)
    set(VK_GAUSS_BITDEFS "16" "32")
    foreach(bits IN LISTS VK_GAUSS_BITDEFS)
        set(bits_defs -DBITS=${bits})
        foreach(entry IN LISTS VK_GAUSS_ENTRIES)
            set(defs ${bits_defs})
            if(entry STREQUAL gauss)
                list(APPEND defs -DENTRY_GAUSS)
            elseif(entry STREQUAL vert)
                list(APPEND defs -DENTRY_VERT)
            else()
                list(APPEND defs -DENTRY_HORIZ)
            endif()
            list(APPEND VK_SPV_OUTPUTS "${VK_SPV_DIR}/gaussblur_${bits}_${entry}.spv")
            add_custom_command(
                OUTPUT "${VK_SPV_DIR}/gaussblur_${bits}_${entry}.spv"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} -O -fshader-stage=compute ${defs}
                        -o "${VK_SPV_DIR}/gaussblur_${bits}_${entry}.spv"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/gaussblur.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gaussblur.comp"
                VERBATIM)
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
    set(VK_DFT_BITS "16" "32")
    foreach(bits IN LISTS VK_DFT_BITS)
        foreach(entry IN LISTS VK_DFT_ENTRIES)
            set(defs -DBITS=${bits})
            if(entry STREQUAL pad_slot)
                list(APPEND defs -DENTRY_PAD_SLOT)
            elseif(entry STREQUAL pad_direct)
                list(APPEND defs -DENTRY_PAD_DIRECT)
            else()
                list(APPEND defs -DENTRY_COL2IM)
            endif()
            set(raw_spv "${VK_SPV_DIR}/dfttest_${bits}_${entry}.spv")
            list(APPEND VK_SPV_OUTPUTS "${raw_spv}")
            add_custom_command(
                OUTPUT "${raw_spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute ${defs}
                        -o "${raw_spv}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/dfttest.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/dfttest.comp"
                VERBATIM)
        endforeach()
        foreach(radius RANGE 0 3)
            set(raw_spv "${VK_SPV_DIR}/dfttest_${bits}_fused_r${radius}.spv")
            list(APPEND VK_SPV_OUTPUTS "${raw_spv}")
            add_custom_command(
                OUTPUT "${raw_spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute -DBITS=${bits}
                        -DENTRY_FUSED -DRADIUS=${radius}
                        -o "${raw_spv}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/dfttest.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/dfttest.comp"
                VERBATIM)
        endforeach()
    endforeach()
    # NLMeans: weight / accumulation / finish / pad entry points per io
    # variant (16-bit int, 32-bit float); everything else is a specialization
    # constant.
    set(VK_NLM_ENTRIES weight acc finish pad)
    set(VK_NLM_BITS "16" "32")
    foreach(bits IN LISTS VK_NLM_BITS)
        foreach(entry IN LISTS VK_NLM_ENTRIES)
            set(defs -DBITS=${bits})
            if(entry STREQUAL weight)
                list(APPEND defs -DENTRY_WEIGHT)
            elseif(entry STREQUAL acc)
                list(APPEND defs -DENTRY_ACC)
            elseif(entry STREQUAL pad)
                list(APPEND defs -DENTRY_PAD)
            else()
                list(APPEND defs -DENTRY_FINISH)
            endif()
            set(raw_spv "${VK_SPV_DIR}/nlmeans_${bits}_${entry}.spv")
            list(APPEND VK_SPV_OUTPUTS "${raw_spv}")
            add_custom_command(
                OUTPUT "${raw_spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} -O -fshader-stage=compute ${defs}
                        -o "${raw_spv}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/nlmeans.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/nlmeans.comp"
                VERBATIM)
        endforeach()
    endforeach()
    # EEDI3: row kernel (one 32-lane WG per interp row, subgroup-register DP
    # with shuffles) + vcheck kernel (single WG serial row walk), per io
    # variant (16 int, 32 float). The row kernel needs subgroup ops + the
    # Vulkan memory model, so SPIR-V 1.3+ (vulkan1.3 target) is required.
    set(VK_EEDI3_ENTRIES row vcheck)
    set(VK_EEDI3_BITS "16" "32")
    foreach(bits IN LISTS VK_EEDI3_BITS)
        foreach(entry IN LISTS VK_EEDI3_ENTRIES)
            set(defs -DBITS=${bits})
            if(entry STREQUAL row)
                list(APPEND defs -DENTRY_ROW)
            else()
                list(APPEND defs -DENTRY_VCHECK)
            endif()
            set(raw_spv "${VK_SPV_DIR}/eedi3_${bits}_${entry}.spv")
            list(APPEND VK_SPV_OUTPUTS "${raw_spv}")
            add_custom_command(
                OUTPUT "${raw_spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute ${defs}
                        -o "${raw_spv}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/eedi3.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/eedi3.comp"
                VERBATIM)
        endforeach()
    endforeach()
    # EEDI3 upload kernels: mirror-pad builder + empty-row vcheck copy per io
    # variant (plain 1D kernels, fixed local size, no subgroup ops);
    # vulkan1.3 target keeps one rule for the file.
    set(VK_EEDI3_UP_ENTRIES pad vcopy)
    foreach(bits IN LISTS VK_EEDI3_BITS)
        foreach(entry IN LISTS VK_EEDI3_UP_ENTRIES)
            if(entry STREQUAL pad)
                set(entry_def -DENTRY_PAD)
            else()
                set(entry_def -DENTRY_VCOPY)
            endif()
            set(raw_spv "${VK_SPV_DIR}/eedi3_${bits}_${entry}.spv")
            list(APPEND VK_SPV_OUTPUTS "${raw_spv}")
            add_custom_command(
                OUTPUT "${raw_spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute -DBITS=${bits} ${entry_def}
                        -o "${raw_spv}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/eedi3.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/eedi3.comp"
                VERBATIM)
        endforeach()
    endforeach()
    # NNEDI3: clamp pad builder + fused main kernel per io variant (plain
    # kernels, no subgroup ops); vulkan1.3 target keeps one rule for the file.
    set(VK_NNEDI3_ENTRIES pad main)
    set(VK_NNEDI3_BITS "16" "32")
    foreach(bits IN LISTS VK_NNEDI3_BITS)
        foreach(entry IN LISTS VK_NNEDI3_ENTRIES)
            if(entry STREQUAL pad)
                set(entry_def -DENTRY_PAD)
            else()
                set(entry_def -DENTRY_MAIN)
            endif()
            set(raw_spv "${VK_SPV_DIR}/nnedi3_${bits}_${entry}.spv")
            list(APPEND VK_SPV_OUTPUTS "${raw_spv}")
            add_custom_command(
                OUTPUT "${raw_spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute -DBITS=${bits} ${entry_def}
                        -o "${raw_spv}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/nnedi3.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/nnedi3.comp"
                VERBATIM)
        endforeach()
    endforeach()
else()
    message(FATAL_ERROR "glslc not found")
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/gen_spirv_header.py"
            "${VK_SPV_DIR}/bilateral_shared_16.spv"
            "${VK_SPV_DIR}/bilateral_shared_32.spv"
            "${VK_SPV_DIR}/bilateral_plain_16.spv"
            "${VK_SPV_DIR}/bilateral_plain_32.spv"
            "${VK_SPV_DIR}/bm3d.spv"
            "${VK_SPV_DIR}/bm3d_agg.spv"
            "${VK_SPV_DIR}/gaussblur_16_gauss.spv"
            "${VK_SPV_DIR}/gaussblur_16_vert.spv"
            "${VK_SPV_DIR}/gaussblur_16_horiz.spv"
            "${VK_SPV_DIR}/gaussblur_32_gauss.spv"
            "${VK_SPV_DIR}/gaussblur_32_vert.spv"
            "${VK_SPV_DIR}/gaussblur_32_horiz.spv"
            "${VK_SPV_DIR}/dfttest_16_pad_slot.spv"
            "${VK_SPV_DIR}/dfttest_16_pad_direct.spv"
            "${VK_SPV_DIR}/dfttest_16_col2im.spv"
            "${VK_SPV_DIR}/dfttest_16_fused_r0.spv"
            "${VK_SPV_DIR}/dfttest_16_fused_r1.spv"
            "${VK_SPV_DIR}/dfttest_16_fused_r2.spv"
            "${VK_SPV_DIR}/dfttest_16_fused_r3.spv"
            "${VK_SPV_DIR}/dfttest_32_pad_slot.spv"
            "${VK_SPV_DIR}/dfttest_32_pad_direct.spv"
            "${VK_SPV_DIR}/dfttest_32_col2im.spv"
            "${VK_SPV_DIR}/dfttest_32_fused_r0.spv"
            "${VK_SPV_DIR}/dfttest_32_fused_r1.spv"
            "${VK_SPV_DIR}/dfttest_32_fused_r2.spv"
            "${VK_SPV_DIR}/dfttest_32_fused_r3.spv"
            "${VK_SPV_DIR}/nlmeans_16_weight.spv"
            "${VK_SPV_DIR}/nlmeans_16_acc.spv"
            "${VK_SPV_DIR}/nlmeans_16_finish.spv"
            "${VK_SPV_DIR}/nlmeans_16_pad.spv"
            "${VK_SPV_DIR}/nlmeans_32_weight.spv"
            "${VK_SPV_DIR}/nlmeans_32_acc.spv"
            "${VK_SPV_DIR}/nlmeans_32_finish.spv"
            "${VK_SPV_DIR}/nlmeans_32_pad.spv"
            "${VK_SPV_DIR}/eedi3_16_row.spv"
            "${VK_SPV_DIR}/eedi3_16_vcheck.spv"
            "${VK_SPV_DIR}/eedi3_32_row.spv"
            "${VK_SPV_DIR}/eedi3_32_vcheck.spv"
            "${VK_SPV_DIR}/eedi3_16_pad.spv"
            "${VK_SPV_DIR}/eedi3_32_pad.spv"
            "${VK_SPV_DIR}/eedi3_16_vcopy.spv"
            "${VK_SPV_DIR}/eedi3_32_vcopy.spv"
            "${VK_SPV_DIR}/nnedi3_16_pad.spv"
            "${VK_SPV_DIR}/nnedi3_16_main.spv"
            "${VK_SPV_DIR}/nnedi3_32_pad.spv"
            "${VK_SPV_DIR}/nnedi3_32_main.spv"
            "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    DEPENDS ${VK_SPV_OUTPUTS}
    VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h" PROPERTIES GENERATED TRUE)

# NNEDI3 weights: link the raw blob straight into the binary (objcopy turns
# it into an object exposing _binary_nnedi3_weights_bin_{start,end}). The
# blob stays out of C++ sources so the 13.5 MB never passes through the
# compiler; the wheel remains self-contained (reference/ is not shipped).
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
    "${CMAKE_CURRENT_BINARY_DIR}/nnedi3_weights.o")
configure_common(vsfeel)
target_compile_options(vsfeel PRIVATE -mavx2)
target_link_libraries(vsfeel PRIVATE Vulkan::Vulkan)
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

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)
    if(VS_FOUND)
        message(STATUS "Found VapourSynth R${VS_VERSION}")
        if(NOT VSFEEL_INSTALL_DIR)
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -c "import vapoursynth; print(vapoursynth.get_plugin_dir())"
                OUTPUT_VARIABLE VS_PLUGIN_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            string(STRIP "${VS_PLUGIN_DIR}" VS_PLUGIN_DIR)
        endif()

        target_include_directories(vsfeel PRIVATE ${VS_INCLUDE_DIRS})

        if(VSFEEL_INSTALL_DIR OR VS_PLUGIN_DIR)
            set(_vsfeel_install_dir "${VSFEEL_INSTALL_DIR}")
            if(NOT _vsfeel_install_dir)
                set(_vsfeel_install_dir "${VS_PLUGIN_DIR}/vsfeel")
            endif()
            install(TARGETS vsfeel LIBRARY DESTINATION "${_vsfeel_install_dir}")
            install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" DESTINATION "${_vsfeel_install_dir}")
        else()
            # Isolated build environments (pip/pdm-build) can import no
            # vapoursynth module; stage into <prefix>/lib|bin + <prefix>/
            # so packaging hooks can collect the artifacts.
            install(TARGETS vsfeel LIBRARY)
            install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" DESTINATION .)
        endif()
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")
    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(vsfeel PRIVATE ${VS_INCLUDE_DIR})
    install(TARGETS vsfeel LIBRARY RUNTIME)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" DESTINATION .)
endif()
