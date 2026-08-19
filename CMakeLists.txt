cmake_minimum_required(VERSION 3.21)

project(VapourSynth-BilateralHIP LANGUAGES CXX)

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
    message(STATUS "VapourSynth-BilateralHIP ${VCS_TAG}")
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
# Each kernel is compiled for 8/16/32-bit staging formats.
set(VK_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/vk_spv")

set(VK_KERNEL_COMPONENTS shared plain)
set(VK_BITS 8 16 32)

if(VK_GLSLC)
    set(VK_SPV_OUTPUTS)
    foreach(component IN LISTS VK_KERNEL_COMPONENTS)
        foreach(bits IN LISTS VK_BITS)
            list(APPEND VK_SPV_OUTPUTS "${VK_SPV_DIR}/kernel_${component}_${bits}.spv")
            add_custom_command(
                OUTPUT "${VK_SPV_DIR}/kernel_${component}_${bits}.spv"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${VK_SPV_DIR}"
                COMMAND ${VK_GLSLC} -O -fshader-stage=compute -DBITS=${bits}
                        -o "${VK_SPV_DIR}/kernel_${component}_${bits}.spv"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_${component}.comp"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_${component}.comp"
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
    # large-path passes), per io variant (8/16 int, 16 half, 32 float)
    set(VK_GAUSS_ENTRIES gauss vert horiz)
    set(VK_GAUSS_BITDEFS "8" "16" "16h" "32")
    foreach(bits IN LISTS VK_GAUSS_BITDEFS)
        set(bits_defs)
        if(bits STREQUAL "16h")
            list(APPEND bits_defs -DBITS=16 -DHALF=1)
        else()
            list(APPEND bits_defs -DBITS=${bits})
        endif()
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
else()
    message(FATAL_ERROR "glslc not found")
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/gen_spirv_header.py"
            "${VK_SPV_DIR}/kernel_shared_8.spv"
            "${VK_SPV_DIR}/kernel_shared_16.spv"
            "${VK_SPV_DIR}/kernel_shared_32.spv"
            "${VK_SPV_DIR}/kernel_plain_8.spv"
            "${VK_SPV_DIR}/kernel_plain_16.spv"
            "${VK_SPV_DIR}/kernel_plain_32.spv"
            "${VK_SPV_DIR}/bm3d.spv"
            "${VK_SPV_DIR}/bm3d_agg.spv"
            "${VK_SPV_DIR}/gaussblur_8_gauss.spv"
            "${VK_SPV_DIR}/gaussblur_8_vert.spv"
            "${VK_SPV_DIR}/gaussblur_8_horiz.spv"
            "${VK_SPV_DIR}/gaussblur_16_gauss.spv"
            "${VK_SPV_DIR}/gaussblur_16_vert.spv"
            "${VK_SPV_DIR}/gaussblur_16_horiz.spv"
            "${VK_SPV_DIR}/gaussblur_16h_gauss.spv"
            "${VK_SPV_DIR}/gaussblur_16h_vert.spv"
            "${VK_SPV_DIR}/gaussblur_16h_horiz.spv"
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
            "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    DEPENDS ${VK_SPV_OUTPUTS}
    VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h" PROPERTIES GENERATED TRUE)

# Plugin
add_library(vsfeel SHARED
    src/vsfeel.cpp
    src/bilateral.cpp
    src/bm3d.cpp
    src/gaussblur.cpp
    src/dfttest.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h")
configure_common(vsfeel)
target_compile_options(vsfeel PRIVATE -mavx2)
target_link_libraries(vsfeel PRIVATE Vulkan::Vulkan)
target_include_directories(vsfeel PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# Config header
configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# VapourSynth discovery & installation
find_package(PkgConfig QUIET MODULE)

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)
    if(VS_FOUND)
        message(STATUS "Found VapourSynth R${VS_VERSION}")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import vapoursynth; print(vapoursynth.get_plugin_dir())"
            OUTPUT_VARIABLE VS_PLUGIN_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        target_include_directories(vsfeel PRIVATE ${VS_INCLUDE_DIRS})
        install(TARGETS vsfeel LIBRARY DESTINATION "${VS_PLUGIN_DIR}/vsfeel")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk/manifest.vs" DESTINATION "${VS_PLUGIN_DIR}/vsfeel")
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
