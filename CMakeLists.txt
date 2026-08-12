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
            "${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h"
    DEPENDS ${VK_SPV_OUTPUTS}
    VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/spirv_binaries.h" PROPERTIES GENERATED TRUE)

# Plugin
add_library(vsfeel SHARED
    src/vsfeel.cpp
    src/bilateral.cpp
    src/bm3d.cpp
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
