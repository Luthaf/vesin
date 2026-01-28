cmake_minimum_required(VERSION 3.16)

# if the vesin target already exists, do not create one again.
if (TARGET vesin)
    return()
endif()

file(READ "VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

project(vesin LANGUAGES C CXX VERSION ${VESIN_VERSION})

#[[
 This function wraps the input file with the appropriate string to allow us to
 static-initialize string variables, for example:
     static const char* CUDA_CODE =
#include "generated/cuda_impl.cu"
        ;
]]
function(make_includeable INPUT_FILE OUTPUT_FILE)
    if(NOT EXISTS ${INPUT_FILE})
        message(FATAL_ERROR "Error: The input file '${INPUT_FILE}' does not exist.")
    endif()
    file(STRINGS ${INPUT_FILE} content)

    file(WRITE ${OUTPUT_FILE} "")
    foreach(line IN LISTS content)
        # Write multiple raw string literals to workaound MSVC size limitations
        # on string literals. https://learn.microsoft.com/en-us/cpp/c-language/maximum-string-length
        file(APPEND ${OUTPUT_FILE} "R\"======(${line})======\"\n")
    endforeach()
endfunction()


if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(VESIN_MAIN_PROJECT ON)
else()
    set(VESIN_MAIN_PROJECT OFF)
endif()

if (VESIN_MAIN_PROJECT)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'Release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "Release"
            CACHE STRING
            "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug MinSizeRel None)
    endif()
endif()

option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)
option(VESIN_BUILD_TESTS "Build and run Vesin's unit tests" OFF)
option(VESIN_INSTALL "Install Vesin's headers and libraries" ${VESIN_MAIN_PROJECT})
option(VESIN_ENABLE_NVTX "Enable NVTX profiling markers" OFF)

set(VESIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vesin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_cell_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vesin_cuda.cpp
)

# create two targets: `vesin` is the main one, while `vesin_objects` will
# be used by language bindings to embed vesin inside another (SHARED) library
add_library(vesin ${VESIN_SOURCES})
add_library(vesin_objects OBJECT ${VESIN_SOURCES})

target_include_directories(vesin_objects PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Generate the CUDA source files for NVRTC compilation
make_includeable(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_bruteforce.cu ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_bruteforce.cu)
make_includeable(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_cell_list.cu ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_cell_list.cu)

include(FetchContent)
FetchContent_Declare(
    gpulite
    # GIT_REPOSITORY https://github.com/rubber-duck-debug/gpu-lite.git
    # GIT_TAG 78b4bad091e329b332def47ee9692e367a28ea85 # v1.0.0
    GIT_REPOSITORY https://github.com/Luthaf/gpu-lite.git
    GIT_TAG f59550e6eafdf51a3fd266a50ec080f640c0991c
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(gpulite)

target_link_libraries(vesin_objects PRIVATE gpulite)
target_link_libraries(vesin PRIVATE gpulite)

# Create generated directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_include_directories(vesin_objects PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(vesin PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# NVTX profiling support
if (VESIN_ENABLE_NVTX)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND AND TARGET CUDA::nvToolsExt)
        message(STATUS "Found NVTX via CUDAToolkit")
        target_compile_definitions(vesin PRIVATE VESIN_ENABLE_NVTX)
        target_compile_definitions(vesin_objects PRIVATE VESIN_ENABLE_NVTX)
        target_link_libraries(vesin PRIVATE CUDA::nvToolsExt)
    else()
        message(WARNING "VESIN_ENABLE_NVTX=ON but CUDA::nvToolsExt target not found")
    endif()
endif()

target_include_directories(vesin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(vesin_objects PRIVATE cxx_std_17)
set_target_properties(vesin_objects PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    EXCLUDE_FROM_ALL ON
)

target_compile_features(vesin PRIVATE cxx_std_17)
set_target_properties(vesin PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

if (BUILD_SHARED_LIBS)
    target_compile_definitions(vesin PUBLIC VESIN_SHARED)
    target_compile_definitions(vesin PRIVATE VESIN_EXPORTS)
endif()

if (VESIN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
if (VESIN_INSTALL)
    message(STATUS "Installing vesin target")
    install(TARGETS vesin
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
    )

    install(FILES "include/vesin.h" DESTINATION "include")
else()
    set_target_properties(vesin PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
