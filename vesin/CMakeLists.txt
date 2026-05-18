cmake_minimum_required(VERSION 3.16)

# if the vesin target already exists, do not create one again.
if (TARGET vesin)
    return()
endif()

file(READ "VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(vesin LANGUAGES C CXX VERSION ${VESIN_VERSION})

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cluster_pair_search.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vesin_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/verlet.cpp
)

if (BUILD_VESIN_FOR_PYTHON)
    list(APPEND VESIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/shared_libraries.cpp)
endif()

# Generate the CUDA source files for NVRTC compilation
file(GLOB CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
foreach(cuda_source IN LISTS CUDA_SOURCES)
    get_filename_component(cuda_filename ${cuda_source} NAME)
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/generated/${cuda_filename}.inc")
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${cuda_source}
            -DOUTPUT_FILE=${output_file}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/make_hex_array.cmake
        DEPENDS ${cuda_source}
        COMMENT "Generating ${output_file}"
    )
    list(APPEND GENERATED_CUDA_HEX ${output_file})
endforeach()

add_custom_target(generate_cuda_hex DEPENDS ${GENERATED_CUDA_HEX})

# create two targets: `vesin` is the main one, while `vesin_objects` will
# be used by language bindings to embed vesin inside another (SHARED) library
add_library(vesin ${VESIN_SOURCES} ${GENERATED_CUDA_HEX})
add_library(vesin_objects OBJECT ${VESIN_SOURCES} ${GENERATED_CUDA_HEX})

target_include_directories(vesin_objects PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

include(FetchContent)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/gpulite.tar.gz)
    FetchContent_Declare(
        gpulite
        URL ${CMAKE_CURRENT_SOURCE_DIR}/external/gpulite.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP
        EXCLUDE_FROM_ALL
    )
else()
    FetchContent_Declare(
        gpulite
        GIT_REPOSITORY https://github.com/metatensor/gpu-lite.git
        GIT_TAG 7e8e9e929b254c3faefc556fb348068027e73319 # v1.1.1
        EXCLUDE_FROM_ALL
    )
endif()

FetchContent_MakeAvailable(gpulite)

# Google Highway for portable SIMD (used in cluster-pair search)
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.2.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# Force Highway to compile as a static library so libvesin.so is self-contained.
# Without this libvesin links libhwy.so.1 dynamically, which then has to be
# bundled by delocate/auditwheel when packaging Python wheels (it can't find
# the library and the build fails). Restore the saved value afterwards so
# vesin itself still honors the caller's BUILD_SHARED_LIBS choice.
set(_VESIN_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)
set(BUILD_SHARED_LIBS ${_VESIN_SAVED_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

target_link_libraries(vesin_objects PRIVATE gpulite hwy)
target_link_libraries(vesin PRIVATE gpulite hwy)
# Mark the SIMD code paths as available so the source can #ifdef around the
# Highway-using kernels and provide scalar fallbacks for builds without it
# (the single-file dist build does not have Highway on its include path).
target_compile_definitions(vesin_objects PRIVATE VESIN_HAVE_HIGHWAY)
target_compile_definitions(vesin PRIVATE VESIN_HAVE_HIGHWAY)

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
