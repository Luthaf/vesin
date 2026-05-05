cmake_minimum_required(VERSION 3.16)

file(READ "vesin/VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

project(vesin-torch LANGUAGES CXX VERSION ${VESIN_VERSION})

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

option(VESIN_INSTALL "Install Vesin's headers and libraries" ${VESIN_MAIN_PROJECT})
option(VESIN_TORCH_BUILD_TESTS "Build and run the vesin-torch C++ tests" OFF)

# PyTorch requires passing std::string & co. across DLL boundaries, which fails
# when using a different runtime library from the one used by torch.dll & co.
# This must be set before `add_subdirectory(vesin)` so that all targets (and not
# only vesin_torch) are compiled against the same MSVC runtime library.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

add_subdirectory(vesin)

find_package(Torch 2.3 REQUIRED)

add_library(vesin_torch SHARED src/vesin_torch.cpp)

target_link_libraries(vesin_torch PRIVATE vesin_objects)
target_link_libraries(vesin_torch PUBLIC torch)

if (${Torch_VERSION} VERSION_GREATER_EQUAL 2.13)
    # Torch 2.13+ requires C++20
    target_compile_features(vesin_torch PUBLIC cxx_std_20)
else()
    target_compile_features(vesin_torch PUBLIC cxx_std_17)
endif()

set_target_properties(vesin_torch PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if (VESIN_TORCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
if (VESIN_INSTALL)
    message(STATUS "Installing vesin-torch target")
    install(TARGETS vesin_torch
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
    )
else()
    set_target_properties(vesin_torch PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
