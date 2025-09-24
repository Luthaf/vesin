# This is not the actual CMakeLists.txt for this project, see
# `vesin/CMakeLists.txt` for it. Instead, this file is here to enable `cmake ..`
# from a git checkout, `add_subdirectory` and `FetchContent` without having to
# specify a subdirectory.

cmake_minimum_required(VERSION 3.16)

project(vesin-git LANGUAGES NONE)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(VESIN_MAIN_PROJECT ON)
else()
    set(VESIN_MAIN_PROJECT OFF)
endif()

if (VESIN_MAIN_PROJECT)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

option(VESIN_TORCH "Build the vesin_torch library" OFF)

set(VESIN_INSTALL ${VESIN_MAIN_PROJECT} CACHE BOOL "Install Vesin's headers and libraries")
add_subdirectory(vesin)

if (VESIN_TORCH)
    add_subdirectory(torch)
endif()
