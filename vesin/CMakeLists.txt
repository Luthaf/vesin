cmake_minimum_required(VERSION 3.16)

file(READ "VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

project(vesin LANGUAGES C CXX VERSION ${VESIN_VERSION})

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

option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)
option(VESIN_BUILD_TESTS "Build and run Vesin's unit tests" OFF)
option(VESIN_INSTALL "Install Vesin's headers and libraries" ${VESIN_MAIN_PROJECT})
option(VESIN_TORCH "Build the vesin_torch library" OFF)

set(VESIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vesin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_cell_list.cpp
)
add_library(vesin ${VESIN_SOURCES})

target_include_directories(vesin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_compile_features(vesin PRIVATE cxx_std_17)

set_target_properties(vesin PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(vesin PRIVATE VESIN_EXPORTS)
if (BUILD_SHARED_LIBS)
    target_compile_definitions(vesin PUBLIC VESIN_SHARED)
endif()

if (VESIN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (VESIN_TORCH)
    add_subdirectory(torch)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
if (VESIN_INSTALL)
    install(TARGETS vesin
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
    )

    install(FILES "include/vesin.h" DESTINATION "include")
endif()
