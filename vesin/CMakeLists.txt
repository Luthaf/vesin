cmake_minimum_required(VERSION 3.16)

# if the vesin target already exists, do not create one again.
if (TARGET vesin)
    return()
endif()

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

set(VESIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vesin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_cell_list.cpp
)

# create two targets: `vesin` is the main one, while `vesin_objects` will
# be used by language bindings to embed vesin inside another (SHARED) library
add_library(vesin ${VESIN_SOURCES})
add_library(vesin_objects OBJECT ${VESIN_SOURCES})

target_include_directories(vesin_objects PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

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
