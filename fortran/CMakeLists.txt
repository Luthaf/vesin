cmake_minimum_required(VERSION 3.16)

file(READ "vesin/VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

project(vesin-torch LANGUAGES Fortran CXX VERSION ${VESIN_VERSION})

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

add_subdirectory(vesin)

add_library(vesin_fortran src/vesin.f90)

target_link_libraries(vesin_fortran PRIVATE vesin_objects)
set_target_properties(vesin_fortran PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

target_include_directories(vesin_fortran PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (VESIN_BUILD_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()

if (VESIN_INSTALL)
    install(TARGETS vesin_fortran
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
    )

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/vesin.mod" DESTINATION "include")
else()
    set_target_properties(vesin_fortran PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
