cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0077)
    # Override options with variables
    cmake_policy(SET CMP0077 NEW)
endif()

project(vesin C CXX)

option(VESIN_BUILD_TESTS "build unit tests" OFF)

add_library(vesin
    src/vesin.cpp
    src/cpu_cell_list.cpp
)

target_include_directories(vesin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_compile_features(vesin PRIVATE cxx_std_17)

if (VESIN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
install(TARGETS vesin
    ARCHIVE DESTINATION "lib"
    LIBRARY DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)

install(DIRECTORY "include" DESTINATION "include")
