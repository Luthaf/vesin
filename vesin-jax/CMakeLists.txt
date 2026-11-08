cmake_minimum_required(VERSION 3.16)

file(READ "vesin/VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

project(vesin-jax LANGUAGES CXX VERSION ${VESIN_VERSION})

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

# Directory containing the XLA FFI headers (`xla/ffi/api/ffi.h`), these are
# distributed with jaxlib. By default, we ask the jax installation of the
# current Python interpreter for them.
set(XLA_FFI_INCLUDE_DIR "" CACHE PATH "Directory containing the XLA FFI headers")
if ("${XLA_FFI_INCLUDE_DIR}" STREQUAL "")
    find_package(Python REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir())"
        RESULT_VARIABLE XLA_FFI_INCLUDE_DIR_STATUS
        OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT XLA_FFI_INCLUDE_DIR_STATUS EQUAL 0)
        message(FATAL_ERROR "could not find the XLA FFI headers, is jax installed?")
    endif()
endif()
message(STATUS "Using XLA FFI headers from ${XLA_FFI_INCLUDE_DIR}")

add_subdirectory(vesin)

add_library(vesin_jax SHARED src/vesin_jax.cpp)
target_link_libraries(vesin_jax PRIVATE vesin_objects gpulite)
# the XLA headers are external code, we don't want to see their warnings
target_include_directories(vesin_jax SYSTEM PRIVATE ${XLA_FFI_INCLUDE_DIR})
target_compile_features(vesin_jax PRIVATE cxx_std_17)

set_target_properties(vesin_jax PROPERTIES
    # hide non-exported symbols by default, the FFI handlers are explicitly
    # exported in the code
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS vesin_jax
    ARCHIVE DESTINATION "lib"
    LIBRARY DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)
