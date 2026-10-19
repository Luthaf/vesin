cmake_minimum_required(VERSION 3.16)

file(READ "vesin/VERSION" VESIN_VERSION)
string(STRIP ${VESIN_VERSION} VESIN_VERSION)

project(vesin-jax LANGUAGES CXX VERSION ${VESIN_VERSION})

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
option(VESIN_JAX_BUILD_TESTS "Build and run the vesin-jax C++ tests" OFF)

add_subdirectory(vesin)

find_path(XLA_INCLUDE_DIR "xla/ffi/api/ffi.h")

if (NOT XLA_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find XLA_INCLUDE_DIR, please define it in cmake's options")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/check-xla.cpp"
    "#include \"xla/ffi/api/c_api.h\"\n"
    "#include \"xla/ffi/api/ffi.h\"\n"
    "int main() { return 0; }\n"
)

try_compile(CAN_COMPILE_XLA
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/check-xla.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${XLA_INCLUDE_DIR}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED true
    OUTPUT_VARIABLE TRY_COMPILE_OUTPUT
)

if (NOT CAN_COMPILE_XLA)
    message(FATAL_ERROR "Could not compile code using XLA, is XLA_INCLUDE_DIR correct?\n\n${TRY_COMPILE_OUTPUT}")
endif()

add_library(xla INTERFACE IMPORTED)
target_include_directories(xla INTERFACE ${XLA_INCLUDE_DIR})

add_library(vesin_jax SHARED src/vesin_jax.cpp)
target_compile_features(vesin_jax PUBLIC cxx_std_17)

target_link_libraries(vesin_jax PRIVATE vesin_objects)
target_link_libraries(vesin_jax PRIVATE xla)

set_target_properties(vesin_jax PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if (VESIN_JAX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
if (VESIN_INSTALL)
    message(STATUS "Installing vesin-jax target")
    install(TARGETS vesin_jax
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
    )
else()
    set_target_properties(vesin_jax PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
