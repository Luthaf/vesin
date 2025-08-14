cmake_minimum_required(VERSION 3.12)
project(gpuLite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create header-only interface library
add_library(gpuLite INTERFACE)

# Include directories
target_include_directories(gpuLite INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# System dependencies
find_package(Threads REQUIRED)
target_link_libraries(gpuLite INTERFACE 
    ${CMAKE_DL_LIBS}
    Threads::Threads
)

# Build examples by default
option(BUILD_EXAMPLES "Build gpuLite examples" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples/basic_vector_add)
    add_subdirectory(examples/matrix_multiply)
    add_subdirectory(examples/templated_kernels)
endif()

# Installation rules
install(TARGETS gpuLite EXPORT gpuLiteConfig)
install(FILES 
    dynamic_cuda.hpp 
    cuda_cache.hpp 
    cuda_types_wrapper.hpp 
    DESTINATION include
)
install(EXPORT gpuLiteConfig DESTINATION lib/cmake/gpuLite)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gpuLiteConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gpuLiteConfigVersion.cmake"
    DESTINATION lib/cmake/gpuLite
)