find_package(Torch 1.12 REQUIRED)

# PyTorch requires passing std::string & co. across DLL boundaries, which fails
# when using a different runtime library from the one used by torch.dll & co.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Find CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # gpuLite does not depend directly on the CUDAToolkit
    # so we do not need to enable_language(CUDA) or find_package(CUDAToolkit)
else()
    message(STATUS "Could not find a CUDA compiler")
endif()

add_library(vesin_torch SHARED src/vesin_torch.cpp ${VESIN_SOURCES})

if (CMAKE_CUDA_COMPILER AND VESIN_ENABLE_CUDA)
    # Include gpuLite headers
    target_include_directories(vesin_torch PRIVATE ../external/gpuLite)
    # Include the current binary directory for generated files
    target_include_directories(vesin_torch PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)
    # Link against the dynamic library loader
    target_link_libraries(vesin_torch PRIVATE ${CMAKE_DL_LIBS})

    # Create generated directory
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../generated)
    
    # Generate the CUDA source file for NVRTC compilation
    make_includeable(${CMAKE_CURRENT_SOURCE_DIR}/../src/mic_neighbourlist.cu ${CMAKE_CURRENT_BINARY_DIR}/../generated/mic_neighbourlist.cu)
endif()

target_link_libraries(vesin_torch PRIVATE torch)

target_include_directories(vesin_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_compile_features(vesin_torch PUBLIC cxx_std_17)

set_target_properties(vesin_torch PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
if (VESIN_INSTALL)
    install(TARGETS vesin_torch
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
    )

    install(FILES "include/vesin_torch.hpp" DESTINATION "include")
endif()
