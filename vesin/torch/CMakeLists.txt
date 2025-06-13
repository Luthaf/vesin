find_package(Torch 1.12 REQUIRED)

# PyTorch requires passing std::string & co. across DLL boundaries, which fails
# when using a different runtime library from the one used by torch.dll & co.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

add_library(vesin_torch SHARED src/vesin_torch.cpp ${VESIN_SOURCES})

target_link_libraries(vesin_torch torch)

target_include_directories(vesin_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include>
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
