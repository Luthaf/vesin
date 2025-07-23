cmake_minimum_required(VERSION 3.16)

project(vesin-fortran Fortran)

add_library(vesin_fortran src/vesin.f90)

target_link_libraries(vesin_fortran PUBLIC vesin)
set_target_properties(vesin_fortran PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(vesin_fortran PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
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

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/vesin.mod" DESTINATION "include")
endif()
