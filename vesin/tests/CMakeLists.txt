include(FetchContent)

# Override options with variables
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
)

set(CATCH_CONFIG_FAST_COMPILE ON)
FetchContent_MakeAvailable(Catch2)

find_program(VALGRIND valgrind)
if (VALGRIND)
    message(STATUS "Running tests using valgrind")
    set(TEST_COMMAND
        "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
        "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
        "--gen-suppressions=all"
    )
else()
    set(TEST_COMMAND "")
endif()

find_package(CUDAToolkit)


file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} vesin Catch2WithMain)

    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
    )

    if (CUDAToolkit_FOUND)
        target_compile_definitions(${_name_} PRIVATE VESIN_TESTS_WITH_CUDA)
        target_link_libraries(${_name_} CUDA::cudart)
    endif()
endforeach()
