include(FetchContent)

# Override options with variables
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
)

set(CATCH_CONFIG_FAST_COMPILE ON)
FetchContent_MakeAvailable(Catch2)

# Make sure Catch2 uses the same C++ standard library ABI as torch (torch < 2.7
# on Linux uses `-D_GLIBCXX_USE_CXX11_ABI=0`), otherwise the tests can not link
# with Catch2.
separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(Catch2 PUBLIC ${TORCH_CXX_FLAGS_LIST})

# The models used by the tests are generated with Python, using the vesin-torch
# Python package, which must be installed in this Python environment.
find_package(Python COMPONENTS Interpreter REQUIRED)

set(VESIN_TORCH_TESTS_MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
add_test(
    NAME generate-models
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate-models.py ${VESIN_TORCH_TESTS_MODELS_DIR}
)
set_tests_properties(generate-models PROPERTIES FIXTURES_SETUP models)

# AOTInductor links the compiled models with the libraries in the Python library
# directory, without setting a rpath. When Python comes from conda, this
# includes conda's C++ standard library, which must then be available when
# loading the models outside of Python.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (APPLE)
    set(TESTS_ENVIRONMENT "DYLD_FALLBACK_LIBRARY_PATH=${PYTHON_LIBDIR}:/usr/local/lib:/usr/lib")
elseif (UNIX)
    set(TESTS_ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${PYTHON_LIBDIR}")
else()
    # The tests (and the models loaded by AOTInductor) link against torch.dll
    # & co., which are not in a directory searched by default on Windows. Add
    # their directory to PATH the same way `import torch` sets up the DLL
    # search paths in Python.
    set(TESTS_ENVIRONMENT "PATH=${TORCH_INSTALL_PREFIX}/lib\;$ENV{PATH}")
endif()

file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)

    if (${_name_} STREQUAL "aoti" AND ${Torch_VERSION} VERSION_LESS 2.7)
        # AOTInductor supports TorchBind objects since torch 2.7
        message(STATUS "Skipping AOTInductor tests with torch v${Torch_VERSION}")
        continue()
    endif()

    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} torch Catch2WithMain ${CMAKE_DL_LIBS})

    if (${Torch_VERSION} VERSION_GREATER_EQUAL 2.13)
        target_compile_features(${_name_} PRIVATE cxx_std_20)
    else()
        target_compile_features(${_name_} PRIVATE cxx_std_17)
    endif()

    # make sure string literals (i.e. the paths below) are encoded in UTF-8
    target_compile_options(${_name_} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    # vesin_torch does not export any symbol, the operators and classes are
    # registered when loading the library. The tests load it at runtime, the
    # same way `torch.ops.load_library` does in Python.
    add_dependencies(${_name_} vesin_torch)
    target_compile_definitions(${_name_} PRIVATE
        VESIN_TORCH_LIBRARY="$<TARGET_FILE:vesin_torch>"
        VESIN_TORCH_TESTS_MODELS_DIR="${VESIN_TORCH_TESTS_MODELS_DIR}"
    )

    add_test(NAME ${_name_} COMMAND $<TARGET_FILE:${_name_}>)
    set_tests_properties(${_name_} PROPERTIES
        FIXTURES_REQUIRED models
        ENVIRONMENT "${TESTS_ENVIRONMENT}"
    )
endforeach()
