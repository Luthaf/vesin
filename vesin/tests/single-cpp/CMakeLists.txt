cmake_minimum_required(VERSION 3.16)

project(test_vesin_single_build LANGUAGES C CXX)

add_library(vesin-single-build ${SINGLE_BUILD_DIR}/vesin-single-build.cpp)
add_library(vesin-single-build-nocuda ${SINGLE_BUILD_DIR}/vesin-single-build-nocuda.cpp)

target_include_directories(vesin-single-build PUBLIC ${SINGLE_BUILD_DIR})
target_include_directories(vesin-single-build-nocuda PUBLIC ${SINGLE_BUILD_DIR})

target_compile_features(vesin-single-build PUBLIC cxx_std_17)
target_compile_features(vesin-single-build-nocuda PUBLIC cxx_std_17)

add_executable(main main.cpp)
add_executable(main-nocuda main.cpp)

target_link_libraries(main vesin-single-build)
target_link_libraries(main-nocuda vesin-single-build-nocuda)
