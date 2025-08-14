cmake_minimum_required(VERSION 3.12)
project(VectorAddExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(vector_add vector_add.cpp)

# Include gpuLite headers (assuming examples are in gpuLite/examples/)
target_include_directories(vector_add PRIVATE ../..)

# Link required system libraries
target_link_libraries(vector_add PRIVATE ${CMAKE_DL_LIBS})

# Set output directory
set_target_properties(vector_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)