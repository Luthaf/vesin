# gpuLite

A lightweight C++ library for dynamic CUDA runtime compilation and kernel caching. gpuLite simplifies building and deploying CUDA-dependent applications by providing runtime symbol resolution and automated kernel compilation with caching.

**🚀 No CUDA SDK Required at Build Time!** - Compile your applications without installing the CUDA SDK.

## Features

- **No Build-Time CUDA Dependencies**: Compiles without CUDA SDK installed - only requires C++17 compiler
- **Dynamic Symbol Resolution**: Loads CUDA libraries (libcuda.so, libcudart.so, libnvrtc.so) at runtime
- **Runtime Compilation**: Compiles CUDA kernels using NVRTC with automatic compute capability detection
- **Kernel Caching**: Intelligent caching system to avoid recompilation of identical kernels
- **Easy Integration**: Header-only design for simple project integration
- **Cross-Platform Support**: Currently supports Linux (with plans for additional platforms)

## Why gpuLite?

Traditional CUDA applications require the CUDA SDK to be installed at build time and often have complex deployment requirements. gpuLite solves this by:

1. **Eliminating build-time CUDA dependencies** - No need for CUDA SDK during compilation
2. **Simplifying deployment** - Applications can run on any system with CUDA drivers installed
3. **Reducing compilation overhead** - Kernels are compiled once and cached for subsequent runs
4. **Providing runtime flexibility** - Kernels can be modified or optimized at runtime

## Quick Start

### Basic Usage

```cpp
#include "dynamic_cuda.hpp"
#include "cuda_cache.hpp"

int main() {
    // Check if CUDA is available
    if (!CUDA_DRIVER_INSTANCE.loaded() || !NVRTC_INSTANCE.loaded()) {
        throw std::runtime_error("CUDA runtime not available");
    }

    // Your CUDA kernel code as a string
    const char* kernel_source = R"(
        extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    )";

    // Create kernel factory and cache the kernel
    auto& factory = KernelFactory::instance();
    auto* kernel = factory.create(
        "vector_add",           // kernel name
        kernel_source,          // kernel source code
        "vector_add.cu",        // virtual source filename
        {"-std=c++17"}          // compilation options
    );

    // Allocate device memory and launch kernel
    float *d_a, *d_b, *d_c;
    int n = 1024;
    
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_a), n * sizeof(float)));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_b), n * sizeof(float)));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_c), n * sizeof(float)));

    // Prepare kernel arguments
    void* d_a_ptr = static_cast<void*>(d_a);
    void* d_b_ptr = static_cast<void*>(d_b);
    void* d_c_ptr = static_cast<void*>(d_c);
    std::vector<void*> args = {&d_a_ptr, &d_b_ptr, &d_c_ptr, const_cast<void*>(static_cast<const void*>(&n))};
    
    // Launch kernel
    kernel->launch(
        dim3((n + 255) / 256),  // grid size
        dim3(256),              // block size
        0,                      // shared memory size
        nullptr,                // stream
        args,                   // kernel arguments
        true                    // synchronize after launch
    );

    // Clean up
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_a));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_b));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_c));

    return 0;
}
```

**Compilation:**

```bash
# Save the above code as main.cpp, then compile:
g++ -std=c++17 main.cpp -ldl -o my_gpu_app

# Run the application:
./my_gpu_app
```

**Requirements:**
- C++17 compatible compiler (GCC 7+, Clang 5+)
- CUDA SDK installed at run-time, but not at build time!
- Linux

### Loading Kernels from Files

```cpp
// Load kernel from a .cu file
auto* kernel = KernelFactory::instance().createFromSource(
    "my_kernel",
    "/path/to/kernel.cu",
    "kernel.cu",
    {"-std=c++17"}
);
```

### Template Kernel Names

For templated kernels, use the `getKernelName` helper:

```cpp
// For a templated kernel like: template<typename T> __global__ void process(T* data)
std::string kernel_name = getKernelName<float>("process");
auto* kernel = factory.create(kernel_name, source, "template_kernel.cu", {});
```

## CMake Integration

### Method 1: Header-Only Integration

Add gpuLite as a subdirectory to your project:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Add gpuLite headers
add_subdirectory(external/gpuLite)

# Your executable
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE gpuLite)

# Link system libraries required by gpuLite
target_link_libraries(my_app PRIVATE ${CMAKE_DL_LIBS})
```

Create a simple CMakeLists.txt in the gpuLite directory:

```cmake
# gpuLite/CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(gpuLite LANGUAGES CXX)

# Create header-only interface library
add_library(gpuLite INTERFACE)

# Include directories
target_include_directories(gpuLite INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# C++17 requirement
target_compile_features(gpuLite INTERFACE cxx_std_17)

# System dependencies
find_package(Threads REQUIRED)
target_link_libraries(gpuLite INTERFACE 
    ${CMAKE_DL_LIBS}
    Threads::Threads
)

# Installation rules
install(TARGETS gpuLite EXPORT gpuLiteConfig)
install(FILES dynamic_cuda.hpp cuda_cache.hpp DESTINATION include)
install(EXPORT gpuLiteConfig DESTINATION lib/cmake/gpuLite)
```

### Method 2: Manual Integration

Simply copy the header files to your project and include them:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Your executable
add_executable(my_app 
    main.cpp
    # Copy these headers to your project
    gpuLite/dynamic_cuda.hpp
    gpuLite/cuda_cache.hpp
)

# Link required system libraries
target_link_libraries(my_app PRIVATE ${CMAKE_DL_LIBS})
```

### Method 3: Using FetchContent

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyProject LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    gpuLite
    GIT_REPOSITORY https://github.com/nickjbrowning/gpuLite.git
    GIT_TAG main
)

FetchContent_MakeAvailable(gpuLite)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE gpuLite)
```

## Advanced Usage

### Custom Compilation Options

```cpp
std::vector<std::string> options = {
    "-std=c++17",                   // C++ standard
    "--use_fast_math",              // Fast math operations
    "-DBLOCK_SIZE=256",             // Preprocessor definitions
    "--maxrregcount=32"             // Maximum register count
};

auto* kernel = factory.create("optimized_kernel", source, "kernel.cu", options);
```

### Kernel Function Attributes

```cpp
// Set maximum dynamic shared memory
kernel->setFuncAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8192);

// Get kernel register usage
int reg_count = kernel->getFuncAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
std::cout << "Kernel uses " << reg_count << " registers per thread" << std::endl;
```

### Asynchronous Execution

```cpp
// Create CUDA stream
cudaStream_t stream;
CUDART_SAFE_CALL(CUDART_INSTANCE.cudaStreamCreate(&stream));

// Launch kernel asynchronously
kernel->launch(
    grid, block, shared_mem_size, 
    reinterpret_cast<void*>(stream), 
    args, 
    false  // don't synchronize
);

// Do other work...

// Synchronize when needed
CUDART_SAFE_CALL(CUDART_INSTANCE.cudaStreamSynchronize(stream));
CUDART_SAFE_CALL(CUDART_INSTANCE.cudaStreamDestroy(stream));
```

## Error Handling

gpuLite provides comprehensive error checking with detailed error messages:

```cpp
try {
    auto* kernel = factory.create("my_kernel", source, "kernel.cu", {});
    kernel->launch(grid, block, 0, nullptr, args);
} catch (const std::runtime_error& e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    // Error message includes file and line information
}
```

## Requirements

### Runtime Requirements
- NVIDIA GPU with CUDA capability 3.0 or higher
- NVIDIA CUDA drivers installed
- Linux operating system (currently supported)

### Build Requirements (No CUDA SDK Needed!)
- C++17 compatible compiler (GCC 7+, Clang 5+)
- CMake 3.12 or higher
- Standard C++ library with threading support
- **No CUDA SDK installation required** - gpuLite uses minimal type wrappers

### CUDA Libraries (loaded dynamically at runtime)
- `libcuda.so` - CUDA Driver API
- `libcudart.so` - CUDA Runtime API  
- `libnvrtc.so` - NVIDIA Runtime Compilation
- **Note**: These libraries are only required at runtime, provided by NVIDIA drivers

## Platform Support

| Platform | Status |
|----------|--------|
| Linux    | ✅ Supported |
| Windows  | 🚧 Planned |
| macOS    | ❌ Not applicable (no CUDA support) |

## CUDA/HIP Support

| Platform | Status |
|----------|--------|
| CUDA    | ✅ Supported |
| HIP  | 🚧 Planned |

## Performance Considerations

- **First Launch**: Kernels are compiled on first use, which may add initial latency
- **Subsequent Launches**: Cached kernels launch immediately with minimal overhead
- **Memory Usage**: Compiled kernels are kept in memory for the application lifetime
- **Context Switching**: gpuLite automatically handles CUDA context management

## Troubleshooting

### "CUDA runtime not available" Error
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Verify CUDA libraries are in library path: `ldconfig -p | grep cuda`

### Compilation Errors
- Check kernel syntax using `nvcc` offline
- Verify compute capability compatibility
- Review compilation options for conflicts

### Runtime Errors
- Enable CUDA error checking in debug builds
- Use `cuda-gdb` and `compute-sanitizer` for kernel debugging
- Check memory alignment and access patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA CUDA Toolkit documentation
- NVRTC API reference
- Contributors and early adopters

---

For more examples and advanced usage patterns, see the `examples/` directory in this repository.