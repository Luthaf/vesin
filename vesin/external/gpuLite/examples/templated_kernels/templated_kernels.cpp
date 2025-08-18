#include "../../cuda_cache.hpp"
#include "../../dynamic_cuda.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

template <typename T>
void run_templated_example(const std::string& type_name) {
    std::cout << "\n=== " << type_name << " Template Example ===" << std::endl;

    const int N = 1024 * 256;
    const size_t size = N * sizeof(T);

    // Initialize host data
    std::vector<T> h_input(N), h_output(N);

    // Fill with test data
    if constexpr (std::is_integral_v<T>) {
        for (int i = 0; i < N; i++) {
            h_input[i] = static_cast<T>(i % 100);
        }
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(1.0, 10.0);
        for (int i = 0; i < N; i++) {
            h_input[i] = dist(gen);
        }
    }

    // CUDA kernel template source - uses C++ template syntax
    std::string kernel_source = R"(
template<typename T>
__device__ T square(T x) {
    return x * x;
}

template<typename T>
__device__ T cube(T x) {
    return x * x * x;
}

extern "C" __global__ void process_array_)" +
                                type_name + R"(()" +
                                (std::is_same_v<T, float> ? "float" : std::is_same_v<T, double> ? "double"
                                                                  : std::is_same_v<T, int>      ? "int"
                                                                                                : "long long") +
                                R"(* input, )" +
                                (std::is_same_v<T, float> ? "float" : std::is_same_v<T, double> ? "double"
                                                                  : std::is_same_v<T, int>      ? "int"
                                                                                                : "long long") +
                                R"(* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        auto val = input[idx];
        // Apply some mathematical operations
        output[idx] = square(val) + cube(val) / (val + )" +
                                (std::is_integral_v<T> ? "1" : "1.0") + R"();
    }
}
)";

    try {
        // Allocate device memory
        T *d_input, *d_output;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_input), size));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_output), size));

        // Copy input data to device
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

        // Create templated kernel name using the helper function
        std::string kernel_name = "process_array_" + type_name;

        // Create and cache kernel
        auto& factory = KernelFactory::instance();
        std::cout << "Compiling " << type_name << " kernel: " << kernel_name << std::endl;

        auto compile_start = std::chrono::high_resolution_clock::now();
        auto* kernel = factory.create(
            kernel_name,                      // templated kernel name
            kernel_source,                    // kernel source code
            "templated_kernel.cu",            // virtual source filename
            {"-std=c++17", "--use_fast_math"} // compilation options
        );
        auto compile_end = std::chrono::high_resolution_clock::now();

        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start);
        std::cout << "Kernel compiled in: " << compile_time.count() << " ms" << std::endl;

        // Launch configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Prepare kernel arguments
        void* d_input_ptr = static_cast<void*>(d_input);
        void* d_output_ptr = static_cast<void*>(d_output);
        std::vector<void*> args = {&d_input_ptr, &d_output_ptr, const_cast<void*>(static_cast<const void*>(&N))};

        // Launch kernel
        auto kernel_start = std::chrono::high_resolution_clock::now();
        kernel->launch(
            dim3(blocksPerGrid),
            dim3(threadsPerBlock),
            0,
            nullptr,
            args,
            true
        );
        auto kernel_end = std::chrono::high_resolution_clock::now();

        auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
        std::cout << "Kernel executed in: " << kernel_time.count() << " μs" << std::endl;

        // Copy result back to host
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));

        // Verify results (spot check)
        bool success = true;
        for (int i = 0; i < std::min(10, N); i++) {
            T expected;
            if constexpr (std::is_integral_v<T>) {
                expected = h_input[i] * h_input[i] + (h_input[i] * h_input[i] * h_input[i]) / (h_input[i] + 1);
            } else {
                expected = h_input[i] * h_input[i] + (h_input[i] * h_input[i] * h_input[i]) / (h_input[i] + static_cast<T>(1.0));
            }

            T tolerance = std::is_integral_v<T> ? T(0) : static_cast<T>(1e-5);
            if (std::abs(h_output[i] - expected) > tolerance) {
                std::cout << "Error at index " << i << ": expected " << expected
                          << ", got " << h_output[i] << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "SUCCESS: " << type_name << " templated kernel executed correctly!" << std::endl;

            // Performance metrics
            double bandwidth = (2.0 * N * sizeof(T)) / (kernel_time.count() * 1e-6) / 1e9;
            std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(2)
                      << bandwidth << " GB/s" << std::endl;
        } else {
            std::cout << "FAILURE: " << type_name << " templated kernel produced incorrect results." << std::endl;
        }

        // Clean up
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_input));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_output));

    } catch (const std::exception& e) {
        std::cerr << "Error in " << type_name << " example: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Check if CUDA is available
        if (!CUDA_DRIVER_INSTANCE.loaded() || !NVRTC_INSTANCE.loaded() || !CUDART_INSTANCE.loaded()) {
            std::cout << "CUDA runtime libraries not available. Please install NVIDIA drivers." << std::endl;
            return 1;
        }

        std::cout << "Templated Kernels Example" << std::endl;
        std::cout << "This example demonstrates how to create type-specific kernels" << std::endl;
        std::cout << "for different data types using runtime compilation." << std::endl;

        // Run examples with different data types
        run_templated_example<float>("float");
        run_templated_example<double>("double");
        run_templated_example<int>("int");

        std::cout << "\n=== Advanced Template Example ===" << std::endl;

        // Example showing how to use the getKernelName helper for true C++ templates
        const char* template_kernel = R"(
template<typename T, int BLOCK_SIZE>
__global__ void reduction_sum(T* input, T* output, int n) {
    extern __shared__ T sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Perform reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Explicit instantiation for float with BLOCK_SIZE=256
template __global__ void reduction_sum<float, 256>(float*, float*, int);

extern "C" __global__ void reduction_sum_float_256(float* input, float* output, int n) {
    reduction_sum<float, 256>(input, output, n);
}
)";

        // This demonstrates how you would handle complex templated kernels
        // by using explicit instantiation and wrapper functions
        std::cout << "Complex template kernels require explicit instantiation" << std::endl;
        std::cout << "and wrapper functions for runtime compilation." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}