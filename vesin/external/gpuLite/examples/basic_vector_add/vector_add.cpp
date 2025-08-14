#include "../../dynamic_cuda.hpp"
#include "../../cuda_cache.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

int main() {
    try {
        // Check if CUDA is available
        if (!CUDA_DRIVER_INSTANCE.loaded() || !NVRTC_INSTANCE.loaded() || !CUDART_INSTANCE.loaded()) {
            std::cout << "CUDA runtime libraries not available. Please install the CUDA SDK and drivers." << std::endl;
            return 1;
        }

        // Vector size
        const int N = 1024 * 1024;
        const size_t size = N * sizeof(float);

        std::cout << "Vector Addition Example" << std::endl;
        std::cout << "Vector size: " << N << " elements" << std::endl;

        // Initialize host vectors
        std::vector<float> h_a(N), h_b(N), h_c(N);
        
        // Fill vectors with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        
        for (int i = 0; i < N; i++) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }

        // CUDA kernel source code
        const char* kernel_source = R"(
extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

        // Allocate device memory
        float *d_a, *d_b, *d_c;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_a), size));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_b), size));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(reinterpret_cast<void**>(&d_c), size));

        // Copy data to device
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));

        // Create and cache kernel
        auto& factory = KernelFactory::instance();
        std::cout << "Compiling kernel..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto* kernel = factory.create(
            "vector_add",           // kernel name
            kernel_source,          // kernel source code
            "vector_add.cu",        // virtual source filename
            {"-std=c++17"}          // compilation options
        );
        auto end = std::chrono::high_resolution_clock::now();
        
        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Kernel compiled in: " << compile_time.count() << " ms" << std::endl;

        // Launch configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "Launching kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

        // Prepare kernel arguments
        void* d_a_ptr = static_cast<void*>(d_a);
        void* d_b_ptr = static_cast<void*>(d_b);
        void* d_c_ptr = static_cast<void*>(d_c);
        std::vector<void*> args = {&d_a_ptr, &d_b_ptr, &d_c_ptr, const_cast<void*>(static_cast<const void*>(&N))};
        
        // Launch kernel and measure execution time
        start = std::chrono::high_resolution_clock::now();
        kernel->launch(
            dim3(blocksPerGrid),    // grid size
            dim3(threadsPerBlock),  // block size
            0,                      // shared memory size
            nullptr,                // stream (default)
            args,                   // kernel arguments
            true                    // synchronize after launch
        );
        end = std::chrono::high_resolution_clock::now();
        
        auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Kernel executed in: " << kernel_time.count() << " μs" << std::endl;

        // Copy result back to host
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));

        // Verify results
        std::cout << "Verifying results..." << std::endl;
        bool success = true;
        for (int i = 0; i < N; i++) {
            float expected = h_a[i] + h_b[i];
            if (std::abs(h_c[i] - expected) > 1e-5) {
                std::cout << "Error at index " << i << ": expected " << expected << ", got " << h_c[i] << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "SUCCESS: Vector addition completed correctly!" << std::endl;
            
            // Calculate performance metrics
            double bandwidth = (3.0 * N * sizeof(float)) / (kernel_time.count() * 1e-6) / 1e9;
            std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
        } else {
            std::cout << "FAILURE: Results do not match expected values." << std::endl;
        }

        // Clean up device memory
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_a));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_b));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(d_c));

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}