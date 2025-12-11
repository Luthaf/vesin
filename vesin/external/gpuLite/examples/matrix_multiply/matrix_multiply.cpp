#include "../../cuda_cache.hpp"
#include "../../dynamic_cuda.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
    try {
        // Check if CUDA is available
        if (!CUDA_DRIVER_INSTANCE.loaded() || !NVRTC_INSTANCE.loaded() || !CUDART_INSTANCE.loaded()) {
            std::cout << "CUDA runtime libraries not available. Please install NVIDIA drivers." << std::endl;
            return 1;
        }

        // Matrix dimensions (N x N matrices)
        const int N = 1024;
        const size_t size = N * N * sizeof(float);

        std::cout << "Matrix Multiplication Example" << std::endl;
        std::cout << "Matrix size: " << N << "x" << N << std::endl;

        // Initialize host matrices
        std::vector<float> h_a(N * N), h_b(N * N), h_c(N * N), h_c_ref(N * N);

        // Fill matrices with random data
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducible results
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < N * N; i++) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
            h_c[i] = 0.0f;
        }

        // CUDA kernel source code - optimized tiled matrix multiplication
        const char* kernel_source = R"(
#define TILE_SIZE 16

extern "C" __global__ void matrix_multiply(float* A, float* B, float* C, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate the row and column of C that this thread will compute
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        
        if (a_row < N && a_col < N) {
            As[ty][tx] = A[a_row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (b_row < N && b_col < N) {
            Bs[ty][tx] = B[b_row * N + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
)";

        // CPU reference implementation for verification
        std::cout << "Computing CPU reference..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += h_a[i * N + k] * h_b[k * N + j];
                }
                h_c_ref[i * N + j] = sum;
            }
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        std::cout << "CPU computation time: " << cpu_time.count() << " ms" << std::endl;

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

        auto compile_start = std::chrono::high_resolution_clock::now();
        auto* kernel = factory.create(
            "matrix_multiply",                // kernel name
            kernel_source,                    // kernel source code
            "matrix_multiply.cu",             // virtual source filename
            {"-std=c++17", "--use_fast_math"} // compilation options
        );
        auto compile_end = std::chrono::high_resolution_clock::now();

        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start);
        std::cout << "Kernel compiled in: " << compile_time.count() << " ms" << std::endl;

        // Launch configuration - using 16x16 thread blocks to match TILE_SIZE
        const int TILE_SIZE = 16;
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        std::cout << "Launching kernel with " << gridSize.x << "x" << gridSize.y
                  << " blocks of " << blockSize.x << "x" << blockSize.y << " threads" << std::endl;

        // Prepare kernel arguments
        void* d_a_ptr = static_cast<void*>(d_a);
        void* d_b_ptr = static_cast<void*>(d_b);
        void* d_c_ptr = static_cast<void*>(d_c);
        std::vector<void*> args = {&d_a_ptr, &d_b_ptr, &d_c_ptr, const_cast<void*>(static_cast<const void*>(&N))};

        // Warm up run
        kernel->launch(gridSize, blockSize, 0, nullptr, args, true);

        // Launch kernel and measure execution time
        auto gpu_start = std::chrono::high_resolution_clock::now();
        kernel->launch(gridSize, blockSize, 0, nullptr, args, true);
        auto gpu_end = std::chrono::high_resolution_clock::now();

        auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
        std::cout << "GPU kernel executed in: " << gpu_time.count() << " μs" << std::endl;

        // Copy result back to host
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));

        // Verify results
        std::cout << "Verifying results..." << std::endl;
        bool success = true;
        float max_error = 0.0f;

        for (int i = 0; i < N * N; i++) {
            float error = std::abs(h_c[i] - h_c_ref[i]);
            max_error = std::max(max_error, error);
            if (error > 1e-3) { // Relaxed tolerance for floating-point arithmetic
                std::cout << "Error at index " << i << ": expected " << h_c_ref[i] << ", got " << h_c[i]
                          << " (error: " << error << ")" << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "SUCCESS: Matrix multiplication completed correctly!" << std::endl;
            std::cout << "Maximum error: " << std::scientific << std::setprecision(2) << max_error << std::endl;

            // Calculate performance metrics
            double gflops = (2.0 * N * N * N) / (gpu_time.count() * 1e-6) / 1e9;
            double speedup = (double)cpu_time.count() / (gpu_time.count() * 1e-3);

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "GPU Performance: " << gflops << " GFLOPS" << std::endl;
            std::cout << "Speedup over CPU: " << speedup << "x" << std::endl;
        } else {
            std::cout << "FAILURE: Results do not match expected values." << std::endl;
            std::cout << "Maximum error: " << std::scientific << std::setprecision(2) << max_error << std::endl;
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