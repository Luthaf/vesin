#include "vesin_cuda.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "cuda_cache.hpp"
#include "dynamic_cuda.hpp"

using namespace vesin::cuda;
using namespace std;

static std::optional<cudaPointerAttributes> getPtrAttributes(const void* ptr) {
    if (!ptr) {
        return std::nullopt;
    }

    try {
        cudaPointerAttributes attr;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaPointerGetAttributes(&attr, ptr));
        return attr;
    } catch (const std::runtime_error& e) {
        return std::nullopt;
    }
}

static bool is_device_ptr(const std::optional<cudaPointerAttributes>& maybe_attr, const char* name) {
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        return (attr.type == cudaMemoryTypeDevice);
    } else {
        throw std::runtime_error(
            "failed to resolve attributes for pointer: " + std::string(name)
        );
    }
}

static int get_device_id(const void* ptr) {
    if (!ptr) {
        return -1;
    }
    auto maybe_attr = getPtrAttributes(ptr);
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        if (attr.type != cudaMemoryTypeDevice) {
            return -1;
        }
        return attr.device;
    }
    return -1;
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    if (this->length_ptr) {
        CUDART_INSTANCE.cudaFree(this->length_ptr);
    }
    if (this->cell_check_ptr) {
        CUDART_INSTANCE.cudaFree(this->cell_check_ptr);
    }
}

vesin::cuda::CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    if (!neighbors->opaque) {
        neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
        vesin::cuda::CudaNeighborListExtras* test =
            static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
}

static void reset(VesinNeighborList& neighbors) {

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    if (neighbors.pairs && is_device_ptr(getPtrAttributes(neighbors.pairs), "pairs")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.pairs));
    }
    if (neighbors.shifts && is_device_ptr(getPtrAttributes(neighbors.shifts), "shifts")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.shifts));
    }
    if (neighbors.distances && is_device_ptr(getPtrAttributes(neighbors.distances), "distances")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.distances));
    }
    if (neighbors.vectors && is_device_ptr(getPtrAttributes(neighbors.vectors), "vectors")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
    extras->length_ptr = nullptr;

    *extras = CudaNeighborListExtras();
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {

    assert(neighbors.device.type == VesinCUDA);

    int curr_device = -1;
    int device_id = -1;

    if (neighbors.pairs) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaGetDevice(&curr_device));
        device_id = get_device_id(neighbors.pairs);

        if (device_id && curr_device != device_id) {
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device_id));
        }
    }

    reset(neighbors);

    if (device_id && curr_device != device_id) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(curr_device));
    }

    if (neighbors.opaque) {
        delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
        neighbors.opaque = nullptr;
    }
}

void checkCuda() {
    if (!CUDA_DRIVER_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load libcuda.so. Try appending the directory containing this library to "
            "your $LD_LIBRARY_PATH environment variable."
        );
    }

    if (!CUDART_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load libcudart.so. Try appending the directory containing this library to "
            "your $LD_LIBRARY_PATH environment variable."
        );
    }

    if (!NVRTC_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load libnvrtc.so. Try appending the directory containing this library to "
            "your $LD_LIBRARY_PATH environment variable."
        );
    }
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    static const char* CUDA_CODE =
#include "generated/mic_neighbourlist.cu"
        ;

    assert(neighbors.device.type == VesinCUDA);
    assert(!options.sorted && "Sorting is not supported in CUDA version of Vesin");

    // Check if CUDA is available
    checkCuda();

    // assert both points and box are device pointers
    assert(is_device_ptr(getPtrAttributes(points)) && "points pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(box)) && "box pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(periodic)) && "periodic pointer is not allocated on a CUDA device");

    int device = get_device_id(points);
    // assert both points and box are on the same device
    assert((device == get_device_id(box)) && "`points` and `box` do not exist on the same device");
    assert((device == get_device_id(periodic)) && "`points` and `periodic` do not exist on the same device");
    assert((device == neighbors.device.device_id) && "`points`, `box` and `periodic` device differs from input neighbors device_id");

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    // if allocated_device is different from the input device, we need to reset
    if (extras->allocated_device != device) {
        // first switch to previous device
        if (extras->allocated_device >= 0) {
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(extras->allocated_device));
        }
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device));
        extras->allocated_device = device;
    }

    // make sure the allocations can fit n_points
    if (extras->capacity >= n_points &&
        extras->length_ptr) {
        // allocation fits, so just memset set the length_ptr to 0
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int)));
    } else {
        // need a new allocation, so reset and reallocate
        reset(neighbors);
        auto max_pairs = static_cast<size_t>(1.2 * n_points * VESIN_CUDA_MAX_PAIRS_PER_POINT);

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * max_pairs * 2));

        if (options.return_shifts) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * max_pairs * 3)
            );
        }

        if (options.return_distances) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.distances, sizeof(double) * max_pairs)
            );
        }

        if (options.return_vectors) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.vectors, sizeof(double) * max_pairs * 3)
            );
        }

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        CUDART_SAFE_CALL(
            CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t))
        );

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->cell_check_ptr, sizeof(int)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int)));

        extras->capacity = static_cast<size_t>(1.2 * n_points);
    }

    const double* d_positions = reinterpret_cast<const double*>(points);
    const double* d_box = reinterpret_cast<const double*>(box);
    const bool* d_periodic = periodic;

    size_t* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    int* d_shifts = reinterpret_cast<int*>(neighbors.shifts);
    double* d_distances = reinterpret_cast<double*>(neighbors.distances);
    double* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    size_t* d_pair_counter = extras->length_ptr;
    int* d_cell_check = extras->cell_check_ptr;

    // Get or create kernel factory
    auto& factory = KernelFactory::instance();

    // First check box dimensions with mic_box_check kernel
    auto* box_check_kernel = factory.create(
        "mic_box_check",
        CUDA_CODE,
        "mic_neighbourlist.cu",
        {"-std=c++17"}
    );

    // Prepare arguments for box check kernel
    std::vector<void*> box_check_args = {&d_box, &d_periodic, &options.cutoff, &d_cell_check};

    // Launch box check kernel
    box_check_kernel->launch(
        dim3(1),         // grid size
        dim3(32),        // block size
        0,               // shared memory
        nullptr,         // stream
        box_check_args,  // arguments
        true             // synchronize
    );

    // Check box validity, assume fail
    int h_cell_check = 1;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_cell_check != 0) {
        throw std::runtime_error("Invalid cutoff: too large for box dimensions");
    }

    // Prepare arguments for neighbor computation kernel
    std::vector<void*> args = {
        &d_positions, &d_box, &d_periodic, &n_points, &options.cutoff, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
    };

    // Launch appropriate neighbor computation kernel
    const int WARP_SIZE = 32;
    const int NWARPS = 4;
    dim3 blockDim(WARP_SIZE * NWARPS);

    if (options.full) {
        // Full neighbor list kernel
        auto* full_kernel = factory.create(
            "compute_mic_neighbours_full_impl",
            CUDA_CODE,
            "mic_neighbourlist.cu",
            {"-std=c++17", "-DNWARPS=" + std::to_string(NWARPS), "-DWARP_SIZE=" + std::to_string(WARP_SIZE)}
        );

        dim3 gridDim(std::max((int)(n_points + NWARPS - 1) / NWARPS, 1));

        full_kernel->launch(gridDim, blockDim, 0, nullptr, args, false);

    } else {
        // Half neighbor list kernel
        auto* half_kernel = factory.create(
            "compute_mic_neighbours_half_impl",
            CUDA_CODE,
            "mic_neighbourlist.cu",
            {"-std=c++17", "-DNWARPS=" + std::to_string(NWARPS), "-DWARP_SIZE=" + std::to_string(WARP_SIZE)}
        );

        const size_t num_all_pairs = n_points * (n_points - 1) / 2;
        int threads_per_block = WARP_SIZE * NWARPS;
        int num_blocks = (num_all_pairs + threads_per_block - 1) / threads_per_block;
        dim3 gridDim(std::max(num_blocks, 1));

        half_kernel->launch(gridDim, blockDim, 0, nullptr, args, false);
    }

    // Copy final pair count back to host
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(&neighbors.length, d_pair_counter, sizeof(size_t), cudaMemcpyDeviceToHost));
}
