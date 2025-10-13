#include "vesin_cuda.hpp"
#include "mic_neighbourlist.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <optional>
#include <stdexcept>

using namespace vesin::cuda;
using namespace std;

#define CUDA_CHECK(expr)                                                                                                \
    do {                                                                                                                \
        cudaError_t err = (expr);                                                                                       \
        if (err != cudaSuccess) {                                                                                       \
            throw std::runtime_error(                                                                                   \
                std::string("CUDA error at " __FILE__ ":") + std::to_string(__LINE__) + " - " + cudaGetErrorString(err) \
            );                                                                                                          \
        }                                                                                                               \
    } while (0)

static std::optional<cudaPointerAttributes> get_ptr_attributes(const void* ptr) {
    if (!ptr) {
        return std::nullopt;
    }

    try {
        cudaPointerAttributes attr;
        CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
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
    auto maybe_attr = get_ptr_attributes(ptr);
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
        cudaFree(this->length_ptr);
    }
    if (this->cell_check_ptr) {
        cudaFree(this->cell_check_ptr);
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

    if (neighbors.pairs && is_device_ptr(get_ptr_attributes(neighbors.pairs), "pairs")) {
        CUDA_CHECK(cudaFree(neighbors.pairs));
    }
    if (neighbors.shifts && is_device_ptr(get_ptr_attributes(neighbors.shifts), "shifts")) {
        CUDA_CHECK(cudaFree(neighbors.shifts));
    }
    if (neighbors.distances && is_device_ptr(get_ptr_attributes(neighbors.distances), "distances")) {
        CUDA_CHECK(cudaFree(neighbors.distances));
    }
    if (neighbors.vectors && is_device_ptr(get_ptr_attributes(neighbors.vectors), "vectors")) {
        CUDA_CHECK(cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
    *extras = CudaNeighborListExtras();
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    int curr_device = -1;
    int device_id = -1;

    if (neighbors.pairs) {
        CUDA_CHECK(cudaGetDevice(&curr_device));
        device_id = get_device_id(neighbors.pairs);

        if (device_id && curr_device != device_id) {
            CUDA_CHECK(cudaSetDevice(device_id));
        }
    }

    reset(neighbors);

    if (device_id && curr_device != device_id) {
        CUDA_CHECK(cudaSetDevice(curr_device));
    }

    if (neighbors.opaque) {
        delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
        neighbors.opaque = nullptr;
    }
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double cell[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(neighbors.device.type == VesinCUDA);
    assert(!options.sorted && "Sorting is not supported in CUDA version of Vesin");

    // assert both points and cell are device pointers
    assert(is_device_ptr(get_ptr_attributes(points)) && "points pointer is not allocated on a CUDA device");

    int device = get_device_id(points);
    auto any_periodic = periodic[0] || periodic[1] || periodic[2];
    if (any_periodic) {
        assert(cell != nullptr && "periodic calculations require a non-null cell pointer");
        assert(is_device_ptr(get_ptr_attributes(cell)) && "cell pointer is not allocated on a CUDA device");
        // assert both points and cell are on the same device
        assert((device == get_device_id(cell)) && "points and cell pointers do not exist on the same device");
    }
    assert((device == neighbors.device.device_id) && "points and cell device differs from input neighbors device_id");

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    // if allocated_device is different from the input device, we need to reset
    if (extras->allocated_device != device) {
        // first switch to previous device
        if (extras->allocated_device >= 0) {
            CUDA_CHECK(cudaSetDevice(extras->allocated_device));
        }
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        CUDA_CHECK(cudaSetDevice(device));
        extras->allocated_device = device;
    }

    // make sure the allocations can fit n_points
    if (extras->capacity >= n_points && extras->length_ptr) {
        // allocation fits, so just memset set the length_ptr to 0
        CUDA_CHECK(cudaMemset(extras->length_ptr, 0, sizeof(extras->length_ptr)));
        CUDA_CHECK(cudaMemset(extras->cell_check_ptr, 0, sizeof(extras->cell_check_ptr)));
    } else {
        // need a new allocation, so reset and reallocate
        reset(neighbors);
        auto max_pairs = static_cast<size_t>(
            1.2 * n_points * VESIN_CUDA_MAX_PAIRS_PER_POINT
        );

        CUDA_CHECK(
            cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * max_pairs * 2)
        );

        if (options.return_shifts) {
            CUDA_CHECK(
                cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * max_pairs * 3)
            );
        }

        if (options.return_distances) {
            CUDA_CHECK(
                cudaMalloc((void**)&neighbors.distances, sizeof(double) * max_pairs)
            );
        }

        if (options.return_vectors) {
            CUDA_CHECK(
                cudaMalloc((void**)&neighbors.vectors, sizeof(double) * max_pairs * 3)
            );
        }

        CUDA_CHECK(cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        CUDA_CHECK(cudaMemset(extras->length_ptr, 0, sizeof(size_t)));

        CUDA_CHECK(cudaMalloc((void**)&extras->cell_check_ptr, sizeof(int)));
        CUDA_CHECK(cudaMemset(extras->cell_check_ptr, 0, sizeof(int)));

        extras->capacity = static_cast<size_t>(1.2 * n_points);
    }

    vesin::cuda::compute_mic_neighbourlist(
        points,
        n_points,
        any_periodic ? cell : nullptr,
        periodic,
        extras->cell_check_ptr,
        options,
        neighbors
    );
}
