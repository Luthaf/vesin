#include "vesin_cuda.hpp"
#include "mic_neighbourlist.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

using namespace vesin::cuda;
using namespace std;

#define WARP_SIZE 32
#define NWARPS 4

#define CUDA_CHECK(expr)                                                                                                                       \
    do {                                                                                                                                       \
        cudaError_t err = (expr);                                                                                                              \
        if (err != cudaSuccess) {                                                                                                              \
            throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        }                                                                                                                                      \
    } while (0)

static std::optional<cudaPointerAttributes> getPtrAttributes(const void* ptr) {
    if (!ptr)
        return std::nullopt;

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

vesin::cuda::CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    if (!neighbors->opaque) {
        neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
        vesin::cuda::CudaNeighborListExtras* test =
            static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
}

void reset(VesinNeighborList& neighbors) {

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    if (neighbors.pairs && is_device_ptr(getPtrAttributes(neighbors.pairs), "pairs")) {
        CUDA_CHECK(cudaFree(neighbors.pairs));
    }
    if (neighbors.shifts && is_device_ptr(getPtrAttributes(neighbors.shifts), "shifts")) {
        CUDA_CHECK(cudaFree(neighbors.shifts));
    }
    if (neighbors.distances && is_device_ptr(getPtrAttributes(neighbors.distances), "distances")) {
        CUDA_CHECK(cudaFree(neighbors.distances));
    }
    if (neighbors.vectors && is_device_ptr(getPtrAttributes(neighbors.vectors), "vectors")) {
        CUDA_CHECK(cudaFree(neighbors.vectors));
    }
    if (extras->length_ptr &&
        is_device_ptr(getPtrAttributes(extras->length_ptr), "extras->length_ptr")) {
        CUDA_CHECK(cudaFree(extras->length_ptr));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
    extras->length_ptr = nullptr;
    extras->capacity = 0;
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {

    assert(neighbors.device == VesinCUDA);

    int curr_device = -1, device_id = -1;

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

void vesin::cuda::neighbors(const double (*points)[3], long n_points, const double cell[3][3], VesinOptions options, VesinNeighborList& neighbors) {

    assert(neighbors.device == VesinCUDA);
    assert(!options.sorted && "Sorting is not supported in CUDA version of Vesin");

    // assert both points and cell are device pointers
    assert(is_device_ptr(getPtrAttributes(points)) && "points pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(cell)) && "cell pointer is not allocated on a CUDA device");

    int device = get_device_id(points);
    // assert both points and cell are on the same device
    assert((device == get_device_id(cell)) && "points and cell pointers do not exist on the same device");

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    // if allocated_device is different from the input device, we need to reset
    if (extras->allocated_device != device) {
        // first switch to previous device
        if (extras->allocated_device >= 0)
            CUDA_CHECK(cudaSetDevice(extras->allocated_device));
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        CUDA_CHECK(cudaSetDevice(device));
        extras->allocated_device = device;
    }

    // make sure the allocations can fit n_points
    if (extras->capacity >= n_points &&
        extras->length_ptr) {
        // allocation fits, so just memset set the length_ptr to 0
        CUDA_CHECK(cudaMemset(extras->length_ptr, 0, sizeof(unsigned long)));
    } else {
        // need a new allocation, so reset and reallocate
        reset(neighbors);
        unsigned long max_pairs =
            static_cast<unsigned long>(1.2 * n_points * VESIN_CUDA_MAX_PAIRS_PER_POINT);

        CUDA_CHECK(cudaMalloc((void**)&neighbors.pairs, sizeof(unsigned long) * max_pairs * 2));
        CUDA_CHECK(
            cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * max_pairs * 3)
        );
        CUDA_CHECK(
            cudaMalloc((void**)&neighbors.distances, sizeof(double) * max_pairs)
        );
        CUDA_CHECK(
            cudaMalloc((void**)&neighbors.vectors, sizeof(double) * max_pairs * 3)
        );

        CUDA_CHECK(cudaMalloc((void**)&extras->length_ptr, sizeof(unsigned long)));

        CUDA_CHECK(
            cudaMemset(extras->length_ptr, 0, sizeof(unsigned long))
        );

        extras->capacity = static_cast<unsigned long>(1.2 * n_points);
    }

    vesin::cuda::compute_mic_neighbourlist(points, n_points, cell, options, neighbors);
}