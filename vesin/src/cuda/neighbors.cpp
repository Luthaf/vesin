#include <cassert>
#include <cstdint>
#include <limits>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include <gpulite/gpulite.hpp>

#include "./bruteforce.hpp"
#include "./cell_list.hpp"
#include "./extra_data.hpp"
#include "./neighbors.hpp"
#include "./pick_algo.hpp"
#include "./verlet.hpp"

#include "kernels/bruteforce.cuh"

using namespace vesin::cuda;

static std::optional<cudaPointerAttributes> get_ptr_attributes(const void* ptr) {
    if (ptr == nullptr) {
        return std::nullopt;
    }

    try {
        cudaPointerAttributes attr;
        GPULITE_CUDART_CALL(cudaPointerGetAttributes(&attr, ptr));
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

static int32_t get_device_id(const void* ptr) {
    if (ptr == nullptr) {
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

static void free_output_buffers(VesinNeighborList& neighbors) {
    if ((neighbors.pairs != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.pairs), "pairs")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.pairs));
    }
    if ((neighbors.shifts != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.shifts), "shifts")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.shifts));
    }
    if ((neighbors.distances != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.distances), "distances")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.distances));
    }
    if ((neighbors.vectors != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.vectors), "vectors")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    GPULITE_CUDART_CALL(cudaSetDevice(neighbors.device.device_id));

    free_output_buffers(neighbors);

    delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
    neighbors.opaque = nullptr;
}

static void check_cuda_is_available() {
    std::string cuda_libname;
    std::string cudart_libname;
    std::string nvrtc_libname;
    std::string suggestion;
#if defined(__linux__)
    cuda_libname = "libcuda.so";
    cudart_libname = "libcudart.so(.*)";
    nvrtc_libname = "libnvrtc.so(.*)";
    suggestion = ("Try appending the directory containing this library to "
                  "your $LD_LIBRARY_PATH environment variable.");

#elif defined(_WIN32)
    cuda_libname = "nvcuda.dll";
    cudart_libname = "cudart64_*.dll";
    nvrtc_libname = "nvrtc64_*.dll";
    suggestion = ("Try adding the directory containing this library to your "
                  "system PATH, or making sure that CUDA_PATH is properly set "
                  "to your CUDA installation directory.");
#else
    cuda_libname = "cuda";
    cudart_libname = "cudart";
    nvrtc_libname = "nvrtc";
    suggestion = "Unsupported platform: unable to load CUDA libraries.";
#endif
    if (!gpulite::CUDADriver::loaded()) {
        throw std::runtime_error(
            "Failed to load " + cuda_libname + ". " + suggestion
        );
    }

    if (!gpulite::CUDART::loaded()) {
        throw std::runtime_error(
            "Failed to load " + cudart_libname + ". " + suggestion
        );
    }

    if (!gpulite::NVRTC::loaded()) {
        throw std::runtime_error(
            "Failed to load " + nvrtc_libname + ". " + suggestion
        );
    }
}

// Allocate the small, fixed-size scratch buffers held by `extras`. These are
// independent of the pair capacity and are only ever allocated once.
static void allocate_scratch_buffers(CudaNeighborListExtras* extras) {
    if (extras->d_length_ptr == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_length_ptr, sizeof(size_t)));
    }

    if (extras->pinned == nullptr) {
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras->pinned,
            sizeof(PinnedReadback),
            cudaHostAllocDefault
        ));
    }

    if (extras->d_all_periodic == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_all_periodic, sizeof(bool)));
    }

    if (extras->d_overflow_flag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_overflow_flag, sizeof(int32_t)));
    }

    if (extras->d_padding_overflow == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_padding_overflow, sizeof(int32_t)));
    }
}

void vesin::cuda::allocate_output_buffers(
    VesinNeighborList& neighbors,
    size_t n_pairs,
    VesinOptions options
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    allocate_scratch_buffers(extras);

    bool missing_requested_output =
        (options.return_shifts && neighbors.shifts == nullptr) ||
        (options.return_distances && neighbors.distances == nullptr) ||
        (options.return_vectors && neighbors.vectors == nullptr);

    if (extras->pairs_capacity >= n_pairs && !missing_requested_output) {
        return;
    }

    free_output_buffers(neighbors);

    size_t capacity = std::max(extras->pairs_capacity, n_pairs);
    extras->pairs_capacity = 0;

    GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * capacity * 2));

    if (options.return_shifts) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * capacity * 3));
    }

    if (options.return_distances) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.distances, sizeof(double) * capacity));
    }

    if (options.return_vectors) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.vectors, sizeof(double) * capacity * 3));
    }

    extras->pairs_capacity = capacity;
}

// Should this calculation use the cell list rather than the brute force
// algorithm? This throws if the user explicitly asked for the brute force
// algorithm, but it can not handle this system.
static bool use_cell_list(
    const BoxChecks& box_checks,
    size_t n_points,
    VesinOptions options,
    int32_t device_id
) {
    switch (options.algorithm) {
    case VesinBruteForce: {
        if (n_points > VESIN_MAX_BRUTE_FORCE_POINTS) {
            throw std::runtime_error(
                "too many points (" + std::to_string(n_points) + ") for the "
                "brute force algorithm (the limit is " + std::to_string(VESIN_MAX_BRUTE_FORCE_POINTS) + "); "
                "use the cell list instead"
            );
        }

        if (box_checks.n_images > VESIN_MAX_BRUTE_FORCE_IMAGES) {
            throw std::runtime_error(
                "the cutoff (" + std::to_string(options.cutoff) + ") is so large "
                "compared to the box that the brute force algorithm would have "
                "to check " + std::to_string(box_checks.n_images) + " periodic images of "
                "every pair of points (the limit is " +
                std::to_string(VESIN_MAX_BRUTE_FORCE_IMAGES) + "); use the cell list "
                "instead"
            );
        }

        return false;
    }
    case VesinCellList:
        return true;
    case VesinAutoAlgorithm:
    default:
        return vesin::cuda::auto_use_cell_list(
            static_cast<double>(n_points),
            static_cast<double>(box_checks.n_images),
            options.cutoff,
            options.sorted,
            device_id
        );
    }
}

void vesin::cuda::neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(neighbors.device.type == VesinCUDA);
    check_cuda_is_available();

    if (n_points == 0) {
        // Nothing to do, and the kernels below can not be launched with an
        // empty grid anyway.
        neighbors.length = 0;
        return;
    }

    if (n_points > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "Too many points (" + std::to_string(n_points) + ") for the CUDA "
            "neighbour list (limit is ~2.1 billion)."
        );
    }

    if (options.padded_to > 0) {
        // should already be checked
        assert(options.full);

        // The kernels use `int32_t` offsets in the output arrays
        auto max_padded_to = static_cast<size_t>(std::numeric_limits<int32_t>::max()) / n_points;
        if (options.padded_to > max_padded_to) {
            throw std::runtime_error(
                "`padded_to` (" + std::to_string(options.padded_to) + ") is too large "
                "for the CUDA neighbour list with " + std::to_string(n_points) +
                " points (the limit is 2^31 pairs in total)"
            );
        }
    }

    // check that all pointers are are device pointers
    if (!is_device_ptr(get_ptr_attributes(d_points), "points")) {
        throw std::runtime_error("`points` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(get_ptr_attributes(d_box), "box")) {
        throw std::runtime_error("`box` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(get_ptr_attributes(d_periodic), "periodic")) {
        throw std::runtime_error("`periodic` pointer is not allocated on a CUDA device");
    }

    auto device_id = get_device_id(d_points);
    if (device_id != get_device_id(d_box)) {
        throw std::runtime_error("`points` and `box` do not exist on the same device");
    }

    if (device_id != get_device_id(d_periodic)) {
        throw std::runtime_error("`points` and `periodic` do not exist on the same device");
    }

    if (device_id != neighbors.device.device_id) {
        throw std::runtime_error("`points`, `box` and `periodic` device differs from input neighbors device_id");
    }

    auto* extras = vesin::cuda::get_cuda_extras(neighbors);
    allocate_scratch_buffers(extras);

    if (options.skin > 0.0) {
        extras->verlet_cache.run(
            d_points, n_points, d_box, d_periodic, options, neighbors
        );
    } else {
        // no skin, free any remaining verlet buffers from a previous call
        extras->verlet_cache = VerletCache();

        auto box_checks = vesin::cuda::check_box(neighbors, d_box, d_periodic, options.cutoff);
        if (use_cell_list(box_checks, n_points, options, device_id)) {
            vesin::cuda::cell_list_neighbors(
                d_points,
                n_points,
                d_box,
                d_periodic,
                options,
                box_checks.all_periodic,
                neighbors
            );
        } else {
            vesin::cuda::brute_force_neighbors(
                d_points,
                n_points,
                d_box,
                d_periodic,
                options,
                neighbors
            );
        }
    }
}
