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
#include "./utils.hpp"

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

// Free the output arrays currently in `neighbors`
static void free_output_buffers(VesinNeighborList& neighbors, cudaStream_t stream) {
    if ((neighbors.pairs != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.pairs), "pairs")) {
        GPULITE_CUDART_CALL(cudaFreeAsync(neighbors.pairs, stream));
    }

    if ((neighbors.shifts != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.shifts), "shifts")) {
        GPULITE_CUDART_CALL(cudaFreeAsync(neighbors.shifts, stream));
    }

    if ((neighbors.distances != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.distances), "distances")) {
        GPULITE_CUDART_CALL(cudaFreeAsync(neighbors.distances, stream));
    }

    if ((neighbors.vectors != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.vectors), "vectors")) {
        GPULITE_CUDART_CALL(cudaFreeAsync(neighbors.vectors, stream));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
}

// Free the output arrays allocated by vesin in previous calculations, in stream
// order on `stream`
static void free_allocated_outputs(CudaNeighborListExtras* extras, cudaStream_t stream) {
    GPULITE_CUDART_CALL(cudaFreeAsync(extras->allocated_outputs.pairs, stream));
    GPULITE_CUDART_CALL(cudaFreeAsync(extras->allocated_outputs.shifts, stream));
    GPULITE_CUDART_CALL(cudaFreeAsync(extras->allocated_outputs.distances, stream));
    GPULITE_CUDART_CALL(cudaFreeAsync(extras->allocated_outputs.vectors, stream));
    extras->allocated_outputs = vesin::OutputPointers();
    extras->pairs_capacity = 0;
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    if (neighbors.opaque == nullptr) {
        // vesin did not allocate anything for this neighbor list
        return;
    }

    auto device_guard = DeviceGuard(neighbors.device.device_id);

    auto* extras = static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);

    // The stream of the last call might no longer exist, so everything
    // below uses `cudaFree`. This does not wait for pending work using memory
    // from `cudaMallocAsync`, so we have to wait for the end of the last
    // calculation first.
    if (extras->last_call != nullptr) {
        GPULITE_CUDART_CALL(cudaEventSynchronize(extras->last_call));
    }

    // only release the arrays allocated by vesin, not the pre-allocated ones
    GPULITE_CUDART_CALL(cudaFree(extras->allocated_outputs.pairs));
    GPULITE_CUDART_CALL(cudaFree(extras->allocated_outputs.shifts));
    GPULITE_CUDART_CALL(cudaFree(extras->allocated_outputs.distances));
    GPULITE_CUDART_CALL(cudaFree(extras->allocated_outputs.vectors));

    delete extras;
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
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&extras->d_length_ptr, sizeof(size_t), extras->stream));
    }

    if (extras->pinned == nullptr) {
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras->pinned,
            sizeof(PinnedReadback),
            cudaHostAllocDefault
        ));
    }

    if (extras->d_all_periodic == nullptr) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&extras->d_all_periodic, sizeof(bool), extras->stream));
    }

    if (extras->d_overflow_flag == nullptr) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&extras->d_overflow_flag, sizeof(int32_t), extras->stream));
    }

    if (extras->d_padding_overflow == nullptr) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&extras->d_padding_overflow, sizeof(int32_t), extras->stream));
    }
}

void vesin::cuda::allocate_output_buffers(
    VesinNeighborList& neighbors,
    size_t n_pairs,
    VesinOptions options,
    size_t headroom
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    allocate_scratch_buffers(extras);

    if (extras->pre_allocated_outputs) {
        if (extras->pairs_capacity < n_pairs) {
            throw vesin::pre_allocated_too_small(extras->pairs_capacity, n_pairs);
        }
        return;
    }

    n_pairs += headroom;

    bool missing_requested_output =
        (options.return_shifts && neighbors.shifts == nullptr) ||
        (options.return_distances && neighbors.distances == nullptr) ||
        (options.return_vectors && neighbors.vectors == nullptr);

    if (extras->pairs_capacity >= n_pairs && !missing_requested_output) {
        return;
    }

    free_output_buffers(neighbors, extras->stream);

    size_t capacity = std::max(extras->pairs_capacity, n_pairs);
    extras->pairs_capacity = 0;

    GPULITE_CUDART_CALL(cudaMallocAsync((void**)&neighbors.pairs, sizeof(size_t) * capacity * 2, extras->stream));

    if (options.return_shifts) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&neighbors.shifts, sizeof(int32_t) * capacity * 3, extras->stream));
    }

    if (options.return_distances) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&neighbors.distances, sizeof(double) * capacity, extras->stream));
    }

    if (options.return_vectors) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&neighbors.vectors, sizeof(double) * capacity * 3, extras->stream));
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
    void* stream,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(neighbors.device.type == VesinCUDA);
    check_cuda_is_available();

    // Everything below (allocations, default streams, …) must happen on the
    // device of the neighbor list, regardless of the caller's current device.
    // We check below that this is also the device of the input data.
    auto device_guard = DeviceGuard(neighbors.device.device_id);

    auto* extras = vesin::cuda::get_cuda_extras(neighbors);
    extras->stream = static_cast<cudaStream_t>(stream);

    // The previous calculation with this neighbor list might have used a
    // different stream, and still be using the buffers in `extras` (or the
    // outputs) on it. Make sure the work on the current stream only starts once
    // the previous calculation is finished.
    if (extras->last_call == nullptr) {
        GPULITE_CUDART_CALL(cudaEventCreateWithFlags(&extras->last_call, cudaEventDisableTiming));
    } else {
        GPULITE_CUDART_CALL(cudaStreamWaitEvent(extras->stream, extras->last_call, cudaEventWaitDefault));
    }

    // Mark the end of this calculation on the current stream, for the next
    // calculation to wait on. This also runs if the calculation fails.
    struct LastCallGuard {
        CudaNeighborListExtras* extras;
        ~LastCallGuard() {
            // errors are ignored, since we can not throw from a destructor
            gpulite::CUDART::instance().cudaEventRecord(extras->last_call, extras->stream);
        }
    };
    auto last_call_guard = LastCallGuard{extras};

    vesin::check_output_arrays(neighbors, extras->allocated_outputs, options);

    if (options.pre_allocated_outputs) {
        // the arrays allocated by vesin in previous calls are no longer needed
        free_allocated_outputs(extras, extras->stream);
        // the capacity of the pre-allocated arrays is given by the caller
        extras->pairs_capacity = neighbors.length;
    } else if (extras->pre_allocated_outputs) {
        // the previous call used pre-allocated arrays, we have none of our own
        extras->pairs_capacity = 0;
    }
    extras->pre_allocated_outputs = options.pre_allocated_outputs;

    // Record the arrays allocated by vesin at the end of the calculation, even
    // if it fails
    struct RecordAllocatedOutputs {
        CudaNeighborListExtras* extras;
        VesinNeighborList& neighbors;
        ~RecordAllocatedOutputs() {
            if (!extras->pre_allocated_outputs) {
                extras->allocated_outputs = vesin::OutputPointers::from(neighbors);
            }
        }
    };
    auto record = RecordAllocatedOutputs{extras, neighbors};

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

    if (options.pre_allocated_outputs) {
        auto pointers = vesin::OutputPointers::from(neighbors);
        for (const void* ptr : {pointers.pairs, pointers.shifts, pointers.distances, pointers.vectors}) {
            if (ptr != nullptr && get_device_id(ptr) != device_id) {
                throw std::runtime_error(
                    "the pre-allocated output arrays in `neighbors` must be "
                    "allocated on CUDA device " + std::to_string(device_id)
                );
            }
        }
    }

    allocate_scratch_buffers(extras);

    if (options.skin > 0.0) {
        extras->verlet_cache.run(
            d_points, n_points, d_box, d_periodic, options, neighbors
        );
    } else {
        // no skin, free any remaining verlet buffers from a previous call
        extras->verlet_cache.free_buffers(extras->stream);

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
