#include <cassert>
#include <cmath>
#include <cstdint>

#include <algorithm>
#include <stdexcept>
#include <string>

#include <gpulite/gpulite.hpp>

#include "./bruteforce.hpp"
#include "./extra_data.hpp"
#include "./neighbors.hpp"
#include "./padding.hpp"
#include "./sort_pairs.hpp"

using namespace vesin::cuda;

#include "kernels/bruteforce.cuh"
const uint8_t CUDA_BRUTEFORCE_HEX[] = {
#include "generated/kernels/bruteforce.cu.inc"
};
static const char* CUDA_BRUTEFORCE_CODE = reinterpret_cast<const char*>(CUDA_BRUTEFORCE_HEX);

#define VESIN_BRUTEFORCE_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_BRUTEFORCE_CODE, "bruteforce.cu", {"-std=c++17", "-default-device"})

BoxChecks vesin::cuda::check_box(
    VesinNeighborList& neighbors,
    const double d_box[3][3],
    const bool d_periodic[3],
    double cutoff
) {
    BoxChecks results;
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    if (extras->d_n_search == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_n_search, sizeof(int32_t) * 3));
    }
    if (extras->d_inv_box == nullptr) {
        double (*ptr)[3] = nullptr;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&ptr, sizeof(double[3][3])));
        extras->d_inv_box = ptr;
    }

    auto* box_check_kernel = VESIN_BRUTEFORCE_KERNEL(factory, analyze_box);

    // `analyze_box` is a single-thread kernel, there are only 18 numbers to
    // look at and everything downstream needs the answer before it can start
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);

    box_check_kernel->launch(
        config,
        d_box,
        d_periodic,
        cutoff,
        extras->d_all_periodic,
        extras->d_n_search,
        extras->d_inv_box
    );

    auto* pinned = extras->pinned;
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &pinned->all_periodic, extras->d_all_periodic,
        sizeof(bool), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        pinned->n_search, extras->d_n_search,
        sizeof(int32_t) * 3, cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    results.all_periodic = pinned->all_periodic;

    const int32_t* n_search = pinned->n_search;
    results.n_images = 1;
    for (size_t dim = 0; dim < 3; dim++) {
        results.n_images *= 2 * static_cast<size_t>(n_search[dim]) + 1;
    }

    return results;
}

// Run the brute force kernels, leaving the pairs they found in `neighbors`.
//
// Returns false when the output buffers were too small to hold them all. The
// exact number of pairs the search found will be in `neighbors.length`.
static bool run_brute_force(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(n_points <= VESIN_MAX_BRUTE_FORCE_POINTS);

    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    GPULITE_CUDART_CALL(cudaMemset(extras->d_length_ptr, 0, sizeof(size_t)));
    GPULITE_CUDART_CALL(cudaMemset(extras->d_overflow_flag, 0, sizeof(int32_t)));

    size_t THREADS_PER_BLOCK = 128;
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3((n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, n_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);

    if (options.full) {
        auto* kernel = VESIN_BRUTEFORCE_KERNEL(factory, brute_force_full);

        kernel->launch(
            config,
            d_points,
            d_box,
            extras->d_inv_box,
            d_periodic,
            extras->d_n_search,
            options.cutoff * options.cutoff,
            options.return_shifts,
            options.return_distances,
            options.return_vectors,
            extras->d_length_ptr,
            neighbors.pairs,
            neighbors.shifts,
            neighbors.distances,
            neighbors.vectors,
            extras->pairs_capacity,
            extras->d_overflow_flag,
            static_cast<int32_t>(options.padded_to),
            extras->d_pairs_per_point
        );
    } else {
        auto* kernel = VESIN_BRUTEFORCE_KERNEL(factory, brute_force_half);

        kernel->launch(
            config,
            d_points,
            d_box,
            extras->d_inv_box,
            d_periodic,
            extras->d_n_search,
            options.cutoff * options.cutoff,
            options.return_shifts,
            options.return_distances,
            options.return_vectors,
            extras->d_length_ptr,
            neighbors.pairs,
            neighbors.shifts,
            neighbors.distances,
            neighbors.vectors,
            extras->pairs_capacity,
            extras->d_overflow_flag
        );
    }

    if (options.padded_to > 0) {
        // The output is sorted by construction when padding, and always has
        // room for exactly `padded_to` pairs per point: points with more
        // neighbors make `finish_padding` throw instead.
        vesin::cuda::finish_padding(neighbors, n_points, options);
        return true;
    }

    // Read back the pair count from the GPU-side atomic counter, and the
    // overflow flag.
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->length,
        extras->d_length_ptr,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->overflow,
        extras->d_overflow_flag,
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    neighbors.length = extras->pinned->length;

    if (extras->pinned->overflow != 0) {
        return false;
    }

    // Brute-force output is unsorted; sort if requested
    if (options.sorted && neighbors.length > 1) {
        vesin::cuda::sort_pairs(n_points, options, neighbors);
    }

    return true;
}

void vesin::cuda::brute_force_neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    if (options.padded_to > 0) {
        // With padding, the output size is known in advance
        vesin::cuda::prepare_padding(neighbors, n_points, options);
        auto buffer_large_enough = run_brute_force(d_points, n_points, d_box, d_periodic, options, neighbors);
        assert(buffer_large_enough);
        (void)buffer_large_enough;
        return;
    }

    // Brute force writes its pairs through an atomic counter, and needs a
    // pre-allocated output. We guess a starting size here, and will grow it
    // below if needed
    size_t max_pairs_per_point = static_cast<size_t>(std::max(
        128.0, std::ceil(std::pow(options.cutoff, 3))
    ));

    vesin::cuda::allocate_output_buffers(
        neighbors, n_points * max_pairs_per_point, options
    );

    auto buffer_large_enough = run_brute_force(d_points, n_points, d_box, d_periodic, options, neighbors);

    if (!buffer_large_enough) {
        // The guess was too small, but the number of pairs is in
        // `neighbors.length`. We re-allocate the outputs with 10% headroom in
        // case the number of pairs fluctuate between now and the next call.
        auto needed = static_cast<size_t>(1.1 * static_cast<double>(neighbors.length));
        vesin::cuda::allocate_output_buffers(neighbors, needed, options);

        buffer_large_enough = run_brute_force(d_points, n_points, d_box, d_periodic, options, neighbors);

        if (!buffer_large_enough) {
            throw std::runtime_error(
                "the brute force search overflowed its output buffers "
                "twice in a row, the second time with room for " +
                std::to_string(needed) + " pairs against the " +
                std::to_string(neighbors.length) + " the first search "
                "counted. This should not happen, please report it as a bug"
            );
        }
    }
}
