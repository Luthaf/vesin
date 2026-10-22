#include <cassert>
#include <cstdint>

#include <limits>
#include <stdexcept>
#include <string>

#include <gpulite/gpulite.hpp>

#include "./extra_data.hpp"
#include "./neighbors.hpp"
#include "./padding.hpp"

using namespace vesin::cuda;

#include "kernels/padding.cuh"
const uint8_t CUDA_PADDING_HEX[] = {
#include "generated/kernels/padding.cu.inc"
};
static const char* CUDA_PADDING_CODE = reinterpret_cast<const char*>(CUDA_PADDING_HEX);

#define VESIN_PADDING_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_PADDING_CODE, "padding.cu", {"-std=c++17", "-default-device"})

void vesin::cuda::reset_padding_overflow(VesinNeighborList& neighbors) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    // The copy only reads from the pinned memory when it executes, but nothing
    // else touches this field until `check_padding_overflow` synchronizes.
    extras->pinned->padding_overflow = std::numeric_limits<int32_t>::max();
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        extras->d_padding_overflow, &extras->pinned->padding_overflow,
        sizeof(int32_t), cudaMemcpyHostToDevice, nullptr
    ));
}

void vesin::cuda::check_padding_overflow(VesinNeighborList& neighbors, VesinOptions options) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->padding_overflow, extras->d_padding_overflow,
        sizeof(int32_t), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    auto point = extras->pinned->padding_overflow;
    if (point != std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(
            "point " + std::to_string(point) +
            " has more neighbors than the requested `padded_to` (" +
            std::to_string(options.padded_to) + ")"
        );
    }
}

void vesin::cuda::prepare_padding(
    VesinNeighborList& neighbors,
    size_t n_points,
    VesinOptions options
) {
    assert(options.padded_to > 0);
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    vesin::cuda::allocate_output_buffers(neighbors, n_points * options.padded_to, options);

    if (extras->pairs_per_point_capacity < n_points) {
        GPULITE_CUDART_CALL(cudaFree(extras->d_pairs_per_point));
        extras->d_pairs_per_point = nullptr;
        extras->pairs_per_point_capacity = 0;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_pairs_per_point, sizeof(int32_t) * n_points));
        extras->pairs_per_point_capacity = n_points;
    }

    GPULITE_CUDART_CALL(cudaMemset(extras->d_pairs_per_point, 0, sizeof(int32_t) * n_points));
    vesin::cuda::reset_padding_overflow(neighbors);
}

void vesin::cuda::finish_padding(
    VesinNeighborList& neighbors,
    size_t n_points,
    VesinOptions options
) {
    assert(options.padded_to > 0);
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    auto n_slots = n_points * options.padded_to;
    size_t threads = 256;
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3((n_slots + threads - 1) / threads);
    config.blockDim = dim3(threads);

    auto* kernel = VESIN_PADDING_KERNEL(factory, pad_neighbors);
    kernel->launch(
        config,
        extras->d_pairs_per_point,
        n_points,
        static_cast<int32_t>(options.padded_to),
        extras->d_padding_overflow,
        options.return_shifts,
        options.return_distances,
        options.return_vectors,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors
    );

    vesin::cuda::check_padding_overflow(neighbors, options);
    neighbors.length = n_slots;
}
