#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <stdexcept>

#define NOMINMAX
#include <gpulite/gpulite.hpp>

#include "vesin_cuda.hpp"
#include "verlet_cuda.hpp"

using namespace vesin::cuda;

// NVTX for profiling (optional, enabled if available)
#ifdef VESIN_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) \
    do {                \
    } while (0)
#define NVTX_POP() \
    do {           \
    } while (0)
#endif

#include "cuda/verlet.cuh"
const unsigned char CUDA_VERLET_HEX[] = {
#include "generated/cuda/verlet.cu.inc"
};
const auto CUDA_VERLET_CODE = std::string(reinterpret_cast<const char*>(CUDA_VERLET_HEX), sizeof(CUDA_VERLET_HEX));

VerletCache::~VerletCache() {
    try {
        this->free_buffers();
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing VerletCache buffers: " << e.what() << std::endl;
    }
}

void VerletCache::free_buffers() {
    if (this->candidates.device.type == VesinCUDA) {
        vesin::cuda::free_neighbors(this->candidates);
    }

    this->candidates = VesinNeighborList();
    GPULITE_CUDART_CALL(cudaFree(this->d_ref_positions));
    GPULITE_CUDART_CALL(cudaFree(this->d_rebuild_flag));

    this->d_ref_positions = nullptr;
    this->d_rebuild_flag = nullptr;
    this->h_ref_capacity = 0;
    this->h_n_points = 0;
    this->h_options = VesinOptions();
    this->h_half_skin_sq = 0.0;
    this->h_has_cache = false;
}

bool VerletCache::options_changed(VesinOptions options) const {
    return this->h_options.cutoff != options.cutoff ||
           this->h_options.skin != options.skin ||
           this->h_options.full != options.full;
}

bool VerletCache::box_changed(const double h_box[9], const bool h_periodic[3]) const {
    for (size_t i = 0; i < 9; i++) {
        if (std::abs(this->h_ref_box[i] - h_box[i]) > 1e-12) {
            return true;
        }
    }

    for (size_t i = 0; i < 3; i++) {
        if (this->h_ref_periodic[i] != h_periodic[i]) {
            return true;
        }
    }

    return false;
}

void VerletCache::allocate_ref_buffers(size_t n_points) {
    if (this->h_ref_capacity < n_points) {
        GPULITE_CUDART_CALL(cudaFree(this->d_ref_positions));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_ref_positions, sizeof(double) * n_points * 3));
        this->h_ref_capacity = n_points;
    }

    if (this->d_rebuild_flag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_rebuild_flag, sizeof(int32_t)));
    }
}

void VerletCache::rebuild_cache(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    int32_t device_id,
    VesinOptions options,
    const double h_box[9],
    const bool h_periodic[3]
) {
    if (this->h_has_cache &&
        this->h_n_points == n_points &&
        !this->box_changed(h_box, h_periodic)) {

        auto& factory = gpulite::KernelFactory::instance(device_id);

        GPULITE_CUDART_CALL(cudaMemset(this->d_rebuild_flag, 0, sizeof(int32_t)));

        auto* kernel = factory.create<decltype(check_verlet_displacements)>(
            "check_verlet_displacements",
            CUDA_VERLET_CODE,
            "cuda_verlet.cu",
            {"-std=c++17", "-default-device"}
        );

        size_t threads = 256;
        size_t blocks = (n_points + threads - 1) / threads;

        auto config = gpulite::LaunchConfig();
        config.gridDim = dim3(std::max(blocks, static_cast<size_t>(1)));
        config.blockDim = dim3(threads);

        kernel->launch(
            config,
            reinterpret_cast<const double*>(points),
            this->d_ref_positions,
            n_points,
            this->h_half_skin_sq,
            this->d_rebuild_flag
        );

        int32_t h_rebuild = 0;
        GPULITE_CUDART_CALL(cudaMemcpy(
            &h_rebuild, this->d_rebuild_flag, sizeof(int32_t), cudaMemcpyDeviceToHost
        ));

        if (h_rebuild == 0) {
            return;
        }
    }

    if (this->candidates.device.type == VesinCUDA) {
        vesin::cuda::free_neighbors(this->candidates);
    }

    this->candidates = VesinNeighborList();
    this->candidates.device = {VesinCUDA, device_id};

    auto build_options = VesinOptions();
    build_options.cutoff = options.cutoff + options.skin;
    build_options.full = options.full;
    build_options.sorted = false;
    build_options.algorithm = VesinCellList;
    build_options.return_shifts = true;
    build_options.return_distances = false;
    build_options.return_vectors = false;
    build_options.skin = 0.0;

    vesin::cuda::neighbors(
        points,
        n_points,
        box,
        periodic,
        build_options,
        this->candidates
    );

    this->allocate_ref_buffers(n_points);
    GPULITE_CUDART_CALL(cudaMemcpy(
        this->d_ref_positions,
        points,
        sizeof(double) * n_points * 3,
        cudaMemcpyDeviceToDevice
    ));

    std::memcpy(this->h_ref_box, h_box, sizeof(double) * 9);
    std::memcpy(this->h_ref_periodic, h_periodic, sizeof(bool) * 3);
    this->h_n_points = n_points;
    this->h_options = options;
    this->h_half_skin_sq = (options.skin / 2.0) * (options.skin / 2.0);
    this->h_has_cache = true;
}


void VerletCache::filter_neighbors(
    const double (*d_positions)[3],
    size_t n_points,
    const double d_box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
) const {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    vesin::cuda::allocate_output_buffers(neighbors, this->candidates.length, options);

    auto* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    auto* d_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    auto* d_distances = neighbors.distances;
    auto* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    auto* d_pair_counter = extras->d_length_ptr;
    auto* d_candidate_pairs = reinterpret_cast<size_t*>(this->candidates.pairs);
    auto* d_candidate_shifts = reinterpret_cast<int32_t*>(this->candidates.shifts);

    size_t threads = 256;
    size_t blocks = (this->candidates.length + threads - 1) / threads;
    auto* kernel = factory.create<decltype(filter_verlet_candidates)>(
        "filter_verlet_candidates",
        CUDA_VERLET_CODE,
        "cuda_verlet.cu",
        {"-std=c++17", "-default-device"}
    );

    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3(std::max(blocks, static_cast<size_t>(1)));
    config.blockDim = dim3(threads);

    kernel->launch(
        config,
        reinterpret_cast<const double*>(d_positions),
        reinterpret_cast<const double*>(d_box),
        d_candidate_pairs,
        d_candidate_shifts,
        this->candidates.length,
        options.cutoff,
        d_pair_counter,
        d_pair_indices,
        d_shifts,
        d_distances,
        d_vectors,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );
}

void VerletCache::run(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto device_id = neighbors.device.device_id;

    if (this->options_changed(options)) {
        this->free_buffers();
    }

    double h_box[9];
    bool h_periodic[3];
    GPULITE_CUDART_CALL(cudaMemcpy(h_box, d_box, sizeof(double) * 9, cudaMemcpyDeviceToHost));
    GPULITE_CUDART_CALL(cudaMemcpy(h_periodic, d_periodic, sizeof(bool) * 3, cudaMemcpyDeviceToHost));

    this->rebuild_cache(
        d_points, n_points, d_box, d_periodic, device_id, options, h_box, h_periodic
    );

    this->filter_neighbors(d_points, n_points, d_box, options, neighbors);
}
