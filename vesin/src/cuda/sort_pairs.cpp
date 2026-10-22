#include <cstdint>
#include <iostream>

#include <limits>
#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "./extra_data.hpp"
#include "./scan.hpp"
#include "./sort_pairs.hpp"

#include "kernels/scan.cuh"

using namespace vesin::cuda;

#include "kernels/sort_pairs.cuh"
const uint8_t CUDA_SORT_PAIRS_HEX[] = {
#include "generated/kernels/sort_pairs.cu.inc"
};
static const char* CUDA_SORT_PAIRS_CODE = reinterpret_cast<const char*>(CUDA_SORT_PAIRS_HEX);

#define VESIN_SORT_PAIRS_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_SORT_PAIRS_CODE, "sort_pairs.cu", {"-std=c++17", "-default-device"})

SortBuffers::~SortBuffers() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_pairs_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_shifts_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_distances_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_vectors_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals2));
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
    }
}

SortBuffers::SortBuffers(SortBuffers&& other) noexcept:
    SortBuffers() {
    *this = std::move(other);
}

SortBuffers& SortBuffers::operator=(SortBuffers&& other) noexcept {
    if (this != &other) {
        this->~SortBuffers();

        this->capacity = other.capacity;
        other.capacity = 0;

        this->d_pairs_tmp = other.d_pairs_tmp;
        other.d_pairs_tmp = nullptr;

        this->d_shifts_tmp = other.d_shifts_tmp;
        other.d_shifts_tmp = nullptr;

        this->d_distances_tmp = other.d_distances_tmp;
        other.d_distances_tmp = nullptr;

        this->d_vectors_tmp = other.d_vectors_tmp;
        other.d_vectors_tmp = nullptr;

        this->points_capacity = other.points_capacity;
        other.points_capacity = 0;

        this->d_counts = other.d_counts;
        other.d_counts = nullptr;

        this->d_offsets = other.d_offsets;
        other.d_offsets = nullptr;

        this->d_block_totals = other.d_block_totals;
        other.d_block_totals = nullptr;

        this->d_block_offsets = other.d_block_offsets;
        other.d_block_offsets = nullptr;

        this->d_block_totals2 = other.d_block_totals2;
        other.d_block_totals2 = nullptr;
    }
    return *this;
}

void SortBuffers::allocate(
    size_t n_pairs,
    size_t n_points,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    // The per-pair and per-point arrays are sized by different quantities and
    // grown independently, so that getting more pairs does not throw away a
    // per-point buffer that is still big enough, and vice versa.
    if (this->capacity < n_pairs) {
        GPULITE_CUDART_CALL(cudaFree(this->d_pairs_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_shifts_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_distances_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_vectors_tmp));
        this->d_pairs_tmp = nullptr;
        this->d_shifts_tmp = nullptr;
        this->d_distances_tmp = nullptr;
        this->d_vectors_tmp = nullptr;
        this->capacity = 0;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_pairs_tmp, sizeof(size_t) * n_pairs * 2));
        this->capacity = n_pairs;
    }

    // The optional buffers are allocated the first time they are asked for,
    // which can be a later call than the one that set the capacity, so they are
    // sized from the capacity rather than from this call's `n_pairs`.
    if (return_shifts && this->d_shifts_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_shifts_tmp, sizeof(int32_t) * this->capacity * 3));
    }

    if (return_distances && this->d_distances_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_distances_tmp, sizeof(double) * this->capacity));
    }

    if (return_vectors && this->d_vectors_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_vectors_tmp, sizeof(double) * this->capacity * 3));
    }

    if (this->points_capacity < n_points) {
        GPULITE_CUDART_CALL(cudaFree(this->d_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals2));
        this->d_counts = nullptr;
        this->d_offsets = nullptr;
        this->d_block_totals = nullptr;
        this->d_block_offsets = nullptr;
        this->d_block_totals2 = nullptr;
        this->points_capacity = 0;

        // Same two-level scan as the cell list, see `device_exclusive_scan`
        size_t n_blocks = (n_points + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;
        size_t n_blocks_2 = (n_blocks + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_counts, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_offsets, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals, sizeof(int32_t) * n_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_offsets, sizeof(int32_t) * n_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals2, sizeof(int32_t) * n_blocks_2));

        this->points_capacity = n_points;
    }
}

// Reorder the output pair list (and any optional aligned outputs) so that pairs
// come out sorted by the first particle index. Used by the bruteforce and
// Verlet paths — the cell-list path produces output already sorted.
void vesin::cuda::sort_pairs(
    size_t n_points,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    if (neighbors.length > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "too many pairs to sort on the GPU (the limit is 2^31); "
            "please open an issue on GitHub if you need more than that"
        );
    }

    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    auto& sort = extras->sort_buffers;
    sort.allocate(
        neighbors.length,
        n_points,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    auto* copy_and_count_kernel = VESIN_SORT_PAIRS_KERNEL(factory, sort_pairs_copy_and_count);
    auto* scatter_kernel = VESIN_SORT_PAIRS_KERNEL(factory, sort_pairs_scatter);

    const size_t sort_threads = 256;
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3((neighbors.length + sort_threads - 1) / sort_threads);
    config.blockDim = dim3(sort_threads);

    // The pass below accumulates into `d_counts` with atomics, so it has to
    // start from zero on every call.
    GPULITE_CUDART_CALL(cudaMemset(sort.d_counts, 0, sizeof(int32_t) * n_points));

    copy_and_count_kernel->launch(
        config,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors,
        sort.d_pairs_tmp,
        sort.d_shifts_tmp,
        sort.d_distances_tmp,
        sort.d_vectors_tmp,
        sort.d_counts,
        neighbors.length,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    vesin::cuda::device_exclusive_scan(
        factory,
        sort.d_counts,
        sort.d_offsets,
        sort.d_block_totals,
        sort.d_block_offsets,
        sort.d_block_totals2,
        n_points
    );

    scatter_kernel->launch(
        config,
        sort.d_pairs_tmp,
        sort.d_shifts_tmp,
        sort.d_distances_tmp,
        sort.d_vectors_tmp,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors,
        sort.d_offsets,
        neighbors.length,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );
}
