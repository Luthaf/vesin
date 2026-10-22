#ifndef VESIN_CUDA_SORT_PAIRS_HPP
#define VESIN_CUDA_SORT_PAIRS_HPP

#include <cstddef>
#include <cstdint>

#include "vesin.h"

namespace vesin::cuda {

struct SortBuffers {
    size_t capacity = 0;                  // Capacity for the pair buffers below (number of pairs)
    size_t (*d_pairs_tmp)[2] = nullptr;   // [capacity] temporary pair indices for sorting
    int32_t (*d_shifts_tmp)[3] = nullptr; // [capacity] temporary shifts for sorting
    double* d_distances_tmp = nullptr;    // [capacity] temporary distances for sorting
    double (*d_vectors_tmp)[3] = nullptr; // [capacity * 3] temporary vectors for sorting

    // Counting sort over the first index of each pair: how many pairs each
    // point owns, and where the pairs of each point start in the output.
    size_t points_capacity = 0;          // Capacity for the per-point arrays below
    int32_t* d_counts = nullptr;         // [points_capacity] pairs per point
    int32_t* d_offsets = nullptr;        // [points_capacity] exclusive scan of d_counts

    // Scratch for the multi-block scan of d_counts → d_offsets
    int32_t* d_block_totals = nullptr;  // [ceil(points_capacity/1024)]
    int32_t* d_block_offsets = nullptr; // [ceil(points_capacity/1024)]
    int32_t* d_block_totals2 = nullptr; // [ceil(n_blocks1/1024)]

    void allocate(
        size_t n_pairs,
        size_t n_points,
        bool return_shifts,
        bool return_distances,
        bool return_vectors
    );

    SortBuffers() = default;
    ~SortBuffers();

    SortBuffers(SortBuffers&& other) noexcept;
    SortBuffers& operator=(SortBuffers&& other) noexcept;

    SortBuffers(const SortBuffers&) = delete;
    SortBuffers& operator=(const SortBuffers&) = delete;
};

/// Sort the pairs in `neighbors` by the first particle index, using a counting
/// sort on the GPU over the `n_points` possible values of that index.
void sort_pairs(size_t n_points, VesinOptions options, VesinNeighborList& neighbors);

} // namespace vesin::cuda


#endif // VESIN_CUDA_SORT_PAIRS_HPP
