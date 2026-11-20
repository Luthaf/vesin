#ifndef VESIN_CUDA_BRUTEFORCE_CUH
#define VESIN_CUDA_BRUTEFORCE_CUH

#include <cstddef>

#include "integer.cuh"

// Largest number of periodic images the brute force kernels will enumerate for
// a single pair of atoms, i.e. `prod(2 * n_search + 1)`.
#define VESIN_MAX_BRUTE_FORCE_IMAGES 4096

// Largest number of points the brute force kernels accept, from the maximal
// possible value for `gridDim.y`. Should not be a limitation in practice, since
// we want to switch to cell list earlier anyway.
#define VESIN_MAX_BRUTE_FORCE_POINTS 65535

// Bound on `n_search`, only there to keep the value finite for a box so small
// that `cutoff / face_distance` overflows. This needs to be set high enough to
// ensure `2 * VESIN_MAX_N_SEARCH + 1 > VESIN_MAX_BRUTE_FORCE_IMAGES`.
#define VESIN_MAX_N_SEARCH 2048

__global__ void brute_force_half(
    const double (*points)[3],
    const double box[3][3],
    const double inv_box[3][3],
    const bool periodic[3],
    const int32 n_search[3],
    double cutoff2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int32 (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int32* overflow_flag
);

__global__ void brute_force_full(
    const double (*points)[3],
    const double box[3][3],
    const double inv_box[3][3],
    const bool periodic[3],
    const int32 n_search[3],
    double cutoff2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int32 (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int32* overflow_flag,
    int32 padded_to,
    int32* pairs_per_point
);

// Block size of `analyze_box`, which must be launched with a single block. This
// must be a power of two for the reduction, and is used to size shared memory.
#define VESIN_ANALYZE_BOX_BLOCK 256

// Compute `all_periodic`, the number of periodic images to search (`n_search`)
// and the (regularized) inverse box.
//
// When `compute_volume` is true, this also writes in `points_volume` the volume
// taken by the points: the periodic part of the box, times the extent of the
// points (at least `cutoff`) along each non-periodic direction. This is used to
// estimate the typical distance between points.
__global__ void analyze_box(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    const double (*points)[3],
    size_t n_points,
    bool compute_volume,
    bool* all_periodic,
    int32 n_search[3],
    double inv_box_out[3][3],
    double* points_volume
);

#endif // VESIN_CUDA_BRUTEFORCE_CUH
