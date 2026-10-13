#ifndef VESIN_CUDA_BRUTEFORCE_CUH
#define VESIN_CUDA_BRUTEFORCE_CUH

#include <cstddef>

// Largest number of periodic images the brute force kernels will enumerate for
// a single pair of atoms, i.e. `prod(2 * n_search + 1)`.
#define MAX_BRUTE_FORCE_IMAGES 4096

// Largest number of points the brute force kernels accept, from the maximal
// possible value for `gridDim.y`. Should not be a limitation in practice, since
// we want to switch to cell list earlier anyway.
#define MAX_BRUTE_FORCE_POINTS 65535

// Bound on `n_search`, only there to keep the value finite for a box so small
// that `cutoff / face_distance` overflows. This needs to be set high enough to
// ensure `2 * MAX_N_SEARCH + 1 > MAX_BRUTE_FORCE_IMAGES`.
#define MAX_N_SEARCH 2048

__global__ void brute_force_half(
    const double (*points)[3],
    const double box[3][3],
    const double inv_box[3][3],
    const bool periodic[3],
    const int n_search[3],
    double cutoff2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int* overflow_flag
);

__global__ void brute_force_full(
    const double (*points)[3],
    const double box[3][3],
    const double inv_box[3][3],
    const bool periodic[3],
    const int n_search[3],
    double cutoff2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int* overflow_flag
);

__global__ void analyze_box(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    bool* all_periodic,
    int n_search[3],
    double inv_box_out[3][3]
);

#endif // VESIN_CUDA_BRUTEFORCE_CUH
