#ifndef VESIN_CUDA_BRUTEFORCE_CUH
#define VESIN_CUDA_BRUTEFORCE_CUH

#include <cstddef>

__global__ void brute_force_half_orthogonal(
    const double (*points)[3],
    size_t n_points,
    const double box_diag[3],
    const bool periodic[3],
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

__global__ void brute_force_full_orthogonal(
    const double (*points)[3],
    size_t n_points,
    const double box_diag[3],
    const bool periodic[3],
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

__global__ void brute_force_half_general(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const double inv_box[3][3],
    const bool periodic[3],
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

__global__ void brute_force_full_general(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const double inv_box[3][3],
    const bool periodic[3],
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

// Status flags for mic_box_check
#define BOX_STATUS_ERROR 1
#define BOX_STATUS_ORTHOGONAL 2

__global__ void mic_box_check(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    int* status,
    double box_diag[3],
    double inv_box_out[3][3]
);

#endif // VESIN_CUDA_BRUTEFORCE_CUH
