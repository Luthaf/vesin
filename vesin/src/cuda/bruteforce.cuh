#ifndef VESIN_CUDA_BRUTEFORCE_CUH
#define VESIN_CUDA_BRUTEFORCE_CUH

#include <cstddef>

__global__ void brute_force_half_orthogonal(
    const double* points,
    const double* box_diag,
    const bool* periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
);

__global__ void brute_force_full_orthogonal(
    const double* points,
    const double* box_diag,
    const bool* periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
);

__global__ void brute_force_half_general(
    const double* points,
    const double* box,
    const double* inv_box,
    const bool* periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
);

__global__ void brute_force_full_general(
    const double* points,
    const double* box,
    const double* inv_box,
    const bool* periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
);

// Status flags for mic_box_check
#define BOX_STATUS_ERROR 1
#define BOX_STATUS_ORTHOGONAL 2

__global__ void mic_box_check(
    const double* box,
    const bool* periodic,
    double cutoff,
    int* status,
    double* box_diag,   // Output: [Lx, Ly, Lz] for orthogonal boxes (can be nullptr)
    double* inv_box_out // Output: 9-element inverse box matrix (can be nullptr)
);

#endif // VESIN_CUDA_BRUTEFORCE_CUH
