#ifndef VESIN_CUDA_VERLET_CUH
#define VESIN_CUDA_VERLET_CUH

#include <cstddef>

// CUDA Verlet cache support uses the existing cell-list builder to create
// candidate pairs at cutoff + skin. These kernels validate the cached
// reference points and filter those cached candidates at the exact cutoff.

__global__ void check_verlet_displacements(
    const double (*points)[3],
    const double* ref_points,
    size_t n_points,
    double half_skin_sq,
    int* rebuild_flag
);

__global__ void filter_verlet_candidates(
    const double (*points)[3],
    const double box[3][3],
    double cutoff,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    const size_t (*candidate_pairs)[2],
    const int (*candidate_shifts)[3],
    size_t candidate_length,
    size_t* length,
    size_t (*pair_indices)[2],
    int (*shifts_out)[3],
    double* distances,
    double (*vectors)[3]
);

#endif // VESIN_CUDA_VERLET_CUH
