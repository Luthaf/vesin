#ifndef VESIN_CUDA_VERLET_CUH
#define VESIN_CUDA_VERLET_CUH

#include <cstddef>

// CUDA Verlet cache support uses the existing cell-list builder to create
// candidate pairs at cutoff + skin. These kernels validate the cached
// reference positions and filter those cached candidates at the exact cutoff.

__global__ void check_verlet_displacements(
    const double* positions,
    const double* ref_positions,
    size_t n_points,
    double half_skin_sq,
    int* rebuild_flag
);

__global__ void filter_verlet_candidates(
    const double* positions,
    const double* box,
    const size_t* candidate_pairs,
    const int* candidate_shifts,
    size_t candidate_length,
    double cutoff,
    size_t* length,
    size_t* pair_indices,
    int* shifts_out,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

#endif // VESIN_CUDA_VERLET_CUH
