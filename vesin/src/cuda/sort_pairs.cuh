#ifndef VESIN_CUDA_SORT_PAIRS_CUH
#define VESIN_CUDA_SORT_PAIRS_CUH

#include <cstddef>

__global__ void sort_pairs_fill_buffers(
    const size_t* pairs_in,
    const int* shifts_in,
    const double* distances_in,
    const double* vectors_in,
    size_t* pairs_tmp,
    int* shifts_tmp,
    double* distances_tmp,
    double* vectors_tmp,
    size_t length,
    size_t sort_capacity,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

__global__ void sort_pairs_bitonic_step(
    size_t* pairs,
    int* shifts,
    double* distances,
    double* vectors,
    size_t sort_capacity,
    size_t j,
    size_t k,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

__global__ void sort_pairs_copy_back(
    size_t* pairs_out,
    int* shifts_out,
    double* distances_out,
    double* vectors_out,
    const size_t* pairs_tmp,
    const int* shifts_tmp,
    const double* distances_tmp,
    const double* vectors_tmp,
    size_t length,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

#endif // VESIN_CUDA_SORT_PAIRS_CUH
