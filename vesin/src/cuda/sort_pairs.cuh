#ifndef VESIN_CUDA_SORT_PAIRS_CUH
#define VESIN_CUDA_SORT_PAIRS_CUH

#include <cstddef>

__global__ void sort_pairs_fill_buffers(
    const size_t (*pairs_in)[2],
    const int (*shifts_in)[3],
    const double* distances_in,
    const double (*vectors_in)[3],
    size_t (*pairs_tmp)[2],
    int (*shifts_tmp)[3],
    double* distances_tmp,
    double (*vectors_tmp)[3],
    size_t length,
    size_t sort_capacity,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

__global__ void sort_pairs_bitonic_step(
    size_t (*pairs)[2],
    int (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t sort_capacity,
    size_t j,
    size_t k,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

__global__ void sort_pairs_copy_back(
    size_t (*pairs_out)[2],
    int (*shifts_out)[3],
    double* distances_out,
    double (*vectors_out)[3],
    const size_t (*pairs_tmp)[2],
    const int (*shifts_tmp)[3],
    const double* distances_tmp,
    const double (*vectors_tmp)[3],
    size_t length,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
);

#endif // VESIN_CUDA_SORT_PAIRS_CUH
