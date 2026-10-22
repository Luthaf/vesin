#ifndef VESIN_CUDA_PADDING_CUH
#define VESIN_CUDA_PADDING_CUH

#include <cstddef>

#include "integer.cuh"

// Fill the output slots not used by actual neighbors with padding pairs
// `(i, SIZE_MAX)`, with zero shift, distance and vector. This is used by the
// paths that write pairs with atomics (brute force and Verlet), where point `i`
// owns the slots `[i * padded_to, (i + 1) * padded_to)` and `pairs_per_point[i]`
// is how many neighbors it tried to write there. The smallest index of a point
// with more than `padded_to` neighbors is stored in `padding_overflow` (which
// must be initialized to INT_MAX).
//
// This uses one thread per output slot.
__global__ void pad_neighbors(
    const int32* pairs_per_point,
    size_t n_points,
    int32 padded_to,
    int32* padding_overflow,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t (*pair_indices)[2],
    int32 (*shifts)[3],
    double* distances,
    double (*vectors)[3]
);

#endif // VESIN_CUDA_PADDING_CUH
