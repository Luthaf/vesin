#include "padding.cuh"

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
) {
    size_t slot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= n_points * static_cast<size_t>(padded_to)) {
        return;
    }

    size_t point = slot / padded_to;
    int32 index_in_point = static_cast<int32>(slot % padded_to);
    int32 count = pairs_per_point[point];

    if (index_in_point == 0 && count > padded_to) {
        atomicMin(padding_overflow, static_cast<int32>(point));
    }

    if (index_in_point < count) {
        // this slot contains an actual neighbor
        return;
    }

    pair_indices[slot][0] = point;
    pair_indices[slot][1] = static_cast<size_t>(-1);
    if (return_shifts) {
        shifts[slot][0] = 0;
        shifts[slot][1] = 0;
        shifts[slot][2] = 0;
    }
    if (return_distances) {
        distances[slot] = 0.0;
    }
    if (return_vectors) {
        vectors[slot][0] = 0.0;
        vectors[slot][1] = 0.0;
        vectors[slot][2] = 0.0;
    }
}
