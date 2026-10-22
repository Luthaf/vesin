#include "sort_pairs.cuh"

// Sorting the pairs by their first index using a counting sort.

// First pass: copy the pair list (and any optional aligned output) to the
// temporary buffers, while counting how many pairs each point owns.
//
// The copy is what makes the scatter below possible at all: it reads and writes
// the same slots, so it needs its input somewhere else. Counting here instead
// of in a separate kernel is free, since this pass already has to read the
// first index of every pair.
__global__ void sort_pairs_copy_and_count(
    const size_t (*pairs_in)[2],
    const int32 (*shifts_in)[3],
    const double* __restrict__ distances_in,
    const double (*vectors_in)[3],
    size_t (*pairs_tmp)[2],
    int32 (*shifts_tmp)[3],
    double* __restrict__ distances_tmp,
    double (*vectors_tmp)[3],
    int32* __restrict__ counts,
    size_t length,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }

    size_t first = pairs_in[idx][0];
    pairs_tmp[idx][0] = first;
    pairs_tmp[idx][1] = pairs_in[idx][1];

    atomicAdd(&counts[first], 1);

    if (return_shifts) {
        shifts_tmp[idx][0] = shifts_in[idx][0];
        shifts_tmp[idx][1] = shifts_in[idx][1];
        shifts_tmp[idx][2] = shifts_in[idx][2];
    }

    if (return_distances) {
        distances_tmp[idx] = distances_in[idx];
    }

    if (return_vectors) {
        vectors_tmp[idx][0] = vectors_in[idx][0];
        vectors_tmp[idx][1] = vectors_in[idx][1];
        vectors_tmp[idx][2] = vectors_in[idx][2];
    }
}

// Second pass: move every pair back to its final slot. `offsets[i]` enters this
// kernel pointing at the first slot owned by point `i` and is bumped as pairs
// are placed there, so all the pairs sharing a first index come out contiguous.
//
// Which of them lands where inside that range depends on the order the atomics
// resolve in, and is not reproducible from one call to the next.
__global__ void sort_pairs_scatter(
    const size_t (*pairs_tmp)[2],
    const int32 (*shifts_tmp)[3],
    const double* __restrict__ distances_tmp,
    const double (*vectors_tmp)[3],
    size_t (*pairs_out)[2],
    int32 (*shifts_out)[3],
    double* __restrict__ distances_out,
    double (*vectors_out)[3],
    int32* __restrict__ offsets,
    size_t length,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }

    size_t first = pairs_tmp[idx][0];
    auto out = static_cast<size_t>(atomicAdd(&offsets[first], 1));

    pairs_out[out][0] = first;
    pairs_out[out][1] = pairs_tmp[idx][1];

    if (return_shifts) {
        shifts_out[out][0] = shifts_tmp[idx][0];
        shifts_out[out][1] = shifts_tmp[idx][1];
        shifts_out[out][2] = shifts_tmp[idx][2];
    }

    if (return_distances) {
        distances_out[out] = distances_tmp[idx];
    }

    if (return_vectors) {
        vectors_out[out][0] = vectors_tmp[idx][0];
        vectors_out[out][1] = vectors_tmp[idx][1];
        vectors_out[out][2] = vectors_tmp[idx][2];
    }
}
