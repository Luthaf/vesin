#include "sort_pairs.cuh"

__device__ inline bool pair_less(
    const size_t (*pairs)[2],
    size_t a,
    size_t b
) {
    size_t ai = pairs[a][0];
    size_t bi = pairs[b][0];

    return ai < bi;
}

__device__ inline void swap_pair_payload(
    size_t (*pairs)[2],
    int (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t a,
    size_t b,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t pair0 = pairs[a][0];
    size_t pair1 = pairs[a][1];
    pairs[a][0] = pairs[b][0];
    pairs[a][1] = pairs[b][1];
    pairs[b][0] = pair0;
    pairs[b][1] = pair1;

    if (return_shifts) {
        int sx = shifts[a][0];
        int sy = shifts[a][1];
        int sz = shifts[a][2];
        shifts[a][0] = shifts[b][0];
        shifts[a][1] = shifts[b][1];
        shifts[a][2] = shifts[b][2];
        shifts[b][0] = sx;
        shifts[b][1] = sy;
        shifts[b][2] = sz;
    }

    if (return_distances) {
        double dist = distances[a];
        distances[a] = distances[b];
        distances[b] = dist;
    }

    if (return_vectors) {
        double vx = vectors[a][0];
        double vy = vectors[a][1];
        double vz = vectors[a][2];
        vectors[a][0] = vectors[b][0];
        vectors[a][1] = vectors[b][1];
        vectors[a][2] = vectors[b][2];
        vectors[b][0] = vx;
        vectors[b][1] = vy;
        vectors[b][2] = vz;
    }
}

__global__ void sort_pairs_fill_buffers(
    const size_t (*pairs_in)[2],
    const int (*shifts_in)[3],
    const double* __restrict__ distances_in,
    const double (*vectors_in)[3],
    size_t (*pairs_tmp)[2],
    int (*shifts_tmp)[3],
    double* __restrict__ distances_tmp,
    double (*vectors_tmp)[3],
    size_t length,
    size_t sort_capacity,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sort_capacity) {
        return;
    }

    if (idx < length) {
        pairs_tmp[idx][0] = pairs_in[idx][0];
        pairs_tmp[idx][1] = pairs_in[idx][1];

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
    } else {
        pairs_tmp[idx][0] = static_cast<size_t>(-1);
        pairs_tmp[idx][1] = static_cast<size_t>(-1);

        if (return_shifts) {
            shifts_tmp[idx][0] = 0;
            shifts_tmp[idx][1] = 0;
            shifts_tmp[idx][2] = 0;
        }

        if (return_distances) {
            distances_tmp[idx] = 0.0;
        }

        if (return_vectors) {
            vectors_tmp[idx][0] = 0.0;
            vectors_tmp[idx][1] = 0.0;
            vectors_tmp[idx][2] = 0.0;
        }
    }
}

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
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sort_capacity) {
        return;
    }

    size_t ixj = idx ^ j;
    if (ixj <= idx || ixj >= sort_capacity) {
        return;
    }

    bool ascending = ((idx & k) == 0);
    bool idx_less = pair_less(pairs, idx, ixj);
    bool should_swap = ascending ? !idx_less : idx_less;

    if (should_swap) {
        swap_pair_payload(
            pairs,
            shifts,
            distances,
            vectors,
            idx,
            ixj,
            return_shifts,
            return_distances,
            return_vectors
        );
    }
}

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
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }

    pairs_out[idx][0] = pairs_tmp[idx][0];
    pairs_out[idx][1] = pairs_tmp[idx][1];

    if (return_shifts) {
        shifts_out[idx][0] = shifts_tmp[idx][0];
        shifts_out[idx][1] = shifts_tmp[idx][1];
        shifts_out[idx][2] = shifts_tmp[idx][2];
    }

    if (return_distances) {
        distances_out[idx] = distances_tmp[idx];
    }

    if (return_vectors) {
        vectors_out[idx][0] = vectors_tmp[idx][0];
        vectors_out[idx][1] = vectors_tmp[idx][1];
        vectors_out[idx][2] = vectors_tmp[idx][2];
    }
}
