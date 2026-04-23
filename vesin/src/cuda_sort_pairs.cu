__device__ inline bool pair_less(
    const size_t* pairs,
    const int* shifts,
    size_t a,
    size_t b,
    bool with_shifts
) {
    size_t ai = pairs[a * 2 + 0];
    size_t aj = pairs[a * 2 + 1];
    size_t bi = pairs[b * 2 + 0];
    size_t bj = pairs[b * 2 + 1];

    if (ai < bi) {
        return true;
    }
    if (ai > bi) {
        return false;
    }
    if (aj < bj) {
        return true;
    }
    if (aj > bj) {
        return false;
    }

    if (!with_shifts) {
        return false;
    }

    int asx = shifts[a * 3 + 0];
    int asy = shifts[a * 3 + 1];
    int asz = shifts[a * 3 + 2];
    int bsx = shifts[b * 3 + 0];
    int bsy = shifts[b * 3 + 1];
    int bsz = shifts[b * 3 + 2];

    if (asx < bsx) {
        return true;
    }
    if (asx > bsx) {
        return false;
    }

    if (asy < bsy) {
        return true;
    }
    if (asy > bsy) {
        return false;
    }

    return asz < bsz;
}

__device__ inline void swap_pair_payload(
    size_t* pairs,
    int* shifts,
    double* distances,
    double* vectors,
    size_t a,
    size_t b,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t pair0 = pairs[a * 2 + 0];
    size_t pair1 = pairs[a * 2 + 1];
    pairs[a * 2 + 0] = pairs[b * 2 + 0];
    pairs[a * 2 + 1] = pairs[b * 2 + 1];
    pairs[b * 2 + 0] = pair0;
    pairs[b * 2 + 1] = pair1;

    if (return_shifts) {
        int sx = shifts[a * 3 + 0];
        int sy = shifts[a * 3 + 1];
        int sz = shifts[a * 3 + 2];
        shifts[a * 3 + 0] = shifts[b * 3 + 0];
        shifts[a * 3 + 1] = shifts[b * 3 + 1];
        shifts[a * 3 + 2] = shifts[b * 3 + 2];
        shifts[b * 3 + 0] = sx;
        shifts[b * 3 + 1] = sy;
        shifts[b * 3 + 2] = sz;
    }

    if (return_distances) {
        double dist = distances[a];
        distances[a] = distances[b];
        distances[b] = dist;
    }

    if (return_vectors) {
        double vx = vectors[a * 3 + 0];
        double vy = vectors[a * 3 + 1];
        double vz = vectors[a * 3 + 2];
        vectors[a * 3 + 0] = vectors[b * 3 + 0];
        vectors[a * 3 + 1] = vectors[b * 3 + 1];
        vectors[a * 3 + 2] = vectors[b * 3 + 2];
        vectors[b * 3 + 0] = vx;
        vectors[b * 3 + 1] = vy;
        vectors[b * 3 + 2] = vz;
    }
}

__global__ void sort_pairs_fill_buffers(
    const size_t* __restrict__ pairs_in,
    const int* __restrict__ shifts_in,
    const double* __restrict__ distances_in,
    const double* __restrict__ vectors_in,
    size_t* __restrict__ pairs_tmp,
    int* __restrict__ shifts_tmp,
    double* __restrict__ distances_tmp,
    double* __restrict__ vectors_tmp,
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
        pairs_tmp[idx * 2 + 0] = pairs_in[idx * 2 + 0];
        pairs_tmp[idx * 2 + 1] = pairs_in[idx * 2 + 1];

        if (return_shifts) {
            shifts_tmp[idx * 3 + 0] = shifts_in[idx * 3 + 0];
            shifts_tmp[idx * 3 + 1] = shifts_in[idx * 3 + 1];
            shifts_tmp[idx * 3 + 2] = shifts_in[idx * 3 + 2];
        }

        if (return_distances) {
            distances_tmp[idx] = distances_in[idx];
        }

        if (return_vectors) {
            vectors_tmp[idx * 3 + 0] = vectors_in[idx * 3 + 0];
            vectors_tmp[idx * 3 + 1] = vectors_in[idx * 3 + 1];
            vectors_tmp[idx * 3 + 2] = vectors_in[idx * 3 + 2];
        }
    } else {
        pairs_tmp[idx * 2 + 0] = static_cast<size_t>(-1);
        pairs_tmp[idx * 2 + 1] = static_cast<size_t>(-1);

        if (return_shifts) {
            shifts_tmp[idx * 3 + 0] = 0;
            shifts_tmp[idx * 3 + 1] = 0;
            shifts_tmp[idx * 3 + 2] = 0;
        }

        if (return_distances) {
            distances_tmp[idx] = 0.0;
        }

        if (return_vectors) {
            vectors_tmp[idx * 3 + 0] = 0.0;
            vectors_tmp[idx * 3 + 1] = 0.0;
            vectors_tmp[idx * 3 + 2] = 0.0;
        }
    }
}

__global__ void sort_pairs_bitonic_step(
    size_t* __restrict__ pairs,
    int* __restrict__ shifts,
    double* __restrict__ distances,
    double* __restrict__ vectors,
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
    bool idx_less = pair_less(pairs, shifts, idx, ixj, return_shifts);
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
    size_t* __restrict__ pairs_out,
    int* __restrict__ shifts_out,
    double* __restrict__ distances_out,
    double* __restrict__ vectors_out,
    const size_t* __restrict__ pairs_tmp,
    const int* __restrict__ shifts_tmp,
    const double* __restrict__ distances_tmp,
    const double* __restrict__ vectors_tmp,
    size_t length,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }

    pairs_out[idx * 2 + 0] = pairs_tmp[idx * 2 + 0];
    pairs_out[idx * 2 + 1] = pairs_tmp[idx * 2 + 1];

    if (return_shifts) {
        shifts_out[idx * 3 + 0] = shifts_tmp[idx * 3 + 0];
        shifts_out[idx * 3 + 1] = shifts_tmp[idx * 3 + 1];
        shifts_out[idx * 3 + 2] = shifts_tmp[idx * 3 + 2];
    }

    if (return_distances) {
        distances_out[idx] = distances_tmp[idx];
    }

    if (return_vectors) {
        vectors_out[idx * 3 + 0] = vectors_tmp[idx * 3 + 0];
        vectors_out[idx * 3 + 1] = vectors_tmp[idx * 3 + 1];
        vectors_out[idx * 3 + 2] = vectors_tmp[idx * 3 + 2];
    }
}
