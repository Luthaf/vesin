// CUDA Verlet cache support uses the existing cell-list builder to create
// candidate pairs at cutoff + skin. These kernels validate the cached
// reference positions and filter those cached candidates at the exact cutoff.
__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

__global__ void check_verlet_displacements(
    const double* __restrict__ positions,
    const double* __restrict__ ref_positions,
    size_t n_points,
    double half_skin_sq,
    int* rebuild_flag
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    double dx = positions[i * 3 + 0] - ref_positions[i * 3 + 0];
    double dy = positions[i * 3 + 1] - ref_positions[i * 3 + 1];
    double dz = positions[i * 3 + 2] - ref_positions[i * 3 + 2];
    double disp_sq = dx * dx + dy * dy + dz * dz;

    if (disp_sq > half_skin_sq) {
        atomicExch(rebuild_flag, 1);
    }
}

__global__ void filter_verlet_candidates(
    const double* __restrict__ positions,
    const double* __restrict__ box,
    const size_t* __restrict__ candidate_pairs,
    const int* __restrict__ candidate_shifts,
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
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_length) {
        return;
    }

    const double cutoff2 = cutoff * cutoff;
    const size_t i = candidate_pairs[idx * 2 + 0];
    const size_t j = candidate_pairs[idx * 2 + 1];
    const int sx = candidate_shifts[idx * 3 + 0];
    const int sy = candidate_shifts[idx * 3 + 1];
    const int sz = candidate_shifts[idx * 3 + 2];

    const double* ri = &positions[i * 3];
    const double* rj = &positions[j * 3];

    double shift_x = sx * box[0] + sy * box[3] + sz * box[6];
    double shift_y = sx * box[1] + sy * box[4] + sz * box[7];
    double shift_z = sx * box[2] + sy * box[5] + sz * box[8];

    double vx = rj[0] - ri[0] + shift_x;
    double vy = rj[1] - ri[1] + shift_y;
    double vz = rj[2] - ri[2] + shift_z;
    double dist_sq = vx * vx + vy * vy + vz * vz;

    if (dist_sq < cutoff2) {
        size_t out = atomicAdd_size_t(length, 1);
        pair_indices[out * 2 + 0] = i;
        pair_indices[out * 2 + 1] = j;

        if (return_shifts) {
            shifts_out[out * 3 + 0] = sx;
            shifts_out[out * 3 + 1] = sy;
            shifts_out[out * 3 + 2] = sz;
        }

        if (return_distances) {
            distances[out] = sqrt(dist_sq);
        }

        if (return_vectors) {
            vectors[out * 3 + 0] = vx;
            vectors[out * 3 + 1] = vy;
            vectors[out * 3 + 2] = vz;
        }
    }
}
