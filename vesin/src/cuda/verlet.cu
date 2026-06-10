#include "verlet.cuh"

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

__global__ void check_verlet_displacements(
    const double (*__restrict__ points)[3],
    const double* __restrict__ ref_points,
    size_t n_points,
    double half_skin_sq,
    int* rebuild_flag
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    double dx = points[i][0] - ref_points[i * 3 + 0];
    double dy = points[i][1] - ref_points[i * 3 + 1];
    double dz = points[i][2] - ref_points[i * 3 + 2];
    double disp_sq = dx * dx + dy * dy + dz * dz;

    if (disp_sq > half_skin_sq) {
        atomicExch(rebuild_flag, 1);
    }
}

__global__ void filter_verlet_candidates(
    const double (*__restrict__ points)[3],
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
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_length) {
        return;
    }

    double cutoff2 = cutoff * cutoff;
    size_t i = candidate_pairs[idx][0];
    size_t j = candidate_pairs[idx][1];
    int sx = candidate_shifts[idx][0];
    int sy = candidate_shifts[idx][1];
    int sz = candidate_shifts[idx][2];

    const double* ri = points[i];
    const double* rj = points[j];

    double shift_x = sx * box[0][0] + sy * box[1][0] + sz * box[2][0];
    double shift_y = sx * box[0][1] + sy * box[1][1] + sz * box[2][1];
    double shift_z = sx * box[0][2] + sy * box[1][2] + sz * box[2][2];

    double vx = rj[0] - ri[0] + shift_x;
    double vy = rj[1] - ri[1] + shift_y;
    double vz = rj[2] - ri[2] + shift_z;
    double dist_sq = vx * vx + vy * vy + vz * vz;

    if (dist_sq < cutoff2) {
        size_t out = atomicAdd_size_t(length, 1);
        pair_indices[out][0] = i;
        pair_indices[out][1] = j;

        if (return_shifts) {
            shifts_out[out][0] = sx;
            shifts_out[out][1] = sy;
            shifts_out[out][2] = sz;
        }

        if (return_distances) {
            distances[out] = sqrt(dist_sq);
        }

        if (return_vectors) {
            vectors[out][0] = vx;
            vectors[out][1] = vy;
            vectors[out][2] = vz;
        }
    }
}
