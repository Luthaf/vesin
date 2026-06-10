#include "cell_list.cuh"

// Cell list neighbor finding: bin particles into cells, then search neighboring cells.
// Particles are sorted by cell for memory coalescing. Multiple threads per particle
// cooperate on neighbor search. Output buffering reduces atomic contention.

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

// Vector math helpers for double3
__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double dot3(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 cross3(const double3& a, const double3& b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline double norm3(const double3& v) {
    return sqrt(dot3(v, v));
}

__device__ inline double3 normalize3(const double3& v) {
    double n = norm3(v);
    return make_double3(v.x / n, v.y / n, v.z / n);
}

__device__ void invert_matrix(const double3 box[3], double3 inverse[3]) {
    double a = box[0].x;
    double b = box[0].y;
    double c = box[0].z;
    double d = box[1].x;
    double e = box[1].y;
    double f = box[1].z;
    double g = box[2].x;
    double h = box[2].y;
    double i = box[2].z;

    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = 1.0 / det;

    inverse[0] = make_double3(
        (e * i - f * h) * invdet,
        (c * h - b * i) * invdet,
        (b * f - c * e) * invdet
    );
    inverse[1] = make_double3(
        (f * g - d * i) * invdet,
        (a * i - c * g) * invdet,
        (c * d - a * f) * invdet
    );
    inverse[2] = make_double3(
        (d * h - e * g) * invdet,
        (b * g - a * h) * invdet,
        (a * e - b * d) * invdet
    );
}

__global__ void compute_bounding_box(
    const double (*__restrict__ points)[3],
    size_t n_points,
    double* __restrict__ face_distances,
    double* __restrict__ bounding_min
) {
    // 256 here must match the number of threads used to launch this kernel.
    __shared__ double shared_min[3][256];
    __shared__ double shared_max[3][256];

    int tid = static_cast<int>(threadIdx.x);
    constexpr double MAX_DOUBLE = 1e300;
    constexpr double MIN_DOUBLE = -1e300;

    double local_min[3] = {MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE};
    double local_max[3] = {MIN_DOUBLE, MIN_DOUBLE, MIN_DOUBLE};

    const auto* pos3 = reinterpret_cast<const double3*>(points);
    for (size_t idx = tid; idx < n_points; idx += blockDim.x) {
        double3 point = pos3[idx];
        local_min[0] = min(local_min[0], point.x);
        local_min[1] = min(local_min[1], point.y);
        local_min[2] = min(local_min[2], point.z);
        local_max[0] = max(local_max[0], point.x);
        local_max[1] = max(local_max[1], point.y);
        local_max[2] = max(local_max[2], point.z);
    }

    shared_min[0][tid] = local_min[0];
    shared_min[1][tid] = local_min[1];
    shared_min[2][tid] = local_min[2];
    shared_max[0][tid] = local_max[0];
    shared_max[1][tid] = local_max[1];
    shared_max[2][tid] = local_max[2];
    __syncthreads();

    if (tid == 0) {
        double final_min[3] = {shared_min[0][0], shared_min[1][0], shared_min[2][0]};
        double final_max[3] = {shared_max[0][0], shared_max[1][0], shared_max[2][0]};

        for (int i = 1; i < blockDim.x; i++) {
            final_min[0] = min(final_min[0], shared_min[0][i]);
            final_min[1] = min(final_min[1], shared_min[1][i]);
            final_min[2] = min(final_min[2], shared_min[2][i]);
            final_max[0] = max(final_max[0], shared_max[0][i]);
            final_max[1] = max(final_max[1], shared_max[1][i]);
            final_max[2] = max(final_max[2], shared_max[2][i]);
        }

        for (int dim = 0; dim < 3; dim++) {
            bounding_min[dim] = final_min[dim];

            // Add 1% margin to bounding box size to ensure particles on the
            // edge are included in cells
            face_distances[dim] = (final_max[dim] - final_min[dim]) * 1.01;

            // make sure the distance is not too small (to prevent searching too
            // many cells down the line). This can happen if all point are in
            // the same plane in this direction
            if (face_distances[dim] < 1.0) {
                face_distances[dim] = 1.0;
            }
        }
    }
}

__global__ void compute_cell_grid_params(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    size_t max_cells,
    double inv_box[3][3],
    int* __restrict__ n_cells,
    int* __restrict__ n_search,
    int* __restrict__ n_cells_total,
    double* __restrict__ face_distances
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    // Normalize non-periodic box directions like the CPU BoundingBox logic,
    // then use the normalized matrix for inverse and distance calculations.
    double3 rows[3] = {
        make_double3(box[0][0], box[0][1], box[0][2]),
        make_double3(box[1][0], box[1][1], box[1][2]),
        make_double3(box[2][0], box[2][1], box[2][2]),
    };

    int n_periodic = 0;
    int periodic_idx_1 = -1;
    int periodic_idx_2 = -1;
    for (int d = 0; d < 3; d++) {
        if (periodic[d]) {
            n_periodic += 1;
            if (periodic_idx_1 == -1) {
                periodic_idx_1 = d;
            } else if (periodic_idx_2 == -1) {
                periodic_idx_2 = d;
            }
        }
    }

    if (n_periodic == 0) {
        rows[0] = make_double3(1.0, 0.0, 0.0);
        rows[1] = make_double3(0.0, 1.0, 0.0);
        rows[2] = make_double3(0.0, 0.0, 1.0);
    } else if (n_periodic == 1) {
        double3 a = rows[periodic_idx_1];
        double3 b = make_double3(0.0, 1.0, 0.0);
        if (abs(dot3(normalize3(a), b)) > 0.9) {
            b = make_double3(0.0, 0.0, 1.0);
        }
        double3 c = normalize3(cross3(a, b));
        b = normalize3(cross3(c, a));

        rows[(periodic_idx_1 + 1) % 3] = b;
        rows[(periodic_idx_1 + 2) % 3] = c;
    } else if (n_periodic == 2) {
        double3 a = rows[periodic_idx_1];
        double3 b = rows[periodic_idx_2];
        double3 c = normalize3(cross3(a, b));
        rows[3 - periodic_idx_1 - periodic_idx_2] = c;
    }

    // Inverse box matrix
    invert_matrix(rows, reinterpret_cast<double3(*)>(inv_box));

    // Box vectors
    double va[3] = {rows[0].x, rows[0].y, rows[0].z};
    double vb[3] = {rows[1].x, rows[1].y, rows[1].z};
    double vc[3] = {rows[2].x, rows[2].y, rows[2].z};

    // Cross products for face normals
    double bc[3] = {vb[1] * vc[2] - vb[2] * vc[1], vb[2] * vc[0] - vb[0] * vc[2], vb[0] * vc[1] - vb[1] * vc[0]};
    double ca[3] = {vc[1] * va[2] - vc[2] * va[1], vc[2] * va[0] - vc[0] * va[2], vc[0] * va[1] - vc[1] * va[0]};
    double ab[3] = {va[1] * vb[2] - va[2] * vb[1], va[2] * vb[0] - va[0] * vb[2], va[0] * vb[1] - va[1] * vb[0]};

    double bc_norm = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
    double ca_norm = sqrt(ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]);
    double ab_norm = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);

    // Compute the distances between opposite faces
    // For non-periodic directions, `face_distances` already contains the
    // bounding box extent (with 1% margin) computed by `compute_bounding_box`
    if (periodic[0]) {
        face_distances[0] = abs(va[0] * bc[0] + va[1] * bc[1] + va[2] * bc[2]) / bc_norm;
    }

    if (periodic[1]) {
        face_distances[1] = abs(vb[0] * ca[0] + vb[1] * ca[1] + vb[2] * ca[2]) / ca_norm;
    }

    if (periodic[2]) {
        face_distances[2] = abs(vc[0] * ab[0] + vc[1] * ab[1] + vc[2] * ab[2]) / ab_norm;
    }

    // Compute number of cells based on cutoff (one cell per cutoff distance)
    n_cells[0] = max(1, (int)floor(face_distances[0] / cutoff));
    n_cells[1] = max(1, (int)floor(face_distances[1] / cutoff));
    n_cells[2] = max(1, (int)floor(face_distances[2] / cutoff));

    int total = n_cells[0] * n_cells[1] * n_cells[2];

    // Limit total cells to effective maximum
    if (total > max_cells) {
        double ratio = cbrt((double)max_cells / total);
        n_cells[0] = max(1, (int)floor(n_cells[0] * ratio));
        n_cells[1] = max(1, (int)floor(n_cells[1] * ratio));
        n_cells[2] = max(1, (int)floor(n_cells[2] * ratio));
        total = n_cells[0] * n_cells[1] * n_cells[2];

        // total might still be above max_cells due to separate rounding for
        // separate dimensions. Decrease the largest dimension until the product
        // fits
        while (total > max_cells) {
            if (n_cells[0] >= n_cells[1] && n_cells[0] >= n_cells[2] && n_cells[0] > 1) {
                n_cells[0] -= 1;
            } else if (n_cells[1] >= n_cells[0] && n_cells[1] >= n_cells[2] && n_cells[1] > 1) {
                n_cells[1] -= 1;
            } else if (n_cells[2] > 1) {
                n_cells[2] -= 1;
            } else {
                break;
            }

            total = n_cells[0] * n_cells[1] * n_cells[2];
        }
    }
    n_cells_total[0] = total;

    // Compute search range - how many cells to search in each direction
    // When cells are larger than cutoff, we need to search more cells
    for (int dim = 0; dim < 3; dim++) {
        double cell_size = face_distances[dim] / n_cells[dim];
        n_search[dim] = max(1, (int)ceil(cutoff / cell_size));
    }

    // When there is mixed periodicity, increase search in non-periodic
    // dimensions to account for periodic shifts that can bring atoms
    // together across the bounding box.
    if (n_periodic > 0 && n_periodic < 3) {
        for (int dim = 0; dim < 3; dim++) {
            if (!periodic[dim]) {
                n_search[dim] = max(n_search[dim], n_cells[dim] - 1);
            }
        }
    }
}

__global__ void assign_cell_indices(
    const double (*__restrict__ points)[3],
    size_t n_points,
    const double inv_box[3][3],
    const bool periodic[3],
    const int* __restrict__ n_cells,
    const double* __restrict__ face_distances,
    const double* __restrict__ bounding_min,
    int* __restrict__ cell_indices,
    int* __restrict__ particle_shifts
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    // Vectorized position load
    const auto* pos3 = reinterpret_cast<const double3*>(points);
    double3 pos = pos3[i];

    // frac = pos @ inv_box (inv_box stored row-major)
    double frac[3];
    frac[0] = pos.x * inv_box[0][0] + pos.y * inv_box[1][0] + pos.z * inv_box[2][0];
    frac[1] = pos.x * inv_box[0][1] + pos.y * inv_box[1][1] + pos.z * inv_box[2][1];
    frac[2] = pos.x * inv_box[0][2] + pos.y * inv_box[1][2] + pos.z * inv_box[2][2];

    double pos_arr[3] = {pos.x, pos.y, pos.z};

    // For non-periodic dimensions, compute fractional coordinates using the
    // bounding box instead of the box matrix inverse
    for (int d = 0; d < 3; d++) {
        if (!periodic[d]) {
            frac[d] = (pos_arr[d] - bounding_min[d]) / face_distances[d];
        }
    }

    int cell_idx[3];
    int shift[3];

    for (int d = 0; d < 3; d++) {
        // Use divmod for both periodic and non-periodic dimensions.
        // For non-periodic dimensions, the fractional coordinates should be
        // in [0, 1) due to the bounding box margin, so shift will be 0.
        shift[d] = static_cast<int>(floor(frac[d]));
        frac[d] -= shift[d];

        cell_idx[d] = static_cast<int>(frac[d] * n_cells[d]);
        cell_idx[d] = max(0, min(n_cells[d] - 1, cell_idx[d]));
    }

    cell_indices[i] = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1];
    particle_shifts[i * 3 + 0] = shift[0];
    particle_shifts[i * 3 + 1] = shift[1];
    particle_shifts[i * 3 + 2] = shift[2];
}

__global__ void count_particles_per_cell(
    const int* __restrict__ cell_indices,
    size_t n_points,
    int* __restrict__ cell_counts
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }
    atomicAdd(&cell_counts[cell_indices[i]], 1);
}

__global__ void prefix_sum_cells(
    const int* __restrict__ cell_counts,
    int* __restrict__ cell_starts,
    const int* __restrict__ n_cells_total_ptr
) {
    extern __shared__ int shared[];
    int tid = static_cast<int>(threadIdx.x);
    int nthreads = static_cast<int>(blockDim.x);
    int n_cells_total = n_cells_total_ptr[0];

    // Each thread computes sum for its chunk
    int chunk_size = (n_cells_total + nthreads - 1) / nthreads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, n_cells_total);

    // Local scan within chunk
    int local_sum = 0;
    for (int i = start; i < end; i++) {
        int val = cell_counts[i];
        cell_starts[i] = local_sum;
        local_sum += val;
    }

    // Store chunk totals in shared memory
    shared[tid] = local_sum;
    __syncthreads();

    // Thread 0 computes prefix sum of chunk totals
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < nthreads; i++) {
            int val = shared[i];
            shared[i] = sum;
            sum += val;
        }
    }
    __syncthreads();

    // Add chunk offset to local results
    int offset = shared[tid];
    for (int i = start; i < end; i++) {
        cell_starts[i] += offset;
    }
}

__global__ void scatter_particles(
    const double (*__restrict__ points)[3],
    size_t n_points,
    const int* __restrict__ cell_indices,
    const int* __restrict__ particle_shifts,
    int* __restrict__ cell_offsets,
    double* __restrict__ sorted_points,
    int* __restrict__ sorted_indices,
    int* __restrict__ sorted_shifts,
    int* __restrict__ sorted_cell_indices
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    int cell = cell_indices[i];
    int slot = atomicAdd(&cell_offsets[cell], 1);

    // Vectorized points copy
    const auto* pos_in = reinterpret_cast<const double3*>(points);
    auto* pos_out = reinterpret_cast<double3*>(sorted_points);
    pos_out[slot] = pos_in[i];

    sorted_indices[slot] = static_cast<int>(i);
    sorted_shifts[slot * 3 + 0] = particle_shifts[i * 3 + 0];
    sorted_shifts[slot * 3 + 1] = particle_shifts[i * 3 + 1];
    sorted_shifts[slot * 3 + 2] = particle_shifts[i * 3 + 2];
    sorted_cell_indices[slot] = cell;
}

// THREADS_PER_PARTICLE threads cooperate on each particle's neighbor search
#define THREADS_PER_PARTICLE 8
// Buffer this many pairs before writing to global memory (reduces atomics)
#define MAX_BUFFERED_PAIRS 8

__global__ void find_neighbors_cell_list(
    const double* __restrict__ sorted_points,
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    const int* __restrict__ n_cells,
    const int* __restrict__ n_search,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ sorted_shifts,
    const int* __restrict__ sorted_cell_indices,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    double cutoff,
    bool full_list,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int (*shifts_out)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int* overflow_flag
) {
    // Thread organization: 32 threads/warp, THREADS_PER_PARTICLE threads per particle
    // Example with THREADS_PER_PARTICLE=8: 4 particles per warp, each gets 8 threads
    int warp_id = static_cast<int>(threadIdx.x) / 32;
    int lane_id = static_cast<int>(threadIdx.x) % 32;
    int particle_in_warp = lane_id / THREADS_PER_PARTICLE; // which particle in warp
    int thread_in_group = lane_id % THREADS_PER_PARTICLE;  // thread's role within group
    int particles_per_warp = 32 / THREADS_PER_PARTICLE;

    int warps_per_block = static_cast<int>(blockDim.x) / 32;
    size_t base_particle = (size_t)(blockIdx.x * warps_per_block + warp_id) * particles_per_warp;
    size_t i = base_particle + particle_in_warp; // global particle index

    if (i >= n_points) {
        return;
    }

    double cutoff2 = cutoff * cutoff;

    int nc_x = __ldg(&n_cells[0]);
    int nc_y = __ldg(&n_cells[1]);
    int nc_z = __ldg(&n_cells[2]);
    int nc_xy = nc_x * nc_y;
    int ns_x = __ldg(&n_search[0]);
    int ns_y = __ldg(&n_search[1]);
    int ns_z = __ldg(&n_search[2]);
    bool pbc_x = periodic[0];
    bool pbc_y = periodic[1];
    bool pbc_z = periodic[2];

    // Load box matrix rows as double3
    const auto* box3 = reinterpret_cast<const double3*>(box);
    const double3 box_row0 = box3[0];
    const double3 box_row1 = box3[1];
    const double3 box_row2 = box3[2];

    // Vectorized position load
    const auto* pos3 = reinterpret_cast<const double3*>(sorted_points);
    const double3 ri = pos3[i];
    int orig_i = __ldg(&sorted_indices[i]);
    int shift_i_x = __ldg(&sorted_shifts[i * 3 + 0]);
    int shift_i_y = __ldg(&sorted_shifts[i * 3 + 1]);
    int shift_i_z = __ldg(&sorted_shifts[i * 3 + 2]);

    int cell_i = __ldg(&sorted_cell_indices[i]);
    int cell_iz = cell_i / nc_xy;
    int cell_iy = (cell_i % nc_xy) / nc_x;
    int cell_ix = cell_i % nc_x;

    // Per-thread output buffer to reduce atomic contention
    int buffered_count = 0;
    int buffered_j[MAX_BUFFERED_PAIRS];
    int buffered_shift[MAX_BUFFERED_PAIRS * 3];
    double buffered_dist[MAX_BUFFERED_PAIRS];
    double buffered_vec[MAX_BUFFERED_PAIRS * 3];

    // Threads in group split neighbor cells: thread 0 does cells 0,8,16,...; thread 1 does 1,9,17,...
    int total_neighbor_cells = (2 * ns_x + 1) * (2 * ns_y + 1) * (2 * ns_z + 1);

    for (int cell_idx = thread_in_group; cell_idx < total_neighbor_cells; cell_idx += THREADS_PER_PARTICLE) {
        // Convert linear cell_idx to 3D offset (dx,dy,dz) from particle's cell
        int temp = cell_idx;
        int dx = (temp % (2 * ns_x + 1)) - ns_x;
        temp /= (2 * ns_x + 1);
        int dy = (temp % (2 * ns_y + 1)) - ns_y;
        int dz = (temp / (2 * ns_y + 1)) - ns_z;

        int cell_jx = cell_ix + dx;
        int cell_jy = cell_iy + dy;
        int cell_jz = cell_iz + dz;
        int cell_shift_x = 0;
        int cell_shift_y = 0;
        int cell_shift_z = 0;

        // Wrap cell indices for PBC, track shift; skip out-of-bounds for non-PBC
        if (pbc_x) {
            while (cell_jx < 0) {
                cell_jx += nc_x;
                cell_shift_x -= 1;
            }
            while (cell_jx >= nc_x) {
                cell_jx -= nc_x;
                cell_shift_x += 1;
            }
        } else {
            if (cell_jx < 0 || cell_jx >= nc_x) {
                continue;
            }
        }

        if (pbc_y) {
            while (cell_jy < 0) {
                cell_jy += nc_y;
                cell_shift_y -= 1;
            }
            while (cell_jy >= nc_y) {
                cell_jy -= nc_y;
                cell_shift_y += 1;
            }
        } else {
            if (cell_jy < 0 || cell_jy >= nc_y) {
                continue;
            }
        }

        if (pbc_z) {
            while (cell_jz < 0) {
                cell_jz += nc_z;
                cell_shift_z -= 1;
            }
            while (cell_jz >= nc_z) {
                cell_jz -= nc_z;
                cell_shift_z += 1;
            }
        } else {
            if (cell_jz < 0 || cell_jz >= nc_z) {
                continue;
            }
        }

        int cell_j = cell_jx + cell_jy * nc_x + cell_jz * nc_xy;
        int start_j = __ldg(&cell_starts[cell_j]);
        int count_j = __ldg(&cell_counts[cell_j]);

        for (int k = start_j; k < start_j + count_j; k++) {
            int orig_j = __ldg(&sorted_indices[k]);

            int shift_j_x = __ldg(&sorted_shifts[k * 3 + 0]);
            int shift_j_y = __ldg(&sorted_shifts[k * 3 + 1]);
            int shift_j_z = __ldg(&sorted_shifts[k * 3 + 2]);

            int total_shift_x = shift_i_x - shift_j_x + cell_shift_x;
            int total_shift_y = shift_i_y - shift_j_y + cell_shift_y;
            int total_shift_z = shift_i_z - shift_j_z + cell_shift_z;

            bool shift_is_zero = (total_shift_x == 0 && total_shift_y == 0 && total_shift_z == 0);

            if (orig_i == orig_j && shift_is_zero) {
                continue;
            }

            // Half-list: only keep pair once (i<j, or i==j with positive shift direction)
            if (!full_list) {
                if (orig_i > orig_j) {
                    continue;
                }
                if (orig_i == orig_j) {
                    // For self-images, use lexicographic ordering on shift vector
                    int shift_sum = total_shift_x + total_shift_y + total_shift_z;
                    if (shift_sum < 0) {
                        continue;
                    }
                    if (shift_sum == 0) {
                        if (total_shift_z < 0 || (total_shift_z == 0 && total_shift_y < 0)) {
                            continue;
                        }
                    }
                }
            }

            // Vectorized position load for particle j
            const double3 rj = pos3[k];

            // Convert integer shift to Cartesian displacement: shift @ box
            const double3 shift_cart = make_double3(
                total_shift_x * box_row0.x + total_shift_y * box_row1.x + total_shift_z * box_row2.x,
                total_shift_x * box_row0.y + total_shift_y * box_row1.y + total_shift_z * box_row2.y,
                total_shift_x * box_row0.z + total_shift_y * box_row1.z + total_shift_z * box_row2.z
            );

            // Vector from i to j (accounting for PBC)
            const double3 vec = make_double3(
                rj.x - ri.x + shift_cart.x,
                rj.y - ri.y + shift_cart.y,
                rj.z - ri.z + shift_cart.z
            );

            double dist2 = dot3(vec, vec);

            if (dist2 < cutoff2 && dist2 > 0.0) {
                buffered_j[buffered_count] = orig_j;
                buffered_shift[buffered_count * 3 + 0] = total_shift_x;
                buffered_shift[buffered_count * 3 + 1] = total_shift_y;
                buffered_shift[buffered_count * 3 + 2] = total_shift_z;
                buffered_dist[buffered_count] = sqrt(dist2);
                buffered_vec[buffered_count * 3 + 0] = vec.x;
                buffered_vec[buffered_count * 3 + 1] = vec.y;
                buffered_vec[buffered_count * 3 + 2] = vec.z;
                buffered_count++;

                // Flush buffer when full
                if (buffered_count >= MAX_BUFFERED_PAIRS) {
                    size_t base_idx = atomicAdd_size_t(length, buffered_count);

                    // Check if we are about to exceed max_pairs
                    if (base_idx + buffered_count > max_pairs) {
                        atomicExch(overflow_flag, 1);
                        return;
                    }

                    for (int b = 0; b < buffered_count; b++) {
                        pair_indices[base_idx + b][0] = orig_i;
                        pair_indices[base_idx + b][1] = buffered_j[b];
                        if (return_shifts) {
                            shifts_out[base_idx + b][0] = buffered_shift[b * 3 + 0];
                            shifts_out[base_idx + b][1] = buffered_shift[b * 3 + 1];
                            shifts_out[base_idx + b][2] = buffered_shift[b * 3 + 2];
                        }
                        if (return_distances) {
                            distances[base_idx + b] = buffered_dist[b];
                        }
                        if (return_vectors) {
                            vectors[base_idx + b][0] = buffered_vec[b * 3 + 0];
                            vectors[base_idx + b][1] = buffered_vec[b * 3 + 1];
                            vectors[base_idx + b][2] = buffered_vec[b * 3 + 2];
                        }
                    }
                    buffered_count = 0;
                }
            }
        }
    }

    // Flush remaining buffered pairs
    if (buffered_count > 0) {
        size_t base_idx = atomicAdd_size_t(length, buffered_count);

        // Check if we are about to exceed max_pairs
        if (base_idx + buffered_count > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }

        for (int b = 0; b < buffered_count; b++) {
            pair_indices[base_idx + b][0] = orig_i;
            pair_indices[base_idx + b][1] = buffered_j[b];
            if (return_shifts) {
                shifts_out[base_idx + b][0] = buffered_shift[b * 3 + 0];
                shifts_out[base_idx + b][1] = buffered_shift[b * 3 + 1];
                shifts_out[base_idx + b][2] = buffered_shift[b * 3 + 2];
            }

            if (return_distances) {
                distances[base_idx + b] = buffered_dist[b];
            }

            if (return_vectors) {
                vectors[base_idx + b][0] = buffered_vec[b * 3 + 0];
                vectors[base_idx + b][1] = buffered_vec[b * 3 + 1];
                vectors[base_idx + b][2] = buffered_vec[b * 3 + 2];
            }
        }
    }
}
