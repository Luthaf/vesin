// Cell list neighbor finding: bin particles into cells, then search neighboring cells.
// Particles are sorted by cell for memory coalescing. Multiple threads per particle
// cooperate on neighbor search. Output buffering reduces atomic contention.

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

// Vector math helpers for double3
__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double dot3(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void compute_bounding_box(
    const double* __restrict__ positions,
    size_t n_points,
    double* __restrict__ bounding_min,
    double* __restrict__ bounding_max
) {
    // 256 here must match the number of threads used to launch this kernel.
    __shared__ double shared_min[3][256];
    __shared__ double shared_max[3][256];

    int tid = static_cast<int>(threadIdx.x);
    constexpr double MAX_DOUBLE = 1e300;
    constexpr double MIN_DOUBLE = -1e300;

    double local_min[3] = {MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE};
    double local_max[3] = {MIN_DOUBLE, MIN_DOUBLE, MIN_DOUBLE};

    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    for (size_t idx = tid; idx < n_points; idx += blockDim.x) {
        double3 point = pos3[idx];
        local_min[0] = fmin(local_min[0], point.x);
        local_min[1] = fmin(local_min[1], point.y);
        local_min[2] = fmin(local_min[2], point.z);
        local_max[0] = fmax(local_max[0], point.x);
        local_max[1] = fmax(local_max[1], point.y);
        local_max[2] = fmax(local_max[2], point.z);
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
            final_min[0] = fmin(final_min[0], shared_min[0][i]);
            final_min[1] = fmin(final_min[1], shared_min[1][i]);
            final_min[2] = fmin(final_min[2], shared_min[2][i]);
            final_max[0] = fmax(final_max[0], shared_max[0][i]);
            final_max[1] = fmax(final_max[1], shared_max[1][i]);
            final_max[2] = fmax(final_max[2], shared_max[2][i]);
        }

        for (int dim = 0; dim < 3; dim++) {
            bounding_min[dim] = final_min[dim];
            bounding_max[dim] = final_max[dim];

            // if all atoms have the same coordinate in this dimension, pretend
            // that the bounding box is at least 1 unit wide
            if (bounding_max[dim] - bounding_min[dim] < 1e-6) {
                bounding_max[dim] = bounding_min[dim] + 1;
            }
        }
    }
}

// Compute inv_box, n_cells, n_search from box matrix and cutoff (single thread)
__global__ void compute_cell_grid_params(
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    double cutoff,
    size_t max_cells,
    double* __restrict__ inv_box,
    int* __restrict__ n_cells,
    int* __restrict__ n_search,
    int* __restrict__ n_cells_total,
    const double* __restrict__ bounding_min,
    const double* __restrict__ bounding_max
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    // Box matrix elements
    double a = box[0];
    double b = box[1];
    double c = box[2];
    double d = box[3];
    double e = box[4];
    double f = box[5];
    double g = box[6];
    double h = box[7];
    double i = box[8];

    // Determinant
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = 1.0 / det;

    // Inverse box matrix
    inv_box[0] = (e * i - f * h) * invdet;
    inv_box[1] = (c * h - b * i) * invdet;
    inv_box[2] = (b * f - c * e) * invdet;
    inv_box[3] = (f * g - d * i) * invdet;
    inv_box[4] = (a * i - c * g) * invdet;
    inv_box[5] = (c * d - a * f) * invdet;
    inv_box[6] = (d * h - e * g) * invdet;
    inv_box[7] = (b * g - a * h) * invdet;
    inv_box[8] = (a * e - b * d) * invdet;

    // Box vectors
    double va[3] = {box[0], box[1], box[2]};
    double vb[3] = {box[3], box[4], box[5]};
    double vc[3] = {box[6], box[7], box[8]};

    // Cross products for face normals
    double bc[3] = {vb[1] * vc[2] - vb[2] * vc[1], vb[2] * vc[0] - vb[0] * vc[2], vb[0] * vc[1] - vb[1] * vc[0]};
    double ca[3] = {vc[1] * va[2] - vc[2] * va[1], vc[2] * va[0] - vc[0] * va[2], vc[0] * va[1] - vc[1] * va[0]};
    double ab[3] = {va[1] * vb[2] - va[2] * vb[1], va[2] * vb[0] - va[0] * vb[2], va[0] * vb[1] - va[1] * vb[0]};

    double bc_norm = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
    double ca_norm = sqrt(ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]);
    double ab_norm = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);

    // Distances between opposite faces
    // For non-periodic directions, use the bounding box extent (with 1% margin)
    // instead of the box matrix geometry
    double distances[3];

    if (periodic[0]) {
        distances[0] = fabs(va[0] * bc[0] + va[1] * bc[1] + va[2] * bc[2]) / bc_norm;
    } else {
        distances[0] = bounding_max[0] * 1.01 - bounding_min[0];
    }

    if (periodic[1]) {
        distances[1] = fabs(vb[0] * ca[0] + vb[1] * ca[1] + vb[2] * ca[2]) / ca_norm;
    } else {
        distances[1] = bounding_max[1] * 1.01 - bounding_min[1];
    }

    if (periodic[2]) {
        distances[2] = fabs(vc[0] * ab[0] + vc[1] * ab[1] + vc[2] * ab[2]) / ab_norm;
    } else {
        distances[2] = bounding_max[2] * 1.01 - bounding_min[2];
    }

    // Compute number of cells based on cutoff (one cell per cutoff distance)
    n_cells[0] = max(1, (int)floor(distances[0] / cutoff));
    n_cells[1] = max(1, (int)floor(distances[1] / cutoff));
    n_cells[2] = max(1, (int)floor(distances[2] / cutoff));

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
        double cell_size = distances[dim] / n_cells[dim];
        n_search[dim] = max(1, (int)ceil(cutoff / cell_size));
    }
}

// Map particles to cells via fractional coords, record periodic wrap shifts
__global__ void assign_cell_indices(
    const double* __restrict__ positions,
    const double* __restrict__ inv_box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    size_t n_points,
    int* __restrict__ cell_indices,
    int* __restrict__ particle_shifts,
    const double* __restrict__ bounding_min,
    const double* __restrict__ bounding_max
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    // Vectorized position load
    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    double3 pos = pos3[i];

    // frac = pos @ inv_box (inv_box stored row-major: rows are inv_box[0..2], inv_box[3..5], inv_box[6..8])
    double frac[3];
    frac[0] = pos.x * inv_box[0] + pos.y * inv_box[3] + pos.z * inv_box[6];
    frac[1] = pos.x * inv_box[1] + pos.y * inv_box[4] + pos.z * inv_box[7];
    frac[2] = pos.x * inv_box[2] + pos.y * inv_box[5] + pos.z * inv_box[8];

    double pos_arr[3] = {pos.x, pos.y, pos.z};

    // For non-periodic dimensions, compute fractional coordinates using the
    // bounding box instead of the box matrix inverse
    for (int d = 0; d < 3; d++) {
        if (!periodic[d]) {
            double dist = bounding_max[d] * 1.01 - bounding_min[d];
            frac[d] = (pos_arr[d] - bounding_min[d]) / dist;
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

// Count particles per cell (histogram)
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

// Exclusive prefix sum of cell counts -> cell_starts (single block, uses shared mem)
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

// Reorder particles by cell for coalesced access in neighbor search
__global__ void scatter_particles(
    const double* __restrict__ positions,
    const int* __restrict__ cell_indices,
    const int* __restrict__ particle_shifts,
    int* __restrict__ cell_offsets,
    size_t n_points,
    double* __restrict__ sorted_positions,
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

    // Vectorized position copy
    const double3* pos_in = reinterpret_cast<const double3*>(positions);
    double3* pos_out = reinterpret_cast<double3*>(sorted_positions);
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

// Main neighbor search kernel: each particle searches neighboring cells,
// threads within a group split the work across neighbor cells.
// Uses output buffering to batch writes and reduce atomic contention.
__global__ void find_neighbors_optimized(
    const double* __restrict__ sorted_positions,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ sorted_shifts,
    const int* __restrict__ sorted_cell_indices,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    const int* __restrict__ n_search,
    size_t n_points,
    double cutoff,
    bool full_list,
    size_t* length,
    size_t* pair_indices,
    int* shifts_out,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
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
    const bool pbc_x = periodic[0];
    const bool pbc_y = periodic[1];
    const bool pbc_z = periodic[2];

    // Load box matrix rows as double3
    const double3* box3 = reinterpret_cast<const double3*>(box);
    const double3 box_row0 = box3[0]; // box[0], box[1], box[2]
    const double3 box_row1 = box3[1]; // box[3], box[4], box[5]
    const double3 box_row2 = box3[2]; // box[6], box[7], box[8]

    // Vectorized position load
    const double3* pos3 = reinterpret_cast<const double3*>(sorted_positions);
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

            const bool shift_is_zero = (total_shift_x == 0 && total_shift_y == 0 && total_shift_z == 0);

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
                        pair_indices[(base_idx + b) * 2] = orig_i;
                        pair_indices[(base_idx + b) * 2 + 1] = buffered_j[b];
                        if (return_shifts) {
                            shifts_out[(base_idx + b) * 3 + 0] = buffered_shift[b * 3 + 0];
                            shifts_out[(base_idx + b) * 3 + 1] = buffered_shift[b * 3 + 1];
                            shifts_out[(base_idx + b) * 3 + 2] = buffered_shift[b * 3 + 2];
                        }
                        if (return_distances) {
                            distances[base_idx + b] = buffered_dist[b];
                        }
                        if (return_vectors) {
                            vectors[(base_idx + b) * 3 + 0] = buffered_vec[b * 3 + 0];
                            vectors[(base_idx + b) * 3 + 1] = buffered_vec[b * 3 + 1];
                            vectors[(base_idx + b) * 3 + 2] = buffered_vec[b * 3 + 2];
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
            pair_indices[(base_idx + b) * 2] = orig_i;
            pair_indices[(base_idx + b) * 2 + 1] = buffered_j[b];
            if (return_shifts) {
                shifts_out[(base_idx + b) * 3 + 0] = buffered_shift[b * 3 + 0];
                shifts_out[(base_idx + b) * 3 + 1] = buffered_shift[b * 3 + 1];
                shifts_out[(base_idx + b) * 3 + 2] = buffered_shift[b * 3 + 2];
            }

            if (return_distances) {
                distances[base_idx + b] = buffered_dist[b];
            }

            if (return_vectors) {
                vectors[(base_idx + b) * 3 + 0] = buffered_vec[b * 3 + 0];
                vectors[(base_idx + b) * 3 + 1] = buffered_vec[b * 3 + 1];
                vectors[(base_idx + b) * 3 + 2] = buffered_vec[b * 3 + 2];
            }
        }
    }
}
