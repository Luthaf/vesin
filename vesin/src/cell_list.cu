// NVRTC type definitions
#if defined(__CUDACC_RTC__)
typedef int int32_t;
typedef unsigned int uint32_t;
#endif

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

// Compute cell grid parameters on device
__global__ void compute_cell_grid_params(
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    double cutoff,
    int max_cells,
    size_t n_points,
    int min_particles_per_cell,
    double* __restrict__ inv_box,
    int* __restrict__ n_cells,
    int* __restrict__ n_search,
    int* __restrict__ n_cells_total
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    // Box matrix elements
    double a = box[0], b = box[1], c = box[2];
    double d = box[3], e = box[4], f = box[5];
    double g = box[6], h = box[7], i = box[8];

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
    double dist_a = fabs(va[0] * bc[0] + va[1] * bc[1] + va[2] * bc[2]) / bc_norm;
    double dist_b = fabs(vb[0] * ca[0] + vb[1] * ca[1] + vb[2] * ca[2]) / ca_norm;
    double dist_c = fabs(vc[0] * ab[0] + vc[1] * ab[1] + vc[2] * ab[2]) / ab_norm;
    double distances[3] = {dist_a, dist_b, dist_c};

    // Compute number of cells based on cutoff (one cell per cutoff distance)
    n_cells[0] = max(1, (int)floor(distances[0] / cutoff));
    n_cells[1] = max(1, (int)floor(distances[1] / cutoff));
    n_cells[2] = max(1, (int)floor(distances[2] / cutoff));

    int total = n_cells[0] * n_cells[1] * n_cells[2];

    // Compute effective max cells based on minimum particles per cell target
    // This ensures we have enough work per cell for good GPU utilization
    int max_cells_from_particles = max(1, (int)(n_points / min_particles_per_cell));
    int effective_max_cells = min(max_cells, max_cells_from_particles);

    // Limit total cells to effective maximum
    if (total > effective_max_cells) {
        double ratio = cbrt((double)effective_max_cells / total);
        n_cells[0] = max(1, (int)(n_cells[0] * ratio));
        n_cells[1] = max(1, (int)(n_cells[1] * ratio));
        n_cells[2] = max(1, (int)(n_cells[2] * ratio));
        total = n_cells[0] * n_cells[1] * n_cells[2];
    }
    n_cells_total[0] = total;

    // Compute search range - how many cells to search in each direction
    // When cells are larger than cutoff, we need to search more cells
    for (int dim = 0; dim < 3; dim++) {
        double cell_size = distances[dim] / n_cells[dim];
        n_search[dim] = max(1, (int)ceil(cutoff / cell_size));
        if (!periodic[dim] && n_cells[dim] == 1) {
            n_search[dim] = 0;
        }
    }
}

// Assign particles to cells and record periodic wrapping shifts
__global__ void assign_cell_indices(
    const double* __restrict__ positions,
    const double* __restrict__ inv_box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    size_t n_points,
    int* __restrict__ cell_indices,
    int32_t* __restrict__ particle_shifts
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    double pos[3] = {positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]};

    // frac = pos @ inv_box
    double frac[3];
    frac[0] = pos[0] * inv_box[0] + pos[1] * inv_box[3] + pos[2] * inv_box[6];
    frac[1] = pos[0] * inv_box[1] + pos[1] * inv_box[4] + pos[2] * inv_box[7];
    frac[2] = pos[0] * inv_box[2] + pos[1] * inv_box[5] + pos[2] * inv_box[8];

    int cell_idx[3];
    int32_t shift[3];

    for (int d = 0; d < 3; d++) {
        if (periodic[d]) {
            shift[d] = static_cast<int32_t>(floor(frac[d]));
            frac[d] -= shift[d];
        } else {
            shift[d] = 0;
            frac[d] = fmax(0.0, fmin(frac[d], 1.0 - 1e-10));
        }
        cell_idx[d] = static_cast<int>(frac[d] * n_cells[d]);
        cell_idx[d] = max(0, min(n_cells[d] - 1, cell_idx[d]));
    }

    cell_indices[i] = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1];
    particle_shifts[i * 3 + 0] = shift[0];
    particle_shifts[i * 3 + 1] = shift[1];
    particle_shifts[i * 3 + 2] = shift[2];
}

// Count particles per cell using atomic increment
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

// Parallel exclusive prefix sum
__global__ void prefix_sum_cells(
    int* __restrict__ cell_counts,
    int* __restrict__ cell_starts,
    const int* __restrict__ n_cells_total_ptr
) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
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

// Scatter particles into cell-sorted order
__global__ void scatter_particles(
    const double* __restrict__ positions,
    const int* __restrict__ cell_indices,
    const int32_t* __restrict__ particle_shifts,
    int* __restrict__ cell_offsets,
    size_t n_points,
    double* __restrict__ sorted_positions,
    int* __restrict__ sorted_indices,
    int32_t* __restrict__ sorted_shifts,
    int* __restrict__ sorted_cell_indices
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    int cell = cell_indices[i];
    int slot = atomicAdd(&cell_offsets[cell], 1);

    sorted_positions[slot * 3 + 0] = positions[i * 3 + 0];
    sorted_positions[slot * 3 + 1] = positions[i * 3 + 1];
    sorted_positions[slot * 3 + 2] = positions[i * 3 + 2];
    sorted_indices[slot] = static_cast<int>(i);
    sorted_shifts[slot * 3 + 0] = particle_shifts[i * 3 + 0];
    sorted_shifts[slot * 3 + 1] = particle_shifts[i * 3 + 1];
    sorted_shifts[slot * 3 + 2] = particle_shifts[i * 3 + 2];
    sorted_cell_indices[slot] = cell;
}

#define THREADS_PER_PARTICLE 8
#define MAX_BUFFERED_PAIRS 8

// Multi-threaded neighbor finding with output buffering
__global__ void find_neighbors_optimized(
    const double* __restrict__ sorted_positions,
    const int* __restrict__ sorted_indices,
    const int32_t* __restrict__ sorted_shifts,
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
    int32_t* shifts_out,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int particle_in_warp = lane_id / THREADS_PER_PARTICLE;
    const int thread_in_group = lane_id % THREADS_PER_PARTICLE;
    const int particles_per_warp = 32 / THREADS_PER_PARTICLE;

    // Global particle index
    const int warps_per_block = blockDim.x / 32;
    const size_t base_particle = (size_t)(blockIdx.x * warps_per_block + warp_id) * particles_per_warp;
    const size_t i = base_particle + particle_in_warp;

    if (i >= n_points) {
        return;
    }

    const double cutoff2 = cutoff * cutoff;

    const int nc_x = __ldg(&n_cells[0]);
    const int nc_y = __ldg(&n_cells[1]);
    const int nc_z = __ldg(&n_cells[2]);
    const int nc_xy = nc_x * nc_y;
    const int ns_x = __ldg(&n_search[0]);
    const int ns_y = __ldg(&n_search[1]);
    const int ns_z = __ldg(&n_search[2]);
    const bool pbc_x = periodic[0];
    const bool pbc_y = periodic[1];
    const bool pbc_z = periodic[2];

    const double box00 = __ldg(&box[0]), box01 = __ldg(&box[1]), box02 = __ldg(&box[2]);
    const double box10 = __ldg(&box[3]), box11 = __ldg(&box[4]), box12 = __ldg(&box[5]);
    const double box20 = __ldg(&box[6]), box21 = __ldg(&box[7]), box22 = __ldg(&box[8]);

    const double ri_x = __ldg(&sorted_positions[i * 3 + 0]);
    const double ri_y = __ldg(&sorted_positions[i * 3 + 1]);
    const double ri_z = __ldg(&sorted_positions[i * 3 + 2]);
    const int orig_i = __ldg(&sorted_indices[i]);
    const int32_t shift_i_x = __ldg(&sorted_shifts[i * 3 + 0]);
    const int32_t shift_i_y = __ldg(&sorted_shifts[i * 3 + 1]);
    const int32_t shift_i_z = __ldg(&sorted_shifts[i * 3 + 2]);

    const int cell_i = __ldg(&sorted_cell_indices[i]);
    const int cell_iz = cell_i / nc_xy;
    const int cell_iy = (cell_i % nc_xy) / nc_x;
    const int cell_ix = cell_i % nc_x;

    int buffered_count = 0;
    int buffered_j[MAX_BUFFERED_PAIRS];
    int32_t buffered_shift[MAX_BUFFERED_PAIRS * 3];
    double buffered_dist[MAX_BUFFERED_PAIRS];
    double buffered_vec[MAX_BUFFERED_PAIRS * 3];

    int total_neighbor_cells = (2 * ns_x + 1) * (2 * ns_y + 1) * (2 * ns_z + 1);

    for (int cell_idx = thread_in_group; cell_idx < total_neighbor_cells; cell_idx += THREADS_PER_PARTICLE) {
        int temp = cell_idx;
        int dx = (temp % (2 * ns_x + 1)) - ns_x;
        temp /= (2 * ns_x + 1);
        int dy = (temp % (2 * ns_y + 1)) - ns_y;
        int dz = (temp / (2 * ns_y + 1)) - ns_z;

        int cell_jx = cell_ix + dx;
        int cell_jy = cell_iy + dy;
        int cell_jz = cell_iz + dz;
        int32_t cell_shift_x = 0, cell_shift_y = 0, cell_shift_z = 0;

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

        const int cell_j = cell_jx + cell_jy * nc_x + cell_jz * nc_xy;
        const int start_j = __ldg(&cell_starts[cell_j]);
        const int count_j = __ldg(&cell_counts[cell_j]);

        for (int k = start_j; k < start_j + count_j; k++) {
            const int orig_j = __ldg(&sorted_indices[k]);

            const int32_t shift_j_x = __ldg(&sorted_shifts[k * 3 + 0]);
            const int32_t shift_j_y = __ldg(&sorted_shifts[k * 3 + 1]);
            const int32_t shift_j_z = __ldg(&sorted_shifts[k * 3 + 2]);

            const int32_t total_shift_x = shift_i_x - shift_j_x + cell_shift_x;
            const int32_t total_shift_y = shift_i_y - shift_j_y + cell_shift_y;
            const int32_t total_shift_z = shift_i_z - shift_j_z + cell_shift_z;

            const bool shift_is_zero = (total_shift_x == 0 && total_shift_y == 0 && total_shift_z == 0);

            if (orig_i == orig_j && shift_is_zero) {
                continue;
            }

            if (!full_list) {
                if (orig_i > orig_j) {
                    continue;
                }
                if (orig_i == orig_j) {
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

            const double rj_x = __ldg(&sorted_positions[k * 3 + 0]);
            const double rj_y = __ldg(&sorted_positions[k * 3 + 1]);
            const double rj_z = __ldg(&sorted_positions[k * 3 + 2]);

            const double shift_cart_x = total_shift_x * box00 + total_shift_y * box10 + total_shift_z * box20;
            const double shift_cart_y = total_shift_x * box01 + total_shift_y * box11 + total_shift_z * box21;
            const double shift_cart_z = total_shift_x * box02 + total_shift_y * box12 + total_shift_z * box22;

            const double vec_x = rj_x - ri_x + shift_cart_x;
            const double vec_y = rj_y - ri_y + shift_cart_y;
            const double vec_z = rj_z - ri_z + shift_cart_z;

            const double dist2 = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

            if (dist2 < cutoff2 && dist2 > 0.0) {
                buffered_j[buffered_count] = orig_j;
                buffered_shift[buffered_count * 3 + 0] = total_shift_x;
                buffered_shift[buffered_count * 3 + 1] = total_shift_y;
                buffered_shift[buffered_count * 3 + 2] = total_shift_z;
                buffered_dist[buffered_count] = sqrt(dist2);
                buffered_vec[buffered_count * 3 + 0] = vec_x;
                buffered_vec[buffered_count * 3 + 1] = vec_y;
                buffered_vec[buffered_count * 3 + 2] = vec_z;
                buffered_count++;

                if (buffered_count >= MAX_BUFFERED_PAIRS) {
                    size_t base_idx = atomicAdd_size_t(length, buffered_count);
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

    if (buffered_count > 0) {
        size_t base_idx = atomicAdd_size_t(length, buffered_count);
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

// Maximum particles per cell that can be cached in shared memory
#define MAX_PARTICLES_PER_CELL 384

// Cell-pair-centric neighbor finding with shared memory caching
// One block per cell, particles from current cell cached in shared memory
__global__ void find_neighbors_cell_pairs_smem(
    const double* __restrict__ sorted_positions,
    const int* __restrict__ sorted_indices,
    const int32_t* __restrict__ sorted_shifts,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    const int* __restrict__ n_search,
    const int* __restrict__ n_cells_total_ptr,
    double cutoff,
    bool full_list,
    size_t* __restrict__ length,
    size_t* __restrict__ pair_indices,
    int32_t* __restrict__ shifts_out,
    double* __restrict__ distances,
    double* __restrict__ vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    // Shared memory for caching particles in current cell
    __shared__ double smem_pos[MAX_PARTICLES_PER_CELL * 3];
    __shared__ int smem_idx[MAX_PARTICLES_PER_CELL];
    __shared__ int32_t smem_shift[MAX_PARTICLES_PER_CELL * 3];
    __shared__ int smem_count;

    int cell_i = blockIdx.x;
    int n_cells_total = n_cells_total_ptr[0];
    if (cell_i >= n_cells_total) {
        return;
    }

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    double cutoff2 = cutoff * cutoff;

    int nc_x = n_cells[0];
    int nc_y = n_cells[1];
    int nc_z = n_cells[2];
    int nc_xy = nc_x * nc_y;

    int cell_iz = cell_i / nc_xy;
    int cell_iy = (cell_i % nc_xy) / nc_x;
    int cell_ix = cell_i % nc_x;

    int start_i = cell_starts[cell_i];
    int count_i = cell_counts[cell_i];

    // Limit to shared memory capacity
    if (tid == 0) {
        smem_count = min(count_i, MAX_PARTICLES_PER_CELL);
    }
    __syncthreads();
    int cached_count = smem_count;

    // Cooperatively load particles from current cell into shared memory
    for (int k = tid; k < cached_count; k += nthreads) {
        int idx = start_i + k;
        smem_pos[k * 3 + 0] = sorted_positions[idx * 3 + 0];
        smem_pos[k * 3 + 1] = sorted_positions[idx * 3 + 1];
        smem_pos[k * 3 + 2] = sorted_positions[idx * 3 + 2];
        smem_idx[k] = sorted_indices[idx];
        smem_shift[k * 3 + 0] = sorted_shifts[idx * 3 + 0];
        smem_shift[k * 3 + 1] = sorted_shifts[idx * 3 + 1];
        smem_shift[k * 3 + 2] = sorted_shifts[idx * 3 + 2];
    }
    __syncthreads();

    // Iterate over all neighboring cells
    for (int dz = -n_search[2]; dz <= n_search[2]; dz++) {
        for (int dy = -n_search[1]; dy <= n_search[1]; dy++) {
            for (int dx = -n_search[0]; dx <= n_search[0]; dx++) {

                int cell_jx = cell_ix + dx;
                int cell_jy = cell_iy + dy;
                int cell_jz = cell_iz + dz;
                int32_t cell_shift[3] = {0, 0, 0};

                if (periodic[0]) {
                    while (cell_jx < 0) {
                        cell_jx += nc_x;
                        cell_shift[0] -= 1;
                    }
                    while (cell_jx >= nc_x) {
                        cell_jx -= nc_x;
                        cell_shift[0] += 1;
                    }
                } else {
                    if (cell_jx < 0 || cell_jx >= nc_x) {
                        continue;
                    }
                }

                if (periodic[1]) {
                    while (cell_jy < 0) {
                        cell_jy += nc_y;
                        cell_shift[1] -= 1;
                    }
                    while (cell_jy >= nc_y) {
                        cell_jy -= nc_y;
                        cell_shift[1] += 1;
                    }
                } else {
                    if (cell_jy < 0 || cell_jy >= nc_y) {
                        continue;
                    }
                }

                if (periodic[2]) {
                    while (cell_jz < 0) {
                        cell_jz += nc_z;
                        cell_shift[2] -= 1;
                    }
                    while (cell_jz >= nc_z) {
                        cell_jz -= nc_z;
                        cell_shift[2] += 1;
                    }
                } else {
                    if (cell_jz < 0 || cell_jz >= nc_z) {
                        continue;
                    }
                }

                int cell_j = cell_jx + cell_jy * nc_x + cell_jz * nc_xy;
                int start_j = cell_starts[cell_j];
                int count_j = cell_counts[cell_j];

                // Total pairs: cached_count × count_j
                int n_pairs = cached_count * count_j;

                for (int pair_idx = tid; pair_idx < n_pairs; pair_idx += nthreads) {
                    int local_i = pair_idx / count_j;
                    int local_j = pair_idx % count_j;

                    int idx_j = start_j + local_j;

                    // Load from shared memory for particle i
                    int orig_i = smem_idx[local_i];
                    double ri[3] = {
                        smem_pos[local_i * 3 + 0],
                        smem_pos[local_i * 3 + 1],
                        smem_pos[local_i * 3 + 2]
                    };
                    int32_t shift_i[3] = {
                        smem_shift[local_i * 3 + 0],
                        smem_shift[local_i * 3 + 1],
                        smem_shift[local_i * 3 + 2]
                    };

                    // Load from global memory for particle j
                    int orig_j = sorted_indices[idx_j];
                    double rj[3] = {
                        sorted_positions[idx_j * 3 + 0],
                        sorted_positions[idx_j * 3 + 1],
                        sorted_positions[idx_j * 3 + 2]
                    };
                    int32_t shift_j[3] = {
                        sorted_shifts[idx_j * 3 + 0],
                        sorted_shifts[idx_j * 3 + 1],
                        sorted_shifts[idx_j * 3 + 2]
                    };

                    int32_t total_shift[3];
                    total_shift[0] = shift_i[0] - shift_j[0] + cell_shift[0];
                    total_shift[1] = shift_i[1] - shift_j[1] + cell_shift[1];
                    total_shift[2] = shift_i[2] - shift_j[2] + cell_shift[2];

                    bool shift_is_zero = (total_shift[0] == 0 && total_shift[1] == 0 && total_shift[2] == 0);

                    if (orig_i == orig_j && shift_is_zero) {
                        continue;
                    }

                    if (!full_list) {
                        if (orig_i > orig_j) {
                            continue;
                        }
                        if (orig_i == orig_j) {
                            int shift_sum = total_shift[0] + total_shift[1] + total_shift[2];
                            if (shift_sum < 0) {
                                continue;
                            }
                            if (shift_sum == 0) {
                                if (total_shift[2] < 0 || (total_shift[2] == 0 && total_shift[1] < 0)) {
                                    continue;
                                }
                            }
                        }
                    }

                    double shift_cart[3];
                    shift_cart[0] = total_shift[0] * box[0] + total_shift[1] * box[3] + total_shift[2] * box[6];
                    shift_cart[1] = total_shift[0] * box[1] + total_shift[1] * box[4] + total_shift[2] * box[7];
                    shift_cart[2] = total_shift[0] * box[2] + total_shift[1] * box[5] + total_shift[2] * box[8];

                    double vector[3];
                    vector[0] = rj[0] - ri[0] + shift_cart[0];
                    vector[1] = rj[1] - ri[1] + shift_cart[1];
                    vector[2] = rj[2] - ri[2] + shift_cart[2];

                    double dist2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];

                    if (dist2 < cutoff2 && dist2 > 0.0) {
                        size_t out_idx = atomicAdd_size_t(&length[0], 1);
                        pair_indices[out_idx * 2] = orig_i;
                        pair_indices[out_idx * 2 + 1] = orig_j;

                        if (return_shifts) {
                            shifts_out[out_idx * 3 + 0] = total_shift[0];
                            shifts_out[out_idx * 3 + 1] = total_shift[1];
                            shifts_out[out_idx * 3 + 2] = total_shift[2];
                        }
                        if (return_vectors) {
                            vectors[out_idx * 3 + 0] = vector[0];
                            vectors[out_idx * 3 + 1] = vector[1];
                            vectors[out_idx * 3 + 2] = vector[2];
                        }
                        if (return_distances) {
                            distances[out_idx] = sqrt(dist2);
                        }
                    }
                }
            }
        }
    }

    // Handle overflow particles not cached in shared memory (if any)
    // These are processed without shared memory benefit
    for (int extra_i = cached_count; extra_i < count_i; extra_i++) {
        int idx_i = start_i + extra_i;
        int orig_i = sorted_indices[idx_i];
        double ri[3] = {
            sorted_positions[idx_i * 3 + 0],
            sorted_positions[idx_i * 3 + 1],
            sorted_positions[idx_i * 3 + 2]
        };
        int32_t shift_i[3] = {
            sorted_shifts[idx_i * 3 + 0],
            sorted_shifts[idx_i * 3 + 1],
            sorted_shifts[idx_i * 3 + 2]
        };

        for (int dz = -n_search[2]; dz <= n_search[2]; dz++) {
            for (int dy = -n_search[1]; dy <= n_search[1]; dy++) {
                for (int dx = -n_search[0]; dx <= n_search[0]; dx++) {

                    int cell_jx = cell_ix + dx;
                    int cell_jy = cell_iy + dy;
                    int cell_jz = cell_iz + dz;
                    int32_t cell_shift[3] = {0, 0, 0};

                    if (periodic[0]) {
                        while (cell_jx < 0) {
                            cell_jx += nc_x;
                            cell_shift[0] -= 1;
                        }
                        while (cell_jx >= nc_x) {
                            cell_jx -= nc_x;
                            cell_shift[0] += 1;
                        }
                    } else {
                        if (cell_jx < 0 || cell_jx >= nc_x) {
                            continue;
                        }
                    }

                    if (periodic[1]) {
                        while (cell_jy < 0) {
                            cell_jy += nc_y;
                            cell_shift[1] -= 1;
                        }
                        while (cell_jy >= nc_y) {
                            cell_jy -= nc_y;
                            cell_shift[1] += 1;
                        }
                    } else {
                        if (cell_jy < 0 || cell_jy >= nc_y) {
                            continue;
                        }
                    }

                    if (periodic[2]) {
                        while (cell_jz < 0) {
                            cell_jz += nc_z;
                            cell_shift[2] -= 1;
                        }
                        while (cell_jz >= nc_z) {
                            cell_jz -= nc_z;
                            cell_shift[2] += 1;
                        }
                    } else {
                        if (cell_jz < 0 || cell_jz >= nc_z) {
                            continue;
                        }
                    }

                    int cell_j = cell_jx + cell_jy * nc_x + cell_jz * nc_xy;
                    int start_j = cell_starts[cell_j];
                    int count_j = cell_counts[cell_j];

                    for (int local_j = tid; local_j < count_j; local_j += nthreads) {
                        int idx_j = start_j + local_j;
                        int orig_j = sorted_indices[idx_j];

                        int32_t shift_j[3] = {
                            sorted_shifts[idx_j * 3 + 0],
                            sorted_shifts[idx_j * 3 + 1],
                            sorted_shifts[idx_j * 3 + 2]
                        };

                        int32_t total_shift[3];
                        total_shift[0] = shift_i[0] - shift_j[0] + cell_shift[0];
                        total_shift[1] = shift_i[1] - shift_j[1] + cell_shift[1];
                        total_shift[2] = shift_i[2] - shift_j[2] + cell_shift[2];

                        bool shift_is_zero = (total_shift[0] == 0 && total_shift[1] == 0 && total_shift[2] == 0);

                        if (orig_i == orig_j && shift_is_zero) {
                            continue;
                        }

                        if (!full_list) {
                            if (orig_i > orig_j) {
                                continue;
                            }
                            if (orig_i == orig_j) {
                                int shift_sum = total_shift[0] + total_shift[1] + total_shift[2];
                                if (shift_sum < 0) {
                                    continue;
                                }
                                if (shift_sum == 0) {
                                    if (total_shift[2] < 0 || (total_shift[2] == 0 && total_shift[1] < 0)) {
                                        continue;
                                    }
                                }
                            }
                        }

                        double rj[3] = {
                            sorted_positions[idx_j * 3 + 0],
                            sorted_positions[idx_j * 3 + 1],
                            sorted_positions[idx_j * 3 + 2]
                        };

                        double shift_cart[3];
                        shift_cart[0] = total_shift[0] * box[0] + total_shift[1] * box[3] + total_shift[2] * box[6];
                        shift_cart[1] = total_shift[0] * box[1] + total_shift[1] * box[4] + total_shift[2] * box[7];
                        shift_cart[2] = total_shift[0] * box[2] + total_shift[1] * box[5] + total_shift[2] * box[8];

                        double vector[3];
                        vector[0] = rj[0] - ri[0] + shift_cart[0];
                        vector[1] = rj[1] - ri[1] + shift_cart[1];
                        vector[2] = rj[2] - ri[2] + shift_cart[2];

                        double dist2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];

                        if (dist2 < cutoff2 && dist2 > 0.0) {
                            size_t out_idx = atomicAdd_size_t(&length[0], 1);
                            pair_indices[out_idx * 2] = orig_i;
                            pair_indices[out_idx * 2 + 1] = orig_j;

                            if (return_shifts) {
                                shifts_out[out_idx * 3 + 0] = total_shift[0];
                                shifts_out[out_idx * 3 + 1] = total_shift[1];
                                shifts_out[out_idx * 3 + 2] = total_shift[2];
                            }
                            if (return_vectors) {
                                vectors[out_idx * 3 + 0] = vector[0];
                                vectors[out_idx * 3 + 1] = vector[1];
                                vectors[out_idx * 3 + 2] = vector[2];
                            }
                            if (return_distances) {
                                distances[out_idx] = sqrt(dist2);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Cell-pair-centric neighbor finding: one block per cell, threads cooperate on pairs
// This approach mirrors the CPU implementation's iteration order for better efficiency
__global__ void find_neighbors_cell_pairs(
    const double* __restrict__ sorted_positions,
    const int* __restrict__ sorted_indices,
    const int32_t* __restrict__ sorted_shifts,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    const int* __restrict__ n_search,
    const int* __restrict__ n_cells_total_ptr,
    double cutoff,
    bool full_list,
    size_t* __restrict__ length,
    size_t* __restrict__ pair_indices,
    int32_t* __restrict__ shifts_out,
    double* __restrict__ distances,
    double* __restrict__ vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    // Each block handles one cell
    int cell_i = blockIdx.x;
    int n_cells_total = n_cells_total_ptr[0];
    if (cell_i >= n_cells_total) {
        return;
    }

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    double cutoff2 = cutoff * cutoff;

    // Decompose cell_i into 3D indices
    int nc_x = n_cells[0];
    int nc_y = n_cells[1];
    int nc_z = n_cells[2];
    int nc_xy = nc_x * nc_y;

    int cell_iz = cell_i / nc_xy;
    int cell_iy = (cell_i % nc_xy) / nc_x;
    int cell_ix = cell_i % nc_x;

    // Get particles in this cell
    int start_i = cell_starts[cell_i];
    int count_i = cell_counts[cell_i];

    // Iterate over all neighboring cells (including self)
    for (int dz = -n_search[2]; dz <= n_search[2]; dz++) {
        for (int dy = -n_search[1]; dy <= n_search[1]; dy++) {
            for (int dx = -n_search[0]; dx <= n_search[0]; dx++) {

                int cell_jx = cell_ix + dx;
                int cell_jy = cell_iy + dy;
                int cell_jz = cell_iz + dz;
                int32_t cell_shift[3] = {0, 0, 0};

                // Wrap cell indices with periodic boundary conditions
                if (periodic[0]) {
                    while (cell_jx < 0) {
                        cell_jx += nc_x;
                        cell_shift[0] -= 1;
                    }
                    while (cell_jx >= nc_x) {
                        cell_jx -= nc_x;
                        cell_shift[0] += 1;
                    }
                } else {
                    if (cell_jx < 0 || cell_jx >= nc_x) {
                        continue;
                    }
                }

                if (periodic[1]) {
                    while (cell_jy < 0) {
                        cell_jy += nc_y;
                        cell_shift[1] -= 1;
                    }
                    while (cell_jy >= nc_y) {
                        cell_jy -= nc_y;
                        cell_shift[1] += 1;
                    }
                } else {
                    if (cell_jy < 0 || cell_jy >= nc_y) {
                        continue;
                    }
                }

                if (periodic[2]) {
                    while (cell_jz < 0) {
                        cell_jz += nc_z;
                        cell_shift[2] -= 1;
                    }
                    while (cell_jz >= nc_z) {
                        cell_jz -= nc_z;
                        cell_shift[2] += 1;
                    }
                } else {
                    if (cell_jz < 0 || cell_jz >= nc_z) {
                        continue;
                    }
                }

                int cell_j = cell_jx + cell_jy * nc_x + cell_jz * nc_xy;
                int start_j = cell_starts[cell_j];
                int count_j = cell_counts[cell_j];

                // Total number of pairs between these two cells
                int n_pairs = count_i * count_j;

                // Distribute pairs among threads
                for (int pair_idx = tid; pair_idx < n_pairs; pair_idx += nthreads) {
                    int local_i = pair_idx / count_j;
                    int local_j = pair_idx % count_j;

                    int idx_i = start_i + local_i;
                    int idx_j = start_j + local_j;

                    int orig_i = sorted_indices[idx_i];
                    int orig_j = sorted_indices[idx_j];

                    // Compute total shift
                    int32_t shift_i[3] = {
                        sorted_shifts[idx_i * 3],
                        sorted_shifts[idx_i * 3 + 1],
                        sorted_shifts[idx_i * 3 + 2]
                    };
                    int32_t shift_j[3] = {
                        sorted_shifts[idx_j * 3],
                        sorted_shifts[idx_j * 3 + 1],
                        sorted_shifts[idx_j * 3 + 2]
                    };

                    int32_t total_shift[3];
                    total_shift[0] = shift_i[0] - shift_j[0] + cell_shift[0];
                    total_shift[1] = shift_i[1] - shift_j[1] + cell_shift[1];
                    total_shift[2] = shift_i[2] - shift_j[2] + cell_shift[2];

                    bool shift_is_zero = (total_shift[0] == 0 && total_shift[1] == 0 && total_shift[2] == 0);

                    // Skip self-pairs without shift
                    if (orig_i == orig_j && shift_is_zero) {
                        continue;
                    }

                    // For half list, apply filtering
                    if (!full_list) {
                        if (orig_i > orig_j) {
                            continue;
                        }
                        if (orig_i == orig_j) {
                            int shift_sum = total_shift[0] + total_shift[1] + total_shift[2];
                            if (shift_sum < 0) {
                                continue;
                            }
                            if (shift_sum == 0) {
                                if (total_shift[2] < 0 || (total_shift[2] == 0 && total_shift[1] < 0)) {
                                    continue;
                                }
                            }
                        }
                    }

                    // Load positions
                    double ri[3] = {
                        sorted_positions[idx_i * 3],
                        sorted_positions[idx_i * 3 + 1],
                        sorted_positions[idx_i * 3 + 2]
                    };
                    double rj[3] = {
                        sorted_positions[idx_j * 3],
                        sorted_positions[idx_j * 3 + 1],
                        sorted_positions[idx_j * 3 + 2]
                    };

                    // Compute shift in Cartesian coordinates
                    double shift_cart[3];
                    shift_cart[0] = total_shift[0] * box[0] + total_shift[1] * box[3] + total_shift[2] * box[6];
                    shift_cart[1] = total_shift[0] * box[1] + total_shift[1] * box[4] + total_shift[2] * box[7];
                    shift_cart[2] = total_shift[0] * box[2] + total_shift[1] * box[5] + total_shift[2] * box[8];

                    double vector[3];
                    vector[0] = rj[0] - ri[0] + shift_cart[0];
                    vector[1] = rj[1] - ri[1] + shift_cart[1];
                    vector[2] = rj[2] - ri[2] + shift_cart[2];

                    double dist2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];

                    if (dist2 < cutoff2 && dist2 > 0.0) {
                        size_t out_idx = atomicAdd_size_t(&length[0], 1);
                        pair_indices[out_idx * 2] = orig_i;
                        pair_indices[out_idx * 2 + 1] = orig_j;

                        if (return_shifts) {
                            shifts_out[out_idx * 3 + 0] = total_shift[0];
                            shifts_out[out_idx * 3 + 1] = total_shift[1];
                            shifts_out[out_idx * 3 + 2] = total_shift[2];
                        }
                        if (return_vectors) {
                            vectors[out_idx * 3 + 0] = vector[0];
                            vectors[out_idx * 3 + 1] = vector[1];
                            vectors[out_idx * 3 + 2] = vector[2];
                        }
                        if (return_distances) {
                            distances[out_idx] = sqrt(dist2);
                        }
                    }
                }
            }
        }
    }
}

// Legacy particle-centric kernel (kept for reference/comparison)
// Find neighbors using cell list. For small cells (< cutoff), the same cell
// is searched multiple times with different periodic shifts.
__global__ void find_neighbors_cell_list(
    const double* __restrict__ sorted_positions,
    const int* __restrict__ sorted_indices,
    const int32_t* __restrict__ sorted_shifts,
    const int* __restrict__ cell_indices,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    const int* __restrict__ n_search,
    size_t n_points,
    double cutoff,
    bool full_list,
    size_t* __restrict__ length,
    size_t* __restrict__ pair_indices,
    int32_t* __restrict__ shifts,
    double* __restrict__ distances,
    double* __restrict__ vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    double cutoff2 = cutoff * cutoff;

    double ri[3] = {
        sorted_positions[i * 3],
        sorted_positions[i * 3 + 1],
        sorted_positions[i * 3 + 2]
    };
    int32_t shift_i[3] = {
        sorted_shifts[i * 3],
        sorted_shifts[i * 3 + 1],
        sorted_shifts[i * 3 + 2]
    };
    int orig_i = sorted_indices[i];

    int cell_i = cell_indices[i];
    int n_cells_xy = n_cells[0] * n_cells[1];
    int cell_iz = cell_i / n_cells_xy;
    int cell_iy = (cell_i % n_cells_xy) / n_cells[0];
    int cell_ix = cell_i % n_cells[0];

    for (int dz = -n_search[2]; dz <= n_search[2]; dz++) {
        for (int dy = -n_search[1]; dy <= n_search[1]; dy++) {
            for (int dx = -n_search[0]; dx <= n_search[0]; dx++) {

                int cell_jx = cell_ix + dx;
                int cell_jy = cell_iy + dy;
                int cell_jz = cell_iz + dz;
                int32_t cell_shift[3] = {0, 0, 0};

                // Wrap cell indices into [0, n_cells) with periodic shifts
                if (periodic[0]) {
                    while (cell_jx < 0) {
                        cell_jx += n_cells[0];
                        cell_shift[0] -= 1;
                    }
                    while (cell_jx >= n_cells[0]) {
                        cell_jx -= n_cells[0];
                        cell_shift[0] += 1;
                    }
                } else {
                    if (cell_jx < 0 || cell_jx >= n_cells[0]) {
                        continue;
                    }
                }

                if (periodic[1]) {
                    while (cell_jy < 0) {
                        cell_jy += n_cells[1];
                        cell_shift[1] -= 1;
                    }
                    while (cell_jy >= n_cells[1]) {
                        cell_jy -= n_cells[1];
                        cell_shift[1] += 1;
                    }
                } else {
                    if (cell_jy < 0 || cell_jy >= n_cells[1]) {
                        continue;
                    }
                }

                if (periodic[2]) {
                    while (cell_jz < 0) {
                        cell_jz += n_cells[2];
                        cell_shift[2] -= 1;
                    }
                    while (cell_jz >= n_cells[2]) {
                        cell_jz -= n_cells[2];
                        cell_shift[2] += 1;
                    }
                } else {
                    if (cell_jz < 0 || cell_jz >= n_cells[2]) {
                        continue;
                    }
                }

                int cell_j = cell_jx + cell_jy * n_cells[0] + cell_jz * n_cells_xy;
                int start = cell_starts[cell_j];
                int count = cell_counts[cell_j];

                for (int k = start; k < start + count; k++) {
                    int orig_j = sorted_indices[k];

                    // total_shift combines particle wrapping shifts and cell boundary crossings
                    int32_t total_shift[3];
                    total_shift[0] = shift_i[0] - sorted_shifts[k * 3 + 0] + cell_shift[0];
                    total_shift[1] = shift_i[1] - sorted_shifts[k * 3 + 1] + cell_shift[1];
                    total_shift[2] = shift_i[2] - sorted_shifts[k * 3 + 2] + cell_shift[2];

                    bool shift_is_zero = (total_shift[0] == 0 && total_shift[1] == 0 && total_shift[2] == 0);

                    if (orig_i == orig_j && shift_is_zero) {
                        continue;
                    }

                    if (!full_list) {
                        if (orig_i > orig_j) {
                            continue;
                        }
                        if (orig_i == orig_j) {
                            // For self-pairs with periodic shifts, keep only positive half-space
                            int shift_sum = total_shift[0] + total_shift[1] + total_shift[2];
                            if (shift_sum < 0) {
                                continue;
                            }
                            if (shift_sum == 0) {
                                if (total_shift[2] < 0 || (total_shift[2] == 0 && total_shift[1] < 0)) {
                                    continue;
                                }
                            }
                        }
                    }

                    double rj[3] = {
                        sorted_positions[k * 3],
                        sorted_positions[k * 3 + 1],
                        sorted_positions[k * 3 + 2]
                    };

                    // shift_cart = total_shift @ box
                    double shift_cart[3];
                    shift_cart[0] = total_shift[0] * box[0] + total_shift[1] * box[3] + total_shift[2] * box[6];
                    shift_cart[1] = total_shift[0] * box[1] + total_shift[1] * box[4] + total_shift[2] * box[7];
                    shift_cart[2] = total_shift[0] * box[2] + total_shift[1] * box[5] + total_shift[2] * box[8];

                    double vector[3];
                    vector[0] = rj[0] - ri[0] + shift_cart[0];
                    vector[1] = rj[1] - ri[1] + shift_cart[1];
                    vector[2] = rj[2] - ri[2] + shift_cart[2];

                    double dist2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];

                    if (dist2 < cutoff2 && dist2 > 0.0) {
                        size_t idx = atomicAdd_size_t(&length[0], 1);
                        pair_indices[idx * 2] = orig_i;
                        pair_indices[idx * 2 + 1] = orig_j;

                        if (return_shifts) {
                            // Convention: D = positions[j] - positions[i] + shift @ box
                            shifts[idx * 3 + 0] = total_shift[0];
                            shifts[idx * 3 + 1] = total_shift[1];
                            shifts[idx * 3 + 2] = total_shift[2];
                        }
                        if (return_vectors) {
                            vectors[idx * 3 + 0] = vector[0];
                            vectors[idx * 3 + 1] = vector[1];
                            vectors[idx * 3 + 2] = vector[2];
                        }
                        if (return_distances) {
                            distances[idx] = sqrt(dist2);
                        }
                    }
                }
            }
        }
    }
}
