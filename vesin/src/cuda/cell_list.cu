#include "cell_list.cuh"

// Cell list neighbor finding: bin particles into cells, then search neighboring cells.
// Particles are sorted by cell for memory coalescing. Multiple threads per particle
// cooperate on neighbor search. Output buffering reduces atomic contention.

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    auto* address_as_ull = reinterpret_cast<uint64*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<uint64>(val)));
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

// Block size for the two bounding box kernels. Must be a power of two, for the
// tree reduction below. Keep `BOUNDING_BOX_BLOCK` in vesin_cuda.cpp in sync.
#define BOUNDING_BOX_BLOCK 256

__device__ const double BOUNDING_BOX_MAX = 1e300;
__device__ const double BOUNDING_BOX_MIN = -1e300;

// Reduce `local_min` / `local_max` across the block, leaving the result in
// element 0 of the shared arrays. `blockDim.x` must be a power of two.
__device__ inline void reduce_bounding_box_block(
    double shared_min[3][BOUNDING_BOX_BLOCK],
    double shared_max[3][BOUNDING_BOX_BLOCK],
    const double local_min[3],
    const double local_max[3]
) {
    int32 tid = static_cast<int32>(threadIdx.x);
    for (int32 dim = 0; dim < 3; dim++) {
        shared_min[dim][tid] = local_min[dim];
        shared_max[dim][tid] = local_max[dim];
    }
    __syncthreads();

    for (int32 stride = static_cast<int32>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int32 dim = 0; dim < 3; dim++) {
                shared_min[dim][tid] = min(shared_min[dim][tid], shared_min[dim][tid + stride]);
                shared_max[dim][tid] = max(shared_max[dim][tid], shared_max[dim][tid + stride]);
            }
        }
        __syncthreads();
    }
}

// Turn a min/max corner into the outputs the cell list wants.
__device__ inline void write_bounding_box(
    const double* corner_min,
    const double* corner_max,
    double* __restrict__ face_distances,
    double* __restrict__ bounding_min
) {
    for (int32 dim = 0; dim < 3; dim++) {
        bounding_min[dim] = corner_min[dim];

        // Add 1% margin to bounding box size to ensure particles on the
        // edge are included in cells
        face_distances[dim] = (corner_max[dim] - corner_min[dim]) * 1.01;

        // make sure the distance is not too small (to prevent searching too
        // many cells down the line). This can happen if all point are in
        // the same plane in this direction
        if (face_distances[dim] < 1.0) {
            face_distances[dim] = 1.0;
        }
    }
}

// First step of `compute_bounding_box`: per-block partial bounding boxes.
//
// When launched with a single block there is nothing left to combine, so it
// writes the final answer directly and the caller can skip the second step.
__global__ void compute_bounding_box(
    const double (*__restrict__ points)[3],
    size_t n_points,
    double* __restrict__ partials,
    double* __restrict__ face_distances,
    double* __restrict__ bounding_min
) {
    __shared__ double shared_min[3][BOUNDING_BOX_BLOCK];
    __shared__ double shared_max[3][BOUNDING_BOX_BLOCK];

    double local_min[3] = {BOUNDING_BOX_MAX, BOUNDING_BOX_MAX, BOUNDING_BOX_MAX};
    double local_max[3] = {BOUNDING_BOX_MIN, BOUNDING_BOX_MIN, BOUNDING_BOX_MIN};

    const auto* pos3 = reinterpret_cast<const double3*>(points);
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = gridDim.x * blockDim.x;
    for (size_t idx = start; idx < n_points; idx += step) {
        double3 point = pos3[idx];
        local_min[0] = min(local_min[0], point.x);
        local_min[1] = min(local_min[1], point.y);
        local_min[2] = min(local_min[2], point.z);
        local_max[0] = max(local_max[0], point.x);
        local_max[1] = max(local_max[1], point.y);
        local_max[2] = max(local_max[2], point.z);
    }

    reduce_bounding_box_block(shared_min, shared_max, local_min, local_max);

    if (threadIdx.x != 0) {
        return;
    }

    double corner_min[3] = {shared_min[0][0], shared_min[1][0], shared_min[2][0]};
    double corner_max[3] = {shared_max[0][0], shared_max[1][0], shared_max[2][0]};

    if (gridDim.x == 1) {
        write_bounding_box(corner_min, corner_max, face_distances, bounding_min);
    } else {
        auto* out = partials + blockIdx.x * 6;
        for (int32 dim = 0; dim < 3; dim++) {
            out[dim] = corner_min[dim];
            out[3 + dim] = corner_max[dim];
        }
    }
}

// Second step of `compute_bounding_box`: combine the partials bounding boxes.
// Launched with a single block; only needed when the first step used more than
// one block.
__global__ void reduce_bounding_box(
    const double* __restrict__ partials,
    size_t n_partials,
    double* __restrict__ face_distances,
    double* __restrict__ bounding_min
) {
    __shared__ double shared_min[3][BOUNDING_BOX_BLOCK];
    __shared__ double shared_max[3][BOUNDING_BOX_BLOCK];

    double local_min[3] = {BOUNDING_BOX_MAX, BOUNDING_BOX_MAX, BOUNDING_BOX_MAX};
    double local_max[3] = {BOUNDING_BOX_MIN, BOUNDING_BOX_MIN, BOUNDING_BOX_MIN};

    for (size_t i = threadIdx.x; i < n_partials; i += blockDim.x) {
        const auto* in = partials + i * 6;
        for (int32 dim = 0; dim < 3; dim++) {
            local_min[dim] = min(local_min[dim], in[dim]);
            local_max[dim] = max(local_max[dim], in[3 + dim]);
        }
    }

    reduce_bounding_box_block(shared_min, shared_max, local_min, local_max);

    if (threadIdx.x == 0) {
        double corner_min[3] = {shared_min[0][0], shared_min[1][0], shared_min[2][0]};
        double corner_max[3] = {shared_max[0][0], shared_max[1][0], shared_max[2][0]};
        write_bounding_box(corner_min, corner_max, face_distances, bounding_min);
    }
}


// True when no point of the cell `(dx, dy, dz)` cells away can be within the
// cutoff of any point of the cell at the origin, so the whole cell can be
// skipped without looking at what is in it.
//
// Two points, one in each cell, are separated by `(delta + w) @ cell` for some
// `w` in `[-1, 1]^3`, so projecting on `v = delta @ cell` bounds their distance
// from below by
//
//     |v| - sum_d |cell_d . v| / |v|
//
// Writing `gap = |v|^2 - sum_d |cell_d . v|`, the cell is out of reach when
// `gap > 0 && gap^2 >= cutoff^2 * |v|^2`, which needs no square root. The cell
// itself (`v == 0`) always passes.
__device__ inline bool cell_offset_out_of_reach(
    const double3& cell_a,
    const double3& cell_b,
    const double3& cell_c,
    int32 dx,
    int32 dy,
    int32 dz,
    double cutoff2
) {
    double3 v = make_double3(
        cell_a.x * dx + cell_b.x * dy + cell_c.x * dz,
        cell_a.y * dx + cell_b.y * dy + cell_c.y * dz,
        cell_a.z * dx + cell_b.z * dy + cell_c.z * dz
    );

    double v2 = dot3(v, v);
    double reach = abs(dot3(cell_a, v)) + abs(dot3(cell_b, v)) + abs(dot3(cell_c, v));
    double gap = v2 - reach;

    return gap > 0.0 && gap * gap >= cutoff2 * v2;
}

// How many entries are in the cell reachability mask.
//
// A box much thinner than the cutoff can still exceed it, in which case no
// pruning will be done.
#define MAX_OUT_OF_REACH_MASK 32768

// Block size of `compute_cell_grid_params`. One thread picks the grid, then the
// whole block fills the offset mask, so this only has to be large enough to
// cover `MAX_OUT_OF_REACH_MASK` in a few iterations each. Must match
// `GRID_PARAMS_BLOCK` in vesin_cuda.cpp.
#define GRID_PARAMS_BLOCK 256

// Shrink `n_cells` until the three dimensions multiply to `max_cells` or less,
// and return that product.
__device__ inline double fit_cells_in_budget(double n_cells[3], double max_cells) {
    double total = n_cells[0] * n_cells[1] * n_cells[2];
    if (total <= max_cells) {
        return total;
    }

    // Scale all three at once to get close in one step,
    double ratio = cbrt(max_cells / total);
    for (int32 dim = 0; dim < 3; dim++) {
        n_cells[dim] = max(1.0, floor(n_cells[dim] * ratio));
    }
    total = n_cells[0] * n_cells[1] * n_cells[2];

    // then walk the largest dimension down, since rounding each dimension
    // separately can leave the product just above the budget.
    while (total > max_cells) {
        int32 largest = 0;
        if (n_cells[1] > n_cells[largest]) {
            largest = 1;
        }
        if (n_cells[2] > n_cells[largest]) {
            largest = 2;
        }

        if (n_cells[largest] <= 1.0) {
            break;
        }

        n_cells[largest] -= 1.0;
        total = n_cells[0] * n_cells[1] * n_cells[2];
    }

    return total;
}

__global__ void compute_cell_grid_params(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    size_t n_points,
    size_t max_cells,
    double cell_visit_cost,
    double inv_box[3][3],
    int32* __restrict__ n_cells,
    int32* __restrict__ n_search,
    int32* __restrict__ n_cells_total,
    double* __restrict__ face_distances,
    bool* __restrict__ cell_out_of_reach
) {
    if (blockIdx.x != 0) {
        return;
    }

    // The grid is picked by a single thread and handed to the rest of the block
    // through shared memory.
    __shared__ double shared_cell_vectors[9];
    __shared__ int32 shared_n_search[3];

    // Only one thread picks the grid; the rest wait, then fill the offset mask
    // together at the end.
    if (threadIdx.x == 0) {
        // Normalize non-periodic box directions like the CPU BoundingBox logic,
        // then use the normalized matrix for inverse and distance calculations.
        double3 rows[3] = {
            make_double3(box[0][0], box[0][1], box[0][2]),
            make_double3(box[1][0], box[1][1], box[1][2]),
            make_double3(box[2][0], box[2][1], box[2][2]),
        };

        int32 n_periodic = 0;
        int32 periodic_idx_1 = -1;
        int32 periodic_idx_2 = -1;
        for (int32 d = 0; d < 3; d++) {
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

        // Pick how many cells to put across one cutoff.
        //
        // Cells one cutoff wide is the textbook choice, but then the search sweeps
        // a 3x3x3 block of them, i.e. (3 * cutoff)^3 of volume, where the cutoff
        // sphere only occupies (4 pi / 3) * cutoff^3 -- a factor of 6.4 more
        // candidate pairs than there are neighbors to find. Splitting each cutoff
        // into `k` cells brings the swept span per dimension down from 3 * cutoff
        // towards the (2 + 1/k) * cutoff floor, at the price of (2k+1)^3 cell
        // visits instead of 27.
        //
        // That trade only pays off when the cells hold enough atoms for the
        // candidates to dominate the cell visits. Minimising
        //
        //     cost(k) = atoms_per_cell * (2 + 1/k)^3 + cell_visit_cost * (2k+1)^3
        //
        // over k gives k = (atoms_per_cell / (2 * cell_visit_cost))^(1/4), where
        // `atoms_per_cell` is measured on the one-cell-per-cutoff grid.
        auto atoms_per_cell = (double)n_points;
        for (int32 dim = 0; dim < 3; dim++) {
            double count = floor(face_distances[dim] / cutoff);
            count = min(max(count, 1.0), (double)max_cells);
            atoms_per_cell /= count;
        }

        double k = sqrt(sqrt(atoms_per_cell / (2.0 * cell_visit_cost)));

        // Cap cells_per_cutoff to 6, to prevent having too many cells to search in
        // extreme cases.
        int32 cells_per_cutoff = max(1, min((int32)round(k), 6));

        // Number of cells to put in each dimension.
        double n_cells_per_dim[3];
        for (int32 dim = 0; dim < 3; dim++) {
            // Refining a non-periodic dimension under mixed periodicity buys
            // nothing: `n_search` there is widened below to cover the whole extent
            // regardless, so a finer grid sweeps the same atoms in more cells.
            int32 dim_cells_per_cutoff = cells_per_cutoff;
            if (n_periodic > 0 && n_periodic < 3 && !periodic[dim]) {
                dim_cells_per_cutoff = 1;
            }

            double count = floor(dim_cells_per_cutoff * face_distances[dim] / cutoff);
            n_cells_per_dim[dim] = min(max(count, 1.0), (double)max_cells);
        }

        double total = fit_cells_in_budget(n_cells_per_dim, (double)max_cells);

        // `max_cells` is at most 2^21, so both the per-dimension counts and the
        // total fit in an `int` by this point.
        n_cells[0] = (int32)n_cells_per_dim[0];
        n_cells[1] = (int32)n_cells_per_dim[1];
        n_cells[2] = (int32)n_cells_per_dim[2];
        n_cells_total[0] = (int32)total;

        // Compute search range - how many cells to search in each direction
        // When cells are larger than cutoff, we need to search more cells
        for (int32 dim = 0; dim < 3; dim++) {
            double cell_size = face_distances[dim] / n_cells[dim];
            n_search[dim] = max(1, (int32)ceil(cutoff / cell_size));

            if (!periodic[dim]) {
                // A non-periodic direction never searches past its own extent, so
                // we can reduce the search size.
                int32 extent = n_cells[dim] - 1;
                n_search[dim] = (n_periodic > 0) ? extent : min(n_search[dim], extent);
            }
        }

        // Cartesian vector spanning one cell along each grid direction, used
        // just below to decide which cell offsets are out of reach.
        for (int32 i = 0; i < 9; i++) {
            shared_cell_vectors[i] = 0.0;
        }

        if (n_periodic == 3) {
            for (int32 dim = 0; dim < 3; dim++) {
                double scale = 1.0 / n_cells[dim];
                shared_cell_vectors[3 * dim + 0] = rows[dim].x * scale;
                shared_cell_vectors[3 * dim + 1] = rows[dim].y * scale;
                shared_cell_vectors[3 * dim + 2] = rows[dim].z * scale;
            }
        } else if (n_periodic == 0) {
            for (int32 dim = 0; dim < 3; dim++) {
                shared_cell_vectors[3 * dim + dim] = face_distances[dim] / n_cells[dim];
            }
        }

        for (int32 i = 0; i < 3; i++) {
            shared_n_search[i] = n_search[i];
        }
    }

    __syncthreads();

    // Evaluate `cell_offset_out_of_reach` once per cell offset, into a mask
    // that `traverse_neighbors` reads back as a single bool per offset.
    auto cell_a = make_double3(shared_cell_vectors[0], shared_cell_vectors[1], shared_cell_vectors[2]);
    auto cell_b = make_double3(shared_cell_vectors[3], shared_cell_vectors[4], shared_cell_vectors[5]);
    auto cell_c = make_double3(shared_cell_vectors[6], shared_cell_vectors[7], shared_cell_vectors[8]);

    bool can_prune_offsets = (cell_a.x != 0.0) || (cell_a.y != 0.0) || (cell_a.z != 0.0);

    int64 n_x = 2 * shared_n_search[0] + 1;
    int64 n_y = 2 * shared_n_search[1] + 1;
    int64 n_z = 2 * shared_n_search[2] + 1;
    int64 n_offsets = n_x * n_y * n_z;

    if (n_offsets > MAX_OUT_OF_REACH_MASK) {
        return;
    }

    double cutoff2 = cutoff * cutoff;
    for (int64 i = threadIdx.x; i < n_offsets; i += blockDim.x) {
        if (!can_prune_offsets) {
            cell_out_of_reach[i] = false;
            continue;
        }

        int32 dx = (int32)(i % n_x) - shared_n_search[0];
        int32 dy = (int32)((i / n_x) % n_y) - shared_n_search[1];
        int32 dz = (int32)(i / (n_x * n_y)) - shared_n_search[2];

        cell_out_of_reach[i] = cell_offset_out_of_reach(cell_a, cell_b, cell_c, dx, dy, dz, cutoff2);
    }
}

__global__ void assign_cell_indices(
    const double (*__restrict__ points)[3],
    size_t n_points,
    const double inv_box[3][3],
    const bool periodic[3],
    const int32* __restrict__ n_cells,
    const double* __restrict__ face_distances,
    const double* __restrict__ bounding_min,
    int32* __restrict__ cell_indices,
    int32* __restrict__ particle_shifts
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
    for (int32 d = 0; d < 3; d++) {
        if (!periodic[d]) {
            frac[d] = (pos_arr[d] - bounding_min[d]) / face_distances[d];
        }
    }

    int32 cell_idx[3];
    int32 shift[3];

    for (int32 d = 0; d < 3; d++) {
        // Use divmod for both periodic and non-periodic dimensions.
        // For non-periodic dimensions, the fractional coordinates should be
        // in [0, 1) due to the bounding box margin, so shift will be 0.
        shift[d] = static_cast<int32>(floor(frac[d]));
        frac[d] -= shift[d];

        cell_idx[d] = static_cast<int32>(frac[d] * n_cells[d]);
        cell_idx[d] = max(0, min(n_cells[d] - 1, cell_idx[d]));
    }

    cell_indices[i] = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1];
    particle_shifts[i * 3 + 0] = shift[0];
    particle_shifts[i * 3 + 1] = shift[1];
    particle_shifts[i * 3 + 2] = shift[2];
}

__global__ void count_particles_per_cell(
    const int32* __restrict__ cell_indices,
    size_t n_points,
    int32* __restrict__ cell_counts
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }
    atomicAdd(&cell_counts[cell_indices[i]], 1);
}

__global__ void scatter_particles(
    const double (*__restrict__ points)[3],
    size_t n_points,
    const int32* __restrict__ cell_indices,
    const int32* __restrict__ particle_shifts,
    const double box[3][3],
    int32* __restrict__ cell_offsets,
    double* __restrict__ sorted_points,
    double* __restrict__ wrapped_positions,
    int32* __restrict__ sorted_indices,
    int32* __restrict__ sorted_shifts,
    int32* __restrict__ sorted_cell_indices
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    int32 cell = cell_indices[i];
    int32 slot = atomicAdd(&cell_offsets[cell], 1);

    // Vectorized points copy
    const auto* pos_in = reinterpret_cast<const double3*>(points);
    auto* pos_out = reinterpret_cast<double3*>(sorted_points);
    auto* wrapped_out = reinterpret_cast<double3*>(wrapped_positions);
    const double3 pos = pos_in[i];
    pos_out[slot] = pos;

    // Compute wrapped position: wrapped = pos - shift @ box
    int32 shift_x = particle_shifts[i * 3 + 0];
    int32 shift_y = particle_shifts[i * 3 + 1];
    int32 shift_z = particle_shifts[i * 3 + 2];
    const auto* box3 = reinterpret_cast<const double3*>(box);
    double3 box_a = box3[0];
    double3 box_b = box3[1];
    double3 box_c = box3[2];
    wrapped_out[slot] = make_double3(
        pos.x - (shift_x * box_a.x + shift_y * box_b.x + shift_z * box_c.x),
        pos.y - (shift_x * box_a.y + shift_y * box_b.y + shift_z * box_c.y),
        pos.z - (shift_x * box_a.z + shift_y * box_b.z + shift_z * box_c.z)
    );

    sorted_indices[slot] = static_cast<int32>(i);
    sorted_shifts[slot * 3 + 0] = shift_x;
    sorted_shifts[slot * 3 + 1] = shift_y;
    sorted_shifts[slot * 3 + 2] = shift_z;
    sorted_cell_indices[slot] = cell;
}

// -------------------------------------------------------------------------- //
// Two-pass count/fill neighbor search
//
// Pass 1 (count_neighbors_cell_list): one thread per sorted position, counts
//   neighbors and writes d_counts[orig_i].
// Pass 2 (fill_neighbors_cell_list): same traversal, writes pairs/shifts/
//   distances/vectors at pre-computed offsets from d_write_offsets[orig_i].
//
// The output is naturally sorted by orig_i (the prefix sum assigns contiguous
// write slots per original atom), so no separate sort pass is needed.
// -------------------------------------------------------------------------- //

// `THREADS_PER_PARTICLE` is the number of threads cooperating on one atom in
// the neighbor search, and the resulting number of atoms a block covers. Must
// divide 32, so that a group never straddles two warps.
//
// Keep `CELL_LIST_BLOCK` and `THREADS_PER_PARTICLE` in vesin_cuda.cpp in sync
// with these: the host sizes the launch from them, and they size the shared
// arrays here.
#define CELL_LIST_BLOCK 256
#define THREADS_PER_PARTICLE 8
#define PARTICLES_PER_BLOCK (CELL_LIST_BLOCK / THREADS_PER_PARTICLE)

// Everything about the cell grid that is the same for every thread of a block.
// Loading it once into shared memory, instead of once per thread into
// registers, keeps this data out of the traversal's register budget. That
// budget is what decides how many blocks fit on an SM, and the traversal is
// large enough for it to bind on big systems.
struct SharedCellGrid {
    double3 box_a;
    double3 box_b;
    double3 box_c;

    int32 n_cells_x;
    int32 n_cells_y;
    int32 n_cells_z;
    int32 n_cells_xy;

    int32 n_search_x;
    int32 n_search_y;
    int32 n_search_z;

    int32 n_search_cells_x;
    int32 n_search_cells_y;
    int32 n_search_cells_z;

    bool pbc_x;
    bool pbc_y;
    bool pbc_z;

    bool use_offset_mask;
};

// Fill `grid` from the device buffers. Every thread of the block must call
// this, and a `__syncthreads()` must follow before `grid` is read.
__device__ inline void load_cell_grid(
    SharedCellGrid& grid,
    const double box[3][3],
    const bool periodic[3],
    const int32* __restrict__ n_cells,
    const int32* __restrict__ n_search
) {
    if (threadIdx.x != 0) {
        return;
    }

    const auto* box_d3 = reinterpret_cast<const double3*>(box);
    grid.box_a = box_d3[0];
    grid.box_b = box_d3[1];
    grid.box_c = box_d3[2];

    grid.n_cells_x = n_cells[0];
    grid.n_cells_y = n_cells[1];
    grid.n_cells_z = n_cells[2];
    grid.n_cells_xy = grid.n_cells_x * grid.n_cells_y;

    grid.n_search_x = n_search[0];
    grid.n_search_y = n_search[1];
    grid.n_search_z = n_search[2];

    grid.n_search_cells_x = 2 * grid.n_search_x + 1;
    grid.n_search_cells_y = 2 * grid.n_search_y + 1;
    grid.n_search_cells_z = 2 * grid.n_search_z + 1;

    grid.pbc_x = periodic[0];
    grid.pbc_y = periodic[1];
    grid.pbc_z = periodic[2];

    int64 n_offsets = (int64)grid.n_search_cells_x
                        * (int64)grid.n_search_cells_y
                        * (int64)grid.n_search_cells_z;
    grid.use_offset_mask = n_offsets <= MAX_OUT_OF_REACH_MASK;
}

// Shared neighbor traversal logic. `THREADS_PER_PARTICLE` threads cooperate on
// the sorted position `sorted_i`, each walking the subset of neighbor cells
// given by its `thread_in_group` (0 <= thread_in_group < THREADS_PER_PARTICLE).
//
// `Emit` is a functor called as emit(orig_j, shift_x, shift_y, shift_z, vec,
// dist2) for each pair that passes the cutoff and half-list filters.
template <typename Emit>
__device__ inline void traverse_neighbors(
    size_t sorted_i,
    int32 thread_in_group,
    const double* __restrict__ wrapped_positions,
    const SharedCellGrid& grid,
    const bool* __restrict__ cell_out_of_reach,
    const int32* __restrict__ sorted_indices,
    const int32* __restrict__ sorted_shifts,
    const int32* __restrict__ sorted_cell_indices,
    const int32* __restrict__ cell_starts,
    const int32* __restrict__ cell_counts,
    double cutoff2,
    bool full_list,
    Emit&& emit
) {
    const auto* wrapped_positions_d3 = reinterpret_cast<const double3*>(wrapped_positions);
    auto wrapped_i = wrapped_positions_d3[sorted_i];
    auto shift_i_x = sorted_shifts[sorted_i * 3 + 0];
    auto shift_i_y = sorted_shifts[sorted_i * 3 + 1];
    auto shift_i_z = sorted_shifts[sorted_i * 3 + 2];

    int32 cell_i = sorted_cell_indices[sorted_i];
    int32 cell_i_z = cell_i / grid.n_cells_xy;
    int32 cell_i_y = (cell_i % grid.n_cells_xy) / grid.n_cells_x;
    int32 cell_i_x = cell_i % grid.n_cells_x;

    // The (dx, dy, dz) offsets of the cells to search form a box of
    // `n_search_cells_x * n_search_cells_y * n_search_cells_z` entries. Each
    // thread of the group walks a strided subset of that box: it decodes its
    // starting point from the flattened index once, then advances by
    // `n_threads`.
    int32 index_x = thread_in_group % grid.n_search_cells_x;
    int32 index_rest = thread_in_group / grid.n_search_cells_x;
    int32 index_y = index_rest % grid.n_search_cells_y;
    int32 index_z = index_rest / grid.n_search_cells_y;
    int32 flat_offset = thread_in_group;

    // `index_z >= n_search_cells_z` already here means there are fewer cells to
    // search than threads in the group, and this thread has nothing to do.
    while (index_z < grid.n_search_cells_z) {
        int32 dx = index_x - grid.n_search_x;
        int32 dy = index_y - grid.n_search_y;
        int32 dz = index_z - grid.n_search_z;

        // Advance to the next cell handled by this thread. This is done up
        // front so that the `continue`s below can not skip it.
        index_x += THREADS_PER_PARTICLE;
        int32 this_offset = flat_offset;
        flat_offset += THREADS_PER_PARTICLE;
        while (index_x >= grid.n_search_cells_x) {
            index_x -= grid.n_search_cells_x;
            index_y += 1;
        }
        while (index_y >= grid.n_search_cells_y) {
            index_y -= grid.n_search_cells_y;
            index_z += 1;
        }

        // For a half list, visit only a canonical half of the offset box: for
        // every nonzero offset keep exactly one of `{v, -v}`, the one whose
        // first nonzero component is positive in (x, y, z) order. Each pair of
        // cells is then visited once instead of twice, and no pair has to be
        // discarded after the fact.
        //
        // `(0, 0, 0)` is the same-cell case, which is kept and handled by the
        // `first_k` bound below.
        //
        // The test sits before the wrap arithmetic and the `cell_starts` /
        // `cell_counts` gathers, so a skipped offset costs three comparisons
        // rather than a whole cell visit.
        //
        // Must stay identical to `build_half_offsets()` in cpu_cell_list.cpp:
        // which half is canonical is arbitrary, but it decides which way round
        // a pair comes out, and CPU and CUDA output is compared pair by pair.
        bool same_cell = (dx == 0 && dy == 0 && dz == 0);
        if (!full_list && !same_cell) {
            bool canonical = (dx > 0)
                || (dx == 0 && dy > 0)
                || (dx == 0 && dy == 0 && dz > 0);

            if (!canonical) {
                continue;
            }
        }

        // Skip cells too far away to hold any neighbor of this one.
        if (grid.use_offset_mask && cell_out_of_reach[this_offset]) {
            continue;
        }

        int32 cell_j_x = cell_i_x + dx;
        int32 cell_j_y = cell_i_y + dy;
        int32 cell_j_z = cell_i_z + dz;
        int32 cell_shift_x = 0;
        int32 cell_shift_y = 0;
        int32 cell_shift_z = 0;

        if (grid.pbc_x) {
            while (cell_j_x < 0) {
                cell_j_x += grid.n_cells_x;
                cell_shift_x -= 1;
            }

            while (cell_j_x >= grid.n_cells_x) {
                cell_j_x -= grid.n_cells_x;
                cell_shift_x += 1;
            }
        } else if (cell_j_x < 0 || cell_j_x >= grid.n_cells_x) {
            continue;
        }

        if (grid.pbc_y) {
            while (cell_j_y < 0) {
                cell_j_y += grid.n_cells_y;
                cell_shift_y -= 1;
            }

            while (cell_j_y >= grid.n_cells_y) {
                cell_j_y -= grid.n_cells_y;
                cell_shift_y += 1;
            }
        } else if (cell_j_y < 0 || cell_j_y >= grid.n_cells_y) {
            continue;
        }

        if (grid.pbc_z) {
            while (cell_j_z < 0) {
                cell_j_z += grid.n_cells_z;
                cell_shift_z -= 1;
            }

            while (cell_j_z >= grid.n_cells_z) {
                cell_j_z -= grid.n_cells_z;
                cell_shift_z += 1;
            }
        } else if (cell_j_z < 0 || cell_j_z >= grid.n_cells_z) {
            continue;
        }

        // Full cell shift vector: cell_shift @ box
        double3 cell_shift_vec = make_double3(
            grid.box_a.x * cell_shift_x + grid.box_b.x * cell_shift_y + grid.box_c.x * cell_shift_z,
            grid.box_a.y * cell_shift_x + grid.box_b.y * cell_shift_y + grid.box_c.y * cell_shift_z,
            grid.box_a.z * cell_shift_x + grid.box_b.z * cell_shift_y + grid.box_c.z * cell_shift_z
        );

        int32 cell_j = cell_j_x + cell_j_y * grid.n_cells_x + cell_j_z * grid.n_cells_xy;
        int32 start_j = cell_starts[cell_j];
        int32 count_j = cell_counts[cell_j];

        // Within its own cell a half list takes only the atoms after this one,
        // which is what makes each within-cell pair appear exactly once. A full
        // list takes them all, skipping only the atom itself.
        int32 first_k = start_j;
        if (!full_list && same_cell) {
            first_k = max(start_j, static_cast<int32>(sorted_i) + 1);
        }

        for (int32 k = first_k; k < start_j + count_j; k++) {
            if (full_list && same_cell && k == static_cast<int32>(sorted_i)) {
                continue;
            }

            // Distance from wrapped positions (no per-pair matrix multiply)
            const double3 wrapped_j = wrapped_positions_d3[k];
            double3 vec = make_double3(
                wrapped_j.x - wrapped_i.x + cell_shift_vec.x,
                wrapped_j.y - wrapped_i.y + cell_shift_vec.y,
                wrapped_j.z - wrapped_i.z + cell_shift_vec.z
            );
            double dist2 = dot3(vec, vec);

            if (dist2 >= cutoff2) {
                continue;
            }

            // Compute shift only for pairs that pass the cutoff
            int32 orig_j = sorted_indices[k];
            int32 shift_j_x = sorted_shifts[k * 3 + 0];
            int32 shift_j_y = sorted_shifts[k * 3 + 1];
            int32 shift_j_z = sorted_shifts[k * 3 + 2];

            int32 total_shift_x = shift_i_x - shift_j_x + cell_shift_x;
            int32 total_shift_y = shift_i_y - shift_j_y + cell_shift_y;
            int32 total_shift_z = shift_i_z - shift_j_z + cell_shift_z;

            emit(orig_j, total_shift_x, total_shift_y, total_shift_z, vec, dist2);
        }
    }
}

// Which atom this thread helps with, and where it sits in its group.
struct ParticleGroup {
    size_t sorted_i;     // sorted position of the atom handled by the group
    int32 thread_in_group; // 0 <= thread_in_group < THREADS_PER_PARTICLE
    int32 slot;            // index of the group in the per-block shared arrays
};

__device__ inline ParticleGroup particle_group() {
    int32 lane = static_cast<int32>(threadIdx.x) % 32;
    int32 warp = static_cast<int32>(threadIdx.x) / 32;

    ParticleGroup group;
    group.thread_in_group = lane % THREADS_PER_PARTICLE;
    group.slot = warp * (32 / THREADS_PER_PARTICLE) + lane / THREADS_PER_PARTICLE;
    group.sorted_i = static_cast<size_t>(blockIdx.x) * PARTICLES_PER_BLOCK +
                     static_cast<size_t>(group.slot);

    return group;
}

// Pass 1: count neighbors per original atom.
__global__ void count_neighbors_cell_list(
    const double* __restrict__ wrapped_positions,
    size_t n_points,
    const double box[3][3],
    const bool* __restrict__ cell_out_of_reach,
    const bool periodic[3],
    const int32* __restrict__ n_cells,
    const int32* __restrict__ n_search,
    const int32* __restrict__ sorted_indices,
    const int32* __restrict__ sorted_shifts,
    const int32* __restrict__ sorted_cell_indices,
    const int32* __restrict__ cell_starts,
    const int32* __restrict__ cell_counts,
    double cutoff,
    bool full_list,
    int32 padded_to,
    int32* __restrict__ padding_overflow,
    int32* __restrict__ counts  // [n_points] output
) {
    __shared__ int32 shared_counts[PARTICLES_PER_BLOCK];
    __shared__ SharedCellGrid grid;

    auto group = particle_group();

    if (static_cast<int32>(threadIdx.x) < PARTICLES_PER_BLOCK) {
        shared_counts[threadIdx.x] = 0;
    }
    load_cell_grid(grid, box, periodic, n_cells, n_search);
    __syncthreads();

    // Threads past the end of the system still take part in the reduction and
    // the barriers below, they just contribute a count of zero.
    int32 count = 0;
    if (group.sorted_i < n_points) {
        traverse_neighbors(
            group.sorted_i,
            group.thread_in_group,
            wrapped_positions,
            grid,
            cell_out_of_reach,
            sorted_indices,
            sorted_shifts,
            sorted_cell_indices,
            cell_starts,
            cell_counts,
            cutoff * cutoff,
            full_list,
            [&](int32, int32, int32, int32, double3, double) { count++; }
        );
    }

    atomicAdd(&shared_counts[group.slot], count);
    __syncthreads();

    if (group.thread_in_group == 0 && group.sorted_i < n_points) {
        int32 point_i = sorted_indices[group.sorted_i];
        int32 point_count = shared_counts[group.slot];
        counts[point_i] = point_count;

        // Record the first point with too many neighbors, so the error message
        // does not depend on scheduling
        if (padded_to > 0 && point_count > padded_to) {
            atomicMin(padding_overflow, point_i);
        }
    }
}

// Pass 2: fill output arrays at pre-computed offsets.
__global__ void fill_neighbors_cell_list(
    const double* __restrict__ wrapped_positions,
    size_t n_points,
    const double box[3][3],
    const bool* __restrict__ cell_out_of_reach,
    const bool periodic[3],
    const int32* __restrict__ n_cells,
    const int32* __restrict__ n_search,
    const int32* __restrict__ sorted_indices,
    const int32* __restrict__ sorted_shifts,
    const int32* __restrict__ sorted_cell_indices,
    const int32* __restrict__ cell_starts,
    const int32* __restrict__ cell_counts,
    double cutoff,
    bool full_list,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    int32 padded_to,
    const int32* __restrict__ write_offsets,
    size_t (*pair_indices)[2],
    int32 (*shifts_out)[3],
    double* distances,
    double (*vectors)[3]
) {
    // Write cursor for each group, so that the threads sharing an atom hand out
    // distinct slots inside that atom's [write_offsets[i], +count_i) range.
    __shared__ int32 shared_cursors[PARTICLES_PER_BLOCK];
    __shared__ SharedCellGrid grid;

    auto group = particle_group();

    int32 point_i = -1;
    if (group.sorted_i < n_points) {
        point_i = sorted_indices[group.sorted_i];
    }

    if (group.thread_in_group == 0 && group.sorted_i < n_points) {
        if (padded_to > 0) {
            // With padding, each atom owns exactly `padded_to` slots
            shared_cursors[group.slot] = point_i * padded_to;
        } else {
            shared_cursors[group.slot] = write_offsets[point_i];
        }
    }
    load_cell_grid(grid, box, periodic, n_cells, n_search);
    __syncthreads();

    // Threads past the end of the system can not return early, since they
    // still need to reach the barrier below when padding.
    if (group.sorted_i < n_points) {
        traverse_neighbors(
            group.sorted_i,
            group.thread_in_group,
            wrapped_positions,
            grid,
            cell_out_of_reach,
            sorted_indices,
            sorted_shifts,
            sorted_cell_indices,
            cell_starts,
            cell_counts,
            cutoff * cutoff,
            full_list,
            [&](int32 point_j, int32 sx, int32 sy, int32 sz, double3 vec, double dist2) {
                int32 offset = atomicAdd(&shared_cursors[group.slot], 1);

                pair_indices[offset][0] = point_i;
                pair_indices[offset][1] = point_j;
                if (return_shifts) {
                    shifts_out[offset][0] = sx;
                    shifts_out[offset][1] = sy;
                    shifts_out[offset][2] = sz;
                }
                if (return_distances) {
                    distances[offset] = sqrt(dist2);
                }
                if (return_vectors) {
                    vectors[offset][0] = vec.x;
                    vectors[offset][1] = vec.y;
                    vectors[offset][2] = vec.z;
                }
            }
        );
    }

    if (padded_to > 0) {
        // `padded_to` is the same for the whole block, so all threads reach
        // this barrier. After it, the cursor of each group points just after
        // the last actual neighbor of its atom.
        __syncthreads();

        if (group.sorted_i < n_points) {
            int32 end = (point_i + 1) * padded_to;
            int32 start = shared_cursors[group.slot] + group.thread_in_group;
            for (int32 offset = start; offset < end; offset += THREADS_PER_PARTICLE) {
                pair_indices[offset][0] = point_i;
                pair_indices[offset][1] = static_cast<size_t>(-1);
                if (return_shifts) {
                    shifts_out[offset][0] = 0;
                    shifts_out[offset][1] = 0;
                    shifts_out[offset][2] = 0;
                }
                if (return_distances) {
                    distances[offset] = 0.0;
                }
                if (return_vectors) {
                    vectors[offset][0] = 0.0;
                    vectors[offset][1] = 0.0;
                    vectors[offset][2] = 0.0;
                }
            }
        }
    }
}
