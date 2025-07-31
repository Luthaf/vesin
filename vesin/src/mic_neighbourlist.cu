#include "mic_neighbourlist.cuh"

#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define NWARPS 4
#define WARP_SIZE 32

__device__ inline unsigned long atomicAdd(unsigned long* address, unsigned long val) {
    unsigned long long* address_as_ull =
        reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, static_cast<unsigned long long>(val + static_cast<unsigned long>(assumed)));
    } while (assumed != old);

    return static_cast<unsigned long>(old);
}

// ops for vector types
__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ void invert_cell_matrix(const double* cell, double* inv_cell) {
    double a = cell[0], b = cell[1], c = cell[2];
    double d = cell[3], e = cell[4], f = cell[5];
    double g = cell[6], h = cell[7], i = cell[8];

    double det =
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = double(1.0) / det;

    inv_cell[0] = (e * i - f * h) * invdet;
    inv_cell[1] = (c * h - b * i) * invdet;
    inv_cell[2] = (b * f - c * e) * invdet;
    inv_cell[3] = (f * g - d * i) * invdet;
    inv_cell[4] = (a * i - c * g) * invdet;
    inv_cell[5] = (c * d - a * f) * invdet;
    inv_cell[6] = (d * h - e * g) * invdet;
    inv_cell[7] = (b * g - a * h) * invdet;
    inv_cell[8] = (a * e - b * d) * invdet;
}
__device__ void
apply_periodic_boundary(double3& displacement, int3& shift, const double* cell, const double* inv_cell) {
    using vec_t = double3;

    vec_t frac;
    frac.x = displacement.x * inv_cell[0] + displacement.y * inv_cell[1] +
             displacement.z * inv_cell[2];
    frac.y = displacement.x * inv_cell[3] + displacement.y * inv_cell[4] +
             displacement.z * inv_cell[5];
    frac.z = displacement.x * inv_cell[6] + displacement.y * inv_cell[7] +
             displacement.z * inv_cell[8];

    int sx = static_cast<int>(round(frac.x));
    int sy = static_cast<int>(round(frac.y));
    int sz = static_cast<int>(round(frac.z));

    shift.x = sx;
    shift.y = sy;
    shift.z = sz;

    frac.x -= sx;
    frac.y -= sy;
    frac.z -= sz;

    vec_t wrapped;
    wrapped.x = frac.x * cell[0] + frac.y * cell[3] + frac.z * cell[6];
    wrapped.y = frac.x * cell[1] + frac.y * cell[4] + frac.z * cell[7];
    wrapped.z = frac.x * cell[2] + frac.y * cell[5] + frac.z * cell[8];

    displacement = wrapped;
}

__global__ void compute_mic_neighbours_full_impl(
    const double* positions,
    const double* cell,
    long nnodes,
    double cutoff,
    unsigned long* pair_counter,
    unsigned long* edge_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {

    using vec_t = double3;

    __shared__ double scell[9];
    __shared__ double sinv_cell[9];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int thread_id = threadIdx.x % WARP_SIZE;

    const int node_index = blockIdx.x * NWARPS + warp_id;
    const double cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    if (cell != nullptr && threadIdx.x == 0)
        invert_cell_matrix(scell, sinv_cell);

    // Ensure inv_cell is ready
    __syncthreads();

    if (node_index >= nnodes)
        return;

    vec_t ri = *reinterpret_cast<const vec_t*>(&positions[node_index * 3]);

    for (long j = thread_id; j < nnodes; j += WARP_SIZE) {
        vec_t rj = *reinterpret_cast<const vec_t*>(&positions[j * 3]);

        vec_t disp = ri - rj;
        int3 shift = make_int3(0, 0, 0);
        if (cell != nullptr)
            apply_periodic_boundary(disp, shift, scell, sinv_cell);

        double dist2 = dot(disp, disp);
        bool is_valid = (dist2 < cutoff2 && dist2 > double(0.0));

        unsigned int mask = __activemask();
        unsigned int ballot = __ballot_sync(mask, is_valid);
        int local_offset = __popc(ballot & ((1U << thread_id) - 1));
        int warp_total = __popc(ballot);

        int base_edge_index = -1;
        if (is_valid && local_offset == 0) {
            base_edge_index = atomicAdd(&pair_counter[0], warp_total);
        }

        base_edge_index = __shfl_sync(mask, base_edge_index, __ffs(ballot) - 1);

        if (is_valid) {
            long edge_index = base_edge_index + local_offset;
            edge_indices[edge_index * 2 + 0] = node_index;
            edge_indices[edge_index * 2 + 1] = j;

            if (return_shifts) {
                reinterpret_cast<int3&>(shifts[edge_index * 3]) = shift;
            }
            if (return_vectors) {
                reinterpret_cast<vec_t&>(vectors[edge_index * 3]) = disp;
            }
            if (return_distances) {
                distances[edge_index] = sqrt(dist2);
            }
        }
    }
}

__global__ void compute_mic_neighbours_half_impl(
    const double* positions,
    const double* cell,
    long nnodes,
    double cutoff,
    unsigned long* pair_counter,
    unsigned long* edge_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {

    using vec_t = double3;

    const long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long num_all_pairs = nnodes * (nnodes - 1) / 2;

    if (index >= num_all_pairs)
        return;

    __shared__ double scell[9];
    __shared__ double sinv_cell[9];

    const double cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    if (cell != nullptr && threadIdx.x == 0)
        invert_cell_matrix(scell, sinv_cell);

    // Ensure inv_cell is ready
    __syncthreads();

    long row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index)
        row--;
    const long column = index - row * (row - 1) / 2;

    vec_t ri = *reinterpret_cast<const vec_t*>(&positions[column * 3]);
    vec_t rj = *reinterpret_cast<const vec_t*>(&positions[row * 3]);

    vec_t disp = ri - rj;
    int3 shift = make_int3(0, 0, 0);
    if (cell != nullptr)
        apply_periodic_boundary(disp, shift, scell, sinv_cell);

    double dist2 = dot(disp, disp);
    bool is_valid = (dist2 < cutoff2 && dist2 > double(0.0));

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_rank = threadIdx.x % WARP_SIZE;

    unsigned int mask = __activemask();
    unsigned int ballot = __ballot_sync(mask, is_valid);
    int local_offset = __popc(ballot & ((1U << warp_rank) - 1));
    int warp_total = __popc(ballot);

    int base_edge_index = -1;
    if (is_valid && local_offset == 0) {
        base_edge_index = atomicAdd(&pair_counter[0], warp_total);
    }

    base_edge_index = __shfl_sync(mask, base_edge_index, __ffs(ballot) - 1);

    if (is_valid) {
        long edge_index = base_edge_index + local_offset;
        edge_indices[edge_index * 2 + 0] = column;
        edge_indices[edge_index * 2 + 1] = row;

        if (return_shifts) {
            reinterpret_cast<int3&>(shifts[edge_index * 3]) = shift;
        }
        if (return_vectors) {
            reinterpret_cast<vec_t&>(vectors[edge_index * 3]) = disp;
        }
        if (return_distances) {
            distances[edge_index] = sqrt(dist2);
        }
    }
}

__global__ void mic_cell_check(const double* cell, const double cutoff, int* status) {

    __shared__ double scell[9];
    __shared__ double sinv_cell[9];

    const double cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {

        // Extract lattice vectors
        double ax = scell[0], ay = scell[1], az = scell[2];
        double bx = scell[3], by = scell[4], bz = scell[5];
        double cx = scell[6], cy = scell[7], cz = scell[8];

        // Compute norms
        double a_norm = sqrt(ax * ax + ay * ay + az * az);
        double b_norm = sqrt(bx * bx + by * by + bz * bz);
        double c_norm = sqrt(cx * cx + cy * cy + cz * cz);

        // Dot products
        double ab_dot = ax * bx + ay * by + az * bz;
        double ac_dot = ax * cx + ay * cy + az * cz;
        double bc_dot = bx * cx + by * cy + bz * cz;

        // Orthogonality check (relative tolerance)
        double tol = 1e-6;
        bool is_orthogonal = (fabs(ab_dot) < tol * a_norm * b_norm) &&
                             (fabs(ac_dot) < tol * a_norm * c_norm) &&
                             (fabs(bc_dot) < tol * b_norm * c_norm);

        double min_dim;

        if (is_orthogonal) {
            min_dim = fminf(a_norm, fminf(b_norm, c_norm));
        } else {
            // General case
            double bc_x = by * cz - bz * cy;
            double bc_y = bz * cx - bx * cz;
            double bc_z = bx * cy - by * cx;
            double ac_x = ay * cz - az * cy;
            double ac_y = az * cx - ax * cz;
            double ac_z = ax * cy - ay * cx;
            double ab_x = ay * bz - az * by;
            double ab_y = az * bx - ax * bz;
            double ab_z = ax * by - ay * bx;

            double bc_norm = sqrt(bc_x * bc_x + bc_y * bc_y + bc_z * bc_z);
            double ac_norm = sqrt(ac_x * ac_x + ac_y * ac_y + ac_z * ac_z);
            double ab_norm = sqrt(ab_x * ab_x + ab_y * ab_y + ab_z * ab_z);

            double V = fabs(ax * bc_x + ay * bc_y + az * bc_z);

            double d_a = V / bc_norm;
            double d_b = V / ac_norm;
            double d_c = V / ab_norm;

            min_dim = fminf(d_a, fminf(d_b, d_c));
        }

        if (cutoff * 2.0 > min_dim) {
            status[0] = 1; // ERROR
        } else {
            status[0] = 0;
        }
    }
}

void vesin::cuda::compute_mic_neighbourlist(
    const double (*points)[3],
    long n_points,
    const double cell[3][3],
    int* d_cell_check,
    VesinOptions options,
    VesinNeighborList& neighbors
) {

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    const double* d_positions = reinterpret_cast<const double*>(points);
    const double* d_cell = reinterpret_cast<const double*>(cell);

    unsigned long* d_pair_indices =
        reinterpret_cast<unsigned long*>(neighbors.pairs);
    int* d_shifts = reinterpret_cast<int*>(neighbors.shifts);
    double* d_distances = reinterpret_cast<double*>(neighbors.distances);
    double* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    unsigned long* d_pair_counter = extras->length_ptr;

    dim3 blockDim(WARP_SIZE * NWARPS);

    mic_cell_check<<<1, 32>>>(d_cell, options.cutoff, d_cell_check);
    int h_cell_check = 0;
    cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_cell_check != 0) {
        throw std::runtime_error("Invalid cutoff: too large for cell dimensions");
    }

    if (options.full) {
        dim3 gridDim(max((int)(n_points + NWARPS - 1) / NWARPS, 1));

        compute_mic_neighbours_full_impl<<<gridDim, blockDim>>>(
            d_positions,
            d_cell,
            n_points,
            options.cutoff,
            d_pair_counter,
            d_pair_indices,
            d_shifts,
            d_distances,
            d_vectors,
            options.return_shifts,
            options.return_distances,
            options.return_vectors
        );

    } else {
        const long num_all_pairs = n_points * (n_points - 1) / 2;
        int threads_per_block = WARP_SIZE * NWARPS;
        int num_blocks =
            (num_all_pairs + threads_per_block - 1) / threads_per_block;
        dim3 gridDim(max(num_blocks, 1));

        compute_mic_neighbours_half_impl<<<gridDim, blockDim>>>(
            d_positions,
            d_cell,
            n_points,
            options.cutoff,
            d_pair_counter,
            d_pair_indices,
            d_shifts,
            d_distances,
            d_vectors,
            options.return_shifts,
            options.return_distances,
            options.return_vectors
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}