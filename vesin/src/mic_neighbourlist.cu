#include "mic_neighbourlist.cuh"

#include "vesin_cuda.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#define NWARPS 4
#define WARP_SIZE 32

__device__ inline size_t atomicAdd(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            static_cast<unsigned long long>(val + static_cast<size_t>(assumed))
        );
    } while (assumed != old);

    return static_cast<size_t>(old);
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
__device__ void apply_periodic_boundary(
    double3& vector,
    int3& shift,
    const double* cell,
    const double* inv_cell,
    char3 periodic_mask
) {
    double3 fractional;
    fractional.x = vector.x * inv_cell[0] + vector.y * inv_cell[1] + vector.z * inv_cell[2];
    fractional.y = vector.x * inv_cell[3] + vector.y * inv_cell[4] + vector.z * inv_cell[5];
    fractional.z = vector.x * inv_cell[6] + vector.y * inv_cell[7] + vector.z * inv_cell[8];

    int32_t sx = static_cast<int32_t>(round(fractional.x));
    int32_t sy = static_cast<int32_t>(round(fractional.y));
    int32_t sz = static_cast<int32_t>(round(fractional.z));

    if (periodic_mask.x != 0) {
        shift.x = sx;
        fractional.x -= sx;
    }
    if (periodic_mask.y != 0) {
        shift.y = sy;
        fractional.y -= sy;
    }
    if (periodic_mask.z != 0) {
        shift.z = sz;
        fractional.z -= sz;
    }

    double3 wrapped;
    wrapped.x = fractional.x * cell[0] + fractional.y * cell[3] + fractional.z * cell[6];
    wrapped.y = fractional.x * cell[1] + fractional.y * cell[4] + fractional.z * cell[7];
    wrapped.z = fractional.x * cell[2] + fractional.y * cell[5] + fractional.z * cell[8];

    vector = wrapped;
}

__global__ void compute_mic_neighbours_full_impl(
    const double* positions,
    const double* cell,
    char3 periodic_mask,
    size_t n_points,
    double cutoff,
    size_t* length,
    size_t* pair_indices,
    int32_t* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    __shared__ double scell[9];
    __shared__ double sinv_cell[9];
    const int32_t warp_id = threadIdx.x / WARP_SIZE;
    const int32_t thread_id = threadIdx.x % WARP_SIZE;

    const int32_t point_i = blockIdx.x * NWARPS + warp_id;
    const double cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    if (cell != nullptr && threadIdx.x == 0) {
        invert_cell_matrix(scell, sinv_cell);
    }

    // Ensure inv_cell is ready
    __syncthreads();

    if (point_i >= n_points) {
        return;
    }

    double3 ri = *reinterpret_cast<const double3*>(&positions[point_i * 3]);

    for (size_t j = thread_id; j < n_points; j += WARP_SIZE) {
        double3 rj = *reinterpret_cast<const double3*>(&positions[j * 3]);

        double3 vector = ri - rj;
        int3 shift = make_int3(0, 0, 0);
        if (cell != nullptr) {
            apply_periodic_boundary(vector, shift, scell, sinv_cell, periodic_mask);
        }

        double distance2 = dot(vector, vector);
        auto is_valid = (distance2 < cutoff2 && distance2 > double(0.0));

        auto mask = __activemask();
        auto ballot = __ballot_sync(mask, is_valid);
        auto local_offset = __popc(ballot & ((1U << thread_id) - 1));
        auto warp_total = __popc(ballot);

        auto base_pair_index = -1;
        if (is_valid && local_offset == 0) {
            base_pair_index = atomicAdd(&length[0], warp_total);
        }

        base_pair_index = __shfl_sync(mask, base_pair_index, __ffs(ballot) - 1);
        if (is_valid) {
            size_t current_pair = base_pair_index + local_offset;
            pair_indices[current_pair * 2 + 0] = point_i;
            pair_indices[current_pair * 2 + 1] = j;

            if (return_shifts) {
                reinterpret_cast<int3&>(shifts[current_pair * 3]) = shift;
            }
            if (return_vectors) {
                reinterpret_cast<double3&>(vectors[current_pair * 3]) = vector;
            }
            if (return_distances) {
                distances[current_pair] = sqrt(distance2);
            }
        }
    }
}

__global__ void compute_mic_neighbours_half_impl(
    const double* positions,
    const double* cell,
    char3 periodic_mask,
    size_t n_points,
    double cutoff,
    size_t* length,
    size_t* pair_indices,
    int32_t* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_all_pairs = n_points * (n_points - 1) / 2;

    if (index >= num_all_pairs) {
        return;
    }

    __shared__ double scell[9];
    __shared__ double sinv_cell[9];

    const double cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    if (cell != nullptr && threadIdx.x == 0) {
        invert_cell_matrix(scell, sinv_cell);
    }

    // Ensure inv_cell is ready
    __syncthreads();

    size_t row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index) {
        row--;
    }
    const size_t column = index - row * (row - 1) / 2;

    double3 ri = *reinterpret_cast<const double3*>(&positions[column * 3]);
    double3 rj = *reinterpret_cast<const double3*>(&positions[row * 3]);

    double3 vector = ri - rj;
    int3 shift = make_int3(0, 0, 0);
    if (cell != nullptr) {
        apply_periodic_boundary(vector, shift, scell, sinv_cell, periodic_mask);
    }

    double distance2 = dot(vector, vector);
    bool is_valid = (distance2 < cutoff2 && distance2 > double(0.0));

    int32_t warp_rank = threadIdx.x % WARP_SIZE;

    uint32_t mask = __activemask();
    uint32_t ballot = __ballot_sync(mask, is_valid);
    int32_t local_offset = __popc(ballot & ((1U << warp_rank) - 1));
    int32_t warp_total = __popc(ballot);

    int32_t base_pair_index = -1;
    if (is_valid && local_offset == 0) {
        base_pair_index = atomicAdd(&length[0], warp_total);
    }

    base_pair_index = __shfl_sync(mask, base_pair_index, __ffs(ballot) - 1);

    if (is_valid) {
        size_t pair_index = base_pair_index + local_offset;
        pair_indices[pair_index * 2 + 0] = column;
        pair_indices[pair_index * 2 + 1] = row;

        if (return_shifts) {
            reinterpret_cast<int3&>(shifts[pair_index * 3]) = shift;
        }
        if (return_vectors) {
            reinterpret_cast<double3&>(vectors[pair_index * 3]) = vector;
        }
        if (return_distances) {
            distances[pair_index] = sqrt(distance2);
        }
    }
}

__global__ void mic_cell_check(const double* cell, const double cutoff, int32_t* status) {

    __shared__ double scell[9];

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
    size_t n_points,
    const double cell[3][3],
    const bool periodic[3],
    int32_t* d_cell_check,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    const double* d_positions = reinterpret_cast<const double*>(points);
    const double* d_cell = reinterpret_cast<const double*>(cell);

    size_t* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    int32_t* d_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    double* d_distances = reinterpret_cast<double*>(neighbors.distances);
    double* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    size_t* d_pair_counter = extras->length_ptr;

    dim3 blockDim(WARP_SIZE * NWARPS);

    mic_cell_check<<<1, 32>>>(d_cell, options.cutoff, d_cell_check);
    int32_t h_cell_check = 0;
    cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (h_cell_check != 0) {
        throw std::runtime_error(
            "Cutoff it too large for the current box, the CUDA implementation "
            "of vesin uses minimum image convention (MIC). Each box dimension "
            "must be at least twice the cutoff."
        );
    }

    char3 periodic_mask = make_char3(periodic[0], periodic[1], periodic[2]);

    if (options.full) {
        dim3 gridDim(max((int32_t)(n_points + NWARPS - 1) / NWARPS, 1));

        compute_mic_neighbours_full_impl<<<gridDim, blockDim>>>(
            d_positions,
            d_cell,
            periodic_mask,
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
        size_t num_all_pairs = n_points * (n_points - 1) / 2;
        auto threads_per_block = WARP_SIZE * NWARPS;
        auto num_blocks = static_cast<unsigned long long>(
            (num_all_pairs + threads_per_block - 1) / threads_per_block
        );
        dim3 gridDim(max(num_blocks, 1ull));

        compute_mic_neighbours_half_impl<<<gridDim, blockDim>>>(
            d_positions,
            d_cell,
            periodic_mask,
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

    // set the length from the cuda extra data
    cudaMemcpy(&neighbors.length, d_pair_counter, sizeof(size_t), cudaMemcpyDeviceToHost);
}
