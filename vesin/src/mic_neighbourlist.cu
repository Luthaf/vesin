#include "mic_neighbourlist.cuh"

#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define NWARPS 4
#define WARP_SIZE 32

#ifndef MAX_NEIGHBOURS_PER_ATOM
#define MAX_NEIGHBOURS_PER_ATOM 1024 // Make configurable
#endif

__device__ inline long atomicAdd(long* address, long val) {
    unsigned long long* address_as_ull =
        reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed, static_cast<unsigned long long>(val + static_cast<long>(assumed))
        );
    } while (assumed != old);

    return static_cast<long>(old);
}

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

// ops for vector type deduction
__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Vector3IO template structure for handling vectorized types
template <typename scalar_t>
struct Vector3IO;

/* template structure for dealing with float3, double3 vectorized types */
template <>
struct Vector3IO<float> {
    using scalar_t = float;
    using vec_t = float3;

    __device__ static void unpack(const vec_t& v, scalar_t& x0, scalar_t& x1, scalar_t& x2) {
        x0 = v.x;
        x1 = v.y;
        x2 = v.z;
    }

    __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2) {
        return {x0, x1, x2};
    }
};

template <>
struct Vector3IO<double> {
    using scalar_t = double;
    using vec_t = double3;

    __device__ static void unpack(const vec_t& v, scalar_t& x0, scalar_t& x1, scalar_t& x2) {
        x0 = v.x;
        x1 = v.y;
        x2 = v.z;
    }

    __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2) {
        return {x0, x1, x2};
    }
};

template <typename scalar_t>
__device__ typename Vector3IO<scalar_t>::vec_t
operator+(const typename Vector3IO<scalar_t>::vec_t& a, const typename Vector3IO<scalar_t>::vec_t& b) {
    return Vector3IO<scalar_t>::pack(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <typename scalar_t>
__device__ typename Vector3IO<scalar_t>::vec_t
operator-(const typename Vector3IO<scalar_t>::vec_t& a, const typename Vector3IO<scalar_t>::vec_t& b) {
    return Vector3IO<scalar_t>::pack(a.x - b.x, a.y - b.y, a.z - b.z);
}

template <typename scalar_t>
__device__ scalar_t dot(const typename Vector3IO<scalar_t>::vec_t& a, const typename Vector3IO<scalar_t>::vec_t& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename scalar_t>
__device__ void check_rcut(const scalar_t* cell, scalar_t rcut) {
    // Extract lattice vectors
    scalar_t ax = cell[0], ay = cell[1], az = cell[2];
    scalar_t bx = cell[3], by = cell[4], bz = cell[5];
    scalar_t cx = cell[6], cy = cell[7], cz = cell[8];

    // Compute norms
    scalar_t a_norm = sqrt(ax * ax + ay * ay + az * az);
    scalar_t b_norm = sqrt(bx * bx + by * by + bz * bz);
    scalar_t c_norm = sqrt(cx * cx + cy * cy + cz * cz);

    // Dot products
    scalar_t ab_dot = ax * bx + ay * by + az * bz;
    scalar_t ac_dot = ax * cx + ay * cy + az * cz;
    scalar_t bc_dot = bx * cx + by * cy + bz * cz;

    // Orthogonality check (relative tolerance)
    scalar_t tol = 1e-6;
    bool is_orthogonal = (fabs(ab_dot) < tol * a_norm * b_norm) &&
                         (fabs(ac_dot) < tol * a_norm * c_norm) &&
                         (fabs(bc_dot) < tol * b_norm * c_norm);

    scalar_t min_dim;

    if (is_orthogonal) {
        min_dim = fminf(a_norm, fminf(b_norm, c_norm));
    } else {
        // General triclinic case
        scalar_t bc_x = by * cz - bz * cy;
        scalar_t bc_y = bz * cx - bx * cz;
        scalar_t bc_z = bx * cy - by * cx;
        scalar_t ac_x = ay * cz - az * cy;
        scalar_t ac_y = az * cx - ax * cz;
        scalar_t ac_z = ax * cy - ay * cx;
        scalar_t ab_x = ay * bz - az * by;
        scalar_t ab_y = az * bx - ax * bz;
        scalar_t ab_z = ax * by - ay * bx;

        scalar_t bc_norm = sqrt(bc_x * bc_x + bc_y * bc_y + bc_z * bc_z);
        scalar_t ac_norm = sqrt(ac_x * ac_x + ac_y * ac_y + ac_z * ac_z);
        scalar_t ab_norm = sqrt(ab_x * ab_x + ab_y * ab_y + ab_z * ab_z);

        scalar_t V = fabs(ax * bc_x + ay * bc_y + az * bc_z);

        scalar_t d_a = V / bc_norm;
        scalar_t d_b = V / ac_norm;
        scalar_t d_c = V / ab_norm;

        min_dim = fminf(d_a, fminf(d_b, d_c));
    }

    if (rcut * 2.0 > min_dim) {
        printf("ERROR: rcut (%g) must be <= half the smallest cell dimension (%g)\n", (double)rcut, (double)(min_dim * 0.5));
        assert(false);
    }
}

template <typename scalar_t>
__device__ void invert_cell_matrix(const scalar_t* cell, scalar_t* inv_cell) {
    scalar_t a = cell[0], b = cell[1], c = cell[2];
    scalar_t d = cell[3], e = cell[4], f = cell[5];
    scalar_t g = cell[6], h = cell[7], i = cell[8];

    scalar_t det =
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    scalar_t invdet = scalar_t(1.0) / det;

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
template <typename scalar_t>
__device__ void
apply_periodic_boundary(typename Vector3IO<scalar_t>::vec_t& displacement, int3& shift, const scalar_t* cell, const scalar_t* inv_cell) {
    using vec_t = typename Vector3IO<scalar_t>::vec_t;

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

template <typename scalar_t>
__global__ void compute_mic_neighbours_full_impl(
    const scalar_t* positions, const scalar_t* cell, long nnodes, scalar_t cutoff, unsigned long* pair_counter, unsigned long* edge_indices, int* shifts, scalar_t* distances, scalar_t* vectors, bool return_shifts, bool return_distances, bool return_vectors
) {

    using vec_t = typename Vector3IO<scalar_t>::vec_t;

    __shared__ scalar_t scell[9];
    __shared__ scalar_t sinv_cell[9];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int thread_id = threadIdx.x % WARP_SIZE;

    const int node_index = blockIdx.x * NWARPS + warp_id;
    const scalar_t cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    check_rcut(cell, cutoff);

    if (cell != nullptr && thread_id == 0 && warp_id == 0)
        invert_cell_matrix(scell, sinv_cell);

    // Ensure inv_cell is ready
    __syncthreads();

    if (node_index >= nnodes)
        return;
    vec_t ri = Vector3IO<scalar_t>::pack(positions[node_index * 3 + 0], positions[node_index * 3 + 1], positions[node_index * 3 + 2]);

    // vec_t ri = *reinterpret_cast<const vec_t *>(&positions[node_index * 3]);

    for (long j = thread_id; j < nnodes; j += WARP_SIZE) {
        vec_t rj = Vector3IO<scalar_t>::pack(
            positions[j * 3 + 0], positions[j * 3 + 1], positions[j * 3 + 2]
        );
        // vec_t rj = *reinterpret_cast<const vec_t *>(&positions[j * 3]);

        vec_t disp = ri - rj;
        int3 shift = make_int3(0, 0, 0);
        if (cell != nullptr)
            apply_periodic_boundary<scalar_t>(disp, shift, scell, sinv_cell);

        scalar_t dist2 = dot(disp, disp);
        bool is_valid = (dist2 < cutoff2 && dist2 > scalar_t(0.0));

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

template <typename scalar_t>
__global__ void compute_mic_neighbours_half_impl(
    const scalar_t* positions, const scalar_t* cell, long nnodes, scalar_t cutoff, unsigned long* pair_counter, unsigned long* edge_indices, int* shifts, scalar_t* distances, scalar_t* vectors, bool return_shifts, bool return_distances, bool return_vectors
) {

    using vec_t = typename Vector3IO<scalar_t>::vec_t;

    const long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long num_all_pairs = nnodes * (nnodes - 1) / 2;

    if (index >= num_all_pairs)
        return;

    __shared__ scalar_t scell[9];
    __shared__ scalar_t sinv_cell[9];

    const scalar_t cutoff2 = cutoff * cutoff;

    if (cell != nullptr) {
        if (threadIdx.x < 9) {
            scell[threadIdx.x] = cell[threadIdx.x];
        }
    }

    __syncthreads();

    check_rcut(scell, cutoff);

    if (cell != nullptr && threadIdx.x == 0)
        invert_cell_matrix(scell, sinv_cell);

    // Ensure inv_cell is ready
    __syncthreads();

    long row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index)
        row--;
    const long column = index - row * (row - 1) / 2;

    vec_t ri = Vector3IO<scalar_t>::pack(positions[column * 3 + 0], positions[column * 3 + 1], positions[column * 3 + 2]);
    vec_t rj = Vector3IO<scalar_t>::pack(
        positions[row * 3 + 0], positions[row * 3 + 1], positions[row * 3 + 2]
    );

    // vec_t ri = *reinterpret_cast<const vec_t *>(&positions[column * 3]);
    // vec_t rj = *reinterpret_cast<const vec_t *>(&positions[row * 3]);

    vec_t disp = ri - rj;
    int3 shift = make_int3(0, 0, 0);
    if (cell != nullptr)
        apply_periodic_boundary<scalar_t>(disp, shift, scell, sinv_cell);

    scalar_t dist2 = dot(disp, disp);
    bool is_valid = (dist2 < cutoff2 && dist2 > scalar_t(0.0));

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
            int* sh_ptr = &shifts[edge_index * 3];
            sh_ptr[0] = shift.x;
            sh_ptr[1] = shift.y;
            sh_ptr[2] = shift.z;
            // reinterpret_cast<int3 &>(shifts[edge_index * 3]) = shift;
        }
        if (return_vectors) {
            scalar_t* vec_ptr = &vectors[edge_index * 3];
            vec_ptr[0] = disp.x;
            vec_ptr[1] = disp.y;
            vec_ptr[2] = disp.z;
            // reinterpret_cast<vec_t &>(vectors[edge_index * 3]) = disp;
        }
        if (return_distances) {
            distances[edge_index] = sqrt(dist2);
        }
    }
}

static void ensure_is_device_pointer(const void* p, const char* name) {

    if (!p) {
        throw std::runtime_error(std::string(name) + " is not defined.");
        return;
    }

    cudaPointerAttributes attr;

    cudaError_t err = cudaPointerGetAttributes(&attr, p);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaPointerGetAttributes failed for ") + name + ": " +
            cudaGetErrorString(err)
        );
    }
    if (attr.type != cudaMemoryTypeDevice) {
        throw std::runtime_error(
            std::string(name) +
            " is not a device pointer (type=" + std::to_string(attr.type) + ")"
        );
    }
}

template <typename scalar_t>
void vesin::cuda::compute_mic_neighbourlist(
    const scalar_t* positions, const scalar_t* cell, long nnodes, scalar_t cutoff, unsigned long* pair_counter, unsigned long* edge_indices, int* shifts, scalar_t* distances, scalar_t* vectors, bool return_shifts, bool return_distances, bool return_vectors, bool full
) {

    dim3 blockDim(WARP_SIZE * NWARPS);

    // --- BEGIN DEVICE-PTR CHECKS ---
    ensure_is_device_pointer(positions, "points");
    ensure_is_device_pointer(cell, "cell");
    ensure_is_device_pointer(edge_indices, "pairs");
    ensure_is_device_pointer(shifts, "shifts");
    ensure_is_device_pointer(distances, "distances");
    ensure_is_device_pointer(vectors, "vectors");
    ensure_is_device_pointer(pair_counter, "length_ptr");
    // --- END DEVICE-PTR CHECKS ---

    if (full) {
        dim3 gridDim(max((int)(nnodes + NWARPS - 1) / NWARPS, 1));
        compute_mic_neighbours_full_impl<scalar_t><<<gridDim, blockDim>>>(
            positions, cell, nnodes, cutoff, pair_counter, edge_indices, shifts, distances, vectors, return_shifts, return_distances, return_vectors
        );
    } else {
        const long num_all_pairs = nnodes * (nnodes - 1) / 2;
        int threads_per_block = WARP_SIZE * NWARPS;
        int num_blocks =
            (num_all_pairs + threads_per_block - 1) / threads_per_block;
        dim3 gridDim(max(num_blocks, 1));

        compute_mic_neighbours_half_impl<scalar_t><<<gridDim, blockDim>>>(
            positions, cell, nnodes, cutoff, pair_counter, edge_indices, shifts, distances, vectors, return_shifts, return_distances, return_vectors
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Explicit instantiation for double
template void vesin::cuda::compute_mic_neighbourlist<double>(
    const double* positions, const double* cell, long nnodes, double cutoff, unsigned long* pair_counter, unsigned long* edge_indices, int* shifts, double* distances, double* vectors, bool return_shifts, bool return_distances, bool return_vectors, bool full
);

// Explicit instantiation for float
template void vesin::cuda::compute_mic_neighbourlist<float>(
    const float* positions, const float* cell, long nnodes, float cutoff, unsigned long* pair_counter, unsigned long* edge_indices, int* shifts, float* distances, float* vectors, bool return_shifts, bool return_distances, bool return_vectors, bool full
);