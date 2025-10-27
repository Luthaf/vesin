#include "mic_neighbourlist.cuh"

#include "vesin_cuda.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#define NWARPS 4
#define WARP_SIZE 32

static_assert(sizeof(int3) == 3 * sizeof(int32_t));
static_assert(sizeof(ulong2) == 2 * sizeof(size_t));

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
__device__ inline double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 cross(double3 a, double3 b) {
    auto x = a.y * b.z - a.z * b.y;
    auto y = a.z * b.x - a.x * b.z;
    auto z = a.x * b.y - a.y * b.x;
    return {x, y, z};
}

__device__ inline double norm(double3 a) {
    return sqrt(dot(a, a));
}

__device__ void invert_matrix(const double3 matrix[3], double3 inverse[3]) {
    double a = matrix[0].x, b = matrix[0].y, c = matrix[0].z;
    double d = matrix[1].x, e = matrix[1].y, f = matrix[1].z;
    double g = matrix[2].x, h = matrix[2].y, i = matrix[2].z;

    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = double(1.0) / det;

    inverse[0] = {
        (e * i - f * h) * invdet,
        (c * h - b * i) * invdet,
        (b * f - c * e) * invdet
    };
    inverse[1] = {
        (f * g - d * i) * invdet,
        (a * i - c * g) * invdet,
        (c * d - a * f) * invdet
    };
    inverse[2] = {
        (d * h - e * g) * invdet,
        (b * g - a * h) * invdet,
        (a * e - b * d) * invdet
    };
}
__device__ void apply_periodic_boundary(
    double3& vector,
    int3& shift,
    const double3 box[3],
    const double3 inv_box[3],
    const bool periodic[3]
) {
    double3 fractional;
    fractional.x = dot(vector, inv_box[0]);
    fractional.y = dot(vector, inv_box[1]);
    fractional.z = dot(vector, inv_box[2]);

    // the multiplication by `periodic` allows to set the shift to zero
    // for non-periodic directions
    shift.x = static_cast<int32_t>(periodic[0]) * static_cast<int32_t>(round(fractional.x));
    shift.y = static_cast<int32_t>(periodic[1]) * static_cast<int32_t>(round(fractional.y));
    shift.z = static_cast<int32_t>(periodic[2]) * static_cast<int32_t>(round(fractional.z));

    fractional.x -= shift.x;
    fractional.y -= shift.y;
    fractional.z -= shift.z;

    double3 wrapped;
    wrapped.x = fractional.x * box[0].x + fractional.y * box[1].x + fractional.z * box[2].x;
    wrapped.y = fractional.x * box[0].y + fractional.y * box[1].y + fractional.z * box[2].y;
    wrapped.z = fractional.x * box[0].z + fractional.y * box[1].z + fractional.z * box[2].z;

    vector = wrapped;
}

__global__ void compute_mic_neighbours_full_impl(
    const double3* positions,
    const double3 box[3],
    const bool periodic[3],
    size_t n_points,
    double cutoff,
    size_t* length,
    ulong2* pair_indices,
    int3* shifts,
    double* distances,
    double3* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];
    const int32_t warp_id = threadIdx.x / WARP_SIZE;
    const int32_t thread_id = threadIdx.x % WARP_SIZE;

    const size_t point_i = blockIdx.x * NWARPS + warp_id;
    const double cutoff2 = cutoff * cutoff;

    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = box[threadIdx.x];

        if (threadIdx.x == 0 && !periodic[0]) {
            shared_box[0] = make_double3(1, 0, 0);
        } else if (threadIdx.x == 1 && !periodic[1]) {
            shared_box[1] = make_double3(0, 1, 0);
        } else if (threadIdx.x == 2 && !periodic[2]) {
            shared_box[2] = make_double3(0, 0, 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        invert_matrix(shared_box, shared_inv_box);
    }

    // Ensure inv_box is ready
    __syncthreads();

    if (point_i >= n_points) {
        return;
    }

    double3 ri = positions[point_i];

    for (size_t j = thread_id; j < n_points; j += WARP_SIZE) {
        double3 rj = positions[j];

        double3 vector = rj - ri;
        int3 shift = make_int3(0, 0, 0);
        apply_periodic_boundary(vector, shift, shared_box, shared_inv_box, periodic);

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
            pair_indices[current_pair] = {point_i, j};

            if (return_shifts) {
                shifts[current_pair] = shift;
            }
            if (return_vectors) {
                vectors[current_pair] = vector;
            }
            if (return_distances) {
                distances[current_pair] = sqrt(distance2);
            }
        }
    }
}

__global__ void compute_mic_neighbours_half_impl(
    const double3* positions,
    const double3 box[3],
    const bool periodic[3],
    size_t n_points,
    double cutoff,
    size_t* length,
    ulong2* pair_indices,
    int3* shifts,
    double* distances,
    double3* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_all_pairs = n_points * (n_points - 1) / 2;

    if (index >= num_all_pairs) {
        return;
    }

    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];

    const double cutoff2 = cutoff * cutoff;

    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = box[threadIdx.x];

        if (threadIdx.x == 0 && !periodic[0]) {
            shared_box[0] = make_double3(1, 0, 0);
        } else if (threadIdx.x == 1 && !periodic[1]) {
            shared_box[1] = make_double3(0, 1, 0);
        } else if (threadIdx.x == 2 && !periodic[2]) {
            shared_box[2] = make_double3(0, 0, 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        invert_matrix(shared_box, shared_inv_box);
    }

    // Ensure inv_cell is ready
    __syncthreads();

    size_t point_j = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (point_j * (point_j - 1) > 2 * index) {
        point_j--;
    }
    const size_t point_i = index - point_j * (point_j - 1) / 2;

    double3 ri = positions[point_i];
    double3 rj = positions[point_j];

    double3 vector = rj - ri;
    int3 shift = make_int3(0, 0, 0);
    apply_periodic_boundary(vector, shift, shared_box, shared_inv_box, periodic);

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
        pair_indices[pair_index] = {point_i, point_j};

        if (return_shifts) {
            shifts[pair_index] = shift;
        }
        if (return_vectors) {
            vectors[pair_index] = vector;
        }
        if (return_distances) {
            distances[pair_index] = sqrt(distance2);
        }
    }
}

// possible error for mic_box_check
#define CUTOFF_TOO_LARGE 1
#define NOT_ALIGNED_LATTICE_A 2
#define NOT_ALIGNED_LATTICE_B 3
#define NOT_ALIGNED_LATTICE_C 4

__global__ void mic_box_check(
    const double3 box[3],
    const bool periodic[3],
    const double cutoff,
    int32_t* status
) {
    __shared__ double3 shared_box[3];

    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = box[threadIdx.x];
    }

    __syncthreads();

    auto a = shared_box[0];
    auto b = shared_box[1];
    auto c = shared_box[2];

    if (threadIdx.x == 0) {
        // Compute norms
        double a_norm = norm(a);
        double b_norm = norm(b);
        double c_norm = norm(c);

        // Dot products
        double ab_dot = dot(a, b);
        double ac_dot = dot(a, c);
        double bc_dot = dot(b, c);

        // Orthogonality check (relative tolerance)
        double tol = 1e-6;
        bool is_orthogonal = (fabs(ab_dot) < tol * a_norm * b_norm) &&
                             (fabs(ac_dot) < tol * a_norm * c_norm) &&
                             (fabs(bc_dot) < tol * b_norm * c_norm);

        double min_dim = INFINITY;
        if (is_orthogonal) {
            if (periodic[0]) {
                min_dim = a_norm;
            }

            if (periodic[1]) {
                min_dim = fminf(min_dim, b_norm);
            }

            if (periodic[2]) {
                min_dim = fminf(min_dim, c_norm);
            }
        } else {
            // General case
            auto bc = cross(b, c);
            auto ac = cross(a, c);
            auto ab = cross(a, b);

            double bc_norm = norm(bc);
            double ac_norm = norm(ac);
            double ab_norm = norm(ab);

            double V = fabs(dot(a, bc));

            double d_a = V / bc_norm;
            double d_b = V / ac_norm;
            double d_c = V / ab_norm;

            if (periodic[0]) {
                min_dim = d_a;
            }

            if (periodic[1]) {
                min_dim = fminf(min_dim, d_b);
            }

            if (periodic[2]) {
                min_dim = fminf(min_dim, d_c);
            }
        }

        if (cutoff * 2.0 > min_dim) {
            status[0] = CUTOFF_TOO_LARGE;
            return;
        }

        if (!periodic[0] && (fabs(box[1].x) > 1e-6 || fabs(box[2].x) > 1e-6)) {
            status[0] = NOT_ALIGNED_LATTICE_A;
            return;
        }

        if (!periodic[1] && (fabs(box[0].y) > 1e-6 || fabs(box[2].y) > 1e-6)) {
            status[0] = NOT_ALIGNED_LATTICE_B;
            return;
        }

        if (!periodic[2] && (fabs(box[0].z) > 1e-6 || fabs(box[1].z) > 1e-6)) {
            status[0] = NOT_ALIGNED_LATTICE_C;
            return;
        }

        // everything is fine!
        status[0] = 0;
    }
}

void vesin::cuda::compute_mic_neighbourlist(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    int32_t* d_box_check,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    assert(n_points == 0 || points != nullptr);
    assert(box != nullptr);
    assert(periodic != nullptr);

    auto* d_positions = reinterpret_cast<const double3*>(points);
    auto* d_box = reinterpret_cast<const double3*>(box);
    auto* d_periodic = periodic;

    auto* d_pair_indices = reinterpret_cast<ulong2*>(neighbors.pairs);
    auto* d_shifts = reinterpret_cast<int3*>(neighbors.shifts);
    auto* d_distances = reinterpret_cast<double*>(neighbors.distances);
    auto* d_vectors = reinterpret_cast<double3*>(neighbors.vectors);
    size_t* d_pair_counter = extras->length_ptr;

    dim3 blockDim(WARP_SIZE * NWARPS);

    mic_box_check<<<1, 32>>>(d_box, periodic, options.cutoff, d_box_check);
    int32_t h_box_check = 0;
    cudaMemcpy(&h_box_check, d_box_check, sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (h_box_check == CUTOFF_TOO_LARGE) {
        throw std::runtime_error(
            "cutoff it too large for the current box, the CUDA implementation "
            "of vesin uses minimum image convention. Each box dimension "
            "must be at least twice the cutoff."
        );
    } else if (h_box_check == NOT_ALIGNED_LATTICE_A) {
        throw std::runtime_error(
            "periodicity is disabled along the A lattice vector, but the "
            "box is not defined in the yz plane"
        );
    } else if (h_box_check == NOT_ALIGNED_LATTICE_B) {
        throw std::runtime_error(
            "periodicity is disabled along the B lattice vector, but the "
            "box is not defined in the xz plane"
        );
    } else if (h_box_check == NOT_ALIGNED_LATTICE_C) {
        throw std::runtime_error(
            "periodicity is disabled along the CX lattice vector, but the "
            "box is not defined in the xy plane"
        );
    } else if (h_box_check != 0) {
        throw std::runtime_error("unknown error in box check");
    }

    if (options.full) {
        dim3 gridDim(max((int32_t)(n_points + NWARPS - 1) / NWARPS, 1));

        compute_mic_neighbours_full_impl<<<gridDim, blockDim>>>(
            d_positions,
            d_box,
            d_periodic,
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
            d_box,
            d_periodic,
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
