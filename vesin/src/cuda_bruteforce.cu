#define NWARPS 4
#define WARP_SIZE 32

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    return static_cast<size_t>(atomicAdd(
        reinterpret_cast<unsigned long long*>(address),
        static_cast<unsigned long long>(val)
    ));
}

// Vector math helpers for double3
__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 cross(const double3& a, const double3& b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline double norm(const double3& a) {
    return sqrt(dot(a, a));
}

__device__ inline double3 normalize(const double3& a) {
    double n = norm(a);
    return make_double3(a.x / n, a.y / n, a.z / n);
}

__device__ void invert_matrix(const double3 box[3], double3 inverse[3]) {
    double a = box[0].x, b = box[0].y, c = box[0].z;
    double d = box[1].x, e = box[1].y, f = box[1].z;
    double g = box[2].x, h = box[2].y, i = box[2].z;

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

// Helper to compute Cartesian vector from fractional coordinates
// Using row convention: cart = frac @ box (frac as row vector times box matrix)
// cart[j] = sum_i(frac[i] * box[i].j)
__device__ inline double3 frac_to_cart(const double3& frac, const double3 box[3]) {
    return make_double3(
        frac.x * box[0].x + frac.y * box[1].x + frac.z * box[2].x,
        frac.x * box[0].y + frac.y * box[1].y + frac.z * box[2].y,
        frac.x * box[0].z + frac.y * box[1].z + frac.z * box[2].z
    );
}

__device__ void apply_periodic_boundary(
    double3& vector,
    int3& shift,
    const double3 box[3],
    const double3 inv_box[3],
    const bool* periodic,
    bool is_orthogonal
) {
    // Compute fractional coordinates using row convention: frac = vector @ inv_box
    // frac[i] = sum_j(vector[j] * inv_box[j].i)
    double3 fractional = make_double3(
        vector.x * inv_box[0].x + vector.y * inv_box[1].x + vector.z * inv_box[2].x,
        vector.x * inv_box[0].y + vector.y * inv_box[1].y + vector.z * inv_box[2].y,
        vector.x * inv_box[0].z + vector.y * inv_box[1].z + vector.z * inv_box[2].z
    );

    // Compute the initial wrapping to bring fractional coords into [-0.5, 0.5]
    // The multiplication by `periodic` sets the wrap to zero for non-periodic directions
    int3 wrap = make_int3(
        static_cast<int>(periodic[0]) * static_cast<int>(round(fractional.x)),
        static_cast<int>(periodic[1]) * static_cast<int>(round(fractional.y)),
        static_cast<int>(periodic[2]) * static_cast<int>(round(fractional.z))
    );

    if (!is_orthogonal) {
        // For non-orthogonal cells, simple rounding may not find the true minimum image.
        // Search all 27 neighboring images to find the one with minimum distance.
        double min_dist2 = 1e30;
        int3 best_wrap = wrap;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int3 test_wrap = make_int3(
                        (wrap.x + dx) * static_cast<int>(periodic[0]),
                        (wrap.y + dy) * static_cast<int>(periodic[1]),
                        (wrap.z + dz) * static_cast<int>(periodic[2])
                    );

                    double3 test_frac = make_double3(
                        fractional.x - test_wrap.x,
                        fractional.y - test_wrap.y,
                        fractional.z - test_wrap.z
                    );

                    double3 test_vec = frac_to_cart(test_frac, box);
                    double dist2 = dot(test_vec, test_vec);

                    if (dist2 < min_dist2) {
                        min_dist2 = dist2;
                        best_wrap = test_wrap;
                    }
                }
            }
        }
        wrap = best_wrap;
    }

    // The stored shift follows the convention: vector = rj - ri + shift @ box
    // Since we compute wrapped = vector - wrap @ box, the shift is -wrap
    shift = make_int3(-wrap.x, -wrap.y, -wrap.z);

    fractional = make_double3(
        fractional.x - wrap.x,
        fractional.y - wrap.y,
        fractional.z - wrap.z
    );

    vector = frac_to_cart(fractional, box);
}

__global__ void compute_mic_neighbours_full_impl(
    const double* positions,
    const double* box,
    const bool* periodic,
    size_t n_points,
    double cutoff,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];
    __shared__ bool shared_is_orthogonal;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int thread_id = threadIdx.x % WARP_SIZE;

    const size_t point_i = blockIdx.x * NWARPS + warp_id;
    const double cutoff2 = cutoff * cutoff;

    // Load current box to shared memory
    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = make_double3(
            box[threadIdx.x * 3],
            box[threadIdx.x * 3 + 1],
            box[threadIdx.x * 3 + 2]
        );
    }
    __syncthreads();

    // Overwrite non-periodic directions with unit vectors orthogonal to the periodic subspace
    if (threadIdx.x == 0) {
        // Collect periodic / non-periodic indices
        int n_periodic = 0;
        int periodic_idx_1 = -1;
        int periodic_idx_2 = -1;
        for (int i = 0; i < 3; ++i) {
            if (periodic[i]) {
                n_periodic += 1;
                if (periodic_idx_1 == -1) {
                    periodic_idx_1 = i;
                } else if (periodic_idx_2 == -1) {
                    periodic_idx_2 = i;
                }
            }
        }

        if (n_periodic == 0) {
            // Fully non-periodic: any orthonormal basis is fine
            shared_box[0] = make_double3(1.0, 0.0, 0.0);
            shared_box[1] = make_double3(0.0, 1.0, 0.0);
            shared_box[2] = make_double3(0.0, 0.0, 1.0);
        } else if (n_periodic == 1) {
            // 1D periodic: build an orthonormal pair spanning the plane orthogonal to the periodic vector
            double3 a = shared_box[periodic_idx_1];
            double3 b = make_double3(0, 1, 0);
            if (fabs(dot(normalize(a), b)) > 0.9) {
                b = make_double3(0, 0, 1);
            }
            double3 c = normalize(cross(a, b));
            b = normalize(cross(c, a));

            shared_box[(periodic_idx_1 + 1) % 3] = b;
            shared_box[(periodic_idx_1 + 2) % 3] = c;
        } else if (n_periodic == 2) {
            // 2D periodic: set the sole non-periodic direction to the plane normal
            double3 a = shared_box[periodic_idx_1];
            double3 b = shared_box[periodic_idx_2];
            double3 c = normalize(cross(a, b));

            int non_periodic_idx = 3 - periodic_idx_1 - periodic_idx_2;
            shared_box[non_periodic_idx] = c;
        }
        // n_periodic == 3: fully periodic, keep shared_box as-is

        invert_matrix(shared_box, shared_inv_box);

        // Check orthogonality: all off-diagonal dot products should be ~0
        double tol = 1e-10;
        double ab = fabs(dot(shared_box[0], shared_box[1]));
        double ac = fabs(dot(shared_box[0], shared_box[2]));
        double bc = fabs(dot(shared_box[1], shared_box[2]));
        shared_is_orthogonal = (ab < tol) && (ac < tol) && (bc < tol);
    }

    // Ensure inv_box and is_orthogonal are ready
    __syncthreads();

    if (point_i >= n_points) {
        return;
    }

    bool is_orthogonal = shared_is_orthogonal;
    double3 ri = make_double3(
        positions[point_i * 3],
        positions[point_i * 3 + 1],
        positions[point_i * 3 + 2]
    );

    for (size_t j = thread_id; j < n_points; j += WARP_SIZE) {
        double3 rj = make_double3(
            positions[j * 3],
            positions[j * 3 + 1],
            positions[j * 3 + 2]
        );

        double3 vector = rj - ri;
        int3 shift = make_int3(0, 0, 0);
        apply_periodic_boundary(vector, shift, shared_box, shared_inv_box, periodic, is_orthogonal);

        double distance2 = dot(vector, vector);
        bool is_valid = (distance2 < cutoff2 && distance2 > 0.0);

        if (is_valid) {
            size_t current_pair = atomicAdd_size_t(&length[0], 1);
            pair_indices[current_pair * 2] = point_i;
            pair_indices[current_pair * 2 + 1] = j;

            if (return_shifts) {
                shifts[current_pair * 3] = shift.x;
                shifts[current_pair * 3 + 1] = shift.y;
                shifts[current_pair * 3 + 2] = shift.z;
            }
            if (return_vectors) {
                vectors[current_pair * 3] = vector.x;
                vectors[current_pair * 3 + 1] = vector.y;
                vectors[current_pair * 3 + 2] = vector.z;
            }
            if (return_distances) {
                distances[current_pair] = sqrt(distance2);
            }
        }
    }
}

__global__ void compute_mic_neighbours_half_impl(
    const double* positions,
    const double* box,
    const bool* periodic,
    size_t n_points,
    double cutoff,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_all_pairs = n_points * (n_points - 1) / 2;
    const double cutoff2 = cutoff * cutoff;

    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];
    __shared__ bool shared_is_orthogonal;

    // Load current box to shared memory
    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = make_double3(
            box[threadIdx.x * 3],
            box[threadIdx.x * 3 + 1],
            box[threadIdx.x * 3 + 2]
        );
    }
    __syncthreads();

    // Overwrite non-periodic directions with unit vectors orthogonal to the periodic subspace
    if (threadIdx.x == 0) {
        int n_periodic = 0;
        int periodic_idx_1 = -1;
        int periodic_idx_2 = -1;
        for (int i = 0; i < 3; ++i) {
            if (periodic[i]) {
                n_periodic += 1;
                if (periodic_idx_1 == -1) {
                    periodic_idx_1 = i;
                } else if (periodic_idx_2 == -1) {
                    periodic_idx_2 = i;
                }
            }
        }

        if (n_periodic == 0) {
            shared_box[0] = make_double3(1.0, 0.0, 0.0);
            shared_box[1] = make_double3(0.0, 1.0, 0.0);
            shared_box[2] = make_double3(0.0, 0.0, 1.0);
        } else if (n_periodic == 1) {
            double3 a = shared_box[periodic_idx_1];
            double3 b = make_double3(0, 1, 0);
            if (fabs(dot(normalize(a), b)) > 0.9) {
                b = make_double3(0, 0, 1);
            }
            double3 c = normalize(cross(a, b));
            b = normalize(cross(c, a));

            shared_box[(periodic_idx_1 + 1) % 3] = b;
            shared_box[(periodic_idx_1 + 2) % 3] = c;
        } else if (n_periodic == 2) {
            double3 a = shared_box[periodic_idx_1];
            double3 b = shared_box[periodic_idx_2];
            double3 c = normalize(cross(a, b));

            int non_periodic_idx = 3 - periodic_idx_1 - periodic_idx_2;
            shared_box[non_periodic_idx] = c;
        }

        invert_matrix(shared_box, shared_inv_box);

        double tol = 1e-10;
        double ab = fabs(dot(shared_box[0], shared_box[1]));
        double ac = fabs(dot(shared_box[0], shared_box[2]));
        double bc = fabs(dot(shared_box[1], shared_box[2]));
        shared_is_orthogonal = (ab < tol) && (ac < tol) && (bc < tol);
    }

    __syncthreads();

    if (index >= num_all_pairs) {
        return;
    }

    bool is_orthogonal = shared_is_orthogonal;

    size_t point_j = floor((sqrt(8.0 * index + 1.0) + 1.0) / 2.0);
    if (point_j * (point_j - 1) > 2 * index) {
        point_j--;
    }
    const size_t point_i = index - point_j * (point_j - 1) / 2;

    double3 ri = make_double3(
        positions[point_i * 3],
        positions[point_i * 3 + 1],
        positions[point_i * 3 + 2]
    );
    double3 rj = make_double3(
        positions[point_j * 3],
        positions[point_j * 3 + 1],
        positions[point_j * 3 + 2]
    );

    double3 vector = rj - ri;
    int3 shift = make_int3(0, 0, 0);
    apply_periodic_boundary(vector, shift, shared_box, shared_inv_box, periodic, is_orthogonal);

    double distance2 = dot(vector, vector);
    bool is_valid = (distance2 < cutoff2 && distance2 > 0.0);

    if (is_valid) {
        size_t pair_index = atomicAdd_size_t(&length[0], 1);
        pair_indices[pair_index * 2] = point_i;
        pair_indices[pair_index * 2 + 1] = point_j;

        if (return_shifts) {
            shifts[pair_index * 3] = shift.x;
            shifts[pair_index * 3 + 1] = shift.y;
            shifts[pair_index * 3 + 2] = shift.z;
        }
        if (return_vectors) {
            vectors[pair_index * 3] = vector.x;
            vectors[pair_index * 3 + 1] = vector.y;
            vectors[pair_index * 3 + 2] = vector.z;
        }
        if (return_distances) {
            distances[pair_index] = sqrt(distance2);
        }
    }
}

// ============================================================================
// Optimized brute force kernels with precomputed box parameters
// These avoid per-block initialization by having inv_box, is_orthogonal passed in
// ============================================================================

// Simple PBC for orthogonal boxes (most common case)
// For non-periodic directions, no wrapping is applied
__device__ inline void apply_pbc_orthogonal(
    double3& d,
    int3& shift,
    const double3& box_diag,
    const bool* periodic
) {
    shift = make_int3(0, 0, 0);
    if (periodic[0] && box_diag.x > 0) {
        int s = static_cast<int>(round(d.x / box_diag.x));
        d.x -= s * box_diag.x;
        shift.x = -s;
    }
    if (periodic[1] && box_diag.y > 0) {
        int s = static_cast<int>(round(d.y / box_diag.y));
        d.y -= s * box_diag.y;
        shift.y = -s;
    }
    if (periodic[2] && box_diag.z > 0) {
        int s = static_cast<int>(round(d.z / box_diag.z));
        d.z -= s * box_diag.z;
        shift.z = -s;
    }
}

// General PBC using precomputed inverse box
__device__ inline void apply_pbc_general(
    double3& vector,
    int3& shift,
    const double3 box[3],
    const double3 inv_box[3],
    const bool* periodic
) {
    double3 frac = make_double3(
        vector.x * inv_box[0].x + vector.y * inv_box[1].x + vector.z * inv_box[2].x,
        vector.x * inv_box[0].y + vector.y * inv_box[1].y + vector.z * inv_box[2].y,
        vector.x * inv_box[0].z + vector.y * inv_box[1].z + vector.z * inv_box[2].z
    );

    int3 wrap = make_int3(
        periodic[0] ? static_cast<int>(round(frac.x)) : 0,
        periodic[1] ? static_cast<int>(round(frac.y)) : 0,
        periodic[2] ? static_cast<int>(round(frac.z)) : 0
    );

    frac.x -= wrap.x;
    frac.y -= wrap.y;
    frac.z -= wrap.z;

    vector = frac_to_cart(frac, box);

    shift = make_int3(-wrap.x, -wrap.y, -wrap.z);
}

__global__ void brute_force_half_orthogonal(
    const double* __restrict__ positions,
    const double* __restrict__ box_diag,
    const bool* __restrict__ periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_all_pairs = n_points * (n_points - 1) / 2;

    if (index >= num_all_pairs) {
        return;
    }

    size_t j = static_cast<size_t>(floor((sqrt(8.0 * index + 1.0) + 1.0) / 2.0));
    if (j * (j - 1) > 2 * index) {
        j--;
    }
    const size_t i = index - j * (j - 1) / 2;

    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    double3 pi = pos3[i];
    double3 pj = pos3[j];
    double3 d = pj - pi;
    double3 L = make_double3(box_diag[0], box_diag[1], box_diag[2]);

    int3 s;
    apply_pbc_orthogonal(d, s, L, periodic);

    double dist2 = dot(d, d);

    if (dist2 < cutoff2 && dist2 > 0.0) {
        size_t idx = atomicAdd_size_t(length, 1UL);

        // Check if we are about to exceed max_pairs
        if (idx + 1 > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }
        pair_indices[idx * 2] = i;
        pair_indices[idx * 2 + 1] = j;
        if (return_shifts) {
            shifts[idx * 3] = s.x;
            shifts[idx * 3 + 1] = s.y;
            shifts[idx * 3 + 2] = s.z;
        }
        if (return_vectors) {
            vectors[idx * 3] = d.x;
            vectors[idx * 3 + 1] = d.y;
            vectors[idx * 3 + 2] = d.z;
        }
        if (return_distances) {
            distances[idx] = sqrt(dist2);
        }
    }
}

// Triangular indexing: one thread per unordered pair, outputs both (i,j) and (j,i)
__global__ void brute_force_full_orthogonal(
    const double* __restrict__ positions,
    const double* __restrict__ box_diag,
    const bool* __restrict__ periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_half_pairs = n_points * (n_points - 1) / 2;

    if (index >= num_half_pairs) {
        return;
    }

    size_t j = static_cast<size_t>(floor((sqrt(8.0 * index + 1.0) + 1.0) / 2.0));
    if (j * (j - 1) > 2 * index) {
        j--;
    }
    const size_t i = index - j * (j - 1) / 2;

    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    double3 pi = pos3[i];
    double3 pj = pos3[j];
    double3 d = pj - pi;
    double3 L = make_double3(box_diag[0], box_diag[1], box_diag[2]);

    int3 s;
    apply_pbc_orthogonal(d, s, L, periodic);

    double dist2 = dot(d, d);

    if (dist2 < cutoff2) {
        size_t idx = atomicAdd_size_t(length, 2UL);

        // Check if we are about to exceed max_pairs
        if (idx + 2 > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }

        pair_indices[idx * 2] = i;
        pair_indices[idx * 2 + 1] = j;
        pair_indices[(idx + 1) * 2] = j;
        pair_indices[(idx + 1) * 2 + 1] = i;

        if (return_shifts) {
            shifts[idx * 3] = s.x;
            shifts[idx * 3 + 1] = s.y;
            shifts[idx * 3 + 2] = s.z;
            shifts[(idx + 1) * 3] = -s.x;
            shifts[(idx + 1) * 3 + 1] = -s.y;
            shifts[(idx + 1) * 3 + 2] = -s.z;
        }
        if (return_vectors) {
            vectors[idx * 3] = d.x;
            vectors[idx * 3 + 1] = d.y;
            vectors[idx * 3 + 2] = d.z;
            vectors[(idx + 1) * 3] = -d.x;
            vectors[(idx + 1) * 3 + 1] = -d.y;
            vectors[(idx + 1) * 3 + 2] = -d.z;
        }
        if (return_distances) {
            double dist = sqrt(dist2);
            distances[idx] = dist;
            distances[idx + 1] = dist;
        }
    }
}

__global__ void brute_force_half_general(
    const double* __restrict__ positions,
    const double* __restrict__ box,
    const double* __restrict__ inv_box,
    const bool* __restrict__ periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_all_pairs = n_points * (n_points - 1) / 2;

    // Load box into double3 arrays
    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];

    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = make_double3(
            box[threadIdx.x * 3],
            box[threadIdx.x * 3 + 1],
            box[threadIdx.x * 3 + 2]
        );
        shared_inv_box[threadIdx.x] = make_double3(
            inv_box[threadIdx.x * 3],
            inv_box[threadIdx.x * 3 + 1],
            inv_box[threadIdx.x * 3 + 2]
        );
    }
    __syncthreads();

    if (index >= num_all_pairs) {
        return;
    }

    size_t j = static_cast<size_t>(floor((sqrt(8.0 * index + 1.0) + 1.0) / 2.0));
    if (j * (j - 1) > 2 * index) {
        j--;
    }
    const size_t i = index - j * (j - 1) / 2;

    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    double3 pi = pos3[i];
    double3 pj = pos3[j];
    double3 vector = pj - pi;
    int3 shift;

    apply_pbc_general(vector, shift, shared_box, shared_inv_box, periodic);

    double dist2 = dot(vector, vector);

    if (dist2 < cutoff2 && dist2 > 0.0) {
        size_t idx = atomicAdd_size_t(length, 1UL);

        // Check if we are about to exceed max_pairs
        if (idx + 1 > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }
        pair_indices[idx * 2] = i;
        pair_indices[idx * 2 + 1] = j;
        if (return_shifts) {
            shifts[idx * 3] = shift.x;
            shifts[idx * 3 + 1] = shift.y;
            shifts[idx * 3 + 2] = shift.z;
        }
        if (return_vectors) {
            vectors[idx * 3] = vector.x;
            vectors[idx * 3 + 1] = vector.y;
            vectors[idx * 3 + 2] = vector.z;
        }
        if (return_distances) {
            distances[idx] = sqrt(dist2);
        }
    }
}

// Optimized full-list kernel for GENERAL boxes
// NNPOps-style triangular indexing: one thread per unordered pair, outputs both (i,j) and (j,i)
// Uses double3 for vectorized position loads
__global__ void brute_force_full_general(
    const double* __restrict__ positions,
    const double* __restrict__ box,
    const double* __restrict__ inv_box,
    const bool* __restrict__ periodic,
    size_t n_points,
    double cutoff2,
    size_t* length,
    size_t* pair_indices,
    int* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_half_pairs = n_points * (n_points - 1) / 2;

    // Load box into double3 arrays
    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];

    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = make_double3(
            box[threadIdx.x * 3],
            box[threadIdx.x * 3 + 1],
            box[threadIdx.x * 3 + 2]
        );
        shared_inv_box[threadIdx.x] = make_double3(
            inv_box[threadIdx.x * 3],
            inv_box[threadIdx.x * 3 + 1],
            inv_box[threadIdx.x * 3 + 2]
        );
    }
    __syncthreads();

    if (index >= num_half_pairs) {
        return;
    }

    // NNPOps-style triangular indexing for half-list
    size_t j = static_cast<size_t>(floor((sqrt(8.0 * index + 1.0) + 1.0) / 2.0));
    if (j * (j - 1) > 2 * index) {
        j--;
    }
    const size_t i = index - j * (j - 1) / 2;

    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    double3 pi = pos3[i];
    double3 pj = pos3[j];
    double3 vector = pj - pi;
    int3 shift;

    apply_pbc_general(vector, shift, shared_box, shared_inv_box, periodic);

    double dist2 = dot(vector, vector);

    if (dist2 < cutoff2) {
        size_t idx = atomicAdd_size_t(length, 2UL);

        // Check if we are about to exceed max_pairs
        if (idx + 2 > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }

        pair_indices[idx * 2] = i;
        pair_indices[idx * 2 + 1] = j;
        pair_indices[(idx + 1) * 2] = j;
        pair_indices[(idx + 1) * 2 + 1] = i;

        if (return_shifts) {
            shifts[idx * 3] = shift.x;
            shifts[idx * 3 + 1] = shift.y;
            shifts[idx * 3 + 2] = shift.z;
            shifts[(idx + 1) * 3] = -shift.x;
            shifts[(idx + 1) * 3 + 1] = -shift.y;
            shifts[(idx + 1) * 3 + 2] = -shift.z;
        }
        if (return_vectors) {
            vectors[idx * 3] = vector.x;
            vectors[idx * 3 + 1] = vector.y;
            vectors[idx * 3 + 2] = vector.z;
            vectors[(idx + 1) * 3] = -vector.x;
            vectors[(idx + 1) * 3 + 1] = -vector.y;
            vectors[(idx + 1) * 3 + 2] = -vector.z;
        }
        if (return_distances) {
            double dist = sqrt(dist2);
            distances[idx] = dist;
            distances[idx + 1] = dist;
        }
    }
}

// Status flags for mic_box_check
// bit 0: error (cutoff too large)
// bit 1: is_orthogonal
#define BOX_STATUS_ERROR 1
#define BOX_STATUS_ORTHOGONAL 2

__global__ void mic_box_check(
    const double* box,
    const bool* periodic,
    const double cutoff,
    int* status,
    double* box_diag,   // Output: [Lx, Ly, Lz] for orthogonal boxes (can be nullptr)
    double* inv_box_out // Output: 9-element inverse box matrix (can be nullptr)
) {
    __shared__ double3 shared_box[3];

    if (threadIdx.x < 3) {
        shared_box[threadIdx.x] = make_double3(
            box[threadIdx.x * 3],
            box[threadIdx.x * 3 + 1],
            box[threadIdx.x * 3 + 2]
        );
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        double3 a = shared_box[0];
        double3 b = shared_box[1];
        double3 c = shared_box[2];

        double a_norm = norm(a);
        double b_norm = norm(b);
        double c_norm = norm(c);

        // Count periodic directions
        int n_periodic = 0;
        if (periodic[0]) {
            n_periodic++;
        }
        if (periodic[1]) {
            n_periodic++;
        }
        if (periodic[2]) {
            n_periodic++;
        }

        double ab_dot = dot(a, b);
        double ac_dot = dot(a, c);
        double bc_dot = dot(b, c);

        double tol = 1e-6;
        // Treat fully non-periodic systems as orthogonal (no PBC needed)
        // Also treat systems with zero-norm vectors as orthogonal (degenerate case)
        bool is_orthogonal = (n_periodic == 0) ||
                             (a_norm < tol || b_norm < tol || c_norm < tol) ||
                             ((fabs(ab_dot) < tol * a_norm * b_norm) &&
                              (fabs(ac_dot) < tol * a_norm * c_norm) &&
                              (fabs(bc_dot) < tol * b_norm * c_norm));

        if (box_diag != nullptr) {
            box_diag[0] = a_norm;
            box_diag[1] = b_norm;
            box_diag[2] = c_norm;
        }

        if (inv_box_out != nullptr && !is_orthogonal) {
            double3 inv_box[3];
            invert_matrix(shared_box, inv_box);
            inv_box_out[0] = inv_box[0].x;
            inv_box_out[1] = inv_box[0].y;
            inv_box_out[2] = inv_box[0].z;
            inv_box_out[3] = inv_box[1].x;
            inv_box_out[4] = inv_box[1].y;
            inv_box_out[5] = inv_box[1].z;
            inv_box_out[6] = inv_box[2].x;
            inv_box_out[7] = inv_box[2].y;
            inv_box_out[8] = inv_box[2].z;
        }

        double min_dim = 1e30;
        if (is_orthogonal) {
            if (periodic[0]) {
                min_dim = a_norm;
            }

            if (periodic[1]) {
                min_dim = fmin(min_dim, b_norm);
            }

            if (periodic[2]) {
                min_dim = fmin(min_dim, c_norm);
            }
        } else {
            // General case
            double3 bc_cross = cross(b, c);
            double3 ac_cross = cross(a, c);
            double3 ab_cross = cross(a, b);

            double bc_norm = norm(bc_cross);
            double ac_norm = norm(ac_cross);
            double ab_norm = norm(ab_cross);

            double V = fabs(dot(a, bc_cross));

            double d_a = V / bc_norm;
            double d_b = V / ac_norm;
            double d_c = V / ab_norm;

            if (periodic[0]) {
                min_dim = d_a;
            }

            if (periodic[1]) {
                min_dim = fmin(min_dim, d_b);
            }

            if (periodic[2]) {
                min_dim = fmin(min_dim, d_c);
            }
        }

        int result = 0;
        if (cutoff * 2.0 > min_dim) {
            result |= BOX_STATUS_ERROR;
        }
        if (is_orthogonal) {
            result |= BOX_STATUS_ORTHOGONAL;
        }
        status[0] = result;
    }
}
