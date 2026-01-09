// Type definitions for NVRTC compilation (no system headers available)
// Note: size_t, double3, int3 are already defined by NVRTC builtin headers
#if defined(__CUDACC_RTC__)
typedef int int32_t;
typedef unsigned int uint32_t;
#endif

#define NWARPS 4
#define WARP_SIZE 32

__device__ inline size_t atomicAdd(size_t* address, size_t val) {
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
    int32_t* shifts,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    __shared__ double3 shared_box[3];
    __shared__ double3 shared_inv_box[3];
    __shared__ bool shared_is_orthogonal;

    const int32_t warp_id = threadIdx.x / WARP_SIZE;
    const int32_t thread_id = threadIdx.x % WARP_SIZE;

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
            size_t current_pair = atomicAdd(&length[0], 1);
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
    int32_t* shifts,
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
        size_t pair_index = atomicAdd(&length[0], 1);
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

// possible error for mic_box_check
#define CUTOFF_TOO_LARGE 1

__global__ void mic_box_check(
    const double* box,
    const bool* periodic,
    const double cutoff,
    int32_t* status
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

        double ab_dot = dot(a, b);
        double ac_dot = dot(a, c);
        double bc_dot = dot(b, c);

        double tol = 1e-6;
        bool is_orthogonal = (fabs(ab_dot) < tol * a_norm * b_norm) &&
                             (fabs(ac_dot) < tol * a_norm * c_norm) &&
                             (fabs(bc_dot) < tol * b_norm * c_norm);

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

            double bc_cross_norm = norm(bc_cross);
            double ac_cross_norm = norm(ac_cross);
            double ab_cross_norm = norm(ab_cross);

            double V = fabs(dot(a, bc_cross));

            double d_a = V / bc_cross_norm;
            double d_b = V / ac_cross_norm;
            double d_c = V / ab_cross_norm;

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

        if (cutoff * 2.0 > min_dim) {
            status[0] = CUTOFF_TOO_LARGE;
            return;
        }

        // everything is fine!
        status[0] = 0;
    }
}
