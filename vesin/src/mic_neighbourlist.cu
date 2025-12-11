// Type definitions for NVRTC compilation (no system headers available)
// Note: size_t is already defined by NVRTC builtin headers as unsigned long
#if defined(__CUDACC_RTC__)
typedef int int32_t;
typedef unsigned int uint32_t;
#endif

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

// Vector math helpers using flat array indexing
__device__ inline double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline void cross3(const double* a, const double* b, double* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline double norm3(const double* a) {
    return sqrt(dot3(a, a));
}

__device__ inline void normalize3(double* a) {
    double n = norm3(a);
    a[0] /= n;
    a[1] /= n;
    a[2] /= n;
}

__device__ void invert_matrix(const double* matrix, double* inverse) {
    // matrix is row-major: matrix[row*3 + col]
    double a = matrix[0], b = matrix[1], c = matrix[2];
    double d = matrix[3], e = matrix[4], f = matrix[5];
    double g = matrix[6], h = matrix[7], i = matrix[8];

    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = 1.0 / det;

    inverse[0] = (e * i - f * h) * invdet;
    inverse[1] = (c * h - b * i) * invdet;
    inverse[2] = (b * f - c * e) * invdet;
    inverse[3] = (f * g - d * i) * invdet;
    inverse[4] = (a * i - c * g) * invdet;
    inverse[5] = (c * d - a * f) * invdet;
    inverse[6] = (d * h - e * g) * invdet;
    inverse[7] = (b * g - a * h) * invdet;
    inverse[8] = (a * e - b * d) * invdet;
}

// Helper to compute Cartesian vector from fractional coordinates
// Using row convention: cart = frac @ box (frac as row vector times box matrix)
// cart[j] = sum_i(frac[i] * box[i*3+j])
__device__ inline void frac_to_cart(
    const double* frac,
    const double* box,
    double* cart
) {
    cart[0] = frac[0] * box[0] + frac[1] * box[3] + frac[2] * box[6];
    cart[1] = frac[0] * box[1] + frac[1] * box[4] + frac[2] * box[7];
    cart[2] = frac[0] * box[2] + frac[1] * box[5] + frac[2] * box[8];
}

__device__ void apply_periodic_boundary(
    double* vector,
    int32_t* shift,
    const double* box,
    const double* inv_box,
    const bool* periodic,
    bool is_orthogonal
) {
    // Compute fractional coordinates using row convention: frac = vector @ inv_box
    // frac[i] = sum_j(vector[j] * inv_box[j*3+i])
    double fractional[3];
    fractional[0] = vector[0] * inv_box[0] + vector[1] * inv_box[3] + vector[2] * inv_box[6];
    fractional[1] = vector[0] * inv_box[1] + vector[1] * inv_box[4] + vector[2] * inv_box[7];
    fractional[2] = vector[0] * inv_box[2] + vector[1] * inv_box[5] + vector[2] * inv_box[8];

    // Compute the initial wrapping to bring fractional coords into [-0.5, 0.5]
    // The multiplication by `periodic` sets the wrap to zero for non-periodic directions
    int32_t wrap[3];
    wrap[0] = static_cast<int32_t>(periodic[0]) * static_cast<int32_t>(round(fractional[0]));
    wrap[1] = static_cast<int32_t>(periodic[1]) * static_cast<int32_t>(round(fractional[1]));
    wrap[2] = static_cast<int32_t>(periodic[2]) * static_cast<int32_t>(round(fractional[2]));

    if (!is_orthogonal) {
        // For non-orthogonal cells, simple rounding may not find the true minimum image.
        // Search all 27 neighboring images to find the one with minimum distance.
        double min_dist2 = 1e30;
        int32_t best_wrap[3] = {wrap[0], wrap[1], wrap[2]};

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int32_t test_wrap[3];
                    test_wrap[0] = (wrap[0] + dx) * static_cast<int32_t>(periodic[0]);
                    test_wrap[1] = (wrap[1] + dy) * static_cast<int32_t>(periodic[1]);
                    test_wrap[2] = (wrap[2] + dz) * static_cast<int32_t>(periodic[2]);

                    double test_frac[3] = {
                        fractional[0] - test_wrap[0],
                        fractional[1] - test_wrap[1],
                        fractional[2] - test_wrap[2]
                    };

                    double test_vec[3];
                    frac_to_cart(test_frac, box, test_vec);
                    double dist2 = dot3(test_vec, test_vec);

                    if (dist2 < min_dist2) {
                        min_dist2 = dist2;
                        best_wrap[0] = test_wrap[0];
                        best_wrap[1] = test_wrap[1];
                        best_wrap[2] = test_wrap[2];
                    }
                }
            }
        }
        wrap[0] = best_wrap[0];
        wrap[1] = best_wrap[1];
        wrap[2] = best_wrap[2];
    }

    // The stored shift follows the convention: vector = rj - ri + shift @ box
    // Since we compute wrapped = vector - wrap @ box, the shift is -wrap
    shift[0] = -wrap[0];
    shift[1] = -wrap[1];
    shift[2] = -wrap[2];

    fractional[0] -= wrap[0];
    fractional[1] -= wrap[1];
    fractional[2] -= wrap[2];

    frac_to_cart(fractional, box, vector);
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
    __shared__ double shared_box[9];
    __shared__ double shared_inv_box[9];
    __shared__ bool shared_is_orthogonal;

    const int32_t warp_id = threadIdx.x / WARP_SIZE;
    const int32_t thread_id = threadIdx.x % WARP_SIZE;

    const size_t point_i = blockIdx.x * NWARPS + warp_id;
    const double cutoff2 = cutoff * cutoff;

    // Load current box to shared memory
    if (threadIdx.x < 9) {
        shared_box[threadIdx.x] = box[threadIdx.x];
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
            for (int i = 0; i < 9; i++) shared_box[i] = 0.0;
            shared_box[0] = 1.0; // row 0: (1,0,0)
            shared_box[4] = 1.0; // row 1: (0,1,0)
            shared_box[8] = 1.0; // row 2: (0,0,1)
        } else if (n_periodic == 1) {
            // 1D periodic: build an orthonormal pair spanning the plane orthogonal to the periodic vector
            double a[3] = {shared_box[periodic_idx_1*3], shared_box[periodic_idx_1*3+1], shared_box[periodic_idx_1*3+2]};

            double b[3] = {0, 1, 0};
            double a_normalized[3] = {a[0], a[1], a[2]};
            normalize3(a_normalized);
            if (fabs(dot3(a_normalized, b)) > 0.9) {
                b[0] = 0; b[1] = 0; b[2] = 1;
            }
            double c[3];
            cross3(a, b, c);
            normalize3(c);
            cross3(c, a, b);
            normalize3(b);

            int idx1 = (periodic_idx_1 + 1) % 3;
            int idx2 = (periodic_idx_1 + 2) % 3;
            shared_box[idx1*3] = b[0]; shared_box[idx1*3+1] = b[1]; shared_box[idx1*3+2] = b[2];
            shared_box[idx2*3] = c[0]; shared_box[idx2*3+1] = c[1]; shared_box[idx2*3+2] = c[2];
        } else if (n_periodic == 2) {
            // 2D periodic: set the sole non-periodic direction to the plane normal
            double a[3] = {shared_box[periodic_idx_1*3], shared_box[periodic_idx_1*3+1], shared_box[periodic_idx_1*3+2]};
            double b[3] = {shared_box[periodic_idx_2*3], shared_box[periodic_idx_2*3+1], shared_box[periodic_idx_2*3+2]};
            double c[3];
            cross3(a, b, c);
            normalize3(c);

            int non_periodic_idx = 3 - periodic_idx_1 - periodic_idx_2;
            shared_box[non_periodic_idx*3] = c[0];
            shared_box[non_periodic_idx*3+1] = c[1];
            shared_box[non_periodic_idx*3+2] = c[2];
        }
        // n_periodic == 3: fully periodic, keep shared_box as-is

        invert_matrix(shared_box, shared_inv_box);

        // Check orthogonality: all off-diagonal dot products should be ~0
        double tol = 1e-10;
        double row0[3] = {shared_box[0], shared_box[1], shared_box[2]};
        double row1[3] = {shared_box[3], shared_box[4], shared_box[5]};
        double row2[3] = {shared_box[6], shared_box[7], shared_box[8]};
        double ab = fabs(dot3(row0, row1));
        double ac = fabs(dot3(row0, row2));
        double bc = fabs(dot3(row1, row2));
        shared_is_orthogonal = (ab < tol) && (ac < tol) && (bc < tol);
    }

    // Ensure inv_box and is_orthogonal are ready
    __syncthreads();

    if (point_i >= n_points) {
        return;
    }

    bool is_orthogonal = shared_is_orthogonal;
    double ri[3] = {positions[point_i*3], positions[point_i*3+1], positions[point_i*3+2]};

    for (size_t j = thread_id; j < n_points; j += WARP_SIZE) {
        double rj[3] = {positions[j*3], positions[j*3+1], positions[j*3+2]};

        double vector[3] = {rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]};
        int32_t shift[3] = {0, 0, 0};
        apply_periodic_boundary(vector, shift, shared_box, shared_inv_box, periodic, is_orthogonal);

        double distance2 = dot3(vector, vector);
        auto is_valid = (distance2 < cutoff2 && distance2 > 0.0);

        if (is_valid) {
            size_t current_pair = atomicAdd(&length[0], 1);
            pair_indices[current_pair * 2] = point_i;
            pair_indices[current_pair * 2 + 1] = j;

            if (return_shifts) {
                shifts[current_pair * 3] = shift[0];
                shifts[current_pair * 3 + 1] = shift[1];
                shifts[current_pair * 3 + 2] = shift[2];
            }
            if (return_vectors) {
                vectors[current_pair * 3] = vector[0];
                vectors[current_pair * 3 + 1] = vector[1];
                vectors[current_pair * 3 + 2] = vector[2];
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

    __shared__ double shared_box[9];
    __shared__ double shared_inv_box[9];
    __shared__ bool shared_is_orthogonal;

    // Load current box to shared memory
    if (threadIdx.x < 9) {
        shared_box[threadIdx.x] = box[threadIdx.x];
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
            for (int i = 0; i < 9; i++) shared_box[i] = 0.0;
            shared_box[0] = 1.0;
            shared_box[4] = 1.0;
            shared_box[8] = 1.0;
        } else if (n_periodic == 1) {
            double a[3] = {shared_box[periodic_idx_1*3], shared_box[periodic_idx_1*3+1], shared_box[periodic_idx_1*3+2]};

            double b[3] = {0, 1, 0};
            double a_normalized[3] = {a[0], a[1], a[2]};
            normalize3(a_normalized);
            if (fabs(dot3(a_normalized, b)) > 0.9) {
                b[0] = 0; b[1] = 0; b[2] = 1;
            }
            double c[3];
            cross3(a, b, c);
            normalize3(c);
            cross3(c, a, b);
            normalize3(b);

            int idx1 = (periodic_idx_1 + 1) % 3;
            int idx2 = (periodic_idx_1 + 2) % 3;
            shared_box[idx1*3] = b[0]; shared_box[idx1*3+1] = b[1]; shared_box[idx1*3+2] = b[2];
            shared_box[idx2*3] = c[0]; shared_box[idx2*3+1] = c[1]; shared_box[idx2*3+2] = c[2];
        } else if (n_periodic == 2) {
            double a[3] = {shared_box[periodic_idx_1*3], shared_box[periodic_idx_1*3+1], shared_box[periodic_idx_1*3+2]};
            double b[3] = {shared_box[periodic_idx_2*3], shared_box[periodic_idx_2*3+1], shared_box[periodic_idx_2*3+2]};
            double c[3];
            cross3(a, b, c);
            normalize3(c);

            int non_periodic_idx = 3 - periodic_idx_1 - periodic_idx_2;
            shared_box[non_periodic_idx*3] = c[0];
            shared_box[non_periodic_idx*3+1] = c[1];
            shared_box[non_periodic_idx*3+2] = c[2];
        }

        invert_matrix(shared_box, shared_inv_box);

        double tol = 1e-10;
        double row0[3] = {shared_box[0], shared_box[1], shared_box[2]};
        double row1[3] = {shared_box[3], shared_box[4], shared_box[5]};
        double row2[3] = {shared_box[6], shared_box[7], shared_box[8]};
        double ab = fabs(dot3(row0, row1));
        double ac = fabs(dot3(row0, row2));
        double bc = fabs(dot3(row1, row2));
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

    double ri[3] = {positions[point_i*3], positions[point_i*3+1], positions[point_i*3+2]};
    double rj[3] = {positions[point_j*3], positions[point_j*3+1], positions[point_j*3+2]};

    double vector[3] = {rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]};
    int32_t shift[3] = {0, 0, 0};
    apply_periodic_boundary(vector, shift, shared_box, shared_inv_box, periodic, is_orthogonal);

    double distance2 = dot3(vector, vector);
    bool is_valid = (distance2 < cutoff2 && distance2 > 0.0);

    if (is_valid) {
        size_t pair_index = atomicAdd(&length[0], 1);
        pair_indices[pair_index * 2] = point_i;
        pair_indices[pair_index * 2 + 1] = point_j;

        if (return_shifts) {
            shifts[pair_index * 3] = shift[0];
            shifts[pair_index * 3 + 1] = shift[1];
            shifts[pair_index * 3 + 2] = shift[2];
        }
        if (return_vectors) {
            vectors[pair_index * 3] = vector[0];
            vectors[pair_index * 3 + 1] = vector[1];
            vectors[pair_index * 3 + 2] = vector[2];
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
    __shared__ double shared_box[9];

    if (threadIdx.x < 9) {
        shared_box[threadIdx.x] = box[threadIdx.x];
    }

    __syncthreads();

    double a[3] = {shared_box[0], shared_box[1], shared_box[2]};
    double b[3] = {shared_box[3], shared_box[4], shared_box[5]};
    double c[3] = {shared_box[6], shared_box[7], shared_box[8]};

    if (threadIdx.x == 0) {
        double a_norm = norm3(a);
        double b_norm = norm3(b);
        double c_norm = norm3(c);

        double ab_dot = dot3(a, b);
        double ac_dot = dot3(a, c);
        double bc_dot = dot3(b, c);

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
            double bc_cross[3], ac_cross[3], ab_cross[3];
            cross3(b, c, bc_cross);
            cross3(a, c, ac_cross);
            cross3(a, b, ab_cross);

            double bc_norm = norm3(bc_cross);
            double ac_norm = norm3(ac_cross);
            double ab_norm = norm3(ab_cross);

            double V = fabs(dot3(a, bc_cross));

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

        if (cutoff * 2.0 > min_dim) {
            status[0] = CUTOFF_TOO_LARGE;
            return;
        }

        // everything is fine!
        status[0] = 0;
    }
}
