#include "bruteforce.cuh"

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    return static_cast<size_t>(atomicAdd(
        reinterpret_cast<uint64*>(address),
        static_cast<uint64>(val)
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

// ============================================================================
// Brute force kernels
//
// One thread per pair of atoms, walking the periodic images of that pair.
// `analyze_box` below precomputes everything about the box these need: the
// inverse matrix used to wrap the pair vector, and how many images to walk.
// ============================================================================

// Walk every periodic image of every pair of atoms within `cutoff2`, calling
// `emit(i, j, shift, vector, dist2)` for each.
//
// `inv_box` is the *regularized* inverse from `analyze_box`: it only differs
// from the plain inverse along non-periodic directions, which never take a
// non-zero shift.
template <typename Emit>
__device__ inline void brute_force_images(
    const double (*__restrict__ points)[3],
    const double (*__restrict__ box)[3],
    const double (*__restrict__ inv_box)[3],
    const bool* __restrict__ periodic,
    const int32* __restrict__ n_search_in,
    double cutoff2,
    Emit&& emit
) {
    // One thread per pair `i <= j`, with `j` given by the grid row and `i` by
    // the thread within the row. The grid is the full `n_points x n_points`
    // square, so half of it is above the diagonal and leaves here. `j` is a grid
    // row so `j < n_points`, which means keeping `i <= j` also keeps `i` in
    // range without looking at `n_points` at all.
    //
    // The `i == j` threads are only useful when the box is smaller than twice
    // the cutoff, where an atom can be neighbor of its own periodic images.
    // They are one per row, so carrying them costs nothing worth branching over.
    size_t j = blockIdx.y;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i > j) {
        return;
    }

    const auto* pos3 = reinterpret_cast<const double3*>(points);
    double3 delta = pos3[j] - pos3[i];

    // Fractional coordinates `frac = delta @ inv_box`. `inv_box` is only needed
    // here: the loop below works in cartesian space.
    double3 frac = make_double3(
        delta.x * inv_box[0][0] + delta.y * inv_box[1][0] + delta.z * inv_box[2][0],
        delta.x * inv_box[0][1] + delta.y * inv_box[1][1] + delta.z * inv_box[2][1],
        delta.x * inv_box[0][2] + delta.y * inv_box[1][2] + delta.z * inv_box[2][2]
    );

    // Integer offset bringing `frac` into [-0.5, 0.5] along the periodic
    // directions. This is the minimum image convention when it applies, but the
    // loop below does not rely on it picking the closest image, only on the
    // resulting fractional coordinates being bounded by 0.5 -- see the bound on
    // `n_search` in `analyze_box`.
    int3 wrap = make_int3(
        periodic[0] ? static_cast<int32>(round(frac.x)) : 0,
        periodic[1] ? static_cast<int32>(round(frac.y)) : 0,
        periodic[2] ? static_cast<int32>(round(frac.z)) : 0
    );

    // Every image adds an integer combination of the box rows, so these stay
    // live for the whole loop. `box` is the box as the caller gave it, since
    // that is what the `shift` convention refers to.
    double3 box_rows[3] = {
        make_double3(box[0][0], box[0][1], box[0][2]),
        make_double3(box[1][0], box[1][1], box[1][2]),
        make_double3(box[2][0], box[2][1], box[2][2]),
    };
    int3 n_search = make_int3(n_search_in[0], n_search_in[1], n_search_in[2]);

    for (int32 sx = -n_search.x; sx <= n_search.x; sx++) {
        for (int32 sy = -n_search.y; sy <= n_search.y; sy++) {
            for (int32 sz = -n_search.z; sz <= n_search.z; sz++) {
                // shift convention: `vector = rj - ri + shift @ box`
                int3 shift = make_int3(sx - wrap.x, sy - wrap.y, sz - wrap.z);

                if (i == j) {
                    // `i <= j` can not pick a direction for a pair of an atom
                    // with its own periodic image, so keep only the image whose
                    // first non-zero shift component is positive. This also
                    // drops the zero shift, i.e. the atom paired with itself.
                    bool canonical = (shift.x > 0)
                        || (shift.x == 0 && shift.y > 0)
                        || (shift.x == 0 && shift.y == 0 && shift.z > 0);
                    if (!canonical) {
                        continue;
                    }
                }

                double3 vector = make_double3(
                    delta.x + shift.x * box_rows[0].x + shift.y * box_rows[1].x + shift.z * box_rows[2].x,
                    delta.y + shift.x * box_rows[0].y + shift.y * box_rows[1].y + shift.z * box_rows[2].y,
                    delta.z + shift.x * box_rows[0].z + shift.y * box_rows[1].z + shift.z * box_rows[2].z
                );

                double dist2 = dot(vector, vector);
                if (dist2 < cutoff2) {
                    emit(i, j, shift, vector, dist2);
                }
            }
        }
    }
}

__global__ void brute_force_half(
    const double (*__restrict__ points)[3],
    const double (*__restrict__ box)[3],
    const double (*__restrict__ inv_box)[3],
    const bool* __restrict__ periodic,
    const int32* __restrict__ n_search,
    double cutoff2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int32 (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int32* overflow_flag
) {
    brute_force_images(
        points,
        box,
        inv_box,
        periodic,
        n_search,
        cutoff2,
        [&](size_t i, size_t j, int3 shift, double3 vector, double dist2) {
            size_t idx = atomicAdd_size_t(length, 1UL);

            // Check if we are about to exceed max_pairs
            if (idx + 1 > max_pairs) {
                atomicExch(overflow_flag, 1);
                return;
            }

            pair_indices[idx][0] = i;
            pair_indices[idx][1] = j;
            if (return_shifts) {
                shifts[idx][0] = shift.x;
                shifts[idx][1] = shift.y;
                shifts[idx][2] = shift.z;
            }
            if (return_vectors) {
                vectors[idx][0] = vector.x;
                vectors[idx][1] = vector.y;
                vectors[idx][2] = vector.z;
            }
            if (return_distances) {
                distances[idx] = sqrt(dist2);
            }
        }
    );
}

// Write a single pair `(i, j)` at position `idx` in the outputs
__device__ inline void write_pair(
    size_t idx,
    size_t i,
    size_t j,
    int3 shift,
    double3 vector,
    double dist2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t (*pair_indices)[2],
    int32 (*shifts)[3],
    double* distances,
    double (*vectors)[3]
) {
    pair_indices[idx][0] = i;
    pair_indices[idx][1] = j;
    if (return_shifts) {
        shifts[idx][0] = shift.x;
        shifts[idx][1] = shift.y;
        shifts[idx][2] = shift.z;
    }
    if (return_vectors) {
        vectors[idx][0] = vector.x;
        vectors[idx][1] = vector.y;
        vectors[idx][2] = vector.z;
    }
    if (return_distances) {
        distances[idx] = sqrt(dist2);
    }
}

__global__ void brute_force_full(
    const double (*__restrict__ points)[3],
    const double (*__restrict__ box)[3],
    const double (*__restrict__ inv_box)[3],
    const bool* __restrict__ periodic,
    const int32* __restrict__ n_search,
    double cutoff2,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int32 (*shifts)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int32* overflow_flag,
    int32 padded_to,
    int32* pairs_per_point
) {
    // One thread per unordered pair, writing out both (i, j) and (j, i)
    brute_force_images(
        points,
        box,
        inv_box,
        periodic,
        n_search,
        cutoff2,
        [&](size_t i, size_t j, int3 shift, double3 vector, double dist2) {
            if (padded_to > 0) {
                // Each point owns `padded_to` slots in the output. We keep
                // counting past the end to detect overflow in `pad_neighbors`,
                // but only write inside the slots of the point.
                int32 index_i = atomicAdd(&pairs_per_point[i], 1);
                if (index_i < padded_to) {
                    size_t idx = i * padded_to + index_i;
                    write_pair(idx, i, j, shift, vector, dist2, return_shifts, return_distances, return_vectors, pair_indices, shifts, distances, vectors);
                }

                int32 index_j = atomicAdd(&pairs_per_point[j], 1);
                if (index_j < padded_to) {
                    size_t idx = j * padded_to + index_j;
                    auto minus_shift = make_int3(-shift.x, -shift.y, -shift.z);
                    auto minus_vector = make_double3(-vector.x, -vector.y, -vector.z);
                    write_pair(idx, j, i, minus_shift, minus_vector, dist2, return_shifts, return_distances, return_vectors, pair_indices, shifts, distances, vectors);
                }
                return;
            }

            size_t idx = atomicAdd_size_t(length, 2UL);

            // Check if we are about to exceed max_pairs
            if (idx + 2 > max_pairs) {
                atomicExch(overflow_flag, 1);
                return;
            }

            pair_indices[idx][0] = i;
            pair_indices[idx][1] = j;
            pair_indices[idx + 1][0] = j;
            pair_indices[idx + 1][1] = i;

            if (return_shifts) {
                shifts[idx][0] = shift.x;
                shifts[idx][1] = shift.y;
                shifts[idx][2] = shift.z;
                shifts[idx + 1][0] = -shift.x;
                shifts[idx + 1][1] = -shift.y;
                shifts[idx + 1][2] = -shift.z;
            }
            if (return_vectors) {
                vectors[idx][0] = vector.x;
                vectors[idx][1] = vector.y;
                vectors[idx][2] = vector.z;
                vectors[idx + 1][0] = -vector.x;
                vectors[idx + 1][1] = -vector.y;
                vectors[idx + 1][2] = -vector.z;
            }
            if (return_distances) {
                double dist = sqrt(dist2);
                distances[idx] = dist;
                distances[idx + 1] = dist;
            }
        }
    );
}

// Replace the box vectors along non-periodic directions with an orthonormal
// completion of the periodic ones, so that the matrix can be inverted even when
// the caller left them zero or parallel (a 2D system given a zero third vector,
// say). Only periodic directions ever take a non-zero shift, so this does not
// change any output.
//
// This mirrors `compute_cell_grid_params` in cell_list.cu; keep the two in
// agreement.
__device__ void regularize_box(double3 rows[3], const bool periodic[3]) {
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
        if (abs(dot(normalize(a), b)) > 0.9) {
            b = make_double3(0.0, 0.0, 1.0);
        }
        double3 c = normalize(cross(a, b));
        b = normalize(cross(c, a));

        rows[(periodic_idx_1 + 1) % 3] = b;
        rows[(periodic_idx_1 + 2) % 3] = c;
    } else if (n_periodic == 2) {
        double3 a = rows[periodic_idx_1];
        double3 b = rows[periodic_idx_2];
        double3 c = normalize(cross(a, b));
        rows[3 - periodic_idx_1 - periodic_idx_2] = c;
    }
}

// All the work is done by a single thread: there are only 18 numbers to look at,
// and everything downstream needs the answer before it can start.
__global__ void analyze_box(
    const double (*__restrict__ box)[3],
    const bool* __restrict__ periodic,
    double cutoff,
    bool* all_periodic,
    int32* n_search,
    double inv_box_out[3][3]
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    *all_periodic = periodic[0] && periodic[1] && periodic[2];

    double3 rows[3] = {
        make_double3(box[0][0], box[0][1], box[0][2]),
        make_double3(box[1][0], box[1][1], box[1][2]),
        make_double3(box[2][0], box[2][1], box[2][2]),
    };
    regularize_box(rows, periodic);
    invert_matrix(rows, reinterpret_cast<double3*>(inv_box_out));

    // Distances between opposite faces of the (regularized) box, i.e. the
    // spacing of the lattice planes normal to each reciprocal vector.
    double3 bc = cross(rows[1], rows[2]);
    double3 ca = cross(rows[2], rows[0]);
    double3 ab = cross(rows[0], rows[1]);

    double face_distances[3] = {
        abs(dot(rows[0], bc)) / norm(bc),
        abs(dot(rows[1], ca)) / norm(ca),
        abs(dot(rows[2], ab)) / norm(ab),
    };

    // How many periodic images to walk along each direction.
    //
    // `brute_force_images` wraps the pair vector so that its fractional
    // coordinates are in [-0.5, 0.5], which puts the image at integer offset
    // `s` at least `(|s| - 0.5) * face_distance` away along the normal to the
    // other two box vectors. That is inside the cutoff only if
    // `|s| < cutoff / face_distance + 0.5`, hence the bound below.
    //
    // This is exactly 0 when `cutoff <= face_distance / 2`, i.e. whenever the
    // minimum image convention applies: the single wrapped image is then the
    // only one that can be in range, and the loop runs once.
    for (int32 d = 0; d < 3; d++) {
        if (!periodic[d]) {
            n_search[d] = 0;
            continue;
        }

        // `min`/`max` are `fmin`/`fmax`, so this also pins down a NaN coming
        // out of a box with a zero-length or parallel pair of periodic vectors.
        // Hitting `MAX_N_SEARCH` is not a truncation: the caller is told the
        // box needs too many images before the kernels ever run, see the
        // comment on the constant.
        double shifts = ceil(cutoff / face_distances[d] + 0.5) - 1.0;
        n_search[d] = static_cast<int32>(min(max(shifts, 0.0), double(MAX_N_SEARCH)));
    }
}
