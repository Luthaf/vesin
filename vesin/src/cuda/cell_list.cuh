#ifndef VESIN_CUDA_CELL_LIST_CUH
#define VESIN_CUDA_CELL_LIST_CUH

#include <cstddef>

__global__ void compute_bounding_box(
    const double (*points)[3],
    size_t n_points,
    double* face_distances,
    double* bounding_min
);

// Compute inv_box, n_cells, n_search from box matrix and cutoff (single thread)
__global__ void compute_cell_grid_params(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    size_t max_cells,
    double inv_box[3][3],
    int* n_cells,
    int* n_search,
    int* n_cells_total,
    double* face_distances
);

// Map particles to cells via fractional coords, record periodic wrap shifts
__global__ void assign_cell_indices(
    const double (*points)[3],
    size_t n_points,
    const double inv_box[3][3],
    const bool periodic[3],
    const int* n_cells,
    const double* face_distances,
    const double* bounding_min,
    int* cell_indices,
    int* particle_shifts
);

// Count particles per cell (histogram)
__global__ void count_particles_per_cell(
    const int* cell_indices,
    size_t n_points,
    int* cell_counts
);

// Exclusive prefix sum of cell counts -> cell_starts (single block, uses shared mem)
__global__ void prefix_sum_cells(
    const int* cell_counts,
    int* cell_starts,
    const int* n_cells_total_ptr
);

// Reorder particles by cell for coalesced access in neighbor search
__global__ void scatter_particles(
    const double (*points)[3],
    size_t n_points,
    const int* cell_indices,
    const int* particle_shifts,
    int* cell_offsets,
    double* sorted_points,
    int* sorted_indices,
    int* sorted_shifts,
    int* sorted_cell_indices
);

// Main neighbor search kernel: each particle searches neighboring cells,
// threads within a group split the work across neighbor cells.
// Uses output buffering to batch writes and reduce atomic contention.
__global__ void find_neighbors_cell_list(
    const double* sorted_points,
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    const int* n_cells,
    const int* n_search,
    const int* sorted_indices,
    const int* sorted_shifts,
    const int* sorted_cell_indices,
    const int* cell_starts,
    const int* cell_counts,
    double cutoff,
    bool full_list,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t* length,
    size_t (*pair_indices)[2],
    int (*shifts_out)[3],
    double* distances,
    double (*vectors)[3],
    size_t max_pairs,
    int* overflow_flag
);

#endif // VESIN_CUDA_CELL_LIST_CUH
