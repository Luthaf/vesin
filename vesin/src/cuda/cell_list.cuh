#ifndef VESIN_CUDA_CELL_LIST_CUH
#define VESIN_CUDA_CELL_LIST_CUH

#include <cstddef>

// First step of `compute_bounding_box`: per-block partial bounding boxes.
__global__ void compute_bounding_box(
    const double (*points)[3],
    size_t n_points,
    double* partials,
    double* face_distances,
    double* bounding_min
);

// Second step of `compute_bounding_box`: combine the partials bounding boxes
__global__ void reduce_bounding_box(
    const double* partials,
    size_t n_partials,
    double* face_distances,
    double* bounding_min
);

// Compute inv_box, n_cells, n_search from box matrix and cutoff (single thread)
__global__ void compute_cell_grid_params(
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    size_t n_points,
    size_t max_cells,
    double cell_visit_cost,
    double inv_box[3][3],
    int* n_cells,
    int* n_search,
    int* n_cells_total,
    double* face_distances,
    bool* cell_out_of_reach
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

// Reorder particles by cell for coalesced access in neighbor search.
// Also computes wrapped positions (position - shift @ box) per sorted atom.
__global__ void scatter_particles(
    const double (*points)[3],
    size_t n_points,
    const int* cell_indices,
    const int* particle_shifts,
    const double box[3][3],
    int* cell_offsets,
    double* sorted_points,
    double* wrapped_positions,
    int* sorted_indices,
    int* sorted_shifts,
    int* sorted_cell_indices
);

// Two pass cell list, this is the first pass. This kernel search all neighbors,
// and count the number of neighbor per atom.
__global__ void count_neighbors_cell_list(
    const double* wrapped_positions,
    size_t n_points,
    const double box[3][3],
    const bool* cell_out_of_reach,
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
    int* counts
);

// Two pass cell list, this is the second pass. This kernel actually fills the
// output arrays, using pre-computed offsets to write to.
__global__ void fill_neighbors_cell_list(
    const double* wrapped_positions,
    size_t n_points,
    const double box[3][3],
    const bool* cell_out_of_reach,
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
    const int* write_offsets,
    size_t (*pair_indices)[2],
    int (*shifts_out)[3],
    double* distances,
    double (*vectors)[3]
);

#endif // VESIN_CUDA_CELL_LIST_CUH
