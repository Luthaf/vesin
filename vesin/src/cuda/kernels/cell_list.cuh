#ifndef VESIN_CUDA_CELL_LIST_CUH
#define VESIN_CUDA_CELL_LIST_CUH

#include <cstddef>

#include "integer.cuh"

// Block size for the two bounding box kernels. Must be a power of two, for
// their tree reduction. This is also used to size their shared memory.
#define VESIN_BOUNDING_BOX_BLOCK 256

// How many entries are in the cell reachability mask
// (`CellListBuffersCUDA::d_cell_out_of_reach`).
//
// A box much thinner than the cutoff can still exceed it, in which case no
// pruning will be done.
#define VESIN_MAX_OUT_OF_REACH_MASK 32768

// Block size of `compute_cell_grid_params`. One thread picks the grid, then the
// whole block fills the offset mask, so this only has to be large enough to
// cover `VESIN_MAX_OUT_OF_REACH_MASK` in a few iterations each.
#define VESIN_GRID_PARAMS_BLOCK 256

// Block size for the two neighbor search kernels, and the number of threads
// they put on each atom (and the resulting number of atoms a block covers).
// One atom's search is a walk over (2 * n_search + 1)^3 cells, and splitting it
// across threads both multiplies the number of blocks and shortens the
// dependent chain of gather loads within one atom.
//
// The block size must be a multiple of 32 and the thread count must divide 32,
// so that a group never straddles two warps. These are also used to size the
// shared memory of the kernels.
#define VESIN_CELL_LIST_BLOCK 256
#define VESIN_THREADS_PER_PARTICLE 8
#define VESIN_PARTICLES_PER_BLOCK (VESIN_CELL_LIST_BLOCK / VESIN_THREADS_PER_PARTICLE)

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
    int32* n_cells,
    int32* n_search,
    int32* n_cells_total,
    double* face_distances,
    bool* cell_out_of_reach
);

// Map particles to cells via fractional coords, record periodic wrap shifts
__global__ void assign_cell_indices(
    const double (*points)[3],
    size_t n_points,
    const double inv_box[3][3],
    const bool periodic[3],
    const int32* n_cells,
    const double* face_distances,
    const double* bounding_min,
    int32* cell_indices,
    int32* particle_shifts
);

// Count particles per cell (histogram)
__global__ void count_particles_per_cell(
    const int32* cell_indices,
    size_t n_points,
    int32* cell_counts
);

// Reorder particles by cell for coalesced access in neighbor search.
// Also computes wrapped positions (position - shift @ box) per sorted atom.
__global__ void scatter_particles(
    const double (*points)[3],
    size_t n_points,
    const int32* cell_indices,
    const int32* particle_shifts,
    const double box[3][3],
    int32* cell_offsets,
    double* sorted_points,
    double* wrapped_positions,
    int32* sorted_indices,
    int32* sorted_shifts,
    int32* sorted_cell_indices
);

// Two pass cell list, this is the first pass. This kernel search all neighbors,
// and count the number of neighbor per atom.
__global__ void count_neighbors_cell_list(
    const double* wrapped_positions,
    size_t n_points,
    const double box[3][3],
    const bool* cell_out_of_reach,
    const bool periodic[3],
    const int32* n_cells,
    const int32* n_search,
    const int32* sorted_indices,
    const int32* sorted_shifts,
    const int32* sorted_cell_indices,
    const int32* cell_starts,
    const int32* cell_counts,
    double cutoff,
    bool full_list,
    int32 padded_to,
    int32* padding_overflow,
    int32* counts
);

// Two pass cell list, this is the second pass. This kernel actually fills the
// output arrays, using pre-computed offsets to write to.
__global__ void fill_neighbors_cell_list(
    const double* wrapped_positions,
    size_t n_points,
    const double box[3][3],
    const bool* cell_out_of_reach,
    const bool periodic[3],
    const int32* n_cells,
    const int32* n_search,
    const int32* sorted_indices,
    const int32* sorted_shifts,
    const int32* sorted_cell_indices,
    const int32* cell_starts,
    const int32* cell_counts,
    double cutoff,
    bool full_list,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    int32 padded_to,
    const int32* write_offsets,
    size_t (*pair_indices)[2],
    int32 (*shifts_out)[3],
    double* distances,
    double (*vectors)[3]
);

#endif // VESIN_CUDA_CELL_LIST_CUH
