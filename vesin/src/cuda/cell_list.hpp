#ifndef VESIN_CUDA_CELL_LIST_HPP
#define VESIN_CUDA_CELL_LIST_HPP

#include <cstddef>
#include <cstdint>

#include "vesin.h"

namespace vesin::cuda {

/// Most blocks used for the first pass of the bounding box reduction. The
/// second pass reduces this many partials in a single block, and the scratch
/// buffer holding them is allocated once at this size (48 kB).
static constexpr size_t MAX_BOUNDING_BOX_BLOCKS = 1024;

/// @brief Buffers for cell list-based neighbor search
struct CellListBuffersCUDA {
    size_t points_capacity = 0; // Capacity for point-related arrays
    size_t cells_capacity = 0;  // Capacity for cell-related arrays

    // Per-particle arrays (device)
    int32_t* d_cell_indices = nullptr;    // [points_capacity] linear cell index per particle
    int32_t* d_particle_shifts = nullptr; // [points_capacity * 3] shift applied to wrap into cell

    // Per-cell arrays (device)
    int32_t* d_cell_counts = nullptr;  // [cells_capacity] number of particles in each cell
    int32_t* d_cell_starts = nullptr;  // [cells_capacity] starting index in sorted arrays
    int32_t* d_cell_offsets = nullptr; // [cells_capacity] working copy for scatter

    // Scratch for the multi-block scan of d_cell_counts → d_cell_starts
    int32_t* d_cell_block_totals = nullptr;  // [ceil(cells_capacity/1024)]
    int32_t* d_cell_block_offsets = nullptr; // [ceil(cells_capacity/1024)]
    int32_t* d_cell_block_totals2 = nullptr; // [ceil(n_blocks1/1024)]

    // Sorted particle data (device, for coalesced memory access)
    double* d_sorted_points = nullptr;        // [points_capacity * 3] raw positions (kept for compat)
    double* d_wrapped_positions = nullptr;    // [points_capacity * 3] positions wrapped into cell
    int32_t* d_sorted_indices = nullptr;      // [points_capacity] original particle indices
    int32_t* d_sorted_shifts = nullptr;       // [points_capacity * 3] shifts for sorted particles
    int32_t* d_sorted_cell_indices = nullptr; // [points_capacity] cell indices in sorted order

    // Two-pass count/fill scratch (device)
    int32_t* d_counts = nullptr;          // [points_capacity] per-atom neighbor count (pass 1)
    int32_t* d_write_offsets = nullptr;    // [points_capacity + 1] per-atom write offset (prefix sum)
    int32_t* d_block_totals = nullptr;     // [ceil(points_capacity/1024)] multi-block scan scratch
    int32_t* d_block_offsets = nullptr;    // [ceil(points_capacity/1024)] level-2 scan scratch
    int32_t* d_block_totals2 = nullptr;    // [ceil(n_blocks1/1024)] level-2 block totals (≤256 elts)

    // Cell grid parameters (device, computed on device)
    int32_t* d_n_cells = nullptr;       // [3] number of cells in each direction
    int32_t* d_n_search = nullptr;      // [3] search range in each direction
    int32_t* d_n_cells_total = nullptr; // [1] total number of cells

    double* d_face_distances = nullptr; // [3] distances between faces of the box
    double* d_bounding_min = nullptr;   // [3] bottom of the bounding box

    /// Per cell-offset reachability mask: `d_cell_out_of_reach[i]` is true when
    /// no point of the cell at offset `i` can be within the cutoff of any point
    /// of the cell at the origin. `VESIN_MAX_OUT_OF_REACH_MASK` (from
    /// kernels/cell_list.cuh) entries, filled by `compute_cell_grid_params`
    bool* d_cell_out_of_reach = nullptr;

    // [MAX_BOUNDING_BOX_BLOCKS * 6] per-block partial bounding boxes
    double* d_bbox_partials = nullptr;

    void allocate(size_t n_points, size_t n_cells);

    CellListBuffersCUDA() = default;
    ~CellListBuffersCUDA();

    CellListBuffersCUDA(CellListBuffersCUDA&& other) noexcept;
    CellListBuffersCUDA& operator=(CellListBuffersCUDA&& other) noexcept;

    CellListBuffersCUDA(const CellListBuffersCUDA&) = delete;
    CellListBuffersCUDA& operator=(const CellListBuffersCUDA&) = delete;
};

/// Compute the neighbor list with the cell list algorithm, writing the pairs
/// in `neighbors`. The output is sorted by construction.
///
/// `all_periodic` (as computed by `check_box`) allows to skip the calculation
/// of the bounding box of the points when all directions are periodic.
void cell_list_neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    bool all_periodic,
    VesinNeighborList& neighbors
);

} // namespace vesin::cuda


#endif // VESIN_CUDA_CELL_LIST_HPP
