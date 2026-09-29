#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include <cstdint>

#include "verlet_cuda.hpp"
#include "vesin.h"

namespace vesin {
namespace cuda {

struct CudaNeighborListExtras;

/// Most blocks used for the first pass of the bounding box reduction. The
/// second pass reduces this many partials in a single block, and the scratch
/// buffer holding them is allocated once at this size (48 kB).
static constexpr size_t MAX_BOUNDING_BOX_BLOCKS = 1024;

/// Number of entries in `CellListBuffersCUDA::d_cell_out_of_reach`. Must match
/// `MAX_OUT_OF_REACH_MASK` in cuda/cell_list.cu, which fills and reads it.
static constexpr size_t MAX_OUT_OF_REACH_MASK = 32768;

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
    /// of the cell at the origin. `MAX_OUT_OF_REACH_MASK` entries, filled by
    /// `compute_cell_grid_params`
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

struct SortBuffers {
    size_t capacity = 0;                  // Capacity for the pair buffers below (number of pairs)
    size_t (*d_pairs_tmp)[2] = nullptr;   // [capacity] temporary pair indices for sorting
    int32_t (*d_shifts_tmp)[3] = nullptr; // [capacity] temporary shifts for sorting
    double* d_distances_tmp = nullptr;    // [capacity] temporary distances for sorting
    double (*d_vectors_tmp)[3] = nullptr; // [capacity * 3] temporary vectors for sorting

    // Counting sort over the first index of each pair: how many pairs each
    // point owns, and where the pairs of each point start in the output.
    size_t points_capacity = 0;          // Capacity for the per-point arrays below
    int32_t* d_counts = nullptr;         // [points_capacity] pairs per point
    int32_t* d_offsets = nullptr;        // [points_capacity] exclusive scan of d_counts

    // Scratch for the multi-block scan of d_counts → d_offsets
    int32_t* d_block_totals = nullptr;  // [ceil(points_capacity/1024)]
    int32_t* d_block_offsets = nullptr; // [ceil(points_capacity/1024)]
    int32_t* d_block_totals2 = nullptr; // [ceil(n_blocks1/1024)]

    void allocate(
        size_t n_pairs,
        size_t n_points,
        bool return_shifts,
        bool return_distances,
        bool return_vectors
    );

    SortBuffers() = default;
    ~SortBuffers();

    SortBuffers(SortBuffers&& other) noexcept;
    SortBuffers& operator=(SortBuffers&& other) noexcept;

    SortBuffers(const SortBuffers&) = delete;
    SortBuffers& operator=(const SortBuffers&) = delete;
};

/// Host-side staging for the small device-to-host transfers.
///
/// This is pinned memory, which makes `cudaMemcpyAsync` actually asynchronous.
struct PinnedReadback {
    double box[9];
    bool periodic[3];
    bool all_periodic;
    size_t length;
    int32_t n_search[3];
    int32_t last_offset;
    int32_t last_count;
    int32_t overflow;
    int32_t rebuild;
    int32_t padding_overflow;
};

struct CudaNeighborListExtras {
    size_t pairs_capacity = 0;           // Capacity for the pair buffers in the VesinNeighborList
    size_t* d_length_ptr = nullptr;      // GPU-side counter
    bool* d_all_periodic = nullptr;      // GPU-side output of `analyze_box`
    int32_t* d_overflow_flag = nullptr;  // GPU-side flag to detect overflow of pair buffers
    int32_t* d_padding_overflow = nullptr; // GPU-side first point with more than `padded_to` neighbors

    // Per-point number of pairs written so far, used with `padded_to` by the
    // paths writing pairs with atomics (brute force and Verlet)
    int32_t* d_pairs_per_point = nullptr;
    size_t pairs_per_point_capacity = 0;

    // Pinned host staging for every device-to-host readback, see PinnedReadback
    PinnedReadback* pinned = nullptr;

    // Cell list buffers (allocated on demand for large systems)
    CellListBuffersCUDA cell_list;

    // Box parameters computed by `analyze_box`
    int32_t* d_n_search = nullptr;     // [3] periodic images to walk per direction
    double (*d_inv_box)[3] = nullptr; // [3][3] inverse box matrix

    // Temporary buffers for on-device sorting
    SortBuffers sort_buffers;

    // Verlet cache state
    VerletCache verlet_cache;

    CudaNeighborListExtras() = default;
    ~CudaNeighborListExtras();

    CudaNeighborListExtras(CudaNeighborListExtras&& other) noexcept;
    CudaNeighborListExtras& operator=(CudaNeighborListExtras&& other) noexcept;

    CudaNeighborListExtras(const CudaNeighborListExtras&) = delete;
    CudaNeighborListExtras& operator=(const CudaNeighborListExtras&) = delete;
};

/// @brief Frees GPU memory associated with a VesinNeighborList.
///
/// This function should be called to release all CUDA-allocated memory
/// tied to the given neighbor list. It does not delete the structure itself,
/// only the device-side memory buffers.
///
/// @param neighbors Reference to the VesinNeighborList to clean up.
void free_neighbors(VesinNeighborList& neighbors);

/// @brief Computes the neighbor list on the GPU.
///
/// This function only works under Minimum Image Convention for now.
///
/// This function generates a neighbor list for a set of points within a
/// periodic simulation box using GPU acceleration. The output is stored in a
/// `VesinNeighborList` structure, which must be initialized for GPU usage.
///
/// @param points Pointer to an array of 3D points (shape: [n_points][3]).
/// @param n_points Number of points (atoms, particles, etc.).
/// @param box 3×3 matrix defining the bounding box of the system.
/// @param periodic Array of three booleans indicating periodicity in each dimension.
/// @param options Struct holding parameters such as cutoff, symmetry, etc.
/// @param neighbors Output neighbor list (device memory will be allocated as
/// needed).
void neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

/// Get the `CudaNeighborListExtras` stored inside `VesinNeighborList`'s opaque pointer
inline CudaNeighborListExtras* get_cuda_extras(VesinNeighborList& neighbors) {
    if (neighbors.opaque == nullptr) {
        neighbors.opaque = new vesin::cuda::CudaNeighborListExtras();
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
}

/// Allocate output buffers in the `VesinNeighborList` according to the options
/// and the given number of pairs. If the current capacity is sufficient, this
/// is a no-op. Otherwise, existing buffers are freed and new ones are allocated
/// with the requested capacity.
void allocate_output_buffers(VesinNeighborList& neighbors, size_t n_pairs, VesinOptions options);

/// Prepare `neighbors` for a calculation with `options.padded_to > 0` on one
/// of the paths writing pairs with atomics: allocate the output for exactly
/// `n_points * padded_to` pairs, and reset the per-point counters and the
/// overflow detection.
void prepare_padding(VesinNeighborList& neighbors, size_t n_points, VesinOptions options);

/// Finish a calculation with `options.padded_to > 0` started with
/// `prepare_padding`: fill the unused slots with padding pairs, set the length
/// of `neighbors`, and throw an error if any point had more than `padded_to`
/// neighbors.
void finish_padding(VesinNeighborList& neighbors, size_t n_points, VesinOptions options);

/// Sort the pairs in `neighbors` by the first particle index, using a counting
/// sort on the GPU over the `n_points` possible values of that index.
void sort_pairs(size_t n_points, VesinOptions options, VesinNeighborList& neighbors);

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
