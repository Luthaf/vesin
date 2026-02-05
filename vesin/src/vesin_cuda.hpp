#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include "vesin.h"

namespace vesin {
namespace cuda {

#ifndef VESIN_CUDA_MAX_PAIRS_PER_POINT
/// @brief Default maximum number of pairs per point on the GPU (can be
/// overridden).
#define VESIN_CUDA_MAX_PAIRS_PER_POINT 512
#endif

/// @brief Buffers for cell list-based neighbor search
struct CellListBuffers {
    size_t max_points = 0; // Capacity for point-related arrays
    size_t max_cells = 0;  // Capacity for cell-related arrays

    // Per-particle arrays
    int32_t* cell_indices = nullptr;    // [max_points] linear cell index per particle
    int32_t* particle_shifts = nullptr; // [max_points * 3] shift applied to wrap into cell

    // Per-cell arrays
    int32_t* cell_counts = nullptr;  // [max_cells] number of particles in each cell
    int32_t* cell_starts = nullptr;  // [max_cells] starting index in sorted arrays
    int32_t* cell_offsets = nullptr; // [max_cells] working copy for scatter

    // Sorted particle data (for coalesced memory access)
    double* sorted_positions = nullptr;     // [max_points * 3]
    int32_t* sorted_indices = nullptr;      // [max_points] original particle indices
    int32_t* sorted_shifts = nullptr;       // [max_points * 3] shifts for sorted particles
    int32_t* sorted_cell_indices = nullptr; // [max_points] cell indices in sorted order

    // Cell grid parameters (computed on device)
    double* inv_box = nullptr;        // [9] inverse box matrix
    int32_t* n_cells = nullptr;       // [3] number of cells in each direction
    int32_t* n_search = nullptr;      // [3] search range in each direction
    int32_t* n_cells_total = nullptr; // [1] total number of cells
};

struct CudaNeighborListExtras {
    size_t* length_ptr = nullptr;      // GPU-side counter
    size_t capacity = 0;               // Current capacity per device
    size_t max_pairs = 0;              // Maximum number of pairs that can be stored; depends on VESIN_CUDA_MAX_PAIRS_PER_POINT
    int32_t* cell_check_ptr = nullptr; // GPU-side status code for checking cell
    int32_t* overflow_flag = nullptr;      // GPU-side flag to detect overflow of pair buffers
    int32_t allocated_device_id = -1;  // which device are we currently allocated on

    // Pinned host memory for async D2H copy (Approach 2)
    size_t* pinned_length_ptr = nullptr;

    // Cell list buffers (allocated on demand for large systems)
    CellListBuffers cell_list;

    // Buffers for optimized brute force kernels
    double* box_diag = nullptr;      // [3] diagonal elements for orthogonal boxes
    double* inv_box_brute = nullptr; // [9] inverse box matrix for general boxes

    ~CudaNeighborListExtras();
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
CudaNeighborListExtras* get_cuda_extras(VesinNeighborList* neighbors);

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
