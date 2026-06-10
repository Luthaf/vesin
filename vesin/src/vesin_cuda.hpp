#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include <cstdint>

#include "vesin.h"
#include "verlet_cuda.hpp"

namespace vesin {
namespace cuda {

struct CudaNeighborListExtras;

#ifndef VESIN_CUDA_AT_LEAST_PAIRS_PER_POINT
/// Default value for the number of pairs per points in the CUDA implementation.
/// Unless `VESIN_CUDA_MAX_PAIRS_PER_POINT` is set in the environement, the
/// maximal number of pairs is `n_points *
/// max(VESIN_CUDA_AT_LEAST_PAIRS_PER_POINT, cutoff^3)`. This can be overriden
/// at compile time.
#define VESIN_CUDA_AT_LEAST_PAIRS_PER_POINT 128
#endif

/// @brief Buffers for cell list-based neighbor search
struct CellListBuffers {
    size_t h_max_points = 0; // Capacity for point-related arrays
    size_t h_max_cells = 0;  // Capacity for cell-related arrays

    // Per-particle arrays (device)
    int32_t* d_cell_indices = nullptr;    // [h_max_points] linear cell index per particle
    int32_t* d_particle_shifts = nullptr; // [h_max_points * 3] shift applied to wrap into cell

    // Per-cell arrays (device)
    int32_t* d_cell_counts = nullptr;  // [h_max_cells] number of particles in each cell
    int32_t* d_cell_starts = nullptr;  // [h_max_cells] starting index in sorted arrays
    int32_t* d_cell_offsets = nullptr; // [h_max_cells] working copy for scatter

    // Sorted particle data (device, for coalesced memory access)
    double* d_sorted_positions = nullptr;     // [h_max_points * 3]
    int32_t* d_sorted_indices = nullptr;      // [h_max_points] original particle indices
    int32_t* d_sorted_shifts = nullptr;       // [h_max_points * 3] shifts for sorted particles
    int32_t* d_sorted_cell_indices = nullptr; // [h_max_points] cell indices in sorted order

    // Cell grid parameters (device, computed on device)
    double* d_inv_box = nullptr;        // [9] inverse box matrix
    int32_t* d_n_cells = nullptr;       // [3] number of cells in each direction
    int32_t* d_n_search = nullptr;      // [3] search range in each direction
    int32_t* d_n_cells_total = nullptr; // [1] total number of cells

    double* d_face_distances = nullptr; // [3] distances between faces of the box
    double* d_bounding_min = nullptr;   // [3] bottom of the bounding box
};

struct CudaNeighborListExtras {
    size_t* d_length_ptr = nullptr;      // GPU-side counter
    size_t h_max_pairs = 0;              // Maximum number of pairs that can be stored; depends on VESIN_CUDA_MAX_PAIRS_PER_POINT
    int32_t* d_cell_check_ptr = nullptr; // GPU-side status code for checking cell
    int32_t* d_overflow_flag = nullptr;  // GPU-side flag to detect overflow of pair buffers
    int32_t h_allocated_device_id = -1;  // which device are we currently allocated on

    // Pinned host memory for async D2H copy (Approach 2)
    size_t* h_pinned_length_ptr = nullptr;

    // Cell list buffers (allocated on demand for large systems)
    CellListBuffers cell_list;

    // Buffers for optimized brute force kernels
    double* d_box_diag = nullptr;      // [3] diagonal elements for orthogonal boxes
    double* d_inv_box_brute = nullptr; // [9] inverse box matrix for general boxes

    // Temporary buffers for on-device sorting
    size_t* d_sort_pairs_tmp = nullptr;     // [h_sort_capacity * 2]
    int32_t* d_sort_shifts_tmp = nullptr;   // [h_sort_capacity * 3]
    double* d_sort_distances_tmp = nullptr; // [h_sort_capacity]
    double* d_sort_vectors_tmp = nullptr;   // [h_sort_capacity * 3]
    size_t h_sort_capacity = 0;

    // Verlet cache state
    VerletCache verlet_cache;

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

void allocate_output_buffers(VesinNeighborList& neighbors, size_t n_pairs, VesinOptions options);

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
