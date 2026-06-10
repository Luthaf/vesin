#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include <cstdint>

#include "verlet_cuda.hpp"
#include "vesin.h"

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
    size_t points_capacity = 0; // Capacity for point-related arrays
    size_t cells_capacity = 0;  // Capacity for cell-related arrays

    // Per-particle arrays (device)
    int32_t* d_cell_indices = nullptr;    // [points_capacity] linear cell index per particle
    int32_t* d_particle_shifts = nullptr; // [points_capacity * 3] shift applied to wrap into cell

    // Per-cell arrays (device)
    int32_t* d_cell_counts = nullptr;  // [cells_capacity] number of particles in each cell
    int32_t* d_cell_starts = nullptr;  // [cells_capacity] starting index in sorted arrays
    int32_t* d_cell_offsets = nullptr; // [cells_capacity] working copy for scatter

    // Sorted particle data (device, for coalesced memory access)
    double* d_sorted_points = nullptr;        // [points_capacity * 3]
    int32_t* d_sorted_indices = nullptr;      // [points_capacity] original particle indices
    int32_t* d_sorted_shifts = nullptr;       // [points_capacity * 3] shifts for sorted particles
    int32_t* d_sorted_cell_indices = nullptr; // [points_capacity] cell indices in sorted order

    // Cell grid parameters (device, computed on device)
    int32_t* d_n_cells = nullptr;       // [3] number of cells in each direction
    int32_t* d_n_search = nullptr;      // [3] search range in each direction
    int32_t* d_n_cells_total = nullptr; // [1] total number of cells

    double* d_face_distances = nullptr; // [3] distances between faces of the box
    double* d_bounding_min = nullptr;   // [3] bottom of the bounding box

    void allocate(size_t n_points, size_t n_cells);

    CellListBuffers() = default;
    ~CellListBuffers();

    CellListBuffers(CellListBuffers&& other) noexcept;
    CellListBuffers& operator=(CellListBuffers&& other) noexcept;

    CellListBuffers(const CellListBuffers&) = delete;
    CellListBuffers& operator=(const CellListBuffers&) = delete;
};

struct SortBuffers {
    size_t capacity = 0;                  // Capacity for the buffers below (number of pairs)
    size_t (*d_pairs_tmp)[2] = nullptr;   // [capacity] temporary pair indices for sorting
    int32_t (*d_shifts_tmp)[3] = nullptr; // [capacity] temporary shifts for sorting
    double* d_distances_tmp = nullptr;    // [capacity] temporary distances for sorting
    double (*d_vectors_tmp)[3] = nullptr; // [capacity * 3] temporary vectors for sorting

    void allocate(size_t n, bool return_shifts, bool return_distances, bool return_vectors);

    SortBuffers() = default;
    ~SortBuffers();

    SortBuffers(SortBuffers&& other) noexcept;
    SortBuffers& operator=(SortBuffers&& other) noexcept;

    SortBuffers(const SortBuffers&) = delete;
    SortBuffers& operator=(const SortBuffers&) = delete;
};

struct CudaNeighborListExtras {
    size_t pairs_capacity = 0;           // Capacity for the pair buffers in the VesinNeighborList
    size_t* d_length_ptr = nullptr;      // GPU-side counter
    int32_t* d_cell_check_ptr = nullptr; // GPU-side status code for checking cell
    int32_t* d_overflow_flag = nullptr;  // GPU-side flag to detect overflow of pair buffers

    // Pinned host memory for async D2H copy
    size_t* pinned_length_ptr = nullptr;

    // Cell list buffers (allocated on demand for large systems)
    CellListBuffers cell_list;

    // Buffers for optimized brute force kernels
    double* d_box_diag = nullptr;     // [3] diagonal elements for orthogonal boxes
    double (*d_inv_box)[3] = nullptr; // [3][3] inverse box matrix for general boxes

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

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
