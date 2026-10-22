#ifndef VESIN_CUDA_NEIGHBORS_HPP
#define VESIN_CUDA_NEIGHBORS_HPP

#include <cstddef>

#include "vesin.h"

namespace vesin::cuda {

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

/// Allocate output buffers in the `VesinNeighborList` according to the options
/// and the given number of pairs. If the current capacity is sufficient, this
/// is a no-op. Otherwise, existing buffers are freed and new ones are allocated
/// with the requested capacity.
void allocate_output_buffers(VesinNeighborList& neighbors, size_t n_pairs, VesinOptions options);

} // namespace vesin::cuda


#endif // VESIN_CUDA_NEIGHBORS_HPP
