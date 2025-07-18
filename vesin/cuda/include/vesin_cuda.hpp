#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include <vector>

#include "types.hpp"
#include "vesin.h"
#include "cuda_nbrlist_extras.hpp"

namespace vesin {
namespace cuda {

/// @brief Frees GPU memory associated with a VesinNeighborList.
///
/// This function should be called to release all CUDA-allocated memory
/// tied to the given neighbor list. It does not delete the structure itself,
/// only the device-side memory buffers.
///
/// @param neighbors Reference to the VesinNeighborList to clean up.
void free_neighbors(VesinNeighborList &neighbors);

/// @brief Computes the neighbor list on the GPU using the Minimum Image
/// Convention.
///
/// This function generates a neighbor list for a set of points within a
/// periodic simulation cell using GPU acceleration. The output is stored in a
/// `VesinNeighborList` structure, which must be initialized for GPU usage.
///
/// @param points Pointer to an array of 3D points (shape: [n_points][3]).
/// @param n_points Number of points (atoms, particles, etc.).
/// @param cell 3×3 simulation box matrix defining the periodic boundary
/// conditions.
/// @param options Struct holding parameters such as cutoff, symmetry, etc.
/// @param neighbors Output neighbor list (device memory will be allocated as
/// needed).
void neighbors(const double (*points)[3], long n_points,
               const double cell[3][3], VesinOptions options,
               VesinNeighborList &neighbors);

CudaNeighborListExtras * get_cuda_extras(VesinNeighborList *neighbors);

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
