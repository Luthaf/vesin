#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include <vector>

#include "vesin.h"

namespace vesin {
namespace cuda {

struct CudaNeighborListExtras {
    unsigned long* length_ptr = nullptr; // GPU-side counter, 1 per device
    unsigned long capacity = 0;          // Current capacity per device
    int allocated_device = -1;           // which device are we currently allocated on

    ~CudaNeighborListExtras() {
        // cleanup handled in `free_neighbors`
    }
};

/// @brief Frees GPU memory associated with a VesinNeighborList.
///
/// This function should be called to release all CUDA-allocated memory
/// tied to the given neighbor list. It does not delete the structure itself,
/// only the device-side memory buffers.
///
/// @param neighbors Reference to the VesinNeighborList to clean up.
void free_neighbors(VesinNeighborList& neighbors);

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
void neighbors(const double (*points)[3], long n_points, const double cell[3][3], VesinOptions options, VesinNeighborList& neighbors);

// used in front-end and back-end to grab the length ptr
CudaNeighborListExtras* get_cuda_extras(VesinNeighborList* neighbors);

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
