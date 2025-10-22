#ifndef MIC_NEIGHBOURLIST_CUH
#define MIC_NEIGHBOURLIST_CUH

#include "vesin.h"

namespace vesin {
namespace cuda {

/// @brief Compute the Minimum Image Convention (MIC) neighbor list on the GPU.
///
/// This function builds a neighbor list for particles in periodic boundary
/// conditions using the Minimum Image Convention. The list can optionally
/// return shift vectors, distances, and inter-particle vectors depending on the
/// flags provided.
///
/// @param points Pointer to an array of 3D points (shape: [n_points][3]).
/// @param n_points Number of points (atoms, particles, etc.).
/// @param box 3×3 simulation matrix defining the bounding box of the system.
/// @param options Struct holding parameters such as cutoff, symmetry, etc.
/// @param neighbors Output neighbor list (device memory will be allocated as
///      needed)
/// @param d_box_check device-allocated status checking valid box size with
///     respect to cutoff .
void compute_mic_neighbourlist(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    int* d_box_check,
    VesinOptions options,
    VesinNeighborList& neighbors
);

} // namespace cuda
} // namespace vesin

#endif // MIC_NEIGHBOURLIST_CUH
