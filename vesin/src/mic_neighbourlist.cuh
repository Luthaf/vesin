#ifndef MIC_NEIGHBOURLIST_CUH
#define MIC_NEIGHBOURLIST_CUH

#include <cstddef> // for size_t

namespace vesin {
namespace cuda {

#ifndef VESIN_CUDA_MAX_NEDGES_PER_NODE
/// @brief Default maximum number of edges per node on the GPU (can be
/// overridden).
#define VESIN_CUDA_MAX_NEDGES_PER_NODE 1024
#endif

/// @brief Compute the Minimum Image Convention (MIC) neighbor list on the GPU.
///
/// This function builds a neighbor list for particles in periodic boundary
/// conditions using the Minimum Image Convention. The list can optionally
/// return shift vectors, distances, and inter-particle vectors depending on the
/// flags provided.
///
/// @tparam scalar_t The floating-point type (e.g., float or double).
/// @param positions Pointer to an array of particle positions of shape [nnodes,
/// 3].
/// @param cell Pointer to a 3x3 matrix representing the simulation box.
/// @param nnodes Number of particles (nodes).
/// @param cutoff Cutoff distance for including a neighbor.
/// @param pair_counter Pointer to a device-side counter for the number of
/// generated edges.
/// @param edge_indices Pointer to device memory where neighbor index pairs will
/// be stored.
///        Expected shape: [n_max_edges * 2].
/// @param shifts Pointer to device memory for storing shift vectors (if
/// return_shifts is true).
///        Shape: [n_max_edges * 3].
/// @param distances Pointer to device memory for storing distances (if
/// return_distances is true).
///        Shape: [n_max_edges].
/// @param vectors Pointer to device memory for storing displacement vectors (if
/// return_vectors is true).
///        Shape: [n_max_edges * 3].
/// @param return_shifts Whether to compute and store shift vectors.
/// @param return_distances Whether to compute and store pairwise distances.
/// @param return_vectors Whether to compute and store pairwise displacement
/// vectors.
/// @param full Whether to build a symmetric (i,j) + (j,i) neighbor list or only
/// (i,j).
template <typename scalar_t>
void compute_mic_neighbourlist(const scalar_t* positions, const scalar_t* cell, long nnodes, scalar_t cutoff, unsigned long* pair_counter, unsigned long* edge_indices, int* shifts, scalar_t* distances, scalar_t* vectors, bool return_shifts, bool return_distances, bool return_vectors, bool full);

} // namespace cuda
} // namespace vesin

#endif // MIC_NEIGHBOURLIST_CUH
