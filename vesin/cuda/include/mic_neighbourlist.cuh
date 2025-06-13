#ifndef MIC_NEIGHBOURLIST_CUH
#define MIC_NEIGHBOURLIST_CUH

#include <cstddef> // for size_t

namespace vesin {
namespace cuda {

#define VESIN_CUDA_MAX_NEDGES_PER_NODE 1024

template <typename scalar_t>
void compute_mic_neighbourlist(const scalar_t *positions, const scalar_t *cell,
                               long nnodes, scalar_t cutoff,
                               unsigned long *pair_counter,
                               unsigned long *edge_indices, int *shifts,
                               scalar_t *distances, scalar_t *vectors,
                               bool return_shifts, bool return_distances,
                               bool return_vectors, bool full);

} // namespace cuda
} // namespace vesin

#endif // MIC_NEIGHBOURLIST_CUH