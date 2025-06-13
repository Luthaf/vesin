#ifndef SIMPLE_NEIGHBOURLIST_CUH
#define SIMPLE_NEIGHBOURLIST_CUH

#include <cstddef> // for size_t

namespace vesin {
namespace cuda {

#define VESIN_CUDA_MAX_NEDGES_PER_NODE 1024

template <typename scalar_t>
void compute_simple_neighbourlist(const scalar_t *positions,
                                  const scalar_t *cell, long nnodes,
                                  scalar_t cutoff, unsigned long *pair_counter,
                                  unsigned long *edge_indices, int *shifts);

} // namespace cuda
} // namespace vesin

#endif // SIMPLE_NEIGHBOURLIST_CUH