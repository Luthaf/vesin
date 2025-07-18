#ifndef CUDA_NBRLIST_EXTRAS_HPP
#define CUDA_NBRLIST_EXTRAS_HPP

#include <cuda_runtime.h>

namespace vesin {
namespace cuda {
struct CudaNeighborListExtras {
  unsigned long *length_ptr = nullptr; // GPU-side counter
  unsigned long capacity = 0;          // Current capacity

  ~CudaNeighborListExtras() {
    // cleanup handled in `free_neighbors`
  }
};
} // namespace cuda
} // namespace vesin

#endif