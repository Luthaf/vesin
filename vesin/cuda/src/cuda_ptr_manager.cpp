#include "cuda_ptr_manager.hpp"
#include <cassert>
#include <cuda_runtime.h>

namespace vesin {
namespace cuda {

#ifndef VESIN_CUDA_MAX_NEDGES_PER_NODE
#define VESIN_CUDA_MAX_NEDGES_PER_NODE 1024
#endif

CudaPtrManager::CudaPtrManager(VesinNeighborList *neighbors)
    : neighbors(neighbors) {}

CudaPtrManager::~CudaPtrManager() { reset(); }

void CudaPtrManager::update_capacity(unsigned long nnodes) {
  assert(neighbors->device == VesinCUDA);

  if (capacity >= nnodes && length_ptr) {
    cudaMemset(length_ptr, 0, sizeof(unsigned long));
    return;
  }

  reset();

  unsigned long max_edges =
      static_cast<unsigned long>(1.2 * nnodes * VESIN_CUDA_MAX_NEDGES_PER_NODE);

  cudaMalloc(&neighbors->pairs, sizeof(unsigned long) * max_edges * 2);
  cudaMalloc(&neighbors->shifts, sizeof(int32_t) * max_edges * 3);
  cudaMalloc(&neighbors->distances, sizeof(double) * max_edges);
  cudaMalloc(&neighbors->vectors, sizeof(double) * max_edges * 3);

  cudaMalloc(&length_ptr, sizeof(unsigned long));
  cudaMemset(length_ptr, 0, sizeof(unsigned long));

  capacity = static_cast<unsigned long>(1.2 * nnodes);
}

void CudaPtrManager::reset() {
  if (neighbors->pairs)
    cudaFree(neighbors->pairs);
  if (neighbors->shifts)
    cudaFree(neighbors->shifts);
  if (neighbors->distances)
    cudaFree(neighbors->distances);
  if (neighbors->vectors)
    cudaFree(neighbors->vectors);
  if (length_ptr)
    cudaFree(length_ptr);

  neighbors->pairs = nullptr;
  neighbors->shifts = nullptr;
  neighbors->distances = nullptr;
  neighbors->vectors = nullptr;
  length_ptr = nullptr;
  capacity = 0;
}

unsigned long *CudaPtrManager::get_length_device_ptr() const { return length_ptr; }

} // namespace cuda
} // namespace vesin
