#include "cuda_ptr_manager.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vesin {
namespace cuda {

#ifndef VESIN_CUDA_MAX_NEDGES_PER_NODE
#define VESIN_CUDA_MAX_NEDGES_PER_NODE 1024
#endif

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +    \
                               std::to_string(__LINE__) + " - " +              \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

inline bool is_device_ptr(const void *ptr) {
  if (!ptr)
    return false;
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  return (attr.type == cudaMemoryTypeDevice);
}

CudaPtrManager::CudaPtrManager(VesinNeighborList *neighbors)
    : neighbors(neighbors) {}

CudaPtrManager::~CudaPtrManager() { reset(); }

void CudaPtrManager::update_capacity(unsigned long nnodes) {
  assert(neighbors->device == VesinCUDA);

  if (capacity >= nnodes && length_ptr) {
    CUDA_CHECK(cudaMemset(length_ptr, 0, sizeof(unsigned long)));
    return;
  }

  reset();

  unsigned long max_edges =
      static_cast<unsigned long>(1.2 * nnodes * VESIN_CUDA_MAX_NEDGES_PER_NODE);

  CUDA_CHECK(
      cudaMalloc(&neighbors->pairs, sizeof(unsigned long) * max_edges * 2));
  CUDA_CHECK(cudaMalloc(&neighbors->shifts, sizeof(int32_t) * max_edges * 3));
  CUDA_CHECK(cudaMalloc(&neighbors->distances, sizeof(double) * max_edges));
  CUDA_CHECK(cudaMalloc(&neighbors->vectors, sizeof(double) * max_edges * 3));

  CUDA_CHECK(cudaMalloc(&length_ptr, sizeof(unsigned long)));
  CUDA_CHECK(cudaMemset(length_ptr, 0, sizeof(unsigned long)));

  capacity = static_cast<unsigned long>(1.2 * nnodes);
}

void CudaPtrManager::reset() {
  if (neighbors->pairs && is_device_ptr(neighbors->pairs))
    CUDA_CHECK(cudaFree(neighbors->pairs));
  if (neighbors->shifts && is_device_ptr(neighbors->shifts))
    CUDA_CHECK(cudaFree(neighbors->shifts));
  if (neighbors->distances && is_device_ptr(neighbors->distances))
    CUDA_CHECK(cudaFree(neighbors->distances));
  if (neighbors->vectors && is_device_ptr(neighbors->vectors))
    CUDA_CHECK(cudaFree(neighbors->vectors));
  if (length_ptr && is_device_ptr(length_ptr))
    CUDA_CHECK(cudaFree(length_ptr));

  neighbors->pairs = nullptr;
  neighbors->shifts = nullptr;
  neighbors->distances = nullptr;
  neighbors->vectors = nullptr;
  length_ptr = nullptr;
  capacity = 0;
}

unsigned long *CudaPtrManager::get_length_device_ptr() const {
  return length_ptr;
}

} // namespace cuda
} // namespace vesin
