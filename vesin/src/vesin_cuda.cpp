#include "vesin_cuda.hpp"
#include "mic_neighbourlist.cuh"

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

using namespace vesin::cuda;
using namespace std;

#define WARP_SIZE 32
#define NWARPS 4

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +    \
                               std::to_string(__LINE__) + " - " +              \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

static void ensure_is_device_pointer(const void *p, const char *name) {
  cudaPointerAttributes attr;

  CUDA_CHECK(cudaPointerGetAttributes(&attr, p));

  if (attr.type != cudaMemoryTypeDevice) {
    throw std::runtime_error(
        std::string(name) +
        " is not a device pointer (type=" + std::to_string(attr.type) + ")");
  }
}

inline bool is_device_ptr(const void *ptr, const char *name) {
  if (!ptr)
    return false;
  try {
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    return (attr.type == cudaMemoryTypeDevice);
  } catch (const std::runtime_error &e) {
    return false;
  }
}

vesin::cuda::CudaNeighborListExtras *
vesin::cuda::get_cuda_extras(VesinNeighborList *neighbors) {
  if (!neighbors->opaque) {
    try {
      neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
      vesin::cuda::CudaNeighborListExtras *test =
          static_cast<vesin::cuda::CudaNeighborListExtras *>(neighbors->opaque);
    } catch (...) {
      neighbors->opaque = nullptr;
      throw;
    }
  }
  return static_cast<vesin::cuda::CudaNeighborListExtras *>(neighbors->opaque);
}

void reset(VesinNeighborList &neighbors) {

  auto extras = vesin::cuda::get_cuda_extras(&neighbors);

  if (neighbors.pairs && is_device_ptr(neighbors.pairs, "pairs")) {
    CUDA_CHECK(cudaFree(neighbors.pairs));
  }
  if (neighbors.shifts && is_device_ptr(neighbors.shifts, "shifts")) {
    CUDA_CHECK(cudaFree(neighbors.shifts));
  }
  if (neighbors.distances && is_device_ptr(neighbors.distances, "distances")) {
    CUDA_CHECK(cudaFree(neighbors.distances));
  }
  if (neighbors.vectors && is_device_ptr(neighbors.vectors, "vectors")) {
    CUDA_CHECK(cudaFree(neighbors.vectors));
  }
  if (extras->length_ptr &&
      is_device_ptr(extras->length_ptr, "extras->length_ptr")) {
    CUDA_CHECK(cudaFree(extras->length_ptr));
  }

  neighbors.pairs = nullptr;
  neighbors.shifts = nullptr;
  neighbors.distances = nullptr;
  neighbors.vectors = nullptr;
  extras->length_ptr = nullptr;
  extras->capacity = 0;
}

void update_capacity(unsigned long nnodes, VesinNeighborList &neighbors) {
  assert(neighbors.device == VesinCUDA);

  auto extras = vesin::cuda::get_cuda_extras(&neighbors);

  if (extras->capacity >= nnodes && extras->length_ptr) {
    std::cout << "update_capacity: extras->length_ptr: " << extras->length_ptr
              << std::endl;
    CUDA_CHECK(cudaMemset(extras->length_ptr, 0, sizeof(unsigned long)));
    return;
  }

  reset(neighbors);

  unsigned long max_edges =
      static_cast<unsigned long>(1.2 * nnodes * VESIN_CUDA_MAX_NEDGES_PER_NODE);

  CUDA_CHECK(cudaMalloc((void **)&neighbors.pairs,
                        sizeof(unsigned long) * max_edges * 2));
  CUDA_CHECK(
      cudaMalloc((void **)&neighbors.shifts, sizeof(int32_t) * max_edges * 3));
  CUDA_CHECK(
      cudaMalloc((void **)&neighbors.distances, sizeof(double) * max_edges));
  CUDA_CHECK(
      cudaMalloc((void **)&neighbors.vectors, sizeof(double) * max_edges * 3));

  CUDA_CHECK(cudaMalloc((void **)&extras->length_ptr, sizeof(unsigned long)));

  CUDA_CHECK(cudaMemset(extras->length_ptr, 0, sizeof(unsigned long)));

  extras->capacity = static_cast<unsigned long>(1.2 * nnodes);
}

void vesin::cuda::free_neighbors(VesinNeighborList &neighbors) {

  assert(neighbors.device == VesinCUDA);

  reset(neighbors);

  if (neighbors.opaque) {
    delete static_cast<vesin::cuda::CudaNeighborListExtras *>(neighbors.opaque);
    neighbors.opaque = nullptr;
  }
}

void vesin::cuda::neighbors(const double (*points)[3], long n_points,
                            const double cell[3][3], VesinOptions options,
                            VesinNeighborList &neighbors) {

  assert(neighbors.device == VesinCUDA);
  assert(!neighbors.sort &&
         "Sorting is not supported in CUDA version of Vesin");

  auto extras = vesin::cuda::get_cuda_extras(&neighbors);

  update_capacity(n_points, neighbors);

  ensure_is_device_pointer(points, "points");

  if (cell) {
    ensure_is_device_pointer(cell, "cell");
  }

  const double *d_positions = reinterpret_cast<const double *>(points);
  const double *d_cell = reinterpret_cast<const double *>(cell);

  unsigned long *d_edge_indices =
      reinterpret_cast<unsigned long *>(neighbors.pairs);
  int *d_shifts = reinterpret_cast<int *>(neighbors.shifts);
  double *d_distances = reinterpret_cast<double *>(neighbors.distances);
  double *d_vectors = reinterpret_cast<double *>(neighbors.vectors);
  unsigned long *d_pair_counter = extras->length_ptr;

  vesin::cuda::compute_mic_neighbourlist<double>(
      d_positions, d_cell, n_points, (double)options.cutoff, d_pair_counter,
      d_edge_indices, d_shifts, d_distances, d_vectors, options.return_shifts,
      options.return_distances, options.return_vectors, options.full);
}