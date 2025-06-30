#include "vesin_cuda.hpp"
#include "cuda_ptr_registry.hpp"

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

static void ensure_is_device_pointer(const void *p, const char *name) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, p);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaPointerGetAttributes failed for ") + name + ": " +
        cudaGetErrorString(err));
  }

  if (attr.type != cudaMemoryTypeDevice) {
    throw std::runtime_error(
        std::string(name) +
        " is not a device pointer (type=" + std::to_string(attr.type) + ")");
  }
}

void vesin::cuda::free_neighbors(VesinNeighborList &neighbors) {

  assert(neighbors.device == VesinCUDA);

  auto &manager = vesin::cuda::CudaPtrRegistry::get(&neighbors);

  manager.reset();
}

void vesin::cuda::neighbors(const double (*points)[3], long n_points,
                            const double cell[3][3], VesinOptions options,
                            VesinNeighborList &neighbors) {

  assert(neighbors.device == VesinCUDA);
  assert(!neighbors.sort &&
         "Sorting is not supported in CUDA version of Vesin");

  auto &manager = vesin::cuda::CudaPtrRegistry::get(&neighbors);

  manager.update_capacity(n_points * VESIN_CUDA_MAX_NEDGES_PER_NODE);

  const double *d_positions = reinterpret_cast<const double *>(points);
  const double *d_cell = reinterpret_cast<const double *>(cell);

  unsigned long *d_edge_indices =
      reinterpret_cast<unsigned long *>(neighbors.pairs);
  int *d_shifts = reinterpret_cast<int *>(neighbors.shifts);
  double *d_distances = reinterpret_cast<double *>(neighbors.distances);
  double *d_vectors = reinterpret_cast<double *>(neighbors.vectors);
  unsigned long *d_pair_counter = manager.get_length_device_ptr();

  vesin::cuda::compute_mic_neighbourlist<double>(
      d_positions, d_cell, n_points, (double)options.cutoff, d_pair_counter,
      d_edge_indices, d_shifts, d_distances, d_vectors, options.return_shifts,
      options.return_distances, options.return_vectors, options.full);
}