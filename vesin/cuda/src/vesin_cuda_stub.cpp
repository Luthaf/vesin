#include <stdexcept>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

void free_neighbors(VesinNeighborList & neighbors) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}

void ensure_neighborlist_capacity(VesinNeighborList & neighbors, size_t required_capacity) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}

void neighbors(
    const double (*points)[3],          // [n_points][3] on device
    size_t n_points,
    const double cell[3][3],            // [3][3] on device
    VesinOptions options,
    VesinNeighborList& neighbors,   // outputs already allocated on device
    int cuda_stream
) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}