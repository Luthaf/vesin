#include <stdexcept>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

void free_neighbors(VesinNeighborList & neighbors) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}

void neighbors(
    const double (*points)[3],
    size_t n_points,
    const double cell[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}