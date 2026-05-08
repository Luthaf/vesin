#include <stdexcept>

#include "vesin_cuda.hpp"

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    throw std::runtime_error("CUDA neighbor list generation is not included in this build of vesin");
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    throw std::runtime_error("CUDA neighbor list generation is not included in this build of vesin");
}
