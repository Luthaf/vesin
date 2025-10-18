#include <stdexcept>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double cell[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}

CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    return nullptr;
}
