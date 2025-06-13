#ifndef VESIN_CUDA_HPP
#define VESIN_CUDA_HPP

#include <vector>

#include "vesin.h"
#include "types.hpp"

namespace vesin { namespace cuda {

void free_neighbors(VesinNeighborList& neighbors);

void neighbors(
    const double (*points)[3],
    long n_points,
    const double cell[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

} // namespace cuda
} // namespace vesin

#endif