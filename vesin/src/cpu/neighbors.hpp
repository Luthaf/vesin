#ifndef VESIN_CPU_NEIGHBORS_HPP
#define VESIN_CPU_NEIGHBORS_HPP

#include "vesin.h"

#include "./types.hpp"

namespace vesin::cpu {

void neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    VesinNeighborList& neighbors
);

void free_neighbors(VesinNeighborList& neighbors);

}  // namespace vesin::cpu

#endif
