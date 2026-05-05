#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vesin.h"

namespace vesin {

struct VerletState {
    std::vector<double> ref_positions;
    double ref_box[3][3] = {{0.0}};
    bool ref_periodic[3] = {false, false, false};
    size_t n_points = 0;

    std::vector<size_t> pairs_i;
    std::vector<size_t> pairs_j;
    std::vector<int32_t> shifts;
    size_t n_pairs = 0;
    size_t output_capacity = 0;

    double cutoff = 0.0;
    double skin = 0.0;
    double half_skin_sq = 0.0;
    bool full_list = false;

    bool did_rebuild_flag = false;
    bool has_cache = false;
};

void verlet_set_options(VerletState& state, VesinOptions options);

bool verlet_needs_rebuild(
    const VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
);

void verlet_rebuild(
    VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
);

void verlet_recompute(
    VerletState& state,
    const double (*points)[3],
    const double box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

} // namespace vesin

#endif
