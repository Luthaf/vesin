#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "types.hpp"
#include "vesin.h"

namespace vesin {
namespace cpu {

struct VerletState {
    VerletState() = default;
    ~VerletState();

    VerletState(const VerletState&) = delete;
    VerletState& operator=(const VerletState&) = delete;
    VerletState(VerletState&&) = delete;
    VerletState& operator=(VerletState&&) = delete;

    void set_options(VesinOptions options);

    bool needs_rebuild(
        const Vector* points,
        size_t n_points,
        const BoundingBox& box
    ) const;

    void rebuild(
        const Vector* points,
        size_t n_points,
        const BoundingBox& box
    );

    void recompute(
        const Vector* points,
        const BoundingBox& box,
        VesinOptions options,
        VesinNeighborList& neighbors
    );

    size_t candidate_count() const {
        return candidates.length;
    }

    std::vector<double> ref_positions;
    Matrix ref_matrix = {};
    std::array<bool, 3> ref_periodic = {false, false, false};
    size_t n_points = 0;

    VesinNeighborList candidates;
    size_t output_capacity = 0;

    double cutoff = 0.0;
    double skin = 0.0;
    double half_skin_sq = 0.0;
    bool full_list = false;

    bool did_rebuild_flag = false;
    bool has_cache = false;

private:
    void clear_candidates();
};

} // namespace cpu
} // namespace vesin

#endif
