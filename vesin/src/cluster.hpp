#ifndef VESIN_CLUSTER_HPP
#define VESIN_CLUSTER_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "vesin.h"
#include "types.hpp"

namespace vesin {

/// Size of a cluster on CPU (maps to SIMD width for 4-wide float ops)
static constexpr int32_t CLUSTER_SIZE_CPU = 4;

/// A cluster of up to CLUSTER_SIZE_CPU atoms with a bounding box.
struct Cluster {
    int32_t atom_indices[CLUSTER_SIZE_CPU];
    int32_t n_atoms;      // actual count (<= CLUSTER_SIZE_CPU)
    float bb_lower[3];    // bounding box min (float for SIMD efficiency)
    float bb_upper[3];    // bounding box max
};

/// Grid of clusters organized in 3D cells.
struct ClusterGrid {
    std::vector<Cluster> clusters;

    // Grid dimensions (number of cells in each direction)
    std::array<int32_t, 3> n_cells;

    // Which clusters belong to which cell: cell_offsets[cell_idx] to
    // cell_offsets[cell_idx+1] gives the range of cluster indices in
    // the clusters array.
    std::vector<int32_t> cell_offsets; // [n_cells_total + 1], CSR-style

    // Per-atom wrap shift: when an atom's fractional coordinate falls
    // outside [0, n_cells), it is wrapped into the grid and the integer
    // shift is recorded here. Indexed by original atom index.
    std::vector<CellShift> atom_wrap_shifts;
};

/// Build a cluster grid from atom positions.
///
/// Algorithm:
/// 1. Compute grid cell dimensions from box vectors and cutoff
/// 2. Assign atoms to grid cells (fractional coordinate binning)
/// 3. Within each cell, sort atoms by z coordinate, group into clusters
/// 4. Compute cluster bounding boxes (AABB)
ClusterGrid build_cluster_grid(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    double cutoff
);

/// Minimum squared distance between two AABBs.
/// Returns 0 if the boxes overlap.
inline float bb_distance_sq(const Cluster& a, const Cluster& b) {
    float dist_sq = 0.0f;
    for (int d = 0; d < 3; d++) {
        float gap = 0.0f;
        if (a.bb_lower[d] > b.bb_upper[d]) {
            gap = a.bb_lower[d] - b.bb_upper[d];
        } else if (b.bb_lower[d] > a.bb_upper[d]) {
            gap = b.bb_lower[d] - a.bb_upper[d];
        }
        dist_sq += gap * gap;
    }
    return dist_sq;
}

/// Minimum squared distance between two AABBs where cluster_b is shifted
/// by a Cartesian offset (for periodic images). When shift is zero, this
/// is equivalent to bb_distance_sq.
inline float bb_distance_sq_shifted(
    const Cluster& a, const Cluster& b, const float shift[3]
) {
    float dist_sq = 0.0f;
    for (int d = 0; d < 3; d++) {
        float b_lo = b.bb_lower[d] + shift[d];
        float b_hi = b.bb_upper[d] + shift[d];
        float gap = 0.0f;
        if (a.bb_lower[d] > b_hi) {
            gap = a.bb_lower[d] - b_hi;
        } else if (b_lo > a.bb_upper[d]) {
            gap = b_lo - a.bb_upper[d];
        }
        dist_sq += gap * gap;
    }
    return dist_sq;
}

namespace cpu {

/// Cluster-pair neighbor search. Replaces cell_list for N >= 64.
///
/// Output format is identical to the cell-list path: per-atom pairs with
/// optional shifts, distances, and vectors in VesinNeighborList.
void cluster_pair_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox cell,
    VesinOptions options,
    VesinNeighborList& neighbors
);

} // namespace cpu
} // namespace vesin

#endif
