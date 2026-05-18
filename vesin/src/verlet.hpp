#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "cluster.hpp"
#include "types.hpp"
#include "vesin.h"

namespace vesin {
namespace cpu {

/// Fixed-width candidate block for SIMD filtering of cached Verlet pairs.
struct alignas(64) VerletCandidateBlock {
    size_t count = 0;
    alignas(64) size_t first[CLUSTER_SIZE_CPU] = {};
    alignas(64) size_t second[CLUSTER_SIZE_CPU] = {};
    alignas(64) CellShift shifts[CLUSTER_SIZE_CPU] = {};
    alignas(64) double shift_x[CLUSTER_SIZE_CPU] = {};
    alignas(64) double shift_y[CLUSTER_SIZE_CPU] = {};
    alignas(64) double shift_z[CLUSTER_SIZE_CPU] = {};
};

static_assert(alignof(VerletCandidateBlock) >= 64);
static_assert(offsetof(VerletCandidateBlock, first) % 64 == 0);
static_assert(offsetof(VerletCandidateBlock, second) % 64 == 0);
static_assert(offsetof(VerletCandidateBlock, shifts) % 64 == 0);
static_assert(offsetof(VerletCandidateBlock, shift_x) % 64 == 0);
static_assert(offsetof(VerletCandidateBlock, shift_y) % 64 == 0);
static_assert(offsetof(VerletCandidateBlock, shift_z) % 64 == 0);

/// State for a cached, on-CPU Verlet neighbor list.
///
/// The state stores:
/// - a candidate neighbor list generated at `cutoff + skin`
/// - the reference coordinates and box state used to build that cache
/// - configuration parameters that invalidate the cache when changed
struct VerletState {
    /// Initialize an empty cache state.
    VerletState() = default;
    /// Release any GPU/CPU resources owned by the cached candidate list.
    ~VerletState();

    /// Disallow copy construction; neighbor cache state is process-local.
    VerletState(const VerletState&) = delete;
    /// Disallow copy assignment; neighbor cache state is process-local.
    VerletState& operator=(const VerletState&) = delete;
    /// Disallow move construction; candidate storage is handled by explicit resets.
    VerletState(VerletState&&) = delete;
    /// Disallow move assignment; candidate storage is handled by explicit resets.
    VerletState& operator=(VerletState&&) = delete;

    /// Copy options relevant to cache reuse from the user options.
    ///
    /// If the cache-driving options changed (cutoff, skin, full-list flag),
    /// clear existing cached candidates.
    void set_options(VesinOptions options);

    /// Return `true` if the cache should be rebuilt for the current frame.
    ///
    /// The caller passes the current positions and simulation box; a rebuild is
    /// required after any structural change that could invalidate cached
    /// candidates (box topology/periodicity changes, atom count changes, or large
    /// displacement from the reference positions).
    bool needs_rebuild(
        const Vector* points,
        size_t n_points,
        const BoundingBox& box
    ) const;

    /// Build the over-complete candidate cache at `cutoff + skin`.
    void rebuild(
        const Vector* points,
        size_t n_points,
        const BoundingBox& box
    );

    /// Filter cached candidates at the exact user cutoff for the current frame.
    ///
    /// This performs the displacement-based neighbor-list reuse work:
    /// candidate list generation is amortized, while output arrays are filtered by
    /// `cutoff`, and requested quantities (`S`, `d`, `D`) are emitted.
    void recompute(
        const Vector* points,
        const BoundingBox& box,
        VesinOptions options,
        VesinNeighborList& neighbors,
        size_t& output_capacity
    );

    /// Number of pairs currently stored in the cached candidate list.
    size_t candidate_count() const {
        if (use_cluster_candidates) {
            return cluster_candidates.size();
        }
        return candidates.length;
    }

    /// Number of atom-pair candidates packed into SIMD recompute blocks.
    size_t simd_candidate_count() const {
        return simd_candidate_length;
    }

    /// Reference positions at the time the candidates were built.
    std::vector<double> ref_positions;
    /// Box matrix used for candidate generation and displacement validation.
    Matrix ref_matrix = {};
    /// Periodicity flags for the cached box.
    std::array<bool, 3> ref_periodic = {false, false, false};
    /// Number of points that were used to build the cached candidate list.
    size_t n_points = 0;

    /// Over-complete candidate list generated at `cutoff + skin`.
    ///
    /// The list is kept in normal neighbor-list representation so rebuild and
    /// recompute paths can share storage and filtering logic.
    VesinNeighborList candidates;
    /// Cartesian shift vector for each materialized candidate pair.
    std::vector<Vector> candidate_shift_vectors;
    /// Fixed-width blocks used by the SIMD cached-candidate recompute path.
    std::vector<VerletCandidateBlock> simd_candidate_blocks;
    /// Number of atom-pair lanes stored in `simd_candidate_blocks`.
    size_t simd_candidate_length = 0;

    /// Cluster grid used by cluster-backed Verlet candidate caches.
    ClusterGrid cluster_grid;
    /// Over-complete cluster-pair candidates generated at `cutoff + skin`.
    std::vector<ClusterPairCandidate> cluster_candidates;
    /// Whether the active cache is represented by cluster-pair candidates.
    bool use_cluster_candidates = false;

    /// Options used to build the current cache.
    VesinOptions options = {};
    /// Rebuild threshold used to invalidate a cache (`(skin/2)^2`).
    double half_skin_sq = 0.0;

    /// Whether a usable cache is currently available.
    bool has_cache = false;

private:
    /// Release candidate buffers and reset cache metadata.
    ///
    /// This is called when a rebuild becomes invalid or when the state is
    /// destroyed. It clears all retained storage so stale entries are never
    /// reused after neighbor-list invalidation.
    void clear_candidates();
};

} // namespace cpu
} // namespace vesin

#endif
