#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "types.hpp"
#include "vesin.h"

namespace vesin {
namespace cpu {

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

    /// Build the over-complete candidate list at `cutoff + skin`.
    ///
    /// The rebuild operation stores candidates in full `VesinNeighborList` form
    /// and captures the state used to validate future `needs_rebuild` checks.
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
        return candidates.length;
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
