#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "types.hpp"
#include "vesin.h"

namespace vesin {
namespace cpu {

/// On-CPU Verlet neighbor list.
///
/// This class manages an over-complete candidate neighbor list, computed for
/// `cutoff + skin`, and the associated reference state for validating
/// neighbor-list reuse.
struct VerletList {
    /// Initialize an empty Verlet list with no cached candidates.
    VerletList() = default;
    ~VerletList();

    VerletList(const VerletList&) = delete;
    VerletList& operator=(const VerletList&) = delete;
    VerletList(VerletList&&) = delete;
    VerletList& operator=(VerletList&&) = delete;

    /// Copy options relevant to cache reuse from the user options.
    ///
    /// If the cache-driving options changed (cutoff, skin, full-list flag),
    /// clear existing cached candidates.
    void set_options(VesinOptions options);

    /// Return `true` if the cache should be rebuilt for the current `points`.
    ///
    /// The caller passes the current points and box; a rebuild is required
    /// after any structural change that could invalidate cached candidates (box
    /// topology/periodicity changes, points count changes, or large
    /// displacement from the reference points).
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

    /// Filter cached candidates at the exact user cutoff for the current
    /// `points`.
    void filter(
        const Vector* points,
        const BoundingBox& box,
        VesinOptions options,
        VesinNeighborList& neighbors,
        size_t& output_capacity
    ) const;

    /// Number of pairs currently stored in the cached candidate list.
    size_t candidate_count() const {
        return candidates_.length;
    }

private:
    /// Release candidate buffers and reset cache metadata.
    void clear_candidates();

    /// Reference points at the time the candidates were built.
    std::vector<Vector> ref_points_;
    /// Box matrix used for candidate generation and displacement validation.
    Matrix ref_matrix_;
    /// Periodicity flags for the cached box.
    std::array<bool, 3> ref_periodic_ = {false, false, false};

    /// Over-complete candidate list generated at `cutoff + skin`.
    ///
    /// The list is kept in normal neighbor-list representation so rebuild and
    /// recompute paths can share storage and filtering logic.
    VesinNeighborList candidates_;

    /// Options used to build the current cache.
    VesinOptions options_ = {};
    /// Rebuild threshold used to invalidate a cache (`(skin/2)^2`).
    double half_skin_sq_ = 0.0;

    /// Whether a usable cache is currently available.
    bool has_cache_ = false;
};

} // namespace cpu
} // namespace vesin

#endif
