#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "types.hpp"
#include "vesin.h"

namespace vesin {
namespace cpu {

struct CellListBuffersCPU;

/// All reusable allocations and persistent state for the Verlet neighbor list,
/// stored separately from `VerletList` so they persist across cache
/// invalidations and are reused across calls.
struct VerletBuffersCPU {
    VerletBuffersCPU() = default;
    ~VerletBuffersCPU();

    VerletBuffersCPU(const VerletBuffersCPU&) = delete;
    VerletBuffersCPU& operator=(const VerletBuffersCPU&) = delete;
    VerletBuffersCPU(VerletBuffersCPU&&) = delete;
    VerletBuffersCPU& operator=(VerletBuffersCPU&&) = delete;

    /// Options used to build the current cache.
    VesinOptions options;

    /// Whether a usable cache is currently available.
    bool has_cache = false;

    /// Over-complete candidate list generated at `cutoff + skin`.
    VesinNeighborList candidates;

    /// Reference points at the time the candidates were built.
    std::vector<Vector> ref_points;
    /// Box matrix used for candidate generation and displacement validation.
    Matrix ref_matrix;
    /// Periodicity flags for the cached box.
    std::array<bool, 3> ref_periodic;

    /// Scratch buffer containing per-block match count (filter pass 1).
    std::vector<size_t> block_counts;
    /// Scratch buffer containing per-block output offset (prefix sum of
    /// `block_counts`).
    std::vector<size_t> block_offsets;
    /// Scratch buffer containing pair vectors for matching candidates
    std::vector<Vector> candidates_vectors;
    /// Scratch buffer containing squared distances for all candidates
    std::vector<double> candidates_distances_sq;

    /// Measured cost of filtering a single candidate on a single thread, in
    /// microseconds. <0 until the first call to `VerletList::filter()` has
    /// measured it.
    double candidate_cost_us = -1.0;
};

/// On-CPU Verlet neighbor list.
///
/// This class is a lightweight wrapper around a `VerletBuffersCPU` reference,
/// providing the cache-management logic (rebuild validation) and the
/// two-pass parallel filter. It is designed to be constructed per-call,
/// like `CellList`, while the buffers persist in `ExtraDataCpu`.
struct VerletList {
    /// Construct a Verlet list operating on the given `buffers`.
    VerletList(VerletBuffersCPU& buffers) : buffers_(buffers) {}
    ~VerletList() = default;

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
    bool needs_rebuild(
        const Vector* points,
        size_t n_points,
        const BoundingBox& box
    ) const;

    /// Build the over-complete candidate list at `cutoff + skin`.
    void rebuild(
        const Vector* points,
        size_t n_points,
        const BoundingBox& box,
        CellListBuffersCPU& cell_buffers
    );

    /// Filter cached candidates at the exact user cutoff for the current
    /// `points`.
    void filter(
        const Vector* points,
        const BoundingBox& box,
        VesinOptions options,
        VesinNeighborList& neighbors,
        size_t& output_capacity
    );

private:
    /// Release candidate buffers and reset cache metadata.
    void clear_candidates();

    VerletBuffersCPU& buffers_;
};

} // namespace cpu
} // namespace vesin

#endif
