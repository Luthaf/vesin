#ifndef VESIN_VERLET_CUDA_HPP
#define VESIN_VERLET_CUDA_HPP

#include <cstdint>

#include "vesin.h"

namespace vesin {
namespace cuda {

/// Verlet cache state. Candidates are generated at cutoff + skin and filtered
/// at the exact cutoff while the reference points remain valid.
class VerletCache {
public:
    VerletCache() = default;
    ~VerletCache();

    VerletCache(VerletCache&& other) noexcept;
    VerletCache& operator=(VerletCache&& other) noexcept;

    VerletCache(const VerletCache&) = delete;
    VerletCache& operator=(const VerletCache&) = delete;

    /// Run the verlet calculation, recomputing the cache as needed
    void run(
        const double (*points)[3],
        size_t n_points,
        const double box[3][3],
        const bool periodic[3],
        VesinOptions options,
        VesinNeighborList& neighbors
    );

private:
    // reference values for points/box/periodic, used to create the cache and
    // check its validity
    size_t ref_points_capacity_ = 0; // allocated capacity for d_ref_points
    double* d_ref_points_ = nullptr; // [n_ref_points * 3]
    size_t n_ref_points_ = 0;
    double ref_box_[9] = {0.0};
    bool ref_periodic_[3] = {false, false, false};

    VesinOptions options_;
    bool has_cache_ = false;
    VesinNeighborList candidates_;
    // cached allocation for rebuild_flag
    int32_t* d_rebuild_flag_ = nullptr;

    /// Free allocated buffers
    void free_buffers();

    /// Did the options changed since the cache was built?
    bool options_changed(VesinOptions options) const;
    /// Did the box or periodicity changed since the cache was built?
    bool box_changed(const double h_box[9], const bool h_periodic[3]) const;
    /// Allocate the buffer for reference data
    void allocate_ref_buffers(size_t n_points);
    void rebuild_cache(const double (*points)[3], size_t n_points, const double box[3][3], const bool periodic[3], int32_t device_id, VesinOptions options);
    void filter_neighbors(const double (*d_points)[3], size_t n_points, const double d_box[3][3], VesinOptions options, VesinNeighborList& neighbors) const;
};

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
