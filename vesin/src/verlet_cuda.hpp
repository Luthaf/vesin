#ifndef VESIN_VERLET_CUDA_HPP
#define VESIN_VERLET_CUDA_HPP

#include <cstdint>

#include "vesin.h"

namespace vesin {
namespace cuda {

/// Verlet cache state. Candidates are generated at cutoff + skin and filtered
/// at the exact cutoff while the reference positions remain valid.
class VerletCache {
public:

    ~VerletCache();
    void free_buffers();
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
    VesinNeighborList candidates;
    double* d_ref_positions = nullptr; // [h_n_points * 3]
    int32_t* d_rebuild_flag = nullptr; // [1]
    size_t h_ref_capacity = 0;
    size_t h_n_points = 0;
    double h_ref_box[9] = {0.0};
    bool h_ref_periodic[3] = {false, false, false};
    VesinOptions h_options = {};
    double h_half_skin_sq = 0.0;
    bool h_has_cache = false;

    /// Did the options changed since the cache was built?
    bool options_changed(VesinOptions options) const;
    /// Did the box or periodicity changed since the cache was built?
    bool box_changed(const double h_box[9], const bool h_periodic[3]) const;
    /// Allocate the buffer for reference data
    void allocate_ref_buffers(size_t n_points);
    void rebuild_cache(const double (*points)[3], size_t n_points, const double box[3][3], const bool periodic[3], int32_t device_id, VesinOptions options, const double h_box[9], const bool h_periodic[3]);
    void filter_neighbors(const double (*d_positions)[3], size_t n_points, const double d_box[3][3], VesinOptions options, VesinNeighborList& neighbors) const;
};

} // namespace cuda
} // namespace vesin

#endif // VESIN_CUDA_HPP
