#ifndef VESIN_CUDA_BRUTEFORCE_HPP
#define VESIN_CUDA_BRUTEFORCE_HPP

#include <cmath>
#include <cstddef>

#include "vesin.h"

namespace vesin::cuda {

/// Properties of the box, computed on the device by `check_box`.
struct BoxChecks {
    /// Is the box periodic in all three directions?
    bool all_periodic = false;
    /// How many periodic images of each pair the brute force kernels would
    /// walk. This is 1 for a box at least twice the cutoff in every periodic
    /// direction, and grows as the box gets smaller than that.
    size_t n_images = 1;
    /// Volume taken by the points, see `analyze_box` in `kernels/bruteforce.cuh`.
    /// This is only computed when `check_box` is called with
    /// `compute_volume=true`, and NaN otherwise.
    double points_volume = std::nan("");
};

/// Look at the box and compute the parameters needed by the brute force
/// algorithm (the inverse box and the number of periodic images to search in
/// each direction), storing them in the extra data of `neighbors`. This must
/// be called before `brute_force_neighbors`, and synchronizes the stream.
///
/// If `compute_volume` is true, this also measures the volume taken by the
/// points, to be used when picking an algorithm.
BoxChecks check_box(
    VesinNeighborList& neighbors,
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    double cutoff,
    bool compute_volume
);

/// Compute the neighbor list with the brute force algorithm, writing the pairs
/// in `neighbors`. `check_box` must have been called before this function.
void brute_force_neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

} // namespace vesin::cuda


#endif // VESIN_CUDA_BRUTEFORCE_HPP
