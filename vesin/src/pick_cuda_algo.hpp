#ifndef VESIN_PICK_CUDA_ALGO_HPP
#define VESIN_PICK_CUDA_ALGO_HPP

#include <cstddef>
#include <cstdint>

namespace vesin {
namespace cuda {

/// Should `VesinAutoAlgorithm` run the cell list for a search of `n_points`
/// points walking `n_images` periodic images of each pair, rather than the
/// brute force kernels?
///
/// The cell list wins above a number of points, and the brute force below.
/// Where exactly the two cross is measured on `device_id` the first time that
/// device is asked about, by timing both algorithms on a pair of synthetic
/// lattices, and kept for the rest of the process.
bool auto_use_cell_list(
    double n_points,
    double n_images,
    double cutoff,
    bool sorted,
    int32_t device_id
);

} // namespace cuda
} // namespace vesin

#endif // VESIN_PICK_CUDA_ALGO_HPP
