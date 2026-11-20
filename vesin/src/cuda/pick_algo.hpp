#ifndef VESIN_CUDA_PICK_ALGO_HPP
#define VESIN_CUDA_PICK_ALGO_HPP

#include <cstddef>
#include <cstdint>

namespace vesin::cuda {

/// Should `VesinAutoAlgorithm` run the cell list for a search of `n_points`
/// points walking `n_images` periodic images of each pair, rather than the
/// brute force kernels?
///
/// The cell list wins above a number of points, and the brute force below.
/// Where exactly the two cross is measured on `device_id` the first time that
/// device is asked about, by timing both algorithms on a pair of synthetic
/// lattices, and kept for the rest of the process.
///
/// `points_volume` is the volume taken by the points (see `check_box`), used
/// to compare the cutoff with the typical distance between points. This makes
/// the choice independent of the unit of length.
bool auto_use_cell_list(
    double n_points,
    double n_images,
    double cutoff,
    double points_volume,
    int32_t device_id
);

} // namespace vesin::cuda


#endif // VESIN_CUDA_PICK_ALGO_HPP
