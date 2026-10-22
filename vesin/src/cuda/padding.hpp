#ifndef VESIN_CUDA_PADDING_HPP
#define VESIN_CUDA_PADDING_HPP

#include <cstddef>

#include "vesin.h"

namespace vesin::cuda {

/// Prepare `neighbors` for a calculation with `options.padded_to > 0` on one
/// of the paths writing pairs with atomics: allocate the output for exactly
/// `n_points * padded_to` pairs, and reset the per-point counters and the
/// overflow detection.
void prepare_padding(VesinNeighborList& neighbors, size_t n_points, VesinOptions options);

/// Finish a calculation with `options.padded_to > 0` started with
/// `prepare_padding`: fill the unused slots with padding pairs, set the length
/// of `neighbors`, and throw an error if any point had more than `padded_to`
/// neighbors.
void finish_padding(VesinNeighborList& neighbors, size_t n_points, VesinOptions options);

/// Reset the device-side "first point with too many neighbors" to "no point",
/// before launching the kernels that update it with `atomicMin`.
void reset_padding_overflow(VesinNeighborList& neighbors);

/// Throw an error if any point had more than `padded_to` neighbors. This
/// synchronizes the stream.
void check_padding_overflow(VesinNeighborList& neighbors, VesinOptions options);

} // namespace vesin::cuda


#endif // VESIN_CUDA_PADDING_HPP
