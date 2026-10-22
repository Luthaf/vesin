#ifndef VESIN_CUDA_SCAN_HPP
#define VESIN_CUDA_SCAN_HPP

#include <cstddef>
#include <cstdint>

#include <gpulite/gpulite.hpp>

namespace vesin::cuda {

/// Device-side exclusive prefix sum of `d_in[n]` → `d_out[n]`.
///
/// `d_block_totals` and `d_block_offsets` must each hold at least `ceil(n/1024)`
/// elements, and `d_block_totals2` at least `ceil(n/1024^2)`.
void device_exclusive_scan(
    gpulite::KernelFactory& factory,
    const int32_t* d_in,
    int32_t* d_out,
    int32_t* d_block_totals,
    int32_t* d_block_offsets,
    int32_t* d_block_totals2,
    size_t n
);

} // namespace vesin::cuda


#endif // VESIN_CUDA_SCAN_HPP
