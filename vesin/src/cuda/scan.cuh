#ifndef VESIN_CUDA_SCAN_CUH
#define VESIN_CUDA_SCAN_CUH

#include <cstddef>

// This is a device-side exclusive prefix sum (scan) implementation. It is based
// on the Blelloch scan algorithm, which consists of three phases, each
// implemented as a separate kernel.

/// Phase 1: each block does a local exclusive scan of its chunk and stores
/// the block total.
__global__ void exclusive_scan_phase_1(
    const int* d_in,
    int* d_out,
    int* d_block_totals,
    size_t n
);

/// Phase 2: single-block exclusive scan of the block totals array.
__global__ void exclusive_scan_phase_2(
    int* d_block_totals,
    size_t n_blocks
);


/// Phase 3: add block offsets to each block's local scan results.
__global__ void exclusive_scan_phase_3(
    int* d_out,
    const int* d_block_totals,
    size_t n
);

#endif // VESIN_CUDA_SCAN_CUH
