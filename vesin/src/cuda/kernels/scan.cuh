#ifndef VESIN_CUDA_SCAN_CUH
#define VESIN_CUDA_SCAN_CUH

#include <cstddef>

#include "integer.cuh"

// `device_exclusive_scan` in cuda/scan.cpp sizes its launches from these.
// `VESIN_SCAN_THREADS` is the required block size -- the phases size
// `scan_data` from it and index up to `VESIN_SCAN_THREADS - 1`, so a smaller
// block would read uninitialized shared memory instead of failing -- and
// `VESIN_SCAN_MAX_BLOCKS` is the most block totals phase 2 can scan at once,
// since it does so with a single block of `VESIN_SCAN_THREADS` threads.
#define VESIN_SCAN_THREADS 256
#define VESIN_SCAN_ELEMENTS_PER_THREAD 4
#define VESIN_SCAN_ELEMENTS_PER_BLOCK (VESIN_SCAN_THREADS * VESIN_SCAN_ELEMENTS_PER_THREAD)  // 1024
#define VESIN_SCAN_MAX_BLOCKS VESIN_SCAN_THREADS

// This is a device-side exclusive prefix sum (scan) implementation. It is based
// on the Blelloch scan algorithm, which consists of three phases, each
// implemented as a separate kernel.

/// Phase 1: each block does a local exclusive scan of its chunk and stores
/// the block total.
__global__ void exclusive_scan_phase_1(
    const int32* d_in,
    int32* d_out,
    int32* d_block_totals,
    size_t n
);

/// Phase 2: single-block exclusive scan of the block totals array.
__global__ void exclusive_scan_phase_2(
    int32* d_block_totals,
    size_t n_blocks
);


/// Phase 3: add block offsets to each block's local scan results.
__global__ void exclusive_scan_phase_3(
    int32* d_out,
    const int32* d_block_totals,
    size_t n
);

#endif // VESIN_CUDA_SCAN_CUH
