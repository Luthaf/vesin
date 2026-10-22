#include "scan.cuh"

// -------------------------------------------------------------------------- //
// Device-side exclusive prefix sum (3-phase multi-block scan)
//
// This is used to compute the prefix sum of the neighbor counts array, which is
// needed to determine the write offsets for the neighbor pairs. The scan is
// done in three phases:
// 1. Each block computes a local exclusive scan of its chunk and stores the
//    block total.
// 2. A single block computes an exclusive scan of the block totals array.
// 3. Each block adds the block offset to its local scan results to produce the
//    final exclusive scan of the entire array.
// -------------------------------------------------------------------------- //

__global__ void exclusive_scan_phase_1(
    const int32* __restrict__ d_in,
    int32* __restrict__ d_out,
    int32* __restrict__ d_block_totals,
    size_t n
) {
    __shared__ int32 scan_data[VESIN_SCAN_THREADS];
    auto tid = threadIdx.x;
    auto chunk_base = blockIdx.x * VESIN_SCAN_ELEMENTS_PER_BLOCK;
    auto thread_base = chunk_base + tid * VESIN_SCAN_ELEMENTS_PER_THREAD;

    // Each thread loads VESIN_SCAN_ELEMENTS_PER_THREAD elements and sums them
    int32 local_sum = 0;
    for (int32 i = 0; i < VESIN_SCAN_ELEMENTS_PER_THREAD; i++) {
        auto idx = thread_base + i;
        local_sum += (idx < n) ? d_in[idx] : 0;
    }
    scan_data[tid] = local_sum;
    __syncthreads();

    // Hillis-Steele inclusive scan over thread sums
    for (int32 stride = 1; stride < VESIN_SCAN_THREADS; stride *= 2) {
        int32 val = (tid >= stride) ? scan_data[tid - stride] : 0;
        __syncthreads();
        scan_data[tid] += val;
        __syncthreads();
    }
    // scan_data[tid] = inclusive scan of per-thread sums
    // Exclusive prefix for this thread = scan_data[tid] - local_sum

    int32 prefix = scan_data[tid] - local_sum;

    // Write out the exclusive scan for each element
    int32 running = prefix;
    for (int32 i = 0; i < VESIN_SCAN_ELEMENTS_PER_THREAD; i++) {
        auto idx = thread_base + i;
        if (idx < n) {
            d_out[idx] = running;
            running += d_in[idx];
        }
    }

    // Last thread stores the block total
    if (tid == VESIN_SCAN_THREADS - 1) {
        d_block_totals[blockIdx.x] = scan_data[VESIN_SCAN_THREADS - 1];
    }
}


__global__ void exclusive_scan_phase_2(
    int32* __restrict__ d_block_totals,
    size_t n_blocks
) {
    __shared__ int32 scan_data[VESIN_SCAN_THREADS];
    auto tid = threadIdx.x;

    scan_data[tid] = (tid < n_blocks) ? d_block_totals[tid] : 0;
    __syncthreads();

    // Hillis-Steele inclusive scan
    for (int32 stride = 1; stride < VESIN_SCAN_THREADS; stride *= 2) {
        int32 val = (tid >= stride) ? scan_data[tid - stride] : 0;
        __syncthreads();
        scan_data[tid] += val;
        __syncthreads();
    }

    // Convert to exclusive and write back
    int32 exclusive = scan_data[tid] - ((tid < n_blocks) ? d_block_totals[tid] : 0);
    if (tid < n_blocks) {
        d_block_totals[tid] = exclusive;
    }
}


__global__ void exclusive_scan_phase_3(
    int32* __restrict__ d_out,
    const int32* __restrict__ d_block_totals,
    size_t n
) {
    auto tid = threadIdx.x;
    auto chunk_base = blockIdx.x * VESIN_SCAN_ELEMENTS_PER_BLOCK;
    auto thread_base = chunk_base + tid * VESIN_SCAN_ELEMENTS_PER_THREAD;
    auto offset = d_block_totals[blockIdx.x];

    for (int32 i = 0; i < VESIN_SCAN_ELEMENTS_PER_THREAD; i++) {
        auto idx = thread_base + i;
        if (idx < n) {
            d_out[idx] += offset;
        }
    }
}
