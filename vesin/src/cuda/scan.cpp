#include <cstdint>

#include <stdexcept>
#include <string>

#include <gpulite/gpulite.hpp>

#include "./scan.hpp"

#include "kernels/scan.cuh"
const uint8_t CUDA_SCAN_HEX[] = {
#include "generated/kernels/scan.cu.inc"
};
static const char* CUDA_SCAN_CODE = reinterpret_cast<const char*>(CUDA_SCAN_HEX);

#define VESIN_SCAN_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_SCAN_CODE, "scan.cu", {"-std=c++17", "-default-device"})

// When used to scan per-cell counts, this scans the whole `max_cells` range
// rather than just the `n_cells_total` cells actually in use: the count of an
// unused cell is zero (`d_cell_counts` is zeroed before) so it contributes
// nothing, and this keeps the number of elements host-side, avoiding a read
// back of `n_cells_total`.
void vesin::cuda::device_exclusive_scan(
    gpulite::KernelFactory& factory,
    const int32_t* d_in,
    int32_t* d_out,
    int32_t* d_block_totals,
    int32_t* d_block_offsets,
    int32_t* d_block_totals2,
    size_t n
) {
    size_t n_scan_blocks = (n + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;
    size_t n_scan_blocks_2 = (n_scan_blocks + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;

    if (n_scan_blocks_2 > VESIN_SCAN_MAX_BLOCKS) {
        throw std::runtime_error(
            "too many elements (" + std::to_string(n) + ") for the two-level "
            "device prefix scan (the limit is ~68 billion). This should not happen."
        );
    }

    auto* scan_phase_1 = VESIN_SCAN_KERNEL(factory, exclusive_scan_phase_1);
    auto* scan_phase_2 = VESIN_SCAN_KERNEL(factory, exclusive_scan_phase_2);
    auto* scan_phase_3 = VESIN_SCAN_KERNEL(factory, exclusive_scan_phase_3);

    auto config = gpulite::LaunchConfig();
    config.blockDim = dim3(VESIN_SCAN_THREADS);

    // Each block of 256 threads processes 1024 elements (4 per thread):
    //   - Each thread loads and sums its 4 elements
    //   - Hillis-Steele inclusive scan over the 256 per-thread sums (shared mem)
    //   - Each thread writes the exclusive scan of its 4 elements to d_out
    //   - The last thread stores the block's total to d_block_totals[blockIdx.x]
    //
    // After this, d_out holds the correct exclusive scan *within* each block,
    // but blocks don't know their offset relative to earlier blocks.
    // d_block_totals[i] = total count of elements in blocks 0..i.
    config.gridDim = dim3(n_scan_blocks);
    scan_phase_1->launch(config, d_in, d_out, d_block_totals, n);

    if (n_scan_blocks == 1) {
        // A single block covers the whole input, so the local scan phase 1 just
        // wrote is already the global one: phase 2 would only set the single
        // block total to zero, and phase 3 would add that zero back. Skipping
        // both saves two kernel launches, which is worth having for the small
        // systems where the launch latency is the whole cost of the call.
        return;
    }

    if (n_scan_blocks <= VESIN_SCAN_MAX_BLOCKS) {
        config.gridDim = dim3(1);
        scan_phase_2->launch(config, d_block_totals, n_scan_blocks);

        config.gridDim = dim3(n_scan_blocks);
        scan_phase_3->launch(config, d_out, d_block_totals, n);
    } else {
        // The level-1 block totals array has more than 256 elements, so it
        // can't be scanned by a single phase-2 block. We apply the same 3-phase
        // scan to d_block_totals itself, producing per-block offsets that are
        // then added back at both levels.
        //
        // Level 2 phase 1: scan d_block_totals → d_block_offsets (local scan
        //   within each level-2 block), collect level-2 block totals in
        //   d_block_totals2. d_block_offsets now holds correct values within
        //   each level-2 block, but not globally.
        //
        // Level 2 phase 2: exclusive scan of d_block_totals2 in a single block
        //   (≤256 elements by construction). After this, d_block_totals2[i] =
        //   the offset to add to level-2 block i's results.
        //
        // Level 2 phase 3: add level-2 offsets to d_block_offsets. After this,
        //   d_block_offsets holds the correct global exclusive prefix sum of
        //   the level-1 block totals — i.e. d_block_offsets[i] = the global
        //   offset for level-1 block i.
        //
        // Level 1 phase 3: add d_block_offsets[blockIdx.x] to every element in
        //   d_out. After this, d_out holds the correct global exclusive prefix
        //   sum of d_in.

        // Level 2 phase 1
        config.gridDim = dim3(n_scan_blocks_2);
        scan_phase_1->launch(config, d_block_totals, d_block_offsets, d_block_totals2, n_scan_blocks);

        // Level 2 phase 2
        config.gridDim = dim3(1);
        scan_phase_2->launch(config, d_block_totals2, n_scan_blocks_2);

        // Level 2 phase 3: add level-2 offsets to level-1 block offsets
        config.gridDim = dim3(n_scan_blocks_2);
        scan_phase_3->launch(config, d_block_offsets, d_block_totals2, n_scan_blocks);

        // Level 1 phase 3: add level-1 block offsets to per-element scan
        config.gridDim = dim3(n_scan_blocks);
        scan_phase_3->launch(config, d_out, d_block_offsets, n);
    }
}
