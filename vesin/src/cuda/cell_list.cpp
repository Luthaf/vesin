#include <cstdint>
#include <iostream>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "./cell_list.hpp"
#include "./extra_data.hpp"
#include "./neighbors.hpp"
#include "./padding.hpp"
#include "./scan.hpp"

#include "kernels/scan.cuh"

using namespace vesin::cuda;

#include "kernels/cell_list.cuh"
const uint8_t CUDA_CELL_LIST_HEX[] = {
#include "generated/kernels/cell_list.cu.inc"
};
static const char* CUDA_CELL_LIST_CODE = reinterpret_cast<const char*>(CUDA_CELL_LIST_HEX);

/// Compile (or fetch from cache) an NVRTC kernel from the embedded
/// kernels/cell_list.cu source:
///
///   auto* kernel = VESIN_CELL_LIST_KERNEL(factory, count_neighbors_cell_list);
#define VESIN_CELL_LIST_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_CELL_LIST_CODE, "cell_list.cu", {"-std=c++17", "-default-device"})

// Hard upper bound on the number of cells, to keep the per-cell arrays (three
// `int32_t` each) at a sane size. This only kicks in for very large systems:
// the per-system budget below is normally the binding constraint.
static constexpr size_t DEFAULT_MAX_CELLS = 1 << 21;

// Upper bound on the number of cells per particle. This budget only matters for
// dilute systems, where the cell size picked by `compute_cell_grid_params`
// would create far more cells than particles and most of them would be empty.
// Empty cells are pure overhead: they cost a cell visit in the neighbor search
// without contributing any candidate pair.
static constexpr size_t MAX_CELLS_PER_PARTICLE = 2;

// Cost of visiting one neighbor cell, relative to the cost of evaluating one
// candidate pair. `compute_cell_grid_params` uses this to pick the cell size:
// the smaller it is, the finer the grid we can afford.
//
// Fitted from timings of the diamond benchmark: a least-squares fit of
//
//     time = A * total_candidates + B * total_cell_visits + C
//
// gives B/A = 1.23. A cell visit is therefore slightly *more* expensive than a
// candidate evaluation.
static constexpr double CELL_VISIT_COST = 1.23;

CellListBuffersCUDA::~CellListBuffersCUDA() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_particle_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_starts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_block_totals2));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_points));
        GPULITE_CUDART_CALL(cudaFree(this->d_wrapped_positions));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_cell_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_write_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals2));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_cells));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_search));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_cells_total));
        GPULITE_CUDART_CALL(cudaFree(this->d_face_distances));
        GPULITE_CUDART_CALL(cudaFree(this->d_bounding_min));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_out_of_reach));
        GPULITE_CUDART_CALL(cudaFree(this->d_bbox_partials));
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
    }
}

CellListBuffersCUDA::CellListBuffersCUDA(CellListBuffersCUDA&& other) noexcept:
    CellListBuffersCUDA() {
    *this = std::move(other);
}

CellListBuffersCUDA& CellListBuffersCUDA::operator=(CellListBuffersCUDA&& other) noexcept {
    if (this != &other) {
        this->~CellListBuffersCUDA();

        this->points_capacity = other.points_capacity;
        other.points_capacity = 0;

        this->cells_capacity = other.cells_capacity;
        other.cells_capacity = 0;

        this->d_cell_indices = other.d_cell_indices;
        other.d_cell_indices = nullptr;

        this->d_particle_shifts = other.d_particle_shifts;
        other.d_particle_shifts = nullptr;

        this->d_cell_counts = other.d_cell_counts;
        other.d_cell_counts = nullptr;

        this->d_cell_starts = other.d_cell_starts;
        other.d_cell_starts = nullptr;

        this->d_cell_offsets = other.d_cell_offsets;
        other.d_cell_offsets = nullptr;

        this->d_cell_block_totals = other.d_cell_block_totals;
        other.d_cell_block_totals = nullptr;

        this->d_cell_block_offsets = other.d_cell_block_offsets;
        other.d_cell_block_offsets = nullptr;

        this->d_cell_block_totals2 = other.d_cell_block_totals2;
        other.d_cell_block_totals2 = nullptr;

        this->d_sorted_points = other.d_sorted_points;
        other.d_sorted_points = nullptr;

        this->d_wrapped_positions = other.d_wrapped_positions;
        other.d_wrapped_positions = nullptr;

        this->d_sorted_indices = other.d_sorted_indices;
        other.d_sorted_indices = nullptr;

        this->d_sorted_shifts = other.d_sorted_shifts;
        other.d_sorted_shifts = nullptr;

        this->d_sorted_cell_indices = other.d_sorted_cell_indices;
        other.d_sorted_cell_indices = nullptr;

        this->d_counts = other.d_counts;
        other.d_counts = nullptr;

        this->d_write_offsets = other.d_write_offsets;
        other.d_write_offsets = nullptr;

        this->d_block_totals = other.d_block_totals;
        other.d_block_totals = nullptr;

        this->d_block_offsets = other.d_block_offsets;
        other.d_block_offsets = nullptr;

        this->d_block_totals2 = other.d_block_totals2;
        other.d_block_totals2 = nullptr;

        this->d_n_cells = other.d_n_cells;
        other.d_n_cells = nullptr;

        this->d_n_search = other.d_n_search;
        other.d_n_search = nullptr;

        this->d_n_cells_total = other.d_n_cells_total;
        other.d_n_cells_total = nullptr;

        this->d_face_distances = other.d_face_distances;
        other.d_face_distances = nullptr;

        this->d_bounding_min = other.d_bounding_min;
        other.d_bounding_min = nullptr;

        this->d_cell_out_of_reach = other.d_cell_out_of_reach;
        other.d_cell_out_of_reach = nullptr;

        this->d_bbox_partials = other.d_bbox_partials;
        other.d_bbox_partials = nullptr;
    }
    return *this;
}

void CellListBuffersCUDA::allocate(size_t n_points, size_t n_cells) {
    if (this->points_capacity < n_points) {
        // Free old point-related buffers
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_particle_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_points));
        GPULITE_CUDART_CALL(cudaFree(this->d_wrapped_positions));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_cell_indices));

        // And allocate with the new capacity
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_particle_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_points, sizeof(double) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_wrapped_positions, sizeof(double) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_cell_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_counts, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_write_offsets, sizeof(int32_t) * (n_points + 1)));
        size_t n_scan_blocks = (n_points + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals, sizeof(int32_t) * n_scan_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_offsets, sizeof(int32_t) * n_scan_blocks));
        size_t n_scan_blocks2 = (n_scan_blocks + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals2, sizeof(int32_t) * n_scan_blocks2));
        this->points_capacity = n_points;
    }

    if (this->cells_capacity < n_cells) {
        // Free old cell-related buffers
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_starts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_block_totals2));

        // And allocate with the new capacity
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_counts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_starts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_offsets, sizeof(int32_t) * n_cells));

        size_t n_cell_blocks = (n_cells + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_block_totals, sizeof(int32_t) * n_cell_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_block_offsets, sizeof(int32_t) * n_cell_blocks));
        size_t n_cell_blocks2 = (n_cell_blocks + VESIN_SCAN_ELEMENTS_PER_BLOCK - 1) / VESIN_SCAN_ELEMENTS_PER_BLOCK;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_block_totals2, sizeof(int32_t) * n_cell_blocks2));

        this->cells_capacity = n_cells;
    }

    // Allocate cell grid parameter buffers (fixed size, only once)
    if (this->d_n_cells == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_n_cells, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_n_search, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_n_cells_total, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_face_distances, sizeof(double) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_bounding_min, sizeof(double) * 3));
        GPULITE_CUDART_CALL(cudaMalloc(
            (void**)&this->d_cell_out_of_reach, sizeof(bool) * VESIN_MAX_OUT_OF_REACH_MASK
        ));
        GPULITE_CUDART_CALL(cudaMalloc(
            (void**)&this->d_bbox_partials, sizeof(double) * 6 * MAX_BOUNDING_BOX_BLOCKS
        ));
    }
}

// Exclusive prefix sum of the per-atom neighbour counts
// `d_counts[n_points]` → `d_write_offsets[n_points]`, returning the total
// number of pairs.
static size_t device_exclusive_prefix_sum(
    gpulite::KernelFactory& factory,
    CellListBuffersCUDA& cl,
    PinnedReadback* pinned,
    size_t n_points
) {
    vesin::cuda::device_exclusive_scan(
        factory,
        cl.d_counts,
        cl.d_write_offsets,
        cl.d_block_totals,
        cl.d_block_offsets,
        cl.d_block_totals2,
        n_points
    );

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &pinned->last_offset, &cl.d_write_offsets[n_points - 1],
        sizeof(int32_t), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &pinned->last_count, &cl.d_counts[n_points - 1],
        sizeof(int32_t), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    int32_t h_last_offset = pinned->last_offset;
    int32_t h_last_count = pinned->last_count;

    auto total = static_cast<int64_t>(h_last_offset) + h_last_count;
    if (h_last_offset < 0 || h_last_count < 0 || total > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(
            "too many pairs for the CUDA cell list (the limit is 2^31); "
            "please open an issue on GitHub if you need more than that"
        );
    }

    return static_cast<size_t>(total);
}

void vesin::cuda::cell_list_neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    bool all_periodic,
    VesinNeighborList& neighbors
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    size_t max_cells = std::max(n_points * MAX_CELLS_PER_PARTICLE, static_cast<size_t>(1));
    max_cells = std::min(DEFAULT_MAX_CELLS, max_cells);

    extras->cell_list.allocate(n_points, max_cells);
    auto& cl = extras->cell_list;

    auto config = gpulite::LaunchConfig();

    // The bounding box is only ever read for non-periodic directions, skip the
    // work if everything is periodic
    if (!all_periodic) {
        auto* bounding_kernel = VESIN_CELL_LIST_KERNEL(factory, compute_bounding_box);

        size_t bbox_blocks = std::min(
            std::max((n_points + VESIN_BOUNDING_BOX_BLOCK - 1) / VESIN_BOUNDING_BOX_BLOCK, static_cast<size_t>(1)),
            MAX_BOUNDING_BOX_BLOCKS
        );

        config.gridDim = dim3(bbox_blocks);
        config.blockDim = dim3(VESIN_BOUNDING_BOX_BLOCK);
        bounding_kernel->launch(
            config,
            d_points,
            n_points,
            cl.d_bbox_partials,
            cl.d_face_distances,
            cl.d_bounding_min
        );

        // With a single block there is nothing to combine and the kernel above
        // already wrote the final answer.
        if (bbox_blocks > 1) {
            auto* reduce_kernel = VESIN_CELL_LIST_KERNEL(factory, reduce_bounding_box);

            config.gridDim = dim3(1);
            reduce_kernel->launch(
                config,
                cl.d_bbox_partials,
                bbox_blocks,
                cl.d_face_distances,
                cl.d_bounding_min
            );
        }
    }

    auto* grid_kernel = VESIN_CELL_LIST_KERNEL(factory, compute_cell_grid_params);

    config.gridDim = dim3(1);
    config.blockDim = dim3(VESIN_GRID_PARAMS_BLOCK);
    grid_kernel->launch(
        config,
        d_box,
        d_periodic,
        options.cutoff,
        n_points,
        max_cells,
        CELL_VISIT_COST,
        extras->d_inv_box,
        cl.d_n_cells,
        cl.d_n_search,
        cl.d_n_cells_total,
        cl.d_face_distances,
        cl.d_cell_out_of_reach
    );

    GPULITE_CUDART_CALL(cudaMemset(cl.d_cell_counts, 0, sizeof(int32_t) * max_cells));
    GPULITE_CUDART_CALL(cudaMemset(cl.d_cell_starts, 0, sizeof(int32_t) * max_cells));

    size_t THREADS_PER_BLOCK = 256;
    size_t num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    config.gridDim = dim3(num_blocks_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    auto* assign_kernel = VESIN_CELL_LIST_KERNEL(factory, assign_cell_indices);
    assign_kernel->launch(
        config,
        d_points,
        n_points,
        extras->d_inv_box,
        d_periodic,
        cl.d_n_cells,
        cl.d_face_distances,
        cl.d_bounding_min,
        cl.d_cell_indices,
        cl.d_particle_shifts
    );

    auto* count_kernel = VESIN_CELL_LIST_KERNEL(factory, count_particles_per_cell);
    count_kernel->launch(
        config,
        cl.d_cell_indices,
        n_points,
        cl.d_cell_counts
    );

    // Exclusive prefix sum of the per-cell counts.
    vesin::cuda::device_exclusive_scan(
        factory,
        cl.d_cell_counts,
        cl.d_cell_starts,
        cl.d_cell_block_totals,
        cl.d_cell_block_offsets,
        cl.d_cell_block_totals2,
        max_cells
    );

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        cl.d_cell_offsets, cl.d_cell_starts,
        sizeof(int32_t) * max_cells, cudaMemcpyDeviceToDevice, nullptr
    ));

    config.gridDim = dim3(num_blocks_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);

    auto* scatter_kernel = VESIN_CELL_LIST_KERNEL(factory, scatter_particles);
    scatter_kernel->launch(
        config,
        d_points,
        n_points,
        cl.d_cell_indices,
        cl.d_particle_shifts,
        d_box,
        cl.d_cell_offsets,
        cl.d_sorted_points,
        cl.d_wrapped_positions,
        cl.d_sorted_indices,
        cl.d_sorted_shifts,
        cl.d_sorted_cell_indices
    );

    auto* count_neighbors_kernel = VESIN_CELL_LIST_KERNEL(factory, count_neighbors_cell_list);
    size_t num_blocks_search = (n_points + VESIN_PARTICLES_PER_BLOCK - 1) / VESIN_PARTICLES_PER_BLOCK;

    config.gridDim = dim3(num_blocks_search);
    config.blockDim = dim3(VESIN_CELL_LIST_BLOCK);

    if (options.padded_to > 0) {
        vesin::cuda::reset_padding_overflow(neighbors);
    }

    count_neighbors_kernel->launch(
        config,
        cl.d_wrapped_positions,
        n_points,
        d_box,
        cl.d_cell_out_of_reach,
        d_periodic,
        cl.d_n_cells,
        cl.d_n_search,
        cl.d_sorted_indices,
        cl.d_sorted_shifts,
        cl.d_sorted_cell_indices,
        cl.d_cell_starts,
        cl.d_cell_counts,
        options.cutoff,
        options.full,
        static_cast<int32_t>(options.padded_to),
        extras->d_padding_overflow,
        cl.d_counts
    );

    size_t n_pairs = 0;
    if (options.padded_to > 0) {
        // Each atom gets exactly `padded_to` slots in the output, there is no
        // need for the prefix sum. The slots not used by actual neighbors are
        // filled with padding in the second pass.
        vesin::cuda::check_padding_overflow(neighbors, options);
        n_pairs = n_points * options.padded_to;
    } else {
        // Device-side exclusive prefix sum of d_counts → d_write_offsets.
        n_pairs = device_exclusive_prefix_sum(factory, cl, extras->pinned, n_points);
    }

    // Allocate the output buffers. We ask for a little more than the number of
    // pairs we actually found, so that small step-to-step fluctuations don't
    // re-allocate all the buffers on every call.
    auto extra_pairs = std::min(n_pairs / 4, static_cast<size_t>(4096));
    vesin::cuda::allocate_output_buffers(neighbors, n_pairs + extra_pairs, options);

    config.gridDim = dim3(num_blocks_search);
    config.blockDim = dim3(VESIN_CELL_LIST_BLOCK);

    auto* fill_neighbors_kernel = VESIN_CELL_LIST_KERNEL(factory, fill_neighbors_cell_list);
    fill_neighbors_kernel->launch(
        config,
        cl.d_wrapped_positions,
        n_points,
        d_box,
        cl.d_cell_out_of_reach,
        d_periodic,
        cl.d_n_cells,
        cl.d_n_search,
        cl.d_sorted_indices,
        cl.d_sorted_shifts,
        cl.d_sorted_cell_indices,
        cl.d_cell_starts,
        cl.d_cell_counts,
        options.cutoff,
        options.full,
        options.return_shifts,
        options.return_distances,
        options.return_vectors,
        static_cast<int32_t>(options.padded_to),
        cl.d_write_offsets,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors
    );

    neighbors.length = n_pairs;
}
