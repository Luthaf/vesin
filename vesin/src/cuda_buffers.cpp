#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

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
        size_t n_scan_blocks = (n_points + 1023) / 1024;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals, sizeof(int32_t) * n_scan_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_offsets, sizeof(int32_t) * n_scan_blocks));
        size_t n_scan_blocks2 = (n_scan_blocks + 1023) / 1024;
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

        size_t n_cell_blocks = (n_cells + 1023) / 1024;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_block_totals, sizeof(int32_t) * n_cell_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_block_offsets, sizeof(int32_t) * n_cell_blocks));
        size_t n_cell_blocks2 = (n_cell_blocks + 1023) / 1024;
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
            (void**)&this->d_cell_out_of_reach, sizeof(bool) * MAX_OUT_OF_REACH_MASK
        ));
        GPULITE_CUDART_CALL(cudaMalloc(
            (void**)&this->d_bbox_partials, sizeof(double) * 6 * MAX_BOUNDING_BOX_BLOCKS
        ));
    }
}

SortBuffers::~SortBuffers() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_pairs_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_shifts_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_distances_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_vectors_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals2));
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
    }
}

SortBuffers::SortBuffers(SortBuffers&& other) noexcept:
    SortBuffers() {
    *this = std::move(other);
}

SortBuffers& SortBuffers::operator=(SortBuffers&& other) noexcept {
    if (this != &other) {
        this->~SortBuffers();

        this->capacity = other.capacity;
        other.capacity = 0;

        this->d_pairs_tmp = other.d_pairs_tmp;
        other.d_pairs_tmp = nullptr;

        this->d_shifts_tmp = other.d_shifts_tmp;
        other.d_shifts_tmp = nullptr;

        this->d_distances_tmp = other.d_distances_tmp;
        other.d_distances_tmp = nullptr;

        this->d_vectors_tmp = other.d_vectors_tmp;
        other.d_vectors_tmp = nullptr;

        this->points_capacity = other.points_capacity;
        other.points_capacity = 0;

        this->d_counts = other.d_counts;
        other.d_counts = nullptr;

        this->d_offsets = other.d_offsets;
        other.d_offsets = nullptr;

        this->d_block_totals = other.d_block_totals;
        other.d_block_totals = nullptr;

        this->d_block_offsets = other.d_block_offsets;
        other.d_block_offsets = nullptr;

        this->d_block_totals2 = other.d_block_totals2;
        other.d_block_totals2 = nullptr;
    }
    return *this;
}

void SortBuffers::allocate(
    size_t n_pairs,
    size_t n_points,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    // The per-pair and per-point arrays are sized by different quantities and
    // grown independently, so that getting more pairs does not throw away a
    // per-point buffer that is still big enough, and vice versa.
    if (this->capacity < n_pairs) {
        GPULITE_CUDART_CALL(cudaFree(this->d_pairs_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_shifts_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_distances_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_vectors_tmp));
        this->d_pairs_tmp = nullptr;
        this->d_shifts_tmp = nullptr;
        this->d_distances_tmp = nullptr;
        this->d_vectors_tmp = nullptr;
        this->capacity = 0;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_pairs_tmp, sizeof(size_t) * n_pairs * 2));
        this->capacity = n_pairs;
    }

    // The optional buffers are allocated the first time they are asked for,
    // which can be a later call than the one that set the capacity, so they are
    // sized from the capacity rather than from this call's `n_pairs`.
    if (return_shifts && this->d_shifts_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_shifts_tmp, sizeof(int32_t) * this->capacity * 3));
    }

    if (return_distances && this->d_distances_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_distances_tmp, sizeof(double) * this->capacity));
    }

    if (return_vectors && this->d_vectors_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_vectors_tmp, sizeof(double) * this->capacity * 3));
    }

    if (this->points_capacity < n_points) {
        GPULITE_CUDART_CALL(cudaFree(this->d_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_block_totals2));
        this->d_counts = nullptr;
        this->d_offsets = nullptr;
        this->d_block_totals = nullptr;
        this->d_block_offsets = nullptr;
        this->d_block_totals2 = nullptr;
        this->points_capacity = 0;

        // Same two-level scan as the cell list, see `device_exclusive_scan`
        size_t n_blocks = (n_points + 1023) / 1024;
        size_t n_blocks_2 = (n_blocks + 1023) / 1024;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_counts, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_offsets, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals, sizeof(int32_t) * n_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_offsets, sizeof(int32_t) * n_blocks));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_block_totals2, sizeof(int32_t) * n_blocks_2));

        this->points_capacity = n_points;
    }
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_length_ptr));
        if (this->pinned != nullptr) {
            GPULITE_CUDART_CALL(cudaFreeHost(this->pinned));
        }
        GPULITE_CUDART_CALL(cudaFree(this->d_all_periodic));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_search));
        GPULITE_CUDART_CALL(cudaFree(this->d_inv_box));
        GPULITE_CUDART_CALL(cudaFree(this->d_overflow_flag));
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
    }
}

CudaNeighborListExtras::CudaNeighborListExtras(CudaNeighborListExtras&& other) noexcept:
    CudaNeighborListExtras() {
    *this = std::move(other);
}

CudaNeighborListExtras& CudaNeighborListExtras::operator=(CudaNeighborListExtras&& other) noexcept {
    if (this != &other) {
        try {
            GPULITE_CUDART_CALL(cudaFree(this->d_length_ptr));
            if (this->pinned != nullptr) {
                GPULITE_CUDART_CALL(cudaFreeHost(this->pinned));
            }
            GPULITE_CUDART_CALL(cudaFree(this->d_all_periodic));
            GPULITE_CUDART_CALL(cudaFree(this->d_n_search));
            GPULITE_CUDART_CALL(cudaFree(this->d_inv_box));
            GPULITE_CUDART_CALL(cudaFree(this->d_overflow_flag));
        } catch (const std::runtime_error& e) {
            std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
        }

        this->cell_list = std::move(other.cell_list);
        this->sort_buffers = std::move(other.sort_buffers);
        this->verlet_cache = std::move(other.verlet_cache);

        this->d_length_ptr = other.d_length_ptr;
        other.d_length_ptr = nullptr;

        this->pinned = other.pinned;
        other.pinned = nullptr;

        this->d_all_periodic = other.d_all_periodic;
        other.d_all_periodic = nullptr;

        this->d_n_search = other.d_n_search;
        other.d_n_search = nullptr;

        this->d_inv_box = other.d_inv_box;
        other.d_inv_box = nullptr;

        this->d_overflow_flag = other.d_overflow_flag;
        other.d_overflow_flag = nullptr;

        this->pairs_capacity = other.pairs_capacity;
        other.pairs_capacity = 0;
    }
    return *this;
}
