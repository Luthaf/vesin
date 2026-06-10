#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

CellListBuffers::~CellListBuffers() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_particle_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_starts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_offsets));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_points));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_cell_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_inv_box));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_cells));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_search));
        GPULITE_CUDART_CALL(cudaFree(this->d_n_cells_total));
        GPULITE_CUDART_CALL(cudaFree(this->d_face_distances));
        GPULITE_CUDART_CALL(cudaFree(this->d_bounding_min));
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
    }
}

CellListBuffers::CellListBuffers(CellListBuffers&& other) noexcept:
    CellListBuffers() {
    *this = std::move(other);
}

CellListBuffers& CellListBuffers::operator=(CellListBuffers&& other) noexcept {
    if (this != &other) {
        this->~CellListBuffers();

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

        this->d_sorted_points = other.d_sorted_points;
        other.d_sorted_points = nullptr;

        this->d_sorted_indices = other.d_sorted_indices;
        other.d_sorted_indices = nullptr;

        this->d_sorted_shifts = other.d_sorted_shifts;
        other.d_sorted_shifts = nullptr;

        this->d_sorted_cell_indices = other.d_sorted_cell_indices;
        other.d_sorted_cell_indices = nullptr;

        this->d_inv_box = other.d_inv_box;
        other.d_inv_box = nullptr;

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
    }
    return *this;
}

void CellListBuffers::allocate(size_t n_points, size_t n_cells) {
    if (this->points_capacity < n_points) {
        // Free old point-related buffers
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_particle_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_points));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_indices));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_shifts));
        GPULITE_CUDART_CALL(cudaFree(this->d_sorted_cell_indices));

        // And allocate with the new capacity
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_particle_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_points, sizeof(double) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_sorted_cell_indices, sizeof(int32_t) * n_points));
        this->points_capacity = n_points;
    }

    if (this->cells_capacity < n_cells) {
        // Free old cell-related buffers
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_counts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_starts));
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_offsets));

        // And allocate with the new capacity
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_counts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_starts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_cell_offsets, sizeof(int32_t) * n_cells));
        this->cells_capacity = n_cells;
    }

    // Allocate cell grid parameter buffers (fixed size, only once)
    if (this->d_inv_box == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_inv_box, sizeof(double) * 9));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_n_cells, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_n_search, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_n_cells_total, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_face_distances, sizeof(double) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_bounding_min, sizeof(double) * 3));
    }
}

SortBuffers::~SortBuffers() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_pairs_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_shifts_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_distances_tmp));
        GPULITE_CUDART_CALL(cudaFree(this->d_vectors_tmp));
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
    }
    return *this;
}

void SortBuffers::allocate(size_t n_pairs, bool return_shifts, bool return_distances, bool return_vectors) {
    if (this->capacity < n_pairs) {
        *this = SortBuffers{};
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_pairs_tmp, sizeof(size_t) * n_pairs * 2));
        this->capacity = n_pairs;
    }

    if (return_shifts && this->d_shifts_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_shifts_tmp, sizeof(int32_t) * n_pairs * 3));
    }

    if (return_distances && this->d_distances_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_distances_tmp, sizeof(double) * n_pairs));
    }

    if (return_vectors && this->d_vectors_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&this->d_vectors_tmp, sizeof(double) * n_pairs * 3));
    }
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    try {
        GPULITE_CUDART_CALL(cudaFree(this->d_length_ptr));
        if (this->pinned_length_ptr != nullptr) {
            GPULITE_CUDART_CALL(cudaFreeHost(this->pinned_length_ptr));
        }
        GPULITE_CUDART_CALL(cudaFree(this->d_cell_check_ptr));
        GPULITE_CUDART_CALL(cudaFree(this->d_box_diag));
        GPULITE_CUDART_CALL(cudaFree(this->d_inv_box_brute));
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
            if (this->pinned_length_ptr != nullptr) {
                GPULITE_CUDART_CALL(cudaFreeHost(this->pinned_length_ptr));
            }
            GPULITE_CUDART_CALL(cudaFree(this->d_cell_check_ptr));
            GPULITE_CUDART_CALL(cudaFree(this->d_box_diag));
            GPULITE_CUDART_CALL(cudaFree(this->d_inv_box_brute));
            GPULITE_CUDART_CALL(cudaFree(this->d_overflow_flag));
        } catch (const std::runtime_error& e) {
            std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
        }

        this->cell_list = std::move(other.cell_list);
        this->sort_buffers = std::move(other.sort_buffers);
        this->verlet_cache = std::move(other.verlet_cache);

        this->d_length_ptr = other.d_length_ptr;
        other.d_length_ptr = nullptr;

        this->pinned_length_ptr = other.pinned_length_ptr;
        other.pinned_length_ptr = nullptr;

        this->d_cell_check_ptr = other.d_cell_check_ptr;
        other.d_cell_check_ptr = nullptr;

        this->d_box_diag = other.d_box_diag;
        other.d_box_diag = nullptr;

        this->d_inv_box_brute = other.d_inv_box_brute;
        other.d_inv_box_brute = nullptr;

        this->d_overflow_flag = other.d_overflow_flag;
        other.d_overflow_flag = nullptr;

        this->pairs_capacity = other.pairs_capacity;
        other.pairs_capacity = 0;
    }
    return *this;
}
