#include <iostream>

#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "./extra_data.hpp"

using namespace vesin::cuda;

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
        GPULITE_CUDART_CALL(cudaFree(this->d_padding_overflow));
        GPULITE_CUDART_CALL(cudaFree(this->d_pairs_per_point));
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
            GPULITE_CUDART_CALL(cudaFree(this->d_padding_overflow));
            GPULITE_CUDART_CALL(cudaFree(this->d_pairs_per_point));
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

        this->d_padding_overflow = other.d_padding_overflow;
        other.d_padding_overflow = nullptr;

        this->d_pairs_per_point = other.d_pairs_per_point;
        other.d_pairs_per_point = nullptr;

        this->pairs_per_point_capacity = other.pairs_per_point_capacity;
        other.pairs_per_point_capacity = 0;

        this->pairs_capacity = other.pairs_capacity;
        other.pairs_capacity = 0;
    }
    return *this;
}
