#ifndef VESIN_CUDA_EXTRA_DATA_HPP
#define VESIN_CUDA_EXTRA_DATA_HPP

#include <cstddef>
#include <cstdint>

#include <gpulite/gpulite.hpp>

#include "vesin.h"

#include "../outputs.hpp"

#include "./cell_list.hpp"
#include "./sort_pairs.hpp"
#include "./verlet.hpp"

namespace vesin::cuda {

/// Host-side staging for the small device-to-host transfers.
///
/// This is pinned memory, which makes `cudaMemcpyAsync` actually asynchronous.
struct PinnedReadback {
    double box[9];
    bool periodic[3];
    bool all_periodic;
    size_t length;
    int32_t n_search[3];
    int32_t last_offset;
    int32_t last_count;
    int32_t overflow;
    int32_t rebuild;
    int32_t padding_overflow;
};

/// Extra CUDA allocation metadata stored in `VesinNeighborList::opaque`.
struct CudaNeighborListExtras {
    /// Capacity for the pair buffers in the VesinNeighborList
    size_t pairs_capacity = 0;

    /// Output arrays allocated by vesin and left in the neighbor list at the
    /// end of the last calculation
    OutputPointers allocated_outputs;

    /// Is the current calculation using output arrays pre-allocated by the caller?
    bool pre_allocated_outputs = false;

    /// CUDA stream used by the current calculation, from
    /// `VesinOptions::cuda_stream`
    cudaStream_t stream = nullptr;

    /// GPU-side pointer to the length of the neighbor list
    size_t* d_length_ptr = nullptr;
    /// GPU-side output of `analyze_box`
    bool* d_all_periodic = nullptr;
    /// GPU-side flag to detect overflow of pair buffers
    int32_t* d_overflow_flag = nullptr;
    /// GPU-side first point with more than `padded_to` neighbors
    int32_t* d_padding_overflow = nullptr;

    /// Per-point number of pairs written so far, used with `padded_to` by the
    /// paths writing pairs with atomics (brute force and Verlet)
    int32_t* d_pairs_per_point = nullptr;
    size_t pairs_per_point_capacity = 0;

    /// Pinned host staging for every device-to-host readback, see PinnedReadback
    PinnedReadback* pinned = nullptr;

    /// Cell list buffers (allocated on demand for large systems)
    CellListBuffersCUDA cell_list;

    /// Box parameters computed by `analyze_box`
    int32_t* d_n_search = nullptr;     // [3] periodic images to walk per direction
    double (*d_inv_box)[3] = nullptr;  // [3][3] inverse box matrix

    /// Temporary buffers for on-device sorting
    SortBuffers sort_buffers;

    /// Verlet cache state
    VerletCache verlet_cache;

    CudaNeighborListExtras() = default;
    ~CudaNeighborListExtras();

    CudaNeighborListExtras(CudaNeighborListExtras&& other) noexcept;
    CudaNeighborListExtras& operator=(CudaNeighborListExtras&& other) noexcept;

    CudaNeighborListExtras(const CudaNeighborListExtras&) = delete;
    CudaNeighborListExtras& operator=(const CudaNeighborListExtras&) = delete;
};

/// Get the `CudaNeighborListExtras` stored inside `VesinNeighborList`'s opaque pointer
inline CudaNeighborListExtras* get_cuda_extras(VesinNeighborList& neighbors) {
    if (neighbors.opaque == nullptr) {
        neighbors.opaque = new vesin::cuda::CudaNeighborListExtras();
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
}

} // namespace vesin::cuda


#endif // VESIN_CUDA_EXTRA_DATA_HPP
