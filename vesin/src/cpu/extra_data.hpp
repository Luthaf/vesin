#ifndef VESIN_CPU_EXTRA_DATA_HPP
#define VESIN_CPU_EXTRA_DATA_HPP

#include <cstddef>

#include "vesin.h"

#include "./cell_list.hpp"
#include "./verlet.hpp"

namespace vesin::cpu {

/// Extra CPU allocation metadata stored in `VesinNeighborList::opaque`.
struct ExtraDataCpu {
    /// Initialize empty CPU-side metadata.
    ExtraDataCpu() = default;
    /// Release optional Verlet cache state.
    ~ExtraDataCpu() = default;

    /// Disallow copy construction; this object owns CPU-side cache metadata.
    ExtraDataCpu(const ExtraDataCpu&) = delete;
    /// Disallow copy assignment; this object owns CPU-side cache metadata.
    ExtraDataCpu& operator=(const ExtraDataCpu&) = delete;
    /// Disallow move construction; the C API stores this object by pointer.
    ExtraDataCpu(ExtraDataCpu&&) = delete;
    /// Disallow move assignment; the C API stores this object by pointer.
    ExtraDataCpu& operator=(ExtraDataCpu&&) = delete;

    /// Persisted GrowableNeighborList output capacity.
    size_t capacity = 0;
    /// Reusable Verlet cache state and filter scratch for `skin > 0`.
    VerletBuffersCPU verlet_buffers;
    /// Cached allocations for the cell-list path, reused across calls.
    CellListBuffersCPU cell_list_buffers;
};

/// Get the `ExtraDataCpu` stored inside `VesinNeighborList`'s opaque pointer,
/// creating it if needed.
inline ExtraDataCpu& extra_data(VesinNeighborList& neighbors) {
    if (neighbors.opaque == nullptr) {
        neighbors.opaque = new ExtraDataCpu();
    }

    return *static_cast<ExtraDataCpu*>(neighbors.opaque);
}

/// Delete the `ExtraDataCpu` stored inside `VesinNeighborList`'s opaque pointer
inline void free_extra_data(VesinNeighborList& neighbors) {
    delete static_cast<ExtraDataCpu*>(neighbors.opaque);
    neighbors.opaque = nullptr;
}

} // namespace vesin::cpu


#endif // VESIN_CPU_EXTRA_DATA_HPP
