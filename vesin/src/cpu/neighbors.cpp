#include <cassert>
#include <cstdlib>

#include <stdexcept>

#include "./cell_list.hpp"
#include "./extra_data.hpp"
#include "./neighbors.hpp"
#include "./verlet.hpp"

using namespace vesin::cpu;


void vesin::cpu::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCPU);

    if (neighbors.opaque == nullptr) {
        // vesin did not allocate anything for this neighbor list
        return;
    }

    // only release the arrays allocated by vesin, not the pre-allocated ones
    auto& extra = *static_cast<ExtraDataCpu*>(neighbors.opaque);
    std::free(extra.allocated_outputs.pairs);
    std::free(extra.allocated_outputs.shifts);
    std::free(extra.allocated_outputs.distances);
    std::free(extra.allocated_outputs.vectors);

    free_extra_data(neighbors);
}

static void run_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    ExtraDataCpu& extra,
    VesinNeighborList& raw_neighbors
) {
    // with pre-allocated outputs, the capacity is given by the caller in
    // `length`. This must be a copy, since `length` is overwritten with the
    // number of pairs at the end of the calculation.
    size_t pre_allocated_capacity = raw_neighbors.length;
    auto& capacity = options.pre_allocated_outputs ? pre_allocated_capacity : extra.capacity;

    if (options.skin > 0.0) {
        auto state = VerletList(extra.verlet_buffers);
        state.set_options(options);

        if (state.needs_rebuild(points, n_points, box)) {
            state.rebuild(points, n_points, box);
        }

        state.filter(points, box, options, raw_neighbors, capacity);
    } else {
        cell_list_neighbors(
            points,
            n_points,
            std::move(box),
            options,
            extra.cell_list_buffers,
            raw_neighbors,
            capacity
        );
    }
}


void vesin::cpu::neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    VesinNeighborList& raw_neighbors
) {
    auto& extra = extra_data(raw_neighbors);
    check_output_arrays(raw_neighbors, extra.allocated_outputs, options);

    if (options.pre_allocated_outputs) {
        // the arrays allocated by vesin in previous calls are no longer needed
        std::free(extra.allocated_outputs.pairs);
        std::free(extra.allocated_outputs.shifts);
        std::free(extra.allocated_outputs.distances);
        std::free(extra.allocated_outputs.vectors);
        extra.allocated_outputs = OutputPointers();
        extra.capacity = 0;
    }

    if (options.algorithm != VesinAutoAlgorithm && options.algorithm != VesinCellList) {
        throw std::runtime_error("only VesinAutoAlgorithm and VesinCellList are supported on CPU");
    }
    // the cell list is the only algorithm available on CPU
    options.algorithm = VesinCellList;

    try {
        run_neighbors(points, n_points, std::move(box), options, extra, raw_neighbors);
    } catch (...) {
        if (!options.pre_allocated_outputs) {
            extra.allocated_outputs = OutputPointers::from(raw_neighbors);
        }
        throw;
    }

    if (!options.pre_allocated_outputs) {
        extra.allocated_outputs = OutputPointers::from(raw_neighbors);
    }
}
