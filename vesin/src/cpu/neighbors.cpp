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

    if (neighbors.opaque != nullptr) {
        free_extra_data(neighbors);
    }

    std::free(neighbors.pairs);
    std::free(neighbors.shifts);
    std::free(neighbors.vectors);
    std::free(neighbors.distances);
}


void vesin::cpu::neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    VesinNeighborList& raw_neighbors
) {
    if (options.algorithm != VesinAutoAlgorithm && options.algorithm != VesinCellList) {
        throw std::runtime_error("only VesinAutoAlgorithm and VesinCellList are supported on CPU");
    }
    // the cell list is the only algorithm available on CPU
    options.algorithm = VesinCellList;

    auto& extra = extra_data(raw_neighbors);
    if (options.skin > 0.0) {
        auto state = VerletList(extra.verlet_buffers);
        state.set_options(options);

        if (state.needs_rebuild(points, n_points, box)) {
            state.rebuild(points, n_points, box, extra.cell_list_buffers);
        }

        state.filter(points, box, options, raw_neighbors, extra.capacity);
    } else {
        cell_list_neighbors(
            points,
            n_points,
            std::move(box),
            options,
            extra.cell_list_buffers,
            raw_neighbors,
            extra.capacity
        );
    }
}
