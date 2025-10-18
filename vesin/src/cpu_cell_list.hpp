#ifndef VESIN_CPU_CELL_LIST_HPP
#define VESIN_CPU_CELL_LIST_HPP

#include <vector>

#include "vesin.h"

#include "types.hpp"

namespace vesin {
namespace cpu {

void free_neighbors(VesinNeighborList& neighbors);

void neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox cell,
    VesinOptions options,
    VesinNeighborList& neighbors
);

/// The cell list is used to sort atoms inside bins/cells.
///
/// The list of potential pairs is then constructed by looking through all
/// neighboring cells (the number of cells to search depends on the cutoff and
/// the size of the cells) for each atom to create pair candidates.
class CellList {
public:
    /// Create a new `CellList` for the given bounding box and cutoff,
    /// determining all required parameters.
    CellList(BoundingBox box, double cutoff);

    /// Add a single point to the cell list at the given `position`. The point
    /// is uniquely identified by its `index`.
    void add_point(size_t index, Vector position);

    /// Iterate over all possible pairs, calling the given callback every time
    template <typename Function>
    void foreach_pair(Function callback);

private:
    /// How many cells do we need to look at when searching neighbors to include
    /// all neighbors below cutoff
    std::array<int32_t, 3> n_search_;

    /// the cells themselves are a list of points & corresponding
    /// shift to place the point inside the cell
    struct Point {
        size_t index;
        CellShift shift;
    };
    struct Cell: public std::vector<Point> {};

    // raw data for the cells
    std::vector<Cell> cells_;
    // shape of the cell array
    std::array<size_t, 3> cells_shape_;

    BoundingBox box_;

    bool calc_neighbor_cell_shifts_and_check_outside_bounds(
        std::array<int32_t, 3>& neighbor_cell_i,
        CellShift& cell_shift
    ) const;

    Cell& get_cell(std::array<int32_t, 3> index);
};

/// Wrapper around `VesinNeighborList` that behaves like a std::vector,
/// automatically growing memory allocations.
class GrowableNeighborList {
public:
    VesinNeighborList& neighbors;
    size_t capacity;
    VesinOptions options;

    size_t length() const {
        return neighbors.length;
    }

    void increment_length() {
        neighbors.length += 1;
    }

    void set_pair(size_t index, size_t first, size_t second);
    void set_shift(size_t index, vesin::CellShift shift);
    void set_distance(size_t index, double distance);
    void set_vector(size_t index, vesin::Vector vector);

    // reset length to 0, and allocate/deallocate members of
    // `neighbors` according to `options`
    void reset();

    // allocate more memory & update capacity
    void grow();

    // sort the pairs currently in the neighbor list
    void sort();
};

} // namespace cpu
} // namespace vesin

#endif
