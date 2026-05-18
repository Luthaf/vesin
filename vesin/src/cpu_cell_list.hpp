#ifndef VESIN_CPU_CELL_LIST_HPP
#define VESIN_CPU_CELL_LIST_HPP

#include <cstddef>
#include <vector>

#include "vesin.h"

#include "types.hpp"

namespace vesin {
namespace cpu {

struct VerletState;

/// Extra CPU allocation metadata stored in `VesinNeighborList::opaque`.
struct ExtraDataCpu {
    /// Initialize empty CPU-side metadata.
    ExtraDataCpu() = default;
    /// Release optional Verlet cache state.
    ~ExtraDataCpu();

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
    /// Optional cached Verlet state for `skin > 0` calculations.
    VerletState* verlet_state = nullptr;
};

void free_neighbors(VesinNeighborList& neighbors);

void stateless_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    VesinNeighborList& neighbors,
    size_t& capacity
);

void neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
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

    // Ensure `capacity >= required`, growing if needed. Lets the caller
    // hoist the capacity check out of the hot per-pair loop so the
    // unchecked set_* variants below are safe to use.
    void ensure_capacity(size_t required);

    // Unchecked variants of the set_* methods above: the caller must have
    // already called ensure_capacity so that `index < capacity`. The
    // cachegrind profile attributed 34% of total instructions to the
    // per-pair branch + write path; hoisting the check is the win.
    void set_pair_unchecked(size_t index, size_t first, size_t second) {
        this->neighbors.pairs[index][0] = first;
        this->neighbors.pairs[index][1] = second;
    }
    void set_shift_unchecked(size_t index, vesin::CellShift shift) {
        this->neighbors.shifts[index][0] = shift[0];
        this->neighbors.shifts[index][1] = shift[1];
        this->neighbors.shifts[index][2] = shift[2];
    }
    void set_distance_unchecked(size_t index, double distance) {
        this->neighbors.distances[index] = distance;
    }
    void set_vector_unchecked(size_t index, vesin::Vector vector) {
        this->neighbors.vectors[index][0] = vector[0];
        this->neighbors.vectors[index][1] = vector[1];
        this->neighbors.vectors[index][2] = vector[2];
    }

    // sort the pairs currently in the neighbor list
    void sort();
};

} // namespace cpu
} // namespace vesin

#endif
