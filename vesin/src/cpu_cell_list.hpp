#ifndef VESIN_CPU_CELL_LIST_HPP
#define VESIN_CPU_CELL_LIST_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include "vesin.h"

#include "types.hpp"

namespace vesin {
namespace cpu {

struct VerletList;

/// All memory allocations used by the CPU cell-list algorithm, stored
/// separately from `CellList` so they can be reused across calls to
/// `cell_list_neighbors()`. Each call clears and rewrites the data, only the
/// allocations (vector capacity) are kept.
struct CellListBuffersCPU {
    /// `original_index[s]` is the original point index at sorted position `s`.
    std::vector<size_t> original_index;
    /// `sorted_shifts[s]` is the CellShift of the point at sorted position `s`.
    std::vector<CellShift> sorted_shifts;
    /// `cell_start[c]` is the first sorted position in cell `c`.
    std::vector<size_t> cell_start;
    /// `cell_count[c]` is the number of points in cell `c`.
    std::vector<size_t> cell_count;
    /// `wrapped_positions[s]` is the position of the point at sorted position
    /// `s` wrapped back into the unit cell.
    std::vector<Vector> wrapped_positions;

    /// Scratch: per-point neighbor count before the prefix sum (first pass).
    std::vector<size_t> count;
    /// Scratch: per-thread partial counts merged into `count` (first pass).
    std::vector<std::vector<size_t>> local_counts;
};

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
    VerletList* verlet_state = nullptr;
    /// Cached allocations for the cell-list path, reused across calls.
    CellListBuffersCPU cell_list_buffers;
};

class ThreadPool;

void free_neighbors(VesinNeighborList& neighbors);

void cell_list_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    CellListBuffersCPU& buffers,
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

/// The cell list is used to sort points inside bins/cells.
///
/// Points are binned into a 3-D grid of cells whose size is determined by the
/// cutoff and the bounding box. `build()` bins all points into their cells
/// and finalises the cell list into a compressed sparse row (CSR) layout that
/// supports efficient per-point neighbour queries via `foreach_neighbor_of()`.
///
/// The data arrays live in a caller-owned `CellListBuffersCPU`, so the allocated
/// storage can be reused across neighbor list calls; this `CellList` only holds
/// the grid geometry (box, cells per direction, search range) and a reference
/// to the buffers.
///
/// ## CSR layout
///
/// `build()` first bins each point into its cell (computing the periodic wrap
/// shift if the box is periodic), then flattens the per-cell point lists into a
/// single contiguous sequence of "sorted positions", laid out cell by cell in
/// row-major cell order. Three parallel arrays indexed by sorted position `s`
/// store the point data:
///
///  - `original_index[s]` — the original point index;
///  - `sorted_shifts[s]` — the periodic wrap shift;
///  - `wrapped_positions[s]` — the position wrapped back into the unit cell.
///
/// The cell structure is then compressed into two offset arrays, mirroring
/// the row pointers of a CSR matrix:
///
///  - `cell_start[c]` — the first sorted position belonging to cell `c`;
///  - `cell_count[c]` — the number of points in cell `c`.
///
/// The points of cell `c` therefore occupy the contiguous range
/// `[cell_start[c], cell_start[c] + cell_count[c])`. `build()` fills these
/// arrays directly, without any intermediate per-cell storage: a first pass
/// counts the points per cell, a prefix sum derives the `cell_start` offsets,
/// and a second pass scatters each point into its final CSR slot. Traversing a
/// neighbouring cell during a query is then just a sequential scan of that
/// slice.
class CellList {
public:
    /// Create a new `CellList` for the given bounding box and cutoff,
    /// determining all required parameters. All data arrays are stored in
    /// `buffers`, which must outlive this object.
    CellList(CellListBuffersCPU& buffers, BoundingBox box, double cutoff);

    /// Build the cell list for the given points: bin the points into cells,
    /// then produce the CSR layout and the wrapped positions (see the class
    /// documentation). Must be called before `foreach_neighbor_of()`. Can be
    /// called again on the same `CellList` to re-bin a new set of points.
    void build(const Vector* points, size_t n_points);

    /// Iterate over all neighbours of the point at `sorted_i`, calling
    /// `callback(point_j, shift, vector, distance2)` for each pair that passes
    /// the cutoff and the half-list filter (if `options.full` is false).
    template <typename Function>
    void foreach_neighbor_of(size_t sorted_i, VesinOptions options, Function callback) const;

    /// Original index of the point at sorted position `sorted`.
    size_t original_index(size_t sorted) const { return buffers_.original_index[sorted]; }

    /// Number of cells in each direction of the grid.
    const std::array<size_t, 3>& n_cells() const { return n_cells_; }
    /// Get the search range.
    const std::array<int32_t, 3>& n_search() const { return n_search_; }
    /// Get the bounding box.
    const BoundingBox& box() const { return box_; }

private:
    /// Data arrays for the cell list, owned by the caller.
    CellListBuffersCPU& buffers_;

    BoundingBox box_;

    /// How many cells do we need to look at when searching neighbors to include
    /// all neighbors below cutoff
    std::array<int32_t, 3> n_search_;

    /// number of cells in all directions
    std::array<size_t, 3> n_cells_;

    /// Compute the linear index of the cell containing a point at `position`,
    /// together with the periodic wrap shift used to place it inside the cell.
    std::pair<size_t, CellShift> point_cell(Vector position) const;
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

    void set_pair(size_t index, size_t first, size_t second) {
        while (index >= capacity) {
            this->grow();
        }
        neighbors.pairs[index][0] = first;
        neighbors.pairs[index][1] = second;
    }

    void set_shift(size_t index, vesin::CellShift shift) {
        while (index >= capacity) {
            this->grow();
        }
        neighbors.shifts[index][0] = shift[0];
        neighbors.shifts[index][1] = shift[1];
        neighbors.shifts[index][2] = shift[2];
    }

    void set_distance(size_t index, double distance) {
        while (index >= capacity) {
            this->grow();
        }
        neighbors.distances[index] = distance;
    }

    void set_vector(size_t index, vesin::Vector vector) {
        while (index >= capacity) {
            this->grow();
        }
        neighbors.vectors[index][0] = vector[0];
        neighbors.vectors[index][1] = vector[1];
        neighbors.vectors[index][2] = vector[2];
    }

    // reset length to 0, and allocate/deallocate members of
    // `neighbors` according to `options`
    void reset();

    // allocate more memory & update capacity, `new_size = 0` means doubling the
    // current capacity, otherwise grow to at least `new_size` capacity. This
    // function does not update the `length` of `neighbors`.
    void grow(size_t new_size = 0);

    // sort the pairs currently in the neighbor list
    void sort();
};

} // namespace cpu
} // namespace vesin

#endif
